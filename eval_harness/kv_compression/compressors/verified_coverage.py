"""Output-error measurement for verified KV compression (v2 — the CHECK).

This is the "checking" half of the verified-compression design
(``notes/verified_kv_compression_coverage.md``, Rules 4-5). It is **pure
measurement**: given the full cache and a *proposed* keep-set, it reports how
much the attention output would degrade if we committed to that eviction —
the "leftover" that v2 will eventually turn into a random-ambassador count.

Nothing here evicts, samples, or mutates the cache. :class:`VerifiedSketch`
calls it to *log* leftover before deciding a split (Step 1 of the v2 build:
get the numbers on screen before acting on them).

What "output-error" is (concretely)
-----------------------------------
Attention output for one query ``q`` is the softmax-weighted blend of value
vectors, ``out = softmax(q Kᵀ) V``. We compute that blend twice —

* **full**: over all ``T`` cached tokens (the ground truth; still present at
  compression time), and
* **kept**: over only the proposed ``M`` retained tokens —

and report the *relative* distance ``‖out_kept − out_full‖ / ‖out_full‖`` per
(batch, kv-head). Relative (not absolute) so it is comparable across heads and
layers: 0 means the kept set reproduces the answer exactly, 1 means it is as
wrong as keeping nothing.

Which queries (the proxy)
-------------------------
We compress *before the real question exists*, so we probe with the
**document's own queries** — the ``q_proj(hidden_states)`` the layer already
produced — taking the last ``n_probe`` positions (they sit nearest where the
real question will land) and reporting the **worst** (max) error across them,
never the average: a guarantee is a worst-case statement, so one bad probe
must not be hidden by an average. See Rule 5 in the design note for the scope
this buys ("verified against the document's own spotlight").

Assumptions (match the VerifiedSketch constraints)
--------------------------------------------------
* Single-pass ``post_prefill`` prefill, ``attention_method: none`` — so a
  cache slot index equals the token's absolute position, which the causal mask
  below relies on.
* Queries are GQA-pooled to the KV heads (group mean) exactly as Ridge's omega
  does. This is a proxy for the faithful per-query-head attention; a
  per-query-head version is a later refinement (it changes constants, not the
  shape of the signal).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

logger = logging.getLogger(__name__)


@dataclass
class CoverageReport:
    """Result of :func:`measure_output_error`.

    per_head_worst : Tensor [B, H_kv]
        Worst-probe relative output-error for each (batch, kv-head). This is
        the per-head "leftover" that the v2 adaptive split will consume.
    worst : float
        ``per_head_worst`` averaged over heads/batch — a single human-readable
        headline for logging.
    mean : float
        Relative error averaged over every probe and head (less conservative;
        for reference only).
    n_probe : int
        Number of probe queries actually used.
    n_keep : int
        Size of the proposed keep-set (tokens per head).
    n_total : int
        Full cache length ``T``.
    """

    per_head_worst: torch.Tensor
    worst: float
    mean: float
    n_probe: int
    n_keep: int
    n_total: int


def _build_probe_queries(
    module: nn.Module,
    hidden_states: torch.Tensor,
    keys: torch.Tensor,
    kwargs: Optional[dict],
    n_probe: int,
    rotate: bool,
) -> Optional[torch.Tensor]:
    """Return the last ``n_probe`` RoPE-rotated, GQA-pooled queries [B, H_kv, P, D].

    Reuses Ridge's own query construction (q_proj → head reshape → GQA group
    mean → optional RoPE at absolute positions) so the probes match the omega
    path bit-for-bit; only the trailing ``n_probe`` positions are kept.
    Returns ``None`` if queries cannot be built (no ``q_proj`` / head mismatch),
    so the caller can skip measurement gracefully.
    """
    # Imported lazily to avoid a module-load cycle (ridge imports the base,
    # which the compressors package wires up on first registry lookup).
    from eval_harness.kv_compression.compressors.ridge_sketch import RidgeSketch

    B, H_kv, T, D = keys.shape
    builder = RidgeSketch(compression_ratio=0.0, rotate_queries=rotate)
    queries = builder._get_all_queries(
        module=module, hidden_states=hidden_states, keys=keys, kwargs=kwargs or {}
    )
    if queries is None:
        return None
    P = min(n_probe, T)
    return queries[:, :, T - P :, :].contiguous()


@torch.no_grad()
def measure_output_error(
    module: nn.Module,
    hidden_states: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    keep_idx: torch.Tensor,
    kwargs: Optional[dict] = None,
    *,
    n_probe: int = 8,
    rotate: bool = True,
    eps: float = 1e-8,
) -> Optional[CoverageReport]:
    """Worst-case relative output-error of a proposed keep-set.

    Parameters
    ----------
    module, hidden_states, keys, values, kwargs
        The same tensors the compressor's ``compress`` receives. ``keys`` /
        ``values`` are ``[B, H_kv, T, D]`` (RoPE-rotated cache contents).
    keep_idx : Tensor [B, H_kv, M]
        Long indices (into the ``T`` axis) the compressor proposes to KEEP.
        Order is irrelevant; these are treated as a set per head.
    n_probe : int, default 8
        Number of trailing document queries to probe with (recency-weighted by
        construction — the last positions sit where the real question lands).
    rotate : bool, default True
        RoPE-rotate the probe queries so they share the cached keys' rotation
        convention (faithful attention). Falls back to un-rotated with a
        warning if position info is unavailable.

    Returns
    -------
    CoverageReport, or ``None`` if queries could not be built.
    """
    B, H, T, D = keys.shape
    if T == 0 or keep_idx.numel() == 0:
        return None

    q_probe = _build_probe_queries(module, hidden_states, keys, kwargs, n_probe, rotate)
    if q_probe is None:
        return None
    P = q_probe.shape[2]

    # float32 throughout for a stable softmax; the measurement is off the hot
    # path so the extra precision is free.
    q = q_probe.float()
    k = keys.float()
    v = values.float()
    scale = 1.0 / (D ** 0.5)
    neg = torch.finfo(torch.float32).min

    # Absolute positions: single-pass prefill ⇒ cache slot == token position.
    key_pos = torch.arange(T, device=keys.device)  # [T]
    probe_pos = torch.arange(T - P, T, device=keys.device)  # [P]

    def _attend(k_sub: torch.Tensor, v_sub: torch.Tensor, kpos: torch.Tensor) -> torch.Tensor:
        """Softmax(q kᵀ) v with a causal mask; kpos may be per-head [B,H,M] or [T]."""
        logits = torch.einsum("bhpd,bhmd->bhpm", q, k_sub) * scale  # [B,H,P,M]
        if kpos.dim() == 1:
            masked = kpos.view(1, 1, 1, -1) > probe_pos.view(1, 1, P, 1)
        else:
            masked = kpos.unsqueeze(2) > probe_pos.view(1, 1, P, 1)  # [B,H,P,M]
        logits = logits.masked_fill(masked, neg)
        w = torch.softmax(logits, dim=-1)
        # Fully-masked probe rows (no visible key) → zero output rather than NaN.
        w = w.masked_fill(masked, 0.0)
        return torch.einsum("bhpm,bhmd->bhpd", w, v_sub)  # [B,H,P,D]

    out_full = _attend(k, v, key_pos)

    gather = keep_idx.unsqueeze(-1).expand(-1, -1, -1, D)
    k_keep = k.gather(2, gather)
    v_keep = v.gather(2, gather)
    out_keep = _attend(k_keep, v_keep, keep_idx)

    diff = (out_keep - out_full).norm(dim=-1)  # [B,H,P]
    denom = out_full.norm(dim=-1).clamp_min(eps)
    rel = diff / denom  # [B,H,P]

    per_head_worst = rel.max(dim=-1).values  # [B,H] — worst probe per head
    return CoverageReport(
        per_head_worst=per_head_worst,
        worst=float(per_head_worst.mean()),
        mean=float(rel.mean()),
        n_probe=P,
        n_keep=int(keep_idx.shape[-1]),
        n_total=int(T),
    )
