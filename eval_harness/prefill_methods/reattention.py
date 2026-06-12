"""ReAttention prefill method — faithful port of the OpenMOSS baseline.

Reproduces the prefill-time selection of the original ReAttention
implementation (``ReAttention/re_attention/cache_utils_v0921.py``, the
``RECacheV2.update`` method), adapted to the Prism-Test post-attention
forward-hook mechanism.

Reference: "ReAttention: Training-Free Infinite Context with Finite Attention
Scope" (ICLR 2025, arXiv:2407.15176), Liu et al.

Original algorithm (prefill path)
---------------------------------
The KV cache is partitioned into three regions::

    [ global (sink) | middle | local (recent) ]
      global_size          local_size

``global`` and ``local`` tokens are *always* retained.  Middle tokens are
selected by **position-agnostic** top-k attention:

1. Score middle keys with ``Q @ K^T`` where **neither Q nor K carries RoPE**
   (the original cache stores raw, un-rotated K and scores it against the raw
   query ``query_states`` before RoPE is applied — ``recall_type='qk'``).
2. Per (head, query position), take the top ``mid_size`` middle keys
   (``torch.topk(scores, mid_size, dim=middle)``).
3. Flatten across all heads/queries and take the global ``torch.unique`` of
   selected indices.  If more than ``recall_clip`` survive, keep the
   ``recall_clip`` most-frequently-selected ones.
4. **Span expansion**: replace each selected index ``i`` with the window
   ``[i - span_size//2, i + (span_size+1)//2)``, clamp, re-unique, sort.
5. Gather ``[global | selected_middle | local]`` from the cache.

The final attention then applies RoPE to the retained keys using their
**original absolute positions** (``pe_original=True``).

Adaptation to Prism-Test
-------------------------
In Prism-Test's research backend the KV cache stores keys that are **already
RoPE-rotated** (there is no identity-RoPE interceptor — see
``prefill_methods/base.py``).  To recover the position-agnostic scores exactly,
this method:

* **Un-rotates** the cached keys with the very ``(cos, sin)`` used to rotate
  them (taken from ``kwargs['position_embeddings']``), giving the raw ``K``.
* **Re-projects** the raw query from ``hidden_states`` via ``module.q_proj``
  (pre-RoPE), giving the raw ``Q`` — exactly the original ``recall_q``.

By default the retained keys keep their original rotated values, so their
absolute positions are preserved and no re-rotation is needed; subsequent
decode runs standard dense attention against the pruned cache
(``pe_original=True`` in the reference, i.e. ``reposition=False`` here).

Optional compact end-anchored repositioning (``reposition=True``)
----------------------------------------------------------------
ReAttention's selection gives *sparsity* but leaves the surviving keys at their
original, far-apart absolute positions.  Setting ``reposition=True`` adds the
"repositioning" half: the retained per-layer ``R`` keys are re-rotated to the
contiguous, end-anchored positions ``[A - R, A - 1]`` (causal order preserved),
and the question/decode tokens continue at ``[A, A + 1, ...]`` so the decode
query sees bounded relative distances.  End-anchoring (rather than ``0..R-1``)
is required because ``R`` varies per layer while the decode position grid is
global — anchoring the most-recent retained key at ``A - 1`` keeps the
query-to-most-recent distance equal to 1 for every layer.

Scope honesty: in Prism's post-attention-hook architecture the prefill pass's
own attention has already run at the original positions before this hook fires,
so ``reposition=True`` repositions only the *decode-facing* cache (what the
question/generation tokens attend), not the prefill attention itself.  The
architecturally-complete pre-attention treatment (selection + re-rotation
before the attention computes) is what ``reattention_exact.py`` implements
(``pe_original=False``); it is not what this hook does.

This is a *prefill-only* selection method (the base ``_forward_hook`` skips
decode steps), matching ``recall_option='prefill_only'`` semantics of the
baseline.

Uniform retained length (ragged-cache decode fix)
-------------------------------------------------
The original ReAttention replaces every layer's attention forward, so it
tolerates a different retained length per layer.  Prism's research backend
instead decodes through the model's *normal* forward, where HF builds ONE
causal mask / position grid from the layer-0 cache length and shares it across
layers.  Per-layer top-k selection naturally retains a different count per
layer ("ragged" cache), which breaks that contract in two ways:

* a layer retaining MORE than layer 0 ⇒ hard ``RuntimeError`` (mask narrower
  than that layer's keys);
* a layer retaining LESS than layer 0 ⇒ HF silently *slices* the mask to the
  key length, mis-aligning the causal block over the question/decode keys —
  a silent causality leak, no exception.

``uniform_retained`` (default ``True``) therefore equalizes the
*post-span-expansion* middle selection to a single per-prefill target so every
layer retains exactly ``global + target + local`` tokens:

* the target is ``uniform_budget`` when given, else the **first hooked
  layer's** selection size (so layer-0 / single-layer behavior is
  byte-identical to the un-equalized method);
* layers that selected fewer indices are **padded** with the most-recent
  unselected middle tokens (conservative: strictly more context retained);
* layers that selected more are **shrunk** by re-applying the reference's own
  frequency-clip rule at the seed level (largest top-k-by-frequency seed set
  whose span expansion fits), then recency-padded to the exact target.

This is a Prism integration adaptation, not part of the reference algorithm;
set ``uniform_retained=False`` to recover the reference's per-layer (ragged)
selection, which is only safe on single-layer models or custom decode paths.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Set, Tuple

import torch
from torch import nn

from .base import (
    PrefillMethod,
    apply_rotary_pos_emb,
    build_cos_sin,
    get_inv_freq,
    undo_rotary_pos_emb,
)
from .registry import register_prefill_method

logger = logging.getLogger(__name__)

# Fused einsum + top-k Triton kernel (the ReAttention "kernel").  Imported
# lazily/guarded: triton may be missing and the kernel only runs on CUDA.
# Referenced via the module attribute so tests can monkeypatch it.
try:
    from eval_harness.kernels import einsum_topk_func as _einsum_topk_func
except Exception:  # pragma: no cover - triton/CUDA not available
    _einsum_topk_func = None


def _get_num_heads(module: nn.Module) -> Optional[int]:
    """Extract total query head count from an attention module.

    Different model families store this under different names:
    - Llama/Mistral/Qwen: ``module.num_heads`` or ``config.num_attention_heads``
    - Gemma3: ``module.config.num_attention_heads``
    """
    num = getattr(module, "num_heads", None)
    if num is not None:
        return int(num)
    cfg = getattr(module, "config", None)
    if cfg is not None:
        num = getattr(cfg, "num_attention_heads", None)
        if num is not None:
            return int(num)
    return None


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads to query heads (GQA), mirroring HF ``repeat_kv``."""
    if n_rep == 1:
        return x
    B, H_kv, S, D = x.shape
    x = x[:, :, None, :, :].expand(B, H_kv, n_rep, S, D)
    return x.reshape(B, H_kv * n_rep, S, D)


@register_prefill_method("reattention", aliases=["re_attention", "reatt"])
@dataclass
class ReAttentionMethod(PrefillMethod):
    """ReAttention: position-agnostic top-k KV selection during prefill.

    Parameter names match the original ``ReAttentionConfig``.

    Parameters
    ----------
    global_size : int
        Number of initial "sink" tokens always retained (from position 0).
    local_size : int
        Number of recent tokens always retained (the trailing window).
    mid_size : int
        Top-k value: how many middle keys each query selects.  The retained
        middle set is the *union* of every query's top-``mid_size`` picks.
    span_size : int
        Each selected key index is expanded into a window of ``span_size``
        contiguous tokens centred on it (``span_size//2`` before,
        ``(span_size+1)//2`` after).  ``0`` disables span expansion.
    recall_type : str
        Scoring variant: ``'qk'`` (raw Q·K), ``'qkv'`` (Q·K weighted by the
        L1 norm of V), or ``'qkv2'`` (weighted by the L2 norm of V).
    recall_clip : int
        Cap on the number of unique selected middle indices *before* span
        expansion.  ``-1`` means no cap; otherwise the ``recall_clip`` most
        frequently-selected indices are kept.
    use_triton_kernel : str
        Whether to use the fused ``einsum_topk`` Triton kernel for the
        position-agnostic top-k selection (the ReAttention "kernel").

        * ``'auto'`` (default): use the kernel when all of its constraints are
          met — ``mid_size ∈ {1, 4}``, ``head_dim == 128``, CUDA tensors,
          fp16/bf16, and ``n_middle % 128 == 0`` (see ``align_local_to_128``);
          otherwise fall back to the dense ``torch.einsum`` + ``topk`` path.
        * ``'force'``: require the kernel; raise if constraints are unmet.
        * ``'off'``: always use the dense path.

        The kernel mirrors ``cache_utils_v0921.py:618`` of the reference repo.
    align_local_to_128 : bool
        When ``True``, shrink ``local_size`` so the middle region length is a
        multiple of 128 — exactly the alignment trick the reference uses to
        satisfy the kernel's ``seqlen_k % 128 == 0`` constraint.  Default
        ``False`` (preserves the literal ``[global | middle | local]`` split).
    reposition : bool
        Compact end-anchored repositioning of the retained KV cache.  Default
        ``False`` (behavior is byte-identical to the original selection-only
        method).  When ``True``, the retained keys are **re-rotated** from
        their original absolute positions to contiguous, in-window positions
        ``[A - R, A - 1]`` (preserving causal order), and the question/decode
        tokens continue at ``[A, A + 1, ...]``.  This bounds the relative
        distance every decode query sees from the (variable-length) pruned
        cache.

        Mechanism / scope (honest note)
        -------------------------------
        In Prism's post-attention-hook architecture the prefill pass's own
        attention has *already run* at the original positions before this hook
        fires.  This mode therefore repositions only the **decode-facing**
        cache — i.e. it changes the positions that the question/generation
        tokens (and ``compute_question_position_ids``) attend with, not the
        positions used inside the prefill attention itself.  The
        architecturally-complete *pre-attention* version (re-rotating before
        the prefill attention computes) is what ``ReAttentionExactMethod``
        implements (``pe_original=False``) and is **not** what this hook does.
    reposition_window : Optional[int]
        Explicit anchor ``A`` (the contiguous-window upper bound) for
        repositioning.  When ``None`` and ``reposition`` is on, ``A`` is
        derived from ``recall_clip`` as a safe upper bound on the retained
        count ``R`` (see ``_reposition_anchor``).
    uniform_retained : bool
        Equalize the retained length across layers (default ``True``).  HF's
        normal decode shares one causal mask / position grid across all layers
        (sized from layer 0), so the per-layer top-k selection's naturally
        *ragged* cache either crashes decode (a layer longer than layer 0) or
        silently mis-aligns the causal mask (a layer shorter than layer 0).
        With this on, every hooked layer retains exactly
        ``global_size + target + local`` tokens, where ``target`` is the
        per-prefill middle target (see ``uniform_budget``).  The first hooked
        layer is never altered when the target is derived from it, so
        single-layer behavior is byte-identical to ``uniform_retained=False``.
        This is a Prism integration adaptation — the reference tolerates the
        ragged cache because it replaces every layer's attention forward.
    uniform_budget : Optional[int]
        Explicit per-layer middle target (post-span-expansion count of
        retained *middle* tokens) used when ``uniform_retained`` is on.
        ``None`` (default) derives the target from the first hooked layer's
        own selection size.  Clamped to the middle length; must be positive.
    """

    global_size: int = 32
    local_size: int = 4096
    mid_size: int = 4
    span_size: int = 32
    recall_type: str = "qk"
    recall_clip: int = -1
    use_triton_kernel: str = "auto"
    align_local_to_128: bool = False
    reposition: bool = False
    reposition_window: Optional[int] = None
    uniform_retained: bool = True
    uniform_budget: Optional[int] = None

    # Populated lazily when the context manager installs hooks.
    _inv_freq: Optional[torch.Tensor] = None
    # Per-prefill uniform middle target; reset by ``on_prefill_start`` and
    # established by ``uniform_budget`` or the first hooked layer's selection.
    _uniform_mid_target: Optional[int] = None

    # ------------------------------------------------------------------
    # Lifecycle: capture inv_freq as a fallback RoPE source
    # ------------------------------------------------------------------

    def on_prefill_start(self, total_context_length: int) -> None:
        # Each prefill pass establishes its own uniform middle target (the
        # explicit budget, or the first hooked layer's selection size).
        self._uniform_mid_target = None

    def __call__(self, model):  # type: ignore[override]
        # Capture the model's RoPE frequencies so we can un-rotate cached K
        # even when ``position_embeddings`` is absent from the hook kwargs.
        try:
            self._inv_freq = get_inv_freq(model)
        except Exception:
            self._inv_freq = None
        return super().__call__(model)

    # ------------------------------------------------------------------
    # Core selection
    # ------------------------------------------------------------------

    def prefill_forward_hook(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        kwargs: dict,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        B, H_kv, S, D = keys.shape

        # check_recall: nothing to do until the cache exceeds global + local.
        if S <= self.global_size + self.local_size:
            return None

        # Recover position-agnostic raw K once.  Needed for scoring, and also
        # for re-rotation when ``reposition`` is on — so compute it up front so
        # every return path (including the StreamingLLM and key-norm fallbacks)
        # can feed the shared ``_finalize`` (cheap: a single un-rotation).
        raw_k = self._unrotate_keys(keys, kwargs)  # [B, H_kv, S, D]

        if self.mid_size <= 0:
            # Pure StreamingLLM-style retention: [global | local], no middle.
            all_idx = self._sink_local_indices(B, H_kv, S)
            return self._finalize(keys, values, raw_k, all_idx, kwargs)

        # Optionally shrink the local window so the middle length is a multiple
        # of 128 (the reference's kernel-alignment trick).
        local_size = self._effective_local_size(S)

        middle_start = self.global_size
        middle_end = S - local_size
        n_middle = middle_end - middle_start
        if n_middle <= 0:
            return None

        # --- Recover position-agnostic raw Q -------------------------------
        raw_q = self._raw_query(module, hidden_states, D)  # [B, H_q, q_len, D] or None

        if raw_q is None:
            # No q_proj available → fall back to key-norm scoring on raw K.
            scores = raw_k[:, :, middle_start:middle_end, :].norm(dim=-1)  # [B, H_kv, n_middle]
            k = min(self.mid_size, n_middle)
            _, topk_idx = scores.unsqueeze(2).topk(k, dim=-1)  # [B, H_kv, 1, k]
        else:
            recall_k = self._apply_recall_weight(raw_k, values)
            recall_k_mid = recall_k[:, :, middle_start:middle_end, :]  # [B, H_kv, n_middle, D]
            # Per (head, query) top-mid_size — kernel fast path or dense.
            topk_idx = self._select_topk(raw_q, recall_k_mid, n_middle)  # [B, H_q, q_len, k]

        # --- Global unique + recall_clip (matches the prefill update path) -
        sel_rel = self._global_unique(topk_idx, self.recall_clip)  # [n_unique]

        # --- Span expansion -------------------------------------------------
        sel_rel = self._expand_spans(sel_rel, n_middle)  # sorted, unique, [n_sel]

        # --- Layer-uniform retained length (ragged-cache decode fix) -------
        if self.uniform_retained:
            sel_rel = self._apply_uniform_budget(sel_rel, topk_idx, n_middle)

        # --- Assemble [global | selected_middle | local] indices -----------
        device = keys.device
        global_idx = torch.arange(0, self.global_size, device=device)
        mid_idx = sel_rel + self.global_size
        local_idx = torch.arange(S - local_size, S, device=device)
        all_idx = torch.cat([global_idx, mid_idx, local_idx])  # globally ascending

        all_idx = all_idx.view(1, 1, -1).expand(B, H_kv, -1)
        return self._finalize(keys, values, raw_k, all_idx, kwargs)

    # ------------------------------------------------------------------
    # Top-k selection: fused Triton kernel fast path + dense fallback
    # ------------------------------------------------------------------

    def _effective_local_size(self, S: int) -> int:
        """Local-window size, optionally 128-aligned so n_middle % 128 == 0."""
        local_size = self.local_size
        if not self.align_local_to_128:
            return local_size
        raw_mid = S - self.global_size - local_size
        if raw_mid <= 0:
            return local_size
        pad = (-raw_mid) % 128  # grow middle to the next multiple of 128
        return max(0, local_size - pad)

    def _select_topk(
        self, raw_q: torch.Tensor, recall_k_mid: torch.Tensor, n_middle: int,
    ) -> torch.Tensor:
        """Per (query head, query position) top-``mid_size`` middle indices.

        Dispatches to the fused ReAttention Triton kernel when its constraints
        are satisfied (mirroring the reference), else the dense path.  Both
        return ``[B, H_q, q_len, k]`` *relative* middle indices (long), so they
        feed the same ``_global_unique`` + span-expansion machinery.
        """
        if self._should_use_kernel(raw_q, recall_k_mid, n_middle):
            try:
                return self._kernel_topk(raw_q, recall_k_mid)
            except Exception as exc:  # pragma: no cover - GPU-only path
                if (self.use_triton_kernel or "auto").lower() == "force":
                    raise
                logger.warning(
                    "ReAttention Triton kernel failed (%s); falling back to dense.",
                    exc,
                )
        return self._dense_topk(raw_q, recall_k_mid, n_middle)

    def _should_use_kernel(
        self, raw_q: torch.Tensor, recall_k_mid: torch.Tensor, n_middle: int,
    ) -> bool:
        """Whether the fused einsum-topk kernel can/should run.

        Kernel constraints (from ``kernels/einsum_topk.py``): ``mid_size`` is 1
        or 4, CUDA tensors, ``head_dim == 128``, fp16/bf16-castable, and both
        the query length and ``n_middle`` are multiples of 128.  The query
        length is handled by padding in ``_kernel_topk``, so only ``n_middle``
        is gated here.
        """
        mode = (self.use_triton_kernel or "auto").lower()
        if mode == "off":
            return False
        if _einsum_topk_func is None:
            if mode == "force":
                raise RuntimeError(
                    "use_triton_kernel='force' but the Triton kernel is unavailable "
                    "(triton not importable).",
                )
            return False

        D = raw_q.shape[-1]
        ok = (
            self.mid_size in (1, 4)
            and bool(raw_q.is_cuda)
            and bool(recall_k_mid.is_cuda)
            and D == 128
            and n_middle > 0
            and n_middle % 128 == 0
        )
        if mode == "force" and not ok:
            raise RuntimeError(
                "use_triton_kernel='force' but kernel constraints are unmet: "
                f"mid_size={self.mid_size} (need 1 or 4), cuda={raw_q.is_cuda}, "
                f"head_dim={D} (need 128), n_middle={n_middle} (need %128==0). "
                "Set align_local_to_128=True to 128-align the middle region.",
            )
        return ok

    def _kernel_topk(
        self, raw_q: torch.Tensor, recall_k_mid: torch.Tensor,
    ) -> torch.Tensor:
        """Fused einsum + top-k via the Triton kernel.

        Pads the query length to a multiple of 128 (the kernel has no M-axis
        boundary mask) and drops the padded rows from the result.  GQA is
        handled inside the kernel (``num_heads // num_kv_heads``), so K is
        passed per-KV-head — *not* repeat_kv'd.
        """
        B, H_q, q_len, D = raw_q.shape
        topk = self.mid_size

        dtype = raw_q.dtype if raw_q.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
        q = raw_q.to(dtype)
        k = recall_k_mid.to(dtype)

        pad_q = (-q_len) % 128
        if pad_q:
            q = torch.cat([q, q.new_zeros(B, H_q, pad_q, D)], dim=2)

        q = q.contiguous()
        k = k.contiguous()
        idx = _einsum_topk_func(q, k, topk)  # [B, H_q, q_len_pad, topk] int32
        return idx[:, :, :q_len, :].long()

    def _dense_topk(
        self, raw_q: torch.Tensor, recall_k_mid: torch.Tensor, n_middle: int,
    ) -> torch.Tensor:
        """Dense ``torch.einsum`` + ``torch.topk`` selection (universal path)."""
        B, H_kv = recall_k_mid.shape[0], recall_k_mid.shape[1]
        H_q = raw_q.shape[1]
        n_rep = max(1, H_q // H_kv)
        rk = _repeat_kv(recall_k_mid, n_rep)  # [B, H_q, n_middle, D]
        # Position-agnostic affinity:  bhqd,bhmd -> bhqm  (per query × middle)
        scores = torch.einsum("bhqd,bhmd->bhqm", raw_q.float(), rk.float())
        k = min(self.mid_size, n_middle)
        _, topk_idx = scores.topk(k, dim=-1)  # [B, H_q, q_len, k] relative idx
        return topk_idx

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sink_local_indices(self, B: int, H_kv: int, S: int) -> torch.Tensor:
        device = torch.device("cpu")
        # device fixed at gather time; build on caller device instead.
        global_idx = torch.arange(0, self.global_size)
        local_idx = torch.arange(S - self.local_size, S)
        idx = torch.cat([global_idx, local_idx])
        return idx.view(1, 1, -1).expand(B, H_kv, -1)

    @staticmethod
    def _gather(
        keys: torch.Tensor, values: torch.Tensor, idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = idx.to(keys.device)
        D = keys.shape[-1]
        gather_idx = idx.unsqueeze(-1).expand(-1, -1, -1, D)
        new_keys = keys.gather(2, gather_idx).contiguous()
        new_values = values.gather(2, gather_idx).contiguous()
        return new_keys, new_values

    # ------------------------------------------------------------------
    # Layer-uniform retained length (ragged-cache decode fix)
    # ------------------------------------------------------------------

    def _apply_uniform_budget(
        self, sel_rel: torch.Tensor, topk_idx: torch.Tensor, n_middle: int,
    ) -> torch.Tensor:
        """Pad or shrink the expanded selection to the per-prefill target.

        ``sel_rel`` is the sorted, unique, span-expanded middle selection of
        the current layer; ``topk_idx`` the raw per-(head, query) picks it was
        derived from (needed to re-rank seeds by frequency when shrinking).
        Returns a sorted, unique selection of exactly ``target`` indices, so
        every hooked layer retains ``global + target + local`` tokens.
        """
        n_sel = int(sel_rel.numel())
        target = self._resolve_uniform_target(n_sel, n_middle)
        if n_sel == target:
            return sel_rel
        if n_sel < target:
            return self._pad_selection(sel_rel, target, n_middle)
        return self._shrink_selection(topk_idx, target, n_middle)

    def _resolve_uniform_target(self, n_sel: int, n_middle: int) -> int:
        """The uniform middle target for this prefill pass.

        ``uniform_budget`` wins when set; otherwise the first hooked layer's
        selection size becomes the target (leaving that layer untouched).
        Clamped to ``n_middle``, which is identical for every layer in a
        prefill pass, so the stored target is stable across layers.
        """
        if self.uniform_budget is not None:
            target = int(self.uniform_budget)
            if target <= 0:
                raise ValueError(
                    f"ReAttention uniform_budget must be positive, got {target}",
                )
        elif self._uniform_mid_target is None:
            target = n_sel
        else:
            target = int(self._uniform_mid_target)
        target = min(target, n_middle)
        self._uniform_mid_target = target
        return target

    def _pad_selection(
        self, sel_rel: torch.Tensor, target: int, n_middle: int,
    ) -> torch.Tensor:
        """Grow the selection to ``target`` with the most-recent unselected
        middle indices (the tokens just before the local window) — strictly
        more context retained, never less."""
        device = sel_rel.device
        selected = torch.zeros(n_middle, dtype=torch.bool, device=device)
        selected[sel_rel] = True
        unselected = (~selected).nonzero(as_tuple=True)[0]  # ascending
        extra = unselected[unselected.numel() - (target - sel_rel.numel()):]
        return torch.sort(torch.cat([sel_rel, extra])).values

    def _shrink_selection(
        self, topk_idx: torch.Tensor, target: int, n_middle: int,
    ) -> torch.Tensor:
        """Shrink an over-budget selection to exactly ``target`` indices.

        Re-applies the reference's own frequency-clip rule at the seed level:
        keep the largest top-``k``-by-frequency seed prefix whose span
        expansion still fits in ``target`` (binary search — expansion size is
        monotone in ``k``), then recency-pad the remainder to the exact
        target.  If even the single most-frequent seed's span over-fills the
        target, keep the ``target`` expanded indices closest to that seed.
        """
        flat = topk_idx.reshape(-1)
        uniq, counts = torch.unique(flat, return_counts=True)
        if (
            self.recall_clip is not None
            and 0 <= self.recall_clip < uniq.numel()
        ):
            # Identical clip call to _global_unique: the candidate pool must be
            # exactly the seeds the layer's own selection kept.  (torch.topk
            # and a stable argsort break frequency TIES differently, so
            # re-deriving the clip from a sorted ranking could re-introduce
            # seeds the actual recall_clip discarded.)
            _, keep = torch.topk(counts, k=self.recall_clip)
            keep = torch.sort(keep).values  # restore ascending-index order
            uniq, counts = uniq[keep], counts[keep]
        order = torch.argsort(counts, descending=True, stable=True)
        ranked = uniq[order]  # most-frequent first; ties broken by index

        lo, hi = 0, int(ranked.numel())
        best = ranked[:0]
        while lo <= hi:
            k = (lo + hi) // 2
            expanded = self._expand_spans(ranked[:k], n_middle)
            if expanded.numel() <= target:
                best = expanded
                lo = k + 1
            else:
                hi = k - 1

        if best.numel() == 0:
            seed = ranked[:1]
            expanded = self._expand_spans(seed, n_middle)
            dist = (expanded - seed).abs()
            keep = torch.argsort(dist, stable=True)[:target]
            return torch.sort(expanded[keep]).values
        if best.numel() < target:
            return self._pad_selection(best, target, n_middle)
        return best

    # ------------------------------------------------------------------
    # Compact end-anchored repositioning
    # ------------------------------------------------------------------

    def _reposition_anchor(self) -> int:
        """Anchor ``A`` for the compacted window — deterministic from config.

        ``A`` must be uniform across layers (it must NOT depend on the
        per-layer retained count ``R``), so the global decode position grid is
        consistent across layers.  Resolution order:

        * ``reposition_window`` if set (explicit upper bound on ``R``);
        * else ``global_size + local_size + recall_clip * max(1, span_size)``
          when ``recall_clip > 0`` — a safe upper bound on ``R`` (global +
          local always retained; the middle is at most ``recall_clip`` unique
          picks, each expanded to at most ``span_size`` tokens);
        * else raise — the compacted window is otherwise unbounded.
        """
        if self.reposition_window is not None:
            return int(self.reposition_window)
        if self.recall_clip is not None and self.recall_clip > 0:
            return int(
                self.global_size
                + self.local_size
                + self.recall_clip * max(1, self.span_size)
            )
        raise ValueError(
            "ReAttention reposition=True requires reposition_window or "
            "recall_clip>0 to bound the compacted window",
        )

    def _finalize(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        raw_k: torch.Tensor,
        all_idx: torch.Tensor,
        kwargs: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gather the retained KV and, if ``reposition`` is on, re-rotate.

        ``all_idx`` is ``[B, H_kv, R]`` of globally-ascending absolute indices.
        Values carry no RoPE, so they are always gathered from the rotated
        cache directly.  When ``reposition`` is off this returns exactly the
        original ``_gather`` result (byte-identical).  When on, the retained
        keys are re-rotated from their original absolute positions to the
        contiguous, end-anchored positions ``[A - R, A - 1]``.
        """
        gathered_keys, gathered_values = self._gather(keys, values, all_idx)
        if not self.reposition:
            return gathered_keys, gathered_values

        if self._inv_freq is None:
            raise ValueError(
                "ReAttention reposition=True requires inv_freq to re-rotate the "
                "retained keys, but self._inv_freq is None (__call__ should "
                "have populated it from the model's rotary embedding).",
            )

        device = keys.device
        dtype = keys.dtype
        A = self._reposition_anchor()
        R = int(all_idx.shape[-1])
        if R > A:
            raise ValueError(
                f"ReAttention reposition: retained count R={R} exceeds the "
                f"compacted-window anchor A={A}; increase reposition_window or "
                "recall_clip/span_size (or lower uniform_budget) so A >= R.",
            )

        # Gather the position-agnostic raw keys at the retained indices, then
        # re-rotate to the compacted positions [A - R, A - 1] (ascending,
        # preserving the global causal order encoded in all_idx).
        D = raw_k.shape[-1]
        gather_idx = all_idx.to(device).unsqueeze(-1).expand(-1, -1, -1, D)
        raw_k_ret = raw_k.gather(2, gather_idx).contiguous()  # [B, H_kv, R, D]

        new_pos = torch.arange(A - R, A, device=device)  # [R]
        cos, sin = build_cos_sin(
            new_pos, self._inv_freq.to(device), device, torch.float32,
        )
        repositioned_keys = apply_rotary_pos_emb(
            raw_k_ret, cos.to(dtype), sin.to(dtype),
        ).contiguous()
        return repositioned_keys, gathered_values

    def compute_question_position_ids(
        self,
        context_length: int,
        question_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Position IDs for the question/decode tokens.

        When ``reposition`` is off, or the context was too short to reposition
        (``S <= global_size + local_size`` → the hook returned ``None`` and the
        cache keeps its original positions), the query continues from
        ``context_length`` as in the base method.  Otherwise the cache has been
        compacted into ``[A - R, A - 1]``, so the query continues from the
        anchor ``A``.
        """
        if (not self.reposition) or (
            context_length <= self.global_size + self.local_size
        ):
            return torch.arange(
                context_length,
                context_length + question_length,
                device=device,
            ).unsqueeze(0)
        A = self._reposition_anchor()
        return torch.arange(A, A + question_length, device=device).unsqueeze(0)

    def _unrotate_keys(self, keys: torch.Tensor, kwargs: dict) -> torch.Tensor:
        """Recover raw (pre-RoPE) keys from the rotated cache.

        Prefers the exact ``(cos, sin)`` from ``kwargs['position_embeddings']``
        (these are what rotated the keys in this forward pass).  Falls back to
        rebuilding them from ``inv_freq`` over positions ``0..S-1``, and as a
        last resort returns the rotated keys unchanged.
        """
        B, H_kv, S, D = keys.shape
        cos = sin = None

        pos_emb = kwargs.get("position_embeddings")
        if isinstance(pos_emb, (tuple, list)) and len(pos_emb) == 2:
            c, s = pos_emb
            if c is not None and s is not None and c.shape[-1] == D and c.shape[-2] == S:
                cos = c.to(keys.dtype)
                sin = s.to(keys.dtype)
                if cos.dim() == 3:  # [B, S, D] -> [B, 1, S, D]
                    cos = cos.unsqueeze(1)
                    sin = sin.unsqueeze(1)

        if cos is None and self._inv_freq is not None:
            pos = torch.arange(S, device=keys.device)
            cos, sin = build_cos_sin(pos, self._inv_freq.to(keys.device), keys.device, torch.float32)
            cos = cos.to(keys.dtype)
            sin = sin.to(keys.dtype)

        if cos is None:
            logger.warning(
                "ReAttention: no RoPE source available; scoring against rotated keys "
                "(position-agnostic recovery skipped).",
            )
            return keys

        return undo_rotary_pos_emb(keys, cos, sin)

    def _raw_query(
        self, module: nn.Module, hidden_states: torch.Tensor, head_dim: int,
    ) -> Optional[torch.Tensor]:
        """Re-project the raw (pre-RoPE) query from hidden_states.

        Returns ``[B, H_q, q_len, head_dim]`` or ``None`` if unavailable.
        """
        if not hasattr(module, "q_proj"):
            return None
        num_heads = _get_num_heads(module)
        B, q_len, _ = hidden_states.shape
        raw_q = module.q_proj(hidden_states)  # [B, q_len, H_q * head_dim]
        if num_heads is None:
            num_heads = raw_q.shape[-1] // head_dim
        raw_q = raw_q.view(B, q_len, num_heads, head_dim)

        # Apply QK-norm if the model uses it (Gemma3, Qwen3) — keeps the raw Q
        # consistent with how the cached K was normalized before rotation.
        q_norm = getattr(module, "q_norm", None)
        if q_norm is not None:
            try:
                raw_q = q_norm(raw_q)
            except Exception:
                pass

        return raw_q.transpose(1, 2)  # [B, H_q, q_len, head_dim]

    def _apply_recall_weight(
        self, raw_k: torch.Tensor, values: torch.Tensor,
    ) -> torch.Tensor:
        """Weight raw keys by value norms for the qkv / qkv2 recall variants."""
        rt = self.recall_type
        if rt in ("qk", "qk_pe"):
            return raw_k
        if rt in ("qkv", "qkv_pe"):
            return raw_k * values.norm(p=1, dim=-1, keepdim=True)
        if rt in ("qkv2", "qkv2_pe"):
            return raw_k * values.norm(p=2, dim=-1, keepdim=True)
        raise ValueError(
            f"Unsupported recall_type '{rt}'. Use one of: qk, qkv, qkv2.",
        )

    @staticmethod
    def _global_unique(topk_idx: torch.Tensor, recall_clip: int) -> torch.Tensor:
        """Flatten per-query top-k picks into a single deduped index set.

        Mirrors the original prefill update: ``torch.unique`` across all heads
        and queries, optionally capped to the ``recall_clip`` most-frequently
        selected indices.
        """
        flat = topk_idx.reshape(-1)
        if recall_clip is None or recall_clip < 0:
            return torch.unique(flat)
        uniq, counts = torch.unique(flat, return_counts=True)
        if uniq.numel() > recall_clip:
            _, keep = torch.topk(counts, k=recall_clip)
            uniq = uniq[keep]
        return uniq

    def _expand_spans(self, sel_rel: torch.Tensor, n_middle: int) -> torch.Tensor:
        """Expand each selected index into a contiguous span; re-unique + sort."""
        if self.span_size and self.span_size > 0:
            offsets = torch.arange(
                -(self.span_size // 2),
                (self.span_size + 1) // 2,
                device=sel_rel.device,
            )
            expanded = (sel_rel[:, None] + offsets[None, :]).reshape(-1)
        else:
            expanded = sel_rel
        expanded = expanded.clamp(0, n_middle - 1)
        # torch.unique returns sorted ascending values → causal ordering.
        return torch.unique(expanded)

    def supported_backends(self) -> Set[str]:
        return {"research"}
