import logging
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from transformers.models.llama.modeling_llama import rotate_half

from eval_harness.kv_compression.base import KVCompressor
from eval_harness.kv_compression.registry import register_kv_compressor

logger = logging.getLogger(__name__)


@register_kv_compressor("ridge")
@dataclass
class RidgeSketch(KVCompressor):
    """
    Value-aware query-ridge KV compression (fixed-envelope scoring).

    Port of ``RidgePress`` (kvpress/presses/ridge_press.py; a research-fork
    addition in the local kvpress 0.5.1 checkout, not upstream NVIDIA kvpress),
    reduced to the reference's default production path: fixed-envelope score
    combination, top-k selection, mean-normalized query Gram, queries taken at
    the key positions, and sum-normalized score components. The reference's
    alpha-selection machinery (entropy / tail_risk / query_constrained /
    gated_query_constrained), its additive / multiplicative / plain-envelope /
    weighted-envelope combine modes, multinomial selection, and the dormant
    query-metric ablation helpers were unreachable under every production
    configuration and have been removed.

    Scores over the middle region:
      tau_i   = ridge leverage of the L2-normalized key directions
      omega_i = query-key interaction ||Q k_i||_2 (raw keys)
      ||v_i|| = value norm

      score_i = max(tau_c_i, envelope_gamma * omega_c_i) * ||v_i||^value_norm_power
      with tau_c = tau / sum(tau) and omega_c = omega / sum(omega).

    The first ``sink_size`` and last ``local_size`` tokens are always kept.
    Scoring and selection apply to the middle region only; per (batch, kv-head)
    row, ``keep_mid = int(T * (1 - compression_ratio)) - sink - local`` middle
    tokens are kept (indices sorted ascending so temporal order is preserved).
    The kept count is a pure function of ``T`` and the hyperparameters, so the
    cache stays rectangular across heads and layers; only the kept token
    positions differ per head.

    Notes
    -----
    Upstream quirks replicated faithfully:
    - ``compression_ratio=None`` raises at compress time (the research adapter
      injects the adapter-level float when built from config).
    - When ``keep_total < sink + local`` the ``[sink | local]`` concatenation
      is returned, so the kept count can EXCEED the nominal budget
      ``int(T * (1 - compression_ratio))`` (under-compression).
    - Queries come from ``module.q_proj`` WITHOUT rotary applied while the
      cached keys are RoPE-rotated: omega mixes unrotated Q with rotated K.
      This matches the reference; see ``rotate_queries`` for the opt-in fix.
      Under DCA's cyclic-rotated keys the semantics shift further, so this
      sketch is intended for ``attention_method: none``.

    Deviations from kvpress
    -----------------------
    - **Key normalization (2026-07)**: keys are L2-normalized along the head
      dim (in float32, eps-clamped) BEFORE the ridge leverage computation, so
      tau measures direction diversity and is invariant to per-token key
      magnitude. The reference computes leverage on raw rotated keys. omega
      and the value norms still see raw magnitudes: ``||Q k_i||`` is the
      attention-logit-energy proxy, where key magnitude scales the actual
      logits and is physically meaningful.
    - Kept-window defaults ``sink_size=8, local_size=64`` deviate from
      upstream RidgePress (sink=4, local=28): local=28 was an anomalously
      small guaranteed-recency window. Pass ``sink_size=4, local_size=28`` to
      reproduce the upstream window layout (scores still differ due to the
      key-normalization deviation above).
    - ``rotate_queries`` (default False = upstream behavior): when True, RoPE
      is applied to the re-projected queries (``position_embeddings`` from the
      layer forward kwargs, or rebuilt from ``module.rotary_emb``) before
      omega is formed, making ``omega_i = ||Q k_i||`` a faithful proxy for the
      model's relative-position attention energy. Partial rotary (Qwen3.5) is
      handled by rotating only the first ``rotary_dim`` channels. If position
      embeddings are unavailable the path warns and falls back to un-rotated.
    - ``_get_all_queries`` skips the query-aware path (warning + tau-only
      fallback) when ``hidden_states`` and ``keys`` cover different numbers of
      tokens. Upstream assumes they match and would misalign (or crash on the
      reshape) otherwise; in Prism-Test an outer prefill-method hook (e.g.
      reattention) fires before the sketch hook and can leave the cached keys
      shorter than ``hidden_states``.
    """

    compression_ratio: Optional[float] = None
    ridge_lambda: float = 1e-4
    sink_size: int = 8
    local_size: int = 64
    min_tokens_to_compress: int = 64

    # Query-aware options.
    query_aware: bool = True
    rotate_queries: bool = False

    # Fixed-envelope score combination:
    # score_i = max(tau_c_i, envelope_gamma * omega_c_i), envelope_gamma >= 0.
    envelope_gamma: float = 1.0

    value_norm_power: float = 1.0
    eps: float = 1e-8

    def __post_init__(self):
        super().__post_init__()
        if self.compression_ratio is not None:
            assert 0.0 <= self.compression_ratio < 1.0, "compression_ratio must be in [0, 1)"
        assert self.ridge_lambda > 0, "ridge_lambda must be > 0"
        assert self.sink_size >= 0, "sink_size must be >= 0"
        assert self.local_size >= 0, "local_size must be >= 0"
        assert self.min_tokens_to_compress >= 0, "min_tokens_to_compress must be >= 0"
        assert self.envelope_gamma >= 0.0, "envelope_gamma must be >= 0"
        assert self.value_norm_power >= 0, "value_norm_power must be >= 0"

    def _compute_key_ridge_tau(self, keys: torch.Tensor) -> torch.Tensor:
        """tau_i = k_i^T (K^T K + lambda I)^(-1) k_i over L2-normalized keys."""
        B, H, N, D = keys.shape
        if N == 0:
            return torch.zeros(B, H, 0, device=keys.device, dtype=keys.dtype)

        k = keys.float()
        k = k / k.norm(p=2, dim=-1, keepdim=True).clamp_min(self.eps)
        eye = torch.eye(D, device=k.device, dtype=k.dtype).view(1, 1, D, D)
        gram = k.transpose(-2, -1) @ k
        reg = gram + self.ridge_lambda * eye

        try:
            inv_reg = torch.linalg.inv(reg)
        except torch.linalg.LinAlgError:
            inv_reg = torch.linalg.pinv(reg)

        tau = ((k @ inv_reg) * k).sum(dim=-1).clamp_min(0.0)
        return tau.to(keys.dtype)

    def _position_embeddings(
        self, module: nn.Module, hidden_states: torch.Tensor, kwargs: dict
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """(cos, sin) from the layer forward kwargs, else rebuilt from rotary_emb."""
        pos_emb = kwargs.get("position_embeddings") if kwargs else None
        if isinstance(pos_emb, (tuple, list)) and len(pos_emb) == 2 and pos_emb[0] is not None:
            return pos_emb[0], pos_emb[1]

        rotary = getattr(module, "rotary_emb", None)
        if rotary is None:
            return None
        # NOTE: cache_position is the PHYSICAL cache slot. Before any pruning
        # it coincides with the absolute position; after decode-time pruning
        # (streaming_ridge) it diverges, so this fallback would rotate at the
        # wrong angle. The primary path (position_embeddings kwarg, passed by
        # HF from the pipeline's absolute position_ids) is always correct.
        cache_position = kwargs.get("cache_position") if kwargs else None
        if cache_position is not None:
            position_ids = cache_position.unsqueeze(0)
        else:
            position_ids = torch.arange(
                hidden_states.shape[-2], device=hidden_states.device
            ).unsqueeze(0)
        cos, sin = rotary(hidden_states, position_ids)
        return cos, sin

    def _apply_rope_to_queries(
        self, q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """RoPE-rotate queries [B, H, T, D] at absolute positions (partial-rotary aware)."""
        q_len = q.shape[-2]
        cos_q = cos[:, -q_len:, :].unsqueeze(1).to(q.dtype)
        sin_q = sin[:, -q_len:, :].unsqueeze(1).to(q.dtype)
        rotary_dim = cos_q.shape[-1]
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        q_rot = (q_rot * cos_q) + (rotate_half(q_rot) * sin_q)
        return torch.cat([q_rot, q_pass], dim=-1)

    def _project_and_pool_queries(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        H_kv: int,
        D: int,
        dtype: torch.dtype,
        kwargs: Optional[dict] = None,
    ) -> Optional[torch.Tensor]:
        """q_proj + head reshape + GQA pooling (+ optional RoPE): [B, H_kv, T, D].

        No length checks — callers align hidden_states/keys themselves. Shared
        by the batch path (`_get_all_queries`) and the streaming decode path
        (per-step accumulation in StreamingRidgeSketch).
        """
        try:
            q = module.q_proj(hidden_states)
        except Exception as exc:
            logger.warning("Could not compute queries from q_proj: %s", exc)
            return None

        if q.shape[-1] % D != 0:
            logger.warning("q_proj output dim %s is not divisible by head_dim %s", q.shape[-1], D)
            return None

        B, T = hidden_states.shape[0], hidden_states.shape[1]
        H_q = q.shape[-1] // D
        q = q.view(B, T, H_q, D).transpose(1, 2).contiguous()

        if H_q == H_kv:
            pass
        elif H_q > H_kv and H_q % H_kv == 0:
            group = H_q // H_kv
            q = q.view(B, H_kv, group, T, D).mean(dim=2)
        elif H_kv > H_q and H_kv % H_q == 0:
            repeat = H_kv // H_q
            q = q.repeat_interleave(repeat, dim=1)
        else:
            logger.warning("Incompatible query/KV head counts: H_q=%s, H_kv=%s", H_q, H_kv)
            return None

        q = q.to(dtype)

        if self.rotate_queries:
            cos_sin = self._position_embeddings(module, hidden_states, kwargs or {})
            if cos_sin is not None:
                q = self._apply_rope_to_queries(q, cos_sin[0], cos_sin[1])
            else:
                logger.warning(
                    "rotate_queries=True but position embeddings are unavailable; "
                    "falling back to un-rotated queries."
                )

        return q.contiguous()

    def _get_all_queries(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        kwargs: Optional[dict] = None,
    ) -> Optional[torch.Tensor]:
        """Return prefill queries aligned with KV heads: [B, H_kv, T, D]."""
        if hidden_states is None or not hasattr(module, "q_proj"):
            return None

        B, H_kv, T, D = keys.shape

        if hidden_states.shape[1] != T:
            logger.warning(
                "Query/key token mismatch (hidden_states has %s tokens, keys have %s); "
                "skipping query-aware scoring.",
                hidden_states.shape[1],
                T,
            )
            return None

        return self._project_and_pool_queries(
            module, hidden_states, H_kv, D, keys.dtype, kwargs,
        )

    def _compute_query_key_interaction(
        self,
        keys_mid: torch.Tensor,
        queries: torch.Tensor,
    ) -> torch.Tensor:
        """omega_i = ||Q k_i||_2 = sqrt(k_i^T Q^T Q k_i) with a mean-normalized Gram."""
        B, H, N, _ = keys_mid.shape
        T = queries.shape[2]
        if N == 0:
            return torch.zeros(B, H, 0, device=keys_mid.device, dtype=keys_mid.dtype)
        if T == 0:
            return torch.ones(B, H, N, device=keys_mid.device, dtype=keys_mid.dtype)

        k = keys_mid.float()
        q = queries.float()

        G_q = q.transpose(-2, -1) @ q
        G_q = G_q / max(T, 1)

        omega = ((k @ G_q) * k).sum(dim=-1).clamp_min(0.0).sqrt()
        return omega.to(keys_mid.dtype)

    def _normalize_distribution(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float().clamp_min(self.eps)
        return x / x.sum(dim=-1, keepdim=True).clamp_min(self.eps)

    def _scores_from_tau_omega_and_values(
        self,
        tau: torch.Tensor,
        values: torch.Tensor,
        omega: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Fixed-envelope value-aware scores:

            score_i = max(tau_c_i, envelope_gamma * omega_c_i) * ||v_i||^p

        with tau_c = tau / sum(tau), omega_c = omega / sum(omega). When omega
        is unavailable (query-aware scoring skipped) the score is tau_c only.
        """
        tau_f = tau.float().clamp_min(self.eps)

        value_norms = values.float().norm(p=2, dim=-1).clamp_min(self.eps)
        if self.value_norm_power > 0:
            vweight = value_norms.pow(self.value_norm_power)
        else:
            vweight = torch.ones_like(value_norms)

        tau_c = self._normalize_distribution(tau_f)
        if omega is None:
            return tau_c * vweight

        omega_f = omega.float().clamp_min(self.eps)
        omega_c = self._normalize_distribution(omega_f)
        return torch.maximum(tau_c, float(self.envelope_gamma) * omega_c) * vweight

    def _gather_by_token_indices(self, x: torch.Tensor, token_indices: torch.Tensor) -> torch.Tensor:
        gather_idx = token_indices.unsqueeze(-1).expand(-1, -1, -1, x.shape[-1])
        return x.gather(dim=2, index=gather_idx).contiguous()

    def _select_indices_from_scores(self, scores: torch.Tensor, n_keep: int) -> torch.Tensor:
        B, H, N = scores.shape
        if n_keep <= 0:
            return torch.zeros(B, H, 0, device=scores.device, dtype=torch.long)
        if n_keep >= N:
            return torch.arange(N, device=scores.device, dtype=torch.long).view(1, 1, N).expand(B, H, N)

        flat = scores.float().clamp_min(0.0).reshape(B * H, N)
        row_sums = flat.sum(dim=-1, keepdim=True)
        zero_rows = row_sums.squeeze(-1) <= 0
        if zero_rows.any():
            flat = flat.clone()
            flat[zero_rows] = 1.0

        selected = torch.topk(flat, k=n_keep, dim=-1).indices
        selected = selected.view(B, H, n_keep)
        return selected.sort(dim=-1).values

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del attentions

        if self.compression_ratio is None:
            raise ValueError("compression_ratio must be set before RidgeSketch.compress is called")
        if self.compression_ratio == 0:
            return keys, values

        _, _, T, _ = keys.shape
        if T < self.min_tokens_to_compress:
            return keys, values

        sink = min(self.sink_size, T)
        local = min(self.local_size, max(0, T - sink))
        mid_start = sink
        mid_end = T - local
        mid_len = max(0, mid_end - mid_start)
        if mid_len == 0:
            return keys, values

        keep_total = int(T * (1.0 - self.compression_ratio))
        keep_total = max(0, min(keep_total, T))
        keep_mid = min(max(keep_total - sink - local, 0), mid_len)

        if keep_mid <= 0:
            return (
                torch.cat([keys[:, :, :sink, :], keys[:, :, mid_end:, :]], dim=2).contiguous(),
                torch.cat([values[:, :, :sink, :], values[:, :, mid_end:, :]], dim=2).contiguous(),
            )
        if keep_mid == mid_len:
            return keys, values

        keys_mid = keys[:, :, mid_start:mid_end, :]
        values_mid = values[:, :, mid_start:mid_end, :]
        tau = self._compute_key_ridge_tau(keys_mid)

        omega = None
        if self.query_aware:
            queries = self._get_all_queries(
                module=module, hidden_states=hidden_states, keys=keys, kwargs=kwargs
            )
            if queries is not None:
                queries_for_metric = queries[:, :, mid_start:mid_end, :]
                omega = self._compute_query_key_interaction(keys_mid, queries_for_metric)
            else:
                logger.warning("Query-aware reweighting skipped because queries are unavailable.")

        scores = self._scores_from_tau_omega_and_values(
            tau=tau,
            values=values_mid,
            omega=omega,
        )

        keep_idx_mid = self._select_indices_from_scores(scores, keep_mid)
        kept_mid_keys = self._gather_by_token_indices(keys_mid, keep_idx_mid)
        kept_mid_values = self._gather_by_token_indices(values_mid, keep_idx_mid)

        out_keys = torch.cat([keys[:, :, :sink, :], kept_mid_keys, keys[:, :, mid_end:, :]], dim=2)
        out_values = torch.cat([values[:, :, :sink, :], kept_mid_values, values[:, :, mid_end:, :]], dim=2)
        return out_keys.contiguous(), out_values.contiguous()


@register_kv_compressor("random_sketch_press")
@dataclass
class RandomSketchRidgeSketch(RidgeSketch):
    """
    Prefill-time KV compression baseline mirroring RidgeSketch, intended to
    sample with uniform random scores instead of ridge leverage scores.

    Port of ``RandomSketchPress`` (kvpress/presses/random_sketch_press.py; a
    research-fork addition in the local kvpress 0.5.1 checkout, not upstream
    NVIDIA kvpress). Unrelated to upstream ``RandomPress``, which is ported
    separately as ``RandomSketch`` (registry name "random").

    Upstream bug, replicated faithfully: the single override ``_compute_tau``
    is DEAD CODE. ``RidgePress.compress`` calls ``_compute_key_ridge_tau`` and
    nothing in the kvpress checkout ever calls ``_compute_tau``, so as written
    the press is bitwise-identical to ``RidgePress`` under the same
    configuration and no randomness ever executes. This port preserves the
    dead override and the resulting RidgeSketch-equivalent behavior (pinned by
    tests) rather than wiring the documented intent into the live scoring
    path.
    """

    def _compute_tau(self, keys: torch.Tensor) -> torch.Tensor:
        """
        Return uniform random scores in [0, 1) with shape [B, H, N].

        keys shape: [B, H, N, D]
        tau shape:    [B, H, N]
        """
        B, H, N, _ = keys.shape
        if N == 0:
            return torch.zeros(B, H, 0, device=keys.device, dtype=keys.dtype)
        return torch.rand(B, H, N, device=keys.device, dtype=keys.dtype)
