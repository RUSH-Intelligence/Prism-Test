import logging
from dataclasses import dataclass, field
from typing import FrozenSet, Optional

import torch
from torch import nn

from eval_harness.kv_compression.base import CompressionSchedule
from eval_harness.kv_compression.compressors.ridge_sketch import RidgeSketch
from eval_harness.kv_compression.registry import register_kv_compressor

logger = logging.getLogger(__name__)


@register_kv_compressor("streaming_ridge")
@dataclass
class StreamingRidgeSketch(RidgeSketch):
    """RidgeSketch + decode-time streaming compression from running Grams.

    Prefill behavior is IDENTICAL to :class:`RidgeSketch` (the post-prefill
    event delegates to ``super().compress`` bit-for-bit). In addition, the
    compressor fires during decode every ``decode_interval`` generated tokens,
    re-scoring the cached middle region and pruning it back to the budget
    ``int(T_seen * (1 - ratio))`` where ``T_seen`` counts every token EVER seen
    (prefill + question + generated). Budgeting against ``T_seen`` — not the
    current cache length — avoids the geometric over-eviction that repeated
    ratio-of-current-length pruning would cause.

    Streaming statistics (the whole point — both are d x d and exactly
    additive over token chunks, so streaming scores equal batch scores):

    - **Key-direction Gram (append-only)**: ``G_K = sum_i k_hat_i k_hat_i^T``
      over ALL keys ever seen, including evicted ones. tau_i is the ridge
      leverage of cached key i against the FULL-history Gram: the spectral-
      coverage guarantee targets the full key matrix, tau is monotonically
      non-increasing over time (no score oscillation), and the O(D^2) Gram is
      the lossless statistic that survives eviction.
    - **Query second-moment Gram (optionally EMA)**: ``G_Q`` accumulates
      pooled un-rotated queries (same projection/pooling as the batch path;
      the question's queries fold in during the question forward — decode
      events after the question are question-aware). ``query_ema_beta < 1``
      exponentially decays old queries at each decode event (queries are the
      drifting test distribution); the default 1.0 is pure append-only.

    omega semantics match the batch path: ``omega_i^2 = k_i^T (G_Q / W) k_i``
    with ``W`` the (decayed) query count — the mean-normalized Gram.

    Constraints:
    - Single-pass prefill only (like ridge/compactor: chunked prefill drops
      the query-aware term at the post-prefill event; the streaming Grams are
      initialized from the final full-cache event).
    - Single question per prompt (decode pruning breaks the multi-question
      checkpoint/restore contract; the pipeline enforces this).
    - All hooked layers prune to the same length on the same step (budget is
      a pure function of ``T_seen``/ratio, and every layer sees every decode
      forward), preserving the shared-causal-mask invariant.
    """

    # None -> use compression_ratio for decode events too.
    decode_ratio: Optional[float] = None
    # EMA decay applied to the query Gram at each decode event (1.0 = off).
    query_ema_beta: float = 1.0

    schedule: FrozenSet[CompressionSchedule] = field(
        default_factory=lambda: frozenset(
            {CompressionSchedule.POST_PREFILL, CompressionSchedule.DECODE}
        ),
        kw_only=True,
    )
    # First consumer of the base-class field (reserved until now): number of
    # decode tokens between compression events.
    decode_interval: int = field(default=32, kw_only=True)

    # Marks this class safe to install during decode (interval throttling +
    # tokens-ever-seen budgeting); the pipeline refuses decode installs for
    # compressors without this flag.
    decode_capable = True

    def __post_init__(self):
        super().__post_init__()
        assert self.decode_interval >= 1, "decode_interval must be >= 1"
        assert 0.0 < self.query_ema_beta <= 1.0, "query_ema_beta must be in (0, 1]"
        if self.decode_ratio is not None:
            assert 0.0 <= self.decode_ratio < 1.0, "decode_ratio must be in [0, 1)"
        # The streaming state is only coherent when the full-cache prefill
        # event initializes it: decode-only would leak state across prompts
        # (the defensive init keys on first-touch, not per prompt), and
        # streaming/chunked prefill would reset the full-history Gram to the
        # post-eviction cache each chunk, voiding the invariant.
        if CompressionSchedule.DECODE in self.schedule:
            assert CompressionSchedule.POST_PREFILL in self.schedule, (
                "streaming_ridge with the 'decode' schedule requires "
                "'post_prefill' as well (the prefill event initializes the "
                "streaming statistics)"
            )
            assert CompressionSchedule.STREAMING not in self.schedule, (
                "streaming_ridge is incompatible with the 'streaming' "
                "(chunked-prefill) schedule: per-chunk events would reset the "
                "full-history Gram to the post-eviction cache"
            )
        self.reset()

    # ------------------------------------------------------------------
    # Per-layer streaming state (one instance is shared across all layers;
    # everything is keyed by module.layer_idx and fully re-initialized at
    # each post-prefill event, so state cannot leak across prompts).
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._gram_k: dict = {}        # layer -> [B, H_kv, D, D] fp32, append-only
        self._gram_q: dict = {}        # layer -> [B, H_kv, D, D] fp32, EMA-decayed
        self._q_weight: dict = {}      # layer -> float effective query count W
        self._pending_gq: dict = {}    # layer -> accumulated new-query Gram since last event
        self._pending_qn: dict = {}    # layer -> new-query count since last event
        self._tokens_seen: dict = {}   # layer -> total tokens ever seen
        self._since_event: dict = {}   # layer -> tokens since last decode event

    @staticmethod
    def _normalized_gram(keys: torch.Tensor, eps: float) -> torch.Tensor:
        """sum_i k_hat_i k_hat_i^T over [B, H, N, D] keys -> [B, H, D, D] fp32."""
        k = keys.float()
        k = k / k.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
        return k.transpose(-2, -1) @ k

    def _tau_from_gram(self, keys_mid: torch.Tensor, gram: torch.Tensor) -> torch.Tensor:
        """Ridge leverage of normalized keys against a (full-history) Gram."""
        B, H, N, D = keys_mid.shape
        if N == 0:
            return torch.zeros(B, H, 0, device=keys_mid.device, dtype=keys_mid.dtype)
        k = keys_mid.float()
        k = k / k.norm(p=2, dim=-1, keepdim=True).clamp_min(self.eps)
        eye = torch.eye(D, device=k.device, dtype=k.dtype).view(1, 1, D, D)
        reg = gram + self.ridge_lambda * eye
        try:
            inv_reg = torch.linalg.inv(reg)
        except torch.linalg.LinAlgError:
            inv_reg = torch.linalg.pinv(reg)
        tau = ((k @ inv_reg) * k).sum(dim=-1).clamp_min(0.0)
        return tau.to(keys_mid.dtype)

    def _omega_from_gram(
        self, keys_mid: torch.Tensor, gram_q: torch.Tensor, weight: float
    ) -> torch.Tensor:
        """omega_i = sqrt(k_i^T (G_Q / W) k_i) on raw keys (mean-normalized Gram)."""
        B, H, N, _ = keys_mid.shape
        if N == 0:
            return torch.zeros(B, H, 0, device=keys_mid.device, dtype=keys_mid.dtype)
        k = keys_mid.float()
        G = gram_q / max(weight, 1.0)
        omega = ((k @ G) * k).sum(dim=-1).clamp_min(0.0).sqrt()
        return omega.to(keys_mid.dtype)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_len = hidden_states.shape[1] if hidden_states is not None else keys.shape[2]
        if self._resolve_is_decode(module, kwargs, q_len):
            return self._decode_compress(module, hidden_states, keys, values, kwargs)
        return self._prefill_compress(module, hidden_states, keys, values, attentions, kwargs)

    # ------------------------------------------------------------------
    # Prefill: initialize streaming state, then behave exactly like ridge
    # ------------------------------------------------------------------

    def _prefill_compress(self, module, hidden_states, keys, values, attentions, kwargs):
        li = module.layer_idx
        _, H_kv, T, D = keys.shape

        # Full-history key Gram over ALL context keys — captured BEFORE the
        # post-prefill eviction below, so evicted directions stay represented.
        self._gram_k[li] = self._normalized_gram(keys, self.eps)

        gram_q = None
        weight = 0.0
        if self.query_aware and hidden_states is not None:
            queries = self._get_all_queries(module, hidden_states, keys, kwargs)
            if queries is not None:
                qf = queries.float()
                gram_q = qf.transpose(-2, -1) @ qf
                weight = float(T)
        self._gram_q[li] = gram_q
        self._q_weight[li] = weight
        self._pending_gq[li] = None
        self._pending_qn[li] = 0
        self._tokens_seen[li] = T
        self._since_event[li] = 0

        return super().compress(module, hidden_states, keys, values, attentions, kwargs)

    # ------------------------------------------------------------------
    # Decode: accumulate statistics every step, prune every decode_interval
    # ------------------------------------------------------------------

    def _decode_compress(self, module, hidden_states, keys, values, kwargs):
        li = module.layer_idx
        _, H_kv, T, D = keys.shape
        n_new = hidden_states.shape[1] if hidden_states is not None else 1

        if li not in self._gram_k:
            # Decode without a prior prefill event (defensive): initialize the
            # key Gram from the current cache and start counting from here.
            logger.warning(
                "streaming_ridge: decode step before any prefill event on "
                "layer %s; initializing statistics from the current cache.", li,
            )
            self._gram_k[li] = self._normalized_gram(keys, self.eps)
            self._gram_q[li] = None
            self._q_weight[li] = 0.0
            self._pending_gq[li] = None
            self._pending_qn[li] = 0
            self._tokens_seen[li] = T
            self._since_event[li] = 0
            return keys, values

        # 1. Append the new keys (last n_new cache slots) to the key Gram.
        self._gram_k[li] = self._gram_k[li] + self._normalized_gram(
            keys[:, :, T - n_new:, :], self.eps
        )

        # 2. Buffer the new queries' Gram (EMA decay applies at event time).
        # hasattr guard mirrors _get_all_queries: fused-QKV models (Phi3)
        # would otherwise log a projection warning per layer per token.
        if self.query_aware and hidden_states is not None and hasattr(module, "q_proj"):
            q_new = self._project_and_pool_queries(
                module, hidden_states, H_kv, D, keys.dtype, kwargs,
            )
            if q_new is not None:
                qf = q_new.float()
                pending = qf.transpose(-2, -1) @ qf
                if self._pending_gq[li] is None:
                    self._pending_gq[li] = pending
                else:
                    self._pending_gq[li] = self._pending_gq[li] + pending
                self._pending_qn[li] += n_new

        self._tokens_seen[li] += n_new
        self._since_event[li] += n_new

        if self._since_event[li] < self.decode_interval:
            return keys, values

        # ---- compression event ----
        self._since_event[li] = 0

        # Fold pending queries into the EMA query Gram.
        pending = self._pending_gq[li]
        if pending is not None:
            beta = float(self.query_ema_beta)
            if self._gram_q[li] is None:
                self._gram_q[li] = pending
                self._q_weight[li] = float(self._pending_qn[li])
            else:
                self._gram_q[li] = beta * self._gram_q[li] + pending
                self._q_weight[li] = beta * self._q_weight[li] + float(self._pending_qn[li])
            self._pending_gq[li] = None
            self._pending_qn[li] = 0

        ratio = self.decode_ratio if self.decode_ratio is not None else self.compression_ratio
        if ratio is None or ratio == 0:
            return keys, values

        # Budget against tokens EVER seen — NOT the current cache length —
        # so repeated events do not geometrically shrink the cache.
        keep_total = int(self._tokens_seen[li] * (1.0 - ratio))
        keep_total = max(0, min(keep_total, T))

        sink = min(self.sink_size, T)
        local = min(self.local_size, max(0, T - sink))
        mid_start = sink
        mid_end = T - local
        mid_len = max(0, mid_end - mid_start)
        if mid_len == 0:
            return keys, values

        keep_mid = min(max(keep_total - sink - local, 0), mid_len)
        if keep_mid >= mid_len:
            return keys, values  # already at/below budget
        if keep_mid <= 0:
            return (
                torch.cat([keys[:, :, :sink, :], keys[:, :, mid_end:, :]], dim=2).contiguous(),
                torch.cat([values[:, :, :sink, :], values[:, :, mid_end:, :]], dim=2).contiguous(),
            )

        keys_mid = keys[:, :, mid_start:mid_end, :]
        values_mid = values[:, :, mid_start:mid_end, :]

        tau = self._tau_from_gram(keys_mid, self._gram_k[li])
        omega = None
        if self._gram_q[li] is not None and self._q_weight[li] > 0:
            omega = self._omega_from_gram(keys_mid, self._gram_q[li], self._q_weight[li])

        scores = self._scores_from_tau_omega_and_values(
            tau=tau, values=values_mid, omega=omega,
        )
        keep_idx_mid = self._select_indices_from_scores(scores, keep_mid)
        kept_mid_keys = self._gather_by_token_indices(keys_mid, keep_idx_mid)
        kept_mid_values = self._gather_by_token_indices(values_mid, keep_idx_mid)

        out_keys = torch.cat(
            [keys[:, :, :sink, :], kept_mid_keys, keys[:, :, mid_end:, :]], dim=2,
        )
        out_values = torch.cat(
            [values[:, :, :sink, :], kept_mid_values, values[:, :, mid_end:, :]], dim=2,
        )
        return out_keys.contiguous(), out_values.contiguous()
