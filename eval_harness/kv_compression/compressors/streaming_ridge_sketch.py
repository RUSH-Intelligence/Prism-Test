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
    """RidgeSketch prefill + decode-time compression from full-history Grams.

    **Prefill** is IDENTICAL to :class:`RidgeSketch`: the post-prefill event
    delegates to ``super().compress`` bit-for-bit (tau from the Gram of the
    CURRENT middle-region keys, omega from mid-sliced queries). On the side,
    the event initializes the streaming statistics below from the FULL
    context (all keys/queries, before eviction).

    **Decode** fires every ``decode_interval`` generated tokens and prunes the
    cached middle region back to ``keep_total = int(T_seen * (1 - ratio))``,
    where ``T_seen`` counts every token EVER seen (prefill + question +
    generated). Budgeting against ``T_seen`` — never the current cache
    length — avoids geometric over-eviction under repeated events.

    Streaming statistics — NOTE these deliberately DIFFER from the batch
    RidgeSketch scores (which see only the current middle region):

    - **Key-direction Gram (append-only)**: ``G_K = sum_i k_hat_i k_hat_i^T``
      over ALL keys ever seen, including evicted ones. Decode-time
      ``tau_i = k_hat_i^T (G_K + lambda I)^{-1} k_hat_i`` is the ridge
      leverage of cached key i against the FULL HISTORY: the spectral-
      coverage rationale targets the full key matrix, and the O(D^2) Gram is
      the lossless statistic that survives eviction. Monotonicity caveat: as
      ``G_K`` grows, only the RAW leverage of a FIXED key is monotonically
      non-increasing (Loewner ordering of the inverses); sum-NORMALIZED
      scores and cross-token RANKINGS may still change between events.
    - **Query second-moment Gram (optionally EMA)**: ``G_Q`` accumulates
      pooled (GQA group-mean), by-default un-rotated queries — the same
      projection/pooling as the batch path. The question forward's queries
      fold in during decode, so post-question events are question-aware.
      ``query_ema_beta`` is applied ONCE PER COMPRESSION EVENT (not per
      token): ``G_Q <- beta * G_Q + Q_block^T Q_block`` with effective count
      ``W <- beta * W + s``; omega uses the mean-normalized form
      ``omega_i = sqrt(k_i^T (G_Q / W) k_i)`` on raw keys.

    Efficiency contract: between events the per-token cost is O(1) —
    new hidden states are buffered by reference and counters advance; NO
    Gram/rank-one updates run per token. At each event the pending block is
    folded with batched GEMMs (one q_proj over the whole block, one
    ``K_block^T K_block``, one ``Q_block^T Q_block``) and tau is computed via
    a Cholesky solve of the D x D regularized Gram (pinv fallback).

    Event timing keeps the interval REMAINDER: after firing,
    ``_since_event %= decode_interval``, so multi-token decode chunks (the
    question forward, speculative decoding) that overshoot the boundary keep
    the event phase aligned to the token stream instead of drifting.

    RoPE safety (only relevant when ``rotate_queries=True``): decode-time
    rotation uses the ``position_embeddings`` kwarg (absolute, supplied by
    the pipeline) or absolute ``position_ids``; ``cache_position`` is
    accepted only until the first decode-time prune (afterwards physical
    slots diverge from absolute positions). Positions are NEVER regenerated
    from zero during decode; if nothing valid is available the queries are
    accumulated un-rotated with a once-per-layer warning.

    Constraints:
    - Single-pass prefill only; the schedule must be
      ``{post_prefill, decode}`` (asserted — decode-only would leak state
      across prompts, streaming/chunked prefill would reset the full-history
      Gram per chunk).
    - Single question per prompt (decode pruning breaks the multi-question
      checkpoint/restore contract; the pipeline enforces this).
    - All hooked layers prune to the same length on the same step: the
      budget is a pure function of the shared ``T_seen``/ratio and every
      layer sees every decode forward — preserving the shared-causal-mask
      invariant.
    """

    # None -> use compression_ratio for decode events too.
    decode_ratio: Optional[float] = None
    # EMA decay applied to the query Gram once per compression event (1.0 = off).
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
        self._gram_k: dict = {}          # layer -> [B, H_kv, D, D] fp32, append-only
        self._gram_q: dict = {}          # layer -> [B, H_kv, D, D] fp32, per-event EMA
        self._q_weight: dict = {}        # layer -> float effective query count W
        self._tokens_seen: dict = {}     # layer -> total tokens ever seen
        self._since_event: dict = {}     # layer -> tokens since last event (mod interval)
        self._pending_hiddens: dict = {} # layer -> list[Tensor [B, n, d_model]]
        self._pending_cos_sin: dict = {} # layer -> list[Optional[(cos, sin)]]
        self._pending_tokens: dict = {}  # layer -> tokens buffered since last Gram fold
        self._pruned_in_decode: dict = {}  # layer -> bool (physical != absolute slots)
        self._rope_warned: set = set()   # layers already warned about un-rotated fallback

    def _init_layer_state(self, li: int, keys: torch.Tensor) -> None:
        self._gram_k[li] = self._normalized_gram(keys, self.eps)
        self._gram_q[li] = None
        self._q_weight[li] = 0.0
        self._tokens_seen[li] = keys.shape[2]
        self._since_event[li] = 0
        self._pending_hiddens[li] = []
        self._pending_cos_sin[li] = []
        self._pending_tokens[li] = 0
        self._pruned_in_decode[li] = False
        self._rope_warned.discard(li)

    # ------------------------------------------------------------------
    # Math helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalized_gram(keys: torch.Tensor, eps: float) -> torch.Tensor:
        """sum_i k_hat_i k_hat_i^T over [B, H, N, D] keys -> [B, H, D, D] fp32."""
        k = keys.float()
        k = k / k.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
        return k.transpose(-2, -1) @ k

    def _tau_from_gram(self, keys_mid: torch.Tensor, gram: torch.Tensor) -> torch.Tensor:
        """Ridge leverage of normalized keys against a (full-history) Gram.

        Uses a Cholesky solve of the PD matrix ``gram + lambda I`` (error
        proportional to the condition number rather than its square, and no
        explicit inverse is materialized); falls back to pinv if the
        factorization fails.
        """
        B, H, N, D = keys_mid.shape
        if N == 0:
            return torch.zeros(B, H, 0, device=keys_mid.device, dtype=keys_mid.dtype)
        k = keys_mid.float()
        k = k / k.norm(p=2, dim=-1, keepdim=True).clamp_min(self.eps)
        eye = torch.eye(D, device=k.device, dtype=k.dtype).view(1, 1, D, D)
        reg = gram + self.ridge_lambda * eye
        kt = k.transpose(-2, -1)                       # [B, H, D, N]
        try:
            chol = torch.linalg.cholesky(reg)
            sol = torch.cholesky_solve(kt, chol)       # (reg)^{-1} K_hat^T
            tau = (kt * sol).sum(dim=-2)               # [B, H, N]
        except (torch.linalg.LinAlgError, RuntimeError):
            inv_reg = torch.linalg.pinv(reg)
            tau = ((k @ inv_reg) * k).sum(dim=-1)
        return tau.clamp_min(0.0).to(keys_mid.dtype)

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

    def _decode_cos_sin(
        self, module: nn.Module, hidden_states: torch.Tensor, kwargs: dict, li: int
    ):
        """Absolute (cos, sin) for the new decode tokens, or None.

        Accepts, in order: the ``position_embeddings`` kwarg (absolute — the
        pipeline passes absolute position_ids to the model), absolute
        ``position_ids``, and ``cache_position`` ONLY while no decode-time
        prune has occurred on this layer (before that, physical slot ==
        absolute position; afterwards they diverge). Never fabricates
        positions from zero.
        """
        kwargs = kwargs or {}
        pos_emb = kwargs.get("position_embeddings")
        if isinstance(pos_emb, (tuple, list)) and len(pos_emb) == 2 and pos_emb[0] is not None:
            return pos_emb[0], pos_emb[1]
        rotary = getattr(module, "rotary_emb", None)
        if rotary is None:
            return None
        position_ids = kwargs.get("position_ids")
        if position_ids is not None:
            return rotary(hidden_states, position_ids)
        cache_position = kwargs.get("cache_position")
        if cache_position is not None and not self._pruned_in_decode.get(li, False):
            return rotary(hidden_states, cache_position.unsqueeze(0))
        return None

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
        self._init_layer_state(li, keys)

        if self.query_aware and hidden_states is not None:
            queries = self._get_all_queries(module, hidden_states, keys, kwargs)
            if queries is not None:
                qf = queries.float()
                self._gram_q[li] = qf.transpose(-2, -1) @ qf
                self._q_weight[li] = float(T)

        return super().compress(module, hidden_states, keys, values, attentions, kwargs)

    # ------------------------------------------------------------------
    # Decode: O(1) buffering per token; batched fold + prune at events
    # ------------------------------------------------------------------

    def _fold_pending(self, module, keys, li):
        """Fold the buffered block into the Grams with batched GEMMs."""
        _, H_kv, T, D = keys.shape
        pending_n = self._pending_tokens[li]
        if pending_n > 0:
            # Between events nothing prunes, so the last pending_n cache slots
            # are exactly the tokens buffered since the previous fold.
            self._gram_k[li] = self._gram_k[li] + self._normalized_gram(
                keys[:, :, T - pending_n:, :], self.eps
            )

        hiddens = self._pending_hiddens[li]
        if hiddens:
            h_block = torch.cat(hiddens, dim=1)
            cos_sin = None
            if self.rotate_queries:
                entries = self._pending_cos_sin[li]
                if entries and all(e is not None for e in entries):
                    cos_sin = (
                        torch.cat([e[0] for e in entries], dim=1),
                        torch.cat([e[1] for e in entries], dim=1),
                    )
                elif li not in self._rope_warned:
                    self._rope_warned.add(li)
                    logger.warning(
                        "streaming_ridge: rotate_queries=True but no absolute "
                        "position source is available at decode on layer %s; "
                        "accumulating un-rotated queries.", li,
                    )
            q_block = self._project_and_pool_queries(
                module, h_block, H_kv, D, keys.dtype,
                kwargs=None, cos_sin=cos_sin, resolve_positions=False,
            )
            if q_block is not None:
                qf = q_block.float()
                block_gram = qf.transpose(-2, -1) @ qf
                n_q = float(h_block.shape[1])
                beta = float(self.query_ema_beta)
                if self._gram_q[li] is None:
                    self._gram_q[li] = block_gram
                    self._q_weight[li] = n_q
                else:
                    self._gram_q[li] = beta * self._gram_q[li] + block_gram
                    self._q_weight[li] = beta * self._q_weight[li] + n_q

        self._pending_hiddens[li] = []
        self._pending_cos_sin[li] = []
        self._pending_tokens[li] = 0

    def _decode_compress(self, module, hidden_states, keys, values, kwargs):
        li = module.layer_idx
        _, H_kv, T, D = keys.shape
        n_new = hidden_states.shape[1] if hidden_states is not None else 1

        if li not in self._gram_k:
            # Decode without a prior prefill event (defensive): initialize the
            # statistics from the current cache and start counting from here.
            logger.warning(
                "streaming_ridge: decode step before any prefill event on "
                "layer %s; initializing statistics from the current cache.", li,
            )
            self._init_layer_state(li, keys)
            return keys, values

        # Per-token path: buffer references and advance counters — no GEMMs.
        if self.query_aware and hidden_states is not None and hasattr(module, "q_proj"):
            self._pending_hiddens[li].append(hidden_states.detach())
            if self.rotate_queries:
                self._pending_cos_sin[li].append(
                    self._decode_cos_sin(module, hidden_states, kwargs, li)
                )
        self._pending_tokens[li] += n_new
        self._tokens_seen[li] += n_new
        self._since_event[li] += n_new

        if self._since_event[li] < self.decode_interval:
            return keys, values

        # ---- compression event ----
        # Keep the interval remainder so multi-token chunks (question forward,
        # speculative decoding) don't drift the event phase.
        self._since_event[li] %= self.decode_interval

        self._fold_pending(module, keys, li)

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
            self._pruned_in_decode[li] = True
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

        self._pruned_in_decode[li] = True
        out_keys = torch.cat(
            [keys[:, :, :sink, :], kept_mid_keys, keys[:, :, mid_end:, :]], dim=2,
        )
        out_values = torch.cat(
            [values[:, :, :sink, :], kept_mid_values, values[:, :, mid_end:, :]], dim=2,
        )
        return out_keys.contiguous(), out_values.contiguous()
