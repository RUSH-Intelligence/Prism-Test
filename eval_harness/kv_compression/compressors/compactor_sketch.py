import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.llama.modeling_llama import repeat_kv, rotate_half

from eval_harness.kv_compression.registry import register_kv_compressor
from eval_harness.kv_compression.base import ScorerKVCompressor


def _get_prerope_query_states(module: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    """Re-project pre-RoPE queries from hidden_states (kvpress ``utils.get_prerope_query_states``)."""
    bsz, q_len, _ = hidden_states.shape
    num_heads = module.config.num_attention_heads
    head_dim = module.head_dim

    if hasattr(module, "qkv_proj"):
        qkv = module.qkv_proj(hidden_states)
        query_states = qkv[..., : num_heads * head_dim]
    elif hasattr(module, "q_proj"):
        query_states = module.q_proj(hidden_states)
        # Qwen3.5 gated attention fuses [query | gate] per head into q_proj
        # (output dim = num_heads * head_dim * 2). Slice off the gate to recover
        # the pre-RoPE query, matching Qwen3_5Attention.forward's
        # torch.chunk(q_proj(x).view(*, -1, head_dim * 2), 2, dim=-1).
        if query_states.shape[-1] == num_heads * head_dim * 2:
            query_states = query_states.view(bsz, q_len, num_heads, head_dim * 2)[..., :head_dim]
    else:
        raise NotImplementedError(f"CompactorSketch not yet implemented for {module.__class__}.")

    query_states = query_states.reshape(bsz, q_len, num_heads, head_dim).transpose(1, 2)

    q_norm = getattr(module, "q_norm", None)
    if q_norm is not None:
        query_states = q_norm(query_states)
    return query_states


def _get_prerope_key_states(module: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    """Re-project pre-RoPE keys from hidden_states (kvpress ``utils.get_prerope_key_states``)."""
    bsz, k_len, _ = hidden_states.shape
    head_dim = module.head_dim

    if hasattr(module, "qkv_proj"):
        qkv = module.qkv_proj(hidden_states)
        query_pos = module.config.num_attention_heads * head_dim
        key_states = qkv[..., query_pos : query_pos + module.num_key_value_heads * head_dim]
    elif hasattr(module, "k_proj"):
        key_states = module.k_proj(hidden_states)
    else:
        raise NotImplementedError(f"CompactorSketch not yet implemented for {module.__class__}.")

    key_states = key_states.view(bsz, k_len, -1, head_dim).transpose(1, 2)

    k_norm = getattr(module, "k_norm", None)
    if k_norm is not None:
        key_states = k_norm(key_states)
    return key_states


@register_kv_compressor("compactor")
@dataclass
class CompactorSketch(ScorerKVCompressor):
    """Compactor: Calibrated Query-Agnostic KV Cache Compression with Approximate Leverage Scores.

    Port of kvpress ``CompactorPress`` (with its ``LeverageScorePress`` and
    ``NonCausalAttnPress`` components inlined). Compactor blends, over the
    sink-protected interior of the sequence: (1) approximate statistical
    leverage scores of pre-RoPE keys via a right Gaussian sketch and a
    Cholesky solve, and (2) non-causal chunked attention column sums of
    RoPE-rotated re-projected queries against the cached (rotated) keys,
    weighted by value norms and smoothed with a 3-tap average pool. Each
    component is globally z-normalized; the blend is
    ``blending * leverage + attention`` (``blending=None`` resolves to
    ``compression_ratio``), and the protected sink positions are padded back
    in at the global max score. Prefill-only.

    Reference: Chari & Van Durme (2025), "Compactor: Calibrated
    Query-Agnostic KV Cache Compression with Approximate Leverage Scores"
    (https://arxiv.org/pdf/2507.08143v1); kvpress 0.5.1
    ``presses/compactor_press.py``.

    Upstream quirks replicated faithfully: no 1/sqrt(d) scaling on the
    chunked attention dots; padded-key columns of the last chunk are masked
    with -1e-9 (effectively zero, NOT -inf, so padded keys keep softmax mass
    and zeroed padded-query rows add ~uniform mass to every last-chunk
    column); both z-normalizations are global over (B, H, S) despite
    upstream's "head-wise" comment; ``avg_pool1d`` counts the zero padding
    (diluting the boundary scores); the Cholesky receives jitter 1e-2 on the
    first attempt with x10 escalation up to 5 tries; sinks are padded at the
    global max so they tie with the interior argmax under topk.

    Deviations from kvpress
    -----------------------
    - The two child presses are inlined as methods with identical math; the
      ``__setattr__`` parameter-sync machinery is dropped.
    - ``get_prerope_query_states``/``get_prerope_key_states`` are inlined
      with duck-typing (``qkv_proj``/``q_proj``/``k_proj``, optional
      ``q_norm``/``k_norm``) instead of isinstance checks.
    - Qwen3.5 hybrid attention is handled in the query reprojection: when
      ``q_proj`` emits ``num_heads * head_dim * 2`` features the per-head output
      gate is sliced off, and ``_non_causal_scores`` rotates only the first
      ``rotary_dim`` channels (partial rotary), passing the rest through. Both
      reduce to the prior full-head behavior when ``rotary_dim == head_dim`` and
      the gate is absent, so non-hybrid models (Llama/Qwen3/Gemma3) are
      unaffected.
    - ``phi``: optional injected sketch matrix used verbatim (no 1/sqrt(k)
      scaling) for deterministic tests; upstream draws a fresh
      ``torch.randn`` every call, so scores are nondeterministic run-to-run.
    - Empty-interior guard: when sink protection covers the whole sequence,
      uniform zero scores are returned (topk then keeps an arbitrary
      subset); upstream crashes on this input.
    - ``(cos, sin)`` are rebuilt from ``module.rotary_emb`` when
      ``kwargs['position_embeddings']`` is unavailable.

    Do not combine with ``attention_method: dca``: DCA caches keys rotated at
    cyclic positions, which breaks the non-causal q.k logits.

    Parameters
    ----------
    compression_ratio : float, default 0.0
        Fraction of key-value pairs to remove (sinks count against budget).
    sink_size_start : int, default 8
        Number of initial sink tokens to always protect.
    sink_size_end : int, default 4
        Number of most-recent tokens to always protect.
    chunk_size : int, default 256
        Chunk size used in non-causal attention.
    sketch_dimension : int, default 48
        Size of the right Gaussian sketch.
    blending : Optional[float], default None
        Weight on leverage z-scores; ``None`` resolves to ``compression_ratio``.
    phi : Optional[torch.Tensor], default None
        Injected sketch matrix broadcastable to (B, H_kv, head_dim, k); test hook.
    """

    sink_size_start: int = 8
    sink_size_end: int = 4
    chunk_size: int = 256
    sketch_dimension: int = 48
    blending: Optional[float] = None
    phi: Optional[torch.Tensor] = None

    @staticmethod
    def _chol_with_jitter(G: torch.Tensor, jitter: float = 0.0, max_tries: int = 5) -> torch.Tensor:
        """Cholesky factorization with adaptive jitter (kvpress ``LeverageScorePress.chol_with_jitter``)."""
        identity = torch.eye(G.shape[-1], device=G.device, dtype=G.dtype)
        cur = float(jitter)
        for _ in range(max_tries):
            L, info = torch.linalg.cholesky_ex(G + cur * identity, upper=False)
            if bool((info == 0).all()):
                return L
            cur = max(1e-8, (1e-2 if cur == 0.0 else 10.0 * cur))
        raise RuntimeError(f"Cholesky failed after {max_tries} tries.")

    def _compute_leverage_scores(self, key_states: torch.Tensor) -> torch.Tensor:
        """Approximate leverage scores on pre-RoPE keys via right Gaussian sketching."""
        d, k = key_states.shape[-1], self.sketch_dimension
        if self.phi is not None:
            Phi = self.phi.to(device=key_states.device, dtype=key_states.dtype)
        else:
            Phi = torch.randn(
                key_states.shape[0],
                key_states.shape[1],
                d,
                k,
                device=key_states.device,
                dtype=key_states.dtype,
            ) * (1 / math.sqrt(k))

        X = key_states - key_states.mean(dim=-2, keepdim=True)
        X = torch.matmul(X, Phi).to(torch.float32)  # (B, H, S, k)
        XT = X.transpose(-2, -1)  # (B, H, k, S)
        G = XT @ X  # (B, H, k, k)
        L = self._chol_with_jitter(0.5 * (G + G.transpose(-2, -1)), jitter=1e-2, max_tries=5)
        inv_Xt = torch.cholesky_solve(XT, L, upper=False)  # (X^T X)^{-1} X^T
        scores = (X * inv_Xt.transpose(-2, -1)).sum(dim=-1).clamp_min(0)  # (B, H, S)
        return scores

    def _leverage_scores(self, module: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        """``LeverageScorePress.score``: leverage on re-projected pre-RoPE keys, global z-norm."""
        pre_rope_keys = _get_prerope_key_states(module, hidden_states)  # (B, H_kv, S, d)
        scores = self._compute_leverage_scores(pre_rope_keys)  # (B, H_kv, S)
        z_scores = (scores - scores.mean()) / scores.std().clamp_min(1e-6)
        return z_scores

    @staticmethod
    def _non_causal_chunked_attn(q: torch.Tensor, k: torch.Tensor, chunk_size: int) -> torch.Tensor:
        """Non-causal, chunked attention column-sums (kvpress ``NonCausalAttnPress.non_causal_chunked_attn``)."""
        assert chunk_size > 0, "chunk_size must be positive"
        assert q.shape[-2] == k.shape[-2], "only used in prefill"
        B, H, S, d = k.shape
        S_pad = math.ceil(S / chunk_size) * chunk_size
        pad_len = S_pad - S

        if pad_len > 0:
            q_padded = torch.cat([q, torch.zeros(B, H, pad_len, d, device=q.device, dtype=q.dtype)], dim=2)
            k_padded = torch.cat([k, torch.zeros(B, H, pad_len, d, device=k.device, dtype=k.dtype)], dim=2)
            last_chunk_start = (S // chunk_size) * chunk_size
            in_valid = torch.arange(last_chunk_start, S_pad, device=q.device) >= S
            query_mask = key_mask = in_valid.view(1, 1, chunk_size).expand(B, H, chunk_size)
        else:
            q_padded, k_padded = q, k
            last_chunk_start = ((S - 1) // chunk_size) * chunk_size
            in_valid = torch.arange(last_chunk_start, S_pad, device=q.device) >= S
            query_mask = key_mask = in_valid.view(1, 1, chunk_size).expand(B, H, chunk_size)

        num_chunks = S_pad // chunk_size
        q_chunks = q_padded.view(B, H, num_chunks, chunk_size, d)
        k_chunks = k_padded.view(B, H, num_chunks, chunk_size, d)

        dots = torch.matmul(q_chunks, k_chunks.transpose(-2, -1))
        dots[:, :, -1].masked_fill_(query_mask.unsqueeze(-1), 0)
        dots[:, :, -1].masked_fill_(key_mask.unsqueeze(-2), -1e-9)
        attn = torch.softmax(dots.to(torch.float32), dim=-1)
        return attn.sum(dim=-2).view(B, H, S_pad)[..., :S]

    def _non_causal_scores(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """``NonCausalAttnPress.score``: chunked non-causal attention x value norms, pooled, global z-norm."""
        q = _get_prerope_query_states(module, hidden_states)  # (B, H_q, S, d)

        q_len = q.shape[-2]
        num_kv_groups = q.shape[1] // values.shape[1]
        # Partial rotary (Qwen3.5: rotary_dim < head_dim) — rotate only the first
        # rotary_dim channels of q so it matches the (partially) RoPE-rotated
        # cached keys, mirroring the model's apply_rotary_pos_emb. Reduces to full
        # RoPE when rotary_dim == head_dim.
        cos_q, sin_q = cos[:, -q_len:, :].unsqueeze(1), sin[:, -q_len:, :].unsqueeze(1)
        rotary_dim = cos_q.shape[-1]
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        q_rot = (q_rot * cos_q) + (rotate_half(q_rot) * sin_q)
        q = torch.cat([q_rot, q_pass], dim=-1)

        A = self._non_causal_chunked_attn(q, repeat_kv(keys, num_kv_groups), self.chunk_size)  # (B, H_q, S)
        A = A.view(A.shape[0], values.shape[1], -1, A.shape[-1]).mean(dim=-2)  # (B, H_kv, S)

        scores = A * values.norm(dim=-1)  # (B, H_kv, S)
        scores = F.avg_pool1d(scores, kernel_size=3, padding=1, stride=1)
        z_scores = (scores - scores.mean()) / scores.std().clamp_min(1e-6)
        return z_scores

    def _position_embeddings(
        self, module: nn.Module, hidden_states: torch.Tensor, kwargs: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pos_emb = kwargs.get("position_embeddings")
        if isinstance(pos_emb, (tuple, list)) and len(pos_emb) == 2 and pos_emb[0] is not None:
            return pos_emb[0], pos_emb[1]

        rotary = getattr(module, "rotary_emb", None)
        if rotary is None:
            raise ValueError(
                "CompactorSketch requires kwargs['position_embeddings'] or module.rotary_emb"
            )
        cache_position = kwargs.get("cache_position")
        if cache_position is not None:
            position_ids = cache_position.unsqueeze(0)
        else:
            position_ids = torch.arange(hidden_states.shape[-2], device=hidden_states.device).unsqueeze(0)
        cos, sin = rotary(hidden_states, position_ids)
        return cos, sin

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        n_queries = hidden_states.shape[-2]
        assert keys.shape[-2] == n_queries, "CompactorSketch only supports prefill at the moment"
        left_keep = min(self.sink_size_start, n_queries)
        right_keep = min(self.sink_size_end, max(0, n_queries - left_keep))
        start_idx, end_idx = left_keep, (None if right_keep == 0 else -right_keep)

        if n_queries - left_keep - right_keep == 0:
            # Deviation: upstream crashes on an empty interior; return uniform scores instead.
            return torch.zeros(keys.shape[0], keys.shape[1], n_queries, dtype=torch.float32, device=keys.device)

        hs = hidden_states[:, start_idx:end_idx]
        keys = keys[..., start_idx:end_idx, :]
        values = values[..., start_idx:end_idx, :]
        cos, sin = self._position_embeddings(module, hidden_states, kwargs)
        cos = cos[..., start_idx:end_idx, :]
        sin = sin[..., start_idx:end_idx, :]

        l_scores = self._leverage_scores(module, hs)
        attn_scores = self._non_causal_scores(module, hs, keys, values, cos, sin)
        assert attn_scores.shape == l_scores.shape, "CompactorSketch only supports prefill at the moment"
        blending = self.blending if self.blending is not None else self.compression_ratio
        blending = 0.35 if blending is None else blending
        scores = blending * l_scores + attn_scores
        # protect sinks by padding at the global max (sinks tie with the interior argmax)
        scores = F.pad(scores, (left_keep, right_keep), value=scores.detach().max())
        return scores
