import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.llama.modeling_llama import repeat_kv, rotate_half

from eval_harness.kv_compression.registry import register_kv_compressor
from eval_harness.kv_compression.base import ScorerKVCompressor


def _get_prerope_query_states(module: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    """Extract pre-RoPE query states, shape (B, H_q, S, head_dim).

    Port of ``kvpress.utils.get_prerope_query_states`` with the isinstance
    checks (Phi3Attention / Qwen3Attention / Gemma3Attention) replaced by
    duck-typing on ``qkv_proj`` / ``q_proj`` / ``q_norm``.
    """
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
        raise NotImplementedError(f"Sketch not yet implemented for {module.__class__}.")

    query_states = query_states.reshape(bsz, q_len, num_heads, head_dim).transpose(1, 2)

    q_norm = getattr(module, "q_norm", None)
    if q_norm is not None:
        query_states = q_norm(query_states)

    return query_states


@register_kv_compressor("non_causal_attention")
@dataclass
class NonCausalAttnSketch(ScorerKVCompressor):
    """Non-causal, chunked attention scorer.

    This sketch implements the non-causal, chunked attention, sum-over-queries
    scoring used in Compactor. Scores are z-normalized.

    Port of kvpress 0.5.1 ``NonCausalAttnPress``
    (``kvpress/presses/non_causal_attention_press.py``).

    References:
    - Chari & Van Durme (2025): "Compactor: Calibrated Query-Agnostic KV Cache
      Compression with Approximate Leverage Scores"
      (https://arxiv.org/pdf/2507.08143v1)

    Parameters
    ----------
    compression_ratio : float, default ``0.0``
        Fraction of KV pairs pruned (inherited from ``ScorerKVCompressor``).
    chunk_size : int, default ``256``
        Chunk size used in non-causal attention.

    Notes
    -----
    Only supports prefill. Cached keys are RoPE-rotated, so the pre-RoPE
    queries are re-rotated with ``kwargs["position_embeddings"]`` before the
    q.k matmul — no un-rotation of the cache is needed. Do not combine with
    the DCA prefill method (DCA stores keys rotated at cyclic positions,
    breaking the q.k geometry).

    Upstream quirks replicated faithfully (do not "fix"):
    - the chunked dots carry NO ``1/sqrt(d)`` scaling;
    - the padded-key mask constant in the last chunk is ``-1e-9`` (effectively
      unmasked: padded keys soak up softmax mass before being trimmed), and
      zero-filled padded-query rows softmax to a uniform ``1/chunk_size``;
    - ``F.avg_pool1d`` smoothing uses ``count_include_pad=True``, depressing
      scores at positions ``0`` and ``S-1`` (averaged with an implicit zero);
    - the z-normalization is GLOBAL over the whole (B, H_kv, S) tensor despite
      upstream's "head-wise" comment (affine, so top-k selection is unchanged;
      kept for forward-compatibility with a CompactorPress-style blend).

    Deviations from kvpress
    -----------------------
    - ``get_prerope_query_states`` is inlined here (it was not ported to
      ``eval_harness/sketch/utils.py``) with duck-typed ``qkv_proj`` /
      ``q_norm`` checks instead of isinstance checks against Phi3/Qwen3/Gemma3
      attention classes — the established Prism pattern, which also works with
      fake test modules and new qk-norm families.
    """

    chunk_size: int = 256

    @staticmethod
    def non_causal_chunked_attn(q: torch.Tensor, k: torch.Tensor, chunk_size: int) -> torch.Tensor:
        """Compute non-causal, chunked attention column-sums over the sequence.

        The sequence is padded to a multiple of ``chunk_size`` and processed in
        fixed-size tiles.

        Parameters
        ----------
        q, k : torch.Tensor, shape (B, H, S, d)
            Query/Key tensors for a single layer/head group.
        chunk_size : int
            Size of the chunk used to tile the sequence axis.

        Returns
        -------
        torch.Tensor, shape (B, H, S)
            Column-wise non-causal attention accumulations per key position.
        """
        assert chunk_size > 0, "chunk_size must be positive"
        assert q.shape[-2] == k.shape[-2], "only used in prefill"
        B, H, S, d = k.shape
        # pad to a multiple of chunk_size for easy reshaping
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
        # (B, H, num_chunks, chunk_size, d)
        q_chunks = q_padded.view(B, H, num_chunks, chunk_size, d)
        k_chunks = k_padded.view(B, H, num_chunks, chunk_size, d)

        # (B, H, num_chunks, chunk_size, chunk_size)
        dots = torch.matmul(q_chunks, k_chunks.transpose(-2, -1))
        dots[:, :, -1].masked_fill_(query_mask.unsqueeze(-1), 0)
        dots[:, :, -1].masked_fill_(key_mask.unsqueeze(-2), -1e-9)
        attn = torch.softmax(dots.to(torch.float32), dim=-1)
        # sum over query and trim padding
        return attn.sum(dim=-2).view(B, H, S_pad)[..., :S]

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
        assert keys.shape[-2] == n_queries, "NonCausalAttnSketch only supports prefill"

        cos, sin = kwargs["position_embeddings"]
        q = _get_prerope_query_states(module, hidden_states)  # (B, H_q, S, d)

        q_len = q.shape[-2]
        num_kv_groups = q.shape[1] // values.shape[1]
        # apply RoPE to the queries for the last q_len positions
        # Partial rotary (Qwen3.5: rotary_dim < head_dim) — rotate only the first
        # rotary_dim channels and pass the rest through; reduces to full RoPE when
        # rotary_dim == head_dim.
        cos_q, sin_q = cos[:, -q_len:, :].unsqueeze(1), sin[:, -q_len:, :].unsqueeze(1)
        rotary_dim = cos_q.shape[-1]
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        q_rot = (q_rot * cos_q) + (rotate_half(q_rot) * sin_q)
        q = torch.cat([q_rot, q_pass], dim=-1)

        A = self.non_causal_chunked_attn(q, repeat_kv(keys, num_kv_groups), self.chunk_size)  # (B, H_q, S)
        # average across query-head groups back to H_kv
        A = A.view(A.shape[0], values.shape[1], -1, A.shape[-1]).mean(dim=-2)  # (B, H_kv, S)

        scores = A * values.norm(dim=-1)  # (B, H_kv, S)
        scores = F.avg_pool1d(scores, kernel_size=3, padding=1, stride=1)
        z_scores = (scores - scores.mean()) / scores.std().clamp_min(1e-6)  # head-wise z-norm
        return z_scores
