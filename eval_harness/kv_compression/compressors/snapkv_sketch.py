import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.llama.modeling_llama import repeat_kv, rotate_half

from eval_harness.kv_compression.registry import register_kv_compressor
from eval_harness.kv_compression.base import ScorerKVCompressor


def _get_prerope_query_states(module: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    """Extract pre-RoPE query states ``[B, H_q, S, D]``.

    Duck-typed port of kvpress ``utils.get_prerope_query_states``: fused
    ``qkv_proj`` slice (Phi3-style) when present, ``q_proj`` otherwise, with an
    optional ``q_norm`` applied after the head reshape (Qwen3/Gemma3 qk-norm
    families).
    """
    bsz, q_len, _ = hidden_states.shape
    num_heads = module.config.num_attention_heads
    head_dim = module.head_dim

    qkv_proj = getattr(module, "qkv_proj", None)
    if qkv_proj is not None:
        query_states = qkv_proj(hidden_states)[..., : num_heads * head_dim]
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


@register_kv_compressor("snapkv")
@dataclass
class SnapKVSketch(ScorerKVCompressor):
    """
    SnapKV: Attention-based KV cache compression using recent token patterns.

    Uses attention patterns of the most recent ``window_size`` tokens to
    estimate the importance of previous key-value pairs: the last-window
    queries are re-projected (pre-RoPE), rotated to their true absolute
    positions with the layer's own ``kwargs["position_embeddings"]``, and
    matmul-ed against the RoPE-rotated cached keys, reproducing the model's
    real window-attention logits with no un-rotation anywhere. Per-position
    scores are the window-mean attention, smoothed with ``avg_pool1d``,
    group-averaged to KV heads, and the observation window is re-appended with
    the global max score so it is never pruned.

    Based on SnapKV (https://arxiv.org/abs/2404.14469).
    Port of kvpress ``SnapKVPress`` (kvpress/presses/snapkv_press.py).

    Parameters
    ----------
    compression_ratio : float, default=0.0
        Fraction of key-value pairs to remove during compression.
    window_size : int, default=64
        Number of recent tokens to use for computing attention-based
        importance scores; these positions are force-kept via max-score
        padding.
    kernel_size : int, default=5
        Size of the pooling kernel applied to attention weights for smoothing.
        Must be odd (see deviations).

    Deviations from kvpress
    -----------------------
    - ``get_prerope_query_states`` is inlined with duck-typing (``qkv_proj``
      presence for Phi3-style fused projections, ``getattr(module, "q_norm",
      None)`` for qk-norm families) instead of isinstance checks against
      transformers attention classes.
    - Qwen3.5 hybrid attention is handled in the query reprojection: when
      ``q_proj`` emits ``num_heads * head_dim * 2`` features the per-head output
      gate is sliced off (recovering the query the model actually uses), and
      ``compute_window_attention`` rotates only the first ``rotary_dim`` channels
      (partial rotary), passing the rest through. Both reduce to the prior
      full-head behavior on non-hybrid models -- the gate slice is guarded on the
      doubled width, and ``rotary_dim == head_dim`` leaves no passthrough -- so
      Llama/Qwen3/Gemma3 are unaffected. Inherited unchanged by PyramidKVSketch.
    - ``kernel_size`` is asserted odd in ``__post_init__``: an even kernel
      makes the pooled length ``k_len - window_size + 1`` and kvpress crashes
      later in the GQA ``view`` with an opaque shape error.
    - The ``attentions is not None`` branch is kept for parity but is dead
      code in this pipeline (sdpa never returns attention probabilities).

    Quirks kept for kvpress parity: ``avg_pool1d`` uses
    ``count_include_pad=True`` so edge positions are diluted by zero padding;
    the window pad value is ``scores.max().item()`` (a global max across
    batch and heads); kept KV pairs stay in score-descending topk order, not
    positional order; when ``n_kept < window_size`` the max-tied window
    positions are partially dropped with implementation-defined tie-breaking;
    prompts with ``q_len <= window_size`` raise at any nonzero ratio. Do not
    combine with the DCA prefill method: its keys are rotated at cyclic
    positions ``pos % chunk_len``, breaking the absolute-position RoPE
    assumption of the recomputed window queries.
    """

    compression_ratio: float = 0.0
    window_size: int = 64
    kernel_size: int = 5

    def __post_init__(self):
        super().__post_init__()
        assert self.kernel_size % 2 == 1, (
            f"kernel_size must be odd (got {self.kernel_size}): an even kernel changes the "
            f"avg_pool1d output length and breaks the per-group score reshape"
        )

    @staticmethod
    def compute_window_attention(module, hidden_states, keys, window_size, position_embeddings):
        """
        Compute the last window_size queries and associated attention weights for the first q_len - window_size keys.
        """

        bsz, _, k_len, _ = keys.shape
        num_heads = module.config.num_attention_heads
        head_dim = module.head_dim
        num_key_value_groups = num_heads // module.config.num_key_value_heads

        query_states = _get_prerope_query_states(module, hidden_states[:, -window_size:])

        cos, sin = position_embeddings
        cos, sin = cos[:, -window_size:], sin[:, -window_size:]
        # Partial rotary (Qwen3.5: rotary_dim = head_dim * partial_rotary_factor
        # < head_dim) — rotate only the first rotary_dim channels and leave the
        # passthrough channels unchanged, mirroring the model's
        # apply_rotary_pos_emb. Reduces to full RoPE when rotary_dim == head_dim.
        rotary_dim = cos.shape[-1]
        cos_u, sin_u = cos.unsqueeze(1), sin.unsqueeze(1)
        q_rot, q_pass = query_states[..., :rotary_dim], query_states[..., rotary_dim:]
        q_rot = (q_rot * cos_u) + (rotate_half(q_rot) * sin_u)
        query_states = torch.cat([q_rot, q_pass], dim=-1)

        key_states = repeat_kv(keys, num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
        attention_mask = torch.ones_like(attn_weights) * float("-inf")
        attention_mask = torch.triu(attention_mask, diagonal=k_len - window_size + 1)
        attn_weights += attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = attn_weights[..., :-window_size]

        return attn_weights

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:

        bsz, num_key_value_heads, k_len, _ = keys.shape
        num_key_value_groups = module.config.num_attention_heads // num_key_value_heads

        assert (
            hidden_states.shape[1] > self.window_size
        ), f"Query length {hidden_states.shape[1]} should be greater than the window size {self.window_size}"

        if attentions is not None:
            attn_weights = attentions[..., -self.window_size :, : -self.window_size]
        else:
            attn_weights = self.compute_window_attention(
                module, hidden_states, keys, self.window_size, kwargs["position_embeddings"]
            )

        scores = attn_weights.mean(dim=-2)
        scores = F.avg_pool1d(scores, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1)

        scores = scores.view(bsz, num_key_value_heads, num_key_value_groups, k_len - self.window_size)
        scores = scores.mean(2)

        scores = F.pad(scores, (0, self.window_size), value=scores.max().item())

        return scores
