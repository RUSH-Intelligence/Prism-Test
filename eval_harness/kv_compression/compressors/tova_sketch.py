import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.llama.modeling_llama import repeat_kv, rotate_half

from eval_harness.kv_compression.registry import register_kv_compressor
from eval_harness.kv_compression.base import ScorerKVCompressor


def _get_prerope_query_states(module: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    """Extract pre-RoPE query states ``[B, H_q, S, D]``.

    Duck-typed port of kvpress ``utils.get_prerope_query_states``: fused
    ``qkv_proj`` slice (Phi3-style), ``q_proj`` otherwise, with an optional
    ``q_norm`` applied after the head reshape (Qwen3/Gemma3 qk-norm families).
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


@register_kv_compressor("tova")
@dataclass
class TOVASketch(ScorerKVCompressor):
    """
    TOVA: Token-wise Optimal Value Attention for KV cache compression.

    Uses attention weights of the last token (averaged across all query heads)
    to estimate importance of previous key-value pairs. The last token's
    attention pattern provides a good indicator of which historical tokens are
    important. Every KV head receives the same head-averaged score vector, so
    all heads keep an identical position set and the cache stays rectangular.

    Based on TOVA (https://arxiv.org/abs/2401.06104).
    Official implementation: https://github.com/schwartz-lab-NLP/TOVA/blob/main/src/tova_cache.py
    Port of kvpress ``TOVAPress`` (kvpress/presses/tova_press.py); the
    recompute path is the window-size-1 specialization of
    ``SnapKVPress.compute_window_attention`` (kvpress/presses/snapkv_press.py).

    When attention probabilities are unavailable (this pipeline runs sdpa and
    never requests ``output_attentions``, so the hook's ``attentions`` is
    always ``None``), the last attention row is recomputed: the last token's
    pre-RoPE query is re-projected, rotated to its absolute position via
    ``kwargs["position_embeddings"]``, and matmul-ed against the RoPE-rotated
    cached keys — do not un-rotate anything. The last token is then re-appended
    with the global max score so it (almost) always survives compression
    (kvpress quirk; the original TOVA does not enforce this, replicated as-is).

    Parameters
    ----------
    compression_ratio : float, default=0.0
        Fraction of key-value pairs to remove during compression.

    Deviations from kvpress
    -----------------------
    - ``get_prerope_query_states`` is inlined with duck-typing (``qkv_proj``
      presence for Phi3-style fused projections, ``q_proj`` otherwise,
      ``getattr(module, "q_norm", None)`` for qk-norm families) instead of
      isinstance checks against transformers attention classes.
    - The additive ``triu`` mask of ``SnapKVPress.compute_window_attention``
      is dropped: at window size 1 its diagonal is ``k_len``, masking nothing
      on the single query row (numerically a no-op; pinned against the full
      kvpress transcription in tests).
    - The eager ``attentions`` branch is kept for parity but is dead code in
      this pipeline (``attentions`` is always ``None``).

    Notes
    -----
    - Do not compose with the DCA prefill method: DCA stores keys rotated at
      cyclic positions ``pos % chunk_len``, breaking the absolute-position
      RoPE assumption of the recomputed last-token query.
    - A length-1 prefill raises (empty ``scores.max()``), matching kvpress.
    """

    compression_ratio: float = 0.0

    @staticmethod
    def _compute_last_token_attention(
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Recompute the last token's attention row over all but the last key.

        Window-size-1 specialization of ``SnapKVPress.compute_window_attention``.
        Returns ``[B, H_q, 1, S - 1]`` in the query dtype.
        """
        head_dim = module.head_dim
        num_key_value_groups = module.config.num_attention_heads // module.config.num_key_value_heads

        query_states = _get_prerope_query_states(module, hidden_states[:, -1:])

        cos, sin = position_embeddings
        cos, sin = cos[:, -1:], sin[:, -1:]
        # Partial rotary (Qwen3.5: rotary_dim < head_dim) — rotate only the first
        # rotary_dim channels and pass the rest through; reduces to full RoPE when
        # rotary_dim == head_dim.
        rotary_dim = cos.shape[-1]
        cos_u, sin_u = cos.unsqueeze(1), sin.unsqueeze(1)
        q_rot, q_pass = query_states[..., :rotary_dim], query_states[..., rotary_dim:]
        q_rot = (q_rot * cos_u) + (rotate_half(q_rot) * sin_u)
        query_states = torch.cat([q_rot, q_pass], dim=-1)

        key_states = repeat_kv(keys, num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = attn_weights[..., :-1]

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
        if attentions is not None:
            attn_weights = attentions[..., -1:, :-1]
        else:
            attn_weights = self._compute_last_token_attention(
                module, hidden_states, keys, kwargs["position_embeddings"]
            )

        scores = attn_weights.mean(1)
        scores = scores.repeat(1, keys.shape[1], 1)

        scores = F.pad(scores, (0, 1), value=scores.max().item())

        return scores
