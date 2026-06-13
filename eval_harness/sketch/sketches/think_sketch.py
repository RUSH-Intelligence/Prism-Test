from dataclasses import dataclass

import torch
from torch import nn
from transformers.models.llama.modeling_llama import rotate_half

from eval_harness.sketch.sketches.base_sketch import BaseSketch
from eval_harness.sketch.sketches.registry import register_sketch


def _get_prerope_query_states(module: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    """Pre-RoPE query states, transcribed from kvpress/utils.py::get_prerope_query_states.

    The upstream ``isinstance`` checks (Phi3Attention fused projection,
    Qwen3/Gemma3 qk-norm) are replaced with duck-typing on ``qkv_proj`` /
    ``q_norm`` attributes, the established Prism-Test pattern.
    """
    bsz, q_len, _ = hidden_states.shape
    num_heads = module.config.num_attention_heads
    head_dim = module.head_dim

    if hasattr(module, "qkv_proj"):
        qkv = module.qkv_proj(hidden_states)
        query_states = qkv[..., : num_heads * head_dim]
    elif hasattr(module, "q_proj"):
        query_states = module.q_proj(hidden_states)
    else:
        raise NotImplementedError(f"Sketch not yet implemented for {module.__class__}.")

    query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)

    q_norm = getattr(module, "q_norm", None)
    if q_norm is not None:
        query_states = q_norm(query_states)

    return query_states


@register_sketch("think")
@dataclass
class ThinKSketch(BaseSketch):
    """ThinK: channel-wise key compression for transformer attention.

    Port of kvpress 0.5.1 ``ThinKPress``
    (kvpress/presses/think_press.py), based on ThinK
    (https://arxiv.org/pdf/2407.21018).

    ThinK compresses the dimensions (head_dim channels) of the keys, not the
    sequence length. Channel importance is the product of a per-channel query
    moment, computed from the recomputed post-RoPE queries of the last
    ``window_size`` prefill tokens, and a per-channel key moment over all
    cached (post-RoPE) keys. The ``int(head_dim *
    key_channel_compression_ratio)`` lowest-scoring key channels per (batch,
    kv-head) are zeroed in place; key shapes are unchanged and values are
    untouched, so there is no memory gain (matching kvpress), but a zeroed
    key channel contributes exactly 0 to every q.k dot product, which is
    mathematically identical to dropping the channel.

    ``compression_ratio`` is a read-only property equal to
    ``key_channel_compression_ratio / 2`` (keys are half the KV cache); it is
    deliberately not a dataclass field, so ``ResearchAdapter._build_sketch``
    does not inject the adapter-level ratio — configure via
    ``sketch_kwargs={key_channel_compression_ratio, window_size}``.

    Deviations from kvpress
    -----------------------
    - ``get_prerope_query_states`` is not available in
      ``eval_harness/sketch/utils.py``; it is transcribed locally with
      duck-typed module introspection (``qkv_proj`` presence for Phi3-style
      fused projections, ``q_norm`` attribute for Qwen3/Gemma3 qk-norm)
      instead of ``isinstance`` checks.
    - ``__post_init__`` asserts ``0 <= key_channel_compression_ratio < 1``;
      kvpress does not guard (a ratio of 1.0 would silently zero every key
      channel).
    - Combining with ``prefill_method: dca`` is approximate/unsupported: DCA
      caches keys rotated at cyclic positions ``pos % chunk_len``, so the
      cached-key channel statistics and ``kwargs["position_embeddings"]`` no
      longer correspond to DCA's internal positioning.

    Parameters
    ----------
    key_channel_compression_ratio : float, default=0.0
        Fraction of key channels (dimensions) to remove during compression.
    window_size : int, default=32
        Number of recent tokens to use for computing key channel importance.
    """

    key_channel_compression_ratio: float = 0.0
    window_size: int = 32

    def __post_init__(self):
        assert 0 <= self.key_channel_compression_ratio < 1, (
            "key_channel_compression_ratio must be in [0, 1)"
        )

    def compute_window_queries(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Re-compute the last window_size post-RoPE query states."""
        query_states = _get_prerope_query_states(module, hidden_states[:, -self.window_size :])

        cos, sin = position_embeddings
        cos, sin = cos[:, -self.window_size :], sin[:, -self.window_size :]
        query_states = (query_states * cos.unsqueeze(1)) + (rotate_half(query_states) * sin.unsqueeze(1))

        return query_states

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.key_channel_compression_ratio == 0:
            return keys, values

        bsz, num_key_value_heads, k_len, head_dim = keys.shape
        num_key_value_groups = module.config.num_attention_heads // num_key_value_heads

        queries = self.compute_window_queries(module, kwargs["hidden_states"], kwargs["position_embeddings"])
        queries_norm = torch.pow(queries, 2).mean(dim=2)
        queries_norm = queries_norm.view(bsz, num_key_value_heads, num_key_value_groups, module.head_dim).mean(2)
        keys_norm = torch.pow(keys, 2).mean(dim=2)
        key_scores = queries_norm * keys_norm

        n_pruned = int(head_dim * self.key_channel_compression_ratio)
        indices = key_scores.topk(n_pruned, dim=-1, largest=False).indices
        indices = indices.unsqueeze(2).expand(-1, -1, k_len, -1)
        keys = keys.scatter_(-1, indices, 0)

        return keys, values

    @property
    def compression_ratio(self) -> float:
        return self.key_channel_compression_ratio / 2

    @compression_ratio.setter
    def compression_ratio(self, value):
        raise AttributeError(f"compression ratio cannot be set for {type(self).__name__}")
