from dataclasses import dataclass
import math

import torch
from torch import nn

from eval_harness.kv_compression.registry import register_kv_compressor
from eval_harness.kv_compression.base import ScorerKVCompressor


@register_kv_compressor("reattention", aliases=["reattention_sketch"])
@dataclass
class ReAttentionSketch(ScorerKVCompressor):
    """Score keys by query-key affinity and keep top-K positions.

    The score for each key position is the maximum q·k similarity across queries
    in the current forward pass (per KV head).
    """

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        del values, attentions, kwargs

        if not hasattr(module, "q_proj"):
            # Fallback for unsupported attention modules.
            return -keys.norm(dim=-1)

        batch_size, _, _, head_dim = keys.shape
        q_len = hidden_states.shape[1]

        queries = module.q_proj(hidden_states)
        queries = queries.view(batch_size, q_len, module.num_heads, head_dim).transpose(1, 2)

        # Map query heads to KV heads for GQA/MQA models.
        n_kv_heads = keys.shape[1]
        if queries.shape[1] != n_kv_heads:
            group_size = max(1, queries.shape[1] // n_kv_heads)
            queries = queries[:, : n_kv_heads * group_size, :, :]
            queries = queries.reshape(batch_size, n_kv_heads, group_size, q_len, head_dim).mean(dim=2)

        scale = 1.0 / math.sqrt(float(head_dim))
        scores = torch.matmul(queries.float(), keys.float().transpose(-1, -2)) * scale

        # Keep keys that are highly attended by at least one query token.
        return scores.amax(dim=2).to(keys.dtype)
