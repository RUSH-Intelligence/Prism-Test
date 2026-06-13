from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from eval_harness.kv_compression.registry import register_kv_compressor
from eval_harness.kv_compression.base import ScorerKVCompressor


@register_kv_compressor("random", aliases=["random_sketch"])
@dataclass
class RandomSketch(ScorerKVCompressor):
    compression_ratio: float = 0.0
    seed: Optional[int] = None

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        generator = None
        if self.seed is not None:
            generator = torch.Generator()
            generator.manual_seed(self.seed)
        return torch.rand(*keys.shape[:-1], generator=generator, device=keys.device, dtype=keys.dtype)
