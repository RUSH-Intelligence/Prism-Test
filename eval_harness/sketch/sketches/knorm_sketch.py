from dataclasses import dataclass

import torch
from torch import nn

from eval_harness.sketch.sketches.registry import register_sketch
from eval_harness.sketch.sketches.scorer_sketch import ScorerSketch


@register_sketch("knorm", aliases=["knorm_sketch"])
@dataclass
class KnormSketch(ScorerSketch):
    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        return -keys.norm(dim=-1)
