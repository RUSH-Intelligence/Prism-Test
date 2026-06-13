"""Linear Positional Interpolation (Chen et al. 2023, arXiv:2306.15595).

Linear-PI extends context by *compressing* the position indices: every absolute
position is divided by a scaling ``factor`` so positions that would exceed the
pretraining window fall back inside it.  This is a pure **position remap** — the
frequencies are untouched — so it is the textbook Door-1 ``remap_position_ids``
method (mathematically identical to dividing ``inv_freq`` by ``factor``, as HF's
``"linear"`` rope does).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .base import PositionalMethod
from .registry import register_positional_method


@register_positional_method("linear_pi", aliases=["linear", "pi"])
@dataclass
class LinearPIMethod(PositionalMethod):
    """Divide absolute positions by ``factor`` (>= 1)."""

    factor: float = 1.0

    def __post_init__(self) -> None:
        if self.factor < 1.0:
            raise ValueError(f"linear_pi factor must be >= 1, got {self.factor}")

    def remap_position_ids(
        self,
        position_ids: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        del seq_len
        if self.factor == 1.0:
            return position_ids
        return position_ids / self.factor
