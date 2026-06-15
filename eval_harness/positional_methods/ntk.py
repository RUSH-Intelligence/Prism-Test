"""NTK-aware RoPE scaling (static "NTK-aware" base interpolation).

The NTK-aware trick (bloc97, 2023) extends context without fine-tuning by
*scaling the RoPE base* ``theta`` rather than the positions::

    theta' = theta * factor ** (dim / (dim - 2))

so high-frequency dimensions are nearly untouched (extrapolation) while
low-frequency ones are interpolated.  This is a pure **frequency** change, so it
overrides Door 1's :meth:`compute_inv_freq`.  ``factor == 1`` is the identity.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .base import PositionalMethod, recover_base_and_dim
from .registry import register_positional_method


@register_positional_method("ntk", aliases=["ntk_aware", "ntk-aware"])
@dataclass
class NTKMethod(PositionalMethod):
    """Static NTK-aware base scaling by ``factor`` (>= 1)."""

    factor: float = 1.0

    def __post_init__(self) -> None:
        if self.factor < 1.0:
            raise ValueError(f"ntk factor must be >= 1, got {self.factor}")

    def compute_inv_freq(
        self,
        original_inv_freq: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        del seq_len
        if self.factor == 1.0:
            return original_inv_freq

        base, dim = recover_base_and_dim(original_inv_freq)
        new_base = base * (self.factor ** (dim / (dim - 2)))
        exponents = torch.arange(0, dim, 2, dtype=torch.float64) / dim
        inv_freq = 1.0 / (new_base ** exponents)
        return inv_freq.to(
            dtype=original_inv_freq.dtype, device=original_inv_freq.device,
        )
