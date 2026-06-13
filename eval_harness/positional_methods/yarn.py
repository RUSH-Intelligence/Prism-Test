"""YaRN — Yet another RoPE extensioN (Peng et al. 2023, arXiv:2309.00071).

YaRN is "NTK-by-parts": it splits the rotary dimensions by wavelength and
applies the interpolation ``factor`` only to the low-frequency (long-wavelength)
dimensions, leaving high-frequency ones extrapolated, with a linear ramp across
a correction range set by ``beta_fast``/``beta_slow``.  It also raises the
attention logit temperature by an ``attention_factor`` (``mscale``).

This implementation mirrors transformers' ``_compute_yarn_parameters`` term for
term so it can be tested against that reference (see
``tests/test_positional_methods.py``).  ``factor`` scales the frequencies;
``mscale`` is applied to ``(cos, sin)`` by the Door-1 interceptor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

from .base import PositionalMethod, recover_base_and_dim
from .registry import register_positional_method


@register_positional_method("yarn")
@dataclass
class YaRNMethod(PositionalMethod):
    """YaRN frequency scaling + logit-temperature ``mscale``.

    Parameters
    ----------
    factor : float
        The interpolation scaling factor (target / original context).
    original_max_position_embeddings : int
        The pretraining context length (sets the correction range).
    beta_fast, beta_slow : float
        Extrapolation / interpolation boundary rotations (paper defaults 32, 1).
    attention_factor : float, optional
        Explicit logit temperature.  ``None`` → ``0.1·ln(factor)+1`` (or 1 when
        ``factor <= 1``), matching the paper / HF.
    truncate : bool
        Whether to floor/ceil the correction range (HF default True).
    """

    factor: float = 1.0
    original_max_position_embeddings: int = 4096
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    attention_factor: Optional[float] = None
    truncate: bool = True

    def __post_init__(self) -> None:
        if self.factor < 1.0:
            raise ValueError(f"yarn factor must be >= 1, got {self.factor}")
        if self.attention_factor is None:
            self.attention_factor = self._default_mscale(self.factor)
        # The interceptor applies mscale to (cos, sin); YaRN's temperature lives
        # there so prefill and decode both see it.
        self.mscale = float(self.attention_factor)

    @staticmethod
    def _default_mscale(scale: float) -> float:
        if scale <= 1.0:
            return 1.0
        return 0.1 * math.log(scale) + 1.0

    def compute_inv_freq(
        self,
        original_inv_freq: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        del seq_len
        if self.factor == 1.0:
            return original_inv_freq

        base, dim = recover_base_and_dim(original_inv_freq)
        omp = self.original_max_position_embeddings

        pos_freqs = base ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (self.factor * pos_freqs)

        low, high = _find_correction_range(
            self.beta_fast, self.beta_slow, dim, base, omp, self.truncate,
        )
        extrapolation_factor = 1.0 - _linear_ramp_factor(low, high, dim // 2)
        inv_freq = (
            inv_freq_interpolation * (1.0 - extrapolation_factor)
            + inv_freq_extrapolation * extrapolation_factor
        )
        return inv_freq.to(
            dtype=original_inv_freq.dtype, device=original_inv_freq.device,
        )


def _find_correction_dim(
    num_rotations: float, dim: int, base: float, max_position_embeddings: int,
) -> float:
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


def _find_correction_range(
    low_rot: float,
    high_rot: float,
    dim: int,
    base: float,
    max_position_embeddings: int,
    truncate: bool,
) -> tuple[float, float]:
    low = _find_correction_dim(low_rot, dim, base, max_position_embeddings)
    high = _find_correction_dim(high_rot, dim, base, max_position_embeddings)
    if truncate:
        low = math.floor(low)
        high = math.ceil(high)
    return max(low, 0.0), min(high, dim - 1)


def _linear_ramp_factor(minimum: float, maximum: float, dim: int) -> torch.Tensor:
    if minimum == maximum:
        maximum += 0.001  # avoid singularity
    linear = (torch.arange(dim, dtype=torch.float64) - minimum) / (maximum - minimum)
    return torch.clamp(linear, 0.0, 1.0)
