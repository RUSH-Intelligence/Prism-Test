"""BalanceKV — KV cache compression through discrepancy theory.

Faithful port of the official implementation
(https://github.com/ksheth96/BalanceKV, ``src/balanced_walk.py`` +
``src/llama_forward.py``) accompanying the NeurIPS 2025 Spotlight paper
"Streaming Attention Approximation via Discrepancy Theory"
(https://arxiv.org/abs/2502.07861).

Unlike score-and-evict compressors, BalanceKV builds a *coreset* of the cache:
a self-balancing random walk (Banaszczyk vector balancing) selects a balanced
subset of tokens whose geometry — keys *and* values jointly — approximates the
full self-attention output, and the surviving **values are reweighted** to
account for the tokens they stand in for. Because it reweights values rather
than just dropping tokens, it subclasses :class:`KVCompressor` directly instead
of :class:`ScorerKVCompressor`.

Layout, exactly as upstream's ``weightedbw`` prefill path: the cache is split
into ``[sink | middle | window]``; the walk runs only on ``middle``; sink
(first ``n_sink``) and recency window (last ``window_size``) are kept verbatim.
Each balanced-walk iteration halves the middle, so ``itrs`` iterations keep
``len(middle) / 2**itrs`` tokens.

Schedule: ``post_prefill`` (fires once on the full prefill, matching the
official ``kv_cache is None`` branch); decode tokens are appended uncompressed.

Deviations from upstream (intentional, documented):
  - Only the default ``query=None`` walk is ported; the experimental
    query-conditioned kernel and the separate needle-detection variant are
    omitted (they are not BalanceKV's headline method).
  - The walk's kernel math runs in float32 for numerical stability of
    ``exp(...)`` and to avoid int16 weight overflow at larger ``itrs``; token
    *selection* (sign sampling, stable argsort) is unchanged, and the gathered
    keys/values keep their original dtype.
  - ``compression_ratio`` (the framework-wide knob) is mapped to ``itrs`` when
    ``itrs`` is not given explicitly: ``itrs = round(log2(1/(1-ratio)))``
    (clamped ≥ 1). An explicit ``itrs`` wins; with neither set, the paper
    default ``itrs=2`` is used.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
from torch import nn

from eval_harness.kv_compression.base import CompressionSchedule, KVCompressor
from eval_harness.kv_compression.registry import register_kv_compressor


# ----------------------------------------------------------------------
# Balanced-walk core (ported verbatim from BalanceKV/src/balanced_walk.py,
# query=None path).  Operates per (batch, head); halves the token axis each
# iteration and returns indices into the original token order plus per-token
# coreset weights.
# ----------------------------------------------------------------------

def _indexing(key, sort_idx, block_size, value=None):
    indices = sort_idx.unsqueeze(-1).expand(-1, -1, -1, key.shape[-1])
    new_n = math.ceil(sort_idx.shape[-1] / block_size) * block_size
    out_key = torch.nn.functional.pad(
        key.gather(2, indices), (0, 0, 0, new_n - sort_idx.shape[-1]), mode="constant", value=0.0
    )
    out_value = None
    if value is not None:
        out_value = torch.nn.functional.pad(
            value.gather(2, indices), (0, 0, 0, new_n - sort_idx.shape[-1]), mode="constant", value=0.0
        )
    return out_key, out_value


def balanced_walk(key, rng, gamma_, temp_, beta_, itrs, block_size, value=None, sort_idx=None):
    """Self-balancing walk over (key, value) blocks. Returns ``(sort_idx, weight_idx)``."""
    b, h, n, d = key.shape
    if not isinstance(gamma_, list):
        gamma_ = [gamma_] * itrs
    const_denom = 0.025
    if not isinstance(block_size, list):
        block_size = [block_size] * itrs

    weight_idx = None
    for t in range(itrs):
        if sort_idx is not None:
            key_sorted, value_sorted = _indexing(key, sort_idx, block_size[t], value)
            key_sorted = key_sorted.view(b, h, -1, block_size[t], d)
            if value is not None:
                weight_idx_padded = torch.nn.functional.pad(
                    weight_idx, (0, math.ceil(n / block_size[t]) * block_size[t] - weight_idx.shape[-1])
                )
                value_sorted = value_sorted * weight_idx_padded.unsqueeze(-1)
                value_sorted = value_sorted.view(b, h, -1, block_size[t], d)
        else:
            new_n = math.ceil(n / block_size[t]) * block_size[t]
            key_sorted = torch.nn.functional.pad(
                key, (0, 0, 0, new_n - n), mode="constant", value=0.0
            ).view(b, h, -1, block_size[t], d)
            value_sorted = None
            if value is not None:
                value_sorted = torch.nn.functional.pad(
                    value, (0, 0, 0, new_n - n), mode="constant", value=0.0
                ).view(b, h, -1, block_size[t], d)

        normal_keys = key_sorted - torch.mean(key_sorted, dim=-2, keepdim=True)
        kernel_ = torch.exp(
            temp_ * torch.einsum("...nd,...sd->...ns", normal_keys, normal_keys) / math.sqrt(d) - beta_
        )
        if value is not None:
            kernel_ *= 1e-8 + torch.einsum("...nd,...sd->...ns", value_sorted, value_sorted) + const_denom

        signs = torch.zeros(kernel_.shape[:4], dtype=torch.float32, device=kernel_.device)
        signs[:, :, :, 0] = 1
        rand_tensor = torch.rand(signs.shape, generator=rng, device=key.device)
        for i in range(1, kernel_.shape[3]):
            partial_inner_prod = (kernel_[:, :, :, i, :] * signs).sum(dim=-1)
            samp_prb = 0.5 - gamma_[t] * partial_inner_prod
            signs[:, :, :, i] = 2 * (rand_tensor[:, :, :, i] < samp_prb) - 1

        signs = signs.view(b, h, -1)[:, :, :n]
        if signs.shape[-1] == 0:
            sort_idx = signs[:, :, :0].long()
            weight_idx = signs[:, :, :0]
            break

        cumsum_neg = (signs == -1).cumsum(dim=-1)
        cumsum_pos = (signs == 1).cumsum(dim=-1)
        c_neg = torch.argmax((cumsum_neg == n // 2).to(torch.int64), dim=-1)
        c_pos = torch.argmax((cumsum_pos == n // 2).to(torch.int64), dim=-1)
        c = torch.maximum(c_neg, c_pos).to(signs.device)

        weight = signs.clone()
        indices = torch.arange(signs.shape[2], device=signs.device).view(1, 1, -1)
        mask_after_c = indices > c.unsqueeze(-1)
        weight[mask_after_c] = torch.abs(weight[mask_after_c])
        mask_flip_needed = (signs.gather(2, c.unsqueeze(-1)) == 1).squeeze(-1)
        mask_before_c = indices <= c.unsqueeze(-1)
        weight[mask_before_c] *= 2
        flip_mask = mask_before_c & mask_flip_needed.unsqueeze(-1)
        weight[flip_mask] *= -1

        weight_argsort = torch.argsort(-weight, dim=-1, stable=True)

        n = n // 2
        if sort_idx is None:
            sort_idx = weight_argsort[:, :, :n]
            weight_idx = weight.gather(-1, weight_argsort[:, :, :n])
        else:
            sort_idx = sort_idx.gather(2, weight_argsort[:, :, :n])
            weight_idx_1 = weight.gather(-1, weight_argsort[:, :, :n])
            weight_idx = weight_idx.gather(-1, weight_argsort[:, :, :n]) * weight_idx_1

    return sort_idx, weight_idx


@register_kv_compressor("balancekv")
@dataclass
class BalanceKVSketch(KVCompressor):
    """BalanceKV coreset compressor (discrepancy-theory KV compression).

    Parameters
    ----------
    compression_ratio : float, default=0.0
        Framework-wide prune fraction; mapped to ``itrs`` when ``itrs`` is None.
    itrs : int | None, default=None
        Number of balanced-walk halvings on the middle region. Explicit value
        wins over ``compression_ratio``; if both are unset, defaults to 2.
    gamma, temp, beta : float
        Walk hyperparameters (paper/repo defaults 4.0 / 1.0 / 0.0).
    block_size : int, default=128
        Block granularity of the walk.
    n_sink : int, default=32
        Leading tokens kept verbatim (attention sinks).
    window_size : int, default=32
        Trailing recency tokens kept verbatim.
    seed : int, default=42
        Seed for the walk's RNG.
    """

    compression_ratio: float = 0.0
    itrs: Optional[int] = None
    gamma: float = 4.0
    temp: float = 1.0
    beta: float = 0.0
    block_size: int = 128
    n_sink: int = 32
    window_size: int = 32
    seed: int = 42
    schedule: frozenset = field(
        default_factory=lambda: frozenset({CompressionSchedule.POST_PREFILL}),
        kw_only=True,
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        assert 0 <= self.compression_ratio < 1, "compression_ratio must be in [0, 1)"
        if self.itrs is None:
            if self.compression_ratio > 0:
                self.itrs = max(1, round(math.log2(1.0 / (1.0 - self.compression_ratio))))
            else:
                self.itrs = 2
        assert self.itrs >= 1, "itrs must be >= 1"
        self._generators: dict = {}

    def _generator(self, device: torch.device) -> torch.Generator:
        gen = self._generators.get(device)
        if gen is None:
            gen = torch.Generator(device=device)
            gen.manual_seed(self.seed)
            self._generators[device] = gen
        return gen

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        k_len = keys.shape[2]
        # Nothing to compress if the middle region is too small to split.
        middle_len = k_len - self.n_sink - self.window_size
        if middle_len <= 0 or middle_len < (1 << self.itrs):
            return keys, values

        sink_k, sink_v = keys[:, :, : self.n_sink], values[:, :, : self.n_sink]
        win_k, win_v = keys[:, :, -self.window_size :], values[:, :, -self.window_size :]
        mid_k = keys[:, :, self.n_sink : k_len - self.window_size]
        mid_v = values[:, :, self.n_sink : k_len - self.window_size]

        rng = self._generator(keys.device)
        # Selection math in float32 for stability; gather from original dtype.
        indices, weights = balanced_walk(
            mid_k.float(), rng, self.gamma, self.temp, self.beta,
            self.itrs, self.block_size, value=mid_v.float(),
        )

        gather_idx = indices.unsqueeze(-1).expand(-1, -1, -1, keys.shape[-1])
        kept_k = mid_k.gather(2, gather_idx)
        kept_v = mid_v.gather(2, gather_idx)

        # Reweight values by the coreset weights (matches upstream's
        # ``weights / 2**itrs`` value scaling).
        w = (weights / (2 ** self.itrs)).unsqueeze(-1).to(kept_v.dtype)
        kept_v = kept_v * w

        new_keys = torch.cat([sink_k, kept_k, win_k], dim=2).contiguous()
        new_values = torch.cat([sink_v, kept_v, win_v], dim=2).contiguous()
        return new_keys, new_values
