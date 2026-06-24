import math
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from torch import nn

from eval_harness.kv_compression.registry import register_kv_compressor
from eval_harness.kv_compression.base import ScorerKVCompressor


@register_kv_compressor("cur")
@dataclass
class CURSketch(ScorerKVCompressor):
    """
    Sketch based on `CurDKV` (https://arxiv.org/abs/2509.15038) which computes approximate leverage scores
    for keys (k2) and values (v2) and combines them to prune the KV cache.

    If `use_random_leverage` is true (default is False), keys and values are first
    multiplied by a random projection matrix G.
    If `use_local_approximation` is true (default), the scores are averaged over a
    local window of size `local_window_size`.
    Depending on `leverage_type`, returns either k2, v2, (k2 + v2) / 2, or k2 * v2 (default)
    Finally, the first `num_sinks` tokens are set to 1.0 to preserve some initial "attention sinks".

    Port of kvpress 0.5.1 ``CURPress`` (``kvpress/presses/cur_press.py``). Scores are
    computed on the RoPE-rotated cached keys, exactly as upstream; squared key norms
    are invariant to the orthogonal RoPE rotation and values are never rotated.

    Deviations from kvpress
    -----------------------
    - The random projection ``G`` is sampled with ``dtype=keys.dtype``. Upstream samples
      float32 unconditionally, so ``use_random_leverage=True`` raises a matmul dtype
      mismatch on bf16/fp16 models; for float32 inputs the draws (and scores) remain
      bitwise identical to upstream under the same seed.

    Upstream quirks replicated on purpose:
    - When ``seed=None`` (kvpress-default behavior), ``G`` is unseeded and resampled
      on every call (i.e. per layer). Set ``seed`` to a non-None int to make the
      random projection deterministic across reruns — the generator is initialized
      once and advances naturally across layers, so per-layer ``G`` still differs
      (just reproducibly).
    - A local window (or the global normalization) with zero total mass yields
      0/0 = NaN scores.
    - ``num_sinks >= seq_len`` sets every score to 1.0, leaving the topk selection
      tie-break dependent; ``num_sinks > n_kept`` keeps an arbitrary subset of sinks.
    """

    num_sinks: int = 4
    leverage_type: Literal["key", "value", "kv_avg", "kv_product"] = "kv_product"
    use_random_leverage: bool = False
    use_local_approximation: bool = True
    local_window_size: int = 16
    seed: Optional[int] = None

    def _get_generator(self, device: torch.device) -> Optional[torch.Generator]:
        if self.seed is None:
            return None
        gen = getattr(self, "_rng", None)
        if gen is None or gen.device != device:
            gen = torch.Generator(device=device).manual_seed(int(self.seed))
            self._rng = gen
        return gen

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:

        if self.use_random_leverage:
            r = 20
            gen = self._get_generator(keys.device)
            G = torch.randn(
                keys.shape[-1], r, device=keys.device, dtype=keys.dtype, generator=gen,
            ) / math.sqrt(r)
            keys = keys @ G
            values = values @ G

        k2 = (keys**2).sum(dim=-1)
        v2 = (values**2).sum(dim=-1)

        if self.use_local_approximation:
            b, h, n = k2.shape
            w = self.local_window_size
            k2 = F.pad(k2, (0, (w - n % w) % w)).reshape(b, h, -1, w)
            k2 = (k2 / k2.sum(dim=-1, keepdim=True)).reshape(b, h, -1)[:, :, :n]
            v2 = F.pad(v2, (0, (w - n % w) % w)).reshape(b, h, -1, w)
            v2 = (v2 / v2.sum(dim=-1, keepdim=True)).reshape(b, h, -1)[:, :, :n]

        if self.leverage_type == "key":
            scores = k2
        elif self.leverage_type == "value":
            scores = v2
        elif self.leverage_type == "kv_avg":
            scores = (k2 + v2) / 2
        elif self.leverage_type == "kv_product":
            scores = k2 * v2
        else:
            raise ValueError("Unknown leverage type: choose from 'kv_avg', 'key', 'value' or 'kv_product'")

        scores /= scores.sum(dim=-1, keepdim=True)
        scores[:, :, : self.num_sinks] = 1.0

        return scores
