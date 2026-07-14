from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

from eval_harness.kv_compression.registry import register_kv_compressor
from eval_harness.kv_compression.base import ScorerKVCompressor


@register_kv_compressor("top_k_sampling", aliases=["top_k_sampling_sketch"])
@dataclass
class TopKSamplingSketch(ScorerKVCompressor):
    """Deterministic top-k core + uniform random tail sample.

    A simple hybrid eviction baseline. With per-head keep-budget
    ``n_kept = int(T * (1 - compression_ratio))`` (the framework convention,
    matching :meth:`ScorerKVCompressor.compress`):

    1. **Core** — the top ``n_top = round(n_kept * top_frac)`` tokens by
       key-norm score ``-||k||`` (KnormSketch semantics: RoPE is orthogonal so
       norms are position-invariant; no attention probabilities, queries, or
       ``position_embeddings`` needed, so this composes with sdpa/flash-attn
       and hybrid models such as NemotronH out of the box).
    2. **Tail** — the remaining ``n_kept - n_top`` slots are filled by a
       uniform random sample *without replacement* from the not-yet-selected
       tokens, drawn independently per (batch, kv-head).

    ``top_frac=1.0`` reduces to the pure ``knorm`` baseline; ``top_frac=0.0``
    to a pure uniform-random baseline.

    Randomness: a fresh ``torch.Generator`` on the keys' device is seeded with
    ``seed + module.layer_idx`` on every :meth:`compress` call, so repeated
    runs — and repeated prefills within a run — are reproducible, the global
    RNG state is never consumed, and each layer draws its own tail sample
    (a fixed shared seed would give every layer the identical "random"
    pattern). ``seed=None`` falls back to the global RNG. Random draws use
    float32 regardless of the cache dtype so low-precision ties don't bias
    selection toward early positions.

    Like the base scorer, kept tokens come out in score order, not positional
    order. ``post_prefill`` schedule (the default) is the intended use.
    """

    top_frac: float = 0.75
    seed: Optional[int] = 42

    def __post_init__(self) -> None:
        super().__post_init__()
        assert 0.0 <= self.top_frac <= 1.0, "top_frac must be in [0, 1]"

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

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.compression_ratio == 0:
            return keys, values

        bsz, num_kv_heads, k_len, head_dim = keys.shape
        n_kept = int(k_len * (1 - self.compression_ratio))
        n_top = min(int(round(n_kept * self.top_frac)), n_kept)

        generator = None
        if self.seed is not None:
            generator = torch.Generator(device=keys.device)
            layer_idx = getattr(module, "layer_idx", None)
            generator.manual_seed(self.seed + (layer_idx or 0))
        selection = torch.rand(
            bsz, num_kv_heads, k_len,
            generator=generator, device=keys.device, dtype=torch.float32,
        )
        if n_top > 0:
            core_scores = self.score(module, hidden_states, keys, values, attentions, kwargs)
            core_indices = core_scores.topk(n_top, dim=-1).indices
            selection.scatter_(-1, core_indices, torch.inf)

        indices = selection.topk(n_kept, dim=-1).indices
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

        keys = keys.gather(2, indices).contiguous()
        values = values.gather(2, indices).contiguous()

        return keys, values
