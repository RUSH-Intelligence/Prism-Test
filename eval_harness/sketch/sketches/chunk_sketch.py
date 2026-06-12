from dataclasses import dataclass

import torch
from torch import nn
from transformers import PreTrainedModel

from eval_harness.sketch.sketches.base_sketch import BaseSketch
from eval_harness.sketch.sketches.registry import register_sketch
from eval_harness.sketch.sketches.scorer_sketch import ScorerSketch


@register_sketch("chunk")
@dataclass
class ChunkSketch(BaseSketch):
    """
    ChunkPress: Uniform compression through independent chunk processing.

    This wrapper enhances any ScorerSketch by applying compression independently
    to fixed-size chunks of the sequence. Unlike global compression methods that
    may concentrate selection in high-importance regions, chunked compression is
    uniform across the entire context because each chunk is scored and pruned
    separately.

    Based on FINCH (https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00716/125280).
    Direct port of kvpress 0.5.1 ``ChunkPress`` (presses/chunk_press.py).

    Parameters
    ----------
    press : ScorerSketch
        The underlying scoring method to apply to each chunk independently.
    chunk_length : int, default=1024
        Length of each chunk for independent compression. The last chunk may be
        shorter; its kept count uses the actual slice length.

    Replicated kvpress quirks (intentional, do not "fix"):
    - Per-chunk ``n_kept = max(1, int(L * (1 - ratio)))`` keeps at least one
      token per chunk, so effective compression can be lower than the nominal
      ratio for short/last chunks (global ``ScorerSketch`` has no such floor).
    - The ``compression_ratio == 0`` early return precedes the
      ``attentions is None`` assert, so non-None attentions do not raise at
      ratio 0.
    - ``kwargs`` is forwarded to the wrapped scorer UNSLICED: chunk-sliced
      hidden_states/keys/values but full-length position_embeddings /
      cache_position. Moot for kwargs-agnostic scorers (Knorm, Random).
    - Kept positions within each chunk are in descending-score order, not
      positional order (same as ``ScorerSketch``).

    Deviations from kvpress: none in the compression math. Prism-specific
    caveat: prefill-method hooks (e.g. ReAttention) fire before sketch hooks
    and may have already pruned the cache, making ``keys`` shorter than
    ``hidden_states``; the key-indexed chunk slicing of ``hidden_states`` then
    misaligns, so combining with cache-pruning prefill methods is only safe for
    hidden_states-agnostic wrapped scorers.
    """

    press: ScorerSketch
    chunk_length: int = 1024

    def __post_init__(self):
        assert isinstance(self.press, ScorerSketch), "ChunkSketch requires a ScorerSketch as input"

    def post_init_from_model(self, model: PreTrainedModel):
        self.press.post_init_from_model(model)

    @property
    def compression_ratio(self) -> float:
        return self.press.compression_ratio

    @compression_ratio.setter
    def compression_ratio(self, value: float):
        self.press.compression_ratio = value

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if self.press.compression_ratio == 0:
            return keys, values

        assert attentions is None, "ChunkSketch does not support attentions."

        kv_len = keys.shape[2]
        indices = []
        for i in range(0, kv_len, self.chunk_length):
            chunk_scores = self.press.score(
                module,
                hidden_states[:, i : i + self.chunk_length],
                keys[:, :, i : i + self.chunk_length],
                values[:, :, i : i + self.chunk_length],
                attentions,
                kwargs,
            )
            chunk_length = keys[:, :, i : i + self.chunk_length].shape[2]
            n_kept = max(1, int(chunk_length * (1 - self.press.compression_ratio)))
            chunk_indices = i + chunk_scores.topk(n_kept, dim=-1).indices
            indices.append(chunk_indices)

        indices = torch.cat(indices, dim=-1)
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)

        keys = keys.gather(2, indices).contiguous()
        values = values.gather(2, indices).contiguous()

        return keys, values
