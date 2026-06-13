from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from eval_harness.sketch.sketches.base_sketch import BaseSketch
from eval_harness.sketch.sketches.knorm_sketch import KnormSketch
from eval_harness.sketch.sketches.registry import register_sketch
from eval_harness.sketch.sketches.scorer_sketch import ScorerSketch


@register_sketch("chunkkv")
@dataclass
class ChunkKVSketch(BaseSketch):
    """
    ChunkKV: Semantic-preserving compression with chunk-wise token selection.

    Enhances any ScorerSketch by applying chunk-wise token selection instead of
    global selection. Computes global importance scores, then keeps or drops
    whole chunks of tokens to preserve semantic coherence within local contexts.
    Chunk score = mean over the chunk's tokens of the head-summed global score;
    the ``n_chunks_kept = max(1, int(total_chunks * (1 - compression_ratio)))``
    best chunks survive, so at least one chunk is always kept and compression is
    quantized to chunk granularity.

    Port of kvpress ``ChunkKVPress`` (kvpress/presses/chunkkv_press.py).
    Based on ChunkKV (https://arxiv.org/abs/2502.00299).

    Parameters
    ----------
    press : ScorerSketch, optional
        The underlying scoring method used to compute global importance scores.
        Defaults to ``KnormSketch()`` so the sketch is constructible from flat
        registry/config kwargs (kvpress requires an explicit ScorerPress).
    chunk_length : int, default=20
        Length of each chunk for token selection. The sequence is divided into
        ``kv_len // chunk_length`` complete chunks plus an optional trailing
        partial chunk of ``kv_len % chunk_length`` tokens (which competes in the
        top-k with a score averaged over only its tokens, as in kvpress).
    compression_ratio : float, optional
        Constructor convenience: when provided it is assigned to
        ``press.compression_ratio``. After construction the attribute is a
        property delegating to the inner sketch, mirroring kvpress's delegated
        property (chunkkv_press.py L43-49).

    Behavior notes (faithful to kvpress)
    ------------------------------------
    - When ``kv_len < chunk_length`` (no complete chunk), compression is
      delegated wholesale to ``press.compress``, i.e. per-head top-k selection,
      unlike the head-uniform chunk path.
    - ``compression_ratio == 0`` short-circuits before any other check.

    Deviations from kvpress
    -----------------------
    - Uniform retained length: in kvpress a layer that selects the trailing
      partial chunk keeps ``(n_chunks_kept - 1) * chunk_length + r`` tokens
      while another keeps ``n_chunks_kept * chunk_length`` — ragged across
      layers, which breaks this framework's shared decode mask. Here every
      layer keeps exactly ``min(n_chunks_kept * chunk_length, kv_len)`` tokens:
      a short selection is padded with the highest-scoring (head-summed global
      score) not-yet-selected positions. When ``kv_len % chunk_length == 0``
      the selection is bit-identical to kvpress.
    - Batch handling: kvpress builds chunk indices from batch element 0 only
      (chunkkv_press.py L108); the chunk path here asserts ``batch_size == 1``
      instead of silently applying row 0's selection to every row.
    - ``press`` is optional (defaults to ``KnormSketch``) and
      ``compression_ratio`` is accepted at construction, both so the sketch is
      usable through the registry / ``CacheConfig`` wiring.
    """

    press: Optional[ScorerSketch] = None
    chunk_length: int = 20
    compression_ratio: Optional[float] = None

    def __post_init__(self):
        pending_ratio = self.__dict__.pop("_pending_compression_ratio", None)
        if self.press is None:
            self.press = KnormSketch()
        assert isinstance(self.press, ScorerSketch), "ChunkKVSketch requires a ScorerSketch as input"
        if pending_ratio is not None:
            self.press.compression_ratio = pending_ratio

    def post_init_from_model(self, model):
        self.press.post_init_from_model(model)

    def _get_compression_ratio(self) -> float:
        return self.press.compression_ratio

    def _set_compression_ratio(self, value: Optional[float]) -> None:
        if getattr(self, "press", None) is None:
            self.__dict__["_pending_compression_ratio"] = value
        elif value is not None:
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

        assert attentions is None, "ChunkPress does not support attentions."

        kv_len = keys.shape[2]

        global_scores = self.press.score(module, hidden_states, keys, values, attentions, kwargs)

        num_complete_chunks = kv_len // self.chunk_length
        remaining_tokens = kv_len % self.chunk_length

        if num_complete_chunks == 0:
            return self.press.compress(module, hidden_states, keys, values, attentions, kwargs)

        assert keys.shape[0] == 1, "ChunkKVSketch chunk selection only supports batch size 1"

        main_scores = global_scores[..., : num_complete_chunks * self.chunk_length]
        main_chunk_scores = main_scores.sum(dim=1).view(-1, num_complete_chunks, self.chunk_length)
        main_chunk_scores = main_chunk_scores.mean(dim=-1)

        if remaining_tokens > 0:
            remaining_scores = global_scores[..., -remaining_tokens:]
            remaining_chunk_score = remaining_scores.sum(dim=1).mean(dim=-1, keepdim=True)
            chunk_scores = torch.cat([main_chunk_scores, remaining_chunk_score], dim=-1)
        else:
            chunk_scores = main_chunk_scores

        n_chunks_kept = max(
            1,
            int((num_complete_chunks + (remaining_tokens > 0)) * (1 - self.press.compression_ratio)),
        )
        top_chunks = chunk_scores.topk(n_chunks_kept, dim=-1)

        indices = []
        for chunk_idx in top_chunks.indices[0]:
            if chunk_idx < num_complete_chunks:
                start_idx = chunk_idx * self.chunk_length
                chunk_indices = torch.arange(start_idx, start_idx + self.chunk_length, device=keys.device)
            else:
                chunk_indices = torch.arange(num_complete_chunks * self.chunk_length, kv_len, device=keys.device)
            indices.append(chunk_indices)

        indices = torch.cat(indices)

        # Deviation from kvpress: pad partial-chunk selections to a deterministic
        # per-prefill target so all hooked layers retain the same length (§4.14).
        target_len = min(n_chunks_kept * self.chunk_length, kv_len)
        n_padding = target_len - indices.numel()
        if n_padding > 0:
            selected = torch.zeros(kv_len, dtype=torch.bool, device=keys.device)
            selected[indices] = True
            padding_scores = global_scores.sum(dim=1)[0].masked_fill(selected, float("-inf"))
            indices = torch.cat([indices, padding_scores.topk(n_padding).indices])

        indices = indices.sort()[0]
        indices = indices.view(1, 1, -1, 1).expand(keys.shape[0], keys.shape[1], -1, module.head_dim)

        keys = keys.gather(2, indices).contiguous()
        values = values.gather(2, indices).contiguous()

        return keys, values


# ``compression_ratio`` must be both a dataclass field (so the registry/adapter
# wiring injects the adapter-level ratio at construction) and, as in kvpress
# (chunkkv_press.py L43-49), a property delegating to the inner press. Attaching
# the property after class creation keeps the field in ``dataclasses.fields``
# while routing all attribute access through the inner sketch.
ChunkKVSketch.compression_ratio = property(
    ChunkKVSketch._get_compression_ratio,
    ChunkKVSketch._set_compression_ratio,
)
