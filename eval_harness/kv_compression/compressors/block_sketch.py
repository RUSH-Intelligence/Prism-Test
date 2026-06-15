from dataclasses import dataclass

import torch
from torch import nn

from eval_harness.kv_compression.base import KVCompressor
from eval_harness.kv_compression.registry import register_kv_compressor
from eval_harness.kv_compression.base import ScorerKVCompressor


@register_kv_compressor("block")
@dataclass
class BlockSketch(KVCompressor):
    """
    BlockPress: block-wise iterative KV cache compression.

    Simulates the block prompt processing described in the KeyDiff paper
    (https://arxiv.org/abs/2504.15364): the wrapped scorer's global top-k is
    replaced by a streaming top-k that segments the sequence into
    non-overlapping blocks and compresses iteratively — each step scores the
    candidate set (currently kept tokens + next block) with the wrapped
    ``ScorerKVCompressor`` and keeps the ``n_kept`` best. This is not a true
    chunked-prefill implementation: all inputs are computed in a single
    forward pass before block-wise scoring and pruning.

    Direct port of kvpress ``BlockPress`` (kvpress/presses/block_press.py).
    For per-token scorers (e.g. Knorm) the streaming top-k keeps the same SET
    as the plain wrapped scorer; for set-dependent scorers (e.g. KeyDiff)
    results genuinely depend on ``block_size``, which is the paper's intended
    behavior. ``block_size >= k_len`` degenerates to the plain wrapped scorer.

    Upstream quirks replicated faithfully:

    - Kept K/V are stored in descending-score (position-unsorted) order.
    - Once per-head kept indices diverge, the pseudo hidden states passed to
      the inner ``score()`` mix tokens across per-KV-head channel groups
      (block_press.py:71,80-81), so only hidden-state-agnostic scorers
      (Knorm, KeyDiff, Random) are exact; projection-based scorers
      (SnapKV-style) would silently score garbage rows.
    - ``kwargs`` are forwarded unmodified while keys/values are a gathered
      subset: ``cache_position``/``position_embeddings`` still describe the
      full sequence, so the inner scorer must not consume them.
    - Requires ``hidden_states.shape[1] == keys.shape[2]`` (holds in Prism:
      prefill is one full-context pass) and the hidden size divisible by
      ``num_key_value_heads``.

    The wrapped sketch must be constructed programmatically (a nested
    ``ScorerKVCompressor`` is not expressible as flat config kwargs). Under the DCA
    prefill method cached keys are rotated at cyclic positions
    (``pos % chunk_len``); key-similarity inner scorers then score DCA's
    representation — an untested combination.

    Deviations from kvpress
    -----------------------
    - ``press`` field renamed ``sketch`` (framework naming; wraps a
      ``ScorerKVCompressor`` instead of a ``ScorerPress``).
    - ``__post_init__`` additionally asserts ``block_size >= 1``: upstream
      does not validate it, and ``block_size <= 0`` is undefined behavior
      (an invalid or empty ``range`` step that silently keeps the first
      ``n_kept`` tokens).

    Parameters
    ----------
    sketch : ScorerKVCompressor
        The underlying scoring method used to evaluate token importance
        within each block.
    block_size : int, default=128
        Size of each block for iterative compression.
    """

    sketch: ScorerKVCompressor
    block_size: int = 128

    def __post_init__(self):
        super().__post_init__()
        assert isinstance(self.sketch, ScorerKVCompressor), "BlockSketch requires a ScorerKVCompressor"
        assert self.block_size >= 1, "block_size must be a positive integer"

    def post_init_from_model(self, model):
        self.sketch.post_init_from_model(model)

    @property
    def compression_ratio(self):
        return self.sketch.compression_ratio

    @compression_ratio.setter
    def compression_ratio(self, value):
        self.sketch.compression_ratio = value

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.sketch.compression_ratio == 0:
            return keys, values

        assert attentions is None, "BlockPress does not support attentions."

        bsz, num_key_value_heads, k_len, head_dim = keys.shape

        block_size = self.block_size if self.block_size < k_len else k_len
        n_kept = int(k_len * (1 - self.compression_ratio))

        kept_indices = torch.arange(n_kept, device=keys.device).expand(bsz, num_key_value_heads, -1)

        states = hidden_states.view(bsz, k_len, num_key_value_heads, -1).transpose(1, 2)

        for i in range(n_kept, k_len, block_size):
            end = min(i + block_size, k_len)
            current_indices = torch.arange(i, end, device=keys.device).expand(bsz, num_key_value_heads, -1)
            current_indices = torch.cat([kept_indices, current_indices], dim=-1)

            current_states = states.gather(2, current_indices.unsqueeze(-1).expand(-1, -1, -1, states.shape[-1]))
            current_states = current_states.transpose(1, 2).reshape(bsz, -1, hidden_states.shape[-1])

            scores = self.sketch.score(
                module,
                current_states,
                keys.gather(2, current_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)),
                values.gather(2, current_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)),
                attentions,
                kwargs,
            )
            topk_indices = scores.topk(n_kept, dim=-1).indices
            kept_indices = current_indices.gather(-1, topk_indices)

        kept_indices = kept_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        keys = keys.gather(2, kept_indices).contiguous()
        values = values.gather(2, kept_indices).contiguous()

        return keys, values
