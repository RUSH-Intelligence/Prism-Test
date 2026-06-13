from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from eval_harness.sketch.sketches.registry import register_sketch
from eval_harness.sketch.sketches.scorer_sketch import ScorerSketch


@register_sketch("keydiff")
@dataclass
class KeyDiffSketch(ScorerSketch):
    """
    KeyDiff: Key similarity-based KV cache compression.

    Evicts tokens based on key vector similarity to the average key pattern.
    Identifies tokens with most similar keys to the per-head mean key
    direction and removes them, keeping tokens with more distinctive key
    vectors: anchor = mean over the sequence of L2-normalized keys;
    score = -cosine_similarity(key, anchor). Queries, values, and attention
    weights are never used.

    Based on KeyDiff (https://arxiv.org/abs/2504.15364); direct port of
    kvpress ``KeyDiffPress`` (kvpress/presses/keydiff_press.py). Scores are
    computed on the RoPE-rotated keys stored in the cache — exactly what the
    kvpress hook feeds its reference implementation, so this is parity, not a
    deviation.

    Deviations from kvpress
    -----------------------
    None in the scoring math. As in bare ``KeyDiffPress``, this is the
    one-shot variant: the paper's block-wise iterative compression exists
    upstream only via ``BlockPress(press=KeyDiffPress(...), block_size=N)``,
    which has no Prism-Test analog. Note also that under the DCA prefill
    method cached keys are rotated at cyclic positions (``pos % chunk_len``),
    so scores differ from a vanilla-RoPE run (the selection mechanism itself
    remains valid).

    Parameters
    ----------
    compression_ratio : float, default=0.0
        Fraction of key-value pairs to remove during compression.
    """

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        anchor = F.normalize(keys, p=2, dim=-1).mean(dim=2, keepdim=True)
        return -F.cosine_similarity(keys, anchor, dim=-1)
