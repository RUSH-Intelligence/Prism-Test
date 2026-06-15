from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from eval_harness.kv_compression.registry import register_kv_compressor
from eval_harness.kv_compression.base import ScorerKVCompressor


@register_kv_compressor("h2o", aliases=["heavy_hitter_oracle"])
@dataclass
class H2OSketch(ScorerKVCompressor):
    """H2O (Heavy Hitter Oracle) KV cache compression.

    Heavy Hitter Oracle from Zhang et al. (https://arxiv.org/abs/2306.14048):
    keep a recent-token window plus the "heavy hitter" tokens ranked by the
    *accumulated* attention they received. The score for each key is the raw
    column sum of the prefill attention matrix (total attention probability
    that key received across every query able to attend to it), reduced to KV
    heads by averaging over each head's query group; the most recent
    ``window_size`` positions are then force-kept.

    Relationship to ``ObservedAttentionSketch`` (the closest existing scorer)
    ------------------------------------------------------------------------
    H2O differs in exactly two ways:
    1. **Raw accumulated sum, not an average.** ``observed_attention`` divides
       the column sum by the per-position causal-attendee count; H2O keeps the
       raw sum. Because earlier keys are attended by more queries, the raw sum
       structurally favours early tokens — this is the well-known H2O bias and
       is faithful to the paper's cumulative-attention score.
    2. **Recent-window force-keep.** The last ``window_size`` positions are
       always retained (heavy hitters are chosen only from the rest), via the
       same max-score padding idiom as ``SnapKVSketch``.

    Static-prefill approximation
    ----------------------------
    The original H2O is a *streaming* decode-time eviction policy. Here it runs
    as a one-shot ``post_prefill`` scorer over the full prefill attention
    matrix — the standard static simplification (kvpress itself ships only the
    closely-related ``observed_attention`` press for the same reason). Decode is
    a no-op (the hook only fires during prefill).

    Requires ``llm_kwargs: {attn_implementation: "eager"}``: only the eager
    attention forward returns the softmax probabilities to the hook; sdpa/flash
    (the runner default is sdpa) pass ``attentions=None`` and the assert in
    ``score`` fires. Run with ``attention_method='none'`` — a prefill method
    that prunes the cache desynchronizes it from the observed attention matrix
    (guarded below), and DCA replaces the attention forward so ``output[1]`` is
    not eager probabilities.

    Quirks kept for consistency with the existing scorers: the window pad value
    is ``scores.max().item()`` (a global max across batch and heads), so when a
    non-window heavy hitter ties the window max the topk tie-breaking is
    implementation-defined; kept KV pairs stay in score-descending topk order,
    not positional order.

    Parameters
    ----------
    compression_ratio : float, default=0.0
        Fraction of key-value pairs to remove during compression.
    window_size : int, default=64
        Number of most-recent tokens to force-keep. Set ``0`` to disable the
        window (pure heavy-hitter selection).
    """

    compression_ratio: float = 0.0
    window_size: int = 64

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        assert attentions is not None, 'Set attn_implementation="eager" to use this hook'
        if attentions.shape[-1] != keys.shape[2]:
            raise ValueError(
                f"Observed attention covers {attentions.shape[-1]} keys but the cache holds "
                f"{keys.shape[2]}; a prefill method pruned the cache before this sketch fired. "
                "Use attention_method='none' with the h2o sketch."
            )

        bsz, num_key_value_heads, k_len, _ = keys.shape
        w = self.window_size
        assert k_len > w, f"Key length {k_len} must exceed window_size {w}"

        # Accumulated attention each key received (raw column sum, no
        # causal-count normalization — the H2O heavy-hitter score).
        scores = attentions.sum(2)  # [B, H_q, k_len]
        if w > 0:
            scores = scores[..., :-w]  # heavy hitters chosen from non-window keys only

        n = scores.shape[-1]
        scores = scores.view(bsz, num_key_value_heads, -1, n).mean(2)  # group-average to KV heads

        if w > 0:
            # Force-keep the recent window by padding it with the global max score.
            scores = F.pad(scores, (0, w), value=scores.max().item())

        return scores
