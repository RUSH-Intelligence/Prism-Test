from dataclasses import dataclass

import torch
from torch import nn

from eval_harness.sketch.sketches.registry import register_sketch
from eval_harness.sketch.sketches.scorer_sketch import ScorerSketch


@register_sketch("observed_attention")
@dataclass
class ObservedAttentionSketch(ScorerSketch):
    """Observed attention-based KV cache compression.

    Port of kvpress ``ObservedAttentionPress`` (kvpress 0.5.1,
    ``presses/observed_attention_press.py``). Computes importance scores based
    on actual attention weights observed during the prefill forward pass. The
    score for each key-value pair is the average attention weight it receives
    from the query tokens able to attend to it (causal column sum divided by
    the number of attendable queries), reduced to KV heads by averaging over
    each head's query group.

    Related to H2O (https://arxiv.org/abs/2306.14048).

    Requires ``llm_kwargs: {attn_implementation: "eager"}``: in transformers
    5.x the eager attention forward returns the softmax probabilities
    unconditionally, so the hook receives them as ``output[1]``; sdpa/flash
    (the runner default is sdpa) return ``attentions=None`` and the assert in
    ``score`` fires. Run with ``prefill_method: none`` — prefill-method hooks
    fire before sketch hooks and may prune the cache, desynchronizing it from
    the observed attention matrix (see the explicit guard below), and DCA
    replaces the attention forward entirely so ``output[1]`` is not eager
    probabilities.

    Deviations from kvpress
    -----------------------
    - Adds an explicit ``ValueError`` when ``attentions.shape[-1] !=
      keys.shape[2]`` (a prefill-method hook pruned the cache before this
      sketch fired); kvpress would fail with an opaque broadcast/view error.

    Parameters
    ----------
    compression_ratio : float, default=0.0
        Fraction of key-value pairs to remove during compression.
    """

    compression_ratio: float = 0.0

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
                "Use prefill_method='none' with the observed_attention sketch."
            )
        scores = attentions.sum(2)
        bsz, num_key_value_heads, n_tokens, _ = keys.shape
        n_tokens_in_sum = torch.arange(n_tokens, 0, -1).to(attentions.device, attentions.dtype)
        scores = scores / n_tokens_in_sum
        scores = scores.view(bsz, num_key_value_heads, -1, n_tokens).mean(2)
        return scores
