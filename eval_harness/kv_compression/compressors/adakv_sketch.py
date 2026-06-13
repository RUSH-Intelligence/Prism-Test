from dataclasses import dataclass

import torch
from torch import nn
from transformers import PreTrainedModel

from eval_harness.kv_compression.base import KVCompressor
from eval_harness.kv_compression.registry import register_kv_compressor
from eval_harness.kv_compression.base import ScorerKVCompressor


@register_kv_compressor("adakv")
@dataclass
class AdaKVSketch(KVCompressor):
    """
    AdaKV: Adaptive head-wise KV cache compression.

    Performs head-specific compression by selecting top-k tokens across all heads
    based on importance scores. Applies safeguards to ensure each head retains
    a minimum fraction of tokens.

    Based on AdaKV (https://arxiv.org/abs/2407.11550). Port of kvpress
    ``AdaKVPress`` (kvpress/presses/adakv_press.py).

    The cache is never physically pruned: ``compress`` returns keys/values
    unchanged at full length and records ``module.masked_key_indices`` =
    ``(batch_indices, head_indices, seq_indices)`` of the pruned entries. The
    globally installed attention patch (``eval_harness/kv_compression/attention_patch.py``,
    applied at ``eval_harness.kv_compression`` import over ``ALL_ATTENTION_FUNCTIONS``)
    overwrites those key slots with fake keys such that ``exp(<q, k_fake>) == 0``
    on every ``q_len < k_len`` forward (question pass and each decode step), and
    resets the indices on the next prefill (``q_len == k_len``). Consequently:

    - per-head budgets are purely logical — zero memory savings, and logged
      cache lengths stay at the full context length;
    - a non-eager attention implementation is required (eager bypasses
      ``ALL_ATTENTION_FUNCTIONS``; the research runner defaults to sdpa);
    - incompatible with prefill methods that replace ``self_attn.forward``
      wholesale (``dca``, ``reattention_exact``) — the mask would be silently
      ignored. Use with ``prefill_method: none`` or hook-style methods.

    Parameters
    ----------
    press : ScorerKVCompressor
        Inner scorer producing (B, H_kv, S) importance scores. AdaKV and
        attention-weights-based scorers are not supported.
    alpha_safeguard : float, default=0.20
        Minimum fraction of KV pairs that each head must retain
        (``n_safe = int(n_kept * alpha_safeguard)``). Ensures no attention head
        is compressed too aggressively.

    Deviations from kvpress
    -----------------------
    None — ``compress`` is a verbatim transcription of ``AdaKVPress.compress``.
    """

    press: ScorerKVCompressor
    alpha_safeguard: float = 0.20

    def __post_init__(self):
        assert isinstance(self.press, ScorerKVCompressor), "AdaKVSketch requires a ScorerKVCompressor as input"
        assert 0 <= self.alpha_safeguard <= 1, "alpha_safeguard should be in [0, 1]"

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
        if self.compression_ratio == 0:
            return keys, values

        assert module.config._attn_implementation != "eager", "eager mode not supported"

        # Compute scores
        scores = self.press.score(module, hidden_states, keys, values, attentions, kwargs)
        bsz, num_key_value_heads, k_len = scores.shape

        # Make sure to keep at least alpha * (1 - compression_ratio) KV pairs per head
        n_kept = int(k_len * (1 - self.compression_ratio))  # ScorerKVCompressor definition
        n_safe = int(n_kept * self.alpha_safeguard)
        top_indices = torch.topk(scores, n_safe, dim=-1).indices
        scores.scatter_(-1, top_indices, torch.finfo(scores.dtype).max)

        # Compute bottom-k across heads
        n_pruned = num_key_value_heads * (k_len - n_kept)
        indices = torch.topk(-scores.reshape(bsz, -1), n_pruned, dim=1).indices.flatten()

        # Save indices to mask during the attention mechanism. Please refer to attention_patch.py for more details
        batch_indices = torch.arange(bsz).repeat_interleave(n_pruned)
        head_indices = indices // k_len
        seq_indices = indices % k_len
        module.masked_key_indices = (batch_indices, head_indices, seq_indices)
        return keys, values
