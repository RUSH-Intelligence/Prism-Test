from dataclasses import dataclass

import torch
from torch import nn

from eval_harness.kv_compression.registry import register_kv_compressor
from eval_harness.kv_compression.base import ScorerKVCompressor


@register_kv_compressor("streaming_llm")
@dataclass
class StreamingLLMSketch(ScorerKVCompressor):
    """
    StreamingLLM: Window-based KV cache compression with sink tokens.

    Implements sliding window approach preserving first few tokens (sink tokens)
    and most recent tokens, while pruning middle tokens.

    Based on StreamingLLM (https://arxiv.org/abs/2309.17453).
    Direct port of kvpress 0.5.1 ``StreamingLLMPress`` (presses/streaming_llm_press.py).
    As in kvpress, fully matching the paper would additionally require the
    key-rerotation wrapper (ported as the ``key_rerotation`` sketch, mirroring
    kvpress ``KeyRerotationPress``), which is not composed here by default;
    kept keys deliberately stay at their original RoPE phases.

    Parameters
    ----------
    compression_ratio : float, default=0.0
        Fraction of key-value pairs to remove during compression.
    n_sink : int, default=4
        Number of initial tokens to always preserve (sink tokens).
        These tokens are never pruned and serve as "attention sinks" that help
        maintain model stability. Language models often assign high attention
        weights to early tokens regardless of semantic content.

    Replicated kvpress quirks (intentional, do not "fix"):
    - ``compression_ratio > 0`` with ``k_len <= n_sink`` raises AssertionError;
      at ratio 0 the inherited ``compress`` short-circuits before ``score``.
    - When ``n_kept < n_sink`` (very high ratio) the recency window vanishes and
      ``topk`` keeps an unspecified subset of the sink tokens; ``n_kept == 0``
      empties the layer cache and breaks decode, exactly as upstream.
    - ``topk`` over the binary scores leaves the kept-token order unspecified.
    """

    compression_ratio: float = 0.0
    n_sink: int = 4

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:

        k_len = keys.shape[2]
        assert k_len > self.n_sink, f"Input should contain more tokens than n_sink={self.n_sink}"
        n_pruned = k_len - int(k_len * (1 - self.compression_ratio))
        scores = torch.ones_like(keys[..., 0])
        scores[:, :, self.n_sink : self.n_sink + n_pruned] = 0

        return scores
