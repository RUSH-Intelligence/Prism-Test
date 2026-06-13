import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator, Optional, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from eval_harness.kv_compression.base import KVCompressor
from eval_harness.kv_compression.registry import get_kv_compressor, register_kv_compressor
from eval_harness.kv_compression.base import ScorerKVCompressor
from eval_harness.kv_compression.utils import extract_keys_and_values

logger = logging.getLogger(__name__)


@register_kv_compressor("dms")
@dataclass
class DMSSketch(KVCompressor):
    """
    Dynamic Memory Sparsification (DMS, https://arxiv.org/abs/2506.05345) inference,
    ported from kvpress 0.5.1 ``DMSPress`` (kvpress/presses/dms_press.py).

    Wraps a ScorerKVCompressor and evicts keys/values with scores below a given threshold.
    This sketch implements a dense-prefill version of DMS, not the sparse-prefill version.

    Unlike most sketches that use a fixed compression_ratio, DMSSketch uses a score
    threshold to determine which KV pairs to evict. This allows for adaptive compression
    where the actual compression ratio depends on the input content.

    Importantly, this sketch can be used both during prefilling and during decoding
    (if decoding=True).

    A sliding window protects the most recent tokens from eviction, ensuring that
    recently generated tokens are always available for attention.

    Eviction is virtual: the cache is never physically shrunk. Evicted (batch, head,
    token) indices are recorded on ``module.masked_key_indices`` and the globally
    patched attention interface (``eval_harness/kv_compression/attention_patch.py``, applied
    at ``import eval_harness.kv_compression``) overwrites those cache rows with fake keys k
    such that exp(<q, k>) == 0 for every query of subsequent forwards. The same patch
    resets ``masked_key_indices`` on every full prefill forward (q_len == k_len), so
    prefill attention stays exact. ``compression_ratio`` is therefore a read-only
    property (per-layer masked fraction averaged across layers, available only after
    a forward pass) instead of a ScorerKVCompressor field.

    Parameters
    ----------
    press : ScorerKVCompressor or str
        The underlying scorer sketch used to compute importance scores for each token.
        A sketch-registry name (e.g. ``"knorm"``, ``"random"``) is resolved via
        ``get_kv_compressor(press)``. The scorer is always called with ``attentions=None``
        and only the newest ``q_len`` keys/values, so attention-weight-based scorers
        cannot be wrapped.
    threshold : float
        Tokens with scores below this threshold are evicted. The optimal threshold
        depends on the scorer sketch being used. Must be provided.
    sliding_window_size : int, default=128
        Number of recent tokens protected from eviction.
    decoding : bool, default=False
        If True, compression is also applied during the decoding phase (token
        generation). If False, compression only occurs during prefill.

    Deviations from kvpress
    -----------------------
    - ``press`` additionally accepts a sketch-registry name (kvpress requires a
      ScorerPress instance); this keeps DMS constructible from flat ``kv_compressor_kwargs``
      config.
    - ``threshold`` is validated at construction (kvpress defaults to None and only
      fails with a TypeError once the score buffer first exceeds the window).
    - Prefill/decode detection keeps kvpress's ``cache_position[-1] + 1 == q_len``
      test but falls back to ``cache.get_seq_length(layer_idx)`` when
      ``cache_position`` is absent (parity with ``KVCompressor._is_decoding_step``).
    - ``__call__`` fails fast on configurations where the recorded indices would
      silently never be applied or would be misaligned: ``attn_implementation ==
      'eager'`` (the attention patch only wraps ``ALL_ATTENTION_FUNCTIONS`` entries;
      eager dispatches around it) and prefill methods that replace
      ``self_attn.forward`` (dca, reattention_exact) or register cache-pruning
      forward hooks (reattention). Use ``attention_method='none'``.
    - ``decoding=True`` requires the pipeline to keep sketch hooks installed during
      the question/decode phase. kvpress's pipeline re-registers DMS hooks for the
      decode phase (kvpress pipeline.py:227-229); Prism's
      ``ResearchGenerationPipeline`` does not yet have that gate, so a warning is
      emitted at construction — without the gate, decode-time eviction does not run
      in pipeline evals. kvpress's latent multi-question hazard (decode-time masked
      indices outliving a cache restore) is inherited unchanged.
    """

    press: Union[ScorerKVCompressor, str]
    threshold: Optional[float] = None
    sliding_window_size: int = 128
    decoding: bool = False
    scores_buffer: dict[int, torch.Tensor] = field(default_factory=dict, init=False, repr=False)
    compression_ratios: dict[int, float] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        if isinstance(self.press, str):
            self.press = get_kv_compressor(self.press)
        assert isinstance(self.press, ScorerKVCompressor), "DMSSketch requires a ScorerKVCompressor as press"
        assert self.threshold is not None, (
            "DMSSketch requires an explicit threshold (kvpress defaults to None and fails "
            "with a TypeError at the first eviction)"
        )
        if self.decoding:
            logger.warning(
                "DMSSketch(decoding=True): ResearchGenerationPipeline does not re-register "
                "sketch hooks for the decode phase (the kvpress DMSPress pipeline gate is not "
                "ported); decode-time eviction only happens while the hooks remain installed."
            )

    def post_init_from_model(self, model):
        self.press.post_init_from_model(model)

    @property
    def compression_ratio(self):
        """Average compression ratio across all layers (computed after forward pass)."""
        assert len(self.compression_ratios) > 0, "Forward pass must be run to compute the compression ratio"
        return sum(self.compression_ratios.values()) / len(self.compression_ratios)

    @compression_ratio.setter
    def compression_ratio(self, value):
        """Compression ratio is read-only since it depends on threshold and input content."""
        raise AttributeError(f"compression ratio cannot be set for {type(self).__name__}")

    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        hidden_states = kwargs["hidden_states"]
        cache = kwargs["past_key_values"]
        q_len = hidden_states.shape[1]
        cache_position = kwargs.get("cache_position")
        if cache_position is not None:
            cache_len = int(cache_position[-1]) + 1
        else:
            cache_len = int(cache.get_seq_length(module.layer_idx))
        prefilling = cache_len == q_len

        layer_idx: int = module.layer_idx

        if prefilling and (layer_idx == 0):
            self.scores_buffer.clear()
            self.compression_ratios.clear()

        if not prefilling and not self.decoding:
            return output

        keys, values = extract_keys_and_values(cache, layer_idx)
        scores = self.press.score(module, hidden_states, keys[:, :, -q_len:], values[:, :, -q_len:], None, kwargs)

        if prefilling:
            self.scores_buffer[layer_idx] = scores
        else:
            self.scores_buffer[layer_idx] = torch.cat([self.scores_buffer[layer_idx], scores], dim=-1)

        if self.scores_buffer[layer_idx].shape[-1] > self.sliding_window_size:
            n_to_evict = self.scores_buffer[layer_idx].shape[-1] - self.sliding_window_size
            scores_to_evict = self.scores_buffer[layer_idx][..., :n_to_evict]
            self.scores_buffer[layer_idx] = self.scores_buffer[layer_idx][..., n_to_evict:]

            new_masked_key_indices = list(torch.where(scores_to_evict < self.threshold))

            if len(new_masked_key_indices[0]) > 0:
                # Buffer-relative -> cache-absolute token indices (shift == 0 during prefill).
                shift = cache_len - scores_to_evict.shape[2] - self.sliding_window_size
                new_masked_key_indices[-1] += shift

                if module.masked_key_indices is None:
                    module.masked_key_indices = new_masked_key_indices
                else:
                    module.masked_key_indices = list(
                        torch.cat([i, new_i]) for i, new_i in zip(module.masked_key_indices, new_masked_key_indices)
                    )

        if module.masked_key_indices is not None:
            bsz, num_key_value_heads, cache_len, _ = keys.shape
            n_masked = len(module.masked_key_indices[0])
            self.compression_ratios[layer_idx] = n_masked / (bsz * num_key_value_heads * cache_len)
        else:
            self.compression_ratios[layer_idx] = 0

        return output

    @staticmethod
    def _validate_model(model) -> None:
        impl = getattr(getattr(model, "config", None), "_attn_implementation", None)
        if impl == "eager":
            raise ValueError(
                "DMSSketch requires a patched attention interface: attn_implementation='eager' "
                "bypasses ALL_ATTENTION_FUNCTIONS, so masked_key_indices would be recorded but "
                "never applied (silent no-op). Load the model with sdpa or flash attention."
            )
        inner = getattr(model, "model", None)
        if inner is None:
            return
        language_model = inner.language_model if hasattr(inner, "language_model") else inner
        for layer in getattr(language_model, "layers", []):
            attn = getattr(layer, "self_attn", None)
            if attn is None:
                continue
            if "forward" in vars(attn):
                raise ValueError(
                    "DMSSketch is incompatible with prefill methods that replace self_attn.forward "
                    "(e.g. 'dca', 'reattention_exact'); use attention_method='none'."
                )
            if getattr(attn, "_forward_hooks", None):
                raise ValueError(
                    "DMSSketch is incompatible with prefill methods that register cache-pruning "
                    "forward hooks (e.g. 'reattention'); use attention_method='none'."
                )

    @contextmanager
    def __call__(self, model: PreTrainedModel) -> Generator:
        self._validate_model(model)
        with super().__call__(model):
            yield
