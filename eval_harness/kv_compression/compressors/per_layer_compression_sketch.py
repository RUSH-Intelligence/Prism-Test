import inspect
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
from torch import nn
from transformers import PreTrainedModel

from eval_harness.kv_compression.base import KVCompressor
from eval_harness.kv_compression.registry import get_kv_compressor, register_kv_compressor
from eval_harness.kv_compression.base import ScorerKVCompressor

logger = logging.getLogger(__name__)


@register_kv_compressor("per_layer_compression")
@dataclass
class PerLayerCompressionSketch(KVCompressor):
    """
    Per-layer compression: Apply different compression ratios to different layers.

    Wrapper that applies layer-specific compression ratios using any underlying
    ScorerKVCompressor method. Different layers may have different importance patterns,
    so layer-specific compression can improve quality-efficiency trade-offs.

    **Important**: Experimental feature that only works with flash attention.

    Port of kvpress ``PerLayerCompressionPress``
    (``kvpress/presses/per_layer_compression_press.py``). The wrapper computes no
    scores itself: per hooked layer it temporarily overrides the inner sketch's
    ``compression_ratio`` with ``compression_ratios[module.layer_idx]`` and
    delegates to the inner ``forward_hook`` (prefill gate, scoring, top-k gather,
    and cache write-back all run inside the delegate).

    Parameters
    ----------
    press : ScorerKVCompressor or str
        The underlying scoring method to apply with layer-specific compression
        ratios. Must be (or resolve to) a ``ScorerKVCompressor`` exposing
        ``compression_ratio`` in its ``__init__`` signature (both asserted, as in
        kvpress). A string is resolved through the sketch registry (see
        Deviations).
    compression_ratios : List[float]
        Per-layer fraction of tokens to remove, indexed by the global
        ``module.layer_idx``. Length must cover ``config.num_hidden_layers``;
        entries for layers the pipeline does not hook (sliding/linear layers of
        mixed-attention families) are simply unused.
    press_kwargs : dict, optional
        Constructor kwargs for the inner sketch when ``press`` is a registry
        name; passing it together with a sketch instance raises ``ValueError``.

    Deviations from kvpress
    -----------------------
    - ``forward_hook`` restores the inner sketch's ``compression_ratio`` in a
      ``try/finally``; kvpress restores it unguarded, so an exception inside the
      inner hook would leave the inner press mutated.
    - ``press`` may be given as a registry name (with optional ``press_kwargs``)
      so the wrapper is constructible from flat YAML ``kv_compressor_kwargs``; kvpress
      only accepts an instance.
    - ``post_init_from_model`` (no kvpress analog) delegates to the inner sketch
      (in-tree composite convention, cf. ``DecodingSketch``), validates
      ``len(compression_ratios) >= config.num_hidden_layers`` (kvpress would
      ``IndexError`` mid-prefill), and raises — where kvpress only warns at
      construction — when unequal ratios would produce a cross-layer ragged
      cache and the model is not loaded with ``flash_attention_2``. This
      pipeline decodes through HF's standard forward where one causal mask is
      sized from layer 0 (runner default ``sdpa``): a layer retaining more than
      layer 0 hard-errors, a layer retaining less silently leaks causality on
      the multi-token question forward. Equal ratios keep the cache rectangular
      (identical ``int(S * (1 - r))`` at every layer) and are therefore allowed
      under any attention implementation. The raggedness check conservatively
      compares the first ``num_hidden_layers`` entries even though
      mixed-attention families only hook full-attention layers.
    - ``ResearchAdapter._build_kv_compressor`` does not inject the adapter-level
      ``cfg.compression_ratio`` (``compression_ratio`` here is a read-only
      property, not a dataclass field); all configuration flows through
      ``kv_compressor_kwargs``.

    Upstream quirks kept verbatim
    -----------------------------
    - ``compression_ratios`` entries are not range-checked: 1.0 is accepted (the
      per-layer attribute write bypasses ``ScorerKVCompressor.__post_init__``'s
      ``0 <= ratio < 1`` assert) and empties that layer; ``S = 1`` with ratio
      0.5 keeps ``int(0.5) = 0`` tokens.
    - The "experimental ... only works with flash attention" warning is logged
      at construction.
    - ``compression_ratio`` is the read-only mean of ``compression_ratios``;
      assigning it raises ``AttributeError``.
    """

    press: Union[ScorerKVCompressor, str]
    compression_ratios: List[float]
    press_kwargs: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        super().__post_init__()
        logger.warning(
            "Per layer compression wrapper is an experimental feature and only works with flash attention. "
            "Please make sure that the model uses flash attention."
        )
        if isinstance(self.press, str):
            self.press = get_kv_compressor(self.press, **(self.press_kwargs or {}))
        elif self.press_kwargs is not None:
            raise ValueError("press_kwargs is only used when press is a registry name, not a sketch instance")
        assert (
            "compression_ratio" in inspect.signature(self.press.__init__).parameters
        ), f"compression_ratio can't be set in the provided press: {self.press.__class__}"
        assert isinstance(self.press, ScorerKVCompressor), "PerLayerCompressionSketch requires a ScorerKVCompressor as input"

    def post_init_from_model(self, model: PreTrainedModel):
        self.press.post_init_from_model(model)

        config = getattr(model, "config", None)
        num_hidden_layers = getattr(config, "num_hidden_layers", None)
        if num_hidden_layers is None and hasattr(config, "get_text_config"):
            num_hidden_layers = getattr(config.get_text_config(), "num_hidden_layers", None)

        if num_hidden_layers is not None:
            if len(self.compression_ratios) < num_hidden_layers:
                raise ValueError(
                    f"compression_ratios has {len(self.compression_ratios)} entries but the model has "
                    f"{num_hidden_layers} layers; provide one ratio per layer (entries for skipped "
                    "non-full-attention layers are unused but must be present)."
                )
            used_ratios = self.compression_ratios[:num_hidden_layers]
        else:
            used_ratios = list(self.compression_ratios)

        if len(set(used_ratios)) > 1:
            attn_implementation = getattr(config, "_attn_implementation", None)
            if attn_implementation != "flash_attention_2":
                raise ValueError(
                    "PerLayerCompressionSketch with unequal compression_ratios produces a cross-layer "
                    "ragged KV cache, which is only decode-safe under flash_attention_2 (got "
                    f"attn_implementation={attn_implementation!r}): HF sizes the decode causal mask "
                    "from layer 0, so under sdpa/eager a longer layer shape-errors and a shorter "
                    "layer silently leaks causality. Load the model with "
                    "attn_implementation='flash_attention_2' or use equal ratios."
                )

    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        original_compression_ratio = self.press.compression_ratio
        self.press.compression_ratio = self.compression_ratios[module.layer_idx]
        try:
            output = self.press.forward_hook(module, input, kwargs, output)
        finally:
            self.press.compression_ratio = original_compression_ratio
        return output

    @property
    def compression_ratio(self):
        return sum(self.compression_ratios) / len(self.compression_ratios)

    @compression_ratio.setter
    def compression_ratio(self, value):
        raise AttributeError(f"compression ratio cannot be set for {type(self).__name__}")
