import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator

import torch
from torch import nn
from transformers import PreTrainedModel, QuantizedCache

try:
    from transformers import Gemma3ForConditionalGeneration
except ImportError:
    Gemma3ForConditionalGeneration = None  # type: ignore[assignment]

try:
    from transformers import Gemma3ForCausalLM
except ImportError:
    Gemma3ForCausalLM = None  # type: ignore[assignment]

try:
    from transformers import LlamaForCausalLM
except ImportError:
    LlamaForCausalLM = None  # type: ignore[assignment]

try:
    from transformers import MistralForCausalLM
except ImportError:
    MistralForCausalLM = None  # type: ignore[assignment]

try:
    from transformers import Mistral3ForConditionalGeneration
except ImportError:
    Mistral3ForConditionalGeneration = None  # type: ignore[assignment]

try:
    from transformers import Phi3ForCausalLM
except ImportError:
    Phi3ForCausalLM = None  # type: ignore[assignment]

try:
    from transformers import Qwen2ForCausalLM
except ImportError:
    Qwen2ForCausalLM = None  # type: ignore[assignment]

try:
    from transformers import Qwen3ForCausalLM
except ImportError:
    Qwen3ForCausalLM = None  # type: ignore[assignment]

try:
    from transformers import Qwen3_5ForCausalLM
except ImportError:
    Qwen3_5ForCausalLM = None  # type: ignore[assignment]

from eval_harness.sketch.utils import extract_keys_and_values

logger = logging.getLogger(__name__)

SUPPORTED_MODELS = tuple(
    m
    for m in (
        LlamaForCausalLM,
        MistralForCausalLM,
        Mistral3ForConditionalGeneration,
        Phi3ForCausalLM,
        Qwen2ForCausalLM,
        Qwen3ForCausalLM,
        Qwen3_5ForCausalLM,
        Gemma3ForCausalLM,
        Gemma3ForConditionalGeneration,
    )
    if m is not None
)


def _is_non_full_attention_layer(layer: nn.Module) -> bool:
    """Best-effort detection of non-full attention layers.

    For mixed-attention families (Gemma3, Qwen3.5), we should only hook full-softmax layers.
    This helper flags sliding/linear (or any non-full typed layer) to skip compression hooks.
    """
    attn = getattr(layer, "self_attn", None)
    if attn is None:
        return False

    # Some architectures expose per-layer type directly on the decoder layer.
    for attr in ("layer_type", "attention_type"):
        val = getattr(layer, attr, None)
        if isinstance(val, str):
            lowered = val.lower()
            if "full" in lowered:
                return False
            if any(token in lowered for token in ("sliding", "linear")):
                return True
            return True

    # Common explicit flags in some implementations.
    is_sliding = getattr(attn, "is_sliding", None)
    if is_sliding is not None:
        return bool(is_sliding)
    is_linear = getattr(attn, "is_linear", None)
    if is_linear is not None:
        return bool(is_linear)

    # Common config fields used by mixed-attention families.
    cfg = getattr(attn, "config", None)
    for attr in ("layer_type", "attention_type"):
        val = getattr(cfg, attr, None) if cfg is not None else None
        if isinstance(val, str):
            lowered = val.lower()
            if "full" in lowered:
                return False
            if any(token in lowered for token in ("sliding", "linear")):
                return True
            # Conservative fallback: if attention type exists and is not explicit full, skip it.
            return True

    sw = getattr(cfg, "sliding_window", None) if cfg is not None else None
    if isinstance(sw, int):
        return sw > 0

    return False


@dataclass
class BaseSketch:
    def post_init_from_model(self, model: PreTrainedModel):
        pass

    @staticmethod
    def _is_decoding_step(module: nn.Module, kwargs: dict, q_len: int) -> bool:
        """Detect decoding vs prefill across transformers versions."""
        cache_position = kwargs.get("cache_position")
        if cache_position is not None:
            return cache_position[-1] > q_len

        cache = kwargs.get("past_key_values")
        if cache is not None:
            try:
                return cache.get_seq_length(module.layer_idx) > q_len
            except Exception:
                pass

        # Conservative fallback when cache metadata is unavailable.
        return q_len <= 1

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("compress method must be implemented in subclass")

    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        hidden_states = kwargs["hidden_states"]
        cache = kwargs["past_key_values"]
        cache_layer = cache.layers[module.layer_idx]
        q_len = hidden_states.shape[1]

        if self._is_decoding_step(module, kwargs, q_len):
            return output

        keys, values = extract_keys_and_values(cache, module.layer_idx)
        keys, values = self.compress(module, hidden_states, keys, values, output[1], kwargs)

        if isinstance(cache, QuantizedCache):
            cache_layer._quantized_keys = cache_layer._quantize(keys, axis=cache_layer.axis_key)
            cache_layer._quantized_values = cache_layer._quantize(values, axis=cache_layer.axis_value)
            cache_layer.keys = torch.zeros(0, dtype=keys.dtype, device=keys.device)
            cache_layer.values = torch.zeros(0, dtype=keys.dtype, device=keys.device)
            cache_layer.cumulative_length = keys.shape[2]
        else:
            cache_layer.keys = keys
            cache_layer.values = values

        return output

    @contextmanager
    def __call__(self, model: PreTrainedModel) -> Generator:
        if not isinstance(model, SUPPORTED_MODELS):
            logger.warning(f"Model {type(model)} not tested, supported models: {SUPPORTED_MODELS}")

        is_gemma3_family = (
            (Gemma3ForConditionalGeneration is not None and isinstance(model, Gemma3ForConditionalGeneration))
            or (Gemma3ForCausalLM is not None and isinstance(model, Gemma3ForCausalLM))
        )
        if is_gemma3_family or (
            Qwen3_5ForCausalLM is not None and isinstance(model, Qwen3_5ForCausalLM)
        ):
            logger.warning(
                "Compression is only applied to full-softmax attention layers for this model family"
            )

        self.post_init_from_model(model)
        hooks = []
        try:
            language_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
            for layer in language_model.layers:
                if is_gemma3_family and getattr(layer.self_attn, "is_sliding", False):
                    # Keep behavior aligned with base_press: skip Gemma3 sliding-window layers.
                    continue
                if _is_non_full_attention_layer(layer):
                    continue
                layer.self_attn.rotary_emb = language_model.rotary_emb
                hooks.append(layer.self_attn.register_forward_hook(self.forward_hook, with_kwargs=True))
            yield
        finally:
            for forward_hook in hooks:
                forward_hook.remove()
