from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

from transformers import DynamicCache, PreTrainedModel


CacheCheckpoint = Dict[int, int]


def _layer_seq_length(layer) -> Optional[int]:
    if hasattr(layer, "get_seq_length"):
        try:
            return int(layer.get_seq_length())
        except Exception:
            pass
    keys = getattr(layer, "keys", None)
    if keys is not None and hasattr(keys, "shape") and len(keys.shape) >= 3:
        return int(keys.shape[2])
    return None


def _can_slice_attention_kv(layer) -> bool:
    # A sliceable attention layer must hold real K/V tensors. In the unified
    # transformers cache, EVERY per-layer slot is a CacheLayerMixin exposing
    # ``keys``/``values`` attributes — including the mamba/mlp slots of a hybrid
    # model (NemotronH), where they stay ``None`` because no attention layer
    # populates them. ``hasattr`` alone would wrongly include those stateful
    # (non-K/V) slots and then crash checkpoint/restore on ``None[:, :, :n]``.
    keys = getattr(layer, "keys", None)
    values = getattr(layer, "values", None)
    return (
        keys is not None
        and values is not None
        and hasattr(keys, "shape")
        and len(keys.shape) >= 3
    )


class CacheAdapter:
    def initialize_cache(self, cache):
        raise NotImplementedError

    def get_seq_length(self, cache) -> int:
        raise NotImplementedError

    def clone_or_checkpoint_for_multi_question(self, cache) -> CacheCheckpoint:
        raise NotImplementedError

    def restore_after_question(self, cache, checkpoint: CacheCheckpoint) -> None:
        raise NotImplementedError

    def maybe_slice_prefill(self, cache) -> None:
        # Hook for future cache-family-specific prefill slicing policies.
        del cache


@dataclass
class StandardCacheAdapter(CacheAdapter):
    model: PreTrainedModel

    def initialize_cache(self, cache):
        return cache if cache is not None else DynamicCache()

    def get_seq_length(self, cache) -> int:
        return int(cache.get_seq_length())

    def clone_or_checkpoint_for_multi_question(self, cache) -> CacheCheckpoint:
        return {
            layer_idx: seq_len
            for layer_idx in range(len(cache))
            if (seq_len := _layer_seq_length(cache.layers[layer_idx])) is not None
        }

    def restore_after_question(self, cache, checkpoint: CacheCheckpoint) -> None:
        for layer_idx, sequence_length in checkpoint.items():
            layer = cache.layers[layer_idx]
            if not _can_slice_attention_kv(layer):
                continue
            layer.keys = layer.keys[:, :, :sequence_length]
            layer.values = layer.values[:, :, :sequence_length]

            if hasattr(layer, "_quantized_keys") and layer._quantized_keys is not None:
                layer._quantized_keys = layer._quantized_keys[:, :, :sequence_length]
            if hasattr(layer, "_quantized_values") and layer._quantized_values is not None:
                layer._quantized_values = layer._quantized_values[:, :, :sequence_length]


@dataclass
class HybridCacheAdapter(CacheAdapter):
    model: PreTrainedModel

    def initialize_cache(self, cache):
        if cache is None:
            return DynamicCache(config=self.model.config)

        layers = getattr(cache, "layers", None)
        if isinstance(layers, list) and len(layers) == 0:
            # A config-less DynamicCache cannot represent linear-attention layers.
            return DynamicCache(config=self.model.config)

        return cache

    def _iter_attention_layer_indices(self, cache) -> Iterable[int]:
        for idx, layer in enumerate(cache.layers):
            if _can_slice_attention_kv(layer):
                yield idx

    def get_seq_length(self, cache) -> int:
        lengths = [
            seq_len
            for idx in self._iter_attention_layer_indices(cache)
            if (seq_len := _layer_seq_length(cache.layers[idx])) is not None
        ]
        return max(lengths) if lengths else 0

    def clone_or_checkpoint_for_multi_question(self, cache) -> CacheCheckpoint:
        checkpoint: CacheCheckpoint = {}
        for layer_idx in self._iter_attention_layer_indices(cache):
            seq_len = _layer_seq_length(cache.layers[layer_idx])
            if seq_len is not None:
                checkpoint[layer_idx] = seq_len
        return checkpoint

    def restore_after_question(self, cache, checkpoint: CacheCheckpoint) -> None:
        for layer_idx, sequence_length in checkpoint.items():
            layer = cache.layers[layer_idx]
            if not _can_slice_attention_kv(layer):
                continue
            layer.keys = layer.keys[:, :, :sequence_length]
            layer.values = layer.values[:, :, :sequence_length]

            if hasattr(layer, "_quantized_keys") and layer._quantized_keys is not None:
                layer._quantized_keys = layer._quantized_keys[:, :, :sequence_length]
            if hasattr(layer, "_quantized_values") and layer._quantized_values is not None:
                layer._quantized_values = layer._quantized_values[:, :, :sequence_length]


def _is_hybrid_model(model: PreTrainedModel) -> bool:
    cfg = getattr(model, "config", None)
    if cfg is None:
        return False

    try:
        decoder_cfg = cfg.get_text_config(decoder=True)
    except Exception:
        decoder_cfg = cfg

    # Per-layer type list: ``layer_types`` (Qwen3.5/Gemma3) or, for NemotronH,
    # ``layers_block_type`` (native transformers also aliases this onto
    # ``layer_types``). Values look like "full_attention"/"linear_attention"
    # (Qwen3.5) or "attention"/"mamba"/"mlp" (NemotronH Mamba-attention hybrid).
    layer_types = (
        getattr(decoder_cfg, "layer_types", None)
        or getattr(decoder_cfg, "layers_block_type", None)
        or []
    )
    lowered = [str(t).lower() for t in layer_types]
    # "Hybrid" here means some layers hold NO standard K/V cache (linear-attention
    # or mamba), so checkpoint/restore must skip them. Sliding attention still
    # has a K/V cache and therefore does NOT, by itself, make a model hybrid.
    has_stateful = any(("linear" in t or "mamba" in t) for t in lowered)
    has_attention = any("attention" in t for t in lowered)
    if has_stateful and has_attention:
        return True

    # NemotronH configs may carry only the raw pattern string (M=mamba,
    # *=attention, -=mlp) without an expanded per-layer list.
    pattern = getattr(decoder_cfg, "hybrid_override_pattern", None)
    if isinstance(pattern, str) and "M" in pattern and "*" in pattern:
        return True

    return False


def create_cache_adapter(model: PreTrainedModel) -> CacheAdapter:
    if _is_hybrid_model(model):
        return HybridCacheAdapter(model=model)
    return StandardCacheAdapter(model=model)
