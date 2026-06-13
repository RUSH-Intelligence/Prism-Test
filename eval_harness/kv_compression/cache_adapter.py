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
    return hasattr(layer, "keys") and hasattr(layer, "values")


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

    layer_types = getattr(decoder_cfg, "layer_types", None) or []
    lowered = [str(t).lower() for t in layer_types]
    has_linear = any("linear" in t for t in lowered)
    has_full_or_sliding = any(("attention" in t or "sliding" in t) for t in lowered)
    return has_linear and has_full_or_sliding


def create_cache_adapter(model: PreTrainedModel) -> CacheAdapter:
    if _is_hybrid_model(model):
        return HybridCacheAdapter(model=model)
    return StandardCacheAdapter(model=model)
