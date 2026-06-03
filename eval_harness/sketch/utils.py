import torch
from transformers import Cache, QuantizedCache


def dequantize_layer(cache_layer) -> tuple[torch.Tensor, torch.Tensor]:
    keys = cache_layer._dequantize(cache_layer._quantized_keys)
    values = cache_layer._dequantize(cache_layer._quantized_values)
    return keys, values


def extract_keys_and_values(cache: Cache, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(cache, QuantizedCache):
        keys, values = dequantize_layer(cache.layers[layer_idx])
    else:
        keys = cache.layers[layer_idx].keys
        values = cache.layers[layer_idx].values
    return keys, values
