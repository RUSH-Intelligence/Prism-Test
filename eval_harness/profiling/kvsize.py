"""KV-cache accounting for the performance benchmark.

Two correctness rules, both learned from cache shapes that actually occur here:

1. **Sum the ACTUAL per-layer lengths.**  ``seq_len x n_layers x bytes_per_token``
   overstates a ragged cache (PyramidKV keeps ~99.7% at layer 0 and ~60% at the
   deepest layer at r=0.2), which would make the method look like the worst row
   on the memory table when it is not.
2. **Skip non-attention slots.**  In the unified transformers cache every
   per-layer slot exposes ``keys``/``values``, including the mamba/mlp slots of a
   hybrid model (NemotronH), where they stay ``None``.  This module reuses
   ``cache_adapter``'s own predicates rather than re-deriving them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List, Optional

from eval_harness.kv_compression.cache_adapter import (
    _can_slice_attention_kv,
    _layer_seq_length,
)


@dataclass
class KVAccounting:
    bytes_total: int = 0
    layers_total: int = 0
    layers_with_kv: int = 0
    per_layer_seq_len: List[int] = field(default_factory=list)
    seq_len_max: int = 0
    seq_len_min: int = 0
    ragged: bool = False
    kv_dtype: Optional[str] = None
    bytes_per_token: Optional[int] = None
    quantized_layers: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def kv_cache_accounting(cache) -> KVAccounting:
    """Walk a cache and measure it. Never raises on hybrid/empty caches."""
    acc = KVAccounting()
    layers = getattr(cache, "layers", None)
    if layers is None:
        return acc
    acc.layers_total = len(layers)
    for layer in layers:
        if getattr(layer, "_quantized_keys", None) is not None:
            acc.quantized_layers += 1
        if not _can_slice_attention_kv(layer):
            continue
        keys, values = layer.keys, layer.values
        acc.layers_with_kv += 1
        acc.bytes_total += keys.numel() * keys.element_size()
        acc.bytes_total += values.numel() * values.element_size()
        seq = _layer_seq_length(layer)
        acc.per_layer_seq_len.append(int(seq) if seq is not None else 0)
        if acc.kv_dtype is None:
            acc.kv_dtype = str(keys.dtype)
            # bytes/token for ONE layer: 2 (k+v) * heads * head_dim * itemsize
            acc.bytes_per_token = int(
                2 * keys.shape[1] * keys.shape[3] * keys.element_size()
            )
    if acc.per_layer_seq_len:
        acc.seq_len_max = max(acc.per_layer_seq_len)
        acc.seq_len_min = min(acc.per_layer_seq_len)
        acc.ragged = acc.seq_len_max != acc.seq_len_min
    return acc


def analytic_kv_bytes(seq_len: int, n_layers: int, n_kv_heads: int, head_dim: int, itemsize: int) -> int:
    """Closed form, for cross-checking the measured walk."""
    return 2 * n_layers * n_kv_heads * head_dim * itemsize * seq_len
