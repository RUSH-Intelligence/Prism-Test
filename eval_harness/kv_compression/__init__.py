"""Door 3 — KV compression (post-attention cache rewrite, schedule-gated)."""

from .attention_patch import patch_attention_functions
from .base import (
    CompressionOperation,
    CompressionSchedule,
    KVCompressor,
    ScorerKVCompressor,
)
from .compressors.decoding_sketch import DecodingSketch
from .compressors.knorm_sketch import KnormSketch
from .compressors.prefill_decoding_sketch import PrefillDecodingSketch
from .compressors.random_sketch import RandomSketch
from .compressors.reattention_sketch import ReAttentionSketch
from .registry import (
    available_kv_compressors,
    get_kv_compressor,
    get_kv_compressor_class,
    register_kv_compressor,
)

# Patch ALL_ATTENTION_FUNCTIONS so attention-weight scorers (adakv etc.) can
# capture probabilities — applied at ``import eval_harness.kv_compression`` (the
# legacy ``import eval_harness.sketch`` side effect).
patch_attention_functions()

__all__ = [
    "CompressionOperation",
    "CompressionSchedule",
    "KVCompressor",
    "ScorerKVCompressor",
    "KnormSketch",
    "ReAttentionSketch",
    "RandomSketch",
    "DecodingSketch",
    "PrefillDecodingSketch",
    "available_kv_compressors",
    "get_kv_compressor",
    "get_kv_compressor_class",
    "register_kv_compressor",
]
