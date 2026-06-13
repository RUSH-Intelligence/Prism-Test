"""Transition shim — the canonical Door-3 base now lives in
``eval_harness.kv_compression.base``.

During the three-door migration this module re-exports the renamed
:class:`~eval_harness.kv_compression.base.KVCompressor` under the legacy
``BaseSketch`` name (plus the shared layer-detection helpers) so existing
import paths keep working.  It is deleted in the folder rename (step 7); new
code should import from ``eval_harness.kv_compression``.
"""

from eval_harness.kv_compression.base import (  # noqa: F401
    SUPPORTED_MODELS,
    KVCompressor as BaseSketch,
    _is_gemma3,
    _is_non_full_attention_layer,
)

__all__ = [
    "BaseSketch",
    "SUPPORTED_MODELS",
    "_is_gemma3",
    "_is_non_full_attention_layer",
]
