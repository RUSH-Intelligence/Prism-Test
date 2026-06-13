"""Transition shim — the canonical score-based compressor now lives in
``eval_harness.kv_compression.base`` as ``ScorerKVCompressor``.

Re-exported here under the legacy ``ScorerSketch`` name for the migration;
removed in the folder rename (step 7).
"""

from eval_harness.kv_compression.base import ScorerKVCompressor as ScorerSketch  # noqa: F401

__all__ = ["ScorerSketch"]
