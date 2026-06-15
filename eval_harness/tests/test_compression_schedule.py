"""Regression test for the ``compression_schedule`` coercion contract.

``KVCompressor.__post_init__`` (kv_compression/base.py) coerces a user-supplied
``schedule`` (a string or list from YAML) into a ``frozenset`` of
``CompressionSchedule`` members.  Subclasses that override ``__post_init__`` must
chain ``super().__post_init__()`` or the raw value is never coerced and the first
``fires_on_prefill``/``fires_on_decode`` access crashes with
``TypeError: unsupported operand type(s) for &: 'str' and 'set'``.

This guards every compressor that overrides ``__post_init__`` against that
regression by constructing each with ``schedule`` as both a ``str`` and a
``list``.  No model loading.
"""

from __future__ import annotations

import unittest

from eval_harness.kv_compression.base import CompressionSchedule
from eval_harness.kv_compression.compressors.adakv_sketch import AdaKVSketch
from eval_harness.kv_compression.compressors.block_sketch import BlockSketch
from eval_harness.kv_compression.compressors.chunk_sketch import ChunkSketch
from eval_harness.kv_compression.compressors.chunkkv_sketch import ChunkKVSketch
from eval_harness.kv_compression.compressors.composed_sketch import ComposedSketch
from eval_harness.kv_compression.compressors.criticalkv_sketch import CriticalAdaKVSketch
from eval_harness.kv_compression.compressors.decoding_sketch import DecodingSketch
from eval_harness.kv_compression.compressors.dms_sketch import DMSSketch
from eval_harness.kv_compression.compressors.fastkvzip_sketch import FastKVzipSketch
from eval_harness.kv_compression.compressors.finch_sketch import FinchSketch
from eval_harness.kv_compression.compressors.key_rerotation_sketch import KeyRerotationSketch
from eval_harness.kv_compression.compressors.knorm_sketch import KnormSketch
from eval_harness.kv_compression.compressors.kvzip_sketch import KVzipSketch
from eval_harness.kv_compression.compressors.per_layer_compression_sketch import (
    PerLayerCompressionSketch,
)
from eval_harness.kv_compression.compressors.ridge_sketch import RidgeSketch
from eval_harness.kv_compression.compressors.simlayerkv_sketch import SimLayerKVSketch
from eval_harness.kv_compression.compressors.snapkv_sketch import SnapKVSketch
from eval_harness.kv_compression.compressors.think_sketch import ThinKSketch


def _scorer():
    return KnormSketch(compression_ratio=0.5)


# label -> (class, factory for the minimal valid non-schedule kwargs).  Covers
# every compressor that overrides __post_init__: the 15 that previously skipped
# super().__post_init__() plus snapkv/finch as already-correct positive controls.
CASES = {
    "adakv": (AdaKVSketch, lambda: {"press": _scorer()}),
    "block": (BlockSketch, lambda: {"sketch": _scorer()}),
    "chunk": (ChunkSketch, lambda: {"press": _scorer()}),
    "chunkkv": (ChunkKVSketch, lambda: {}),
    "composed": (ComposedSketch, lambda: {"presses": ["knorm"]}),
    "criticalkv": (CriticalAdaKVSketch, lambda: {"press": _scorer()}),
    "decoding": (DecodingSketch, lambda: {"base_sketch": _scorer()}),
    "dms": (DMSSketch, lambda: {"press": _scorer(), "threshold": 0.0}),
    "fastkvzip": (FastKVzipSketch, lambda: {}),
    "key_rerotation": (KeyRerotationSketch, lambda: {"press": _scorer()}),
    "kvzip": (KVzipSketch, lambda: {}),
    "per_layer_compression": (
        PerLayerCompressionSketch,
        lambda: {"press": _scorer(), "compression_ratios": [0.5]},
    ),
    "ridge": (RidgeSketch, lambda: {}),
    "simlayerkv": (SimLayerKVSketch, lambda: {}),
    "think": (ThinKSketch, lambda: {}),
    # positive controls — already chained super() before this fix.
    "snapkv": (SnapKVSketch, lambda: {}),
    "finch": (FinchSketch, lambda: {}),
}


class TestCompressionScheduleCoercion(unittest.TestCase):
    def test_string_schedule_is_coerced(self):
        for label, (cls, kwargs) in CASES.items():
            with self.subTest(compressor=label):
                obj = cls(**kwargs(), schedule="decode")
                self.assertIsInstance(obj.schedule, frozenset)
                self.assertEqual(obj.schedule, frozenset({CompressionSchedule.DECODE}))
                # The properties below are exactly what crashed on the raw str.
                self.assertTrue(obj.fires_on_decode)
                self.assertFalse(obj.fires_on_prefill)

    def test_list_schedule_is_coerced(self):
        for label, (cls, kwargs) in CASES.items():
            with self.subTest(compressor=label):
                obj = cls(**kwargs(), schedule=["decode", "streaming"])
                self.assertIsInstance(obj.schedule, frozenset)
                self.assertEqual(
                    obj.schedule,
                    frozenset({CompressionSchedule.DECODE, CompressionSchedule.STREAMING}),
                )
                self.assertTrue(obj.fires_on_decode)
                self.assertTrue(obj.fires_on_prefill)

    def test_default_schedule_is_post_prefill(self):
        for label, (cls, kwargs) in CASES.items():
            with self.subTest(compressor=label):
                obj = cls(**kwargs())
                self.assertEqual(
                    obj.schedule, frozenset({CompressionSchedule.POST_PREFILL})
                )
                self.assertTrue(obj.fires_on_prefill)
                self.assertFalse(obj.fires_on_decode)


if __name__ == "__main__":
    unittest.main()
