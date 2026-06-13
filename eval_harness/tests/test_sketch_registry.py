from __future__ import annotations

import unittest

from eval_harness.research_adapter import CacheConfig, ResearchAdapter
from eval_harness.kv_compression import (
    DecodingSketch,
    KnormSketch,
    RandomSketch,
    ReAttentionSketch,
    available_kv_compressors,
    get_kv_compressor,
    get_kv_compressor_class,
)


class TestSketchRegistry(unittest.TestCase):
    def test_existing_sketches_registered_with_aliases(self):
        names = available_kv_compressors()
        for name in (
            "knorm", "knorm_sketch",
            "random", "random_sketch",
            "reattention", "reattention_sketch",
        ):
            self.assertIn(name, names)

    def test_get_kv_compressor_class_resolves_aliases(self):
        self.assertIs(get_kv_compressor_class("knorm"), KnormSketch)
        self.assertIs(get_kv_compressor_class("knorm_sketch"), KnormSketch)
        self.assertIs(get_kv_compressor_class("ReAttention"), ReAttentionSketch)

    def test_get_kv_compressor_instantiates_with_kwargs(self):
        sketch = get_kv_compressor("random", compression_ratio=0.25, seed=7)
        self.assertIsInstance(sketch, RandomSketch)
        self.assertAlmostEqual(sketch.compression_ratio, 0.25)
        self.assertEqual(sketch.seed, 7)

    def test_unknown_sketch_lists_available(self):
        with self.assertRaises(ValueError) as ctx:
            get_kv_compressor_class("definitely_not_a_sketch")
        self.assertIn("Available:", str(ctx.exception))
        self.assertIn("knorm", str(ctx.exception))


class TestBuildSketchViaRegistry(unittest.TestCase):
    def _shell(self, cfg: CacheConfig):
        adapter = object.__new__(ResearchAdapter)
        adapter._cache_cfg = cfg
        return adapter

    def test_registry_name_with_adapter_compression_ratio(self):
        cfg = CacheConfig(sketch_name="reattention", compression_ratio=0.4)
        sketch = self._shell(cfg)._build_sketch(cfg)
        self.assertIsInstance(sketch, ReAttentionSketch)
        self.assertAlmostEqual(sketch.compression_ratio, 0.4)

    def test_sketch_kwargs_pass_through_and_override(self):
        cfg = CacheConfig(
            sketch_name="random",
            compression_ratio=0.4,
            sketch_kwargs={"compression_ratio": 0.6, "seed": 3},
        )
        sketch = self._shell(cfg)._build_sketch(cfg)
        self.assertIsInstance(sketch, RandomSketch)
        self.assertAlmostEqual(sketch.compression_ratio, 0.6)
        self.assertEqual(sketch.seed, 3)

    def test_composite_names_still_special_cased(self):
        cfg = CacheConfig(sketch_name="decoding_knorm", compression_interval=9)
        sketch = self._shell(cfg)._build_sketch(cfg)
        self.assertIsInstance(sketch, DecodingSketch)
        self.assertEqual(sketch.compression_interval, 9)

    def test_none_returns_no_sketch(self):
        cfg = CacheConfig(sketch_name="none")
        self.assertIsNone(self._shell(cfg)._build_sketch(cfg))

    def test_unexpected_kwarg_raises_type_error(self):
        cfg = CacheConfig(sketch_name="knorm", sketch_kwargs={"window_size": 5})
        with self.assertRaises(TypeError):
            self._shell(cfg)._build_sketch(cfg)


if __name__ == "__main__":
    unittest.main()
