from __future__ import annotations

import unittest

from eval_harness.research_adapter import CacheConfig


class TestCacheConfigCompatibility(unittest.TestCase):
    def test_legacy_fields_still_exist(self):
        cfg = CacheConfig()
        self.assertTrue(hasattr(cfg, "global_size"))
        self.assertTrue(hasattr(cfg, "local_size"))
        self.assertTrue(hasattr(cfg, "selection"))

    def test_new_sketch_fields_exist(self):
        cfg = CacheConfig()
        self.assertTrue(hasattr(cfg, "sketch_name"))
        self.assertTrue(hasattr(cfg, "compression_ratio"))
        self.assertTrue(hasattr(cfg, "max_context_length"))

    def test_custom_values_roundtrip(self):
        cfg = CacheConfig(
            sketch_name="knorm",
            compression_ratio=0.6,
            max_context_length=65536,
            compression_interval=16,
            target_size=1024,
            hidden_states_buffer_size=64,
        )
        self.assertEqual(cfg.sketch_name, "knorm")
        self.assertAlmostEqual(cfg.compression_ratio, 0.6)
        self.assertEqual(cfg.max_context_length, 65536)
        self.assertEqual(cfg.compression_interval, 16)
        self.assertEqual(cfg.target_size, 1024)
        self.assertEqual(cfg.hidden_states_buffer_size, 64)


if __name__ == "__main__":
    unittest.main()
