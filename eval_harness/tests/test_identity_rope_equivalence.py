from __future__ import annotations

import unittest

from eval_harness.research_adapter import ResearchConfig


class TestResearchConfigCompatibility(unittest.TestCase):
    def test_legacy_fields_removed(self):
        cfg = ResearchConfig()
        self.assertFalse(hasattr(cfg, "global_size"))
        self.assertFalse(hasattr(cfg, "local_size"))
        self.assertFalse(hasattr(cfg, "mid_budget"))
        self.assertFalse(hasattr(cfg, "span_size"))
        self.assertFalse(hasattr(cfg, "selection"))
        self.assertFalse(hasattr(cfg, "chunk_size"))

    def test_new_door_fields_exist(self):
        cfg = ResearchConfig()
        self.assertTrue(hasattr(cfg, "positional_method"))
        self.assertTrue(hasattr(cfg, "attention_method"))
        self.assertTrue(hasattr(cfg, "kv_compressor"))
        self.assertTrue(hasattr(cfg, "compression_ratio"))
        self.assertTrue(hasattr(cfg, "max_context_length"))

    def test_custom_values_roundtrip(self):
        cfg = ResearchConfig(
            kv_compressor="knorm",
            compression_ratio=0.6,
            max_context_length=65536,
            compression_interval=16,
            target_size=1024,
            hidden_states_buffer_size=64,
        )
        self.assertEqual(cfg.kv_compressor, "knorm")
        self.assertAlmostEqual(cfg.compression_ratio, 0.6)
        self.assertEqual(cfg.max_context_length, 65536)
        self.assertEqual(cfg.compression_interval, 16)
        self.assertEqual(cfg.target_size, 1024)
        self.assertEqual(cfg.hidden_states_buffer_size, 64)


if __name__ == "__main__":
    unittest.main()
