from __future__ import annotations

import unittest

import torch

from eval_harness.attention.plugins import get_attention_plugin, list_attention_plugins


class TestAttentionPlugins(unittest.TestCase):
    def test_default_and_none_disable_plugin(self):
        self.assertIsNone(get_attention_plugin(None))
        self.assertIsNone(get_attention_plugin("default"))
        self.assertIsNone(get_attention_plugin("hf_default"))

    def test_sdpa_plugin_registered(self):
        self.assertIn("sdpa", list(list_attention_plugins()))

    def test_sdpa_plugin_returns_expected_shape(self):
        plugin = get_attention_plugin("sdpa")
        self.assertIsNotNone(plugin)

        q = torch.randn(2, 4, 3, 8)
        k = torch.randn(2, 2, 5, 8)
        v = torch.randn(2, 2, 5, 8)

        out, weights = plugin(None, q, k, v, None, scaling=1.0, dropout=0.0)
        self.assertEqual(out.shape, (2, 3, 4, 8))
        self.assertIsNone(weights)

    def test_unknown_plugin_raises(self):
        with self.assertRaises(ValueError):
            get_attention_plugin("unknown_plugin")


if __name__ == "__main__":
    unittest.main()
