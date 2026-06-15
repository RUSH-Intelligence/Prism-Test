"""Regression tests for `_is_non_full_attention_layer` (KV-compression layer gating).

Pins the Qwen3.5 hybrid bug: a linear-attention layer has NO ``self_attn``
submodule but exposes ``layer_type="linear_attention"``. The detector must
classify it as non-full (skip) by reading ``layer_type`` BEFORE touching
``self_attn`` — otherwise the install loop crashes on ``layer.self_attn``.
No model loading; pure fake layers.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace

from eval_harness.kv_compression.base import _is_non_full_attention_layer


class TestNonFullAttentionDetection(unittest.TestCase):
    def test_qwen35_linear_layer_no_self_attn_is_skipped(self):
        # The exact crashing case: linear layer, no self_attn, typed linear.
        layer = SimpleNamespace(layer_type="linear_attention")
        self.assertTrue(_is_non_full_attention_layer(layer))

    def test_qwen35_full_layer_is_hooked(self):
        layer = SimpleNamespace(layer_type="full_attention", self_attn=SimpleNamespace())
        self.assertFalse(_is_non_full_attention_layer(layer))

    def test_attention_type_sliding_is_skipped(self):
        layer = SimpleNamespace(attention_type="sliding_attention", self_attn=SimpleNamespace())
        self.assertTrue(_is_non_full_attention_layer(layer))

    def test_layer_without_self_attn_and_without_type_is_skipped(self):
        # Cannot install a KV hook on a layer with no self_attn → must skip,
        # not crash. (Pre-fix this returned False → AttributeError downstream.)
        layer = SimpleNamespace()
        self.assertTrue(_is_non_full_attention_layer(layer))

    def test_llama_style_full_layer_is_hooked(self):
        # Real self_attn, no layer_type hints → full attention.
        attn = SimpleNamespace(is_sliding=None, is_linear=None, config=None)
        layer = SimpleNamespace(self_attn=attn)
        self.assertFalse(_is_non_full_attention_layer(layer))

    def test_gemma_style_sliding_via_config_window_is_skipped(self):
        attn = SimpleNamespace(is_sliding=None, is_linear=None,
                               config=SimpleNamespace(sliding_window=4096))
        layer = SimpleNamespace(self_attn=attn)
        self.assertTrue(_is_non_full_attention_layer(layer))

    def test_attn_is_sliding_flag_respected(self):
        layer = SimpleNamespace(self_attn=SimpleNamespace(is_sliding=True, is_linear=None, config=None))
        self.assertTrue(_is_non_full_attention_layer(layer))
        layer2 = SimpleNamespace(self_attn=SimpleNamespace(is_sliding=False, is_linear=None, config=None))
        self.assertFalse(_is_non_full_attention_layer(layer2))


if __name__ == "__main__":
    unittest.main()
