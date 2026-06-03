from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from eval_harness.sketch.cache_adapter import (
    HybridCacheAdapter,
    StandardCacheAdapter,
    create_cache_adapter,
)


class _AttentionLayer:
    def __init__(self, seq_len: int):
        self.keys = torch.zeros(1, 1, seq_len, 4)
        self.values = torch.zeros(1, 1, seq_len, 4)

    def get_seq_length(self) -> int:
        return int(self.keys.shape[2])


class _QuantAttentionLayer(_AttentionLayer):
    def __init__(self, seq_len: int):
        super().__init__(seq_len)
        self._quantized_keys = torch.zeros(1, 1, seq_len, 4)
        self._quantized_values = torch.zeros(1, 1, seq_len, 4)


class _LinearLayer:
    def __init__(self):
        self.state = torch.zeros(1)


class _Cache:
    def __init__(self, layers):
        self.layers = layers

    def __len__(self):
        return len(self.layers)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        layer = self.layers[layer_idx]
        if not hasattr(layer, "get_seq_length"):
            raise ValueError("Linear layer does not expose seq length")
        return layer.get_seq_length()


class TestStandardCacheAdapter(unittest.TestCase):
    def test_checkpoint_and_restore_all_attention_layers(self):
        cache = _Cache([_AttentionLayer(5), _AttentionLayer(7), _QuantAttentionLayer(6)])
        adapter = StandardCacheAdapter(model=SimpleNamespace(config=SimpleNamespace()))

        checkpoint = adapter.clone_or_checkpoint_for_multi_question(cache)

        cache.layers[0].keys = torch.zeros(1, 1, 9, 4)
        cache.layers[0].values = torch.zeros(1, 1, 9, 4)
        cache.layers[1].keys = torch.zeros(1, 1, 11, 4)
        cache.layers[1].values = torch.zeros(1, 1, 11, 4)
        cache.layers[2].keys = torch.zeros(1, 1, 10, 4)
        cache.layers[2].values = torch.zeros(1, 1, 10, 4)
        cache.layers[2]._quantized_keys = torch.zeros(1, 1, 10, 4)
        cache.layers[2]._quantized_values = torch.zeros(1, 1, 10, 4)

        adapter.restore_after_question(cache, checkpoint)

        self.assertEqual(cache.layers[0].keys.shape[2], 5)
        self.assertEqual(cache.layers[1].keys.shape[2], 7)
        self.assertEqual(cache.layers[2].keys.shape[2], 6)
        self.assertEqual(cache.layers[2]._quantized_keys.shape[2], 6)
        self.assertEqual(cache.layers[2]._quantized_values.shape[2], 6)


class TestHybridCacheAdapter(unittest.TestCase):
    def test_checkpoint_and_restore_only_attention_layers(self):
        cache = _Cache([_AttentionLayer(8), _LinearLayer(), _AttentionLayer(4)])
        adapter = HybridCacheAdapter(model=SimpleNamespace(config=SimpleNamespace()))

        checkpoint = adapter.clone_or_checkpoint_for_multi_question(cache)

        cache.layers[0].keys = torch.zeros(1, 1, 10, 4)
        cache.layers[0].values = torch.zeros(1, 1, 10, 4)
        cache.layers[2].keys = torch.zeros(1, 1, 9, 4)
        cache.layers[2].values = torch.zeros(1, 1, 9, 4)

        adapter.restore_after_question(cache, checkpoint)

        self.assertEqual(cache.layers[0].keys.shape[2], 8)
        self.assertEqual(cache.layers[2].keys.shape[2], 4)
        self.assertEqual(adapter.get_seq_length(cache), 8)

    def test_create_cache_adapter_detects_hybrid_layer_types(self):
        config = SimpleNamespace(
            get_text_config=lambda decoder=True: SimpleNamespace(
                layer_types=["full_attention", "linear_attention", "full_attention"]
            )
        )
        model = SimpleNamespace(config=config)

        adapter = create_cache_adapter(model)
        self.assertIsInstance(adapter, HybridCacheAdapter)


if __name__ == "__main__":
    unittest.main()
