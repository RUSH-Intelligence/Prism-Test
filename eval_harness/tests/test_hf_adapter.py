import types
import unittest

import torch

import eval_harness.hf_adapter as hf_module
from eval_harness.hf_adapter import HFAdapter
from eval_harness.long_context import LongContextCompressionConfig


class _FakeRegistry:
    def __init__(self):
        self._global_mapping = {}

    def register(self, name, fn):
        self._global_mapping[name] = fn

    def valid_keys(self):
        return list(self._global_mapping.keys())


class _FakeLayer(torch.nn.Module):
    def __init__(self, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.rotary_emb = object()
        self.config = types.SimpleNamespace(_attn_implementation="eager")


class _FakeModel:
    def __init__(self):
        self.config = types.SimpleNamespace(model_type="llama", rope_scaling=None)
        self.layer0 = _FakeLayer(0)
        self.layer1 = _FakeLayer(1)

    def named_modules(self):
        return [
            ("", self),
            ("layer0", self.layer0),
            ("layer1", self.layer1),
        ]


class HFAdapterSparseHookTests(unittest.TestCase):
    def setUp(self):
        self._orig_attn_registry = hf_module.ALL_ATTENTION_FUNCTIONS
        self._orig_mask_registry = hf_module.ALL_MASK_ATTENTION_FUNCTIONS
        hf_module.ALL_ATTENTION_FUNCTIONS = _FakeRegistry()
        hf_module.ALL_MASK_ATTENTION_FUNCTIONS = _FakeRegistry()

    def tearDown(self):
        hf_module.ALL_ATTENTION_FUNCTIONS = self._orig_attn_registry
        hf_module.ALL_MASK_ATTENTION_FUNCTIONS = self._orig_mask_registry

    def _make_adapter(self) -> HFAdapter:
        adapter = object.__new__(HFAdapter)
        adapter._compression_cfg = LongContextCompressionConfig(
            enabled=True,
            max_context_len=5,
            sink_tokens=1,
            local_tokens=2,
            top_k_tokens=2,
            span_tokens=0,
        )
        adapter._enable_sparse_attention_hook = True
        adapter._enable_rope_relative_reposition = True
        adapter._rope_position_correction = 0.05
        adapter._registered_attention_name = None
        adapter._custom_attention_fn = None
        adapter._active_sparse_meta_data = None
        adapter._model = _FakeModel()
        return adapter

    def test_sparse_mode_swaps_and_restores_implementation(self):
        adapter = self._make_adapter()
        sparse_meta_data = {}

        self.assertEqual(adapter._model.layer0.config._attn_implementation, "eager")
        self.assertEqual(adapter._model.layer1.config._attn_implementation, "eager")

        with adapter.enable_sparse_mode(sparse_meta_data=sparse_meta_data):
            name = adapter._registered_attention_name
            self.assertIsNotNone(name)
            self.assertEqual(adapter._model.layer0.config._attn_implementation, name)
            self.assertEqual(adapter._model.layer1.config._attn_implementation, name)
            self.assertIs(adapter._active_sparse_meta_data, sparse_meta_data)

        self.assertEqual(adapter._model.layer0.config._attn_implementation, "eager")
        self.assertEqual(adapter._model.layer1.config._attn_implementation, "eager")
        self.assertIsNone(adapter._active_sparse_meta_data)

        adapter._cleanup_attention_registration()
        self.assertEqual(hf_module.ALL_ATTENTION_FUNCTIONS.valid_keys(), [])

    def test_custom_attention_returns_expected_shape_and_layer_metadata(self):
        adapter = self._make_adapter()
        adapter._active_sparse_meta_data = {}
        custom_attention = adapter.get_custom_attention_function()

        queries = torch.randn(1, 2, 3, 4, dtype=torch.float32)
        keys = torch.randn(1, 2, 6, 4, dtype=torch.float32)
        values = torch.randn(1, 2, 6, 4, dtype=torch.float32)

        output, attn_weights = custom_attention(
            module=adapter._model.layer0,
            queries=queries,
            keys=keys,
            values=values,
            attention_mask=None,
            scaling=1.0,
            dropout=0.0,
        )

        self.assertEqual(tuple(output.shape), (1, 3, 2, 4))
        self.assertIsNone(attn_weights)

        layer_kept_indices = adapter._active_sparse_meta_data["layer_kept_indices"]
        layer_seq_lens = adapter._active_sparse_meta_data["layer_seq_lens"]
        self.assertIn(0, layer_kept_indices)
        self.assertIn(0, layer_seq_lens)
        self.assertEqual(layer_seq_lens[0], 6)
        self.assertLessEqual(len(layer_kept_indices[0]), 5)

    def test_layer_topk_scale_can_disable_middle_budget_per_layer(self):
        adapter = self._make_adapter()
        sparse_meta_data = {"layer_topk_scale": {0: 0.0}}
        adapter._active_sparse_meta_data = sparse_meta_data
        custom_attention = adapter.get_custom_attention_function()

        queries = torch.randn(1, 2, 3, 4, dtype=torch.float32)
        keys = torch.randn(1, 2, 6, 4, dtype=torch.float32)
        values = torch.randn(1, 2, 6, 4, dtype=torch.float32)

        custom_attention(
            module=adapter._model.layer0,
            queries=queries,
            keys=keys,
            values=values,
            attention_mask=None,
            scaling=1.0,
            dropout=0.0,
        )

        # With top-k scale forced to 0, only sink(0) + local(4,5) remain.
        self.assertEqual(sparse_meta_data["layer_kept_indices"][0], [0, 4, 5])

    def test_no_topk_when_seq_len_within_max_context(self):
        adapter = self._make_adapter()
        adapter._compression_cfg.max_context_len = 8
        sparse_meta_data = {}
        adapter._active_sparse_meta_data = sparse_meta_data
        custom_attention = adapter.get_custom_attention_function()

        queries = torch.randn(1, 2, 3, 4, dtype=torch.float32)
        keys = torch.randn(1, 2, 6, 4, dtype=torch.float32)
        values = torch.randn(1, 2, 6, 4, dtype=torch.float32)

        custom_attention(
            module=adapter._model.layer0,
            queries=queries,
            keys=keys,
            values=values,
            attention_mask=None,
            scaling=1.0,
            dropout=0.0,
        )

        self.assertEqual(sparse_meta_data["layer_kept_indices"][0], [0, 1, 2, 3, 4, 5])


if __name__ == "__main__":
    unittest.main()
