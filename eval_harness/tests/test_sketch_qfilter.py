"""Tests for QFilterSketch — port of kvpress QFilterPress (qfilter_press.py).

No model loading, no network: hub access is stubbed via ``unittest.mock`` and
filters are injected as plain tensors (the constructor accepts them).
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from eval_harness.kv_compression.compressors.qfilter_sketch import QFilters, QFilterSketch
from eval_harness.kv_compression.registry import available_kv_compressors, get_kv_compressor, get_kv_compressor_class


class _FakeAttnModule(nn.Module):
    """Minimal fake: QFilterSketch only reads ``layer_idx`` and (via the
    inherited ScorerKVCompressor gather) ``head_dim`` — no q_proj, no GQA fields."""

    def __init__(self, head_dim=4, layer_idx=0):
        super().__init__()
        self.head_dim = head_dim
        self.layer_idx = layer_idx


def _kvpress_qfilter_compress(q_filters, layer_idx, keys, values, compression_ratio, head_dim):
    """In-test transcription of kvpress QFilterPress.score + ScorerPress.compress."""
    q_filter = q_filters[layer_idx][None, :, None]
    scores = -(q_filter * keys).sum(dim=-1)
    n_kept = int(keys.shape[2] * (1 - compression_ratio))
    indices = scores.topk(n_kept, dim=-1).indices
    indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
    return keys.gather(2, indices).contiguous(), values.gather(2, indices).contiguous()


def _fake_model(name_or_path, dtype=torch.float32):
    return SimpleNamespace(config=SimpleNamespace(name_or_path=name_or_path), dtype=dtype)


class TestQFilterRegistry(unittest.TestCase):
    def test_registered_name_resolves(self):
        self.assertIn("qfilter", available_kv_compressors())
        self.assertIs(get_kv_compressor_class("qfilter"), QFilterSketch)

    def test_get_kv_compressor_instantiates_with_kwargs(self):
        sketch = get_kv_compressor("qfilter", compression_ratio=0.3)
        self.assertIsInstance(sketch, QFilterSketch)
        self.assertAlmostEqual(sketch.compression_ratio, 0.3)
        self.assertIsNone(sketch.q_filters)


class TestQFilterScoreAndCompress(unittest.TestCase):
    def test_zero_ratio_noop_without_filters(self):
        sketch = QFilterSketch(compression_ratio=0.0)
        module = _FakeAttnModule(head_dim=4, layer_idx=0)
        keys = torch.randn(1, 2, 8, 4)
        values = torch.randn(1, 2, 8, 4)
        out_keys, out_values = sketch.compress(module, None, keys, values, None, {})
        self.assertIs(out_keys, keys)
        self.assertIs(out_values, values)

    def test_missing_filters_raises_value_error(self):
        sketch = QFilterSketch(compression_ratio=0.5)
        module = _FakeAttnModule(head_dim=4, layer_idx=0)
        keys = torch.randn(1, 2, 8, 4)
        values = torch.randn(1, 2, 8, 4)
        with self.assertRaisesRegex(ValueError, "Q-filters not loaded"):
            sketch.compress(module, None, keys, values, None, {})

    def test_hand_computed_exact_selection(self):
        sketch = QFilterSketch(compression_ratio=0.5, q_filters=torch.tensor([[[1.0, 0.0]]]))
        module = _FakeAttnModule(head_dim=2, layer_idx=0)
        keys = torch.tensor([[[[3.0, 9.0], [1.0, 9.0], [4.0, 9.0], [2.0, 9.0]]]])
        values = torch.arange(4, dtype=torch.float32).view(1, 1, 4, 1).expand(1, 1, 4, 2).contiguous()

        out_keys, out_values = sketch.compress(module, None, keys, values, None, {})

        self.assertEqual(tuple(out_keys.shape), (1, 1, 2, 2))
        self.assertTrue(torch.equal(out_keys[0, 0], torch.tensor([[1.0, 9.0], [2.0, 9.0]])))
        self.assertTrue(torch.equal(out_values[0, 0], torch.tensor([[1.0, 1.0], [3.0, 3.0]])))

    def test_layer_indexing_selects_filter_row(self):
        torch.manual_seed(0)
        h_kv, d = 2, 4
        q_filters = torch.randn(2, h_kv, d)
        keys = torch.randn(1, h_kv, 6, d)
        sketch = QFilterSketch(compression_ratio=0.5, q_filters=q_filters)

        scores = sketch.score(_FakeAttnModule(head_dim=d, layer_idx=1), None, keys, None, None, {})

        expected_layer1 = -(keys * q_filters[1][None, :, None]).sum(-1)
        expected_layer0 = -(keys * q_filters[0][None, :, None]).sum(-1)
        self.assertTrue(torch.equal(scores, expected_layer1))
        self.assertFalse(torch.equal(scores, expected_layer0))

    def test_gqa_shaped_oracle_and_per_head_independence(self):
        torch.manual_seed(42)
        batch, h_kv, seq, d = 2, 4, 16, 8
        keys = torch.randn(batch, h_kv, seq, d)
        values = torch.randn(batch, h_kv, seq, d)
        q_filters = torch.randn(3, h_kv, d)
        sketch = QFilterSketch(compression_ratio=0.25, q_filters=q_filters)
        module = _FakeAttnModule(head_dim=d, layer_idx=2)
        self.assertFalse(hasattr(module, "q_proj"))
        self.assertFalse(hasattr(module, "num_key_value_groups"))

        scores = sketch.score(module, None, keys, values, None, {})
        self.assertEqual(tuple(scores.shape), (batch, h_kv, seq))

        out_keys, out_values = sketch.compress(module, None, keys, values, None, {})

        expected_scores = -torch.einsum("hd,bhsd->bhs", q_filters[2], keys)
        idx = expected_scores.topk(12, dim=-1).indices
        expanded = idx.unsqueeze(-1).expand(-1, -1, -1, d)
        self.assertTrue(torch.equal(out_keys, keys.gather(2, expanded)))
        self.assertTrue(torch.equal(out_values, values.gather(2, expanded)))
        self.assertEqual(tuple(out_keys.shape), (2, 4, 12, 8))

        head_index_sets = [frozenset(idx[0, h].tolist()) for h in range(h_kv)]
        self.assertTrue(any(s != head_index_sets[0] for s in head_index_sets[1:]))

    def test_n_kept_rounding_and_empty_edge(self):
        d = 4
        q_filters = torch.randn(1, 2, d)
        module = _FakeAttnModule(head_dim=d, layer_idx=0)

        sketch = QFilterSketch(compression_ratio=0.5, q_filters=q_filters)
        out_keys, _ = sketch.compress(module, None, torch.randn(1, 2, 5, d), torch.randn(1, 2, 5, d), None, {})
        self.assertEqual(out_keys.shape[2], 2)

        sketch = QFilterSketch(compression_ratio=0.9, q_filters=q_filters)
        out_keys, out_values = sketch.compress(
            module, None, torch.randn(1, 2, 4, d), torch.randn(1, 2, 4, d), None, {}
        )
        self.assertEqual(tuple(out_keys.shape), (1, 2, 0, d))
        self.assertEqual(tuple(out_values.shape), (1, 2, 0, d))

    def test_dtype_promotion_and_device_move(self):
        d = 4
        q_filters = torch.randn(1, 2, d, dtype=torch.float32)
        keys = torch.randn(1, 2, 6, d, dtype=torch.float64)
        sketch = QFilterSketch(compression_ratio=0.5, q_filters=q_filters)

        scores = sketch.score(_FakeAttnModule(head_dim=d, layer_idx=0), None, keys, None, None, {})

        self.assertEqual(scores.dtype, torch.float64)
        self.assertEqual(scores.device, keys.device)


class TestQFilterPostInitFromModel(unittest.TestCase):
    def setUp(self):
        QFilterSketch.load_q_filters.cache_clear()
        self.addCleanup(QFilterSketch.load_q_filters.cache_clear)

    def test_meta_llama_name_mangling_and_dtype_cast(self):
        stub = SimpleNamespace(q_filters=torch.randn(2, 2, 4, dtype=torch.float32))
        sketch = QFilterSketch(compression_ratio=0.5)
        with patch.object(QFilters, "from_pretrained", return_value=stub) as mock_fp:
            sketch.post_init_from_model(
                _fake_model("meta-llama/Meta-Llama-3.1-8B-Instruct", dtype=torch.bfloat16)
            )
        mock_fp.assert_called_once_with("nthngdy/Llama-3.1-8B-Instruct_qfilt")
        self.assertEqual(sketch.q_filters.dtype, torch.bfloat16)
        self.assertEqual(tuple(sketch.q_filters.shape), (2, 2, 4))

    def test_405b_exemption_keeps_meta_prefix(self):
        stub = SimpleNamespace(q_filters=torch.randn(2, 2, 4, dtype=torch.float32))
        sketch = QFilterSketch(compression_ratio=0.5)
        with patch.object(QFilters, "from_pretrained", return_value=stub) as mock_fp:
            sketch.post_init_from_model(_fake_model("nvidia/Meta-Llama-3.1-405B-FP8"))
        mock_fp.assert_called_once_with("nthngdy/Meta-Llama-3.1-405B-FP8_qfilt")

    def test_loader_type_error_becomes_value_error(self):
        sketch = QFilterSketch(compression_ratio=0.5)
        with patch.object(QFilters, "from_pretrained", side_effect=TypeError), patch.object(
            QFilterSketch, "available_qfilters", return_value=["Llama-3.1-8B-Instruct"]
        ):
            with self.assertRaisesRegex(ValueError, "Could not load Q-filters"):
                sketch.post_init_from_model(_fake_model("meta-llama/Llama-3-8B"))

    def test_loader_is_cached_per_model_name(self):
        stub = SimpleNamespace(q_filters=torch.randn(1, 2, 4, dtype=torch.float32))
        with patch.object(QFilters, "from_pretrained", return_value=stub) as mock_fp:
            first = QFilterSketch.load_q_filters("Llama-3.1-8B-Instruct")
            second = QFilterSketch.load_q_filters("Llama-3.1-8B-Instruct")
        mock_fp.assert_called_once()
        self.assertIs(first, second)

    def test_zero_ratio_skips_hub_download(self):
        sketch = QFilterSketch(compression_ratio=0.0)
        with patch.object(QFilters, "from_pretrained") as mock_fp:
            sketch.post_init_from_model(_fake_model("meta-llama/Llama-3.1-8B-Instruct"))
        mock_fp.assert_not_called()
        self.assertIsNone(sketch.q_filters)

    def test_injected_filters_skip_hub_download(self):
        filters = torch.randn(1, 2, 4)
        sketch = QFilterSketch(compression_ratio=0.5, q_filters=filters)
        with patch.object(QFilters, "from_pretrained") as mock_fp:
            sketch.post_init_from_model(_fake_model("meta-llama/Llama-3.1-8B-Instruct"))
        mock_fp.assert_not_called()
        self.assertIs(sketch.q_filters, filters)


class TestQFilterForwardHookIntegration(unittest.TestCase):
    def test_prefill_hook_compresses_cache_then_decode_noop(self):
        from transformers import DynamicCache

        torch.manual_seed(3)
        batch, h_kv, seq, d, hidden_dim = 1, 2, 10, 4, 8
        keys = torch.randn(batch, h_kv, seq, d)
        values = torch.randn(batch, h_kv, seq, d)
        cache = DynamicCache()
        cache.update(keys.clone(), values.clone(), 0)

        module = _FakeAttnModule(head_dim=d, layer_idx=0)
        q_filters = torch.randn(1, h_kv, d)
        sketch = QFilterSketch(compression_ratio=0.4, q_filters=q_filters)

        prefill_kwargs = {
            "hidden_states": torch.randn(batch, seq, hidden_dim),
            "past_key_values": cache,
            "cache_position": torch.arange(seq),
        }
        output = (torch.randn(batch, seq, hidden_dim), None)
        result = sketch.forward_hook(module, [], prefill_kwargs, output)
        self.assertIs(result, output)

        expected_k, expected_v = _kvpress_qfilter_compress(q_filters, 0, keys, values, 0.4, d)
        self.assertEqual(cache.layers[0].keys.shape[2], int(seq * 0.6))
        self.assertTrue(torch.equal(cache.layers[0].keys, expected_k))
        self.assertTrue(torch.equal(cache.layers[0].values, expected_v))

        kept_keys = cache.layers[0].keys.clone()
        kept_values = cache.layers[0].values.clone()
        decode_kwargs = {
            "hidden_states": torch.randn(batch, 1, hidden_dim),
            "past_key_values": cache,
            "cache_position": torch.tensor([seq]),
        }
        sketch.forward_hook(module, [], decode_kwargs, (torch.randn(batch, 1, hidden_dim), None))
        self.assertTrue(torch.equal(cache.layers[0].keys, kept_keys))
        self.assertTrue(torch.equal(cache.layers[0].values, kept_values))


if __name__ == "__main__":
    unittest.main()
