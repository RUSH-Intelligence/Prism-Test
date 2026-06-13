"""Tests for KVzapSketch — port of kvpress/presses/kvzap_press.py.

The reference oracle ``_kvzap_compress_reference`` is an in-test transcription
of kvpress ``KVzapPress.score`` + ``ScorerPress.compress``; no hub access is
needed anywhere (surrogates are constructed randomly or with hand-set weights,
mirroring kvpress tests/default_presses.py::TestKVzapPress).
"""

from __future__ import annotations

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from transformers import DynamicCache

from eval_harness.kv_compression.compressors.kvzap_sketch import KVzapConfig, KVzapModel, KVzapSketch
from eval_harness.kv_compression.registry import available_kv_compressors, get_kv_compressor, get_kv_compressor_class


class _FakeAttnModule(nn.Module):
    """Minimal attention module: KVzap's score only reads layer_idx; compress reads head_dim."""

    def __init__(self, layer_idx=0, head_dim=4, num_heads=4, num_kv_heads=2):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_key_value_heads = num_kv_heads
        self.is_sliding = False


def _linear_surrogate(weights: list[torch.Tensor]) -> KVzapModel:
    """KVzapModel (linear variant) with hand-set per-layer weights and zero biases."""
    out_dim, in_dim = weights[0].shape
    config = KVzapConfig(input_dim=in_dim, output_dim=out_dim, hidden_dim=None, n_modules=len(weights))
    surrogate = KVzapModel(config)
    with torch.no_grad():
        for layer, w in zip(surrogate.layers, weights):
            layer.weight.copy_(w)
            layer.bias.zero_()
    return surrogate


def _random_surrogate(input_dim, output_dim, n_modules, hidden_dim=None, seed=0) -> KVzapModel:
    torch.manual_seed(seed)
    config = KVzapConfig(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim, n_modules=n_modules)
    return KVzapModel(config)


def _position_coded_keys(B, H_kv, S, D, head_offset=0.0):
    """keys[b, h, s, :] == s + head_offset * h — selection order observable in values."""
    pos = torch.arange(S, dtype=torch.float32).view(1, 1, S, 1)
    head = torch.arange(H_kv, dtype=torch.float32).view(1, H_kv, 1, 1) * head_offset
    return (pos + head).expand(B, H_kv, S, D).contiguous()


def _kvzap_compress_reference(surrogate, layer_idx, hidden_states, keys, values, compression_ratio):
    """Transcription of kvpress KVzapPress.score (kvzap_press.py:76-79) +
    ScorerPress.compress (scorer_press.py:86-102)."""
    kvzap_module = copy.deepcopy(surrogate.layers[layer_idx])
    kvzap_module = kvzap_module.to(hidden_states.device, dtype=hidden_states.dtype).eval()
    scores = kvzap_module(hidden_states).transpose(1, 2)

    k_len = keys.shape[2]
    n_kept = int(k_len * (1 - compression_ratio))
    indices = scores.topk(n_kept, dim=-1).indices
    expanded = indices.unsqueeze(-1).expand(-1, -1, -1, keys.shape[-1])
    return keys.gather(2, expanded).contiguous(), values.gather(2, expanded).contiguous(), indices


class TestKVzapRegistry(unittest.TestCase):
    def test_registered_as_kvzap(self):
        self.assertIn("kvzap", available_kv_compressors())
        self.assertIs(get_kv_compressor_class("kvzap"), KVzapSketch)

    def test_get_kv_compressor_instantiates_with_kwargs(self):
        sketch = get_kv_compressor(
            "kvzap", compression_ratio=0.25, model_type="linear", model_name_override="Qwen3-8B"
        )
        self.assertIsInstance(sketch, KVzapSketch)
        self.assertAlmostEqual(sketch.compression_ratio, 0.25)
        self.assertEqual(sketch.model_type, "linear")
        self.assertEqual(sketch.model_name_override, "Qwen3-8B")

    def test_kvzap_model_name_not_an_init_arg(self):
        with self.assertRaises(TypeError):
            KVzapSketch(compression_ratio=0.5, kvzap_model_name="nvidia/KVzap-mlp-Qwen3-8B")


class TestKVzapZeroRatio(unittest.TestCase):
    def test_zero_ratio_returns_same_objects_without_surrogate(self):
        sketch = KVzapSketch(compression_ratio=0.0)
        self.assertFalse(hasattr(sketch, "kvzap_model"))
        module = _FakeAttnModule()
        keys = torch.randn(1, 2, 6, 4)
        values = torch.randn(1, 2, 6, 4)
        out_keys, out_values = sketch.compress(module, torch.randn(1, 6, 8), keys, values, None, {})
        self.assertIs(out_keys, keys)
        self.assertIs(out_values, values)


class TestKVzapHandComputed(unittest.TestCase):
    def test_exact_selection_identity_linear_surrogate(self):
        surrogate = _linear_surrogate([torch.eye(2)])
        sketch = KVzapSketch(compression_ratio=0.5)
        sketch.kvzap_model = surrogate
        module = _FakeAttnModule(layer_idx=0, head_dim=4)

        hidden_states = torch.tensor([[[3.0, 0.0], [1.0, 2.0], [2.0, 1.0], [0.0, 4.0]]])
        # scores = (hs @ I).transpose(1, 2): head0 = [3, 1, 2, 0], head1 = [0, 2, 1, 4]
        # n_kept = int(4 * 0.5) = 2; topk (descending): head0 -> [0, 2], head1 -> [3, 1]
        keys = _position_coded_keys(1, 2, 4, 4)
        values = _position_coded_keys(1, 2, 4, 4, head_offset=100.0)

        out_keys, out_values = sketch.compress(module, hidden_states, keys, values, None, {})

        expected_keys = torch.tensor(
            [[[[0.0] * 4, [2.0] * 4], [[3.0] * 4, [1.0] * 4]]]
        )
        expected_values = torch.tensor(
            [[[[0.0] * 4, [2.0] * 4], [[103.0] * 4, [101.0] * 4]]]
        )
        self.assertTrue(torch.equal(out_keys, expected_keys))
        self.assertTrue(torch.equal(out_values, expected_values))


class TestKVzapReferenceOracle(unittest.TestCase):
    def _run_case(self, hidden_dim, dtype, seed):
        torch.manual_seed(seed)
        S, ratio = 16, 0.25
        hidden_states = torch.randn(1, S, 8, dtype=dtype)
        keys = torch.randn(1, 2, S, 4, dtype=dtype)
        values = torch.randn(1, 2, S, 4, dtype=dtype)
        surrogate = _random_surrogate(8, 2, 1, hidden_dim=hidden_dim, seed=seed)

        ref_keys, ref_values, ref_indices = _kvzap_compress_reference(
            surrogate, 0, hidden_states, keys, values, ratio
        )
        self.assertEqual(ref_indices.shape, (1, 2, int(S * (1 - ratio))))

        sketch = KVzapSketch(compression_ratio=ratio)
        sketch.kvzap_model = surrogate
        out_keys, out_values = sketch.compress(
            _FakeAttnModule(layer_idx=0, head_dim=4), hidden_states, keys, values, None, {}
        )
        self.assertTrue(torch.equal(out_keys, ref_keys))
        self.assertTrue(torch.equal(out_values, ref_values))

    def test_linear_fp32(self):
        self._run_case(hidden_dim=None, dtype=torch.float32, seed=0)

    def test_mlp_fp32(self):
        self._run_case(hidden_dim=5, dtype=torch.float32, seed=1)

    def test_linear_bf16(self):
        self._run_case(hidden_dim=None, dtype=torch.bfloat16, seed=2)

    def test_mlp_bf16(self):
        self._run_case(hidden_dim=5, dtype=torch.bfloat16, seed=3)


class TestKVzapLayerRouting(unittest.TestCase):
    def test_per_layer_module_dispatch(self):
        torch.manual_seed(0)
        w = torch.randn(2, 3)
        surrogate = _linear_surrogate([w, -w])
        sketch = KVzapSketch(compression_ratio=0.5)
        sketch.kvzap_model = surrogate

        hidden_states = torch.randn(1, 6, 3)
        keys = torch.randn(1, 2, 6, 4)
        values = torch.randn(1, 2, 6, 4)
        scores0 = sketch.score(_FakeAttnModule(layer_idx=0), hidden_states, keys, values, None, {})
        scores1 = sketch.score(_FakeAttnModule(layer_idx=1), hidden_states, keys, values, None, {})
        self.assertTrue(torch.equal(scores0, -scores1))


class TestKVzapGQA(unittest.TestCase):
    def test_per_kv_head_scores_no_group_reduction(self):
        surrogate = _linear_surrogate([torch.tensor([[1.0, 0.0], [-1.0, 0.0]])])
        sketch = KVzapSketch(compression_ratio=0.75)
        sketch.kvzap_model = surrogate
        module = _FakeAttnModule(layer_idx=0, head_dim=4, num_heads=4, num_kv_heads=2)

        S = 8
        hidden_states = torch.zeros(1, S, 2)
        hidden_states[0, :, 0] = torch.arange(S, dtype=torch.float32)
        # head0 scores = [0..7] -> topk(2) = [7, 6]; head1 scores = -[0..7] -> topk(2) = [0, 1]
        keys = _position_coded_keys(1, 2, S, 4)
        values = _position_coded_keys(1, 2, S, 4)

        out_keys, out_values = sketch.compress(module, hidden_states, keys, values, None, {})
        self.assertEqual(out_keys.shape, (1, 2, 2, 4))
        expected = torch.tensor([[[[7.0] * 4, [6.0] * 4], [[0.0] * 4, [1.0] * 4]]])
        self.assertTrue(torch.equal(out_keys, expected))
        self.assertTrue(torch.equal(out_values, expected))
        kept_head0 = set(out_keys[0, 0, :, 0].tolist())
        kept_head1 = set(out_keys[0, 1, :, 0].tolist())
        self.assertEqual(kept_head0 & kept_head1, set())


class TestKVzapPostInitFromModel(unittest.TestCase):
    def _fake_model(self, name_or_path):
        return SimpleNamespace(config=SimpleNamespace(name_or_path=name_or_path))

    def test_repo_id_derivation_and_name_guard(self):
        with patch.object(KVzapModel, "from_pretrained", side_effect=lambda name: ("loaded", name)) as mock_fp:
            sketch = KVzapSketch(compression_ratio=0.5)
            model = self._fake_model("Qwen/Qwen3-8B")

            sketch.post_init_from_model(model)
            self.assertEqual(mock_fp.call_count, 1)
            self.assertEqual(mock_fp.call_args.args[0], "nvidia/KVzap-mlp-Qwen3-8B")
            self.assertEqual(sketch.kvzap_model_name, "nvidia/KVzap-mlp-Qwen3-8B")
            self.assertEqual(sketch.kvzap_model, ("loaded", "nvidia/KVzap-mlp-Qwen3-8B"))

            sketch.post_init_from_model(model)
            self.assertEqual(mock_fp.call_count, 1)

            sketch.model_type = "linear"
            sketch.post_init_from_model(model)
            self.assertEqual(mock_fp.call_count, 2)
            self.assertEqual(mock_fp.call_args.args[0], "nvidia/KVzap-linear-Qwen3-8B")

    def test_model_name_override_bypasses_derivation(self):
        with patch.object(KVzapModel, "from_pretrained", side_effect=lambda name: ("loaded", name)) as mock_fp:
            sketch = KVzapSketch(compression_ratio=0.5, model_name_override="Llama-3.1-8B-Instruct")
            model = self._fake_model("/scratch/hf_snapshots/llama31_local_dir")
            sketch.post_init_from_model(model)
            self.assertEqual(mock_fp.call_args.args[0], "nvidia/KVzap-mlp-Llama-3.1-8B-Instruct")


class _TestKVzapSketch(KVzapSketch):
    """Hub-free variant mirroring kvpress tests/default_presses.py::TestKVzapPress."""

    def post_init_from_model(self, model):
        config = KVzapConfig(
            input_dim=model.config.hidden_size,
            output_dim=model.config.num_key_value_heads,
            hidden_dim=None,
            n_modules=model.config.num_hidden_layers,
        )
        self.kvzap_model = KVzapModel(config)


class _HookFakeAttn(nn.Module):
    def __init__(self, layer_idx, head_dim):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = head_dim
        self.is_sliding = False

    def forward(self, hidden_states=None, past_key_values=None, cache_position=None, **kwargs):
        return (hidden_states, None)


class _HookFakeLayer(nn.Module):
    def __init__(self, layer_idx, head_dim):
        super().__init__()
        self.self_attn = _HookFakeAttn(layer_idx, head_dim)


class _HookFakeLanguageModel(nn.Module):
    def __init__(self, n_layers, head_dim):
        super().__init__()
        self.layers = nn.ModuleList([_HookFakeLayer(i, head_dim) for i in range(n_layers)])
        self.rotary_emb = nn.Identity()


class _HookFakeModel(nn.Module):
    def __init__(self, n_layers, hidden_size, num_kv_heads, head_dim):
        super().__init__()
        self.model = _HookFakeLanguageModel(n_layers, head_dim)
        self.config = SimpleNamespace(
            hidden_size=hidden_size,
            num_key_value_heads=num_kv_heads,
            num_hidden_layers=n_layers,
            name_or_path="fake/fake-model",
        )


class TestKVzapHubFreeFullFlow(unittest.TestCase):
    def test_hooked_layers_compress_to_uniform_length(self):
        torch.manual_seed(0)
        n_layers, hidden, n_kv, head_dim, S, ratio = 2, 6, 2, 4, 12, 0.5
        model = _HookFakeModel(n_layers, hidden, n_kv, head_dim)
        sketch = _TestKVzapSketch(compression_ratio=ratio)

        cache = DynamicCache()
        hidden_states = torch.randn(1, S, hidden)
        with sketch(model):
            self.assertIsInstance(sketch.kvzap_model, KVzapModel)
            self.assertEqual(len(sketch.kvzap_model.layers), n_layers)
            for i, layer in enumerate(model.model.layers):
                cache.update(torch.randn(1, n_kv, S, head_dim), torch.randn(1, n_kv, S, head_dim), i)
                layer.self_attn(
                    hidden_states=hidden_states,
                    past_key_values=cache,
                    cache_position=torch.arange(S),
                )

        n_kept = int(S * (1 - ratio))
        lengths = [cache.layers[i].keys.shape[2] for i in range(n_layers)]
        self.assertEqual(lengths, [n_kept] * n_layers)
        self.assertEqual(len(set(lengths)), 1)

        # Decode step: gate must skip compression (cache_position[-1] > q_len).
        with sketch(model):
            for i, layer in enumerate(model.model.layers):
                cache.update(torch.randn(1, n_kv, 1, head_dim), torch.randn(1, n_kv, 1, head_dim), i)
                layer.self_attn(
                    hidden_states=torch.randn(1, 1, hidden),
                    past_key_values=cache,
                    cache_position=torch.tensor([n_kept]),
                )
        for i in range(n_layers):
            self.assertEqual(cache.layers[i].keys.shape[2], n_kept + 1)

    def test_hooks_removed_on_exit(self):
        model = _HookFakeModel(2, 6, 2, 4)
        sketch = _TestKVzapSketch(compression_ratio=0.5)
        with sketch(model):
            pass
        for layer in model.model.layers:
            self.assertEqual(len(layer.self_attn._forward_hooks), 0)


class TestKVzapDtypeCast(unittest.TestCase):
    def test_fp32_surrogate_follows_bf16_hidden_states(self):
        surrogate = _random_surrogate(8, 2, 1, hidden_dim=5, seed=0)
        self.assertEqual(next(surrogate.parameters()).dtype, torch.float32)
        sketch = KVzapSketch(compression_ratio=0.5)
        sketch.kvzap_model = surrogate

        hidden_states = torch.randn(1, 6, 8, dtype=torch.bfloat16)
        keys = torch.randn(1, 2, 6, 4, dtype=torch.bfloat16)
        scores = sketch.score(_FakeAttnModule(layer_idx=0), hidden_states, keys, keys, None, {})
        self.assertEqual(scores.dtype, torch.bfloat16)
        self.assertEqual(scores.shape, (1, 2, 6))


class TestKVzapEdgeCases(unittest.TestCase):
    def _sketch(self, ratio, S, out_dim=2):
        sketch = KVzapSketch(compression_ratio=ratio)
        sketch.kvzap_model = _random_surrogate(4, out_dim, 1, seed=0)
        return sketch

    def test_n_kept_truncation(self):
        sketch = self._sketch(0.5, 5)
        keys = torch.randn(1, 2, 5, 4)
        out_keys, out_values = sketch.compress(
            _FakeAttnModule(layer_idx=0), torch.randn(1, 5, 4), keys, keys.clone(), None, {}
        )
        self.assertEqual(out_keys.shape[2], 2)  # int(5 * 0.5) == 2

    def test_degenerate_single_token_empties_cache(self):
        sketch = self._sketch(0.5, 1)
        keys = torch.randn(1, 2, 1, 4)
        out_keys, out_values = sketch.compress(
            _FakeAttnModule(layer_idx=0), torch.randn(1, 1, 4), keys, keys.clone(), None, {}
        )
        # kvpress-parity: n_kept = int(1 * 0.5) = 0 — empty but no exception.
        self.assertEqual(out_keys.shape, (1, 2, 0, 4))
        self.assertEqual(out_values.shape, (1, 2, 0, 4))

    def test_prepopulated_cache_misalignment_asserts(self):
        sketch = self._sketch(0.5, 7)
        keys = torch.randn(1, 2, 10, 4)  # cache longer than the current pass
        with self.assertRaises(AssertionError):
            sketch.score(_FakeAttnModule(layer_idx=0), torch.randn(1, 7, 4), keys, keys, None, {})


class TestKVzapScoreShapeAndBatch(unittest.TestCase):
    def test_score_shape_batch2(self):
        sketch = KVzapSketch(compression_ratio=0.5)
        sketch.kvzap_model = _random_surrogate(4, 2, 1, hidden_dim=3, seed=0)
        hidden_states = torch.randn(2, 7, 4)
        keys = torch.randn(2, 2, 7, 4)
        scores = sketch.score(_FakeAttnModule(layer_idx=0), hidden_states, keys, keys, None, {})
        self.assertEqual(scores.shape, (2, 2, 7))

    def test_per_batch_independent_selection(self):
        surrogate = _linear_surrogate([torch.tensor([[1.0, 0.0]])])  # H_kv = 1
        sketch = KVzapSketch(compression_ratio=0.5)
        sketch.kvzap_model = surrogate

        S = 7
        hidden_states = torch.zeros(2, S, 2)
        hidden_states[0, :, 0] = torch.arange(S, dtype=torch.float32)          # ascending
        hidden_states[1, :, 0] = torch.arange(S - 1, -1, -1, dtype=torch.float32)  # descending
        pos = torch.arange(S, dtype=torch.float32).view(1, 1, S, 1)
        keys = (pos + torch.tensor([0.0, 10.0]).view(2, 1, 1, 1)).expand(2, 1, S, 4).contiguous()

        out_keys, _ = sketch.compress(
            _FakeAttnModule(layer_idx=0, head_dim=4, num_kv_heads=1), hidden_states, keys, keys.clone(), None, {}
        )
        # n_kept = int(7 * 0.5) = 3; batch0 keeps positions [6, 5, 4], batch1 keeps [0, 1, 2].
        expected = torch.tensor([[[[6.0] * 4, [5.0] * 4, [4.0] * 4]], [[[10.0] * 4, [11.0] * 4, [12.0] * 4]]])
        self.assertTrue(torch.equal(out_keys, expected))


if __name__ == "__main__":
    unittest.main()
