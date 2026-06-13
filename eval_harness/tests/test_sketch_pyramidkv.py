"""Tests for PyramidKVSketch (port of kvpress PyramidKVPress).

The kvpress math (SnapKVPress.score scoring inherited by PyramidKVPress, and
the PyramidKVPress.get_layer_budget formula, kvpress/presses/pyramidkv_press.py)
is re-transcribed locally as a reference oracle; no kvpress import.
"""

import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch import nn

from eval_harness.sketch.sketches.pyramidkv_sketch import PyramidKVSketch
from eval_harness.sketch.sketches.snapkv_sketch import SnapKVSketch


class _FakePyramidAttn(nn.Module):
    def __init__(
        self,
        hidden_dim=24,
        num_heads=4,
        head_dim=6,
        num_kv_heads=2,
        seed=0,
        layer_idx=0,
        num_hidden_layers=2,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.layer_idx = layer_idx
        self.config = SimpleNamespace(
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            num_hidden_layers=num_hidden_layers,
        )
        self.q_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        torch.manual_seed(seed)
        with torch.no_grad():
            self.q_proj.weight.normal_()


class _RaisingProj(nn.Module):
    def forward(self, x):
        raise AssertionError("q_proj must not be called on the attentions branch")


class _FakeCacheLayer:
    def __init__(self, keys, values):
        self.keys = keys
        self.values = values


class _FakeCache:
    def __init__(self, layers):
        self.layers = layers


def _budget_module(layer_idx, num_hidden_layers):
    return SimpleNamespace(
        layer_idx=layer_idx, config=SimpleNamespace(num_hidden_layers=num_hidden_layers)
    )


def _ref_rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _ref_repeat_kv(hidden_states, n_rep):
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def _rope_pos_emb(positions, dim, base=10000.0):
    """Real (cos, sin) of shape [1, S, dim] for the given 1-D positions."""
    half = dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))
    freqs = torch.einsum("s,d->sd", positions.to(torch.float32), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().unsqueeze(0), emb.sin().unsqueeze(0)


def _position_values(B, H_kv, S, D, offset=0.0):
    pos = torch.arange(S, dtype=torch.float32) + offset
    return pos.view(1, 1, S, 1).expand(B, H_kv, S, D).contiguous()


def _kvpress_window_attention(module, hidden_states, keys, window_size, cos, sin):
    """Verbatim transcription of kvpress SnapKVPress.compute_window_attention
    (snapkv_press.py:41-69)."""
    bsz, _, k_len, _ = keys.shape
    num_heads = module.config.num_attention_heads
    head_dim = module.head_dim
    num_key_value_groups = num_heads // module.config.num_key_value_heads

    q = module.q_proj(hidden_states[:, -window_size:])
    q = q.view(bsz, window_size, num_heads, head_dim).transpose(1, 2)

    c, s = cos[:, -window_size:], sin[:, -window_size:]
    q = (q * c.unsqueeze(1)) + (_ref_rotate_half(q) * s.unsqueeze(1))

    k = _ref_repeat_kv(keys, num_key_value_groups)
    w = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(head_dim)
    mask = torch.ones_like(w) * float("-inf")
    mask = torch.triu(mask, diagonal=k_len - window_size + 1)
    w = w + mask
    w = nn.functional.softmax(w, dim=-1, dtype=torch.float32).to(q.dtype)
    return w[..., :-window_size]


def _kvpress_score_reference(
    module, hidden_states, keys, window_size, kernel_size, cos=None, sin=None, attentions=None
):
    """Verbatim transcription of kvpress SnapKVPress.score (snapkv_press.py:71-105),
    inherited unchanged by PyramidKVPress."""
    bsz, H_kv, k_len, _ = keys.shape
    G = module.config.num_attention_heads // H_kv
    if attentions is not None:
        w = attentions[..., -window_size:, :-window_size]
    else:
        w = _kvpress_window_attention(module, hidden_states, keys, window_size, cos, sin)
    s = w.mean(dim=-2)
    s = F.avg_pool1d(s, kernel_size=kernel_size, padding=kernel_size // 2, stride=1)
    s = s.view(bsz, H_kv, G, k_len - window_size).mean(2)
    s = F.pad(s, (0, window_size), value=s.max().item())
    return s


class TestPyramidKVRegistry(unittest.TestCase):
    def test_registered_as_pyramidkv(self):
        from eval_harness.sketch.sketches.registry import (
            available_sketches,
            get_sketch,
            get_sketch_class,
        )

        self.assertIn("pyramidkv", available_sketches())
        self.assertIs(get_sketch_class("pyramidkv"), PyramidKVSketch)
        sketch = get_sketch(
            "pyramidkv", compression_ratio=0.5, window_size=4, kernel_size=3, beta=2
        )
        self.assertIsInstance(sketch, PyramidKVSketch)
        self.assertIsInstance(sketch, SnapKVSketch)
        self.assertAlmostEqual(sketch.compression_ratio, 0.5)
        self.assertEqual(sketch.window_size, 4)
        self.assertEqual(sketch.kernel_size, 3)
        self.assertEqual(sketch.beta, 2)
        self.assertFalse(sketch.uniform_budget)


class TestPyramidKVLayerBudget(unittest.TestCase):
    def test_clamped_case_value_pinned(self):
        sketch = PyramidKVSketch(compression_ratio=0.5, window_size=64, beta=20)
        budgets = [sketch.get_layer_budget(_budget_module(i, 2), 1000) for i in range(2)]
        self.assertEqual(budgets, [936, 64])
        self.assertEqual(sum(budgets), round(1000 * 2 * 0.5))

    def test_unclamped_bankers_rounding_value_pinned(self):
        sketch = PyramidKVSketch(compression_ratio=0.5, window_size=10, beta=2)
        budgets = [sketch.get_layer_budget(_budget_module(i, 5), 100) for i in range(5)]
        self.assertEqual(budgets, [75, 62, 50, 38, 25])
        self.assertEqual(sum(budgets), round(100 * 5 * 0.5))

    def test_fallback_layer_independent(self):
        sketch = PyramidKVSketch(compression_ratio=0.5, window_size=64, beta=20)
        for layer_idx in (0, 3):
            self.assertEqual(sketch.get_layer_budget(_budget_module(layer_idx, 4), 100), 50)

    def test_beta_zero_asserts(self):
        sketch = PyramidKVSketch(compression_ratio=0.5, window_size=4, beta=0)
        with self.assertRaisesRegex(AssertionError, "Beta"):
            sketch.get_layer_budget(_budget_module(0, 2), 100)

    def test_single_layer_zero_division_parity(self):
        sketch = PyramidKVSketch(compression_ratio=0.5, window_size=10, beta=2)
        with self.assertRaises(ZeroDivisionError):
            sketch.get_layer_budget(_budget_module(0, 1), 100)

    def test_uniform_budget_rectangular(self):
        sketch = PyramidKVSketch(
            compression_ratio=0.5, window_size=10, beta=2, uniform_budget=True
        )
        for q_len, expected in ((100, 50), (101, 50), (8, 4)):
            budgets = [sketch.get_layer_budget(_budget_module(i, 5), q_len) for i in range(5)]
            self.assertEqual(budgets, [expected] * 5)

    def test_budget_property_sweep(self):
        for q_len, beta, ratio, window in (
            (1000, 20, 0.5, 64),
            (100, 2, 0.5, 10),
            (512, 4, 0.25, 8),
            (2048, 8, 0.75, 16),
        ):
            sketch = PyramidKVSketch(compression_ratio=ratio, window_size=window, beta=beta)
            num_layers = 6
            budgets = [
                sketch.get_layer_budget(_budget_module(i, num_layers), q_len)
                for i in range(num_layers)
            ]
            for earlier, later in zip(budgets, budgets[1:]):
                self.assertGreaterEqual(earlier, later)
            for budget in budgets:
                self.assertGreaterEqual(budget, window)
            self.assertLessEqual(budgets[0], q_len - window)
            target = num_layers * q_len * (1 - ratio)
            self.assertLessEqual(abs(sum(budgets) - target), num_layers / 2 + 1e-6)


class TestPyramidKVScore(unittest.TestCase):
    def test_zero_ratio_noop(self):
        sketch = PyramidKVSketch(compression_ratio=0.0)
        module = _FakePyramidAttn()
        keys = torch.randn(1, 2, 5, 6)
        values = torch.randn(1, 2, 5, 6)
        with patch.object(PyramidKVSketch, "score", side_effect=AssertionError("score called")):
            out_k, out_v = sketch.compress(module, torch.randn(1, 5, 24), keys, values, None, {})
        self.assertIs(out_k, keys)
        self.assertIs(out_v, values)

    def test_score_matches_kvpress_reference(self):
        module = _FakePyramidAttn(
            hidden_dim=32, num_heads=4, head_dim=8, num_kv_heads=2, seed=3, num_hidden_layers=4
        )
        torch.manual_seed(3)
        B, S, W, D, H_kv = 1, 12, 4, 8, 2
        hidden = torch.randn(B, S, 32)
        keys = torch.randn(B, H_kv, S, D)
        values = torch.randn(B, H_kv, S, D)
        cos, sin = _rope_pos_emb(torch.arange(S), D)
        kwargs = {"position_embeddings": (cos, sin)}
        sketch = PyramidKVSketch(compression_ratio=0.5, window_size=W, kernel_size=3, beta=2)

        scores = sketch.score(module, hidden, keys, values, None, kwargs)
        ref = _kvpress_score_reference(module, hidden, keys, W, 3, cos, sin)
        self.assertEqual(scores.shape, (B, H_kv, S))
        torch.testing.assert_close(scores, ref, rtol=0.0, atol=0.0)

    def test_gqa_group_mean_axis_order(self):
        module = _FakePyramidAttn(
            hidden_dim=32, num_heads=4, head_dim=8, num_kv_heads=2, seed=5, num_hidden_layers=4
        )
        torch.manual_seed(5)
        B, S, W, D, H_kv = 1, 12, 4, 8, 2
        hidden = torch.randn(B, S, 32)
        keys = torch.randn(B, H_kv, S, D)
        cos, sin = _rope_pos_emb(torch.arange(S), D)
        kwargs = {"position_embeddings": (cos, sin)}
        sketch = PyramidKVSketch(compression_ratio=0.5, window_size=W, kernel_size=3, beta=2)

        scores = sketch.score(module, hidden, keys, None, None, kwargs)
        self.assertEqual(scores.shape, (B, H_kv, S))

        w = _kvpress_window_attention(module, hidden, keys, W, cos, sin)
        per_head = F.avg_pool1d(w.mean(dim=-2), kernel_size=3, padding=1, stride=1)
        expected_kv0 = (per_head[:, 0] + per_head[:, 1]) / 2
        expected_kv1 = (per_head[:, 2] + per_head[:, 3]) / 2
        torch.testing.assert_close(scores[:, 0, : S - W], expected_kv0, rtol=0.0, atol=1e-6)
        torch.testing.assert_close(scores[:, 1, : S - W], expected_kv1, rtol=0.0, atol=1e-6)
        self.assertFalse(torch.allclose(scores[:, 0, : S - W], scores[:, 1, : S - W]))

    def test_exact_selection_and_cross_layer_raggedness(self):
        B, H_kv, S, D = 1, 1, 8, 2
        scores = torch.tensor([[[0.0, 5.0, 1.0, 4.0, 2.0, 3.0, 9.0, 8.0]]])
        keys = _position_values(B, H_kv, S, D)
        values = _position_values(B, H_kv, S, D, offset=50.0)
        sketch = PyramidKVSketch(compression_ratio=0.5, window_size=2, kernel_size=3, beta=2)

        module_mid = SimpleNamespace(
            head_dim=D, layer_idx=1, config=SimpleNamespace(num_hidden_layers=3)
        )
        self.assertEqual(sketch.get_layer_budget(module_mid, S), 4)
        with patch.object(PyramidKVSketch, "score", return_value=scores):
            out_k, out_v = sketch.compress(module_mid, torch.randn(B, S, 4), keys, values, None, {})
        self.assertEqual(out_k.shape, (B, H_kv, 4, D))
        self.assertEqual(set(out_k[0, 0, :, 0].tolist()), {1.0, 3.0, 6.0, 7.0})
        self.assertEqual(set(out_v[0, 0, :, 0].tolist()), {51.0, 53.0, 56.0, 57.0})

        module_last = SimpleNamespace(
            head_dim=D, layer_idx=2, config=SimpleNamespace(num_hidden_layers=3)
        )
        self.assertEqual(sketch.get_layer_budget(module_last, S), 2)
        with patch.object(PyramidKVSketch, "score", return_value=scores):
            out_k_last, _ = sketch.compress(
                module_last, torch.randn(B, S, 4), keys, values, None, {}
            )
        self.assertEqual(out_k_last.shape, (B, H_kv, 2, D))
        self.assertEqual(set(out_k_last[0, 0, :, 0].tolist()), {6.0, 7.0})
        self.assertNotEqual(out_k.shape[2], out_k_last.shape[2])

    def test_attentions_branch_skips_projection(self):
        torch.manual_seed(9)
        B, S, W, D, H_q, H_kv = 1, 8, 3, 4, 4, 2
        module = _FakePyramidAttn(
            hidden_dim=16, num_heads=H_q, head_dim=D, num_kv_heads=H_kv, num_hidden_layers=4
        )
        module.q_proj = _RaisingProj()
        attentions = torch.rand(B, H_q, S, S)
        keys = torch.randn(B, H_kv, S, D)
        values = torch.randn(B, H_kv, S, D)
        sketch = PyramidKVSketch(compression_ratio=0.5, window_size=W, kernel_size=3, beta=2)

        scores = sketch.score(module, torch.randn(B, S, 16), keys, values, attentions, {})
        ref = _kvpress_score_reference(module, None, keys, W, 3, attentions=attentions)
        torch.testing.assert_close(scores, ref, rtol=0.0, atol=0.0)

    def test_window_edge_cases(self):
        module = _FakePyramidAttn(
            hidden_dim=8, num_heads=1, head_dim=4, num_kv_heads=1, seed=7, num_hidden_layers=2
        )
        cos, sin = _rope_pos_emb(torch.arange(4), 4)
        sketch = PyramidKVSketch(compression_ratio=0.5, window_size=4, kernel_size=3, beta=2)
        with self.assertRaisesRegex(AssertionError, "greater than the window size"):
            sketch.compress(
                module,
                torch.randn(1, 4, 8),
                torch.randn(1, 1, 4, 4),
                torch.randn(1, 1, 4, 4),
                None,
                {"position_embeddings": (cos, sin)},
            )

        torch.manual_seed(7)
        S, W = 5, 4
        hidden = torch.randn(1, S, 8)
        keys = torch.randn(1, 1, S, 4)
        values = _position_values(1, 1, S, 4)
        cos, sin = _rope_pos_emb(torch.arange(S), 4)
        kwargs = {"position_embeddings": (cos, sin)}
        sketch = PyramidKVSketch(compression_ratio=0.2, window_size=W, kernel_size=3, beta=2)
        self.assertEqual(sketch.get_layer_budget(module, S), 4)
        scores = sketch.score(module, hidden, keys, values, None, kwargs)
        self.assertTrue(torch.all(scores == scores.max()))
        out_k, out_v = sketch.compress(module, hidden, keys, values, None, kwargs)
        self.assertEqual(out_k.shape, (1, 1, 4, 4))
        self.assertEqual(len(set(out_v[0, 0, :, 0].tolist())), 4)

        with self.assertRaisesRegex(AssertionError, "odd"):
            PyramidKVSketch(compression_ratio=0.5, kernel_size=4)


class TestPyramidKVForwardHook(unittest.TestCase):
    def test_forward_hook_ragged_layer_lengths(self):
        B, S, W, hidden_dim, H_q, H_kv, D = 1, 16, 2, 8, 2, 1, 4
        cos, sin = _rope_pos_emb(torch.arange(S), D)
        sketch = PyramidKVSketch(compression_ratio=0.5, window_size=W, kernel_size=3, beta=2)
        torch.manual_seed(43)
        layers = [
            _FakeCacheLayer(torch.randn(B, H_kv, S, D), torch.randn(B, H_kv, S, D))
            for _ in range(2)
        ]
        cache = _FakeCache(layers)
        for layer_idx in range(2):
            module = _FakePyramidAttn(
                hidden_dim, H_q, D, H_kv, seed=layer_idx, layer_idx=layer_idx, num_hidden_layers=2
            )
            kwargs = {
                "hidden_states": torch.randn(B, S, hidden_dim),
                "past_key_values": cache,
                "cache_position": torch.arange(S),
                "position_embeddings": (cos, sin),
            }
            output = (torch.randn(B, S, hidden_dim), None)
            result = sketch.forward_hook(module, [], kwargs, output)
            self.assertIs(result, output)
        self.assertEqual(cache.layers[0].keys.shape, (B, H_kv, 12, D))
        self.assertEqual(cache.layers[0].values.shape, (B, H_kv, 12, D))
        self.assertEqual(cache.layers[1].keys.shape, (B, H_kv, 4, D))
        self.assertEqual(cache.layers[1].values.shape, (B, H_kv, 4, D))

    def test_decode_step_noop(self):
        B, S, hidden_dim, H_kv, D = 1, 8, 24, 2, 6
        module = _FakePyramidAttn(hidden_dim, 4, D, H_kv, seed=1, num_hidden_layers=2)
        keys = torch.randn(B, H_kv, S, D)
        values = torch.randn(B, H_kv, S, D)
        cache = _FakeCache([_FakeCacheLayer(keys, values)])
        sketch = PyramidKVSketch(compression_ratio=0.5, window_size=2, kernel_size=3, beta=2)
        kwargs = {
            "hidden_states": torch.randn(B, 1, hidden_dim),
            "past_key_values": cache,
            "cache_position": torch.tensor([S + 3]),
        }
        output = (torch.randn(B, 1, hidden_dim), None)
        result = sketch.forward_hook(module, [], kwargs, output)
        self.assertIs(result, output)
        self.assertIs(cache.layers[0].keys, keys)
        self.assertIs(cache.layers[0].values, values)


class TestPyramidKVAttnImplGuard(unittest.TestCase):
    @staticmethod
    def _fake_model(attn_impl):
        return SimpleNamespace(config=SimpleNamespace(_attn_implementation=attn_impl))

    def test_sdpa_raises(self):
        sketch = PyramidKVSketch(compression_ratio=0.5)
        with self.assertRaisesRegex(ValueError, "flash_attention_2"):
            sketch.post_init_from_model(self._fake_model("sdpa"))

    def test_eager_raises(self):
        sketch = PyramidKVSketch(compression_ratio=0.5)
        with self.assertRaises(ValueError):
            sketch.post_init_from_model(self._fake_model("eager"))

    def test_flash_attention_2_passes(self):
        sketch = PyramidKVSketch(compression_ratio=0.5)
        sketch.post_init_from_model(self._fake_model("flash_attention_2"))

    def test_uniform_budget_passes_under_sdpa(self):
        sketch = PyramidKVSketch(compression_ratio=0.5, uniform_budget=True)
        sketch.post_init_from_model(self._fake_model("sdpa"))

    def test_zero_ratio_passes_under_sdpa(self):
        sketch = PyramidKVSketch(compression_ratio=0.0)
        sketch.post_init_from_model(self._fake_model("sdpa"))


if __name__ == "__main__":
    unittest.main()
