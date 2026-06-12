"""Tests for RandomSketchRidgeSketch (kvpress RandomSketchPress port).

Upstream fact pinned here: ``RandomSketchPress`` overrides only
``_compute_tau`` (random_sketch_press.py:15-25), but ``RidgePress.compress``
calls ``_compute_key_ridge_tau`` (ridge_press.py:733) and nothing in the
kvpress checkout ever calls ``_compute_tau`` — the override is dead code, so
the press is bitwise-identical to ``RidgePress`` and consumes no randomness.
The port replicates this faithfully; these tests would fail if the random
override were ever wired into the live scoring path (or vice versa).
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
from torch import nn

from eval_harness.sketch.sketches.random_sketch import RandomSketch
from eval_harness.sketch.sketches.registry import (
    available_sketches,
    get_sketch,
    get_sketch_class,
)
from eval_harness.sketch.sketches.ridge_sketch import (
    RandomSketchRidgeSketch,
    RidgeSketch,
)


class _FakeAttnModule(nn.Module):
    def __init__(self, hidden_dim=32, num_heads=4, head_dim=8, num_kv_heads=2, seed=0):
        super().__init__()
        self.num_heads = num_heads
        self.num_key_value_heads = num_kv_heads
        self.head_dim = head_dim
        self.layer_idx = 0
        torch.manual_seed(seed)
        self.q_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)


class _NoQProjModule(nn.Module):
    def __init__(self, head_dim=8):
        super().__init__()
        self.head_dim = head_dim
        self.layer_idx = 0


def _rand_inputs(B=1, H_kv=2, T=80, D=8, hidden_dim=32, seed=0):
    torch.manual_seed(seed)
    keys = torch.randn(B, H_kv, T, D)
    values = torch.randn(B, H_kv, T, D)
    hidden = torch.randn(B, T, hidden_dim)
    return keys, values, hidden


class TestRandomSketchPressRegistry(unittest.TestCase):
    def test_registered_under_assigned_name(self):
        self.assertIn("random_sketch_press", available_sketches())
        self.assertIs(get_sketch_class("random_sketch_press"), RandomSketchRidgeSketch)

    def test_distinct_from_ridge_and_upstream_random_press_port(self):
        cls = get_sketch_class("random_sketch_press")
        self.assertTrue(issubclass(cls, RidgeSketch))
        self.assertIsNot(cls, RidgeSketch)
        self.assertIsNot(cls, RandomSketch)
        self.assertIs(get_sketch_class("random"), RandomSketch)

    def test_get_sketch_accepts_compression_ratio_kwarg(self):
        sketch = get_sketch("random_sketch_press", compression_ratio=0.4)
        self.assertIsInstance(sketch, RandomSketchRidgeSketch)
        self.assertAlmostEqual(sketch.compression_ratio, 0.4)


class TestDeadOverrideFaithfulness(unittest.TestCase):
    def _sketch(self):
        return RandomSketchRidgeSketch(
            compression_ratio=0.5, sink_size=4, local_size=8,
            min_tokens_to_compress=0, query_aware=False,
        )

    def test_compute_tau_is_dead_code(self):
        # Upstream, patching RandomSketchPress._compute_tau to raise leaves
        # RidgePress.compress green because compress calls
        # _compute_key_ridge_tau; the port must preserve that.
        keys, values, hidden = _rand_inputs()
        sketch = self._sketch()
        with patch.object(sketch, "_compute_tau", side_effect=RuntimeError("dead code ran")):
            out_k, out_v = sketch.compress(_NoQProjModule(), hidden, keys, values, None, {})
        self.assertEqual(out_k.shape[2], 40)
        self.assertEqual(out_v.shape[2], 40)

    def test_compute_key_ridge_tau_is_the_live_path(self):
        keys, values, hidden = _rand_inputs()
        sketch = self._sketch()
        with patch.object(sketch, "_compute_key_ridge_tau", side_effect=RuntimeError("live")):
            with self.assertRaisesRegex(RuntimeError, "live"):
                sketch.compress(_NoQProjModule(), hidden, keys, values, None, {})

    def test_no_randomness_is_consumed(self):
        keys, values, hidden = _rand_inputs(seed=4)
        sketch = self._sketch()
        torch.manual_seed(1234)
        state_before = torch.get_rng_state()
        sketch.compress(_NoQProjModule(), hidden, keys, values, None, {})
        self.assertTrue(torch.equal(state_before, torch.get_rng_state()))

    def test_bitwise_identical_to_ridge_sketch_query_aware(self):
        module = _FakeAttnModule(seed=1)
        keys, values, hidden = _rand_inputs(T=128, seed=2)
        ridge = RidgeSketch(compression_ratio=0.5)
        rand = RandomSketchRidgeSketch(compression_ratio=0.5)
        ridge_k, ridge_v = ridge.compress(module, hidden, keys, values, None, {})
        rand_k, rand_v = rand.compress(module, hidden, keys, values, None, {})
        self.assertEqual(rand_k.shape[2], 64)
        self.assertTrue(torch.equal(rand_k, ridge_k))
        self.assertTrue(torch.equal(rand_v, ridge_v))

    def test_bitwise_identical_to_ridge_sketch_tau_only(self):
        keys, values, hidden = _rand_inputs(T=96, seed=3)
        kwargs = dict(compression_ratio=0.5, sink_size=4, local_size=8,
                      min_tokens_to_compress=0, query_aware=False)
        ridge_k, ridge_v = RidgeSketch(**kwargs).compress(
            _NoQProjModule(), hidden, keys, values, None, {})
        rand_k, rand_v = RandomSketchRidgeSketch(**kwargs).compress(
            _NoQProjModule(), hidden, keys, values, None, {})
        self.assertTrue(torch.equal(rand_k, ridge_k))
        self.assertTrue(torch.equal(rand_v, ridge_v))


class TestComputeTauOverrideTranscription(unittest.TestCase):
    def test_uniform_random_scores_shape_dtype_range(self):
        sketch = RandomSketchRidgeSketch(compression_ratio=0.5)
        keys = torch.randn(2, 3, 16, 4)
        tau = sketch._compute_tau(keys)
        self.assertEqual(tau.shape, (2, 3, 16))
        self.assertEqual(tau.dtype, keys.dtype)
        self.assertTrue((tau >= 0).all() and (tau < 1).all())

    def test_empty_keys_return_zeros(self):
        sketch = RandomSketchRidgeSketch(compression_ratio=0.5)
        keys = torch.randn(2, 3, 0, 4)
        tau = sketch._compute_tau(keys)
        self.assertEqual(tau.shape, (2, 3, 0))


class TestInheritedRidgeBehavior(unittest.TestCase):
    def test_zero_ratio_noop_and_none_raises(self):
        keys, values, hidden = _rand_inputs(T=100)
        sketch = RandomSketchRidgeSketch(compression_ratio=0.0)
        out_k, out_v = sketch.compress(_NoQProjModule(), hidden, keys, values, None, {})
        self.assertIs(out_k, keys)
        self.assertIs(out_v, values)
        with self.assertRaisesRegex(ValueError, "compression_ratio"):
            RandomSketchRidgeSketch().compress(_NoQProjModule(), hidden, keys, values, None, {})

    def test_min_tokens_gate(self):
        sketch = RandomSketchRidgeSketch(compression_ratio=0.5)
        keys63, values63, hidden63 = _rand_inputs(T=63)
        out_k, _ = sketch.compress(_NoQProjModule(), hidden63, keys63, values63, None, {})
        self.assertEqual(out_k.shape[2], 63)

        keys64, values64, hidden64 = _rand_inputs(T=64)
        out_k, out_v = sketch.compress(_NoQProjModule(), hidden64, keys64, values64, None, {})
        self.assertEqual(out_k.shape[2], 32)
        self.assertTrue(torch.equal(out_k, torch.cat([keys64[:, :, :4], keys64[:, :, 36:]], dim=2)))

    def test_over_keep_edge_exceeds_nominal_budget(self):
        keys, values, hidden = _rand_inputs(T=64)
        sketch = RandomSketchRidgeSketch(compression_ratio=0.9)
        out_k, out_v = sketch.compress(_NoQProjModule(), hidden, keys, values, None, {})
        self.assertEqual(out_k.shape[2], 32)
        self.assertGreater(out_k.shape[2], int(64 * (1.0 - 0.9)))
        self.assertTrue(torch.equal(out_k, torch.cat([keys[:, :, :4], keys[:, :, 36:]], dim=2)))
        self.assertTrue(torch.equal(out_v, torch.cat([values[:, :, :4], values[:, :, 36:]], dim=2)))

    def test_budget_arithmetic_and_temporal_order(self):
        module = _FakeAttnModule(seed=5)
        keys, values, hidden = _rand_inputs(T=100, seed=6)
        sketch = RandomSketchRidgeSketch(compression_ratio=0.5)
        out_k, out_v = sketch.compress(module, hidden, keys, values, None, {})
        self.assertEqual(out_k.shape[2], 50)
        self.assertTrue(torch.equal(out_k[:, :, :4], keys[:, :, :4]))
        self.assertTrue(torch.equal(out_k[:, :, -28:], keys[:, :, 72:]))
        self.assertTrue(torch.equal(out_v[:, :, :4], values[:, :, :4]))
        self.assertTrue(torch.equal(out_v[:, :, -28:], values[:, :, 72:]))
        for h in range(keys.shape[1]):
            mid_in = keys[0, h, 4:72]
            mid_out = out_k[0, h, 4:22]
            eq = (mid_out.unsqueeze(1) == mid_in.unsqueeze(0)).all(dim=-1)
            self.assertTrue((eq.sum(dim=1) == 1).all())
            pos = eq.float().argmax(dim=1)
            self.assertTrue((pos[1:] > pos[:-1]).all())

    def test_rectangular_output_across_layers(self):
        sketch = RandomSketchRidgeSketch(compression_ratio=0.5, query_aware=False)
        lengths = set()
        for seed in (0, 1):
            keys, values, hidden = _rand_inputs(T=100, seed=seed)
            out_k, _ = sketch.compress(_NoQProjModule(), hidden, keys, values, None, {})
            lengths.add(out_k.shape[2])
        self.assertEqual(lengths, {50})

    def test_bf16_passthrough(self):
        module = _FakeAttnModule(seed=7)
        module.to(torch.bfloat16)
        keys, values, hidden = _rand_inputs(T=80, seed=8)
        keys, values, hidden = keys.bfloat16(), values.bfloat16(), hidden.bfloat16()
        sketch = RandomSketchRidgeSketch(compression_ratio=0.5)
        out_k, out_v = sketch.compress(module, hidden, keys, values, None, {})
        self.assertEqual(out_k.shape[2], 40)
        self.assertEqual(out_k.dtype, torch.bfloat16)
        self.assertEqual(out_v.dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
