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

from eval_harness.kv_compression.compressors.random_sketch import RandomSketch
from eval_harness.kv_compression.registry import (
    available_kv_compressors,
    get_kv_compressor,
    get_kv_compressor_class,
)
from eval_harness.kv_compression.compressors.ridge_sketch import (
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
        self.assertIn("random_sketch_press", available_kv_compressors())
        self.assertIs(get_kv_compressor_class("random_sketch_press"), RandomSketchRidgeSketch)

    def test_distinct_from_ridge_and_upstream_random_press_port(self):
        cls = get_kv_compressor_class("random_sketch_press")
        self.assertTrue(issubclass(cls, RidgeSketch))
        self.assertIsNot(cls, RidgeSketch)
        self.assertIsNot(cls, RandomSketch)
        self.assertIs(get_kv_compressor_class("random"), RandomSketch)

    def test_get_kv_compressor_accepts_compression_ratio_kwarg(self):
        sketch = get_kv_compressor("random_sketch_press", compression_ratio=0.4)
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
        # Keep the real mid-selection query-aware path under the 8/64 defaults
        # (so tau/omega scoring actually runs, not the degenerate keep_mid==0
        # branch). T=256, ratio=0.5: sink=8, local=64, mid_len=120,
        # keep_total=128, keep_mid=128-72=56 (0<56<120) -> out = 8+56+64 = 128.
        module = _FakeAttnModule(seed=1)
        keys, values, hidden = _rand_inputs(T=256, seed=2)
        ridge = RidgeSketch(compression_ratio=0.5)
        rand = RandomSketchRidgeSketch(compression_ratio=0.5)
        ridge_k, ridge_v = ridge.compress(module, hidden, keys, values, None, {})
        rand_k, rand_v = rand.compress(module, hidden, keys, values, None, {})
        self.assertEqual(rand_k.shape[2], 128)
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
        # Below the min_tokens_to_compress=64 gate -> no-op.
        keys63, values63, hidden63 = _rand_inputs(T=63)
        out_k, _ = sketch.compress(_NoQProjModule(), hidden63, keys63, values63, None, {})
        self.assertEqual(out_k.shape[2], 63)

        # Above the gate, compression fires. Under the 8/64 defaults T must be
        # >72 for the mid window to be non-empty; at T=80, ratio=0.5 the budget
        # int(80*0.5)=40 < sink+local=72 so keep_mid=max(40-72,0)=0 -> the
        # keep_mid<=0 branch returns [sink | local] = 8 + 64 = 72 tokens (the
        # same deterministic branch the old 4/28 T=64 case exercised).
        T = 80
        keys80, values80, hidden80 = _rand_inputs(T=T)
        out_k, out_v = sketch.compress(_NoQProjModule(), hidden80, keys80, values80, None, {})
        sink, local = sketch.sink_size, sketch.local_size
        mid_end = T - local
        self.assertEqual(out_k.shape[2], sink + local)
        self.assertTrue(torch.equal(out_k, torch.cat([keys80[:, :, :sink], keys80[:, :, mid_end:]], dim=2)))

    def test_over_keep_edge_exceeds_nominal_budget(self):
        # 8/64 defaults at T=80, ratio=0.9: sink=8, local=64, mid_end=16,
        # mid_len=8>0, keep_total=int(80*0.1)=8, keep_mid=max(8-72,0)=0 ->
        # keep_mid<=0 branch returns [sink | local] = 8 + 64 = 72, which EXCEEDS
        # the nominal budget int(80*0.1)=8 (the over-keep edge).
        T = 80
        keys, values, hidden = _rand_inputs(T=T)
        sketch = RandomSketchRidgeSketch(compression_ratio=0.9)
        out_k, out_v = sketch.compress(_NoQProjModule(), hidden, keys, values, None, {})
        sink, local = sketch.sink_size, sketch.local_size
        mid_end = T - local
        self.assertEqual(out_k.shape[2], sink + local)
        self.assertGreater(out_k.shape[2], int(T * (1.0 - 0.9)))
        self.assertTrue(torch.equal(out_k, torch.cat([keys[:, :, :sink], keys[:, :, mid_end:]], dim=2)))
        self.assertTrue(torch.equal(out_v, torch.cat([values[:, :, :sink], values[:, :, mid_end:]], dim=2)))

    def test_budget_arithmetic_and_temporal_order(self):
        # Keep the real mid-selection path (0 < keep_mid < mid_len) under the
        # 8/64 defaults so sink/local pinning AND temporal ordering of the
        # selected mid tokens are actually exercised. T=200, ratio=0.5:
        # sink=8, local=64, mid_start=8, mid_end=136, mid_len=128,
        # keep_total=100, keep_mid=100-72=28 (0<28<128) -> out = 8+28+64 = 100.
        T = 200
        module = _FakeAttnModule(seed=5)
        keys, values, hidden = _rand_inputs(T=T, seed=6)
        sketch = RandomSketchRidgeSketch(compression_ratio=0.5)
        out_k, out_v = sketch.compress(module, hidden, keys, values, None, {})
        sink, local = sketch.sink_size, sketch.local_size
        mid_start, mid_end = sink, T - local
        keep_mid = int(T * 0.5) - sink - local  # 28
        self.assertEqual(out_k.shape[2], sink + keep_mid + local)
        self.assertTrue(torch.equal(out_k[:, :, :sink], keys[:, :, :sink]))
        self.assertTrue(torch.equal(out_k[:, :, -local:], keys[:, :, mid_end:]))
        self.assertTrue(torch.equal(out_v[:, :, :sink], values[:, :, :sink]))
        self.assertTrue(torch.equal(out_v[:, :, -local:], values[:, :, mid_end:]))
        for h in range(keys.shape[1]):
            mid_in = keys[0, h, mid_start:mid_end]
            mid_out = out_k[0, h, sink:sink + keep_mid]
            eq = (mid_out.unsqueeze(1) == mid_in.unsqueeze(0)).all(dim=-1)
            self.assertTrue((eq.sum(dim=1) == 1).all())
            pos = eq.float().argmax(dim=1)
            self.assertTrue((pos[1:] > pos[:-1]).all())

    def test_rectangular_output_across_layers(self):
        # Keep the real mid-selection path (0 < keep_mid < mid_len) under the
        # 8/64 defaults so per-head selection genuinely diverges yet kept counts
        # stay rectangular. T=200, ratio=0.5: keep_mid=int(200*0.5)-8-64=28
        # (0<28<128) -> out = 8+28+64 = 100 tokens for every seed/layer.
        sketch = RandomSketchRidgeSketch(compression_ratio=0.5, query_aware=False)
        lengths = set()
        for seed in (0, 1):
            keys, values, hidden = _rand_inputs(T=200, seed=seed)
            out_k, _ = sketch.compress(_NoQProjModule(), hidden, keys, values, None, {})
            lengths.add(out_k.shape[2])
        self.assertEqual(lengths, {100})

    def test_bf16_passthrough(self):
        # Keep the real mid-selection path (0 < keep_mid < mid_len) under the
        # 8/64 defaults so the bf16-sensitive scoring/gather code runs. T=160,
        # ratio=0.5: sink=8, local=64, mid_len=88, keep_total=80,
        # keep_mid=80-72=8 (0<8<88) -> out = 8 + 8 + 64 = 80 tokens.
        module = _FakeAttnModule(seed=7)
        module.to(torch.bfloat16)
        keys, values, hidden = _rand_inputs(T=160, seed=8)
        keys, values, hidden = keys.bfloat16(), values.bfloat16(), hidden.bfloat16()
        sketch = RandomSketchRidgeSketch(compression_ratio=0.5)
        out_k, out_v = sketch.compress(module, hidden, keys, values, None, {})
        self.assertEqual(out_k.shape[2], 80)
        self.assertEqual(out_k.dtype, torch.bfloat16)
        self.assertEqual(out_v.dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
