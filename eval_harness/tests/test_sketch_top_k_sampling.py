"""Tests for TopKSamplingSketch (top-k core + uniform random tail baseline).

Semantics pinned here:
- per-head keep-budget ``n_kept = int(T * (1 - compression_ratio))`` (framework
  convention, same as ``ScorerKVCompressor.compress``);
- deterministic core = top ``round(n_kept * top_frac)`` tokens by ``-||k||``
  (KnormSketch scoring), always contained in the kept set;
- tail = uniform sample WITHOUT replacement from the non-core remainder;
- seeded per-call generator: same seed => identical selection, no global RNG
  consumption; ``top_frac=1.0`` reduces to pure knorm top-k.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from eval_harness.kv_compression.compressors.knorm_sketch import KnormSketch
from eval_harness.kv_compression.compressors.top_k_sampling_sketch import TopKSamplingSketch
from eval_harness.kv_compression.registry import (
    available_kv_compressors,
    get_kv_compressor,
    get_kv_compressor_class,
)


def _fake_module(head_dim=8):
    # No q_proj / rotary_emb / position_embeddings needed: the sketch scores
    # keys only, so a bare namespace mirrors hybrid (NemotronH-style) modules.
    return SimpleNamespace(head_dim=head_dim, layer_idx=0)


def _rand_inputs(B=1, H_kv=2, T=100, D=8, hidden_dim=32, seed=0):
    torch.manual_seed(seed)
    keys = torch.randn(B, H_kv, T, D)
    values = torch.randn(B, H_kv, T, D)
    hidden = torch.randn(B, T, hidden_dim)
    return keys, values, hidden


def _kept_position_sets(keys_in, out_k):
    """Map kept output rows back to input positions, per (batch, head)."""
    sets = {}
    B, H = keys_in.shape[0], keys_in.shape[1]
    for b in range(B):
        for h in range(H):
            eq = (out_k[b, h].unsqueeze(1) == keys_in[b, h].unsqueeze(0)).all(dim=-1)
            # Every output row must match exactly one input row.
            assert (eq.sum(dim=1) == 1).all()
            sets[(b, h)] = set(eq.float().argmax(dim=1).tolist())
    return sets


class TestRegistry(unittest.TestCase):
    def test_registered(self):
        self.assertIn("top_k_sampling", available_kv_compressors())
        self.assertIs(get_kv_compressor_class("top_k_sampling"), TopKSamplingSketch)
        self.assertIs(get_kv_compressor_class("top_k_sampling_sketch"), TopKSamplingSketch)

    def test_constructs_with_kwargs(self):
        sketch = get_kv_compressor(
            "top_k_sampling", compression_ratio=0.8, top_frac=0.5, seed=7
        )
        self.assertIsInstance(sketch, TopKSamplingSketch)
        self.assertAlmostEqual(sketch.compression_ratio, 0.8)
        self.assertAlmostEqual(sketch.top_frac, 0.5)
        self.assertEqual(sketch.seed, 7)

    def test_defaults(self):
        sketch = TopKSamplingSketch()
        self.assertAlmostEqual(sketch.top_frac, 0.75)
        self.assertEqual(sketch.seed, 42)

    def test_top_frac_validated(self):
        with self.assertRaises(AssertionError):
            TopKSamplingSketch(top_frac=1.5)
        with self.assertRaises(AssertionError):
            TopKSamplingSketch(top_frac=-0.1)


class TestBudget(unittest.TestCase):
    def test_budget_arithmetic(self):
        for T, r in [(100, 0.8), (160, 0.5), (97, 0.8), (64, 0.25), (129, 0.9)]:
            keys, values, hidden = _rand_inputs(T=T, seed=T)
            sketch = TopKSamplingSketch(compression_ratio=r)
            out_k, out_v = sketch.compress(_fake_module(), hidden, keys, values, None, {})
            n_kept = int(T * (1 - r))
            self.assertEqual(out_k.shape[2], n_kept, f"T={T}, r={r}")
            self.assertEqual(out_v.shape[2], n_kept, f"T={T}, r={r}")

    def test_zero_ratio_noop(self):
        keys, values, hidden = _rand_inputs()
        sketch = TopKSamplingSketch(compression_ratio=0.0)
        out_k, out_v = sketch.compress(_fake_module(), hidden, keys, values, None, {})
        self.assertIs(out_k, keys)
        self.assertIs(out_v, values)

    def test_values_gathered_at_same_positions_as_keys(self):
        keys, values, hidden = _rand_inputs(T=120, seed=3)
        sketch = TopKSamplingSketch(compression_ratio=0.8)
        out_k, out_v = sketch.compress(_fake_module(), hidden, keys, values, None, {})
        key_sets = _kept_position_sets(keys, out_k)
        val_sets = _kept_position_sets(values, out_v)
        self.assertEqual(key_sets, val_sets)


class TestSelectionSemantics(unittest.TestCase):
    def test_core_always_kept(self):
        keys, values, hidden = _rand_inputs(B=2, H_kv=3, T=100, seed=5)
        r, top_frac = 0.8, 0.75
        sketch = TopKSamplingSketch(compression_ratio=r, top_frac=top_frac, seed=11)
        out_k, _ = sketch.compress(_fake_module(), hidden, keys, values, None, {})
        n_kept = int(100 * (1 - r))            # 20
        n_top = int(round(n_kept * top_frac))  # 15
        knorm_top = (-keys.norm(dim=-1)).topk(n_top, dim=-1).indices
        kept = _kept_position_sets(keys, out_k)
        for b in range(2):
            for h in range(3):
                core = set(knorm_top[b, h].tolist())
                self.assertTrue(core.issubset(kept[(b, h)]))
                # Tail slots exist and are disjoint from the core.
                self.assertEqual(len(kept[(b, h)]), n_kept)
                self.assertEqual(len(kept[(b, h)] - core), n_kept - n_top)

    def test_same_seed_identical_selection(self):
        keys, values, hidden = _rand_inputs(T=100, seed=6)
        a = TopKSamplingSketch(compression_ratio=0.8, seed=42)
        b = TopKSamplingSketch(compression_ratio=0.8, seed=42)
        ak, av = a.compress(_fake_module(), hidden, keys, values, None, {})
        bk, bv = b.compress(_fake_module(), hidden, keys, values, None, {})
        self.assertTrue(torch.equal(ak, bk))
        self.assertTrue(torch.equal(av, bv))
        # Per-call determinism: the same instance repeats itself too.
        ak2, _ = a.compress(_fake_module(), hidden, keys, values, None, {})
        self.assertTrue(torch.equal(ak, ak2))

    def test_different_seeds_differ_in_tail_only(self):
        keys, values, hidden = _rand_inputs(T=400, seed=7)
        r, top_frac = 0.8, 0.5
        n_kept = int(400 * (1 - r))            # 80
        n_top = int(round(n_kept * top_frac))  # 40
        knorm_top = (-keys.norm(dim=-1)).topk(n_top, dim=-1).indices
        kept = {}
        for seed in (1, 2):
            sketch = TopKSamplingSketch(compression_ratio=r, top_frac=top_frac, seed=seed)
            out_k, _ = sketch.compress(_fake_module(), hidden, keys, values, None, {})
            kept[seed] = _kept_position_sets(keys, out_k)
        core = set(knorm_top[0, 0].tolist())
        self.assertTrue(core.issubset(kept[1][(0, 0)]))
        self.assertTrue(core.issubset(kept[2][(0, 0)]))
        # With 40 tail slots from 360 candidates, distinct seeds must differ.
        self.assertNotEqual(kept[1][(0, 0)], kept[2][(0, 0)])

    def test_top_frac_one_equals_knorm(self):
        keys, values, hidden = _rand_inputs(T=100, seed=8)
        module = _fake_module()
        sketch = TopKSamplingSketch(compression_ratio=0.8, top_frac=1.0)
        knorm = KnormSketch(compression_ratio=0.8)
        tk, _ = sketch.compress(module, hidden, keys, values, None, {})
        kk, _ = knorm.compress(module, hidden, keys, values, None, {})
        self.assertEqual(_kept_position_sets(keys, tk), _kept_position_sets(keys, kk))

    def test_top_frac_zero_is_pure_uniform(self):
        keys, values, hidden = _rand_inputs(T=100, seed=9)
        sketch = TopKSamplingSketch(compression_ratio=0.8, top_frac=0.0, seed=3)
        out_k, _ = sketch.compress(_fake_module(), hidden, keys, values, None, {})
        self.assertEqual(out_k.shape[2], int(100 * (1 - 0.8)))  # 19: float convention
        # Selection must not depend on key norms: scaling keys leaves the
        # kept positions unchanged (values carry the original rows).
        out_k_scaled, _ = sketch.compress(
            _fake_module(), hidden, keys * 100.0, values, None, {}
        )
        self.assertEqual(
            _kept_position_sets(keys * 100.0, out_k_scaled),
            _kept_position_sets(keys, out_k),
        )

    def test_layers_sample_independently(self):
        # The per-call seed folds in module.layer_idx, so two layers with the
        # same cache shape must NOT keep the identical "random" tail.
        keys, values, hidden = _rand_inputs(T=400, seed=15)
        sketch = TopKSamplingSketch(compression_ratio=0.8, top_frac=0.0, seed=42)
        kept = {}
        for layer_idx in (0, 1):
            module = SimpleNamespace(head_dim=8, layer_idx=layer_idx)
            out_k, _ = sketch.compress(module, hidden, keys, values, None, {})
            kept[layer_idx] = _kept_position_sets(keys, out_k)
        self.assertNotEqual(kept[0][(0, 0)], kept[1][(0, 0)])

    def test_missing_layer_idx_falls_back_to_seed(self):
        keys, values, hidden = _rand_inputs(T=100, seed=16)
        sketch = TopKSamplingSketch(compression_ratio=0.8, seed=42)
        no_idx = SimpleNamespace(head_dim=8)          # hybrid-style bare module
        layer0 = SimpleNamespace(head_dim=8, layer_idx=0)
        a, _ = sketch.compress(no_idx, hidden, keys, values, None, {})
        b, _ = sketch.compress(layer0, hidden, keys, values, None, {})
        self.assertTrue(torch.equal(a, b))

    def test_heads_sample_independently(self):
        keys, values, hidden = _rand_inputs(B=1, H_kv=4, T=400, seed=10)
        sketch = TopKSamplingSketch(compression_ratio=0.8, top_frac=0.0, seed=5)
        out_k, _ = sketch.compress(_fake_module(), hidden, keys, values, None, {})
        kept = _kept_position_sets(keys, out_k)
        head_sets = [frozenset(kept[(0, h)]) for h in range(4)]
        self.assertGreater(len(set(head_sets)), 1)


class TestDeterminismAndDtype(unittest.TestCase):
    def test_no_global_rng_consumption_when_seeded(self):
        keys, values, hidden = _rand_inputs(seed=11)
        sketch = TopKSamplingSketch(compression_ratio=0.8, seed=42)
        torch.manual_seed(1234)
        state_before = torch.get_rng_state()
        sketch.compress(_fake_module(), hidden, keys, values, None, {})
        self.assertTrue(torch.equal(state_before, torch.get_rng_state()))

    def test_seed_none_uses_global_rng(self):
        keys, values, hidden = _rand_inputs(seed=12)
        sketch = TopKSamplingSketch(compression_ratio=0.8, seed=None)
        torch.manual_seed(1234)
        state_before = torch.get_rng_state()
        sketch.compress(_fake_module(), hidden, keys, values, None, {})
        self.assertFalse(torch.equal(state_before, torch.get_rng_state()))

    def test_bf16_passthrough(self):
        keys, values, hidden = _rand_inputs(T=160, seed=13)
        keys, values = keys.bfloat16(), values.bfloat16()
        sketch = TopKSamplingSketch(compression_ratio=0.8)
        out_k, out_v = sketch.compress(_fake_module(), hidden.bfloat16(), keys, values, None, {})
        self.assertEqual(out_k.shape[2], int(160 * (1 - 0.8)))  # 31: float convention
        self.assertEqual(out_k.dtype, torch.bfloat16)
        self.assertEqual(out_v.dtype, torch.bfloat16)

    def test_output_contiguous(self):
        keys, values, hidden = _rand_inputs(seed=14)
        sketch = TopKSamplingSketch(compression_ratio=0.8)
        out_k, out_v = sketch.compress(_fake_module(), hidden, keys, values, None, {})
        self.assertTrue(out_k.is_contiguous())
        self.assertTrue(out_v.is_contiguous())


if __name__ == "__main__":
    unittest.main()
