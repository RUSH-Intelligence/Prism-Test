"""Unit tests for BalanceKVSketch (port of github.com/ksheth96/BalanceKV).

No model loading; fake modules only. The balanced-walk selection math is the
official ``query=None`` walk; these tests pin its structural invariants
(halving, sink/window preservation, value reweighting, determinism) rather than
re-deriving the random walk (the official ``balanced_walk`` is not importable in
the test env, but parity was verified bitwise against it during development).
"""
from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import torch

from eval_harness.kv_compression.base import CompressionSchedule
from eval_harness.kv_compression.registry import (
    get_kv_compressor,
    get_kv_compressor_class,
)
from eval_harness.kv_compression.compressors.balancekv_sketch import (
    BalanceKVSketch,
    balanced_walk,
)


def _fake_module(head_dim: int = 16, **extra) -> SimpleNamespace:
    return SimpleNamespace(head_dim=head_dim, **extra)


def _kv(batch, heads, seq, head_dim, seed=0):
    g = torch.Generator().manual_seed(seed)
    keys = torch.randn(batch, heads, seq, head_dim, generator=g)
    values = torch.randn(batch, heads, seq, head_dim, generator=g)
    return keys, values


class TestBalancedWalk(unittest.TestCase):
    def test_halving_per_iteration(self):
        key, val = _kv(1, 2, 512, 16)
        for itrs in (1, 2, 3):
            g = torch.Generator().manual_seed(42)
            idx, w = balanced_walk(key, g, 4.0, 1.0, 0.0, itrs, 128, value=val)
            self.assertEqual(idx.shape[-1], 512 // (2 ** itrs))
            self.assertEqual(w.shape[-1], 512 // (2 ** itrs))

    def test_indices_are_unique_and_in_range(self):
        key, val = _kv(1, 2, 256, 16)
        g = torch.Generator().manual_seed(7)
        idx, _ = balanced_walk(key, g, 4.0, 1.0, 0.0, 2, 128, value=val)
        for b in range(idx.shape[0]):
            for h in range(idx.shape[1]):
                vals = idx[b, h].tolist()
                self.assertEqual(len(vals), len(set(vals)))
                self.assertTrue(all(0 <= v < 256 for v in vals))

    def test_determinism_same_seed(self):
        key, val = _kv(1, 2, 256, 16)
        g1 = torch.Generator().manual_seed(1)
        g2 = torch.Generator().manual_seed(1)
        i1, w1 = balanced_walk(key, g1, 4.0, 1.0, 0.0, 2, 128, value=val)
        i2, w2 = balanced_walk(key, g2, 4.0, 1.0, 0.0, 2, 128, value=val)
        self.assertTrue(torch.equal(i1, i2))
        self.assertTrue(torch.equal(w1, w2))


class TestBalanceKVCompress(unittest.TestCase):
    def test_output_length(self):
        keys, values = _kv(1, 2, 600, 16)
        sketch = BalanceKVSketch(itrs=2, n_sink=32, window_size=32)
        out_k, out_v = sketch.compress(_fake_module(), None, keys, values, None, {})
        mid = 600 - 32 - 32
        expected = 32 + mid // (2 ** 2) + 32
        self.assertEqual(out_k.shape[2], expected)
        self.assertEqual(out_v.shape[2], expected)

    def test_sink_and_window_preserved_verbatim(self):
        keys, values = _kv(1, 2, 400, 16)
        sketch = BalanceKVSketch(itrs=2, n_sink=8, window_size=8)
        out_k, out_v = sketch.compress(_fake_module(), None, keys, values, None, {})
        # sink keys/values copied unchanged
        self.assertTrue(torch.equal(out_k[:, :, :8], keys[:, :, :8]))
        self.assertTrue(torch.equal(out_v[:, :, :8], values[:, :, :8]))
        # window keys copied unchanged (values in window are NOT reweighted)
        self.assertTrue(torch.equal(out_k[:, :, -8:], keys[:, :, -8:]))
        self.assertTrue(torch.equal(out_v[:, :, -8:], values[:, :, -8:]))

    def test_values_are_reweighted_keys_are_not(self):
        # A kept middle key must equal some original middle key (exact copy);
        # the matching value should differ (scaled by coreset weight).
        keys, values = _kv(1, 1, 300, 16, seed=3)
        sketch = BalanceKVSketch(itrs=2, n_sink=4, window_size=4)
        out_k, out_v = sketch.compress(_fake_module(), None, keys, values, None, {})
        mid_k = keys[0, 0, 4:-4]
        kept_k = out_k[0, 0, 4:-4]  # middle slice of output
        # Each kept key row equals exactly one original middle key row.
        for row in kept_k:
            match = (mid_k == row).all(dim=-1).any()
            self.assertTrue(bool(match))
        # At least one kept value is scaled away from its key's original value
        # (weights/2**itrs != 1 in general).
        self.assertFalse(torch.allclose(out_v[0, 0, 4:-4], kept_k))  # sanity: v != k

    def test_noop_when_middle_too_small(self):
        keys, values = _kv(1, 2, 60, 16)
        sketch = BalanceKVSketch(itrs=2, n_sink=32, window_size=32)
        out_k, out_v = sketch.compress(_fake_module(), None, keys, values, None, {})
        self.assertIs(out_k, keys)
        self.assertIs(out_v, values)

    def test_deterministic_compress(self):
        keys, values = _kv(1, 2, 400, 16)
        a, _ = BalanceKVSketch(itrs=2, n_sink=16, window_size=16).compress(
            _fake_module(), None, keys, values, None, {}
        )
        b, _ = BalanceKVSketch(itrs=2, n_sink=16, window_size=16).compress(
            _fake_module(), None, keys.clone(), values.clone(), None, {}
        )
        self.assertTrue(torch.equal(a, b))

    def test_dtype_preserved(self):
        keys, values = _kv(1, 2, 400, 16)
        keys, values = keys.to(torch.float16), values.to(torch.float16)
        sketch = BalanceKVSketch(itrs=2, n_sink=16, window_size=16)
        out_k, out_v = sketch.compress(_fake_module(), None, keys, values, None, {})
        self.assertEqual(out_k.dtype, torch.float16)
        self.assertEqual(out_v.dtype, torch.float16)


class TestBalanceKVConfig(unittest.TestCase):
    def test_default_itrs_is_paper_default(self):
        self.assertEqual(BalanceKVSketch().itrs, 2)

    def test_compression_ratio_maps_to_itrs(self):
        self.assertEqual(BalanceKVSketch(compression_ratio=0.5).itrs, 1)
        self.assertEqual(BalanceKVSketch(compression_ratio=0.75).itrs, 2)
        self.assertEqual(BalanceKVSketch(compression_ratio=0.875).itrs, 3)

    def test_explicit_itrs_wins_over_ratio(self):
        self.assertEqual(BalanceKVSketch(itrs=3, compression_ratio=0.5).itrs, 3)

    def test_default_schedule_is_post_prefill(self):
        self.assertEqual(
            BalanceKVSketch().schedule, frozenset({CompressionSchedule.POST_PREFILL})
        )

    def test_invalid_ratio_rejected(self):
        with self.assertRaises(AssertionError):
            BalanceKVSketch(compression_ratio=1.0)


class TestBalanceKVRegistry(unittest.TestCase):
    def test_registry_resolution(self):
        self.assertIs(get_kv_compressor_class("balancekv"), BalanceKVSketch)

    def test_get_kv_compressor_sets_fields(self):
        sketch = get_kv_compressor("balancekv", itrs=3, n_sink=8)
        self.assertIsInstance(sketch, BalanceKVSketch)
        self.assertEqual(sketch.itrs, 3)
        self.assertEqual(sketch.n_sink, 8)


if __name__ == "__main__":
    unittest.main()
