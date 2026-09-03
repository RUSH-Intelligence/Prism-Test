"""Tests for VerifiedSketch (v1: deterministic head + uniform random tail).

The wrapped inner supplies ``det_fraction`` of the kept budget; the remainder
is uniformly sampled from the tokens the inner evicted (disjoint by
construction). Total kept == M == int(T * (1 - ratio)), rectangular per head.
No model loading — fake attention module + a knorm inner (needs no q_proj).
"""

from __future__ import annotations

import unittest

import torch
from torch import nn
from transformers import DynamicCache

from eval_harness.kv_compression.compressors.verified_sketch import VerifiedSketch
from eval_harness.kv_compression.compressors.knorm_sketch import KnormSketch
from eval_harness.kv_compression.registry import (
    available_kv_compressors,
    get_kv_compressor,
    get_kv_compressor_class,
)


class _FakeAttnModule(nn.Module):
    def __init__(self, num_kv_heads=2, head_dim=8, layer_idx=0):
        super().__init__()
        self.num_key_value_heads = num_kv_heads
        self.head_dim = head_dim
        self.layer_idx = layer_idx


def _recover_indices(full, subset):
    """Exact-row recovery (test oracle) — subset rows must each match one input row."""
    eq = (subset.unsqueeze(-2) == full.unsqueeze(-3)).all(-1)
    assert (eq.sum(-1) == 1).all(), "subset rows must match exactly one input row"
    return eq.float().argmax(-1)


class TestRegistry(unittest.TestCase):
    def test_registered_name(self):
        self.assertIn("verified", available_kv_compressors())
        self.assertIs(get_kv_compressor_class("verified"), VerifiedSketch)

    def test_instantiates_with_kwargs(self):
        sketch = get_kv_compressor(
            "verified", inner="knorm", det_fraction=0.6, compression_ratio=0.5
        )
        self.assertIsInstance(sketch, VerifiedSketch)
        self.assertAlmostEqual(sketch.det_fraction, 0.6)
        self.assertIsInstance(sketch._inner, KnormSketch)


class TestBudgetAndDisjointness(unittest.TestCase):
    def _run(self, T=200, ratio=0.8, det_fraction=0.75, H=2):
        torch.manual_seed(0)
        module = _FakeAttnModule(num_kv_heads=H, head_dim=8)
        keys = torch.randn(1, H, T, 8)
        values = torch.randn(1, H, T, 8)
        hidden = torch.randn(1, T, 16)
        sketch = VerifiedSketch(
            inner="knorm", det_fraction=det_fraction, compression_ratio=ratio,
            sample_seed=0,
        )
        out_keys, out_values = sketch.compress(module, hidden, keys, values, None, {})
        return keys, values, out_keys, out_values

    def test_total_budget_is_M_and_rectangular(self):
        T, ratio = 200, 0.8
        keys, _, out_keys, out_values = self._run(T=T, ratio=ratio)
        M = int(T * (1 - ratio))  # 40
        self.assertEqual(out_keys.shape, (1, 2, M, 8))
        self.assertEqual(out_values.shape, (1, 2, M, 8))

    def test_split_counts_and_disjoint_pool(self):
        # Deterministic head == knorm top-n_det; random tail drawn from the rest.
        T, ratio, det_fraction, H = 200, 0.8, 0.75, 2
        keys, values, out_keys, _ = self._run(T, ratio, det_fraction, H)
        M = int(T * (1 - ratio))                       # 40
        n_det = int(T * det_fraction * (1 - ratio))    # 30

        kept = _recover_indices(keys, out_keys)        # [1, H, M]
        knorm_top = (-keys.norm(dim=-1)).topk(n_det, dim=-1).indices  # [1, H, n_det]
        for h in range(H):
            kept_set = set(kept[0, h].tolist())
            det_set = set(knorm_top[0, h].tolist())
            self.assertEqual(len(kept_set), M, "no duplicate slots")
            self.assertTrue(det_set.issubset(kept_set), "det head must survive")
            rand_set = kept_set - det_set
            self.assertEqual(len(rand_set), M - n_det, "random tail fills the rest")
            self.assertTrue(rand_set.isdisjoint(det_set), "tail disjoint from head")

    def test_gathered_rows_are_original(self):
        keys, values, out_keys, out_values = self._run()
        idx = _recover_indices(keys, out_keys)
        gather = idx.unsqueeze(-1).expand(-1, -1, -1, keys.shape[-1])
        self.assertTrue(torch.equal(out_keys, keys.gather(2, gather)))
        self.assertTrue(torch.equal(out_values, values.gather(2, gather)))

    def test_det_fraction_one_equals_plain_inner(self):
        # n_rand == 0, so the kept SET must equal plain knorm top-M.
        T, ratio = 200, 0.8
        keys, _, out_keys, _ = self._run(T=T, ratio=ratio, det_fraction=1.0)
        M = int(T * (1 - ratio))
        plain = (-keys.norm(dim=-1)).topk(M, dim=-1).indices.sort(-1).values
        kept = _recover_indices(keys, out_keys).sort(-1).values
        self.assertTrue(torch.equal(kept, plain))

    def test_det_fraction_zero_is_pure_random(self):
        # No deterministic head: knorm keeps 0, entire budget is random.
        T, ratio = 200, 0.8
        keys, _, out_keys, _ = self._run(T=T, ratio=ratio, det_fraction=0.0)
        M = int(T * (1 - ratio))
        self.assertEqual(out_keys.shape[2], M)


class TestNoopPaths(unittest.TestCase):
    def test_zero_ratio_returns_input(self):
        module = _FakeAttnModule()
        keys = torch.randn(1, 2, 64, 8)
        values = torch.randn(1, 2, 64, 8)
        sketch = VerifiedSketch(inner="knorm", compression_ratio=0.0)
        out_keys, out_values = sketch.compress(module, torch.randn(1, 64, 16), keys, values, None, {})
        self.assertIs(out_keys, keys)
        self.assertIs(out_values, values)

    def test_below_min_tokens_returns_input(self):
        module = _FakeAttnModule()
        keys = torch.randn(1, 2, 32, 8)
        values = torch.randn(1, 2, 32, 8)
        sketch = VerifiedSketch(inner="knorm", compression_ratio=0.5, min_tokens_to_compress=64)
        out_keys, out_values = sketch.compress(module, torch.randn(1, 32, 16), keys, values, None, {})
        self.assertIs(out_keys, keys)
        self.assertIs(out_values, values)


class TestForwardHookIntegration(unittest.TestCase):
    def test_prefill_replaces_cache_with_compressed_kv(self):
        torch.manual_seed(2)
        B, H, S, D = 1, 2, 128, 8
        keys, values = torch.randn(B, H, S, D), torch.randn(B, H, S, D)
        hidden = torch.randn(B, S, 16)
        cache = DynamicCache()
        cache.update(keys.clone(), values.clone(), 0)
        module = _FakeAttnModule(num_kv_heads=H, head_dim=D, layer_idx=0)

        sketch = VerifiedSketch(inner="knorm", compression_ratio=0.5, sample_seed=0)
        output = (hidden, None)
        kwargs = {"hidden_states": hidden, "past_key_values": cache, "cache_position": torch.arange(S)}
        result = sketch.forward_hook(module, [], kwargs, output)
        self.assertIs(result, output)
        self.assertEqual(cache.layers[0].keys.shape, (B, H, S // 2, D))

    def test_decode_step_leaves_cache_untouched(self):
        torch.manual_seed(4)
        B, H, S, D = 1, 2, 128, 8
        cache = DynamicCache()
        cache.update(torch.randn(B, H, S, D), torch.randn(B, H, S, D), 0)
        cache.update(torch.randn(B, H, 1, D), torch.randn(B, H, 1, D), 0)
        before_k = cache.layers[0].keys.clone()
        module = _FakeAttnModule(num_kv_heads=H, head_dim=D, layer_idx=0)
        sketch = VerifiedSketch(inner="knorm", compression_ratio=0.5)
        hidden = torch.randn(B, 1, 16)
        kwargs = {"hidden_states": hidden, "past_key_values": cache, "cache_position": torch.tensor([S])}
        sketch.forward_hook(module, [], kwargs, (hidden, None))
        self.assertTrue(torch.equal(cache.layers[0].keys, before_k))


if __name__ == "__main__":
    unittest.main()
