"""Tests for ObservedAttentionSketch (port of kvpress ObservedAttentionPress).

No model loading: fake attention modules + a real ``transformers.DynamicCache``
(constructed in-memory, no hub access). The reference oracle re-implements the
kvpress math in float64 in this file.
"""

from __future__ import annotations

import unittest

import torch
from torch import nn
from transformers import DynamicCache

from eval_harness.sketch.sketches.observed_attention_sketch import ObservedAttentionSketch
from eval_harness.sketch.sketches.registry import (
    available_sketches,
    get_sketch,
    get_sketch_class,
)


class _FakeAttnModule(nn.Module):
    """Minimal attention module: only what ObservedAttentionSketch touches."""

    def __init__(self, head_dim=4, layer_idx=0):
        super().__init__()
        self.head_dim = head_dim
        self.layer_idx = layer_idx


class _FakeEagerAttn(_FakeAttnModule):
    """Fake eager self_attn: forward returns (hidden_states, attention_probs)."""

    def __init__(self, head_dim=4, layer_idx=0):
        super().__init__(head_dim=head_dim, layer_idx=layer_idx)
        self.attn_probs = None

    def forward(self, hidden_states=None, past_key_values=None, cache_position=None):
        return hidden_states, self.attn_probs


def _random_causal_attentions(B, H, S, dtype=torch.float32, seed=0):
    """Valid causal attention probs: softmax(randn + causal -inf mask)."""
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(B, H, S, S, generator=generator)
    mask = torch.full((S, S), float("-inf")).triu(1)
    return torch.softmax(logits + mask, dim=-1).to(dtype)


def _observed_attention_oracle(attentions, num_key_value_heads):
    """Float64 transcription of kvpress observed_attention_press.py:43-48."""
    A = attentions.double()
    B, H_q, _, n = A.shape
    col_sum = A.sum(2)
    col_sum = col_sum / torch.arange(n, 0, -1, dtype=torch.float64)
    return col_sum.view(B, num_key_value_heads, -1, n).mean(2)


class TestObservedAttentionRegistry(unittest.TestCase):
    def test_registered_name(self):
        self.assertIn("observed_attention", available_sketches())
        self.assertIs(get_sketch_class("observed_attention"), ObservedAttentionSketch)

    def test_get_sketch_injects_compression_ratio(self):
        sketch = get_sketch("observed_attention", compression_ratio=0.25)
        self.assertIsInstance(sketch, ObservedAttentionSketch)
        self.assertAlmostEqual(sketch.compression_ratio, 0.25)


class TestObservedAttentionScore(unittest.TestCase):
    def test_zero_ratio_noop_without_attentions(self):
        sketch = ObservedAttentionSketch(compression_ratio=0.0)
        module = _FakeAttnModule(head_dim=4)
        keys = torch.randn(1, 2, 8, 4)
        values = torch.randn(1, 2, 8, 4)
        out_k, out_v = sketch.compress(module, torch.randn(1, 8, 8), keys, values, None, {})
        self.assertIs(out_k, keys)
        self.assertIs(out_v, values)

    def test_missing_attentions_asserts_with_eager_message(self):
        sketch = ObservedAttentionSketch(compression_ratio=0.5)
        module = _FakeAttnModule(head_dim=4)
        keys = torch.randn(1, 2, 8, 4)
        values = torch.randn(1, 2, 8, 4)
        with self.assertRaises(AssertionError) as ctx:
            sketch.compress(module, torch.randn(1, 8, 8), keys, values, None, {})
        self.assertIn("eager", str(ctx.exception))

    def test_hand_computed_scores_and_selection(self):
        sketch = ObservedAttentionSketch(compression_ratio=1 / 3)
        module = _FakeAttnModule(head_dim=2)
        attentions = torch.tensor(
            [[[[1.0, 0.0, 0.0], [0.6, 0.4, 0.0], [0.2, 0.3, 0.5]]]]
        )  # [1, 1, 3, 3], rows sum to 1, strictly causal zeros
        keys = torch.arange(6, dtype=torch.float32).reshape(1, 1, 3, 2)
        values = 10 + torch.arange(6, dtype=torch.float32).reshape(1, 1, 3, 2)

        scores = sketch.score(module, None, keys, values, attentions, {})
        # column sums [1.8, 0.7, 0.5] / [3, 2, 1] -> [0.6, 0.35, 0.5]
        expected = torch.tensor([[[0.6, 0.35, 0.5]]])
        torch.testing.assert_close(scores, expected, atol=1e-7, rtol=0.0)

        out_k, out_v = sketch.compress(module, None, keys, values, attentions, {})
        # n_kept = int(3 * 2/3) = 2; topk order is descending score: [0, 2]
        idx = torch.tensor([0, 2])
        self.assertTrue(torch.equal(out_k, keys[:, :, idx, :]))
        self.assertTrue(torch.equal(out_v, values[:, :, idx, :]))

    def test_gqa_blocked_head_grouping(self):
        sketch = ObservedAttentionSketch(compression_ratio=0.5)
        module = _FakeAttnModule(head_dim=4)
        B, H_q, H_kv, n = 1, 4, 2, 4
        attentions = torch.zeros(B, H_q, n, n)
        for h in (0, 1):  # all causally-valid mass on key 0
            attentions[0, h, :, 0] = 1.0
        for h in (2, 3):  # uniform over valid keys
            for i in range(n):
                attentions[0, h, i, : i + 1] = 1.0 / (i + 1)
        keys = torch.randn(B, H_kv, n, 4)
        values = torch.randn(B, H_kv, n, 4)

        scores = sketch.score(module, None, keys, values, attentions, {})
        self.assertEqual(tuple(scores.shape), (B, H_kv, n))
        # kv head 0 = mean of query heads 0,1 (blocked, NOT interleaved)
        torch.testing.assert_close(
            scores[0, 0], torch.tensor([1.0, 0.0, 0.0, 0.0]), atol=1e-6, rtol=0.0
        )
        # kv head 1 = mean of query heads 2,3: col sums [25/12, 13/12, 7/12, 1/4]
        # divided by [4, 3, 2, 1]
        torch.testing.assert_close(
            scores[0, 1],
            torch.tensor([25 / 48, 13 / 36, 7 / 24, 1 / 4]),
            atol=1e-6,
            rtol=0.0,
        )

    def test_reference_oracle_float32(self):
        sketch = ObservedAttentionSketch(compression_ratio=0.5)
        module = _FakeAttnModule(head_dim=8)
        B, H_q, H_kv, S, D = 2, 4, 2, 16, 8
        attentions = _random_causal_attentions(B, H_q, S, seed=11)
        keys = torch.randn(B, H_kv, S, D)
        values = torch.randn(B, H_kv, S, D)

        scores = sketch.score(module, None, keys, values, attentions, {})
        self.assertEqual(scores.dtype, torch.float32)
        oracle = _observed_attention_oracle(attentions, H_kv)
        torch.testing.assert_close(scores.double(), oracle, rtol=1e-5, atol=1e-7)

    def test_reference_oracle_bfloat16_division_dtype(self):
        sketch = ObservedAttentionSketch(compression_ratio=0.5)
        module = _FakeAttnModule(head_dim=8)
        B, H_q, H_kv, S, D = 2, 4, 2, 16, 8
        attentions = _random_causal_attentions(B, H_q, S, seed=11).to(torch.bfloat16)
        keys = torch.randn(B, H_kv, S, D, dtype=torch.bfloat16)
        values = torch.randn(B, H_kv, S, D, dtype=torch.bfloat16)

        scores = sketch.score(module, None, keys, values, attentions, {})
        self.assertEqual(scores.dtype, torch.bfloat16)
        oracle = _observed_attention_oracle(attentions, H_kv)
        torch.testing.assert_close(scores.double(), oracle, rtol=1e-2, atol=1e-3)

    def test_attention_key_length_mismatch_raises(self):
        sketch = ObservedAttentionSketch(compression_ratio=0.5)
        module = _FakeAttnModule(head_dim=4)
        attentions = _random_causal_attentions(1, 2, 10)
        keys = torch.randn(1, 2, 8, 4)
        values = torch.randn(1, 2, 8, 4)
        with self.assertRaisesRegex(ValueError, "prefill"):
            sketch.score(module, None, keys, values, attentions, {})


class TestObservedAttentionCompress(unittest.TestCase):
    def test_n_kept_floor(self):
        sketch = ObservedAttentionSketch(compression_ratio=0.5)
        module = _FakeAttnModule(head_dim=4)
        B, H_kv, S, D = 1, 2, 5, 4
        attentions = _random_causal_attentions(B, H_kv, S, seed=2)
        keys = torch.randn(B, H_kv, S, D)
        values = torch.randn(B, H_kv, S, D)
        out_k, out_v = sketch.compress(module, None, keys, values, attentions, {})
        self.assertEqual(tuple(out_k.shape), (B, H_kv, 2, D))  # int(2.5) == 2
        self.assertEqual(tuple(out_v.shape), (B, H_kv, 2, D))

    def test_single_token_high_ratio_keeps_zero(self):
        sketch = ObservedAttentionSketch(compression_ratio=0.5)
        module = _FakeAttnModule(head_dim=4)
        B, H_kv, S, D = 1, 2, 1, 4
        attentions = torch.ones(B, H_kv, S, S)
        keys = torch.randn(B, H_kv, S, D)
        values = torch.randn(B, H_kv, S, D)
        out_k, out_v = sketch.compress(module, None, keys, values, attentions, {})
        self.assertEqual(tuple(out_k.shape), (B, H_kv, 0, D))  # kvpress parity
        self.assertEqual(tuple(out_v.shape), (B, H_kv, 0, D))


class TestObservedAttentionHookIntegration(unittest.TestCase):
    def test_prefill_compresses_cache_and_decode_is_noop(self):
        sketch = ObservedAttentionSketch(compression_ratio=0.5)
        module = _FakeEagerAttn(head_dim=4, layer_idx=0)
        B, H_kv, S, D, hidden = 1, 2, 6, 4, 8
        torch.manual_seed(7)
        keys = torch.randn(B, H_kv, S, D)
        values = torch.randn(B, H_kv, S, D)
        cache = DynamicCache()
        cache.update(keys, values, 0)
        module.attn_probs = _random_causal_attentions(B, H_kv, S, seed=3)

        handle = module.register_forward_hook(sketch.forward_hook, with_kwargs=True)
        try:
            module(
                hidden_states=torch.randn(B, S, hidden),
                past_key_values=cache,
                cache_position=torch.arange(S),
            )
            n_kept = int(S * (1 - sketch.compression_ratio))
            self.assertEqual(cache.layers[0].keys.shape[2], n_kept)

            oracle = _observed_attention_oracle(module.attn_probs, H_kv)
            idx = oracle.topk(n_kept, dim=-1).indices.unsqueeze(-1).expand(-1, -1, -1, D)
            self.assertTrue(torch.equal(cache.layers[0].keys, keys.gather(2, idx)))
            self.assertTrue(torch.equal(cache.layers[0].values, values.gather(2, idx)))

            module.attn_probs = None  # decode never reaches the eager assert
            module(
                hidden_states=torch.randn(B, 1, hidden),
                past_key_values=cache,
                cache_position=torch.tensor([S]),
            )
            self.assertEqual(cache.layers[0].keys.shape[2], n_kept)
        finally:
            handle.remove()


if __name__ == "__main__":
    unittest.main()
