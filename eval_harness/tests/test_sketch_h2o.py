"""Tests for H2OSketch (Heavy Hitter Oracle, arXiv:2306.14048).

No model loading: fake attention modules + a real ``transformers.DynamicCache``
(constructed in-memory, no hub access). The reference oracle re-implements the
H2O scoring math in float64 in this file.

H2O differs from ``observed_attention`` in two ways verified here: it uses the
raw accumulated column sum (no causal-attendee normalization), and it
force-keeps the most recent ``window_size`` tokens.
"""

from __future__ import annotations

import unittest

import torch
from torch import nn
from torch.nn import functional as F
from transformers import DynamicCache

from eval_harness.kv_compression.compressors.h2o_sketch import H2OSketch
from eval_harness.kv_compression.registry import (
    available_kv_compressors,
    get_kv_compressor,
    get_kv_compressor_class,
)


class _FakeAttnModule(nn.Module):
    """Minimal attention module: only what H2OSketch touches."""

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


def _h2o_oracle(attentions, num_key_value_heads, window_size):
    """Float64 transcription of H2OSketch.score."""
    A = attentions.double()
    B, H_q, _, k_len = A.shape
    col = A.sum(2)  # raw accumulated attention per key — NO division
    if window_size > 0:
        col = col[..., :-window_size]
    n = col.shape[-1]
    scores = col.view(B, num_key_value_heads, -1, n).mean(2)
    if window_size > 0:
        scores = F.pad(scores, (0, window_size), value=scores.max().item())
    return scores


class TestH2ORegistry(unittest.TestCase):
    def test_registered_name(self):
        self.assertIn("h2o", available_kv_compressors())
        self.assertIs(get_kv_compressor_class("h2o"), H2OSketch)

    def test_alias(self):
        self.assertIs(get_kv_compressor_class("heavy_hitter_oracle"), H2OSketch)

    def test_get_kv_compressor_injects_compression_ratio(self):
        sketch = get_kv_compressor("h2o", compression_ratio=0.25)
        self.assertIsInstance(sketch, H2OSketch)
        self.assertAlmostEqual(sketch.compression_ratio, 0.25)


class TestH2OScore(unittest.TestCase):
    def test_zero_ratio_noop_without_attentions(self):
        sketch = H2OSketch(compression_ratio=0.0)
        module = _FakeAttnModule(head_dim=4)
        keys = torch.randn(1, 2, 8, 4)
        values = torch.randn(1, 2, 8, 4)
        out_k, out_v = sketch.compress(module, torch.randn(1, 8, 8), keys, values, None, {})
        self.assertIs(out_k, keys)
        self.assertIs(out_v, values)

    def test_missing_attentions_asserts_with_eager_message(self):
        sketch = H2OSketch(compression_ratio=0.5)
        module = _FakeAttnModule(head_dim=4)
        keys = torch.randn(1, 2, 8, 4)
        values = torch.randn(1, 2, 8, 4)
        with self.assertRaises(AssertionError) as ctx:
            sketch.compress(module, torch.randn(1, 8, 8), keys, values, None, {})
        self.assertIn("eager", str(ctx.exception))

    def test_raw_sum_no_window_hand_computed(self):
        # window_size=0 isolates the heavy-hitter score (raw accumulated sum),
        # which is the key difference from observed_attention (no division).
        sketch = H2OSketch(compression_ratio=1 / 3, window_size=0)
        module = _FakeAttnModule(head_dim=2)
        attentions = torch.tensor(
            [[[[1.0, 0.0, 0.0], [0.6, 0.4, 0.0], [0.2, 0.3, 0.5]]]]
        )  # [1, 1, 3, 3], rows sum to 1, strictly causal zeros
        keys = torch.arange(6, dtype=torch.float32).reshape(1, 1, 3, 2)
        values = 10 + torch.arange(6, dtype=torch.float32).reshape(1, 1, 3, 2)

        scores = sketch.score(module, None, keys, values, attentions, {})
        # raw column sums [1.8, 0.7, 0.5] — NOT divided by [3, 2, 1]
        expected = torch.tensor([[[1.8, 0.7, 0.5]]])
        torch.testing.assert_close(scores, expected, atol=1e-6, rtol=0.0)

        out_k, out_v = sketch.compress(module, None, keys, values, attentions, {})
        # n_kept = int(3 * 2/3) = 2; topk by raw score: [0, 1]
        idx = torch.tensor([0, 1])
        self.assertTrue(torch.equal(out_k, keys[:, :, idx, :]))
        self.assertTrue(torch.equal(out_v, values[:, :, idx, :]))

    def test_recent_window_force_kept(self):
        # window_size=1 must force the last position to the max score even
        # though its raw accumulated attention (0.5) is the smallest.
        sketch = H2OSketch(compression_ratio=1 / 3, window_size=1)
        module = _FakeAttnModule(head_dim=2)
        attentions = torch.tensor(
            [[[[1.0, 0.0, 0.0], [0.6, 0.4, 0.0], [0.2, 0.3, 0.5]]]]
        )
        keys = torch.arange(6, dtype=torch.float32).reshape(1, 1, 3, 2)
        values = 10 + torch.arange(6, dtype=torch.float32).reshape(1, 1, 3, 2)

        scores = sketch.score(module, None, keys, values, attentions, {})
        # non-window col sums [1.8, 0.7]; window pos padded with max (1.8)
        expected = torch.tensor([[[1.8, 0.7, 1.8]]])
        torch.testing.assert_close(scores, expected, atol=1e-6, rtol=0.0)

        out_k, _ = sketch.compress(module, None, keys, values, attentions, {})
        # n_kept = 2; kept set is the two max-score positions {0, 2} (the
        # recent window survives, the 0.7 middle token is evicted).
        kept = {tuple(k.tolist()) for k in out_k[0, 0]}
        self.assertEqual(kept, {tuple(keys[0, 0, 0].tolist()), tuple(keys[0, 0, 2].tolist())})

    def test_window_must_be_smaller_than_keys(self):
        sketch = H2OSketch(compression_ratio=0.5, window_size=8)
        module = _FakeAttnModule(head_dim=4)
        attentions = _random_causal_attentions(1, 2, 8)
        keys = torch.randn(1, 2, 8, 4)
        values = torch.randn(1, 2, 8, 4)
        with self.assertRaises(AssertionError):
            sketch.score(module, None, keys, values, attentions, {})

    def test_gqa_blocked_head_grouping(self):
        sketch = H2OSketch(compression_ratio=0.5, window_size=0)
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
        # kv head 0 = mean of query heads 0,1 (blocked): raw col sums [4,0,0,0]
        torch.testing.assert_close(
            scores[0, 0], torch.tensor([4.0, 0.0, 0.0, 0.0]), atol=1e-6, rtol=0.0
        )
        # kv head 1 = mean of query heads 2,3: raw col sums [25/12, 13/12, 7/12, 1/4]
        torch.testing.assert_close(
            scores[0, 1],
            torch.tensor([25 / 12, 13 / 12, 7 / 12, 1 / 4]),
            atol=1e-6,
            rtol=0.0,
        )

    def test_reference_oracle_float32(self):
        sketch = H2OSketch(compression_ratio=0.5, window_size=4)
        module = _FakeAttnModule(head_dim=8)
        B, H_q, H_kv, S, D = 2, 4, 2, 16, 8
        attentions = _random_causal_attentions(B, H_q, S, seed=11)
        keys = torch.randn(B, H_kv, S, D)
        values = torch.randn(B, H_kv, S, D)

        scores = sketch.score(module, None, keys, values, attentions, {})
        self.assertEqual(scores.dtype, torch.float32)
        oracle = _h2o_oracle(attentions, H_kv, window_size=4)
        torch.testing.assert_close(scores.double(), oracle, rtol=1e-5, atol=1e-7)

    def test_attention_key_length_mismatch_raises(self):
        sketch = H2OSketch(compression_ratio=0.5)
        module = _FakeAttnModule(head_dim=4)
        attentions = _random_causal_attentions(1, 2, 10)
        keys = torch.randn(1, 2, 8, 4)
        values = torch.randn(1, 2, 8, 4)
        with self.assertRaisesRegex(ValueError, "prefill"):
            sketch.score(module, None, keys, values, attentions, {})


class TestH2OHookIntegration(unittest.TestCase):
    def test_prefill_compresses_cache_and_decode_is_noop(self):
        sketch = H2OSketch(compression_ratio=0.5, window_size=2)
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

            oracle = _h2o_oracle(module.attn_probs, H_kv, window_size=2)
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
