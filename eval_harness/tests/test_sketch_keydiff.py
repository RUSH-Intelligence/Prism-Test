"""Tests for KeyDiffSketch (port of kvpress KeyDiffPress).

Reference math (kvpress/presses/keydiff_press.py:45-46):
    anchor = F.normalize(keys, p=2, dim=-1).mean(dim=2, keepdim=True)
    score  = -F.cosine_similarity(keys, anchor, dim=-1)
"""

from __future__ import annotations

import unittest

import torch
from torch import nn
from torch.nn import functional as F
from transformers import DynamicCache

from eval_harness.kv_compression.compressors.keydiff_sketch import KeyDiffSketch
from eval_harness.kv_compression.compressors.knorm_sketch import KnormSketch
from eval_harness.kv_compression.registry import (
    available_kv_compressors,
    get_kv_compressor,
    get_kv_compressor_class,
)


class _FakeAttnModule(nn.Module):
    """Minimal attention module: KeyDiff's score() touches nothing on the
    module; only ``head_dim`` (gather expand) and ``layer_idx`` (cache access)
    are needed. GQA-shaped by default (num_heads > num_key_value_heads)."""

    def __init__(self, num_heads=4, num_kv_heads=2, head_dim=2, layer_idx=0):
        super().__init__()
        self.num_heads = num_heads
        self.num_key_value_heads = num_kv_heads
        self.head_dim = head_dim
        self.layer_idx = layer_idx


def _keydiff_reference(keys: torch.Tensor) -> torch.Tensor:
    """In-test transcription of kvpress KeyDiffPress.score."""
    anchor = F.normalize(keys, p=2, dim=-1).mean(dim=2, keepdim=True)
    return -F.cosine_similarity(keys, anchor, dim=-1)


def _pinned_keys() -> torch.Tensor:
    """[1, 1, 4, 2] fp32 keys at directions 0/30/90/200 degrees with
    magnitudes 2/5/1/3 (distinct magnitudes prove scale-invariance)."""
    return torch.tensor(
        [[[[2.0, 0.0],
           [4.3301270, 2.5],
           [0.0, 1.0],
           [-2.8190779, -1.0260604]]]],
        dtype=torch.float32,
    )


# Hand-computed expectations for ``_pinned_keys`` (verified numerically):
_PINNED_ANCHOR = torch.tensor([0.2315832, 0.2894950])
_PINNED_SCORES = torch.tensor([-0.6246740, -0.9314264, -0.7808856, 0.8540802])
_PINNED_ORDER = [3, 0, 2, 1]  # descending-score order


class TestKeyDiffRegistry(unittest.TestCase):
    def test_registered_under_keydiff(self):
        self.assertIn("keydiff", available_kv_compressors())
        self.assertIs(get_kv_compressor_class("keydiff"), KeyDiffSketch)

    def test_get_kv_compressor_instantiates_with_ratio(self):
        sketch = get_kv_compressor("keydiff", compression_ratio=0.3)
        self.assertIsInstance(sketch, KeyDiffSketch)
        self.assertAlmostEqual(sketch.compression_ratio, 0.3)


class TestKeyDiffScore(unittest.TestCase):
    def test_hand_pinned_anchor_and_scores(self):
        keys = _pinned_keys()
        anchor = F.normalize(keys, p=2, dim=-1).mean(dim=2, keepdim=True)
        torch.testing.assert_close(
            anchor.flatten(), _PINNED_ANCHOR, atol=1e-5, rtol=0
        )
        scores = KeyDiffSketch(compression_ratio=0.5).score(
            _FakeAttnModule(), None, keys, keys.clone(), None, {}
        )
        self.assertEqual(scores.shape, (1, 1, 4))
        torch.testing.assert_close(
            scores.flatten(), _PINNED_SCORES, atol=1e-5, rtol=0
        )
        self.assertEqual(
            scores.flatten().argsort(descending=True).tolist(), _PINNED_ORDER
        )

    def test_hand_pinned_selection_ratio_half(self):
        keys = _pinned_keys()
        values = torch.arange(8, dtype=torch.float32).reshape(1, 1, 4, 2)
        module = _FakeAttnModule(head_dim=2)
        out_keys, out_values = KeyDiffSketch(compression_ratio=0.5).compress(
            module, None, keys, values, None, {}
        )
        # n_kept = int(4 * 0.5) = 2; kept in topk descending-score order
        # [3, 0] — NOT position order — matching kvpress ScorerPress.
        torch.testing.assert_close(out_keys, keys[:, :, [3, 0]])
        torch.testing.assert_close(out_values, values[:, :, [3, 0]])

    def test_hand_pinned_selection_ratio_three_quarters(self):
        keys = _pinned_keys()
        values = torch.arange(8, dtype=torch.float32).reshape(1, 1, 4, 2)
        module = _FakeAttnModule(head_dim=2)
        out_keys, out_values = KeyDiffSketch(compression_ratio=0.75).compress(
            module, None, keys, values, None, {}
        )
        torch.testing.assert_close(out_keys, keys[:, :, [3]])
        torch.testing.assert_close(out_values, values[:, :, [3]])

    def test_scale_invariance_distinguishes_from_knorm(self):
        torch.manual_seed(3)
        keys = torch.randn(2, 3, 16, 8)
        scale = torch.rand(2, 3, 16, 1) * 4 + 0.1  # positive per-token scalars
        sketch = KeyDiffSketch(compression_ratio=0.5)
        module = _FakeAttnModule(head_dim=8)
        base = sketch.score(module, None, keys, keys, None, {})
        scaled = sketch.score(module, None, keys * scale, keys * scale, None, {})
        torch.testing.assert_close(scaled, base, atol=1e-5, rtol=0)
        # Knorm is NOT scale-invariant — proves the test discriminates.
        knorm = KnormSketch(compression_ratio=0.5)
        self.assertFalse(
            torch.allclose(
                knorm.score(module, None, keys * scale, None, None, {}),
                knorm.score(module, None, keys, None, None, {}),
                atol=1e-5,
            )
        )

    def test_reference_transcription_oracle_fp32_vs_fp64(self):
        torch.manual_seed(0)
        keys = torch.randn(2, 4, 37, 8, dtype=torch.float32)
        scores = KeyDiffSketch(compression_ratio=0.5).score(
            _FakeAttnModule(head_dim=8), None, keys, keys, None, {}
        )
        expected = _keydiff_reference(keys.double())
        torch.testing.assert_close(
            scores.double(), expected, atol=1e-6, rtol=0
        )

    def test_reference_transcription_oracle_bf16(self):
        torch.manual_seed(0)
        keys32 = torch.randn(2, 4, 37, 8, dtype=torch.float32)
        keys = keys32.bfloat16()
        scores = KeyDiffSketch(compression_ratio=0.5).score(
            _FakeAttnModule(head_dim=8), None, keys, keys, None, {}
        )
        self.assertEqual(scores.shape, (2, 4, 37))
        expected = _keydiff_reference(keys32.double())
        torch.testing.assert_close(
            scores.double(), expected, atol=1e-2, rtol=0
        )


class TestKeyDiffCompress(unittest.TestCase):
    def test_zero_ratio_is_identity_noop(self):
        torch.manual_seed(1)
        keys = torch.randn(1, 2, 8, 4)
        values = torch.randn(1, 2, 8, 4)
        out_keys, out_values = KeyDiffSketch(compression_ratio=0.0).compress(
            _FakeAttnModule(head_dim=4), None, keys, values, None, {}
        )
        self.assertIs(out_keys, keys)
        self.assertIs(out_values, values)

    def test_gqa_per_head_independence_and_rectangularity(self):
        head0 = _pinned_keys()[0, 0]  # [4, 2]
        # Head 1: same rows permuted so the distinctive (200-degree) token
        # sits at position 0; the anchor (mean over seq) is permutation-
        # invariant, so head-1 scores are head-0 scores permuted.
        perm = [3, 0, 1, 2]
        head1 = head0[perm]
        keys = torch.stack([head0, head1]).unsqueeze(0)  # [1, 2, 4, 2]
        values = torch.arange(16, dtype=torch.float32).reshape(1, 2, 4, 2)
        module = _FakeAttnModule(num_heads=4, num_kv_heads=2, head_dim=2)
        sketch = KeyDiffSketch(compression_ratio=0.5)

        scores = sketch.score(module, None, keys, values, None, {})
        self.assertEqual(scores.dim(), 3)
        self.assertEqual(scores.shape, (1, 2, 4))

        out_keys, out_values = sketch.compress(
            module, None, keys, values, None, {}
        )
        # Uniform count per head despite per-head-independent indices.
        self.assertEqual(out_keys.shape, (1, 2, 2, 2))
        self.assertEqual(out_values.shape, (1, 2, 2, 2))
        # Head 0 keeps positions [3, 0]; head 1 keeps its own [0, 1].
        torch.testing.assert_close(out_keys[:, 0], keys[:, 0, [3, 0]])
        torch.testing.assert_close(out_keys[:, 1], keys[:, 1, [0, 1]])
        torch.testing.assert_close(out_values[:, 0], values[:, 0, [3, 0]])
        torch.testing.assert_close(out_values[:, 1], values[:, 1, [0, 1]])

    def test_n_kept_truncation(self):
        torch.manual_seed(2)
        keys = torch.randn(1, 2, 10, 4)
        values = torch.randn(1, 2, 10, 4)
        out_keys, out_values = KeyDiffSketch(compression_ratio=0.34).compress(
            _FakeAttnModule(head_dim=4), None, keys, values, None, {}
        )
        # n_kept = int(10 * 0.66) = int(6.6) = 6
        self.assertEqual(out_keys.shape, (1, 2, 6, 4))
        self.assertEqual(out_values.shape, (1, 2, 6, 4))

    def test_n_kept_zero_returns_empty_without_raising(self):
        # S=1, ratio=0.5 -> n_kept = int(0.5) = 0: kvpress parity is an
        # EMPTY cache layer. Decoding from an empty cache breaks downstream
        # (framework-level hazard shared with knorm/random); not
        # special-cased here because kvpress does not special-case it.
        keys = torch.randn(1, 2, 1, 4)
        values = torch.randn(1, 2, 1, 4)
        out_keys, out_values = KeyDiffSketch(compression_ratio=0.5).compress(
            _FakeAttnModule(head_dim=4), None, keys, values, None, {}
        )
        self.assertEqual(out_keys.shape, (1, 2, 0, 4))
        self.assertEqual(out_values.shape, (1, 2, 0, 4))

    def test_degenerate_anchor_no_nan(self):
        # Antipodal keys: normalized keys cancel, anchor is exactly zero;
        # F.cosine_similarity's eps=1e-8 guard yields finite (zero) scores.
        keys = torch.tensor([[[[1.0, 0.0], [-1.0, 0.0]]]])
        values = torch.tensor([[[[10.0, 11.0], [20.0, 21.0]]]])
        sketch = KeyDiffSketch(compression_ratio=0.5)
        module = _FakeAttnModule(head_dim=2)
        scores = sketch.score(module, None, keys, values, None, {})
        self.assertTrue(torch.isfinite(scores).all())
        out_keys, out_values = sketch.compress(
            module, None, keys, values, None, {}
        )
        self.assertEqual(out_keys.shape, (1, 1, 1, 2))
        self.assertTrue(torch.isfinite(out_keys).all())
        self.assertTrue(torch.isfinite(out_values).all())


class TestKeyDiffForwardHook(unittest.TestCase):
    def _hook_kwargs(self, hidden_states, cache, cache_position):
        return {
            "hidden_states": hidden_states,
            "past_key_values": cache,
            "cache_position": cache_position,
        }

    def test_prefill_step_prunes_cache_to_recomputed_selection(self):
        torch.manual_seed(7)
        B, H_kv, S, D = 1, 2, 8, 4
        keys = torch.randn(B, H_kv, S, D)
        values = torch.randn(B, H_kv, S, D)
        cache = DynamicCache()
        cache.update(keys.clone(), values.clone(), 0)

        module = _FakeAttnModule(num_heads=4, num_kv_heads=H_kv, head_dim=D)
        sketch = KeyDiffSketch(compression_ratio=0.25)
        hidden_states = torch.randn(B, S, 16)
        output = (torch.randn(B, S, 16), None)

        result = sketch.forward_hook(
            module,
            [],
            self._hook_kwargs(hidden_states, cache, torch.arange(S)),
            output,
        )
        self.assertIs(result, output)

        # n_kept = int(8 * 0.75) = 6; recompute the expected selection from
        # the reference transcription.
        ref_scores = _keydiff_reference(keys)
        indices = ref_scores.topk(6, dim=-1).indices
        expanded = indices.unsqueeze(-1).expand(-1, -1, -1, D)
        torch.testing.assert_close(
            cache.layers[0].keys, keys.gather(2, expanded)
        )
        torch.testing.assert_close(
            cache.layers[0].values, values.gather(2, expanded)
        )

    def test_decode_step_is_noop(self):
        torch.manual_seed(8)
        B, H_kv, S, D = 1, 2, 8, 4
        cache = DynamicCache()
        cache.update(torch.randn(B, H_kv, S, D), torch.randn(B, H_kv, S, D), 0)
        # Simulate the decode-time cache update for one new token.
        cache.update(torch.randn(B, H_kv, 1, D), torch.randn(B, H_kv, 1, D), 0)
        keys_before = cache.layers[0].keys.clone()
        values_before = cache.layers[0].values.clone()

        module = _FakeAttnModule(num_heads=4, num_kv_heads=H_kv, head_dim=D)
        sketch = KeyDiffSketch(compression_ratio=0.25)
        hidden_states = torch.randn(B, 1, 16)
        output = (torch.randn(B, 1, 16), None)

        # cache_position[-1] = S > q_len = 1 -> _is_decoding_step gates a no-op.
        sketch.forward_hook(
            module,
            [],
            self._hook_kwargs(hidden_states, cache, torch.tensor([S])),
            output,
        )
        self.assertEqual(cache.layers[0].keys.shape[2], S + 1)
        torch.testing.assert_close(cache.layers[0].keys, keys_before)
        torch.testing.assert_close(cache.layers[0].values, values_before)


if __name__ == "__main__":
    unittest.main()
