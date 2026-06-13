"""Tests for AdaKVSketch (port of kvpress AdaKVPress) and its attention-patch
enforcement path.

All expectations are hand-computed from the kvpress math
(kvpress/presses/adakv_press.py, kvpress/attention_patch.py). No model loading.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from torch import nn
from transformers import DynamicCache

from eval_harness.sketch.attention_patch import attention_patch, search_hyperplane
from eval_harness.sketch.sketches.adakv_sketch import AdaKVSketch
from eval_harness.sketch.sketches.decoding_sketch import DecodingSketch
from eval_harness.sketch.sketches.knorm_sketch import KnormSketch
from eval_harness.sketch.sketches.registry import get_sketch_class
from eval_harness.sketch.sketches.scorer_sketch import ScorerSketch


class _FakeAttnModule(nn.Module):
    def __init__(self, num_heads=4, num_kv_heads=2, head_dim=8, layer_idx=0,
                 attn_implementation="sdpa"):
        super().__init__()
        self.num_heads = num_heads
        self.num_key_value_heads = num_kv_heads
        self.head_dim = head_dim
        self.layer_idx = layer_idx
        self.config = SimpleNamespace(_attn_implementation=attn_implementation)


class _StubScorer(ScorerSketch):
    """Returns a fixed (cloned) score tensor and counts score() calls."""

    def __init__(self, scores: torch.Tensor, compression_ratio: float = 0.5):
        super().__init__(compression_ratio=compression_ratio)
        self.fixed_scores = scores
        self.score_calls = 0

    def score(self, module, hidden_states, keys, values, attentions, kwargs):
        self.score_calls += 1
        return self.fixed_scores.clone()


def _plain_attention(module, query, key, value, attention_mask, dropout, **kwargs):
    """Reference sdpa-like attention fn (GQA via repeat over groups)."""
    num_groups = query.shape[1] // key.shape[1]
    k = key.repeat_interleave(num_groups, dim=1)
    v = value.repeat_interleave(num_groups, dim=1)
    logits = query @ k.transpose(-1, -2) / (query.shape[-1] ** 0.5)
    weights = torch.softmax(logits, dim=-1)
    return weights @ v, weights


def _masked_set(module):
    b, h, s = module.masked_key_indices
    return set(zip(b.tolist(), h.tolist(), s.tolist()))


class TestAdaKVRegistryAndConstruction(unittest.TestCase):
    def test_registered_name(self):
        self.assertIs(get_sketch_class("adakv"), AdaKVSketch)

    def test_alpha_out_of_bounds_raises(self):
        with self.assertRaises(AssertionError):
            AdaKVSketch(press=KnormSketch(compression_ratio=0.5), alpha_safeguard=1.5)
        with self.assertRaises(AssertionError):
            AdaKVSketch(press=KnormSketch(compression_ratio=0.5), alpha_safeguard=-0.1)

    def test_non_scorer_press_raises(self):
        with self.assertRaises(AssertionError):
            AdaKVSketch(press="not a sketch")

    def test_compression_ratio_delegates_to_press(self):
        inner = KnormSketch(compression_ratio=0.5)
        sketch = AdaKVSketch(press=inner)
        self.assertAlmostEqual(sketch.compression_ratio, 0.5)
        sketch.compression_ratio = 0.3
        self.assertAlmostEqual(inner.compression_ratio, 0.3)
        inner.compression_ratio = 0.7
        self.assertAlmostEqual(sketch.compression_ratio, 0.7)

    def test_post_init_from_model_delegates(self):
        calls = []

        class _Recorder(KnormSketch):
            def post_init_from_model(self, model):
                calls.append(model)

        sketch = AdaKVSketch(press=_Recorder(compression_ratio=0.5))
        sentinel = object()
        sketch.post_init_from_model(sentinel)
        self.assertEqual(calls, [sentinel])


class TestAdaKVCompress(unittest.TestCase):
    def test_zero_ratio_noop_without_config(self):
        # The ratio==0 early return precedes the eager assert, so a config-less
        # stub module must pass and score must never be called.
        scorer = _StubScorer(torch.zeros(1, 2, 8), compression_ratio=0.0)
        sketch = AdaKVSketch(press=scorer)
        module = SimpleNamespace()
        keys = torch.randn(1, 2, 8, 4)
        values = torch.randn(1, 2, 8, 4)
        out_k, out_v = sketch.compress(module, torch.zeros(1, 8, 8), keys, values, None, {})
        self.assertIs(out_k, keys)
        self.assertIs(out_v, values)
        self.assertEqual(scorer.score_calls, 0)
        self.assertFalse(hasattr(module, "masked_key_indices"))

    def test_hand_computed_masked_indices(self):
        # B=1, H_kv=2, S=8, ratio=0.5 -> n_kept=4; alpha=0.5 -> n_safe=2;
        # n_pruned = 2*(8-4) = 8. Safeguard pins h0{0,1} and h1{0,1}; the
        # global bottom-8 of the remaining 12 = all of h1 idx2..7 plus h0
        # idx7 (3.0) and idx6 (4.0).
        scores = torch.tensor(
            [[[10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0],
              [2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3]]]
        )
        sketch = AdaKVSketch(press=_StubScorer(scores, compression_ratio=0.5), alpha_safeguard=0.5)
        module = _FakeAttnModule(num_kv_heads=2, head_dim=4)
        keys = torch.randn(1, 2, 8, 4)
        values = torch.randn(1, 2, 8, 4)

        out_k, out_v = sketch.compress(module, torch.zeros(1, 8, 8), keys, values, None, {})

        self.assertIs(out_k, keys)
        self.assertIs(out_v, values)
        self.assertEqual(out_k.shape[2], 8)

        expected = {(0, 0, 6), (0, 0, 7)} | {(0, 1, s) for s in range(2, 8)}
        self.assertEqual(_masked_set(module), expected)

        batch_indices, head_indices, _ = module.masked_key_indices
        self.assertEqual(batch_indices.numel(), 8)
        counts = torch.bincount(head_indices, minlength=2)
        # head0 retains 6 (variable budget), head1 retains exactly n_safe=2.
        self.assertEqual(counts.tolist(), [2, 6])

    def test_safeguard_floor_property(self):
        # ratio=0.75 -> n_kept=16, n_safe=int(16*0.2)=3, n_pruned=4*48=192.
        torch.manual_seed(0)
        scores = torch.rand(1, 4, 64)
        sketch = AdaKVSketch(press=_StubScorer(scores, compression_ratio=0.75), alpha_safeguard=0.2)
        module = _FakeAttnModule(num_kv_heads=4, head_dim=4)
        keys = torch.randn(1, 4, 64, 4)
        sketch.compress(module, torch.zeros(1, 64, 16), keys, keys.clone(), None, {})

        batch_indices, head_indices, seq_indices = module.masked_key_indices
        self.assertEqual(batch_indices.numel(), 192)
        counts = torch.bincount(head_indices, minlength=4)
        self.assertEqual(counts.sum().item(), 192)
        self.assertTrue((counts <= 61).all(), f"head retained < n_safe=3: {counts.tolist()}")
        self.assertTrue((seq_indices >= 0).all() and (seq_indices < 64).all())

    def test_alpha_one_reduces_to_per_head_topk(self):
        # alpha=1 -> n_safe=n_kept: each head pins its own top-n_kept, so the
        # masked set is exactly each head's bottom S-n_kept by the inner score.
        torch.manual_seed(1)
        B, H_kv, S, D = 1, 2, 16, 8
        keys = torch.randn(B, H_kv, S, D)
        values = torch.randn(B, H_kv, S, D)
        sketch = AdaKVSketch(press=KnormSketch(compression_ratio=0.5), alpha_safeguard=1.0)
        module = _FakeAttnModule(num_kv_heads=H_kv, head_dim=D)
        sketch.compress(module, torch.zeros(B, S, 16), keys, values, None, {})

        knorm_scores = -keys.norm(dim=-1)
        n_kept = 8
        expected = set()
        for h in range(H_kv):
            top = set(knorm_scores[0, h].topk(n_kept).indices.tolist())
            expected |= {(0, h, s) for s in range(S) if s not in top}
        self.assertEqual(_masked_set(module), expected)

    def test_compression_ratio_accounting(self):
        torch.manual_seed(2)
        B, H_kv, S, D = 1, 2, 128, 8
        keys = torch.randn(B, H_kv, S, D)
        for ratio in (0.2, 0.4, 0.6, 0.8):
            module = _FakeAttnModule(num_kv_heads=H_kv, head_dim=D)
            sketch = AdaKVSketch(press=KnormSketch(compression_ratio=ratio))
            sketch.compress(module, torch.zeros(B, S, 16), keys, keys.clone(), None, {})
            self.assertIsNotNone(module.masked_key_indices)
            n_masked = module.masked_key_indices[0].numel()
            self.assertEqual(n_masked, H_kv * (S - int(S * (1 - ratio))))
            masked_fraction = n_masked / (B * H_kv * S)
            self.assertLess(abs(masked_fraction - ratio), 1e-2)

    def test_edge_n_safe_zero(self):
        # alpha=0 and S=8, ratio=0.875 (exact in fp) -> n_kept=1, n_safe=0:
        # empty topk/scatter_ are no-ops and masking still prunes
        # H_kv*(S-n_kept) = 2*7.
        torch.manual_seed(3)
        scores = torch.rand(1, 2, 8)
        sketch = AdaKVSketch(press=_StubScorer(scores, compression_ratio=0.875), alpha_safeguard=0.0)
        module = _FakeAttnModule(num_kv_heads=2, head_dim=4)
        keys = torch.randn(1, 2, 8, 4)
        out_k, _ = sketch.compress(module, torch.zeros(1, 8, 8), keys, keys.clone(), None, {})
        self.assertEqual(out_k.shape[2], 8)
        _, _, seq_indices = module.masked_key_indices
        self.assertEqual(seq_indices.numel(), 2 * 7)
        self.assertTrue((seq_indices >= 0).all() and (seq_indices < 8).all())

    def test_eager_guard_raises(self):
        sketch = AdaKVSketch(press=KnormSketch(compression_ratio=0.5))
        module = _FakeAttnModule(attn_implementation="eager")
        keys = torch.randn(1, 2, 8, 8)
        with self.assertRaisesRegex(AssertionError, "eager"):
            sketch.compress(module, torch.zeros(1, 8, 16), keys, keys.clone(), None, {})


class TestAttentionPatchEnforcement(unittest.TestCase):
    def test_search_hyperplane_oracle(self):
        # Transcribed from kvpress tests/test_attention_patch.py.
        torch.manual_seed(0)
        X = torch.rand(50, 500, 128)
        Y = search_hyperplane(X)
        self.assertEqual(torch.exp(torch.bmm(X, Y.unsqueeze(-1))).max().item(), 0.0)

    def test_gqa_patch_enforcement(self):
        torch.manual_seed(4)
        module = SimpleNamespace(
            masked_key_indices=(torch.tensor([0, 0]), torch.tensor([0, 1]), torch.tensor([2, 5]))
        )
        q = torch.randn(1, 4, 1, 64)
        k = torch.randn(1, 2, 10, 64)
        v = torch.randn(1, 2, 10, 64)
        k_orig = k.clone()

        recorded = {}

        def recording_fn(module, query, key, value, attention_mask, dropout, **kwargs):
            recorded["key"] = key
            return query, None

        attention_patch(recording_fn)(module, q, k, v, None, 0.0)
        self.assertIs(recorded["key"], k)

        # Masked rows replaced, all others bitwise unchanged.
        self.assertFalse(torch.equal(k[0, 0, 2], k_orig[0, 0, 2]))
        self.assertFalse(torch.equal(k[0, 1, 5], k_orig[0, 1, 5]))
        for g in range(2):
            for s in range(10):
                if (g, s) in {(0, 2), (1, 5)}:
                    continue
                self.assertTrue(torch.equal(k[0, g, s], k_orig[0, g, s]))

        # One fake key per kv-head must nullify every grouped query head.
        for h in range(4):
            g = h // 2
            s = 2 if g == 0 else 5
            self.assertEqual(torch.exp(q[0, h, 0] @ k[0, g, s]).item(), 0.0)

    def test_prefill_reset_hygiene(self):
        module = SimpleNamespace(
            masked_key_indices=(torch.tensor([0]), torch.tensor([0]), torch.tensor([3]))
        )
        q = torch.randn(1, 2, 10, 8)
        k = torch.randn(1, 2, 10, 8)
        k_orig = k.clone()
        attention_patch(_plain_attention)(module, q, k, k.clone(), None, 0.0)
        self.assertIsNone(module.masked_key_indices)
        self.assertTrue(torch.equal(k, k_orig))

    def test_decode_parity_with_physical_pruning(self):
        # Masking via fake keys must equal attention with the masked entries
        # physically removed (exp(<q, k_fake>) == 0 exactly).
        torch.manual_seed(5)
        B, H_q, H_kv, S, D = 1, 4, 2, 16, 8
        keys = torch.randn(B, H_kv, S, D)
        values = torch.randn(B, H_kv, S, D)
        module = _FakeAttnModule(num_heads=H_q, num_kv_heads=H_kv, head_dim=D)
        sketch = AdaKVSketch(press=KnormSketch(compression_ratio=0.5), alpha_safeguard=0.2)
        sketch.compress(module, torch.zeros(B, S, 32), keys, values, None, {})
        self.assertEqual(module.masked_key_indices[0].numel(), H_kv * (S - 8))

        masked_per_head = {g: set() for g in range(H_kv)}
        for _, g, s in zip(*[t.tolist() for t in module.masked_key_indices]):
            masked_per_head[g].add(s)

        q = torch.randn(B, H_q, 1, D)
        out, _ = attention_patch(_plain_attention)(module, q, keys.clone(), values, None, 0.0)

        num_groups = H_q // H_kv
        for h in range(H_q):
            g = h // num_groups
            kept = [s for s in range(S) if s not in masked_per_head[g]]
            logits = q[0, h, 0] @ keys[0, g, kept].T / (D ** 0.5)
            ref = torch.softmax(logits, dim=-1) @ values[0, g, kept]
            torch.testing.assert_close(out[0, h, 0], ref, rtol=1e-5, atol=1e-6)


class TestAdaKVForwardHook(unittest.TestCase):
    def test_no_shrink_cache_semantics(self):
        # The hook must leave every layer's cache at full length S while
        # recording per-layer masked indices (the documented no-memory-savings
        # behavior; no cross-layer raggedness can arise).
        torch.manual_seed(6)
        B, H_kv, S, D = 1, 2, 32, 8
        cache = DynamicCache()
        originals = []
        for layer_idx in range(2):
            k = torch.randn(B, H_kv, S, D)
            v = torch.randn(B, H_kv, S, D)
            originals.append((k.clone(), v.clone()))
            cache.update(k, v, layer_idx)

        sketch = AdaKVSketch(press=KnormSketch(compression_ratio=0.5))
        hidden = torch.randn(B, S, 16)
        for layer_idx in range(2):
            module = _FakeAttnModule(num_kv_heads=H_kv, head_dim=D, layer_idx=layer_idx)
            kwargs = {
                "hidden_states": hidden,
                "past_key_values": cache,
                "cache_position": torch.arange(S),
            }
            sketch.forward_hook(module, [], kwargs, (hidden, None))
            self.assertEqual(cache.layers[layer_idx].keys.shape[2], S)
            self.assertTrue(torch.equal(cache.layers[layer_idx].keys, originals[layer_idx][0]))
            self.assertTrue(torch.equal(cache.layers[layer_idx].values, originals[layer_idx][1]))
            self.assertEqual(module.masked_key_indices[0].numel(), H_kv * (S - 16))


class TestDecodingSketchAdaKVBase(unittest.TestCase):
    def test_decoding_sketch_accepts_adakv_base(self):
        inner = AdaKVSketch(press=KnormSketch(compression_ratio=0.0))
        decoding = DecodingSketch(base_sketch=inner, compression_interval=4, target_size=8)

        torch.manual_seed(7)
        module = _FakeAttnModule(num_kv_heads=2, head_dim=4)
        keys = torch.randn(1, 2, 16, 4)
        values = torch.randn(1, 2, 16, 4)
        out_k, out_v = decoding.compress(module, torch.zeros(1, 4, 8), keys, values, None, {})

        # AdaKV never prunes: identical tensors, masked indices for the
        # temporary target ratio (16 -> 8 kept => 2*8 masked), ratio restored.
        self.assertIs(out_k, keys)
        self.assertIs(out_v, values)
        self.assertEqual(module.masked_key_indices[0].numel(), 16)
        self.assertAlmostEqual(inner.compression_ratio, 0.0)

    def test_decoding_sketch_still_accepts_scorer_and_rejects_others(self):
        DecodingSketch(base_sketch=KnormSketch(compression_ratio=0.5))
        with self.assertRaises(AssertionError):
            DecodingSketch(base_sketch="not a sketch")


if __name__ == "__main__":
    unittest.main()
