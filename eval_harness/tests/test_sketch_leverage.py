"""Tests for LeverageScoreSketch (kvpress LeverageScorePress port).

Reference oracle: in-test transcription of kvpress
``LeverageScorePress.compute_leverage_scores`` (leverage_press.py:58-93) using an
explicit ``torch.linalg.solve`` against the first-attempt-jittered Gram matrix
``0.5*(G+G^T) + 1e-2*I`` — pinning the centering order, the
sketch-in-key-dtype-then-fp32-cast order, and the always-applied first jitter.
"""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from eval_harness.kv_compression.compressors.leverage_sketch import (
    LeverageScoreSketch,
    _get_prerope_key_states,
)
from eval_harness.kv_compression.registry import (
    available_kv_compressors,
    get_kv_compressor,
    get_kv_compressor_class,
)


class _FakeAttnModule(nn.Module):
    """Llama-like fake attention module with q_proj/k_proj and required attrs.

    With ``identity_k`` (requires hidden_dim == num_kv_heads * head_dim), k_proj
    is the identity so pre-RoPE keys equal the reshaped hidden states — letting
    tests control the scored keys exactly. With ``poison_q``, q_proj weights are
    NaN to prove queries are never read.
    """

    def __init__(self, hidden_dim=32, num_heads=4, head_dim=8, num_kv_heads=2,
                 identity_k=False, poison_q=False, seed=0):
        super().__init__()
        self.num_heads = num_heads
        self.num_key_value_heads = num_kv_heads
        self.head_dim = head_dim
        self.layer_idx = 0
        torch.manual_seed(seed)
        self.q_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, num_kv_heads * head_dim, bias=False)
        if identity_k:
            assert hidden_dim == num_kv_heads * head_dim
            with torch.no_grad():
                self.k_proj.weight.copy_(torch.eye(hidden_dim))
        if poison_q:
            with torch.no_grad():
                self.q_proj.weight.fill_(float("nan"))


class _FakePhi3Module(nn.Module):
    """Phi3-like fake with fused qkv_proj (no k_proj)."""

    def __init__(self, hidden_dim=16, num_heads=4, head_dim=4, num_kv_heads=2, seed=0):
        super().__init__()
        self.head_dim = head_dim
        self.num_key_value_heads = num_kv_heads
        self.layer_idx = 0
        self.config = SimpleNamespace(num_attention_heads=num_heads)
        torch.manual_seed(seed)
        self.qkv_proj = nn.Linear(hidden_dim, (num_heads + 2 * num_kv_heads) * head_dim, bias=False)


def _hidden_for_keys(K: torch.Tensor) -> torch.Tensor:
    """Hidden states that an identity k_proj maps exactly to keys K [B,H,S,D]."""
    B, H, S, D = K.shape
    return K.transpose(1, 2).reshape(B, S, H * D)


def _leverage_reference(K: torch.Tensor, k: int) -> torch.Tensor:
    """kvpress compute_leverage_scores transcription with an explicit solve.

    Caller must seed the global RNG identically to the production call so the
    Gaussian sketch Phi matches draw-for-draw.
    """
    B, H, S, d = K.shape
    Phi = torch.randn(B, H, d, k, device=K.device, dtype=K.dtype) * (1 / math.sqrt(k))
    X = K - K.mean(dim=-2, keepdim=True)
    X = torch.matmul(X, Phi).to(torch.float32)
    G = X.transpose(-2, -1) @ X
    M = 0.5 * (G + G.transpose(-2, -1)) + 1e-2 * torch.eye(k, dtype=torch.float32)
    sol = torch.linalg.solve(M, X.transpose(-2, -1))
    return (X * sol.transpose(-2, -1)).sum(dim=-1).clamp_min(0)


class TestComputeLeverageScores(unittest.TestCase):
    def test_reference_oracle_fp32(self):
        torch.manual_seed(123)
        K = torch.randn(1, 2, 12, 8)
        torch.manual_seed(0)
        out = LeverageScoreSketch.compute_leverage_scores(K, 4)
        torch.manual_seed(0)
        ref = _leverage_reference(K, 4)
        self.assertEqual(out.shape, (1, 2, 12))
        self.assertEqual(out.dtype, torch.float32)
        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
        self.assertTrue((out >= 0).all())

    def test_bf16_keys_sketched_in_bf16_then_cast_to_fp32(self):
        torch.manual_seed(7)
        K = torch.randn(1, 2, 10, 8).to(torch.bfloat16)
        torch.manual_seed(0)
        out = LeverageScoreSketch.compute_leverage_scores(K, 4)
        self.assertEqual(out.dtype, torch.float32)
        self.assertTrue(torch.isfinite(out).all())
        torch.manual_seed(0)
        ref = _leverage_reference(K, 4)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

    def test_identical_tokens_give_zero_scores(self):
        K = torch.full((1, 2, 10, 8), 0.7)
        torch.manual_seed(0)
        out = LeverageScoreSketch.compute_leverage_scores(K, 4)
        torch.testing.assert_close(out, torch.zeros(1, 2, 10))

    def test_sketch_dimension_larger_than_seq_and_head_dim(self):
        torch.manual_seed(1)
        K = torch.randn(1, 2, 4, 8)
        torch.manual_seed(0)
        out = LeverageScoreSketch.compute_leverage_scores(K, 48)
        self.assertEqual(out.shape, (1, 2, 4))
        self.assertTrue(torch.isfinite(out).all())


class TestCholWithJitter(unittest.TestCase):
    def test_raises_after_max_tries(self):
        G = -1e6 * torch.eye(3)
        with self.assertRaisesRegex(RuntimeError, "Cholesky failed after 5 tries."):
            LeverageScoreSketch.chol_with_jitter(G, jitter=1e-2, max_tries=5)

    def test_jitter_escalates_until_positive_definite(self):
        G = -0.5 * torch.eye(3)
        L = LeverageScoreSketch.chol_with_jitter(G, jitter=1e-2, max_tries=5)
        self.assertTrue(torch.equal(L, L.tril()))
        torch.testing.assert_close(L @ L.transpose(-2, -1), 0.5 * torch.eye(3), atol=1e-6, rtol=1e-6)

    def test_zero_jitter_succeeds_on_pd_matrix_without_jitter(self):
        G = torch.eye(3) * 2.0
        L = LeverageScoreSketch.chol_with_jitter(G)
        torch.testing.assert_close(L @ L.transpose(-2, -1), G, atol=1e-6, rtol=1e-6)


class TestLeverageScore(unittest.TestCase):
    def _identity_module(self, hidden_dim=16, num_kv_heads=2, head_dim=8):
        return _FakeAttnModule(
            hidden_dim=hidden_dim, num_heads=4, head_dim=head_dim,
            num_kv_heads=num_kv_heads, identity_k=True,
        )

    def test_z_normalization_and_selection_invariance(self):
        module = self._identity_module()
        torch.manual_seed(11)
        K = torch.randn(1, 2, 12, 8)
        hidden = _hidden_for_keys(K)
        garbage_keys = torch.randn(1, 2, 12, 8)
        garbage_values = torch.randn(1, 2, 12, 8)
        sketch = LeverageScoreSketch(compression_ratio=0.5, sketch_dimension=4)
        torch.manual_seed(0)
        z = sketch.score(module, hidden, garbage_keys, garbage_values, None, {})
        torch.manual_seed(0)
        raw = LeverageScoreSketch.compute_leverage_scores(K, 4)
        self.assertEqual(z.shape, (1, 2, 12))
        # Properties of a GLOBAL z-normalization (not per-head): whole-tensor
        # moments are normalized, and the ranking of the un-normalized
        # leverage scores is preserved.  (Deliberately not re-applying the
        # normalization formula here — that would just mirror the
        # implementation.)
        self.assertAlmostEqual(z.mean().item(), 0.0, places=4)
        self.assertAlmostEqual(z.std().item(), 1.0, places=4)
        self.assertTrue(torch.equal(z.topk(6, dim=-1).indices, raw.topk(6, dim=-1).indices))

    def test_score_uses_hidden_states_not_cached_keys(self):
        module = self._identity_module()
        torch.manual_seed(11)
        K = torch.randn(1, 2, 12, 8)
        hidden = _hidden_for_keys(K)
        rotated_a = torch.randn(1, 2, 12, 8)
        rotated_b = torch.randn(1, 2, 12, 8)
        values = torch.randn(1, 2, 12, 8)
        sketch = LeverageScoreSketch(compression_ratio=0.5, sketch_dimension=4)
        torch.manual_seed(0)
        z_a = sketch.score(module, hidden, rotated_a, values, None, {})
        torch.manual_seed(0)
        z_b = sketch.score(module, hidden, rotated_b, values, None, {})
        self.assertTrue(torch.equal(z_a, z_b))

    def test_normalization_is_global_not_per_head(self):
        module = self._identity_module()
        torch.manual_seed(3)
        K = torch.randn(1, 2, 12, 8)
        K[:, 1] *= 1e-3
        hidden = _hidden_for_keys(K)
        garbage = torch.randn(1, 2, 12, 8)
        sketch = LeverageScoreSketch(compression_ratio=0.5, sketch_dimension=4)
        torch.manual_seed(0)
        z = sketch.score(module, hidden, garbage, garbage, None, {})
        self.assertGreater(z[0, 0].mean().item(), 0.2)
        self.assertLess(z[0, 1].mean().item(), -0.2)

    def test_identical_tokens_give_zero_z_scores(self):
        module = self._identity_module()
        K = torch.full((1, 2, 10, 8), 0.7)
        hidden = _hidden_for_keys(K)
        garbage = torch.randn(1, 2, 10, 8)
        sketch = LeverageScoreSketch(compression_ratio=0.5, sketch_dimension=4)
        torch.manual_seed(0)
        z = sketch.score(module, hidden, garbage, garbage, None, {})
        torch.testing.assert_close(z, torch.zeros(1, 2, 10))

    def test_prefill_only_assert(self):
        module = self._identity_module()
        hidden = torch.randn(1, 12, 16)
        keys = torch.randn(1, 2, 15, 8)
        sketch = LeverageScoreSketch(compression_ratio=0.5, sketch_dimension=4)
        with self.assertRaisesRegex(AssertionError, "prefill"):
            sketch.score(module, hidden, keys, keys, None, {})


class TestLeverageCompress(unittest.TestCase):
    def test_zero_ratio_noop_and_score_never_called(self):
        module = _FakeAttnModule()
        sketch = LeverageScoreSketch(compression_ratio=0.0)

        def _boom(*args, **kwargs):
            raise AssertionError("score must not be called at ratio 0")

        sketch.score = _boom
        hidden = torch.randn(1, 6, 32)
        keys = torch.randn(1, 2, 6, 8)
        values = torch.randn(1, 2, 6, 8)
        k_out, v_out = sketch.compress(module, hidden, keys, values, None, {})
        self.assertIs(k_out, keys)
        self.assertIs(v_out, values)

    def test_leverage_outlier_is_retained(self):
        module = _FakeAttnModule(hidden_dim=8, num_heads=1, head_dim=8, num_kv_heads=1, identity_k=True)
        torch.manual_seed(5)
        K = 0.01 * torch.randn(1, 1, 20, 8)
        K[0, 0, 7] = torch.zeros(8)
        K[0, 0, 7, 0] = 50.0
        hidden = _hidden_for_keys(K)
        rotated_keys = torch.randn(1, 1, 20, 8)
        values = torch.arange(20, dtype=torch.float32).view(1, 1, 20, 1).expand(1, 1, 20, 8).contiguous()
        sketch = LeverageScoreSketch(compression_ratio=0.5, sketch_dimension=4)
        torch.manual_seed(0)
        k_out, v_out = sketch.compress(module, hidden, rotated_keys, values, None, {})
        self.assertEqual(k_out.shape, (1, 1, 10, 8))
        self.assertEqual(v_out.shape, (1, 1, 10, 8))
        kept_positions = v_out[0, 0, :, 0].tolist()
        self.assertEqual(len(kept_positions), int(20 * (1 - 0.5)))
        self.assertIn(7.0, kept_positions)

    def test_gqa_shapes_and_queries_never_read(self):
        module = _FakeAttnModule(hidden_dim=32, num_heads=8, head_dim=8, num_kv_heads=2, poison_q=True)
        torch.manual_seed(9)
        hidden = torch.randn(1, 16, 32)
        keys = torch.randn(1, 2, 16, 8)
        values = torch.randn(1, 2, 16, 8)
        sketch = LeverageScoreSketch(compression_ratio=0.25, sketch_dimension=4)
        torch.manual_seed(0)
        z = sketch.score(module, hidden, keys, values, None, {})
        self.assertEqual(z.shape, (1, 2, 16))
        self.assertTrue(torch.isfinite(z).all())
        torch.manual_seed(0)
        k_out, v_out = sketch.compress(module, hidden, keys, values, None, {})
        self.assertEqual(k_out.shape, (1, 2, 12, 8))
        self.assertEqual(v_out.shape, (1, 2, 12, 8))
        self.assertTrue(torch.isfinite(k_out).all())
        self.assertTrue(torch.isfinite(v_out).all())


class TestPreRopeKeyStates(unittest.TestCase):
    def test_llama_branch_matches_manual_projection(self):
        module = _FakeAttnModule(hidden_dim=32, num_heads=4, head_dim=8, num_kv_heads=2)
        torch.manual_seed(4)
        hidden = torch.randn(1, 6, 32)
        out = _get_prerope_key_states(module, hidden)
        expected = module.k_proj(hidden).view(1, 6, 2, 8).transpose(1, 2)
        self.assertEqual(out.shape, (1, 2, 6, 8))
        torch.testing.assert_close(out, expected)

    def test_phi3_branch_slices_fused_qkv(self):
        module = _FakePhi3Module(hidden_dim=16, num_heads=4, head_dim=4, num_kv_heads=2)
        torch.manual_seed(4)
        hidden = torch.randn(1, 5, 16)
        out = _get_prerope_key_states(module, hidden)
        qkv = module.qkv_proj(hidden)
        q_pos = 4 * 4
        expected = qkv[..., q_pos : q_pos + 2 * 4].view(1, 5, 2, 4).transpose(1, 2)
        self.assertEqual(out.shape, (1, 2, 5, 4))
        torch.testing.assert_close(out, expected)

    def test_k_norm_duck_typing_applied(self):
        module = _FakeAttnModule(hidden_dim=16, num_heads=4, head_dim=8, num_kv_heads=2, identity_k=True)
        torch.manual_seed(4)
        hidden = torch.randn(1, 6, 16)
        base = _get_prerope_key_states(module, hidden)
        module.k_norm = lambda x: 2.0 * x
        doubled = _get_prerope_key_states(module, hidden)
        torch.testing.assert_close(doubled, 2.0 * base)

    def test_unsupported_module_raises(self):
        module = SimpleNamespace(head_dim=8)
        with self.assertRaises(NotImplementedError):
            _get_prerope_key_states(module, torch.randn(1, 4, 16))


class TestLeverageHookIntegration(unittest.TestCase):
    def _distinct_rows(self, S, D):
        return (torch.arange(S, dtype=torch.float32).view(1, 1, S, 1) * 100.0
                + torch.arange(D, dtype=torch.float32).view(1, 1, 1, D)).contiguous()

    def test_forward_hook_compresses_cache_via_gather_of_rotated_rows(self):
        from transformers import DynamicCache

        S, D = 16, 8
        module = _FakeAttnModule(hidden_dim=8, num_heads=1, head_dim=8, num_kv_heads=1, identity_k=True)
        torch.manual_seed(2)
        hidden = torch.randn(1, S, 8)
        rotated_keys = self._distinct_rows(S, D)
        values = self._distinct_rows(S, D) + 0.5
        cache = DynamicCache()
        cache.update(rotated_keys.clone(), values.clone(), 0)
        sketch = LeverageScoreSketch(compression_ratio=0.5, sketch_dimension=4)
        output = (torch.zeros(1, S, 8), None)
        torch.manual_seed(0)
        result = sketch.forward_hook(
            module, [],
            {"hidden_states": hidden, "past_key_values": cache, "cache_position": torch.arange(S)},
            output,
        )
        self.assertIs(result, output)
        self.assertEqual(cache.layers[0].keys.shape, (1, 1, int(S * 0.5), D))
        self.assertEqual(cache.layers[0].values.shape, (1, 1, int(S * 0.5), D))
        for row in cache.layers[0].keys[0, 0]:
            self.assertTrue(any(torch.equal(row, orig) for orig in rotated_keys[0, 0]))
        for row in cache.layers[0].values[0, 0]:
            self.assertTrue(any(torch.equal(row, orig) for orig in values[0, 0]))

    def test_forward_hook_noop_on_decoding_step(self):
        from transformers import DynamicCache

        S, D = 16, 8
        module = _FakeAttnModule(hidden_dim=8, num_heads=1, head_dim=8, num_kv_heads=1, identity_k=True)
        cache = DynamicCache()
        cache.update(self._distinct_rows(S + 1, D), self._distinct_rows(S + 1, D), 0)
        sketch = LeverageScoreSketch(compression_ratio=0.5, sketch_dimension=4)
        output = (torch.zeros(1, 1, 8), None)
        sketch.forward_hook(
            module, [],
            {"hidden_states": torch.randn(1, 1, 8), "past_key_values": cache,
             "cache_position": torch.tensor([S])},
            output,
        )
        self.assertEqual(cache.layers[0].keys.shape[2], S + 1)


class TestLeverageRegistry(unittest.TestCase):
    def test_registry_resolution(self):
        self.assertIn("leverage", available_kv_compressors())
        self.assertIs(get_kv_compressor_class("leverage"), LeverageScoreSketch)

    def test_get_kv_compressor_round_trips_fields(self):
        sketch = get_kv_compressor("leverage", compression_ratio=0.5, sketch_dimension=64)
        self.assertIsInstance(sketch, LeverageScoreSketch)
        self.assertAlmostEqual(sketch.compression_ratio, 0.5)
        self.assertEqual(sketch.sketch_dimension, 64)

    def test_default_sketch_dimension(self):
        sketch = get_kv_compressor("leverage", compression_ratio=0.25)
        self.assertEqual(sketch.sketch_dimension, 48)
        self.assertAlmostEqual(sketch.compression_ratio, 0.25)


if __name__ == "__main__":
    unittest.main()
