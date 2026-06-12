"""Tests for RidgeSketch (kvpress RidgePress port).

Reference oracle: in-test transcription of the kvpress ``RidgePress.compress``
default path (ridge_press.py:691-858 with fixed_envelope/topk defaults) —
ridge tau via ``torch.linalg.inv`` of ``K_mid^T K_mid + lambda I``, group-mean
GQA query pooling, mean-normalized query Gram omega, eps-clamped distribution
normalization, ``max(p1, gamma*p2) * ||v||`` scoring, and ascending-sorted
per-row topk over the middle region with sink/local windows pinned.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
from torch import nn

from eval_harness.sketch.sketches.registry import (
    available_sketches,
    get_sketch,
    get_sketch_class,
)
from eval_harness.sketch.sketches.ridge_sketch import RidgeSketch


class _FakeAttnModule(nn.Module):
    """Llama-like fake attention module with q_proj and required attrs.

    With ``identity_q`` (requires hidden_dim == num_heads * head_dim), q_proj
    is the identity so queries equal the reshaped hidden states — letting
    tests control the query Gram exactly.
    """

    def __init__(self, hidden_dim=32, num_heads=4, head_dim=8, num_kv_heads=2,
                 identity_q=False, seed=0):
        super().__init__()
        self.num_heads = num_heads
        self.num_key_value_heads = num_kv_heads
        self.head_dim = head_dim
        self.layer_idx = 0
        torch.manual_seed(seed)
        self.q_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        if identity_q:
            assert hidden_dim == num_heads * head_dim
            with torch.no_grad():
                self.q_proj.weight.copy_(torch.eye(hidden_dim))


class _NoQProjModule(nn.Module):
    """Fake attention module without q_proj (exercises the hasattr guard)."""

    def __init__(self, head_dim=8):
        super().__init__()
        self.head_dim = head_dim
        self.layer_idx = 0


def _ridge_default_reference(module, hidden_states, keys, values, *,
                             compression_ratio, sink_size=4, local_size=28,
                             ridge_lambda=1e-4, envelope_gamma=1.0,
                             value_norm_power=1.0, eps=1e-8):
    """kvpress RidgePress default-path transcription (query-aware fixed_envelope)."""
    B, H_kv, T, D = keys.shape
    sink = min(sink_size, T)
    local = min(local_size, max(0, T - sink))
    mid_start, mid_end = sink, T - local
    mid_len = mid_end - mid_start
    keep_total = max(0, min(int(T * (1.0 - compression_ratio)), T))
    keep_mid = min(max(keep_total - sink - local, 0), mid_len)
    keys_mid = keys[:, :, mid_start:mid_end, :]
    values_mid = values[:, :, mid_start:mid_end, :]

    k = keys_mid.float()
    eye = torch.eye(D, dtype=torch.float32).view(1, 1, D, D)
    inv_reg = torch.linalg.inv(k.transpose(-2, -1) @ k + ridge_lambda * eye)
    tau = ((k @ inv_reg) * k).sum(dim=-1).clamp_min(0.0).to(keys.dtype)

    q = module.q_proj(hidden_states)
    H_q = q.shape[-1] // D
    q = q.view(B, T, H_q, D).transpose(1, 2).contiguous()
    if H_q > H_kv:
        q = q.view(B, H_kv, H_q // H_kv, T, D).mean(dim=2)
    elif H_kv > H_q:
        q = q.repeat_interleave(H_kv // H_q, dim=1)
    q = q.to(keys.dtype).contiguous()
    q_mid = q[:, :, mid_start:mid_end, :]

    km = keys_mid.float()
    qf = q_mid.float()
    G = qf.transpose(-2, -1) @ qf
    G = G / max(q_mid.shape[2], 1)
    omega = ((km @ G) * km).sum(dim=-1).clamp_min(0.0).sqrt().to(keys.dtype)

    p1 = tau.float().clamp_min(eps).clamp_min(eps)
    p1 = p1 / p1.sum(dim=-1, keepdim=True).clamp_min(eps)
    p2 = omega.float().clamp_min(eps).clamp_min(eps)
    p2 = p2 / p2.sum(dim=-1, keepdim=True).clamp_min(eps)
    vnorm = values_mid.float().norm(p=2, dim=-1).clamp_min(eps)
    vweight = vnorm.pow(value_norm_power) if value_norm_power > 0 else torch.ones_like(vnorm)
    scores = torch.maximum(p1, float(envelope_gamma) * p2) * vweight

    flat = scores.float().clamp_min(0.0).reshape(B * H_kv, mid_len)
    idx = torch.topk(flat, k=keep_mid, dim=-1).indices.view(B, H_kv, keep_mid).sort(dim=-1).values
    gather_idx = idx.unsqueeze(-1).expand(-1, -1, -1, D)
    out_k = torch.cat([keys[:, :, :sink, :], keys_mid.gather(2, gather_idx), keys[:, :, mid_end:, :]], dim=2)
    out_v = torch.cat([values[:, :, :sink, :], values_mid.gather(2, gather_idx), values[:, :, mid_end:, :]], dim=2)
    return out_k.contiguous(), out_v.contiguous()


def _rand_inputs(B=1, H_kv=2, T=80, D=8, hidden_dim=32, seed=0):
    torch.manual_seed(seed)
    keys = torch.randn(B, H_kv, T, D)
    values = torch.randn(B, H_kv, T, D)
    hidden = torch.randn(B, T, hidden_dim)
    return keys, values, hidden


class TestRidgeRegistry(unittest.TestCase):
    def test_registered_under_assigned_name(self):
        self.assertIn("ridge", available_sketches())
        self.assertIs(get_sketch_class("ridge"), RidgeSketch)

    def test_get_sketch_accepts_compression_ratio_kwarg(self):
        sketch = get_sketch("ridge", compression_ratio=0.3)
        self.assertIsInstance(sketch, RidgeSketch)
        self.assertAlmostEqual(sketch.compression_ratio, 0.3)


class TestRidgeGuards(unittest.TestCase):
    def test_none_ratio_raises_at_compress(self):
        keys, values, hidden = _rand_inputs()
        sketch = RidgeSketch()
        with self.assertRaisesRegex(ValueError, "compression_ratio"):
            sketch.compress(_NoQProjModule(), hidden, keys, values, None, {})

    def test_invalid_ratio_asserts_at_init(self):
        with self.assertRaises(AssertionError):
            RidgeSketch(compression_ratio=1.0)

    def test_zero_ratio_noop_returns_same_tensors(self):
        keys, values, hidden = _rand_inputs()
        sketch = RidgeSketch(compression_ratio=0.0)
        out_k, out_v = sketch.compress(_NoQProjModule(), hidden, keys, values, None, {})
        self.assertIs(out_k, keys)
        self.assertIs(out_v, values)

    def test_min_tokens_gate_default(self):
        keys, values, hidden = _rand_inputs(T=32)
        sketch = RidgeSketch(compression_ratio=0.5)
        out_k, out_v = sketch.compress(_NoQProjModule(), hidden, keys, values, None, {})
        self.assertIs(out_k, keys)
        self.assertIs(out_v, values)

        keys63, values63, hidden63 = _rand_inputs(T=63)
        out_k, out_v = sketch.compress(_NoQProjModule(), hidden63, keys63, values63, None, {})
        self.assertEqual(out_k.shape[2], 63)
        self.assertTrue(torch.equal(out_k, keys63))

    def test_t64_default_windows_hit_keep_mid_zero_branch(self):
        keys, values, hidden = _rand_inputs(T=64)
        sketch = RidgeSketch(compression_ratio=0.5)
        out_k, out_v = sketch.compress(_NoQProjModule(), hidden, keys, values, None, {})
        self.assertEqual(out_k.shape[2], 32)
        self.assertTrue(torch.equal(out_k, torch.cat([keys[:, :, :4], keys[:, :, 36:]], dim=2)))
        self.assertTrue(torch.equal(out_v, torch.cat([values[:, :, :4], values[:, :, 36:]], dim=2)))

    def test_mid_window_collapse_noop(self):
        keys, values, hidden = _rand_inputs(T=32)
        sketch = RidgeSketch(compression_ratio=0.9, min_tokens_to_compress=0)
        out_k, out_v = sketch.compress(_NoQProjModule(), hidden, keys, values, None, {})
        self.assertIs(out_k, keys)
        self.assertIs(out_v, values)

    def test_window_larger_than_sequence_noop(self):
        keys, values, hidden = _rand_inputs(T=3)
        sketch = RidgeSketch(compression_ratio=0.9, min_tokens_to_compress=0)
        out_k, out_v = sketch.compress(_NoQProjModule(), hidden, keys, values, None, {})
        self.assertIs(out_k, keys)
        self.assertIs(out_v, values)

    def test_keep_mid_zero_under_compression_exceeds_budget(self):
        keys, values, hidden = _rand_inputs(T=100)
        sketch = RidgeSketch(compression_ratio=0.8, min_tokens_to_compress=0)
        out_k, out_v = sketch.compress(_NoQProjModule(), hidden, keys, values, None, {})
        self.assertEqual(out_k.shape[2], 32)
        self.assertGreater(out_k.shape[2], int(100 * (1.0 - 0.8)))
        self.assertTrue(torch.equal(out_k, torch.cat([keys[:, :, :4], keys[:, :, 72:]], dim=2)))
        self.assertTrue(torch.equal(out_v, torch.cat([values[:, :, :4], values[:, :, 72:]], dim=2)))


class TestRidgeTauSelection(unittest.TestCase):
    def _e(self, i):
        v = torch.zeros(2)
        v[i] = 1.0
        return v

    def test_hand_computed_tau_values(self):
        keys_mid = torch.stack([self._e(0), self._e(0), self._e(1)]).view(1, 1, 3, 2)
        sketch = RidgeSketch(compression_ratio=0.35, ridge_lambda=1.0)
        tau = sketch._compute_key_ridge_tau(keys_mid)
        torch.testing.assert_close(tau, torch.tensor([[[1 / 3, 1 / 3, 1 / 2]]]), atol=1e-6, rtol=1e-6)

    def test_hand_computed_tau_selection(self):
        keys = torch.stack(
            [torch.tensor([7.0, 7.0]), self._e(0), self._e(0), self._e(1), torch.tensor([9.0, 9.0])]
        ).view(1, 1, 5, 2)
        values = torch.arange(10, dtype=torch.float32).view(1, 1, 5, 2)
        sketch = RidgeSketch(
            compression_ratio=0.35, ridge_lambda=1.0, sink_size=1, local_size=1,
            min_tokens_to_compress=0, query_aware=False, value_norm_power=0.0,
        )
        out_k, out_v = sketch.compress(_NoQProjModule(head_dim=2), None, keys, values, None, {})
        self.assertEqual(out_k.shape[2], 3)
        expected_idx = torch.tensor([0, 3, 4])
        self.assertTrue(torch.equal(out_k, keys[:, :, expected_idx]))
        self.assertTrue(torch.equal(out_v, values[:, :, expected_idx]))

    def test_value_norm_tiebreak(self):
        keys = torch.stack(
            [torch.tensor([7.0, 7.0]), self._e(0), self._e(0), self._e(0), torch.tensor([9.0, 9.0])]
        ).view(1, 1, 5, 2)
        values = torch.tensor(
            [[3.0, 3.0], [1.0, 0.0], [0.0, 5.0], [2.0, 0.0], [4.0, 4.0]]
        ).view(1, 1, 5, 2)
        sketch = RidgeSketch(
            compression_ratio=0.35, ridge_lambda=1.0, sink_size=1, local_size=1,
            min_tokens_to_compress=0, query_aware=False, value_norm_power=1.0,
        )
        out_k, out_v = sketch.compress(_NoQProjModule(head_dim=2), None, keys, values, None, {})
        expected_idx = torch.tensor([0, 2, 4])
        self.assertTrue(torch.equal(out_k, keys[:, :, expected_idx]))
        self.assertTrue(torch.equal(out_v, values[:, :, expected_idx]))

    def test_missing_qproj_falls_back_to_tau_only(self):
        keys, values, hidden = _rand_inputs(T=40)
        module = _NoQProjModule()
        aware = RidgeSketch(compression_ratio=0.5, sink_size=4, local_size=8,
                            min_tokens_to_compress=0, query_aware=True)
        tau_only = RidgeSketch(compression_ratio=0.5, sink_size=4, local_size=8,
                               min_tokens_to_compress=0, query_aware=False)
        out_aware = aware.compress(module, hidden, keys, values, None, {})
        out_tau = tau_only.compress(module, hidden, keys, values, None, {})
        self.assertTrue(torch.equal(out_aware[0], out_tau[0]))
        self.assertTrue(torch.equal(out_aware[1], out_tau[1]))


class TestRidgeEnvelopeGamma(unittest.TestCase):
    def _setup(self):
        keys = torch.tensor([[5.0, 5.0], [2.0, 0.0], [0.0, 1.0], [7.0, 7.0]]).view(1, 1, 4, 2)
        values = torch.tensor([[1.0, 0.0]] * 4).view(1, 1, 4, 2)
        hidden = torch.tensor([[0.0, 0.0], [0.1, 0.0], [0.0, 3.0], [0.0, 0.0]]).view(1, 4, 2)
        module = _FakeAttnModule(hidden_dim=2, num_heads=1, head_dim=2, num_kv_heads=1, identity_q=True)
        return module, hidden, keys, values

    def _sketch(self, gamma):
        return RidgeSketch(
            compression_ratio=0.25, sink_size=1, local_size=1,
            min_tokens_to_compress=0, envelope_gamma=gamma,
        )

    def test_gamma_zero_recovers_pure_ridge_ordering(self):
        module, hidden, keys, values = self._setup()
        out_k, out_v = self._sketch(0.0).compress(module, hidden, keys, values, None, {})
        lam = 1e-4
        tau = torch.tensor([4.0 / (4.0 + lam), 1.0 / (1.0 + lam)])
        vnorm = values[0, 0, 1:3].norm(dim=-1)
        self.assertEqual((tau * vnorm).argmax().item(), 0)
        expected_idx = torch.tensor([0, 1, 3])
        self.assertTrue(torch.equal(out_k, keys[:, :, expected_idx]))
        self.assertTrue(torch.equal(out_v, values[:, :, expected_idx]))

    def test_gamma_large_query_side_dominates(self):
        module, hidden, keys, values = self._setup()
        for gamma in (1.0, 10.0):
            out_k, out_v = self._sketch(gamma).compress(module, hidden, keys, values, None, {})
            expected_idx = torch.tensor([0, 2, 3])
            self.assertTrue(torch.equal(out_k, keys[:, :, expected_idx]))
            self.assertTrue(torch.equal(out_v, values[:, :, expected_idx]))


class TestRidgeGQA(unittest.TestCase):
    def test_group_mean_pooling_matches_reference(self):
        B, T, H_q, H_kv, D = 1, 6, 4, 2, 4
        module = _FakeAttnModule(hidden_dim=16, num_heads=H_q, head_dim=D, num_kv_heads=H_kv, seed=3)
        torch.manual_seed(11)
        hidden = torch.randn(B, T, 16)
        keys = torch.randn(B, H_kv, T, D)
        sketch = RidgeSketch(compression_ratio=0.5)
        q = sketch._get_all_queries(module, hidden, keys)
        expected = module.q_proj(hidden).view(B, T, H_q, D).transpose(1, 2).contiguous()
        expected = expected.view(B, H_kv, H_q // H_kv, T, D).mean(dim=2).to(keys.dtype).contiguous()
        self.assertEqual(q.shape, (B, H_kv, T, D))
        self.assertTrue(torch.equal(q, expected))

    def test_repeat_interleave_when_kv_heads_exceed_q_heads(self):
        B, T, H_q, H_kv, D = 1, 6, 2, 4, 4
        module = _FakeAttnModule(hidden_dim=16, num_heads=H_q, head_dim=D, num_kv_heads=H_kv, seed=4)
        torch.manual_seed(12)
        hidden = torch.randn(B, T, 16)
        keys = torch.randn(B, H_kv, T, D)
        sketch = RidgeSketch(compression_ratio=0.5)
        q = sketch._get_all_queries(module, hidden, keys)
        expected = module.q_proj(hidden).view(B, T, H_q, D).transpose(1, 2).contiguous()
        expected = expected.repeat_interleave(H_kv // H_q, dim=1).to(keys.dtype).contiguous()
        self.assertEqual(q.shape, (B, H_kv, T, D))
        self.assertTrue(torch.equal(q, expected))

    def test_incompatible_head_counts_disable_query_awareness(self):
        B, T, D = 1, 40, 4
        module = _FakeAttnModule(hidden_dim=16, num_heads=3, head_dim=D, num_kv_heads=2, seed=5)
        torch.manual_seed(13)
        hidden = torch.randn(B, T, 16)
        keys = torch.randn(B, 2, T, D)
        values = torch.randn(B, 2, T, D)
        sketch = RidgeSketch(compression_ratio=0.5, sink_size=4, local_size=8, min_tokens_to_compress=0)
        self.assertIsNone(sketch._get_all_queries(module, hidden, keys))
        tau_only = RidgeSketch(compression_ratio=0.5, sink_size=4, local_size=8,
                               min_tokens_to_compress=0, query_aware=False)
        out_aware = sketch.compress(module, hidden, keys, values, None, {})
        out_tau = tau_only.compress(module, hidden, keys, values, None, {})
        self.assertTrue(torch.equal(out_aware[0], out_tau[0]))
        self.assertTrue(torch.equal(out_aware[1], out_tau[1]))

    def test_non_divisible_qproj_dim_disables_query_awareness(self):
        module = nn.Module()
        module.q_proj = nn.Linear(16, 10, bias=False)
        keys = torch.randn(1, 2, 6, 4)
        hidden = torch.randn(1, 6, 16)
        sketch = RidgeSketch(compression_ratio=0.5)
        self.assertIsNone(sketch._get_all_queries(module, hidden, keys))

    def test_hidden_key_length_mismatch_guard(self):
        # Prism-Test deviation: an outer prefill-method hook can leave the
        # cache shorter than hidden_states; the query-aware path must be
        # skipped instead of silently misaligning query/key positions.
        module = _FakeAttnModule(hidden_dim=16, num_heads=4, head_dim=4, num_kv_heads=2, seed=6)
        torch.manual_seed(14)
        hidden = torch.randn(1, 12, 16)
        keys = torch.randn(1, 2, 10, 4)
        values = torch.randn(1, 2, 10, 4)
        sketch = RidgeSketch(compression_ratio=0.5, sink_size=1, local_size=1, min_tokens_to_compress=0)
        self.assertIsNone(sketch._get_all_queries(module, hidden, keys))
        tau_only = RidgeSketch(compression_ratio=0.5, sink_size=1, local_size=1,
                               min_tokens_to_compress=0, query_aware=False)
        out_aware = sketch.compress(module, hidden, keys, values, None, {})
        out_tau = tau_only.compress(module, hidden, keys, values, None, {})
        self.assertTrue(torch.equal(out_aware[0], out_tau[0]))
        self.assertTrue(torch.equal(out_aware[1], out_tau[1]))

    def test_per_head_selection_diverges_counts_stay_rectangular(self):
        T, D = 12, 2
        e1 = torch.tensor([1.0, 0.0])
        e2 = torch.tensor([0.0, 1.0])
        keys = e1.repeat(1, 2, T, 1).view(1, 2, T, 2).clone()
        keys[0, 0, 4] = e2
        keys[0, 1, 7] = e2
        values = torch.randn(1, 2, T, D)
        sketch = RidgeSketch(
            compression_ratio=0.58, ridge_lambda=1.0, sink_size=2, local_size=2,
            min_tokens_to_compress=0, query_aware=False, value_norm_power=0.0,
        )
        out_k, out_v = sketch.compress(_NoQProjModule(head_dim=2), None, keys, values, None, {})
        self.assertEqual(out_k.shape, (1, 2, 5, D))
        head0_idx = torch.tensor([0, 1, 4, 10, 11])
        head1_idx = torch.tensor([0, 1, 7, 10, 11])
        self.assertTrue(torch.equal(out_k[:, 0], keys[:, 0][:, head0_idx]))
        self.assertTrue(torch.equal(out_k[:, 1], keys[:, 1][:, head1_idx]))
        self.assertTrue(torch.equal(out_v[:, 0], values[:, 0][:, head0_idx]))
        self.assertTrue(torch.equal(out_v[:, 1], values[:, 1][:, head1_idx]))


class TestRidgeReferenceOracle(unittest.TestCase):
    def test_default_fixed_envelope_path_bitwise(self):
        B, H_kv, H_q, T, D, hidden_dim = 2, 2, 4, 96, 8, 32
        module = _FakeAttnModule(hidden_dim=hidden_dim, num_heads=H_q, head_dim=D,
                                 num_kv_heads=H_kv, seed=0)
        torch.manual_seed(0)
        keys = torch.randn(B, H_kv, T, D)
        values = torch.randn(B, H_kv, T, D)
        hidden = torch.randn(B, T, hidden_dim)
        sketch = RidgeSketch(compression_ratio=0.5, sink_size=4, local_size=8,
                             min_tokens_to_compress=0)
        out_k, out_v = sketch.compress(module, hidden, keys, values, None, {})
        ref_k, ref_v = _ridge_default_reference(
            module, hidden, keys, values,
            compression_ratio=0.5, sink_size=4, local_size=8,
        )
        self.assertEqual(out_k.shape, (B, H_kv, 48, D))
        self.assertTrue(torch.equal(out_k, ref_k))
        self.assertTrue(torch.equal(out_v, ref_v))


class TestRidgeWeightedEnvelope(unittest.TestCase):
    def test_disjoint_topk_sets_boost_gamma(self):
        sketch = RidgeSketch(compression_ratio=0.5, combine_mode="weighted_envelope",
                             query_boost_strength=2.0, value_norm_power=0.0)
        tau = torch.tensor([[[10.0, 8.0, 1.0, 1.0]]])
        omega = torch.tensor([[[1.0, 1.0, 10.0, 8.0]]])
        values = torch.randn(1, 1, 4, 2)
        scores = sketch._scores_from_tau_omega_and_values(tau, values, omega=omega, n_keep=2)
        p1 = tau / 20.0
        p2 = omega / 20.0
        expected = torch.maximum(p1, 3.0 * p2)
        torch.testing.assert_close(scores, expected, atol=1e-6, rtol=1e-6)
        idx = sketch._select_indices_from_scores(scores, 2)
        self.assertTrue(torch.equal(idx, torch.tensor([[[2, 3]]])))

    def test_identical_components_reduce_to_plain_envelope(self):
        weighted = RidgeSketch(compression_ratio=0.5, combine_mode="weighted_envelope",
                               query_boost_strength=2.0, value_norm_power=0.0)
        envelope = RidgeSketch(compression_ratio=0.5, combine_mode="envelope",
                               value_norm_power=0.0)
        tau = torch.tensor([[[10.0, 8.0, 1.0, 1.0]]])
        values = torch.randn(1, 1, 4, 2)
        s_weighted = weighted._scores_from_tau_omega_and_values(tau, values, omega=tau.clone(), n_keep=2)
        s_envelope = envelope._scores_from_tau_omega_and_values(tau, values, omega=tau.clone(), n_keep=2)
        self.assertTrue(torch.equal(s_weighted, s_envelope))


class TestRidgeAlphaMachinery(unittest.TestCase):
    def test_tail_risk_picks_min_risk_alpha(self):
        sketch = RidgeSketch(
            compression_ratio=0.5, combine_mode="additive", alpha_selection="tail_risk",
            alpha_grid="0.0,1.0", alpha_validation_split="none", value_norm_power=0.0,
        )
        tau = torch.tensor([[[0.8, 0.15, 0.05]]])
        omega = torch.tensor([[[0.05, 0.9, 0.05]]])
        values = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]]).view(1, 1, 3, 2)
        alpha, risk, ridge_tail, query_tail = sketch._choose_alpha_by_tail_risk(
            tau=tau, omega_score=omega, omega_val=omega, values=values, n_keep=1,
        )
        self.assertEqual(alpha, 0.0)
        self.assertAlmostEqual(risk, 0.85, places=5)
        self.assertAlmostEqual(ridge_tail, 0.85, places=5)
        self.assertAlmostEqual(query_tail, 0.1, places=5)

    def test_tail_risk_alpha_selects_omega_favored_token_end_to_end(self):
        sketch = RidgeSketch(
            compression_ratio=0.5, combine_mode="additive", alpha_selection="tail_risk",
            alpha_grid="0.0,1.0", alpha_validation_split="none", value_norm_power=0.0,
        )
        tau = torch.tensor([[[0.8, 0.15, 0.05]]])
        omega = torch.tensor([[[0.05, 0.9, 0.05]]])
        values = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]]).view(1, 1, 3, 2)
        scores = sketch._scores_from_tau_omega_and_values(tau, values, omega=omega, alpha=0.0)
        idx = sketch._select_indices_from_scores(scores, 1)
        self.assertTrue(torch.equal(idx, torch.tensor([[[1]]])))

    def test_envelope_default_bypasses_alpha_machinery(self):
        module = _FakeAttnModule(hidden_dim=32, num_heads=4, head_dim=8, num_kv_heads=2, seed=1)
        keys, values, hidden = _rand_inputs(T=40, seed=2)
        gated = RidgeSketch(compression_ratio=0.5, sink_size=4, local_size=8,
                            min_tokens_to_compress=0, combine_mode="fixed_envelope",
                            alpha_selection="gated_query_constrained")
        with patch.object(gated, "_choose_alpha_query_constrained",
                          side_effect=AssertionError("alpha machinery must not run")), \
             patch.object(gated, "_ridge_excess_for_alpha",
                          side_effect=AssertionError("alpha machinery must not run")), \
             patch.object(gated, "_choose_alpha_by_tail_risk",
                          side_effect=AssertionError("alpha machinery must not run")):
            out_gated = gated.compress(module, hidden, keys, values, None, {})

        fixed = RidgeSketch(compression_ratio=0.5, sink_size=4, local_size=8,
                            min_tokens_to_compress=0, combine_mode="fixed_envelope",
                            alpha_selection="fixed")
        out_fixed = fixed.compress(module, hidden, keys, values, None, {})
        self.assertTrue(torch.equal(out_gated[0], out_fixed[0]))
        self.assertTrue(torch.equal(out_gated[1], out_fixed[1]))


class TestRidgeSelection(unittest.TestCase):
    def test_multinomial_structure_and_seeded_determinism(self):
        keys, values, hidden = _rand_inputs(T=40, seed=7)
        sketch = RidgeSketch(compression_ratio=0.5, sink_size=4, local_size=8,
                             min_tokens_to_compress=0, query_aware=False,
                             selection_method="multinomial")
        module = _NoQProjModule()
        torch.manual_seed(0)
        out_k1, out_v1 = sketch.compress(module, hidden, keys, values, None, {})
        torch.manual_seed(0)
        out_k2, out_v2 = sketch.compress(module, hidden, keys, values, None, {})
        self.assertEqual(out_k1.shape[2], 20)
        self.assertTrue(torch.equal(out_k1, out_k2))
        self.assertTrue(torch.equal(out_v1, out_v2))
        self.assertTrue(torch.equal(out_k1[:, :, :4], keys[:, :, :4]))
        self.assertTrue(torch.equal(out_k1[:, :, -8:], keys[:, :, 32:]))
        mid_in = keys[:, :, 4:32]
        mid_out = out_k1[:, :, 4:12]
        matches = (mid_out.unsqueeze(3) == mid_in.unsqueeze(2)).all(dim=-1).any(dim=-1)
        self.assertTrue(matches.all())

    def test_multinomial_indices_sorted_unique_in_range(self):
        sketch = RidgeSketch(compression_ratio=0.5, selection_method="multinomial")
        torch.manual_seed(0)
        scores = torch.rand(1, 2, 10) + 0.1
        idx = sketch._select_indices_from_scores(scores, 4)
        self.assertEqual(idx.shape, (1, 2, 4))
        self.assertTrue((idx[..., 1:] > idx[..., :-1]).all())
        self.assertTrue((idx >= 0).all() and (idx < 10).all())

    def test_zero_score_rows_get_uniform_fallback(self):
        sketch = RidgeSketch(compression_ratio=0.5)
        scores = torch.zeros(1, 2, 10)
        scores[0, 1] = torch.arange(10, dtype=torch.float32)
        idx = sketch._select_indices_from_scores(scores, 3)
        self.assertTrue(torch.equal(idx[0, 1], torch.tensor([7, 8, 9])))
        # The zero row is replaced by uniform scores; topk tie order is
        # implementation-defined, so pin only the guaranteed invariants.
        zero_row = idx[0, 0]
        self.assertEqual(zero_row.shape, (3,))
        self.assertTrue((zero_row[1:] > zero_row[:-1]).all())
        self.assertTrue((zero_row >= 0).all() and (zero_row < 10).all())

    def test_select_indices_edge_counts(self):
        sketch = RidgeSketch(compression_ratio=0.5)
        scores = torch.rand(1, 1, 6)
        self.assertEqual(sketch._select_indices_from_scores(scores, 0).shape, (1, 1, 0))
        idx_all = sketch._select_indices_from_scores(scores, 6)
        self.assertTrue(torch.equal(idx_all, torch.arange(6).view(1, 1, 6)))
        idx_over = sketch._select_indices_from_scores(scores, 9)
        self.assertTrue(torch.equal(idx_over, torch.arange(6).view(1, 1, 6)))


class TestRidgeInvariants(unittest.TestCase):
    def test_rectangular_output_across_layers(self):
        sketch = RidgeSketch(compression_ratio=0.5, sink_size=4, local_size=8,
                             min_tokens_to_compress=0, query_aware=False)
        module = _NoQProjModule()
        lengths = []
        for seed in (0, 1):
            keys, values, hidden = _rand_inputs(T=50, seed=seed)
            out_k, out_v = sketch.compress(module, hidden, keys, values, None, {})
            lengths.append((out_k.shape[2], out_v.shape[2]))
        self.assertEqual(lengths[0], lengths[1])
        self.assertEqual(lengths[0], (25, 25))

    def test_bf16_passthrough(self):
        module = _FakeAttnModule(hidden_dim=32, num_heads=4, head_dim=8, num_kv_heads=2, seed=2)
        module.to(torch.bfloat16)
        keys, values, hidden = _rand_inputs(T=80, seed=3)
        keys, values, hidden = keys.bfloat16(), values.bfloat16(), hidden.bfloat16()
        sketch = RidgeSketch(compression_ratio=0.5)
        out_k, out_v = sketch.compress(module, hidden, keys, values, None, {})
        self.assertEqual(out_k.shape[2], 40)
        self.assertEqual(out_k.dtype, torch.bfloat16)
        self.assertEqual(out_v.dtype, torch.bfloat16)

    def test_forward_hook_prefill_compresses_then_decode_noop(self):
        from transformers import DynamicCache

        torch.manual_seed(9)
        B, H_kv, T, D, hidden_dim = 1, 2, 12, 4, 8
        keys = torch.randn(B, H_kv, T, D)
        values = torch.randn(B, H_kv, T, D)
        hidden = torch.randn(B, T, hidden_dim)
        cache = DynamicCache()
        cache.update(keys.clone(), values.clone(), 0)

        module = _FakeAttnModule(hidden_dim=hidden_dim, num_heads=2, head_dim=D,
                                 num_kv_heads=H_kv, seed=8)
        sketch = RidgeSketch(compression_ratio=0.5, sink_size=2, local_size=2,
                             min_tokens_to_compress=0)
        prefill_kwargs = {
            "hidden_states": hidden,
            "past_key_values": cache,
            "cache_position": torch.arange(T),
        }
        output = (torch.randn(B, T, hidden_dim), None)
        result = sketch.forward_hook(module, [], prefill_kwargs, output)
        self.assertIs(result, output)

        expected_k, expected_v = sketch.compress(module, hidden, keys, values, None, {})
        self.assertEqual(cache.layers[0].keys.shape[2], 6)
        self.assertTrue(torch.equal(cache.layers[0].keys, expected_k))
        self.assertTrue(torch.equal(cache.layers[0].values, expected_v))

        kept_k = cache.layers[0].keys.clone()
        kept_v = cache.layers[0].values.clone()
        decode_kwargs = {
            "hidden_states": torch.randn(B, 1, hidden_dim),
            "past_key_values": cache,
            "cache_position": torch.tensor([T]),
        }
        sketch.forward_hook(module, [], decode_kwargs, (torch.randn(B, 1, hidden_dim), None))
        self.assertTrue(torch.equal(cache.layers[0].keys, kept_k))
        self.assertTrue(torch.equal(cache.layers[0].values, kept_v))


if __name__ == "__main__":
    unittest.main()
