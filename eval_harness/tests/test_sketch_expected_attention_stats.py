"""Tests for ExpectedAttentionStatsSketch (port of kvpress ExpectedAttentionStatsPress).

The kvpress math (apply_avg_rope + score with injected per-layer statistics)
is re-transcribed locally with explicit loops as a reference oracle; no
kvpress import and no hub access (stats are injected by attribute assignment,
loaded from a temp folder, or mocked).
"""

import math
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch import nn

from eval_harness.kv_compression.compressors.expected_attention_sketch import ExpectedAttentionSketch
from eval_harness.kv_compression.compressors.expected_attention_stats_sketch import (
    ExpectedAttentionStats,
    ExpectedAttentionStatsSketch,
)
from eval_harness.kv_compression.registry import get_kv_compressor, get_kv_compressor_class


class _StubRotary:
    """Stands in for ``module.rotary_emb``; records received position_ids."""

    def __init__(self, head_dim=None, base=10000.0, fixed_cos=None, fixed_sin=None):
        self.head_dim = head_dim
        self.base = base
        self.fixed_cos = fixed_cos
        self.fixed_sin = fixed_sin
        self.captured_position_ids = None

    def __call__(self, x, position_ids):
        self.captured_position_ids = position_ids
        if self.fixed_cos is not None:
            return (
                self.fixed_cos.to(x.dtype).unsqueeze(0),
                self.fixed_sin.to(x.dtype).unsqueeze(0),
            )
        half = self.head_dim // 2
        inv_freq = 1.0 / (self.base ** (torch.arange(0, half, dtype=torch.float64) / half))
        freqs = torch.einsum("s,d->sd", position_ids[0].to(torch.float64), inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(x.dtype).unsqueeze(0), emb.sin().to(x.dtype).unsqueeze(0)


def _identity_rotary(n_positions, head_dim):
    return _StubRotary(
        fixed_cos=torch.ones(n_positions, head_dim),
        fixed_sin=torch.zeros(n_positions, head_dim),
    )


class _FakeStatsAttn(nn.Module):
    """Minimal module for the stats path: no q_proj needed (stats are injected)."""

    def __init__(self, num_heads=1, head_dim=2, layer_idx=0, rotary=None):
        super().__init__()
        self.head_dim = head_dim
        self.layer_idx = layer_idx
        self.config = SimpleNamespace(num_attention_heads=num_heads)
        self.rotary_emb = rotary


def _ref_rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _expected_scores_loop(mu, cov, keys, values, *, n_sink, use_covariance,
                          use_vnorm, epsilon):
    """Explicit-loop transcription of the inherited score with already-rotated
    per-layer statistics (identity-RoPE tests pass the raw stats directly).

    mu: [H_q, d]; cov: [H_q, d, d]; keys/values: [B, H_kv, S, d].
    """
    B, num_kv_heads, S, d = keys.shape
    num_heads = mu.shape[0]
    groups = num_heads // num_kv_heads
    kt = keys[:, :, n_sink:]
    vt = values[:, :, n_sink:]
    s_keys = kt.shape[2]
    logits = torch.zeros(B, num_heads, s_keys, dtype=keys.dtype)
    for b in range(B):
        for hq in range(num_heads):
            kv = hq // groups
            for n in range(s_keys):
                kvec = kt[b, kv, n]
                val = torch.dot(kvec, mu[hq]) / math.sqrt(d)
                if use_covariance:
                    val = val + (kvec @ cov[hq] @ kvec) / d / 2
                logits[b, hq, n] = val
    probs = torch.softmax(logits, dim=-1)
    scores = torch.zeros(B, num_kv_heads, s_keys, dtype=probs.dtype)
    for b in range(B):
        for kv in range(num_kv_heads):
            for g in range(groups):
                scores[b, kv] += probs[b, kv * groups + g]
    scores = scores / groups
    if use_vnorm:
        scores = (scores + epsilon) * vt.norm(dim=-1)
    return F.pad(scores, (n_sink, 0), value=scores.max().item())


def _make_stats_container(num_layers=2, num_heads=2, head_dim=4, seed=0):
    stats = ExpectedAttentionStats(
        num_layers=num_layers, num_heads=num_heads, head_dim=head_dim,
        dataset_name="d", model_name="m", num_samples=1, sample_seq_len=8, n_sink=1,
    )
    torch.manual_seed(seed)
    stats.query_mean.data = torch.randn(num_layers, num_heads, head_dim)
    stats.query_cov.data = torch.randn(num_layers, num_heads, head_dim, head_dim)
    return stats


class TestRegistry(unittest.TestCase):
    def test_registered_name(self):
        self.assertIs(get_kv_compressor_class("expected_attention_stats"), ExpectedAttentionStatsSketch)

    def test_subclasses_base_port(self):
        self.assertTrue(issubclass(ExpectedAttentionStatsSketch, ExpectedAttentionSketch))

    def test_kwargs_construction_and_defaults(self):
        sketch = get_kv_compressor("expected_attention_stats", compression_ratio=0.3, stats_folder="/tmp/x")
        self.assertIsInstance(sketch, ExpectedAttentionStatsSketch)
        self.assertAlmostEqual(sketch.compression_ratio, 0.3)
        self.assertEqual(sketch.stats_folder, "/tmp/x")
        self.assertEqual(sketch.sample_seq_len, 1000)
        self.assertEqual(sketch.num_samples, 100)
        self.assertEqual(sketch.dataset_name, "kmfoda/booksum")
        self.assertIsNone(sketch.mu)
        self.assertIsNone(sketch.cov)


class TestZeroRatioNoop(unittest.TestCase):
    def test_compress_is_identity_and_score_not_called(self):
        sketch = ExpectedAttentionStatsSketch(compression_ratio=0.0)
        sketch.score = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("score must not be called at ratio 0")
        )
        module = _FakeStatsAttn(rotary=_identity_rotary(2, 2))
        keys = torch.randn(1, 1, 6, 2)
        values = torch.randn(1, 1, 6, 2)
        out_k, out_v = sketch.compress(module, torch.zeros(1, 6, 2), keys, values, None, {})
        self.assertIs(out_k, keys)
        self.assertIs(out_v, values)


class TestApplyAvgRopeWithStatsShapes(unittest.TestCase):
    """Stats are un-batched ([H, d] / [H, d, d]); pin the rotation against an
    explicit-loop R construction and the HF rotate_half formula."""

    def test_oracle_with_real_rope(self):
        d, num_heads, n_future, q_len = 4, 3, 5, 9
        sketch = ExpectedAttentionStatsSketch(n_future_positions=n_future)
        rotary = _StubRotary(head_dim=d)
        module = _FakeStatsAttn(num_heads=num_heads, head_dim=d, rotary=rotary)
        torch.manual_seed(1)
        mu = torch.randn(num_heads, d, dtype=torch.float64)
        a = torch.randn(num_heads, d, d, dtype=torch.float64)
        cov = a @ a.transpose(-1, -2)
        mu_out, cov_out = sketch.apply_avg_rope(module, mu.clone(), cov.clone(), q_len=q_len)

        half = d // 2
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, half, dtype=torch.float64) / half))
        P = torch.zeros(d, d, dtype=torch.float64)
        for i in range(d // 2):
            P[i, i + d // 2] = -1.0
            P[i + d // 2, i] = 1.0
        R = torch.zeros(n_future, d, d, dtype=torch.float64)
        for idx, p in enumerate(range(q_len, q_len + n_future)):
            freqs = p * inv_freq
            cos_p = torch.cat([freqs, freqs]).cos()
            sin_p = torch.cat([freqs, freqs]).sin()
            for i in range(d):
                for j in range(d):
                    R[idx, i, j] = cos_p[j] * (1.0 if i == j else 0.0) + sin_p[j] * P[i, j]
        R_bar = R.mean(dim=0)
        torch.testing.assert_close(mu_out, mu @ R_bar.T, rtol=0, atol=1e-12)
        torch.testing.assert_close(
            cov_out, torch.matmul(R_bar, torch.matmul(cov, R_bar.T)), rtol=0, atol=1e-12
        )
        self.assertTrue(
            torch.equal(
                rotary.captured_position_ids,
                torch.arange(q_len, q_len + n_future).unsqueeze(0),
            )
        )

    def test_single_position_matches_hf_rotate_half_formula(self):
        d, q_len = 8, 11
        sketch = ExpectedAttentionStatsSketch(n_future_positions=1)
        rotary = _StubRotary(head_dim=d)
        module = _FakeStatsAttn(num_heads=2, head_dim=d, rotary=rotary)
        torch.manual_seed(2)
        mu = torch.randn(2, d, dtype=torch.float64)
        mu_out, _ = sketch.apply_avg_rope(module, mu.clone(), None, q_len=q_len)
        half = d // 2
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, half, dtype=torch.float64) / half))
        freqs = q_len * inv_freq
        cos_p = torch.cat([freqs, freqs]).cos()
        sin_p = torch.cat([freqs, freqs]).sin()
        expected = mu * cos_p + _ref_rotate_half(mu) * sin_p
        torch.testing.assert_close(mu_out, expected, rtol=0, atol=1e-12)


class TestMeanOnlyValuePinned(unittest.TestCase):
    def _setup(self):
        sketch = ExpectedAttentionStatsSketch(
            compression_ratio=0.5, n_future_positions=2, n_sink=1,
            use_covariance=False, use_vnorm=False,
        )
        sketch.mu = torch.tensor([[[1.0, 0.0]]])
        sketch.cov = torch.zeros(1, 1, 2, 2)
        module = _FakeStatsAttn(num_heads=1, head_dim=2, rotary=_identity_rotary(2, 2))
        hidden = torch.zeros(1, 4, 2)
        keys = torch.tensor([[9.0, 9.0], [2.0, 0.0], [0.0, 2.0], [1.0, 1.0]]).view(1, 1, 4, 2)
        values = torch.tensor([[5.0, 5.0], [1.0, 0.0], [0.0, 2.0], [3.0, 0.0]]).view(1, 1, 4, 2)
        return sketch, module, hidden, keys, values

    def test_score_values(self):
        sketch, module, hidden, keys, values = self._setup()
        scores = sketch.score(module, hidden, keys, values, None, {})
        p = torch.softmax(torch.tensor([2.0, 0.0, 1.0]) / math.sqrt(2.0), dim=-1)
        expected = torch.cat([p.max().unsqueeze(0), p]).view(1, 1, 4)
        torch.testing.assert_close(scores, expected, rtol=0, atol=1e-6)

    def test_compress_keeps_sink_and_argmax(self):
        sketch, module, hidden, keys, values = self._setup()
        out_k, out_v = sketch.compress(module, hidden, keys, values, None, {})
        self.assertEqual(out_k.shape, (1, 1, 2, 2))
        rows = sorted(out_k[0, 0].tolist())
        self.assertEqual(rows, [[2.0, 0.0], [9.0, 9.0]])
        v_rows = sorted(out_v[0, 0].tolist())
        self.assertEqual(v_rows, [[1.0, 0.0], [5.0, 5.0]])


class TestCovarianceTerm(unittest.TestCase):
    def _run(self, cov_mat):
        sketch = ExpectedAttentionStatsSketch(
            compression_ratio=0.5, n_future_positions=2, n_sink=1,
            use_covariance=True, use_vnorm=False,
        )
        sketch.mu = torch.tensor([[[1.0, 0.0]]])
        sketch.cov = cov_mat.view(1, 1, 2, 2)
        module = _FakeStatsAttn(num_heads=1, head_dim=2, rotary=_identity_rotary(2, 2))
        hidden = torch.zeros(1, 4, 2)
        keys = torch.tensor([[9.0, 9.0], [2.0, 0.0], [0.0, 2.0], [1.0, 1.0]]).view(1, 1, 4, 2)
        values = torch.randn(1, 1, 4, 2)
        got = sketch.score(module, hidden, keys, values, None, {})
        oracle = _expected_scores_loop(
            sketch.mu[0], sketch.cov[0], keys, values,
            n_sink=1, use_covariance=True, use_vnorm=False, epsilon=0.0,
        )
        torch.testing.assert_close(got, oracle, rtol=0, atol=1e-6)
        return got

    def test_diagonal_covariance(self):
        got = self._run(torch.diag(torch.tensor([4.0, 1.0])))
        # logit_n = mu.k_n / sqrt(2) + (4 k0^2 + k1^2) / (2 * 2)
        logits = torch.tensor(
            [
                2.0 / math.sqrt(2.0) + (4 * 4.0 + 0.0) / 4.0,
                0.0 / math.sqrt(2.0) + (0.0 + 4.0) / 4.0,
                1.0 / math.sqrt(2.0) + (4 * 1.0 + 1.0) / 4.0,
            ]
        )
        p = torch.softmax(logits, dim=-1)
        expected = torch.cat([p.max().unsqueeze(0), p]).view(1, 1, 4)
        torch.testing.assert_close(got, expected, rtol=0, atol=1e-6)

    def test_non_diagonal_covariance(self):
        got = self._run(torch.tensor([[2.0, 1.0], [1.0, 3.0]]))
        # k^T cov k for rows [2,0],[0,2],[1,1] -> 8, 12, 7; term = /4
        logits = torch.tensor(
            [
                2.0 / math.sqrt(2.0) + 8.0 / 4.0,
                0.0 + 12.0 / 4.0,
                1.0 / math.sqrt(2.0) + 7.0 / 4.0,
            ]
        )
        p = torch.softmax(logits, dim=-1)
        expected = torch.cat([p.max().unsqueeze(0), p]).view(1, 1, 4)
        torch.testing.assert_close(got, expected, rtol=0, atol=1e-6)


class TestVnormEpsilonPadOrdering(unittest.TestCase):
    def test_pad_value_is_max_of_rescaled_scores(self):
        sketch = ExpectedAttentionStatsSketch(
            compression_ratio=0.5, n_future_positions=2, n_sink=1,
            use_covariance=False, use_vnorm=True, epsilon=0.5,
        )
        sketch.mu = torch.tensor([[[1.0, 0.0]]])
        sketch.cov = torch.zeros(1, 1, 2, 2)
        module = _FakeStatsAttn(num_heads=1, head_dim=2, rotary=_identity_rotary(2, 2))
        hidden = torch.zeros(1, 4, 2)
        keys = torch.tensor([[9.0, 9.0], [2.0, 0.0], [0.0, 2.0], [1.0, 1.0]]).view(1, 1, 4, 2)
        values = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [3.0, 0.0]]).view(1, 1, 4, 2)
        scores = sketch.score(module, hidden, keys, values, None, {})

        p = torch.softmax(torch.tensor([2.0, 0.0, 1.0]) / math.sqrt(2.0), dim=-1)
        rescaled = (p + 0.5) * torch.tensor([1.0, 2.0, 3.0])
        expected = torch.cat([rescaled.max().unsqueeze(0), rescaled]).view(1, 1, 4)
        torch.testing.assert_close(scores, expected, rtol=0, atol=1e-6)
        # The sink pad equals the max of the vnorm-RESCALED scores (pad is
        # applied last), not the max softmax probability.
        self.assertAlmostEqual(scores[0, 0, 0].item(), rescaled[2].item(), places=5)
        self.assertNotAlmostEqual(scores[0, 0, 0].item(), p.max().item(), places=2)


class TestGQA(unittest.TestCase):
    def test_kv_head_major_grouping(self):
        num_heads, num_kv_heads, d, S = 4, 2, 2, 6
        sketch = ExpectedAttentionStatsSketch(
            compression_ratio=0.5, n_future_positions=2, n_sink=1,
            use_covariance=False, use_vnorm=False,
        )
        sketch.mu = torch.tensor(
            [[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]]]
        )
        sketch.cov = torch.zeros(1, num_heads, d, d)
        module = _FakeStatsAttn(num_heads=num_heads, head_dim=d, rotary=_identity_rotary(2, d))
        hidden = torch.zeros(1, S, 4)
        torch.manual_seed(6)
        keys = torch.randn(1, num_kv_heads, S, d)
        keys[:, 1] = keys[:, 1] * 3.0 - 5.0
        values = torch.randn(1, num_kv_heads, S, d)
        got = sketch.score(module, hidden, keys, values, None, {})
        self.assertEqual(got.shape, (1, num_kv_heads, S))
        oracle = _expected_scores_loop(
            sketch.mu[0], sketch.cov[0], keys, values,
            n_sink=1, use_covariance=False, use_vnorm=False, epsilon=0.0,
        )
        torch.testing.assert_close(got, oracle, rtol=0, atol=1e-6)

        # A wrong (interleaved) grouping must give a different answer.
        wrong_mu = sketch.mu[0][[0, 2, 1, 3]]
        wrong = _expected_scores_loop(
            wrong_mu, sketch.cov[0], keys, values,
            n_sink=1, use_covariance=False, use_vnorm=False, epsilon=0.0,
        )
        self.assertFalse(torch.allclose(got, wrong, atol=1e-4))


class TestLayerIndexing(unittest.TestCase):
    def _sketch(self):
        sketch = ExpectedAttentionStatsSketch(
            compression_ratio=0.5, n_future_positions=2, n_sink=1,
            use_covariance=False, use_vnorm=False,
        )
        sketch.mu = torch.tensor([[[1.0, 0.0]], [[2.0, 0.0]], [[3.0, 0.0]]])
        sketch.cov = torch.zeros(3, 1, 2, 2)
        return sketch

    def test_layer_idx_selects_row(self):
        sketch = self._sketch()
        hidden = torch.zeros(1, 4, 2)
        keys = torch.tensor([[9.0, 9.0], [2.0, 0.0], [0.0, 2.0], [1.0, 1.0]]).view(1, 1, 4, 2)
        values = torch.randn(1, 1, 4, 2)
        module2 = _FakeStatsAttn(num_heads=1, head_dim=2, layer_idx=2, rotary=_identity_rotary(2, 2))
        got2 = sketch.score(module2, hidden, keys, values, None, {})
        p = torch.softmax(torch.tensor([3.0 * 2.0, 0.0, 3.0 * 1.0]) / math.sqrt(2.0), dim=-1)
        expected = torch.cat([p.max().unsqueeze(0), p]).view(1, 1, 4)
        torch.testing.assert_close(got2, expected, rtol=0, atol=1e-6)

        module0 = _FakeStatsAttn(num_heads=1, head_dim=2, layer_idx=0, rotary=_identity_rotary(2, 2))
        got0 = sketch.score(module0, hidden, keys, values, None, {})
        self.assertFalse(torch.allclose(got0, got2, atol=1e-4))


class TestEdgeCases(unittest.TestCase):
    def test_seq_len_equal_to_n_sink_raises(self):
        sketch = ExpectedAttentionStatsSketch(compression_ratio=0.5, n_sink=4)
        module = _FakeStatsAttn(rotary=_identity_rotary(2, 2))
        keys = torch.randn(1, 1, 4, 2)
        values = torch.randn(1, 1, 4, 2)
        with self.assertRaisesRegex(AssertionError, "n_sink"):
            sketch.score(module, torch.zeros(1, 4, 2), keys, values, None, {})

    def test_small_budget_shapes(self):
        sketch = ExpectedAttentionStatsSketch(
            compression_ratio=0.4, n_future_positions=2, n_sink=4,
            use_covariance=False, use_vnorm=False,
        )
        sketch.mu = torch.tensor([[[1.0, 0.0]]])
        sketch.cov = torch.zeros(1, 1, 2, 2)
        module = _FakeStatsAttn(num_heads=1, head_dim=2, rotary=_identity_rotary(2, 2))
        keys = torch.randn(1, 1, 5, 2)
        values = torch.randn(1, 1, 5, 2)
        out_k, out_v = sketch.compress(module, torch.zeros(1, 5, 2), keys, values, None, {})
        self.assertEqual(out_k.shape, (1, 1, 3, 2))
        self.assertEqual(out_v.shape, (1, 1, 3, 2))

    def test_degenerate_budget_empties_cache(self):
        # kvpress parity: n_kept = int(8 * 0.1) = 0 -> empty compressed cache
        # (decode would break; config-level concern, not the sketch's).
        sketch = ExpectedAttentionStatsSketch(
            compression_ratio=0.9, n_future_positions=2, n_sink=4,
            use_covariance=False, use_vnorm=False,
        )
        sketch.mu = torch.tensor([[[1.0, 0.0]]])
        sketch.cov = torch.zeros(1, 1, 2, 2)
        module = _FakeStatsAttn(num_heads=1, head_dim=2, rotary=_identity_rotary(2, 2))
        keys = torch.randn(1, 1, 8, 2)
        values = torch.randn(1, 1, 8, 2)
        out_k, out_v = sketch.compress(module, torch.zeros(1, 8, 2), keys, values, None, {})
        self.assertEqual(out_k.shape, (1, 1, 0, 2))
        self.assertEqual(out_v.shape, (1, 1, 0, 2))


class TestBatchBroadcast(unittest.TestCase):
    def test_batch_dim_broadcasts_over_stats(self):
        sketch = ExpectedAttentionStatsSketch(
            compression_ratio=0.5, n_future_positions=2, n_sink=1,
        )
        torch.manual_seed(8)
        sketch.mu = torch.randn(1, 2, 2)
        a = torch.randn(1, 2, 2, 2)
        sketch.cov = a @ a.transpose(-1, -2)
        module = _FakeStatsAttn(num_heads=2, head_dim=2, rotary=_identity_rotary(2, 2))
        keys = torch.randn(2, 2, 5, 2)
        values = torch.randn(2, 2, 5, 2)
        scores = sketch.score(module, torch.zeros(2, 5, 2), keys, values, None, {})
        self.assertEqual(scores.shape, (2, 2, 5))
        self.assertFalse(torch.allclose(scores[0], scores[1], atol=1e-4))


class TestPostInitFromModel(unittest.TestCase):
    def test_stats_folder_roundtrip_and_load_once(self):
        stats = _make_stats_container(num_layers=2, num_heads=2, head_dim=4, seed=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            stats.save_pretrained(tmpdir)
            sketch = ExpectedAttentionStatsSketch(stats_folder=tmpdir)
            model = SimpleNamespace(device=torch.device("cpu"), dtype=torch.float32)
            sketch.post_init_from_model(model)
            self.assertEqual(sketch.mu.shape, (2, 2, 4))
            self.assertEqual(sketch.cov.shape, (2, 2, 4, 4))
            self.assertEqual(sketch.mu.dtype, torch.float32)
            torch.testing.assert_close(sketch.mu, stats.query_mean.data)
            torch.testing.assert_close(sketch.cov, stats.query_cov.data)

            sketch.mu = torch.full_like(sketch.mu, 7.0)
            sketch.post_init_from_model(model)
            self.assertTrue((sketch.mu == 7.0).all(), "load-once guard must not reload")

    def test_injected_stats_skip_loading(self):
        sketch = ExpectedAttentionStatsSketch()
        sketch.mu = torch.ones(1, 1, 2)
        sketch.cov = torch.ones(1, 1, 2, 2)
        sketch.post_init_from_model(SimpleNamespace())  # would crash if it tried to load
        self.assertTrue((sketch.mu == 1.0).all())

    def test_hub_path_uses_stats_id_and_casts_dtype(self):
        container = _make_stats_container(num_layers=2, num_heads=2, head_dim=4, seed=4)
        fake_model = SimpleNamespace(
            config=SimpleNamespace(
                name_or_path="meta-llama/Llama-3.1-8B-Instruct",
                num_hidden_layers=2, num_attention_heads=2, head_dim=4,
            ),
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
        )
        sketch = ExpectedAttentionStatsSketch()
        with patch.object(ExpectedAttentionStats, "from_pretrained", return_value=container) as fp:
            sketch.post_init_from_model(fake_model)
        fp.assert_called_once_with(
            "alessiodevoto/exp_att_stats_meta-llama_Llama-3.1-8B-Instruct_kmfoda_booksum_100_1000_4"
        )
        self.assertEqual(sketch.mu.dtype, torch.bfloat16)
        self.assertEqual(sketch.cov.dtype, torch.bfloat16)

    def test_hub_path_missing_stats_raises_value_error(self):
        fake_model = SimpleNamespace(
            config=SimpleNamespace(
                name_or_path="meta-llama/Meta-Llama-3-8B",
                num_hidden_layers=2, num_attention_heads=2, head_dim=4,
            ),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        sketch = ExpectedAttentionStatsSketch()
        with patch.object(ExpectedAttentionStats, "from_pretrained", side_effect=ValueError("404")):
            with self.assertRaisesRegex(ValueError, "No statistics found"):
                sketch.post_init_from_model(fake_model)


class TestStatsContainer(unittest.TestCase):
    def test_stats_id_naming_parity(self):
        stats = ExpectedAttentionStats(
            num_layers=32, num_heads=32, head_dim=128,
            dataset_name="kmfoda/booksum",
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            num_samples=100, sample_seq_len=1000, n_sink=4,
        )
        self.assertEqual(
            stats.stats_id(),
            "alessiodevoto/exp_att_stats_meta-llama_Llama-3.1-8B-Instruct_kmfoda_booksum_100_1000_4",
        )

    def test_parameter_shapes(self):
        stats = ExpectedAttentionStats(
            num_layers=3, num_heads=5, head_dim=6, dataset_name="d",
            model_name="m", num_samples=1, sample_seq_len=8, n_sink=1,
        )
        self.assertEqual(stats.query_mean.shape, (3, 5, 6))
        self.assertEqual(stats.query_cov.shape, (3, 5, 6, 6))


if __name__ == "__main__":
    unittest.main()
