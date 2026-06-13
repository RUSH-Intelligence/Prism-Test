"""Tests for ExpectedAttentionSketch (port of kvpress ExpectedAttentionPress).

Two oracles, no kvpress import:
- a verbatim vendored transcription of kvpress's vectorized score math
  (expected_attention_press.py:62-165), pinned bitwise against the port;
- an independent explicit-loop re-implementation (per-element rotation-matrix
  construction, per-key logits, explicit kv-head-major group reduction),
  pinned in float64 at tight tolerance.
"""

import math
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from eval_harness.sketch.sketches.expected_attention_sketch import (
    ExpectedAttentionSketch,
    _get_prerope_query_states,
)
from eval_harness.sketch.sketches.registry import get_sketch, get_sketch_class


class _StubRotary:
    """Stands in for ``module.rotary_emb``; records received position_ids.

    Either returns fixed ``[n, d]`` cos/sin tensors (batch dim added, as
    transformers returns ``[B, n, d]``), or computes real RoPE trig from the
    given positions with the standard inv_freq formula.
    """

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


class _FakeAttn(nn.Module):
    def __init__(self, hidden_dim=8, num_heads=4, head_dim=2, num_kv_heads=2,
                 rotary=None, identity_q=False, seed=0, dtype=torch.float32):
        super().__init__()
        self.head_dim = head_dim
        self.layer_idx = 0
        self.config = SimpleNamespace(
            num_attention_heads=num_heads, num_key_value_heads=num_kv_heads
        )
        self.rotary_emb = rotary
        self.q_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False, dtype=dtype)
        if identity_q:
            assert hidden_dim == num_heads * head_dim
            with torch.no_grad():
                self.q_proj.weight.copy_(torch.eye(hidden_dim))
        else:
            torch.manual_seed(seed)
            with torch.no_grad():
                self.q_proj.weight.normal_()


class _FakeFusedAttn(nn.Module):
    """Phi3-style module exposing only a fused qkv_proj."""

    def __init__(self, hidden_dim=8, num_heads=4, head_dim=2, num_kv_heads=2,
                 rotary=None, seed=0, dtype=torch.float32):
        super().__init__()
        self.head_dim = head_dim
        self.layer_idx = 0
        self.config = SimpleNamespace(
            num_attention_heads=num_heads, num_key_value_heads=num_kv_heads
        )
        self.rotary_emb = rotary
        out_dim = (num_heads + 2 * num_kv_heads) * head_dim
        self.qkv_proj = nn.Linear(hidden_dim, out_dim, bias=False, dtype=dtype)
        torch.manual_seed(seed)
        with torch.no_grad():
            self.qkv_proj.weight.normal_()


class _DoubleQNorm(nn.Module):
    def forward(self, x):
        return x * 2.0


def _ref_rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _ref_repeat_kv(hidden_states, n_rep):
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def _kvpress_score_verbatim(module, hidden_states, keys, values, *, n_sink,
                            n_future_positions, use_covariance, use_vnorm, epsilon):
    """Verbatim transcription of kvpress ExpectedAttentionPress.score
    (expected_attention_press.py:62-165, Llama q_proj branch of
    utils.get_prerope_query_states)."""
    q_len_full = hidden_states.shape[1]
    h = hidden_states[:, n_sink:]
    bsz, h_len, _ = h.shape
    num_heads = module.config.num_attention_heads
    head_dim = module.head_dim

    if hasattr(module, "qkv_proj"):
        query_states = module.qkv_proj(h)[..., : num_heads * head_dim]
    else:
        query_states = module.q_proj(h)
    query_states = query_states.view(bsz, h_len, num_heads, head_dim).transpose(1, 2)
    q_norm = getattr(module, "q_norm", None)
    if q_norm is not None:
        query_states = q_norm(query_states)

    mu = query_states.mean(dim=2, keepdim=True)
    cov = None
    if use_covariance:
        centered_states = query_states - mu
        cov = torch.einsum("bnsi,bnsj->bnij", centered_states, centered_states) / h.shape[1]
    mu = mu.squeeze(2)

    position_ids = torch.arange(q_len_full, q_len_full + n_future_positions).unsqueeze(0).to(mu.device)
    cos, sin = module.rotary_emb(mu, position_ids)
    cos, sin = cos[0], sin[0]
    Id = torch.eye(head_dim, device=cos.device, dtype=cos.dtype)
    P = torch.zeros((head_dim, head_dim), device=cos.device, dtype=cos.dtype)
    P[head_dim // 2 :, : head_dim // 2] = torch.eye(head_dim // 2)
    P[: head_dim // 2, head_dim // 2 :] = -torch.eye(head_dim // 2)
    R = cos.unsqueeze(1) * Id + sin.unsqueeze(1) * P
    R = R.mean(dim=0).to(mu.device)
    mu = torch.matmul(mu, R.T)
    if cov is not None:
        cov = torch.matmul(R, torch.matmul(cov, R.T))

    keys = keys[:, :, n_sink:]
    values = values[:, :, n_sink:]
    bsz, num_key_value_heads, k_len, d = keys.shape
    num_key_value_groups = num_heads // num_key_value_heads

    keys = _ref_repeat_kv(keys, num_key_value_groups).transpose(2, 3)
    scores = torch.matmul(mu.unsqueeze(2), keys).squeeze(2) / math.sqrt(d)
    if use_covariance:
        scores += torch.einsum("bhin, bhij, bhjn->bhn", keys, cov, keys) / d / 2
    scores = F.softmax(scores, dim=-1)
    scores = scores.view(bsz, num_key_value_heads, num_key_value_groups, k_len)
    scores = scores.mean(dim=2)
    if use_vnorm:
        scores = (scores + epsilon) * values.norm(dim=-1)
    scores = F.pad(scores, (n_sink, 0), value=scores.max().item())
    return scores


def _loop_oracle(module, hidden_states, keys, values, *, n_sink, n_future_positions,
                 use_covariance, use_vnorm, epsilon):
    """Independent explicit-loop re-implementation of the kvpress math."""
    num_heads = module.config.num_attention_heads
    d = module.head_dim
    B = hidden_states.shape[0]
    q_len_full = hidden_states.shape[1]

    h = hidden_states[:, n_sink:]
    if hasattr(module, "qkv_proj"):
        q = module.qkv_proj(h)[..., : num_heads * d]
    else:
        q = module.q_proj(h)
    q = q.view(B, h.shape[1], num_heads, d).transpose(1, 2)
    q_norm = getattr(module, "q_norm", None)
    if q_norm is not None:
        q = q_norm(q)

    mu = q.mean(dim=2)
    cov = None
    if use_covariance:
        c = q - mu.unsqueeze(2)
        cov = torch.zeros(B, num_heads, d, d, dtype=q.dtype)
        for b in range(B):
            for n in range(num_heads):
                for s in range(c.shape[2]):
                    cov[b, n] += torch.outer(c[b, n, s], c[b, n, s])
        cov = cov / h.shape[1]

    position_ids = torch.arange(q_len_full, q_len_full + n_future_positions).unsqueeze(0)
    cos, sin = module.rotary_emb(mu, position_ids)
    cos, sin = cos[0], sin[0]
    P = torch.zeros(d, d, dtype=cos.dtype)
    for i in range(d // 2):
        P[i, i + d // 2] = -1.0
        P[i + d // 2, i] = 1.0
    n_pos = cos.shape[0]
    R = torch.zeros(n_pos, d, d, dtype=cos.dtype)
    for p in range(n_pos):
        for i in range(d):
            for j in range(d):
                R[p, i, j] = cos[p, j] * (1.0 if i == j else 0.0) + sin[p, j] * P[i, j]
    R_bar = R.mean(dim=0)
    mu = mu @ R_bar.T
    if cov is not None:
        cov = torch.matmul(R_bar, torch.matmul(cov, R_bar.T))

    kt = keys[:, :, n_sink:]
    vt = values[:, :, n_sink:]
    num_kv_heads = kt.shape[1]
    groups = num_heads // num_kv_heads
    s_keys = kt.shape[2]
    logits = torch.zeros(B, num_heads, s_keys, dtype=mu.dtype)
    for b in range(B):
        for hq in range(num_heads):
            kv = hq // groups
            for n in range(s_keys):
                kvec = kt[b, kv, n]
                val = torch.dot(kvec, mu[b, hq]) / math.sqrt(d)
                if use_covariance:
                    val = val + (kvec @ cov[b, hq] @ kvec) / d / 2
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


class TestRegistry(unittest.TestCase):
    def test_registered_name(self):
        self.assertIs(get_sketch_class("expected_attention"), ExpectedAttentionSketch)

    def test_kwargs_construction(self):
        sketch = get_sketch(
            "expected_attention", compression_ratio=0.25, n_future_positions=7,
            n_sink=2, use_covariance=False, use_vnorm=False, epsilon=0.1,
        )
        self.assertIsInstance(sketch, ExpectedAttentionSketch)
        self.assertAlmostEqual(sketch.compression_ratio, 0.25)
        self.assertEqual(sketch.n_future_positions, 7)
        self.assertEqual(sketch.n_sink, 2)
        self.assertFalse(sketch.use_covariance)
        self.assertFalse(sketch.use_vnorm)
        self.assertAlmostEqual(sketch.epsilon, 0.1)

    def test_defaults_match_kvpress(self):
        sketch = ExpectedAttentionSketch()
        self.assertEqual(sketch.compression_ratio, 0.0)
        self.assertEqual(sketch.n_future_positions, 512)
        self.assertEqual(sketch.n_sink, 4)
        self.assertTrue(sketch.use_covariance)
        self.assertTrue(sketch.use_vnorm)
        self.assertEqual(sketch.epsilon, 0.0)


class TestZeroRatioNoop(unittest.TestCase):
    def test_compress_is_identity_and_score_not_called(self):
        sketch = ExpectedAttentionSketch(compression_ratio=0.0)
        sketch.score = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("score must not be called at ratio 0")
        )
        module = _FakeAttn(rotary=_identity_rotary(2, 2))
        keys = torch.randn(1, 2, 6, 2)
        values = torch.randn(1, 2, 6, 2)
        out_k, out_v = sketch.compress(module, torch.randn(1, 6, 8), keys, values, None, {})
        self.assertIs(out_k, keys)
        self.assertIs(out_v, values)


class TestMeanOnlyHandComputed(unittest.TestCase):
    def _setup(self, n_future=3):
        module = _FakeAttn(
            hidden_dim=2, num_heads=1, head_dim=2, num_kv_heads=1,
            rotary=_identity_rotary(n_future, 2), identity_q=True,
        )
        hidden = torch.tensor([[1.0, 0.0]] * 4).unsqueeze(0)
        keys = torch.tensor([[1.0, 0.0], [2.0, 0.0], [0.0, 5.0], [3.0, 0.0]]).view(1, 1, 4, 2)
        values = torch.tensor([[10.0, 0.0], [11.0, 0.0], [12.0, 0.0], [13.0, 0.0]]).view(1, 1, 4, 2)
        return module, hidden, keys, values

    def test_score_values_and_ranking(self):
        module, hidden, keys, values = self._setup()
        sketch = ExpectedAttentionSketch(
            compression_ratio=0.5, n_future_positions=3, n_sink=0,
            use_covariance=False, use_vnorm=False,
        )
        scores = sketch.score(module, hidden, keys, values, None, {})
        expected = torch.softmax(torch.tensor([1.0, 2.0, 0.0, 3.0]) / math.sqrt(2.0), dim=-1)
        torch.testing.assert_close(scores, expected.view(1, 1, 4), rtol=0, atol=1e-6)
        self.assertEqual(scores[0, 0].argsort(descending=True).tolist(), [3, 1, 0, 2])

    def test_future_position_ids(self):
        module, hidden, keys, values = self._setup(n_future=3)
        sketch = ExpectedAttentionSketch(
            compression_ratio=0.5, n_future_positions=3, n_sink=0,
            use_covariance=False, use_vnorm=False,
        )
        sketch.score(module, hidden, keys, values, None, {})
        self.assertTrue(
            torch.equal(
                module.rotary_emb.captured_position_ids,
                torch.arange(4, 7).unsqueeze(0),
            )
        )

    def test_compress_keeps_top_keys_in_descending_score_order(self):
        module, hidden, keys, values = self._setup()
        sketch = ExpectedAttentionSketch(
            compression_ratio=0.5, n_future_positions=3, n_sink=0,
            use_covariance=False, use_vnorm=False,
        )
        out_k, out_v = sketch.compress(module, hidden, keys, values, None, {})
        self.assertEqual(out_k.shape, (1, 1, 2, 2))
        torch.testing.assert_close(out_k[0, 0], torch.tensor([[3.0, 0.0], [2.0, 0.0]]))
        torch.testing.assert_close(out_v[0, 0], torch.tensor([[13.0, 0.0], [11.0, 0.0]]))


class TestApplyAvgRope(unittest.TestCase):
    def test_single_position_quarter_turn(self):
        sketch = ExpectedAttentionSketch(n_future_positions=1)
        module = _FakeAttn(
            hidden_dim=2, num_heads=1, head_dim=2, num_kv_heads=1,
            rotary=_StubRotary(fixed_cos=torch.zeros(1, 2), fixed_sin=torch.ones(1, 2)),
        )
        mu = torch.tensor([[[1.0, 0.0]]])
        cov = torch.eye(2).view(1, 1, 2, 2)
        mu_out, cov_out = sketch.apply_avg_rope(module, mu, cov, q_len=7)
        torch.testing.assert_close(mu_out, torch.tensor([[[0.0, 1.0]]]))
        torch.testing.assert_close(cov_out, torch.eye(2).view(1, 1, 2, 2))

    def test_two_position_average(self):
        sketch = ExpectedAttentionSketch(n_future_positions=2)
        module = _FakeAttn(
            hidden_dim=2, num_heads=1, head_dim=2, num_kv_heads=1,
            rotary=_StubRotary(
                fixed_cos=torch.tensor([[1.0, 1.0], [0.0, 0.0]]),
                fixed_sin=torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
            ),
        )
        mu = torch.tensor([[[1.0, 0.0]]])
        cov = torch.eye(2).view(1, 1, 2, 2)
        mu_out, cov_out = sketch.apply_avg_rope(module, mu, cov, q_len=3)
        # R_bar = (I + P) / 2 = [[0.5, -0.5], [0.5, 0.5]]
        torch.testing.assert_close(mu_out, torch.tensor([[[0.5, 0.5]]]))
        torch.testing.assert_close(cov_out, 0.5 * torch.eye(2).view(1, 1, 2, 2))

    def test_single_position_matches_hf_rotate_half_formula(self):
        d = 8
        sketch = ExpectedAttentionSketch(n_future_positions=1)
        rotary = _StubRotary(head_dim=d)
        module = _FakeAttn(
            hidden_dim=16, num_heads=2, head_dim=d, num_kv_heads=2,
            rotary=rotary, dtype=torch.float64,
        )
        torch.manual_seed(3)
        mu = torch.randn(1, 2, d, dtype=torch.float64)
        q_len = 11
        mu_out, _ = sketch.apply_avg_rope(module, mu.clone(), None, q_len=q_len)

        half = d // 2
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, half, dtype=torch.float64) / half))
        freqs = q_len * inv_freq
        cos_p = torch.cat([freqs, freqs]).cos()
        sin_p = torch.cat([freqs, freqs]).sin()
        expected = mu * cos_p + _ref_rotate_half(mu) * sin_p
        torch.testing.assert_close(mu_out, expected, rtol=0, atol=1e-12)
        self.assertTrue(
            torch.equal(rotary.captured_position_ids, torch.tensor([[q_len]]))
        )

    def test_covariance_conjugation_with_real_rope(self):
        d = 4
        n_future = 5
        sketch = ExpectedAttentionSketch(n_future_positions=n_future)
        rotary = _StubRotary(head_dim=d)
        module = _FakeAttn(
            hidden_dim=4, num_heads=1, head_dim=d, num_kv_heads=1,
            rotary=rotary, dtype=torch.float64,
        )
        torch.manual_seed(4)
        mu = torch.randn(1, 1, d, dtype=torch.float64)
        a = torch.randn(1, 1, d, d, dtype=torch.float64)
        cov = a @ a.transpose(-1, -2)
        q_len = 9
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


class TestOracleParity(unittest.TestCase):
    def _inputs(self, num_heads=2, num_kv_heads=2, d=4, S=10, seed=11):
        rotary = _StubRotary(head_dim=d)
        module = _FakeAttn(
            hidden_dim=3 * num_heads * d, num_heads=num_heads, head_dim=d,
            num_kv_heads=num_kv_heads, rotary=rotary, seed=seed, dtype=torch.float64,
        )
        torch.manual_seed(seed + 1)
        hidden = torch.randn(1, S, 3 * num_heads * d, dtype=torch.float64)
        keys = torch.randn(1, num_kv_heads, S, d, dtype=torch.float64)
        values = torch.randn(1, num_kv_heads, S, d, dtype=torch.float64)
        return module, hidden, keys, values

    def test_all_flag_combinations_match_both_oracles(self):
        module, hidden, keys, values = self._inputs()
        for use_covariance in (False, True):
            for use_vnorm in (False, True):
                with self.subTest(use_covariance=use_covariance, use_vnorm=use_vnorm):
                    params = dict(
                        n_sink=2, n_future_positions=5,
                        use_covariance=use_covariance, use_vnorm=use_vnorm, epsilon=0.3,
                    )
                    sketch = ExpectedAttentionSketch(compression_ratio=0.5, **params)
                    got = sketch.score(module, hidden, keys, values, None, {})
                    verbatim = _kvpress_score_verbatim(module, hidden, keys, values, **params)
                    self.assertTrue(
                        torch.equal(got, verbatim),
                        "port is not a bitwise transcription of the kvpress math",
                    )
                    loop = _loop_oracle(module, hidden, keys, values, **params)
                    torch.testing.assert_close(got, loop, rtol=1e-10, atol=1e-10)


class TestGQA(unittest.TestCase):
    def test_score_shape_and_group_reduction(self):
        num_heads, num_kv_heads, d, S = 4, 2, 2, 6
        rotary = _StubRotary(head_dim=d)
        module = _FakeAttn(
            hidden_dim=num_heads * d, num_heads=num_heads, head_dim=d,
            num_kv_heads=num_kv_heads, rotary=rotary, seed=21, dtype=torch.float64,
        )
        torch.manual_seed(22)
        hidden = torch.randn(1, S, num_heads * d, dtype=torch.float64)
        keys = torch.randn(1, num_kv_heads, S, d, dtype=torch.float64)
        keys[:, 1] = keys[:, 1] * 3.0 - 5.0
        values = torch.randn(1, num_kv_heads, S, d, dtype=torch.float64)
        params = dict(n_sink=1, n_future_positions=3, use_covariance=True,
                      use_vnorm=True, epsilon=0.0)
        sketch = ExpectedAttentionSketch(compression_ratio=0.5, **params)
        got = sketch.score(module, hidden, keys, values, None, {})
        self.assertEqual(got.shape, (1, num_kv_heads, S))
        loop = _loop_oracle(module, hidden, keys, values, **params)
        torch.testing.assert_close(got, loop, rtol=1e-10, atol=1e-10)

    def test_compress_output_keeps_kv_head_count(self):
        num_heads, num_kv_heads, d, S = 4, 2, 2, 8
        module = _FakeAttn(
            hidden_dim=num_heads * d, num_heads=num_heads, head_dim=d,
            num_kv_heads=num_kv_heads, rotary=_StubRotary(head_dim=d), seed=5,
        )
        hidden = torch.randn(1, S, num_heads * d)
        keys = torch.randn(1, num_kv_heads, S, d)
        values = torch.randn(1, num_kv_heads, S, d)
        sketch = ExpectedAttentionSketch(compression_ratio=0.5, n_sink=1, n_future_positions=4)
        out_k, out_v = sketch.compress(module, hidden, keys, values, None, {})
        self.assertEqual(out_k.shape, (1, num_kv_heads, 4, d))
        self.assertEqual(out_v.shape, (1, num_kv_heads, 4, d))


class TestSinkForceKeep(unittest.TestCase):
    def _setup(self):
        module = _FakeAttn(
            hidden_dim=2, num_heads=1, head_dim=2, num_kv_heads=1,
            rotary=_identity_rotary(2, 2), identity_q=True,
        )
        hidden = torch.tensor([[1.0, 0.0]] * 8).unsqueeze(0)
        x = [-9.0, -9.0, 1.0, 2.0, 9.0, 3.0, 0.0, 2.0]
        keys = torch.tensor([[xi, float(i)] for i, xi in enumerate(x)]).view(1, 1, 8, 2)
        values = torch.randn(1, 1, 8, 2)
        return module, hidden, keys, values

    def test_sinks_padded_with_global_max(self):
        module, hidden, keys, values = self._setup()
        sketch = ExpectedAttentionSketch(
            compression_ratio=0.5, n_future_positions=2, n_sink=2,
            use_covariance=False, use_vnorm=False,
        )
        scores = sketch.score(module, hidden, keys, values, None, {})
        self.assertEqual(scores.shape, (1, 1, 8))
        global_max = scores.max()
        torch.testing.assert_close(scores[0, 0, 0], global_max)
        torch.testing.assert_close(scores[0, 0, 1], global_max)

    def test_compress_keeps_sinks_and_argmax(self):
        module, hidden, keys, values = self._setup()
        sketch = ExpectedAttentionSketch(
            compression_ratio=0.5, n_future_positions=2, n_sink=2,
            use_covariance=False, use_vnorm=False,
        )
        out_k, _ = sketch.compress(module, hidden, keys, values, None, {})
        kept_positions = set(out_k[0, 0, :, 1].long().tolist())
        self.assertEqual(out_k.shape, (1, 1, 4, 2))
        self.assertEqual(kept_positions, {0, 1, 4, 5})


class TestEdgeCases(unittest.TestCase):
    def test_seq_len_not_greater_than_n_sink_raises(self):
        module = _FakeAttn(
            hidden_dim=2, num_heads=1, head_dim=2, num_kv_heads=1,
            rotary=_identity_rotary(2, 2), identity_q=True,
        )
        sketch = ExpectedAttentionSketch(compression_ratio=0.5, n_sink=4)
        hidden = torch.randn(1, 4, 2)
        keys = torch.randn(1, 1, 4, 2)
        values = torch.randn(1, 1, 4, 2)
        with self.assertRaisesRegex(AssertionError, "n_sink"):
            sketch.score(module, hidden, keys, values, None, {})

    def test_zero_future_positions_propagates_nan(self):
        # kvpress does not guard n_future_positions=0: the empty rotation mean
        # is NaN and poisons every score. Replicated, pinned as documented.
        module = _FakeAttn(
            hidden_dim=2, num_heads=1, head_dim=2, num_kv_heads=1,
            rotary=_StubRotary(head_dim=2), identity_q=True,
        )
        sketch = ExpectedAttentionSketch(
            compression_ratio=0.5, n_future_positions=0, n_sink=1,
        )
        hidden = torch.randn(1, 6, 2)
        keys = torch.randn(1, 1, 6, 2)
        values = torch.randn(1, 1, 6, 2)
        scores = sketch.score(module, hidden, keys, values, None, {})
        self.assertTrue(torch.isnan(scores).all())

    def test_bf16_smoke(self):
        num_heads, num_kv_heads, d, S = 4, 2, 2, 12
        module = _FakeAttn(
            hidden_dim=num_heads * d, num_heads=num_heads, head_dim=d,
            num_kv_heads=num_kv_heads, rotary=_StubRotary(head_dim=d), seed=7,
            dtype=torch.bfloat16,
        )
        hidden = torch.randn(1, S, num_heads * d, dtype=torch.bfloat16)
        keys = torch.randn(1, num_kv_heads, S, d, dtype=torch.bfloat16)
        values = torch.randn(1, num_kv_heads, S, d, dtype=torch.bfloat16)
        sketch = ExpectedAttentionSketch(compression_ratio=0.25)
        scores = sketch.score(module, hidden, keys, values, None, {})
        self.assertEqual(scores.shape, (1, num_kv_heads, S))
        self.assertEqual(scores.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(scores.float()).all())
        out_k, out_v = sketch.compress(module, hidden, keys, values, None, {})
        self.assertEqual(out_k.shape[2], int(S * 0.75))
        self.assertEqual(out_v.shape[2], int(S * 0.75))
        self.assertTrue(out_k.is_contiguous())
        self.assertTrue(out_v.is_contiguous())


class TestPreRopeQueryExtraction(unittest.TestCase):
    def test_fused_qkv_slice(self):
        num_heads, num_kv_heads, d = 4, 2, 2
        module = _FakeFusedAttn(
            hidden_dim=8, num_heads=num_heads, head_dim=d, num_kv_heads=num_kv_heads, seed=9
        )
        h = torch.randn(2, 5, 8)
        got = _get_prerope_query_states(module, h)
        expected = (
            module.qkv_proj(h)[..., : num_heads * d].view(2, 5, num_heads, d).transpose(1, 2)
        )
        self.assertTrue(torch.equal(got, expected))

    def test_q_norm_applied_after_reshape(self):
        num_heads, d = 4, 2
        module = _FakeAttn(hidden_dim=8, num_heads=num_heads, head_dim=d, num_kv_heads=2, seed=10)
        module.q_norm = _DoubleQNorm()
        h = torch.randn(1, 5, 8)
        got = _get_prerope_query_states(module, h)
        expected = module.q_proj(h).view(1, 5, num_heads, d).transpose(1, 2) * 2.0
        self.assertTrue(torch.equal(got, expected))

    def test_no_projection_raises(self):
        module = SimpleNamespace(
            config=SimpleNamespace(num_attention_heads=2), head_dim=2
        )
        with self.assertRaises(NotImplementedError):
            _get_prerope_query_states(module, torch.randn(1, 3, 4))

    def test_fused_module_scores_match_q_proj_module(self):
        # Not bitwise: slicing a wider GEMM's output is not bit-identical to a
        # narrower GEMM (different blocking), hence float64 + tight tolerance.
        num_heads, num_kv_heads, d, S = 4, 2, 2, 6
        rotary = _StubRotary(head_dim=d)
        fused = _FakeFusedAttn(
            hidden_dim=8, num_heads=num_heads, head_dim=d, num_kv_heads=num_kv_heads,
            rotary=rotary, seed=12, dtype=torch.float64,
        )
        plain = _FakeAttn(
            hidden_dim=8, num_heads=num_heads, head_dim=d, num_kv_heads=num_kv_heads,
            rotary=rotary, seed=12, dtype=torch.float64,
        )
        with torch.no_grad():
            plain.q_proj.weight.copy_(fused.qkv_proj.weight[: num_heads * d])
        torch.manual_seed(13)
        hidden = torch.randn(1, S, 8, dtype=torch.float64)
        keys = torch.randn(1, num_kv_heads, S, d, dtype=torch.float64)
        values = torch.randn(1, num_kv_heads, S, d, dtype=torch.float64)
        sketch = ExpectedAttentionSketch(compression_ratio=0.5, n_sink=1, n_future_positions=3)
        s_fused = sketch.score(fused, hidden, keys, values, None, {})
        s_plain = sketch.score(plain, hidden, keys, values, None, {})
        torch.testing.assert_close(s_fused, s_plain, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
