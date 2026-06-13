"""Tests for CompactorSketch (port of kvpress CompactorPress).

Reference oracles below are independent transcriptions of the kvpress 0.5.1
math (compactor_press.py / leverage_press.py / non_causal_attention_press.py)
using different tensor mechanics (python chunk loops, ``torch.linalg.solve``
instead of Cholesky, manual 3-tap pooling) so a transcription error in the
production code cannot cancel out.
"""

import math
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from eval_harness.kv_compression.compressors.compactor_sketch import (
    CompactorSketch,
    _get_prerope_key_states,
    _get_prerope_query_states,
)


class _RMSNorm(nn.Module):
    def __init__(self, dim, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.weight = nn.Parameter(torch.rand(dim, generator=g) + 0.5)
        self.eps = 1e-6

    def forward(self, x):
        var = x.float().pow(2).mean(-1, keepdim=True)
        return ((x.float() * torch.rsqrt(var + self.eps)) * self.weight.float()).to(x.dtype)


class _FakeAttnModule(nn.Module):
    def __init__(self, hidden_dim=32, num_heads=4, num_kv_heads=2, head_dim=8,
                 seed=0, identity_k=False, qk_norm=False):
        super().__init__()
        self.config = SimpleNamespace(num_attention_heads=num_heads)
        self.num_key_value_heads = num_kv_heads
        self.head_dim = head_dim
        self.layer_idx = 0
        self.q_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, num_kv_heads * head_dim, bias=False)
        torch.manual_seed(seed)
        with torch.no_grad():
            self.q_proj.weight.normal_()
            self.k_proj.weight.normal_()
            if identity_k:
                assert hidden_dim == num_kv_heads * head_dim
                self.k_proj.weight.copy_(torch.eye(hidden_dim))
        if qk_norm:
            self.q_norm = _RMSNorm(head_dim, seed=1)
            self.k_norm = _RMSNorm(head_dim, seed=2)


class _FakeFusedAttnModule(nn.Module):
    """Phi3-style fused qkv_proj module (no q_proj/k_proj attributes)."""

    def __init__(self, hidden_dim=16, num_heads=2, num_kv_heads=1, head_dim=4, seed=4):
        super().__init__()
        self.config = SimpleNamespace(num_attention_heads=num_heads)
        self.num_key_value_heads = num_kv_heads
        self.head_dim = head_dim
        self.layer_idx = 0
        torch.manual_seed(seed)
        self.qkv_proj = nn.Linear(hidden_dim, (num_heads + 2 * num_kv_heads) * head_dim, bias=False)


def _identity_pos_emb(B, S, D):
    return torch.ones(B, S, D), torch.zeros(B, S, D)


def _rope_pos_emb(positions, D, base=10000.0):
    from eval_harness.attention_methods._method_base import build_cos_sin

    half = D // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))
    cos, sin = build_cos_sin(positions, inv_freq, torch.device("cpu"), torch.float32)
    return cos.squeeze(1), sin.squeeze(1)  # [B, S, D]


def _ref_rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def _ref_chunked_attn(q, k, chunk_size):
    """Loop-based transcription of kvpress non_causal_chunked_attn (incl. padding quirks):
    zero-pad to a multiple of chunk_size, per-tile q@k^T with NO 1/sqrt(d) scaling, in the
    last tile padded-query rows := 0 then padded-key cols := -1e-9 (literally), fp32 softmax
    over keys, sum over ALL query rows, trim."""
    B, H, S, d = k.shape
    S_pad = math.ceil(S / chunk_size) * chunk_size
    pad_len = S_pad - S
    if pad_len > 0:
        q = torch.cat([q, torch.zeros(B, H, pad_len, d)], dim=2)
        k = torch.cat([k, torch.zeros(B, H, pad_len, d)], dim=2)
    out = torch.zeros(B, H, S_pad)
    num_chunks = S_pad // chunk_size
    for c in range(num_chunks):
        sl = slice(c * chunk_size, (c + 1) * chunk_size)
        dots = torch.einsum("bhqd,bhkd->bhqk", q[:, :, sl], k[:, :, sl])
        if c == num_chunks - 1:
            padded = torch.arange(c * chunk_size, (c + 1) * chunk_size) >= S
            dots[:, :, padded, :] = 0.0
            dots[:, :, :, padded] = -1e-9
        attn = torch.softmax(dots.to(torch.float32), dim=-1)
        out[:, :, sl] = attn.sum(dim=-2)
    return out[..., :S]


def _ref_leverage_raw(key_states, phi):
    X = key_states - key_states.mean(dim=-2, keepdim=True)
    X = torch.matmul(X, phi).to(torch.float32)
    G = X.transpose(-2, -1) @ X
    G = 0.5 * (G + G.transpose(-2, -1)) + 1e-2 * torch.eye(G.shape[-1])
    sol = torch.linalg.solve(G, X.transpose(-2, -1))
    return (X * sol.transpose(-2, -1)).sum(dim=-1).clamp_min(0)


def _ref_zn(s):
    return (s - s.mean()) / s.std().clamp_min(1e-6)


def _ref_pool3(s):
    out = torch.zeros_like(s)
    S = s.shape[-1]
    for i in range(S):
        left = s[..., i - 1] if i >= 1 else torch.zeros_like(s[..., 0])
        right = s[..., i + 1] if i + 1 < S else torch.zeros_like(s[..., 0])
        out[..., i] = (left + s[..., i] + right) / 3.0
    return out


def _ref_compactor_score(sketch, module, hidden_states, keys, values, cos, sin, phi):
    B, H_kv, S, _ = keys.shape
    left = min(sketch.sink_size_start, S)
    right = min(sketch.sink_size_end, max(0, S - left))
    end = None if right == 0 else -right
    hs = hidden_states[:, left:end]
    k_i = keys[..., left:end, :]
    v_i = values[..., left:end, :]
    cos_i = cos[..., left:end, :]
    sin_i = sin[..., left:end, :]
    Sp = hs.shape[1]
    num_heads = module.config.num_attention_heads
    hd = module.head_dim

    pre_k = module.k_proj(hs).view(B, Sp, -1, hd).transpose(1, 2)
    if getattr(module, "k_norm", None) is not None:
        pre_k = module.k_norm(pre_k)
    l_z = _ref_zn(_ref_leverage_raw(pre_k, phi))

    q = module.q_proj(hs).view(B, Sp, num_heads, hd).transpose(1, 2)
    if getattr(module, "q_norm", None) is not None:
        q = module.q_norm(q)
    q = q * cos_i.unsqueeze(1) + _ref_rotate_half(q) * sin_i.unsqueeze(1)
    groups = num_heads // H_kv
    A = _ref_chunked_attn(q, k_i.repeat_interleave(groups, dim=1), sketch.chunk_size)
    A = A.view(B, H_kv, groups, Sp).mean(dim=2)
    a_z = _ref_zn(_ref_pool3(A * v_i.norm(dim=-1)))

    blend = sketch.blending if sketch.blending is not None else sketch.compression_ratio
    out = blend * l_z + a_z
    full = torch.full((B, H_kv, S), out.max().item())
    full[..., left:S - right] = out
    return full


class TestCompactorRegistry(unittest.TestCase):
    def test_registered_and_resolves(self):
        from eval_harness.kv_compression.registry import (
            available_kv_compressors,
            get_kv_compressor,
            get_kv_compressor_class,
        )

        self.assertIn("compactor", available_kv_compressors())
        self.assertIs(get_kv_compressor_class("compactor"), CompactorSketch)
        sketch = get_kv_compressor("compactor", compression_ratio=0.3, sketch_dimension=16, chunk_size=64)
        self.assertIsInstance(sketch, CompactorSketch)
        self.assertAlmostEqual(sketch.compression_ratio, 0.3)
        self.assertEqual(sketch.sketch_dimension, 16)
        self.assertEqual(sketch.chunk_size, 64)


class TestCompactorFrameworkBehavior(unittest.TestCase):
    def test_zero_ratio_noop_score_never_called(self):
        sketch = CompactorSketch(compression_ratio=0.0)
        module = _FakeAttnModule()
        keys = torch.randn(1, 2, 16, 8)
        values = torch.randn(1, 2, 16, 8)
        with mock.patch.object(CompactorSketch, "score") as spy:
            out_k, out_v = sketch.compress(module, torch.randn(1, 16, 32), keys, values, None, {})
        self.assertIs(out_k, keys)
        self.assertIs(out_v, values)
        spy.assert_not_called()

    def test_decode_step_is_noop(self):
        sketch = CompactorSketch(compression_ratio=0.5)
        module = _FakeAttnModule()
        k0 = torch.randn(1, 2, 32, 8)
        v0 = torch.randn(1, 2, 32, 8)
        cache = SimpleNamespace(layers=[SimpleNamespace(keys=k0, values=v0)])
        output = (torch.randn(1, 1, 32), None)
        kwargs = {
            "hidden_states": torch.randn(1, 1, 32),
            "past_key_values": cache,
            "cache_position": torch.tensor([31]),
        }
        with mock.patch.object(CompactorSketch, "score") as spy:
            result = sketch.forward_hook(module, [], kwargs, output)
        self.assertIs(result, output)
        self.assertIs(cache.layers[0].keys, k0)
        self.assertIs(cache.layers[0].values, v0)
        spy.assert_not_called()

    def test_forward_hook_prefill_compresses_cache(self):
        D = 8
        module = _FakeAttnModule(seed=15)
        S = 32
        torch.manual_seed(22)
        keys = torch.randn(1, 2, S, D)
        values = torch.randn(1, 2, S, D)
        cache = SimpleNamespace(layers=[SimpleNamespace(keys=keys, values=values)])
        cos, sin = _rope_pos_emb(torch.arange(S), D)
        kwargs = {
            "hidden_states": torch.randn(1, S, 32),
            "past_key_values": cache,
            "cache_position": torch.arange(S),
            "position_embeddings": (cos, sin),
        }
        sketch = CompactorSketch(compression_ratio=0.5, phi=torch.randn(D, 4) * 0.5)
        output = (torch.randn(1, S, 32), None)
        result = sketch.forward_hook(module, [], kwargs, output)
        self.assertIs(result, output)
        self.assertEqual(tuple(cache.layers[0].keys.shape), (1, 2, 16, D))
        self.assertEqual(tuple(cache.layers[0].values.shape), (1, 2, 16, D))

    def test_non_prefill_shapes_assert(self):
        module = _FakeAttnModule()
        with self.assertRaises(AssertionError):
            CompactorSketch(compression_ratio=0.5).score(
                module, torch.randn(1, 4, 32), torch.randn(1, 2, 10, 8), torch.randn(1, 2, 10, 8), None, {}
            )

    def test_sink_protection_and_selection(self):
        sketch = CompactorSketch(compression_ratio=0.5, sink_size_start=2, sink_size_end=2)
        module = _FakeAttnModule(hidden_dim=8, num_heads=2, num_kv_heads=1, head_dim=4)
        B, H, S, D = 1, 1, 16, 4
        hidden = torch.randn(B, S, 8)
        keys = torch.arange(S, dtype=torch.float32).view(1, 1, S, 1).expand(B, H, S, D).clone()
        values = keys.clone()
        cos, sin = _identity_pos_emb(B, S, D)
        kwargs = {"position_embeddings": (cos, sin)}
        interior = torch.arange(12, dtype=torch.float32).view(1, 1, 12)
        with mock.patch.object(
            CompactorSketch, "_leverage_scores", new=lambda self, m, hs: torch.zeros(1, 1, hs.shape[1])
        ), mock.patch.object(
            CompactorSketch, "_non_causal_scores", new=lambda self, m, hs, k, v, c, s: interior.clone()
        ):
            scores = sketch.score(module, hidden, keys, values, None, kwargs)
            out_k, out_v = sketch.compress(module, hidden, keys, values, None, kwargs)
        # blending=None -> ratio=0.5; blended interior = 0.5*0 + arange(12); pad value = max = 11
        torch.testing.assert_close(scores[0, 0, [0, 1, 14, 15]], torch.full((4,), 11.0))
        self.assertEqual(scores[0, 0].max().item(), 11.0)
        kept = sorted(out_k[0, 0, :, 0].tolist())
        self.assertEqual(kept, [0.0, 1.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
        torch.testing.assert_close(out_k, out_v)

    def test_blending_semantics(self):
        module = _FakeAttnModule(hidden_dim=8, num_heads=2, num_kv_heads=1, head_dim=4)
        B, S, D = 1, 16, 4
        hidden = torch.randn(B, S, 8)
        keys = torch.randn(B, 1, S, D)
        values = torch.randn(B, 1, S, D)
        cos, sin = _identity_pos_emb(B, S, D)
        kwargs = {"position_embeddings": (cos, sin)}
        with mock.patch.object(
            CompactorSketch, "_leverage_scores", new=lambda self, m, hs: torch.full((1, 1, hs.shape[1]), 2.0)
        ), mock.patch.object(
            CompactorSketch, "_non_causal_scores", new=lambda self, m, hs, k, v, c, s: torch.full((1, 1, hs.shape[1]), 3.0)
        ):
            s_explicit = CompactorSketch(
                compression_ratio=0.5, sink_size_start=2, sink_size_end=2, blending=0.25
            ).score(module, hidden, keys, values, None, kwargs)
            s_fallback = CompactorSketch(
                compression_ratio=0.4, sink_size_start=2, sink_size_end=2
            ).score(module, hidden, keys, values, None, kwargs)
        torch.testing.assert_close(s_explicit, torch.full((1, 1, 16), 0.25 * 2.0 + 3.0))
        torch.testing.assert_close(s_fallback, torch.full((1, 1, 16), 0.4 * 2.0 + 3.0))

    def test_empty_interior_guard(self):
        # Documented deviation: upstream crashes when sink protection covers the
        # whole sequence; the port returns uniform zero scores instead.
        sketch = CompactorSketch(compression_ratio=0.5)  # sink_size_start=8, sink_size_end=4
        module = _FakeAttnModule()
        for S in (6, 12):  # S <= sink_start, and S == sink_start + sink_end
            hidden = torch.randn(1, S, 32)
            keys = torch.randn(1, 2, S, 8)
            values = torch.randn(1, 2, S, 8)
            scores = sketch.score(module, hidden, keys, values, None, {})
            self.assertEqual(tuple(scores.shape), (1, 2, S))
            self.assertTrue(torch.isfinite(scores).all())
            torch.testing.assert_close(scores, torch.zeros(1, 2, S))

    def test_determinism_with_injected_phi(self):
        module = _FakeAttnModule()
        D, S = 8, 32
        torch.manual_seed(17)
        hidden = torch.randn(1, S, 32)
        keys = torch.randn(1, 2, S, D)
        values = torch.randn(1, 2, S, D)
        cos, sin = _rope_pos_emb(torch.arange(S), D)
        kwargs = {"position_embeddings": (cos, sin)}
        sketch = CompactorSketch(compression_ratio=0.5, phi=torch.randn(D, 4) * 0.5)
        s1 = sketch.score(module, hidden, keys, values, None, kwargs)
        s2 = sketch.score(module, hidden, keys, values, None, kwargs)
        self.assertTrue(torch.equal(s1, s2))
        n_kept = int(S * 0.5)
        self.assertTrue(torch.equal(s1.topk(n_kept, dim=-1).indices, s2.topk(n_kept, dim=-1).indices))
        # Upstream parity: without injection a fresh torch.randn sketch is drawn
        # on EVERY call, so scores are nondeterministic run-to-run.
        free = CompactorSketch(compression_ratio=0.5)
        f1 = free.score(module, hidden, keys, values, None, kwargs)
        f2 = free.score(module, hidden, keys, values, None, kwargs)
        self.assertFalse(torch.equal(f1, f2))


class TestLeverageComponent(unittest.TestCase):
    def test_hand_computed_pin_identity_phi(self):
        # rows (2,0),(-1,1),(-1,-1) are zero-mean; with Phi=I, G = diag(6,2) and the
        # first-try jitter adds 1e-2*I, so lev = [4/6.01, 1/6.01+1/2.01, 1/6.01+1/2.01].
        sketch = CompactorSketch(compression_ratio=0.5, sketch_dimension=2, phi=torch.eye(2))
        keys = torch.tensor([[[[2.0, 0.0], [-1.0, 1.0], [-1.0, -1.0]]]])
        raw = sketch._compute_leverage_scores(keys)
        expected = torch.tensor([[[4 / 6.01, 1 / 6.01 + 1 / 2.01, 1 / 6.01 + 1 / 2.01]]])
        torch.testing.assert_close(raw, expected, atol=1e-5, rtol=0)

    def test_z_norm_through_module_projection(self):
        module = _FakeAttnModule(hidden_dim=2, num_heads=1, num_kv_heads=1, head_dim=2, identity_k=True)
        sketch = CompactorSketch(compression_ratio=0.5, sketch_dimension=2, phi=torch.eye(2))
        hidden = torch.tensor([[[2.0, 0.0], [-1.0, 1.0], [-1.0, -1.0]]])
        z = sketch._leverage_scores(module, hidden)
        raw = torch.tensor([4 / 6.01, 1 / 6.01 + 1 / 2.01, 1 / 6.01 + 1 / 2.01])
        expected = ((raw - raw.mean()) / raw.std().clamp_min(1e-6)).view(1, 1, 3)
        torch.testing.assert_close(z, expected, atol=1e-5, rtol=0)

    def test_reference_transcription_oracle(self):
        torch.manual_seed(3)
        B, H, S, d, k = 2, 3, 64, 16, 8
        keys = torch.randn(B, H, S, d)
        phi = torch.randn(B, H, d, k) / math.sqrt(k)
        sketch = CompactorSketch(compression_ratio=0.5, sketch_dimension=k, phi=phi)
        raw = sketch._compute_leverage_scores(keys)
        ref = _ref_leverage_raw(keys, phi)
        torch.testing.assert_close(raw, ref, atol=1e-4, rtol=1e-4)

    def test_constant_input_no_nan(self):
        keys = torch.full((1, 2, 10, 4), 0.7)
        sketch = CompactorSketch(compression_ratio=0.5, sketch_dimension=4)
        raw = sketch._compute_leverage_scores(keys)
        torch.testing.assert_close(raw, torch.zeros(1, 2, 10))

        module = _FakeAttnModule()
        z = CompactorSketch(compression_ratio=0.5)._leverage_scores(module, torch.full((1, 10, 32), 0.3))
        torch.testing.assert_close(z, torch.zeros(1, 2, 10))  # std 0 -> clamp_min(1e-6), z exactly 0

        S = 20
        cos, sin = _identity_pos_emb(1, S, 8)
        scores = CompactorSketch(compression_ratio=0.5).score(
            module,
            torch.full((1, S, 32), 0.3),
            torch.full((1, 2, S, 8), 0.7),
            torch.full((1, 2, S, 8), 0.5),
            None,
            {"position_embeddings": (cos, sin)},
        )
        self.assertTrue(torch.isfinite(scores).all())


class TestNonCausalComponent(unittest.TestCase):
    def test_tiny_exact_pin(self):
        # S=2, chunk=4, q=k=I rows: dots=I padded to 4x4; padded-query rows zeroed,
        # padded-key cols -1e-9 (NOT excluded). Real rows: softmax([1,0,-1e-9,-1e-9]),
        # padded rows uniform ~1/4 -> each column sums to (e+1)/denom + 2/denom2.
        q = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
        out = CompactorSketch._non_causal_chunked_attn(q, q.clone(), 4)
        e = math.e
        denom = e + 1 + 2 * math.exp(-1e-9)
        denom2 = 2 + 2 * math.exp(-1e-9)
        expected_col = (e + 1) / denom + 2 / denom2
        torch.testing.assert_close(out, torch.full((1, 1, 2), expected_col), atol=1e-5, rtol=0)

    def test_padding_quirks_oracle(self):
        torch.manual_seed(7)
        q = torch.randn(2, 3, 5, 4)
        k = torch.randn(2, 3, 5, 4)
        out = CompactorSketch._non_causal_chunked_attn(q, k, 4)
        torch.testing.assert_close(out, _ref_chunked_attn(q, k, 4), atol=1e-6, rtol=1e-6)

    def test_no_padding_multichunk_oracle(self):
        torch.manual_seed(8)
        q = torch.randn(1, 2, 8, 4)
        k = torch.randn(1, 2, 8, 4)
        out = CompactorSketch._non_causal_chunked_attn(q, k, 4)
        torch.testing.assert_close(out, _ref_chunked_attn(q, k, 4), atol=1e-6, rtol=1e-6)

    def test_avg_pool_boundary_and_global_z_norm(self):
        # pre-pool scores [3,6,9,12] -> count_include_pad pooling gives [3,6,9,7]
        module = _FakeAttnModule(hidden_dim=2, num_heads=1, num_kv_heads=1, head_dim=2)
        sketch = CompactorSketch(compression_ratio=0.5, sink_size_start=0, sink_size_end=0, blending=0.0)
        B, S, D = 1, 4, 2
        torch.manual_seed(9)
        hidden = torch.randn(B, S, 2)
        keys = torch.randn(B, 1, S, D)
        values = torch.zeros(B, 1, S, D)
        values[..., 0] = 1.0  # unit value norms
        cos, sin = _identity_pos_emb(B, S, D)
        fixed_A = torch.tensor([[[3.0, 6.0, 9.0, 12.0]]])
        with mock.patch.object(
            CompactorSketch, "_non_causal_chunked_attn", new=staticmethod(lambda q, k, c: fixed_A.clone())
        ):
            scores = sketch.score(module, hidden, keys, values, None, {"position_embeddings": (cos, sin)})
        pooled = torch.tensor([3.0, 6.0, 9.0, 7.0])
        expected = ((pooled - pooled.mean()) / pooled.std()).view(1, 1, 4)
        torch.testing.assert_close(scores, expected, atol=1e-6, rtol=1e-6)


class TestPreRopeProjections(unittest.TestCase):
    def test_qk_norm_applied(self):
        module = _FakeAttnModule(seed=7, qk_norm=True)
        hidden = torch.randn(2, 10, 32)
        q = _get_prerope_query_states(module, hidden)
        k = _get_prerope_key_states(module, hidden)
        plain_q = module.q_proj(hidden).view(2, 10, 4, 8).transpose(1, 2)
        plain_k = module.k_proj(hidden).view(2, 10, 2, 8).transpose(1, 2)
        torch.testing.assert_close(q, module.q_norm(plain_q))
        torch.testing.assert_close(k, module.k_norm(plain_k))
        self.assertFalse(torch.allclose(q, plain_q))
        self.assertFalse(torch.allclose(k, plain_k))

    def test_fused_qkv_path(self):
        module = _FakeFusedAttnModule()
        hidden = torch.randn(1, 6, 16)
        q = _get_prerope_query_states(module, hidden)
        k = _get_prerope_key_states(module, hidden)
        qkv = module.qkv_proj(hidden)
        torch.testing.assert_close(q, qkv[..., :8].view(1, 6, 2, 4).transpose(1, 2))
        torch.testing.assert_close(k, qkv[..., 8:12].view(1, 6, 1, 4).transpose(1, 2))


class TestFullScoreOracle(unittest.TestCase):
    def _run_case(self, sketch, module, B, H_kv, S, D, seed):
        torch.manual_seed(seed)
        hidden = torch.randn(B, S, module.q_proj.in_features)
        keys = torch.randn(B, H_kv, S, D)
        values = torch.randn(B, H_kv, S, D)
        cos, sin = _rope_pos_emb(torch.arange(S), D)
        scores = sketch.score(module, hidden, keys, values, None, {"position_embeddings": (cos, sin)})
        self.assertEqual(tuple(scores.shape), (B, H_kv, S))
        ref = _ref_compactor_score(sketch, module, hidden, keys, values, cos, sin, sketch.phi)
        torch.testing.assert_close(scores, ref, atol=1e-4, rtol=1e-4)
        return scores

    def test_gqa_full_chain(self):
        D = 8
        module = _FakeAttnModule(hidden_dim=32, num_heads=4, num_kv_heads=2, head_dim=D, seed=5)
        torch.manual_seed(50)
        sketch = CompactorSketch(
            compression_ratio=0.5, chunk_size=6, blending=0.7, phi=torch.randn(D, 4) * 0.5
        )
        self._run_case(sketch, module, B=1, H_kv=2, S=32, D=D, seed=11)

    def test_chunk_larger_than_sequence(self):
        D = 8
        module = _FakeAttnModule(hidden_dim=32, num_heads=4, num_kv_heads=2, head_dim=D, seed=6)
        torch.manual_seed(51)
        # blending=None -> compression_ratio in the full chain; interior (8) < chunk (256)
        sketch = CompactorSketch(compression_ratio=0.5, chunk_size=256, phi=torch.randn(D, 4) * 0.5)
        self._run_case(sketch, module, B=1, H_kv=2, S=20, D=D, seed=12)

    def test_qk_norm_full_chain(self):
        D = 8
        module = _FakeAttnModule(hidden_dim=32, num_heads=4, num_kv_heads=2, head_dim=D, seed=9, qk_norm=True)
        torch.manual_seed(52)
        sketch = CompactorSketch(
            compression_ratio=0.25, chunk_size=8, blending=0.5, phi=torch.randn(D, 4) * 0.5
        )
        self._run_case(sketch, module, B=1, H_kv=2, S=24, D=D, seed=13)

    def test_position_embeddings_fallback_rotary(self):
        D, S = 8, 20
        module = _FakeAttnModule(seed=14)
        cos, sin = _rope_pos_emb(torch.arange(S), D)
        module.rotary_emb = lambda hs, position_ids: (cos, sin)
        torch.manual_seed(53)
        sketch = CompactorSketch(compression_ratio=0.5, chunk_size=16, phi=torch.randn(D, 4) * 0.5)
        hidden = torch.randn(1, S, 32)
        keys = torch.randn(1, 2, S, D)
        values = torch.randn(1, 2, S, D)
        explicit = sketch.score(module, hidden, keys, values, None, {"position_embeddings": (cos, sin)})
        fallback = sketch.score(module, hidden, keys, values, None, {"cache_position": torch.arange(S)})
        torch.testing.assert_close(explicit, fallback, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
