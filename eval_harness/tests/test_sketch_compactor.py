"""Tests for CompactorSketch (faithful port of the authors' reference engine).

Reference oracles below are independent in-test transcriptions of the
compactor-vllm reference (arXiv 2507.08143):

* ``compression/compactor.py`` — ``approximate_leverage_scores`` (incl.
  ``split_into_chunks``, per-chunk centering, fp32 Gram + 5e-3 diagonal,
  ``SV = V * S^-1/2``, QR fallback) and ``non_causal_attn_scores`` /
  ``_non_causal_attn_kernel`` (block-diagonal softmax with ``sm_scale=1.0``,
  column sums over query rows, GQA SUM into the KV head, padded query rows at
  ``BLOCK_M=64`` tile granularity each adding exactly ``1/chunk_size``);
* ``_zscore_per_batch_epilogue_no_window`` — fp32 z-score with biased
  variance clamped at 0 and NO epsilon (``invstd = 1/sqrt(var)``);
* ``utils/arguments.py`` — the ONE shared PHI: fp32 ``randn`` from a
  dedicated generator seeded 42, cast to model dtype THEN scaled by
  ``1/sqrt(k)``.

The transcriptions use different tensor mechanics (per-batch/head/chunk
python loops, ``torch.linalg.solve`` identities) so a transcription error in
the production code cannot cancel out.

Framework note (``ScorerKVCompressor.compress``): the kept tokens are gathered
in SCORE order, not position order — selection is therefore asserted through
membership/sets and gather-at-topk equality, never through positions.
"""

import math
import unittest
from dataclasses import fields as dataclass_fields
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from eval_harness.kv_compression.registry import (
    available_kv_compressors,
    get_kv_compressor,
    get_kv_compressor_class,
)
from eval_harness.kv_compression.compressors.compactor_sketch import (
    CompactorSketch,
    _get_prerope_key_states,
    _get_prerope_query_states,
)


# ----------------------------------------------------------------------
# Fake modules (no real weights, per repo test conventions)
# ----------------------------------------------------------------------


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
    """Llama-like fake attention module with q_proj/k_proj and required attrs.

    Weights use std 1/sqrt(hidden_dim) so raw q.k logits stay in a range where
    the (un-scaled, ``sm_scale=1.0``) softmax is neither uniform nor one-hot.
    ``identity_q``/``identity_k`` make the projection the identity so tests
    can control queries/pre-RoPE keys exactly.
    """

    def __init__(self, hidden_dim=32, num_heads=4, num_kv_heads=2, head_dim=8,
                 seed=0, identity_q=False, identity_k=False, qk_norm=False):
        super().__init__()
        self.config = SimpleNamespace(num_attention_heads=num_heads)
        self.num_key_value_heads = num_kv_heads
        self.head_dim = head_dim
        self.layer_idx = 0
        torch.manual_seed(seed)
        self.q_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, num_kv_heads * head_dim, bias=False)
        with torch.no_grad():
            self.q_proj.weight.normal_(std=hidden_dim ** -0.5)
            self.k_proj.weight.normal_(std=hidden_dim ** -0.5)
            if identity_q:
                assert hidden_dim == num_heads * head_dim
                self.q_proj.weight.copy_(torch.eye(hidden_dim))
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
        self.qkv_proj = nn.Linear(
            hidden_dim, (num_heads + 2 * num_kv_heads) * head_dim, bias=False,
        )


class _FakeGatedQProjModule(nn.Module):
    """Qwen3.5-style gated attention: q_proj emits [query | gate] per head
    (output dim = num_heads * head_dim * 2)."""

    def __init__(self, hidden_dim=16, num_heads=2, num_kv_heads=1, head_dim=4, seed=5):
        super().__init__()
        self.config = SimpleNamespace(num_attention_heads=num_heads)
        self.num_key_value_heads = num_kv_heads
        self.head_dim = head_dim
        self.layer_idx = 0
        torch.manual_seed(seed)
        self.q_proj = nn.Linear(hidden_dim, num_heads * head_dim * 2, bias=False)
        self.k_proj = nn.Linear(hidden_dim, num_kv_heads * head_dim, bias=False)


class _RecordingRotary:
    """rotary_emb stand-in that records the position_ids it is called with."""

    def __init__(self, cos, sin):
        self.cos, self.sin = cos, sin
        self.calls = []

    def __call__(self, hidden_states, position_ids):
        self.calls.append(position_ids)
        return self.cos, self.sin


# ----------------------------------------------------------------------
# RoPE / projection helpers (test-side, independent of production helpers)
# ----------------------------------------------------------------------


def _rope_cos_sin(S, dim, base=10000.0):
    """HF-style (cos, sin) of shape [1, S, dim] for positions 0..S-1."""
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    freqs = torch.outer(torch.arange(S, dtype=torch.float32), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().unsqueeze(0), emb.sin().unsqueeze(0)


def _identity_pos_emb(B, S, D):
    return torch.ones(B, S, D), torch.zeros(B, S, D)


def _ref_rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def _ref_apply_rope(x, cos, sin):
    """Rotate the first cos.shape[-1] channels of x [B, H, S, D] (partial
    rotary aware; reduces to full RoPE when cos covers head_dim)."""
    rd = cos.shape[-1]
    x_rot, x_pass = x[..., :rd], x[..., rd:]
    x_rot = x_rot * cos.unsqueeze(1) + _ref_rotate_half(x_rot) * sin.unsqueeze(1)
    return torch.cat([x_rot, x_pass], dim=-1)


def _manual_pre_rope_q(module, hidden):
    """Manual pre-RoPE query re-projection (incl. gated q_proj slice, q_norm)."""
    B, S, _ = hidden.shape
    nh, hd = module.config.num_attention_heads, module.head_dim
    q = module.q_proj(hidden)
    if q.shape[-1] == nh * hd * 2:
        q = q.view(B, S, nh, 2 * hd)[..., :hd]
    q = q.reshape(B, S, nh, hd).transpose(1, 2)
    if getattr(module, "q_norm", None) is not None:
        q = module.q_norm(q)
    return q


def _manual_pre_rope_k(module, hidden):
    B, S, _ = hidden.shape
    k = module.k_proj(hidden).view(B, S, -1, module.head_dim).transpose(1, 2)
    if getattr(module, "k_norm", None) is not None:
        k = module.k_norm(k)
    return k


# ----------------------------------------------------------------------
# Reference transcriptions (compactor-vllm math, different mechanics)
# ----------------------------------------------------------------------


def _ref_zscore_flat(vals):
    """``_zscore_per_batch_epilogue_no_window`` over ALL entries of ``vals``:
    fp32, biased variance clamped at 0, ``invstd = 1/sqrt(var)`` — NO eps."""
    v = vals.float()
    mean = v.mean()
    var = torch.clamp((v * v).mean() - mean * mean, min=0.0)
    return (v - mean) / var.sqrt()


def _ref_chunk_lens(n, chunk_size):
    """Reference ``split_into_chunks`` for one sequence: ``n // cs`` full
    chunks plus one shorter epilogue; ``cs <= 0`` -> single chunk."""
    if chunk_size <= 0:
        return [n]
    lens = [chunk_size] * (n // chunk_size)
    epilogue = n - (n // chunk_size) * chunk_size
    if epilogue > 0:
        lens.append(epilogue)
    return lens


def _ref_chunk_bounds(n, chunk_size):
    bounds, start = [], 0
    for L in _ref_chunk_lens(n, chunk_size):
        bounds.append((start, start + L))
        start += L
    return bounds


def _ref_leverage_component(pre_rope_k, phi, leverage_chunk_size=512,
                            regularizer=5e-3):
    """``approximate_leverage_scores(normalize=True)`` transcription.

    X = K @ PHI; per-chunk mean-centering; fp32 Gram + reg*I; SVD ->
    SV = V * S^-1/2 cast back to input dtype; lev = ||X_c @ SV||^2 clamped at
    0; z-scored PER CHUNK over (heads x chunk tokens). Per-batch/head/chunk
    python loops. Returns fp32 [B, H, S].
    """
    B, H, S, _ = pre_rope_k.shape
    k = phi.shape[-1]
    out = torch.empty(B, H, S, dtype=torch.float32)
    eye = torch.eye(k, dtype=torch.float32)
    for b in range(B):
        for (s0, s1) in _ref_chunk_bounds(S, leverage_chunk_size):
            lev = torch.empty(H, s1 - s0, dtype=pre_rope_k.dtype)
            for h in range(H):
                Xc = pre_rope_k[b, h, s0:s1] @ phi
                Xc = Xc - Xc.mean(dim=0, keepdim=True)
                G = (Xc.transpose(0, 1) @ Xc).to(torch.float32) + regularizer * eye
                V, Sv, _ = torch.linalg.svd(G, full_matrices=False)
                SV = (V * Sv.rsqrt().unsqueeze(0)).to(Xc.dtype)
                U = Xc @ SV
                lev[h] = (U * U).sum(dim=-1).clamp_min(0.0)
            out[b, :, s0:s1] = _ref_zscore_flat(lev)
    return out


def _ref_non_causal_component(post_rope_q, post_rope_k, chunk_size=128,
                              normalize=True, pad_block_m=64):
    """``_non_causal_attn_kernel`` transcription.

    Per chunk: fp32 softmax(q @ k^T) with ``sm_scale = 1.0`` over the chunk's
    REAL keys only; column sums over query rows; GQA SUM over group members
    into the KV head; padded query rows of the short chunk (tile granularity
    ``BLOCK_M = 64``) each add exactly ``1/chunk_size`` to every column of
    that chunk. Optionally z-scored per sequence over (H_kv x S).
    Returns fp32 [B, H_kv, S].
    """
    B, H_q, S, _ = post_rope_q.shape
    H_kv = post_rope_k.shape[1]
    n_groups = H_q // H_kv
    raw = torch.zeros(B, H_kv, S, dtype=torch.float32)
    bounds = _ref_chunk_bounds(S, chunk_size)
    for b in range(B):
        for kv in range(H_kv):
            for g in range(n_groups):
                qh = kv * n_groups + g
                for (s0, s1) in bounds:
                    logits = (
                        post_rope_q[b, qh, s0:s1]
                        @ post_rope_k[b, kv, s0:s1].transpose(0, 1)
                    ).float()
                    p = torch.softmax(logits, dim=-1)
                    cols = p.sum(dim=0)
                    m = s1 - s0
                    pad_rows = math.ceil(m / pad_block_m) * pad_block_m - m
                    raw[b, kv, s0:s1] += cols + pad_rows * (1.0 / chunk_size)
    if not normalize:
        return raw
    out = torch.empty_like(raw)
    for b in range(B):
        out[b] = _ref_zscore_flat(raw[b])
    return out


def _reference_compactor_scores(pre_rope_k, post_rope_q, post_rope_k, *, phi,
                                chunk_size=128, leverage_chunk_size=512,
                                regularizer=5e-3, blending=0.5,
                                first=16, last=64):
    """Full reference scores: attn_z + blending * lev_z, then +inf on the
    protected spans AFTER scoring (protected tokens stay inside the z stats)."""
    attn_z = _ref_non_causal_component(post_rope_q, post_rope_k, chunk_size)
    lev_z = _ref_leverage_component(
        pre_rope_k, phi, leverage_chunk_size, regularizer,
    )
    scores = attn_z + blending * lev_z
    T = scores.shape[-1]
    f = min(first, T)
    l = min(last, max(0, T - f))
    if f > 0:
        scores[:, :, :f] = torch.inf
    if l > 0:
        scores[:, :, T - l:] = torch.inf
    return scores


def _build_case(module, B, S, seed, rotary_dim=None, base=10000.0):
    """Consistent inputs: hidden, pre-RoPE k, post-RoPE q, post-RoPE cached
    keys (what the framework would cache), values, and (cos, sin)."""
    torch.manual_seed(seed)
    hidden = torch.randn(B, S, module.q_proj.in_features)
    D = module.head_dim
    cos, sin = _rope_cos_sin(S, rotary_dim or D, base)
    pre_q = _manual_pre_rope_q(module, hidden)
    pre_k = _manual_pre_rope_k(module, hidden)
    post_q = _ref_apply_rope(pre_q, cos, sin)
    post_k = _ref_apply_rope(pre_k, cos, sin)
    values = torch.randn(B, pre_k.shape[1], S, D)
    return hidden, pre_k, post_q, post_k, values, cos, sin


def _gen_phi(head_dim, k, seed=100):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(head_dim, k, generator=g) / math.sqrt(k)


def _standardize(x):
    """Biased z-score over ALL entries (for affine-invariance pins)."""
    xf = x.float()
    return (xf - xf.mean()) / xf.var(unbiased=False).sqrt()


# ----------------------------------------------------------------------
# 1. Registry
# ----------------------------------------------------------------------


class TestCompactorRegistry(unittest.TestCase):
    def test_registered_and_class(self):
        self.assertIn("compactor", available_kv_compressors())
        self.assertIs(get_kv_compressor_class("compactor"), CompactorSketch)

    def test_compression_ratio_kwarg_injection(self):
        sketch = get_kv_compressor(
            "compactor", compression_ratio=0.3, sketch_dimension=16,
            chunk_size=64, seed=7, blending=0.25,
        )
        self.assertIsInstance(sketch, CompactorSketch)
        self.assertAlmostEqual(sketch.compression_ratio, 0.3)
        self.assertEqual(sketch.sketch_dimension, 16)
        self.assertEqual(sketch.chunk_size, 64)
        self.assertEqual(sketch.seed, 7)
        self.assertAlmostEqual(sketch.blending, 0.25)
        # the runner injects the adapter-level ratio only when the class
        # declares a `compression_ratio` dataclass field — it must
        self.assertIn(
            "compression_ratio",
            [f.name for f in dataclass_fields(CompactorSketch)],
        )

    def test_reference_engine_defaults(self):
        sketch = CompactorSketch()
        self.assertAlmostEqual(sketch.compression_ratio, 0.0)
        self.assertEqual(sketch.sink_size_start, 16)   # protected_first_tokens
        self.assertEqual(sketch.sink_size_end, 64)     # protected_last_tokens
        self.assertEqual(sketch.chunk_size, 128)       # CompactorCompression.chunk_size
        self.assertEqual(sketch.leverage_chunk_size, 512)  # BatchCompressionParams
        self.assertEqual(sketch.sketch_dimension, 48)  # leverage_sketch_size
        self.assertAlmostEqual(sketch.regularizer, 5e-3)
        self.assertAlmostEqual(sketch.blending, 0.5)   # hardcoded at the call site
        self.assertEqual(sketch.seed, 42)              # engine PHI seed
        self.assertIsNone(sketch.phi)
        self.assertEqual(CompactorSketch._PAD_BLOCK_M, 64)  # kernel BLOCK_M

    def test_invalid_params_assert(self):
        with self.assertRaises(AssertionError):
            CompactorSketch(chunk_size=0)
        with self.assertRaises(AssertionError):
            CompactorSketch(sketch_dimension=0)
        with self.assertRaises(AssertionError):
            CompactorSketch(regularizer=-1e-3)
        with self.assertRaises(AssertionError):
            CompactorSketch(compression_ratio=1.0)


# ----------------------------------------------------------------------
# 2. Full reference oracle (the centerpiece)
# ----------------------------------------------------------------------


class TestReferenceOracle(unittest.TestCase):
    TOL = dict(atol=2e-4, rtol=2e-4)

    def _check(self, module, sketch, B, S, seed, phi, tol=None):
        hidden, pre_k, post_q, post_k, values, cos, sin = _build_case(
            module, B, S, seed,
        )
        scores = sketch.score(
            module, hidden, post_k, values, None,
            {"position_embeddings": (cos, sin)},
        )
        self.assertEqual(scores.dtype, torch.float32)
        self.assertEqual(tuple(scores.shape), (B, post_k.shape[1], S))
        ref = _reference_compactor_scores(
            pre_k, post_q, post_k, phi=phi,
            chunk_size=sketch.chunk_size,
            leverage_chunk_size=sketch.leverage_chunk_size,
            regularizer=sketch.regularizer,
            blending=sketch.blending,
            first=sketch.sink_size_start,
            last=sketch.sink_size_end,
        )
        torch.testing.assert_close(scores, ref, **(tol or self.TOL))
        return scores

    def test_t700_full_chunk_plus_epilogues_gqa(self):
        # leverage: one full 512-chunk + 188 epilogue; attention: 5 full
        # 128-chunks + a 60-token epilogue (pad_rows = 64 - 60 = 4)
        module = _FakeAttnModule(
            hidden_dim=32, num_heads=4, num_kv_heads=2, head_dim=8, seed=1,
        )
        phi = _gen_phi(8, 8)
        sketch = CompactorSketch(compression_ratio=0.5, sketch_dimension=8, phi=phi)
        self._check(module, sketch, B=1, S=700, seed=11, phi=phi)

    def test_t1024_exact_multiples_no_epilogue_mha(self):
        # 1024 = 2*512 = 8*128: no leverage epilogue, no attention pad rows;
        # MHA (H_q == H_kv) exercises num_groups == 1
        module = _FakeAttnModule(
            hidden_dim=16, num_heads=2, num_kv_heads=2, head_dim=8, seed=2,
        )
        phi = _gen_phi(8, 8, seed=101)
        sketch = CompactorSketch(compression_ratio=0.5, sketch_dimension=8, phi=phi)
        self._check(module, sketch, B=1, S=1024, seed=12, phi=phi)

    def test_t300_epilogue_only_default_seeded_phi_qk_norm(self):
        # 300 < 512: leverage is a single epilogue chunk; attention has two
        # full chunks + M=44 epilogue (pad_rows = 20).  No phi injection: the
        # sketch draws the shared seeded PHI (seed=42, k=48) and the oracle
        # uses the reference transcription of that draw.  qk-norm module.
        module = _FakeAttnModule(
            hidden_dim=32, num_heads=4, num_kv_heads=2, head_dim=8, seed=3,
            qk_norm=True,
        )
        gen = torch.Generator(device="cpu").manual_seed(42)
        phi = torch.randn(8, 48, generator=gen).to(torch.float32) * (
            1.0 / math.sqrt(48)
        )
        sketch = CompactorSketch(compression_ratio=0.5)
        self._check(
            module, sketch, B=1, S=300, seed=13, phi=phi,
            tol=dict(atol=5e-4, rtol=5e-4),
        )

    def test_batch2_matches_oracle_and_z_norm_is_per_batch(self):
        module = _FakeAttnModule(
            hidden_dim=32, num_heads=4, num_kv_heads=2, head_dim=8, seed=4,
        )
        phi = _gen_phi(8, 8, seed=102)
        sketch = CompactorSketch(compression_ratio=0.5, sketch_dimension=8, phi=phi)
        hidden, pre_k, post_q, post_k, values, cos, sin = _build_case(
            module, 2, 300, seed=14,
        )
        kwargs = {"position_embeddings": (cos, sin)}
        scores = sketch.score(module, hidden, post_k, values, None, kwargs)
        ref = _reference_compactor_scores(pre_k, post_q, post_k, phi=phi)
        torch.testing.assert_close(scores, ref, **self.TOL)

        # per-batch independence: change batch 1's data only — batch 0's
        # scores (incl. every z-normalization) must be unchanged
        torch.manual_seed(99)
        hidden_b = hidden.clone()
        hidden_b[1] = torch.randn(300, module.q_proj.in_features)
        post_k_b = post_k.clone()
        post_k_b[1] = torch.randn_like(post_k[1])
        scores_b = sketch.score(module, hidden_b, post_k_b, values, None, kwargs)
        torch.testing.assert_close(scores_b[0], scores[0])
        self.assertFalse(torch.allclose(scores_b[1], scores[1]))


# ----------------------------------------------------------------------
# 3. Leverage component
# ----------------------------------------------------------------------


class TestLeverageComponent(unittest.TestCase):
    def test_regularized_leverage_solve_identity(self):
        # ||X_c @ SV||^2 with SV = V * S^-1/2 of the regularized Gram is
        # algebraically x^T (X^T X + reg I)^{-1} x — pin via torch.linalg.solve
        torch.manual_seed(2)
        x = torch.randn(2, 3, 16, 4)
        x = x - x.mean(dim=-2, keepdim=True)
        sketch = CompactorSketch(regularizer=5e-3)
        gram = (x.transpose(-1, -2) @ x).to(torch.float32)
        sv = sketch._sv_from_gram(gram, torch.float32)
        u = x @ sv
        lev = (u * u).sum(dim=-1).clamp_min(0.0)
        G = gram + 5e-3 * torch.eye(4)
        sol = torch.linalg.solve(G, x.float().transpose(-1, -2))
        expected = (x.float() * sol.transpose(-1, -2)).sum(dim=-1).clamp_min(0.0)
        torch.testing.assert_close(lev, expected, atol=1e-5, rtol=1e-4)
        self.assertTrue((lev >= 0).all())
        # _sv_from_gram regularizes a clone — the caller's Gram is not mutated
        torch.testing.assert_close(
            gram, (x.transpose(-1, -2) @ x).to(torch.float32), rtol=0, atol=0,
        )

    def test_single_batched_svd_over_all_chunks(self):
        # the reference concatenates ALL chunk Grams (full 512-chunks AND the
        # epilogue) into ONE batched SVD call — never one call per chunk-group
        module = _FakeAttnModule(
            hidden_dim=4, num_heads=1, num_kv_heads=1, head_dim=4,
            identity_k=True, seed=5,
        )
        phi = _gen_phi(4, 3, seed=117)
        sketch = CompactorSketch(leverage_chunk_size=8, sketch_dimension=3, phi=phi)
        torch.manual_seed(25)
        hidden = torch.randn(1, 22, 4)  # full chunks [8, 8] + epilogue [6]
        spy = mock.Mock(side_effect=torch.linalg.svd)
        with mock.patch.object(torch.linalg, "svd", spy):
            out = sketch._leverage_scores(module, hidden)
        self.assertEqual(spy.call_count, 1)
        self.assertEqual(tuple(out.shape), (1, 1, 22))

    def test_per_chunk_centering_invariance(self):
        # adding one constant vector to every key of a chunk leaves that
        # chunk's (mean-centered) leverage — and hence all scores — unchanged
        module = _FakeAttnModule(
            hidden_dim=4, num_heads=1, num_kv_heads=1, head_dim=4,
            identity_k=True, seed=3,
        )
        phi = _gen_phi(4, 3, seed=103)
        sketch = CompactorSketch(leverage_chunk_size=8, sketch_dimension=3, phi=phi)
        torch.manual_seed(21)
        # chunks: [8, 8, epilogue 6] — the epilogue is longer than k+1 so its
        # leverage is not (near-)constant and the eps-free per-chunk z-score
        # does not amplify fp noise into the comparison
        hidden = torch.randn(1, 22, 4)
        base = sketch._leverage_scores(module, hidden)

        shifted = hidden.clone()
        shifted[:, 8:16] += torch.tensor([5.0, -3.0, 2.0, 7.0])  # full chunk 2
        torch.testing.assert_close(
            sketch._leverage_scores(module, shifted), base, atol=1e-4, rtol=1e-4,
        )
        shifted_epi = hidden.clone()
        shifted_epi[:, 16:] += torch.tensor([1.0, 2.0, 3.0, 4.0])  # epilogue
        torch.testing.assert_close(
            sketch._leverage_scores(module, shifted_epi), base, atol=1e-4, rtol=1e-4,
        )

    def test_per_chunk_znorm_independence(self):
        # z-scores are per chunk: replacing chunk 2's data leaves chunk 1's
        # and the epilogue's z-scores untouched
        module = _FakeAttnModule(
            hidden_dim=4, num_heads=1, num_kv_heads=1, head_dim=4,
            identity_k=True, seed=3,
        )
        phi = _gen_phi(4, 3, seed=104)
        sketch = CompactorSketch(leverage_chunk_size=8, sketch_dimension=3, phi=phi)
        torch.manual_seed(22)
        hidden = torch.randn(1, 20, 4)
        base = sketch._leverage_scores(module, hidden)

        modified = hidden.clone()
        modified[:, 8:16] = torch.randn(8, 4) * 3.0
        out = sketch._leverage_scores(module, modified)
        torch.testing.assert_close(out[..., :8], base[..., :8])
        torch.testing.assert_close(out[..., 16:], base[..., 16:])
        self.assertFalse(torch.allclose(out[..., 8:16], base[..., 8:16]))

    def test_znorm_couples_heads_within_chunk(self):
        # the reference kernel normalizes over (heads x chunk tokens) jointly:
        # changing head 1's keys must move head 0's z-scores
        module = _FakeAttnModule(
            hidden_dim=8, num_heads=2, num_kv_heads=2, head_dim=4,
            identity_k=True, seed=4,
        )
        phi = _gen_phi(4, 3, seed=105)
        sketch = CompactorSketch(sketch_dimension=3, phi=phi)  # single epilogue chunk
        torch.manual_seed(23)
        hidden = torch.randn(1, 12, 8)
        base = sketch._leverage_scores(module, hidden)

        modified = hidden.clone()
        modified[..., 4:] = torch.randn(1, 12, 4) * 5.0  # head 1 channels only
        out = sketch._leverage_scores(module, modified)
        self.assertFalse(torch.allclose(out[:, 0], base[:, 0]))

    def test_nonpositive_leverage_chunk_size_single_chunk(self):
        # leverage_chunk_size <= 0 treats the whole sequence as one chunk (as
        # upstream); equal to the oracle and to any chunk size > S
        module = _FakeAttnModule(
            hidden_dim=4, num_heads=1, num_kv_heads=1, head_dim=4,
            identity_k=True, seed=5,
        )
        phi = _gen_phi(4, 3, seed=106)
        torch.manual_seed(24)
        hidden = torch.randn(1, 20, 4)
        out0 = CompactorSketch(
            leverage_chunk_size=0, sketch_dimension=3, phi=phi,
        )._leverage_scores(module, hidden)
        ref = _ref_leverage_component(
            _manual_pre_rope_k(module, hidden), phi, leverage_chunk_size=0,
        )
        torch.testing.assert_close(out0.float(), ref, atol=1e-5, rtol=1e-4)
        out_big = CompactorSketch(
            leverage_chunk_size=512, sketch_dimension=3, phi=phi,
        )._leverage_scores(module, hidden)
        torch.testing.assert_close(out0, out_big)

    def test_shared_phi_determinism_across_instances(self):
        # two fresh instances (default seed=42) give IDENTICAL scores; the
        # global torch seed is irrelevant (PHI comes from its own generator)
        module = _FakeAttnModule(seed=6)
        hidden, _, _, post_k, values, cos, sin = _build_case(module, 1, 100, seed=31)
        kwargs = {"position_embeddings": (cos, sin)}

        torch.manual_seed(0)
        r1 = CompactorSketch().score(module, hidden, post_k, values, None, kwargs)
        torch.manual_seed(12345)
        r2 = CompactorSketch().score(module, hidden, post_k, values, None, kwargs)
        self.assertTrue(torch.equal(r1, r2))

    def test_phi_cached_per_head_dim(self):
        sketch = CompactorSketch()
        p1 = sketch._get_phi(8, torch.device("cpu"), torch.float32)
        p2 = sketch._get_phi(8, torch.device("cpu"), torch.float32)
        self.assertIs(p1, p2)  # same object — drawn once, reused
        p3 = sketch._get_phi(4, torch.device("cpu"), torch.float32)
        self.assertIsNot(p3, p1)
        self.assertEqual(tuple(p1.shape), (8, 48))
        self.assertEqual(tuple(p3.shape), (4, 48))

    def test_phi_matches_reference_transcription(self):
        # PackedTensorArguments.PHI: fp32 randn from Generator(seed) -> cast
        # to model dtype -> * 1/sqrt(k)
        sketch = CompactorSketch()
        phi = sketch._get_phi(8, torch.device("cpu"), torch.float32)
        gen = torch.Generator(device="cpu").manual_seed(42)
        expected = torch.randn(8, 48, generator=gen).to(torch.float32) * (
            1.0 / math.sqrt(48)
        )
        self.assertTrue(torch.equal(phi, expected))

    def test_phi_bf16_cast_then_scale_order(self):
        # in low precision the reference casts BEFORE scaling — pin the order
        sketch = CompactorSketch()
        phi = sketch._get_phi(8, torch.device("cpu"), torch.bfloat16)
        self.assertEqual(phi.dtype, torch.bfloat16)
        gen = torch.Generator(device="cpu").manual_seed(42)
        expected = torch.randn(8, 48, generator=gen).to(torch.bfloat16) * (
            1.0 / math.sqrt(48)
        )
        self.assertTrue(torch.equal(phi, expected))
        gen2 = torch.Generator(device="cpu").manual_seed(42)
        scale_then_cast = (
            torch.randn(8, 48, generator=gen2) * (1.0 / math.sqrt(48))
        ).to(torch.bfloat16)
        self.assertFalse(torch.equal(phi, scale_then_cast))

    def test_seed_changes_phi_and_scores(self):
        module = _FakeAttnModule(seed=6)
        hidden, _, _, post_k, values, cos, sin = _build_case(module, 1, 100, seed=32)
        kwargs = {"position_embeddings": (cos, sin)}
        r42 = CompactorSketch(seed=42).score(module, hidden, post_k, values, None, kwargs)
        r7 = CompactorSketch(seed=7).score(module, hidden, post_k, values, None, kwargs)
        self.assertFalse(torch.allclose(r42, r7))
        phi42 = CompactorSketch(seed=42)._get_phi(8, torch.device("cpu"), torch.float32)
        phi7 = CompactorSketch(seed=7)._get_phi(8, torch.device("cpu"), torch.float32)
        self.assertFalse(torch.equal(phi42, phi7))

    def test_injected_phi_overrides_seed(self):
        module = _FakeAttnModule(seed=6)
        hidden, _, _, post_k, values, cos, sin = _build_case(module, 1, 100, seed=33)
        kwargs = {"position_embeddings": (cos, sin)}
        phi = _gen_phi(8, 5, seed=107)
        a = CompactorSketch(seed=42, phi=phi)
        b = CompactorSketch(seed=7, phi=phi)
        self.assertTrue(torch.equal(
            a._get_phi(8, torch.device("cpu"), torch.float32), phi,
        ))
        ra = a.score(module, hidden, post_k, values, None, kwargs)
        rb = b.score(module, hidden, post_k, values, None, kwargs)
        self.assertTrue(torch.equal(ra, rb))  # seed irrelevant with injection
        r_default = CompactorSketch().score(module, hidden, post_k, values, None, kwargs)
        self.assertFalse(torch.allclose(ra, r_default))


# ----------------------------------------------------------------------
# 4. Non-causal chunked attention component
# ----------------------------------------------------------------------


class TestNonCausalComponent(unittest.TestCase):
    def test_two_token_hand_pin(self):
        # two tokens, two MHA heads, identity RoPE: hand-computed fp32
        # softmax column sums with sm_scale = 1.0 (NO 1/sqrt(d)), plus the
        # M=2 epilogue pad rows ceil(2/64)*64 - 2 = 62, each adding 1/128;
        # z-scored over the 4 (head, token) entries with python floats.
        module = _FakeAttnModule(
            hidden_dim=4, num_heads=2, num_kv_heads=2, head_dim=2,
            identity_q=True, seed=7,
        )
        hidden = torch.tensor([[[1.0, 0.0, 3.0, 0.0], [0.0, 2.0, 0.0, 1.0]]])
        # q head0 = [[1,0],[0,2]], q head1 = [[3,0],[0,1]]; keys = I rows
        keys = torch.tensor(
            [[[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]]
        )
        cos, sin = _identity_pos_emb(1, 2, 2)
        out = CompactorSketch()._non_causal_scores(module, hidden, keys, cos, sin)

        e = math.e
        pad = 62.0 / 128.0
        h0c0 = e / (e + 1) + 1 / (1 + e ** 2) + pad
        h0c1 = 1 / (e + 1) + e ** 2 / (1 + e ** 2) + pad
        h1c0 = e ** 3 / (e ** 3 + 1) + 1 / (1 + e) + pad
        h1c1 = 1 / (e ** 3 + 1) + e / (1 + e) + pad
        vals = [h0c0, h0c1, h1c0, h1c1]
        mean = sum(vals) / 4
        var = sum(v * v for v in vals) / 4 - mean * mean
        expected = torch.tensor(
            [[[(h0c0 - mean), (h0c1 - mean)], [(h1c0 - mean), (h1c1 - mean)]]]
        ) / math.sqrt(var)
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)

    def test_gqa_group_sum_pin(self):
        # both group members' column sums are SUMMED into the KV head (the
        # kernel's tl.sum over BLOCK_M * QUERY_GROUP_SIZE rows).  A global
        # positive scale is invisible after the z-score, so this pins
        # combination-by-sum up to that inherent scale invariance — plus that
        # the SECOND group member genuinely contributes.
        module = _FakeAttnModule(
            hidden_dim=4, num_heads=2, num_kv_heads=1, head_dim=2,
            identity_q=True, seed=8,
        )
        torch.manual_seed(41)
        hidden = torch.randn(1, 6, 4)
        keys = torch.randn(1, 1, 6, 2)
        cos, sin = _identity_pos_emb(1, 6, 2)
        out = CompactorSketch()._non_causal_scores(module, hidden, keys, cos, sin)

        qa, qb = hidden[0, :, :2], hidden[0, :, 2:4]
        k0 = keys[0, 0]

        def cols(q):
            return torch.softmax((q @ k0.transpose(0, 1)).float(), dim=-1).sum(dim=0)

        pad = (math.ceil(6 / 64) * 64 - 6) * (1.0 / 128.0)  # 58 pad rows/head
        raw = (cols(qa) + pad) + (cols(qb) + pad)
        expected = _ref_zscore_flat(raw).view(1, 1, 6)
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)

        hidden2 = hidden.clone()
        hidden2[..., 2:4] = torch.randn(1, 6, 2)  # second group member only
        out2 = CompactorSketch()._non_causal_scores(module, hidden2, keys, cos, sin)
        self.assertFalse(torch.allclose(out2, out))

    def test_epilogue_pad_rows_constant_m44(self):
        # S = 172 -> chunks [128, 44]; the M=44 epilogue has
        # ceil(44/64)*64 - 44 = 20 padded query rows PER QUERY HEAD, each
        # adding 1/128 to every column: total 20 * (1/128) * H_g with H_g=2.
        module = _FakeAttnModule(
            hidden_dim=16, num_heads=2, num_kv_heads=1, head_dim=8, seed=9,
        )
        torch.manual_seed(42)
        S = 172
        hidden = torch.randn(1, S, 16)
        keys = torch.randn(1, 1, S, 8)
        cos, sin = _identity_pos_emb(1, S, 8)
        out = CompactorSketch()._non_causal_scores(module, hidden, keys, cos, sin)

        q = module.q_proj(hidden).view(1, S, 2, 8).transpose(1, 2)
        raw = torch.zeros(1, 1, S)
        for h in range(2):
            for (s0, s1) in ((0, 128), (128, S)):
                p = torch.softmax(
                    (q[0, h, s0:s1] @ keys[0, 0, s0:s1].transpose(0, 1)).float(),
                    dim=-1,
                )
                raw[0, 0, s0:s1] += p.sum(dim=0)

        with_pad = raw.clone()
        with_pad[..., 128:] += 2 * 20 * (1.0 / 128.0)
        torch.testing.assert_close(
            out, _ref_zscore_flat(with_pad[0]).view(1, 1, S), atol=1e-5, rtol=1e-5,
        )
        # neither zero padding nor pad-to-chunk_size (128 - 44 = 84) matches:
        # the tile height is BLOCK_M = 64 and the denominator is chunk_size
        for wrong_pad_rows in (0, 84):
            wrong = raw.clone()
            wrong[..., 128:] += 2 * wrong_pad_rows * (1.0 / 128.0)
            self.assertFalse(
                torch.allclose(out, _ref_zscore_flat(wrong[0]).view(1, 1, S), atol=1e-3)
            )

    def test_block_diagonal_chunks_and_sequence_znorm_coupling(self):
        # raw column sums are block-diagonal in 128-chunks (oracle pin), but
        # the z-score is per SEQUENCE, so chunk-2 changes shift chunk-1 z
        # values only through an affine (mean/var) transform.
        torch.manual_seed(51)
        q = torch.randn(1, 2, 256, 4)
        k = torch.randn(1, 2, 256, 4)
        q2, k2 = q.clone(), k.clone()
        q2[..., 128:, :] = torch.randn(1, 2, 128, 4)
        k2[..., 128:, :] = torch.randn(1, 2, 128, 4)

        raw_a = _ref_non_causal_component(q, k, 128, normalize=False)
        raw_b = _ref_non_causal_component(q2, k2, 128, normalize=False)
        self.assertTrue(torch.equal(raw_a[..., :128], raw_b[..., :128]))
        self.assertFalse(torch.allclose(raw_a[..., 128:], raw_b[..., 128:]))

        # implementation-level: with blending=0 and no sinks, scores are the
        # per-sequence z of the raw sums; re-standardizing the chunk-1 slice
        # removes that affine map, so it must be invariant to chunk-2 edits
        module = _FakeAttnModule(
            hidden_dim=8, num_heads=2, num_kv_heads=2, head_dim=4,
            identity_q=True, seed=10,
        )
        hidden_a = q.transpose(1, 2).reshape(1, 256, 8)
        hidden_b = q2.transpose(1, 2).reshape(1, 256, 8)
        values = torch.randn(1, 2, 256, 4)
        cos, sin = _identity_pos_emb(1, 256, 4)
        sketch = CompactorSketch(blending=0.0, sink_size_start=0, sink_size_end=0)
        kwargs = {"position_embeddings": (cos, sin)}
        s_a = sketch.score(module, hidden_a, k, values, None, kwargs)
        s_b = sketch.score(module, hidden_b, k2, values, None, kwargs)
        self.assertFalse(torch.allclose(s_a[..., :128], s_b[..., :128]))  # z couples
        torch.testing.assert_close(
            _standardize(s_a[0, :, :128]), _standardize(s_b[0, :, :128]),
            atol=1e-5, rtol=1e-5,
        )


# ----------------------------------------------------------------------
# 5. Blend + sink protection
# ----------------------------------------------------------------------


class TestBlendAndSinks(unittest.TestCase):
    def _case(self, S=200, seed=15):
        module = _FakeAttnModule(
            hidden_dim=32, num_heads=4, num_kv_heads=2, head_dim=8, seed=seed,
        )
        phi = _gen_phi(8, 8, seed=108)
        hidden, pre_k, post_q, post_k, values, cos, sin = _build_case(
            module, 1, S, seed=seed + 100,
        )
        kwargs = {"position_embeddings": (cos, sin)}
        return module, phi, hidden, pre_k, post_q, post_k, values, kwargs

    def test_blending_default_and_zero_blend_attention_only(self):
        module, phi, hidden, pre_k, post_q, post_k, values, kwargs = self._case()
        default = CompactorSketch(phi=phi)
        self.assertAlmostEqual(default.blending, 0.5)
        s_half = default.score(module, hidden, post_k, values, None, kwargs)
        s_zero = CompactorSketch(phi=phi, blending=0.0).score(
            module, hidden, post_k, values, None, kwargs,
        )
        ref_zero = _reference_compactor_scores(
            pre_k, post_q, post_k, phi=phi, blending=0.0,
        )
        ref_half = _reference_compactor_scores(
            pre_k, post_q, post_k, phi=phi, blending=0.5,
        )
        torch.testing.assert_close(s_zero, ref_zero, atol=2e-4, rtol=2e-4)
        torch.testing.assert_close(s_half, ref_half, atol=2e-4, rtol=2e-4)
        # leverage genuinely contributes at the default blending
        self.assertFalse(
            torch.allclose(s_zero[..., 16:136], s_half[..., 16:136])
        )
        # and the difference is exactly 0.5 * lev_z (blend is linear)
        lev_z = _ref_leverage_component(pre_k, phi)
        torch.testing.assert_close(
            (s_half - s_zero)[..., 16:136], (0.5 * lev_z)[..., 16:136],
            atol=5e-4, rtol=5e-4,
        )

    def test_inf_protection_spans_all_heads(self):
        module, phi, hidden, _, _, post_k, values, kwargs = self._case(seed=16)
        scores = CompactorSketch(phi=phi).score(
            module, hidden, post_k, values, None, kwargs,
        )
        T = 200
        self.assertTrue(torch.isinf(scores[:, :, :16]).all())
        self.assertTrue((scores[:, :, :16] > 0).all())
        self.assertTrue(torch.isinf(scores[:, :, T - 64:]).all())
        self.assertTrue(torch.isfinite(scores[:, :, 16:T - 64]).all())

    def test_protected_tokens_included_in_z_norm_stats(self):
        # +inf is applied AFTER scoring: a huge-norm key at protected position
        # 3 stays inside the per-sequence attention z statistics.  Columns of
        # chunks >= 1 (tokens 128..236) have block-diagonal raw sums that
        # cannot see position 3, and the leverage input (hidden) is untouched,
        # so any change there is purely the z-norm mean/var shifting.
        module, phi, hidden, _, _, post_k, values, kwargs = self._case(
            S=300, seed=17,
        )
        sketch = CompactorSketch(phi=phi)
        base = sketch.score(module, hidden, post_k, values, None, kwargs)
        boosted_k = post_k.clone()
        boosted_k[..., 3, :] *= 50.0
        boosted = sketch.score(module, hidden, boosted_k, values, None, kwargs)
        self.assertFalse(
            torch.allclose(base[..., 128:236], boosted[..., 128:236])
        )
        for s in (base, boosted):  # spans still protected in both runs
            self.assertTrue(torch.isinf(s[:, :, :16]).all())
            self.assertTrue(torch.isinf(s[:, :, 236:]).all())

    def test_protection_covers_sequence_compress_noop(self):
        # T <= sink_size_start + sink_size_end: the engine forces
        # compression_ratio = 1.0 (keep everything) — compress must be a
        # no-op returning the SAME tensor objects, without scoring
        module = _FakeAttnModule(seed=18)
        sketch = CompactorSketch(compression_ratio=0.5)
        for T in (70, 80):
            torch.manual_seed(T)
            hidden = torch.randn(1, T, 32)
            keys = torch.randn(1, 2, T, 8)
            values = torch.randn(1, 2, T, 8)
            with mock.patch.object(CompactorSketch, "score") as spy:
                out_k, out_v = sketch.compress(module, hidden, keys, values, None, {})
            self.assertIs(out_k, keys)
            self.assertIs(out_v, values)
            spy.assert_not_called()

    def test_score_protection_clamps_to_cover_short_sequences(self):
        # T=70: first = min(16, 70) = 16, last = min(64, 70-16) = 54 — the
        # clamped spans cover every position, so scores are all +inf
        module = _FakeAttnModule(seed=19)
        hidden, _, _, post_k, values, cos, sin = _build_case(module, 1, 70, seed=35)
        scores = CompactorSketch().score(
            module, hidden, post_k, values, None,
            {"position_embeddings": (cos, sin)},
        )
        self.assertTrue(torch.isinf(scores).all())

    def test_just_above_protection_compresses(self):
        module = _FakeAttnModule(seed=20)
        hidden, _, _, post_k, values, cos, sin = _build_case(module, 1, 81, seed=36)
        sketch = CompactorSketch(compression_ratio=0.5, phi=_gen_phi(8, 8, seed=109))
        out_k, out_v = sketch.compress(
            module, hidden, post_k, values, None,
            {"position_embeddings": (cos, sin)},
        )
        self.assertEqual(out_k.shape[2], int(81 * 0.5))
        self.assertEqual(out_v.shape[2], int(81 * 0.5))


# ----------------------------------------------------------------------
# 6. Compress / budget (framework contract)
# ----------------------------------------------------------------------


class TestCompressBudget(unittest.TestCase):
    def _case(self, S=200, ratio=0.4, seed=25):
        module = _FakeAttnModule(
            hidden_dim=32, num_heads=4, num_kv_heads=2, head_dim=8, seed=seed,
        )
        phi = _gen_phi(8, 8, seed=110)
        sketch = CompactorSketch(compression_ratio=ratio, phi=phi)
        hidden, _, _, post_k, values, cos, sin = _build_case(
            module, 1, S, seed=seed + 200,
        )
        kwargs = {"position_embeddings": (cos, sin)}
        return module, sketch, hidden, post_k, values, kwargs

    def test_budget_exact_and_gather_in_score_order(self):
        module, sketch, hidden, keys, values, kwargs = self._case()
        out_k, out_v = sketch.compress(module, hidden, keys, values, None, kwargs)
        n_kept = int(200 * (1 - 0.4))
        self.assertEqual(tuple(out_k.shape), (1, 2, n_kept, 8))
        self.assertEqual(tuple(out_v.shape), (1, 2, n_kept, 8))
        # deterministic phi -> re-scoring reproduces the selection: output is
        # the gather at topk(score) indices, in SCORE order (not sorted)
        scores = sketch.score(module, hidden, keys, values, None, kwargs)
        idx = scores.topk(n_kept, dim=-1).indices
        gidx = idx.unsqueeze(-1).expand(-1, -1, -1, 8)
        self.assertTrue(torch.equal(out_k, keys.gather(2, gidx)))
        self.assertTrue(torch.equal(out_v, values.gather(2, gidx)))

    def test_protected_indices_always_kept(self):
        module, sketch, hidden, keys, values, kwargs = self._case()
        n_kept = int(200 * (1 - 0.4))  # 120 >= 80 protected
        scores = sketch.score(module, hidden, keys, values, None, kwargs)
        idx = scores.topk(n_kept, dim=-1).indices
        protected = set(range(16)) | set(range(200 - 64, 200))
        for h in range(2):
            kept = set(idx[0, h].tolist())
            self.assertTrue(protected.issubset(kept))
            self.assertEqual(len(kept), n_kept)

    def test_prefill_only_assertion(self):
        module = _FakeAttnModule(seed=26)
        with self.assertRaisesRegex(AssertionError, "prefill"):
            CompactorSketch(compression_ratio=0.5).score(
                module, torch.randn(1, 4, 32),
                torch.randn(1, 2, 100, 8), torch.randn(1, 2, 100, 8), None, {},
            )

    def test_zero_ratio_noop_score_never_called(self):
        module = _FakeAttnModule(seed=27)
        keys = torch.randn(1, 2, 100, 8)  # above the protection no-op range
        values = torch.randn(1, 2, 100, 8)
        sketch = CompactorSketch(compression_ratio=0.0)
        with mock.patch.object(CompactorSketch, "score") as spy:
            out_k, out_v = sketch.compress(
                module, torch.randn(1, 100, 32), keys, values, None, {},
            )
        self.assertIs(out_k, keys)
        self.assertIs(out_v, values)
        spy.assert_not_called()

    def test_forward_hook_prefill_then_decode_noop(self):
        from transformers import DynamicCache

        module = _FakeAttnModule(
            hidden_dim=16, num_heads=2, num_kv_heads=2, head_dim=8, seed=28,
        )
        phi = _gen_phi(8, 8, seed=111)
        sketch = CompactorSketch(compression_ratio=0.5, phi=phi)
        hidden, _, _, post_k, values, cos, sin = _build_case(module, 1, 100, seed=61)
        cache = DynamicCache()
        cache.update(post_k.clone(), values.clone(), 0)
        prefill_kwargs = {
            "hidden_states": hidden,
            "past_key_values": cache,
            "cache_position": torch.arange(100),
            "position_embeddings": (cos, sin),
        }
        output = (torch.randn(1, 100, 16), None)
        result = sketch.forward_hook(module, [], prefill_kwargs, output)
        self.assertIs(result, output)
        expected_k, expected_v = sketch.compress(
            module, hidden, post_k, values, None,
            {"position_embeddings": (cos, sin)},
        )
        self.assertEqual(tuple(cache.layers[0].keys.shape), (1, 2, 50, 8))
        self.assertTrue(torch.equal(cache.layers[0].keys, expected_k))
        self.assertTrue(torch.equal(cache.layers[0].values, expected_v))

        kept_k, kept_v = cache.layers[0].keys, cache.layers[0].values
        decode_kwargs = {
            "hidden_states": torch.randn(1, 1, 16),
            "past_key_values": cache,
            "cache_position": torch.tensor([100]),
        }
        with mock.patch.object(CompactorSketch, "score") as spy:
            sketch.forward_hook(
                module, [], decode_kwargs, (torch.randn(1, 1, 16), None),
            )
        spy.assert_not_called()
        self.assertIs(cache.layers[0].keys, kept_k)
        self.assertIs(cache.layers[0].values, kept_v)


# ----------------------------------------------------------------------
# 7. Projection / position-embedding paths
# ----------------------------------------------------------------------


class TestProjectionPaths(unittest.TestCase):
    def test_qk_norm_applied(self):
        module = _FakeAttnModule(seed=7, qk_norm=True)
        torch.manual_seed(70)
        hidden = torch.randn(2, 10, 32)
        q = _get_prerope_query_states(module, hidden)
        k = _get_prerope_key_states(module, hidden)
        plain_q = module.q_proj(hidden).view(2, 10, 4, 8).transpose(1, 2)
        plain_k = module.k_proj(hidden).view(2, 10, 2, 8).transpose(1, 2)
        torch.testing.assert_close(q, module.q_norm(plain_q))
        torch.testing.assert_close(k, module.k_norm(plain_k))
        self.assertFalse(torch.allclose(q, plain_q))
        self.assertFalse(torch.allclose(k, plain_k))

    def test_fused_qkv_slicing(self):
        module = _FakeFusedAttnModule()
        torch.manual_seed(71)
        hidden = torch.randn(1, 6, 16)
        q = _get_prerope_query_states(module, hidden)
        k = _get_prerope_key_states(module, hidden)
        qkv = module.qkv_proj(hidden)  # [q (8) | k (4) | v (4)]
        torch.testing.assert_close(q, qkv[..., :8].view(1, 6, 2, 4).transpose(1, 2))
        torch.testing.assert_close(k, qkv[..., 8:12].view(1, 6, 1, 4).transpose(1, 2))

    def test_gated_qproj_slicing(self):
        module = _FakeGatedQProjModule()
        torch.manual_seed(72)
        hidden = torch.randn(1, 6, 16)
        q = _get_prerope_query_states(module, hidden)
        manual = (
            module.q_proj(hidden).view(1, 6, 2, 8)[..., :4]
            .reshape(1, 6, 2, 4).transpose(1, 2)
        )
        torch.testing.assert_close(q, manual)
        self.assertEqual(tuple(q.shape), (1, 2, 6, 4))

    def test_unsupported_module_raises(self):
        module = nn.Module()
        module.config = SimpleNamespace(num_attention_heads=2)
        module.head_dim = 4
        with self.assertRaises(NotImplementedError):
            _get_prerope_query_states(module, torch.randn(1, 3, 8))
        with self.assertRaises(NotImplementedError):
            _get_prerope_key_states(module, torch.randn(1, 3, 8))

    def test_partial_rotary_rotates_first_channels_only(self):
        # rotary_dim = 4 < head_dim = 8 (Qwen3.5-style): only the first 4
        # channels of q are rotated before the non-causal q.k logits
        module = _FakeAttnModule(
            hidden_dim=16, num_heads=2, num_kv_heads=1, head_dim=8, seed=29,
        )
        S = 40
        torch.manual_seed(73)
        hidden = torch.randn(1, S, 16)
        cos, sin = _rope_cos_sin(S, 4)
        pre_q = _manual_pre_rope_q(module, hidden)
        pre_k = _manual_pre_rope_k(module, hidden)
        post_q = _ref_apply_rope(pre_q, cos, sin)
        post_k = _ref_apply_rope(pre_k, cos, sin)
        out = CompactorSketch()._non_causal_scores(module, hidden, post_k, cos, sin)
        expected = _ref_non_causal_component(post_q, post_k, 128)
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)

    def test_position_embeddings_kwarg_preferred(self):
        module = _FakeAttnModule(seed=30)
        hidden, _, _, post_k, values, cos, sin = _build_case(module, 1, 100, seed=74)
        phi = _gen_phi(8, 8, seed=112)
        sketch = CompactorSketch(phi=phi)
        baseline = sketch.score(
            module, hidden, post_k, values, None,
            {"position_embeddings": (cos, sin)},
        )

        def _must_not_be_called(*args, **kwargs):
            raise AssertionError("rotary_emb must not be called")

        module.rotary_emb = _must_not_be_called
        with_rotary = sketch.score(
            module, hidden, post_k, values, None,
            {"position_embeddings": (cos, sin)},
        )
        self.assertTrue(torch.equal(with_rotary, baseline))

    def test_rotary_emb_fallback_uses_cache_position(self):
        module = _FakeAttnModule(seed=30)
        hidden, _, _, post_k, values, cos, sin = _build_case(module, 1, 100, seed=75)
        phi = _gen_phi(8, 8, seed=113)
        sketch = CompactorSketch(phi=phi)
        explicit = sketch.score(
            module, hidden, post_k, values, None,
            {"position_embeddings": (cos, sin)},
        )
        rotary = _RecordingRotary(cos, sin)
        module.rotary_emb = rotary
        fallback = sketch.score(module, hidden, post_k, values, None, {})
        self.assertTrue(torch.equal(fallback, explicit))
        self.assertTrue(torch.equal(
            rotary.calls[0], torch.arange(100).unsqueeze(0),
        ))
        cache_position = torch.arange(100) + 3
        sketch.score(
            module, hidden, post_k, values, None,
            {"cache_position": cache_position},
        )
        self.assertTrue(torch.equal(rotary.calls[1], cache_position.unsqueeze(0)))

    def test_no_rope_identity_fallback(self):
        # NemotronH-style: no rotary_emb, no position_embeddings ->
        # cos = 1 / sin = 0, i.e. raw q against raw cached keys
        module = _FakeAttnModule(seed=31)
        self.assertFalse(hasattr(module, "rotary_emb"))
        torch.manual_seed(76)
        hidden = torch.randn(1, 100, 32)
        keys = torch.randn(1, 2, 100, 8)
        values = torch.randn(1, 2, 100, 8)
        sketch = CompactorSketch(phi=_gen_phi(8, 8, seed=114))
        cos, sin = sketch._position_embeddings(module, hidden, {})
        self.assertTrue(torch.equal(cos, torch.ones(1, 100, 8)))
        self.assertTrue(torch.equal(sin, torch.zeros(1, 100, 8)))
        implicit = sketch.score(module, hidden, keys, values, None, {})
        ones, zeros = _identity_pos_emb(1, 100, 8)
        explicit = sketch.score(
            module, hidden, keys, values, None,
            {"position_embeddings": (ones, zeros)},
        )
        self.assertTrue(torch.equal(implicit, explicit))


# ----------------------------------------------------------------------
# 8. QR fallback (reference's un-regularized leverage path)
# ----------------------------------------------------------------------


class TestQRFallback(unittest.TestCase):
    def test_qr_helpers_match_unregularized_reference(self):
        # both branches: row norms^2 of Q from the reduced QR of the centered
        # fp32 chunk == the UN-regularized exact leverage x^T (X^T X)^{-1} x
        torch.manual_seed(80)
        x = torch.randn(1, 1, 12, 3)
        x = x - x.mean(dim=-2, keepdim=True)
        Q, _ = torch.linalg.qr(x.float(), mode="reduced")
        row_norms_sq = (Q * Q).sum(dim=-1)
        G = x[0, 0].transpose(0, 1) @ x[0, 0]
        sol = torch.linalg.solve(G, x[0, 0].transpose(0, 1))
        exact = (x[0, 0] * sol.transpose(0, 1)).sum(dim=-1)
        for helper in (
            CompactorSketch._leverage_qr_full,
            CompactorSketch._leverage_qr_epilogue,
        ):
            lev = helper(x)
            torch.testing.assert_close(lev, row_norms_sq, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(lev[0, 0], exact, atol=1e-5, rtol=1e-4)

    def test_qr_branch_dtype_order_quirk_bf16(self):
        # replicated upstream quirk: the full-chunk branch casts Q back to
        # model dtype BEFORE squaring (compactor.py:308-311), the epilogue
        # branch squares in fp32 THEN casts (compactor.py:324-325) — the two
        # orders differ in bf16
        torch.manual_seed(82)
        x = (torch.randn(1, 1, 12, 3) * 1.7).to(torch.bfloat16)
        x = x - x.mean(dim=-2, keepdim=True)
        Q, _ = torch.linalg.qr(x.to(torch.float32), mode="reduced")
        Qb = Q.to(torch.bfloat16)
        full_expected = (Qb * Qb).sum(dim=-1).clamp_min(0.0)
        epi_expected = (Q * Q).sum(dim=-1).to(torch.bfloat16)
        lev_full = CompactorSketch._leverage_qr_full(x)
        lev_epi = CompactorSketch._leverage_qr_epilogue(x)
        self.assertEqual(lev_full.dtype, torch.bfloat16)
        self.assertEqual(lev_epi.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(lev_full, full_expected))
        self.assertTrue(torch.equal(lev_epi, epi_expected))
        self.assertFalse(torch.equal(lev_full, lev_epi))

    def test_qr_fallback_after_two_svd_failures_matches_unregularized_oracle(self):
        # the ONE batched SVD fails, the 10x-regularizer retry fails ->
        # the ENTIRE sequence (full chunks AND epilogue) falls back to the
        # un-regularized QR leverage, z-scored per chunk as usual
        module = _FakeAttnModule(
            hidden_dim=4, num_heads=1, num_kv_heads=1, head_dim=4,
            identity_k=True, seed=32,
        )
        phi = _gen_phi(4, 3, seed=115)
        sketch = CompactorSketch(leverage_chunk_size=8, sketch_dimension=3, phi=phi)
        torch.manual_seed(81)
        # full chunks [8, 8] + epilogue [6].  (An epilogue of exactly k+1
        # tokens would make the UN-regularized leverage constant at
        # (L-1)/L -> var 0 -> NaN on both sides; use 6 > k+1 tokens.)
        hidden = torch.randn(1, 22, 4)
        svd_mock = mock.Mock(side_effect=RuntimeError("synthetic SVD failure"))
        with mock.patch.object(torch.linalg, "svd", svd_mock):
            out = sketch._leverage_scores(module, hidden)
        # first try + the 10x-regularizer retry both failed before QR
        self.assertEqual(svd_mock.call_count, 2)

        # oracle: per-chunk UN-regularized leverage via solve, z per chunk
        pre_k = _manual_pre_rope_k(module, hidden)
        expected = torch.empty(1, 1, 22)
        for (s0, s1) in _ref_chunk_bounds(22, 8):
            Xc = pre_k[0, 0, s0:s1] @ phi
            Xc = (Xc - Xc.mean(dim=0, keepdim=True)).float()
            sol = torch.linalg.solve(Xc.transpose(0, 1) @ Xc, Xc.transpose(0, 1))
            lev = (Xc * sol.transpose(0, 1)).sum(dim=-1).clamp_min(0.0)
            expected[0, :, s0:s1] = _ref_zscore_flat(lev.view(1, -1))
        torch.testing.assert_close(out.float(), expected, atol=1e-4, rtol=1e-4)


# ----------------------------------------------------------------------
# 9. Replicated hazards (NaN from the eps-free z-score)
# ----------------------------------------------------------------------


class TestReplicatedHazards(unittest.TestCase):
    def test_zero_phi_leverage_nan(self):
        # phi = 0 -> X = 0 -> centered 0 -> lev all 0 -> biased var 0 ->
        # 0/0 = NaN, exactly as the eps-free reference kernel would produce
        module = _FakeAttnModule(seed=33)
        hidden, _, _, post_k, values, cos, sin = _build_case(module, 1, 100, seed=90)
        sketch = CompactorSketch(phi=torch.zeros(8, 4))
        lev = sketch._leverage_scores(module, hidden)
        self.assertTrue(torch.isnan(lev).all())
        scores = sketch.score(
            module, hidden, post_k, values, None,
            {"position_embeddings": (cos, sin)},
        )
        # interior (16..36 for T=100, last span = min(64, 84) = 64) is NaN;
        # the protection overwrite still lands +inf on the spans afterward
        self.assertTrue(torch.isnan(scores[:, :, 16:36]).all())
        self.assertTrue(torch.isinf(scores[:, :, :16]).all())
        self.assertTrue(torch.isinf(scores[:, :, 36:]).all())

    def test_constant_keys_attention_nan(self):
        # constant cached keys -> every softmax row is uniform -> every raw
        # column sum is identical -> per-sequence var 0 -> NaN (attention
        # side), while the leverage side stays finite (random hidden)
        module = _FakeAttnModule(seed=34)
        torch.manual_seed(91)
        S = 256  # exact 128-multiple: no pad-row asymmetry between chunks
        hidden = torch.randn(1, S, 32)
        keys = torch.ones(1, 2, S, 8)
        values = torch.randn(1, 2, S, 8)
        cos, sin = _identity_pos_emb(1, S, 8)
        sketch = CompactorSketch(phi=_gen_phi(8, 8, seed=116))
        attn = sketch._non_causal_scores(module, hidden, keys, cos, sin)
        self.assertTrue(torch.isnan(attn).all())
        lev = sketch._leverage_scores(module, hidden)
        self.assertTrue(torch.isfinite(lev).all())
        scores = sketch.score(
            module, hidden, keys, values, None,
            {"position_embeddings": (cos, sin)},
        )
        self.assertTrue(torch.isnan(scores[:, :, 16:S - 64]).all())


# ----------------------------------------------------------------------
# 10. Dtype
# ----------------------------------------------------------------------


class TestDtype(unittest.TestCase):
    def test_bf16_end_to_end(self):
        module = _FakeAttnModule(
            hidden_dim=32, num_heads=4, num_kv_heads=2, head_dim=8, seed=35,
        ).to(torch.bfloat16)
        S = 200
        torch.manual_seed(92)
        hidden = torch.randn(1, S, 32).bfloat16()
        keys = torch.randn(1, 2, S, 8).bfloat16()
        values = torch.randn(1, 2, S, 8).bfloat16()
        cos, sin = _rope_cos_sin(S, 8)
        kwargs = {"position_embeddings": (cos.bfloat16(), sin.bfloat16())}
        sketch = CompactorSketch(compression_ratio=0.5)
        scores = sketch.score(module, hidden, keys, values, None, kwargs)
        self.assertEqual(scores.dtype, torch.float32)  # scores stay fp32
        self.assertTrue(torch.isinf(scores[:, :, :16]).all())
        self.assertTrue(torch.isinf(scores[:, :, S - 64:]).all())
        self.assertTrue(torch.isfinite(scores[:, :, 16:S - 64]).all())
        out_k, out_v = sketch.compress(module, hidden, keys, values, None, kwargs)
        self.assertEqual(out_k.dtype, torch.bfloat16)
        self.assertEqual(out_v.dtype, torch.bfloat16)
        self.assertEqual(tuple(out_k.shape), (1, 2, 100, 8))
        self.assertEqual(tuple(out_v.shape), (1, 2, 100, 8))


if __name__ == "__main__":
    unittest.main()
