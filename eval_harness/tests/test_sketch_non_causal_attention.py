"""Tests for the NonCausalAttnSketch port of kvpress's NonCausalAttnPress.

The oracles below are independent transcriptions of the kvpress 0.5.1 math
(``kvpress/presses/non_causal_attention_press.py``): a loop-based chunked
attention (vs the production vectorized reshape), a shift-and-add 3-tap
average pooling (vs ``F.avg_pool1d``), and an explicit stack-based GQA group
mean (vs the production ``view(...).mean``).  Upstream quirks pinned: no
``1/sqrt(d)`` scaling, the ``-1e-9`` (effectively unmasked) key-pad constant,
``count_include_pad`` edge depression, and the global z-normalization.
"""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM

from eval_harness.kv_compression.cache_adapter import create_cache_adapter
from eval_harness.research_pipeline import ResearchGenerationPipeline
from eval_harness.kv_compression.compressors.non_causal_attention_sketch import NonCausalAttnSketch
from eval_harness.kv_compression.registry import get_kv_compressor, get_kv_compressor_class


# ======================================================================
# Fakes and reference oracles (local to this module by convention)
# ======================================================================


class _FakeAttnModule(nn.Module):
    """Minimal Llama-like attention module: q_proj + the attrs score()/compress() read."""

    def __init__(self, hidden_dim=8, num_heads=2, head_dim=4, num_kv_heads=None,
                 identity_q=False, seed=0):
        super().__init__()
        num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.config = SimpleNamespace(num_attention_heads=num_heads)
        self.num_key_value_heads = num_kv_heads
        self.head_dim = head_dim
        self.layer_idx = 0
        self.q_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        if identity_q:
            assert hidden_dim == num_heads * head_dim
            with torch.no_grad():
                self.q_proj.weight.copy_(torch.eye(hidden_dim))
        else:
            torch.manual_seed(seed)
            with torch.no_grad():
                self.q_proj.weight.normal_()


class _FakePhi3AttnModule(nn.Module):
    """Phi3-style fused qkv_proj module (no q_proj attribute)."""

    def __init__(self, hidden_dim, num_heads, head_dim, num_kv_heads, seed=0):
        super().__init__()
        self.config = SimpleNamespace(num_attention_heads=num_heads)
        self.num_key_value_heads = num_kv_heads
        self.head_dim = head_dim
        self.layer_idx = 0
        out = (num_heads + 2 * num_kv_heads) * head_dim
        self.qkv_proj = nn.Linear(hidden_dim, out, bias=False)
        torch.manual_seed(seed)
        with torch.no_grad():
            self.qkv_proj.weight.normal_()


class _DoublingNorm(nn.Module):
    def forward(self, x):
        return x * 2.0


class _FakeCacheLayer:
    def __init__(self, keys, values):
        self.keys = keys
        self.values = values


class _FakeCache:
    def __init__(self, keys, values):
        self.layers = [_FakeCacheLayer(keys, values)]


def _identity_pos_emb(B, S, D):
    """RoPE that is a no-op: cos=1, sin=0."""
    return torch.ones(B, S, D), torch.zeros(B, S, D)


def _manual_rope(S, D, base=10000.0):
    """Real (cos, sin) of shape [1, S, D] for positions 0..S-1, unit amplitude."""
    half = D // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))
    freqs = torch.arange(S, dtype=torch.float32)[:, None] * inv_freq[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos()[None], emb.sin()[None]


def _rotate_half_ref(x):
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def _chunked_attn_reference(q, k, chunk_size, key_pad_fill=-1e-9):
    """Loop-based transcription of kvpress non_causal_chunked_attn (:63-93).

    ``key_pad_fill`` defaults to the upstream ``-1e-9`` quirk; ``-1e9`` builds
    the would-be "properly masked" variant the tests prove the production code
    does NOT compute.
    """
    B, H, S, d = k.shape
    S_pad = math.ceil(S / chunk_size) * chunk_size
    pad_len = S_pad - S
    if pad_len > 0:
        qp = torch.cat([q, torch.zeros(B, H, pad_len, d, dtype=q.dtype)], dim=2)
        kp = torch.cat([k, torch.zeros(B, H, pad_len, d, dtype=k.dtype)], dim=2)
    else:
        qp, kp = q, k
    num_chunks = S_pad // chunk_size
    out = torch.zeros(B, H, S_pad, dtype=torch.float32)
    for c in range(num_chunks):
        lo, hi = c * chunk_size, (c + 1) * chunk_size
        dots = torch.matmul(qp[:, :, lo:hi], kp[:, :, lo:hi].transpose(-2, -1))
        if c == num_chunks - 1 and pad_len > 0:
            n_valid = S - lo
            dots[:, :, n_valid:, :] = 0.0
            dots[:, :, :, n_valid:] = key_pad_fill
        attn = torch.softmax(dots.to(torch.float32), dim=-1)
        out[:, :, lo:hi] = attn.sum(dim=-2)
    return out[..., :S]


def _avg_pool3_reference(scores):
    """kernel=3, stride=1, padding=1, count_include_pad=True — by shift-and-add."""
    zero = torch.zeros_like(scores[..., :1])
    padded = torch.cat([zero, scores, zero], dim=-1)
    return (padded[..., :-2] + padded[..., 1:-1] + padded[..., 2:]) / 3.0


def _score_reference(q_raw, keys, values, cos, sin, chunk_size):
    """End-to-end transcription of NonCausalAttnPress.score (:104-122).

    ``q_raw``: pre-RoPE queries [B, H_q, S, d]; ``keys``: cached (already
    rotated) keys [B, H_kv, S, d]; ``cos``/``sin``: [B, S, d].
    """
    B, H_kv, S, d = keys.shape
    H_q = q_raw.shape[1]
    group = H_q // H_kv
    q = q_raw * cos.unsqueeze(1) + _rotate_half_ref(q_raw) * sin.unsqueeze(1)
    k_rep = keys[:, :, None].expand(B, H_kv, group, S, d).reshape(B, H_q, S, d)
    A = _chunked_attn_reference(q, k_rep, chunk_size)
    A_kv = torch.stack(
        [A[:, g * group:(g + 1) * group].mean(dim=1) for g in range(H_kv)], dim=1
    )
    scores = A_kv * values.norm(dim=-1)
    pooled = _avg_pool3_reference(scores)
    return (pooled - pooled.mean()) / pooled.std().clamp_min(1e-6)


# ======================================================================
# Registry
# ======================================================================


class TestRegistry(unittest.TestCase):
    def test_registered_name_resolves(self):
        self.assertIs(get_kv_compressor_class("non_causal_attention"), NonCausalAttnSketch)

    def test_instantiation_with_kwargs(self):
        sketch = get_kv_compressor("non_causal_attention", compression_ratio=0.25, chunk_size=64)
        self.assertIsInstance(sketch, NonCausalAttnSketch)
        self.assertAlmostEqual(sketch.compression_ratio, 0.25)
        self.assertEqual(sketch.chunk_size, 64)

    def test_default_params(self):
        sketch = NonCausalAttnSketch()
        self.assertEqual(sketch.chunk_size, 256)
        self.assertEqual(sketch.compression_ratio, 0.0)


# ======================================================================
# Zero-ratio no-op
# ======================================================================


class TestZeroRatioNoOp(unittest.TestCase):
    def test_compress_returns_identity_and_never_scores(self):
        sketch = NonCausalAttnSketch(compression_ratio=0.0)
        module = _FakeAttnModule()
        keys = torch.randn(1, 2, 6, 4)
        values = torch.randn(1, 2, 6, 4)
        hidden = torch.randn(1, 6, 8)
        with mock.patch.object(
            NonCausalAttnSketch, "score", side_effect=AssertionError("score must not be called")
        ):
            out_k, out_v = sketch.compress(module, hidden, keys, values, None, {})
        self.assertIs(out_k, keys)
        self.assertIs(out_v, values)


# ======================================================================
# Chunked-attention oracle
# ======================================================================


class TestChunkedAttnOracle(unittest.TestCase):
    def test_padded_case_matches_reference(self):
        torch.manual_seed(0)
        q = torch.randn(1, 2, 10, 4)
        k = torch.randn(1, 2, 10, 4)
        out = NonCausalAttnSketch.non_causal_chunked_attn(q, k, 4)
        ref = _chunked_attn_reference(q, k, 4)
        self.assertEqual(out.shape, (1, 2, 10))
        torch.testing.assert_close(out, ref, atol=1e-6, rtol=1e-6)

    def test_pad_mask_is_minus_1e_minus_9_not_minus_1e9(self):
        """The upstream quirk: padded keys are effectively UNMASKED.  A
        "fixed" -1e9 mask gives different last-chunk column sums."""
        torch.manual_seed(0)
        q = torch.randn(1, 2, 10, 4)
        k = torch.randn(1, 2, 10, 4)
        out = NonCausalAttnSketch.non_causal_chunked_attn(q, k, 4)
        fixed = _chunked_attn_reference(q, k, 4, key_pad_fill=-1e9)
        self.assertGreater((out[..., 8:] - fixed[..., 8:]).abs().max().item(), 1e-3)

    def test_exact_multiple_no_masking(self):
        torch.manual_seed(1)
        q = torch.randn(1, 2, 8, 4)
        k = torch.randn(1, 2, 8, 4)
        out = NonCausalAttnSketch.non_causal_chunked_attn(q, k, 4)
        ref = _chunked_attn_reference(q, k, 4)
        torch.testing.assert_close(out, ref, atol=1e-6, rtol=1e-6)
        # Without padding every query row softmaxes to 1 → total mass == S.
        torch.testing.assert_close(
            out.sum(dim=-1), torch.full((1, 2), 8.0), atol=1e-5, rtol=1e-5
        )

    def test_chunk_larger_than_sequence(self):
        torch.manual_seed(2)
        q = torch.randn(1, 2, 5, 4)
        k = torch.randn(1, 2, 5, 4)
        out = NonCausalAttnSketch.non_causal_chunked_attn(q, k, 256)
        self.assertEqual(out.shape, (1, 2, 5))
        self.assertTrue(torch.isfinite(out).all())
        ref = _chunked_attn_reference(q, k, 256)
        torch.testing.assert_close(out, ref, atol=1e-6, rtol=1e-6)

    def test_chunk_size_one_is_all_ones(self):
        torch.manual_seed(3)
        q = torch.randn(1, 2, 6, 4)
        k = torch.randn(1, 2, 6, 4)
        out = NonCausalAttnSketch.non_causal_chunked_attn(q, k, 1)
        torch.testing.assert_close(out, torch.ones(1, 2, 6), atol=0.0, rtol=0.0)


# ======================================================================
# Hand-computed selection pin (edge-pooling inversion)
# ======================================================================


class TestHandComputedSelection(unittest.TestCase):
    """d=1, H=1, S=4, chunk_size=2; q = [1,1,1,1], k = [2,0,0,2], ||v|| = 1.

    Per chunk, softmax([2,0]) = [a, 1-a] with a = e^2/(e^2+1), so the raw
    column sums are A = [2a, 2(1-a), 2(1-a), 2a] — positions 0 and 3 dominate.
    avg_pool1d's implicit edge zeros then INVERT the order:
    pooled = [2/3, (2+2(1-a))/3, (2+2(1-a))/3, 2/3], so top-2 must be {1, 2}.
    """

    def _setup(self):
        module = _FakeAttnModule(hidden_dim=1, num_heads=1, head_dim=1, identity_q=True)
        hidden = torch.ones(1, 4, 1)
        keys = torch.tensor([2.0, 0.0, 0.0, 2.0]).view(1, 1, 4, 1)
        values = torch.ones(1, 1, 4, 1)
        cos, sin = _identity_pos_emb(1, 4, 1)
        kwargs = {"position_embeddings": (cos, sin)}
        return module, hidden, keys, values, kwargs

    def _expected_pooled(self):
        a = math.exp(2.0) / (math.exp(2.0) + 1.0)
        A0, A1 = 2.0 * a, 2.0 * (1.0 - a)
        return torch.tensor(
            [[[(A0 + A1) / 3, (A0 + 2 * A1) / 3, (2 * A1 + A0) / 3, (A1 + A0) / 3]]],
            dtype=torch.float32,
        )

    def test_score_values_pinned(self):
        module, hidden, keys, values, kwargs = self._setup()
        sketch = NonCausalAttnSketch(compression_ratio=0.5, chunk_size=2)
        z = sketch.score(module, hidden, keys, values, None, kwargs)
        pooled = self._expected_pooled()
        expected_z = (pooled - pooled.mean()) / pooled.std().clamp_min(1e-6)
        torch.testing.assert_close(z, expected_z, atol=1e-5, rtol=1e-5)

    def test_edge_inversion_and_topk_indices(self):
        module, hidden, keys, values, kwargs = self._setup()
        sketch = NonCausalAttnSketch(compression_ratio=0.5, chunk_size=2)
        z = sketch.score(module, hidden, keys, values, None, kwargs)
        # Inversion: positions 0/3 carry the attention mass, yet pooling with
        # implicit edge zeros ranks the interior positions higher.
        self.assertLess(z[0, 0, 0].item(), z[0, 0, 1].item())
        self.assertLess(z[0, 0, 3].item(), z[0, 0, 2].item())
        topk = z.topk(2, dim=-1).indices[0, 0].sort().values
        self.assertEqual(topk.tolist(), [1, 2])

    def test_compress_keeps_interior_keys(self):
        module, hidden, keys, values, kwargs = self._setup()
        sketch = NonCausalAttnSketch(compression_ratio=0.5, chunk_size=2)
        out_k, out_v = sketch.compress(module, hidden, keys, values, None, kwargs)
        self.assertEqual(out_k.shape, (1, 1, 2, 1))
        self.assertEqual(out_v.shape, (1, 1, 2, 1))
        # keys at {1, 2} are both 0.0 ({0,3} would give [2,2], {0,1} [0,2]).
        torch.testing.assert_close(out_k.flatten().sort().values, torch.zeros(2))


# ======================================================================
# GQA group mean + rectangular compress
# ======================================================================


class TestGQA(unittest.TestCase):
    def test_score_matches_reference_and_shape(self):
        torch.manual_seed(4)
        B, H_q, H_kv, S, d = 1, 4, 2, 6, 8
        module = _FakeAttnModule(hidden_dim=32, num_heads=H_q, head_dim=d,
                                 num_kv_heads=H_kv, seed=5)
        hidden = torch.randn(B, S, 32)
        keys = torch.randn(B, H_kv, S, d)
        values = torch.randn(B, H_kv, S, d)
        cos, sin = _manual_rope(S, d)
        kwargs = {"position_embeddings": (cos, sin)}

        sketch = NonCausalAttnSketch(compression_ratio=0.5, chunk_size=4)
        z = sketch.score(module, hidden, keys, values, None, kwargs)
        self.assertEqual(z.shape, (B, H_kv, S))

        q_raw = module.q_proj(hidden).view(B, S, H_q, d).transpose(1, 2)
        ref = _score_reference(q_raw, keys, values, cos, sin, chunk_size=4)
        torch.testing.assert_close(z, ref, atol=1e-5, rtol=1e-5)

    def test_compress_is_rectangular_across_kv_heads(self):
        torch.manual_seed(4)
        B, H_q, H_kv, S, d = 1, 4, 2, 6, 8
        module = _FakeAttnModule(hidden_dim=32, num_heads=H_q, head_dim=d,
                                 num_kv_heads=H_kv, seed=5)
        hidden = torch.randn(B, S, 32)
        keys = torch.randn(B, H_kv, S, d)
        values = torch.randn(B, H_kv, S, d)
        cos, sin = _manual_rope(S, d)
        kwargs = {"position_embeddings": (cos, sin)}

        sketch = NonCausalAttnSketch(compression_ratio=0.5, chunk_size=4)
        out_k, out_v = sketch.compress(module, hidden, keys, values, None, kwargs)
        n_kept = int(S * (1 - 0.5))
        self.assertEqual(out_k.shape, (B, H_kv, n_kept, d))
        self.assertEqual(out_v.shape, (B, H_kv, n_kept, d))


# ======================================================================
# RoPE-rotation parity
# ======================================================================


class TestRoPEParity(unittest.TestCase):
    def test_inline_rotation_matches_repo_helper(self):
        from eval_harness.attention_methods._method_base import apply_rotary_pos_emb, build_cos_sin

        torch.manual_seed(6)
        B, H, S, d = 1, 2, 12, 4
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, d // 2, dtype=torch.float32) / (d // 2)))
        cos_b, sin_b = build_cos_sin(
            torch.arange(S), inv_freq, torch.device("cpu"), torch.float32
        )  # [B, 1, S, d]
        q_raw = torch.randn(B, H, S, d)
        manual = q_raw * cos_b.squeeze(1).unsqueeze(1) + _rotate_half_ref(q_raw) * sin_b.squeeze(1).unsqueeze(1)
        helper = apply_rotary_pos_emb(q_raw, cos_b, sin_b)
        torch.testing.assert_close(manual, helper, atol=1e-6, rtol=1e-6)

    def test_score_on_rotated_cache_matches_full_transcription(self):
        from eval_harness.attention_methods._method_base import apply_rotary_pos_emb, build_cos_sin

        torch.manual_seed(7)
        B, H, S, d = 1, 2, 12, 4
        module = _FakeAttnModule(hidden_dim=8, num_heads=H, head_dim=d, seed=8)
        hidden = torch.randn(B, S, 8)
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, d // 2, dtype=torch.float32) / (d // 2)))
        cos_b, sin_b = build_cos_sin(
            torch.arange(S), inv_freq, torch.device("cpu"), torch.float32
        )
        cos, sin = cos_b.squeeze(1), sin_b.squeeze(1)  # [B, S, d]

        k_raw = torch.randn(B, H, S, d)
        keys = apply_rotary_pos_emb(k_raw, cos_b, sin_b)  # cache stores rotated K
        values = torch.randn(B, H, S, d)
        kwargs = {"position_embeddings": (cos, sin)}

        sketch = NonCausalAttnSketch(compression_ratio=0.5, chunk_size=8)
        z = sketch.score(module, hidden, keys, values, None, kwargs)

        q_raw = module.q_proj(hidden).view(B, S, H, d).transpose(1, 2)
        ref = _score_reference(q_raw, keys, values, cos, sin, chunk_size=8)
        torch.testing.assert_close(z, ref, atol=1e-5, rtol=1e-5)


# ======================================================================
# Degenerate chunk_size=1 selection (ordering decided purely by ||v||)
# ======================================================================


class TestChunkSizeOneSelection(unittest.TestCase):
    def test_selection_follows_pooled_value_norms(self):
        module = _FakeAttnModule(hidden_dim=1, num_heads=1, head_dim=1, identity_q=True)
        S = 6
        hidden = torch.ones(1, S, 1)
        keys = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0]).view(1, 1, S, 1)
        values = torch.tensor([1.0, 5.0, 1.0, 9.0, 1.0, 1.0]).view(1, 1, S, 1)
        cos, sin = _identity_pos_emb(1, S, 1)
        kwargs = {"position_embeddings": (cos, sin)}

        sketch = NonCausalAttnSketch(compression_ratio=0.5, chunk_size=1)
        z = sketch.score(module, hidden, keys, values, None, kwargs)
        # A == 1 everywhere → pooled = [2, 7/3, 5, 11/3, 11/3, 2/3] → top-3 {2,3,4}.
        topk = z.topk(3, dim=-1).indices[0, 0].sort().values
        self.assertEqual(topk.tolist(), [2, 3, 4])

        out_k, _ = sketch.compress(module, hidden, keys, values, None, kwargs)
        torch.testing.assert_close(
            out_k.flatten().sort().values, torch.tensor([30.0, 40.0, 50.0])
        )


# ======================================================================
# Prefill-only assert + decode gate
# ======================================================================


class TestPrefillGate(unittest.TestCase):
    def test_score_raises_when_cache_longer_than_queries(self):
        module = _FakeAttnModule()
        sketch = NonCausalAttnSketch(compression_ratio=0.5)
        hidden = torch.randn(1, 5, 8)
        keys = torch.randn(1, 2, 8, 4)
        values = torch.randn(1, 2, 8, 4)
        cos, sin = _identity_pos_emb(1, 5, 4)
        with self.assertRaisesRegex(AssertionError, "only supports prefill"):
            sketch.score(module, hidden, keys, values, None,
                         {"position_embeddings": (cos, sin)})

    def test_forward_hook_noop_on_decoding_step(self):
        module = _FakeAttnModule(hidden_dim=1, num_heads=1, head_dim=1, identity_q=True)
        sketch = NonCausalAttnSketch(compression_ratio=0.5, chunk_size=2)
        keys = torch.randn(1, 1, 8, 1)
        values = torch.randn(1, 1, 8, 1)
        cache = _FakeCache(keys, values)
        kwargs = {
            "hidden_states": torch.ones(1, 1, 1),
            "past_key_values": cache,
            "cache_position": torch.tensor([8]),
        }
        output = [torch.zeros(1), None]
        result = sketch.forward_hook(module, [], kwargs, output)
        self.assertIs(result, output)
        self.assertIs(cache.layers[0].keys, keys)
        self.assertIs(cache.layers[0].values, values)

    def test_forward_hook_compresses_on_prefill(self):
        module = _FakeAttnModule(hidden_dim=1, num_heads=1, head_dim=1, identity_q=True)
        sketch = NonCausalAttnSketch(compression_ratio=0.5, chunk_size=2)
        keys = torch.tensor([2.0, 0.0, 0.0, 2.0]).view(1, 1, 4, 1)
        values = torch.ones(1, 1, 4, 1)
        cache = _FakeCache(keys, values)
        cos, sin = _identity_pos_emb(1, 4, 1)
        kwargs = {
            "hidden_states": torch.ones(1, 4, 1),
            "past_key_values": cache,
            "cache_position": torch.arange(4),
            "position_embeddings": (cos, sin),
        }
        sketch.forward_hook(module, [], kwargs, [torch.zeros(1), None])
        self.assertEqual(cache.layers[0].keys.shape[2], 2)
        self.assertEqual(cache.layers[0].values.shape[2], 2)


# ======================================================================
# dtype + degenerate-score edge cases
# ======================================================================


class TestDtypeAndEdgeCases(unittest.TestCase):
    def test_bf16_inputs_fp32_scores_bf16_cache(self):
        """bf16 model dtype end-to-end: the float32 softmax upcast (:91) makes
        the returned scores float32 while compress keeps the cache bf16.

        (cos, sin) are bf16 like a real bf16 model's position_embeddings —
        fp32 trig with bf16 keys raises a mixed-dtype matmul error in the
        kvpress original just the same.
        """
        torch.manual_seed(9)
        B, H, S, d = 1, 2, 8, 4
        module = _FakeAttnModule(hidden_dim=8, num_heads=H, head_dim=d, seed=9).to(torch.bfloat16)
        hidden = torch.randn(B, S, 8, dtype=torch.bfloat16)
        keys = torch.randn(B, H, S, d, dtype=torch.bfloat16)
        values = torch.randn(B, H, S, d, dtype=torch.bfloat16)
        cos, sin = _identity_pos_emb(B, S, d)
        cos, sin = cos.to(torch.bfloat16), sin.to(torch.bfloat16)

        sketch = NonCausalAttnSketch(compression_ratio=0.5, chunk_size=4)
        kwargs = {"position_embeddings": (cos, sin)}
        z = sketch.score(module, hidden, keys, values, None, kwargs)
        self.assertEqual(z.dtype, torch.float32)
        self.assertTrue(torch.isfinite(z).all())

        out_k, out_v = sketch.compress(module, hidden, keys, values, None, kwargs)
        self.assertEqual(out_k.dtype, torch.bfloat16)
        self.assertEqual(out_v.dtype, torch.bfloat16)
        self.assertEqual(out_k.shape, (B, H, int(S * 0.5), d))

    def test_constant_scores_std_clamp_no_nan(self):
        """All-zero values → all-zero scores → std=0 path: clamp_min(1e-6)
        yields exact zeros (no NaN), and topk still returns n_kept indices."""
        module = _FakeAttnModule(hidden_dim=8, num_heads=2, head_dim=4, seed=10)
        B, H, S, d = 1, 2, 6, 4
        hidden = torch.randn(B, S, 8)
        keys = torch.randn(B, H, S, d)
        values = torch.zeros(B, H, S, d)
        cos, sin = _identity_pos_emb(B, S, d)
        kwargs = {"position_embeddings": (cos, sin)}

        sketch = NonCausalAttnSketch(compression_ratio=0.5, chunk_size=4)
        z = sketch.score(module, hidden, keys, values, None, kwargs)
        self.assertTrue(torch.equal(z, torch.zeros(B, H, S)))
        out_k, _ = sketch.compress(module, hidden, keys, values, None, kwargs)
        self.assertEqual(out_k.shape, (B, H, 3, d))


# ======================================================================
# Pre-RoPE query extraction (duck-typed Phi3 / qk-norm paths)
# ======================================================================


class TestPreropeQueryStates(unittest.TestCase):
    def test_phi3_fused_qkv_slice_matches_q_proj(self):
        B, H_q, H_kv, S, d = 1, 2, 1, 6, 4
        hidden_dim = 8
        phi3 = _FakePhi3AttnModule(hidden_dim, H_q, d, H_kv, seed=11)
        llama = _FakeAttnModule(hidden_dim=hidden_dim, num_heads=H_q, head_dim=d,
                                num_kv_heads=H_kv, seed=11)
        with torch.no_grad():
            llama.q_proj.weight.copy_(phi3.qkv_proj.weight[: H_q * d, :])

        torch.manual_seed(12)
        hidden = torch.randn(B, S, hidden_dim)
        keys = torch.randn(B, H_kv, S, d)
        values = torch.randn(B, H_kv, S, d)
        cos, sin = _manual_rope(S, d)
        kwargs = {"position_embeddings": (cos, sin)}

        sketch = NonCausalAttnSketch(compression_ratio=0.5, chunk_size=4)
        z_phi3 = sketch.score(phi3, hidden, keys, values, None, kwargs)
        z_llama = sketch.score(llama, hidden, keys, values, None, kwargs)
        torch.testing.assert_close(z_phi3, z_llama, atol=1e-6, rtol=1e-6)

    def test_q_norm_applied_when_present(self):
        B, H, S, d = 1, 2, 6, 4
        module = _FakeAttnModule(hidden_dim=8, num_heads=H, head_dim=d, seed=13)
        module.q_norm = _DoublingNorm()
        torch.manual_seed(14)
        hidden = torch.randn(B, S, 8)
        keys = torch.randn(B, H, S, d)
        values = torch.randn(B, H, S, d)
        cos, sin = _manual_rope(S, d)
        kwargs = {"position_embeddings": (cos, sin)}

        sketch = NonCausalAttnSketch(compression_ratio=0.5, chunk_size=4)
        z = sketch.score(module, hidden, keys, values, None, kwargs)
        q_raw = module.q_proj(hidden).view(B, S, H, d).transpose(1, 2)
        ref = _score_reference(2.0 * q_raw, keys, values, cos, sin, chunk_size=4)
        torch.testing.assert_close(z, ref, atol=1e-5, rtol=1e-5)


# ======================================================================
# Integration: real pipeline prefill + rectangular decode
# ======================================================================


class _StubTokenizer:
    model_max_length = 8192

    def decode(self, ids, skip_special_tokens=True):  # noqa: ARG002
        return "x" * len(ids)


def _build_model(num_hidden_layers: int = 2) -> LlamaForCausalLM:
    cfg = LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=256,
        max_position_embeddings=8192,
        rope_theta=10000.0,
        attn_implementation="eager",
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg).eval()
    if model.generation_config.eos_token_id is None:
        model.generation_config.eos_token_id = 2
    return model


class TestPipelineIntegration(unittest.TestCase):
    def test_prefill_compression_uniform_layers_then_decode(self):
        model = _build_model(num_hidden_layers=2)
        pipe = object.__new__(ResearchGenerationPipeline)
        pipe.model = model
        pipe.tokenizer = _StubTokenizer()

        sketch = NonCausalAttnSketch(compression_ratio=0.5, chunk_size=8)
        cache_adapter = create_cache_adapter(model)
        cache = cache_adapter.initialize_cache(None)

        torch.manual_seed(0)
        inputs = {
            "context_ids": torch.randint(0, 256, (1, 30)),
            "questions_ids": [torch.randint(0, 256, (1, 4))],
        }
        answers = pipe._forward(
            inputs,
            max_new_tokens=5,
            kv_compressor=sketch,
            attention_method=None,
            cache=cache,
            cache_adapter=cache_adapter,
        )
        self.assertEqual(len(answers), 1)
        self.assertIsInstance(answers[0], str)

        lengths = [int(layer.keys.shape[2]) for layer in cache.layers]
        self.assertEqual(lengths, [15, 15])  # int(30 * (1 - 0.5)), all layers


if __name__ == "__main__":
    unittest.main()
