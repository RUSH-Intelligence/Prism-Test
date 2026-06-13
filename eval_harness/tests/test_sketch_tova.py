"""Tests for TOVASketch (port of kvpress TOVAPress).

The kvpress math (TOVAPress.score + the window-1 path of
SnapKVPress.compute_window_attention, including its no-op triu mask) is
re-transcribed locally as a reference oracle; no kvpress import.
"""

import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch import nn

from eval_harness.sketch.sketches.tova_sketch import TOVASketch


class _FakeTOVAAttn(nn.Module):
    def __init__(self, hidden_dim=24, num_heads=4, head_dim=6, num_kv_heads=2, seed=0):
        super().__init__()
        self.head_dim = head_dim
        self.layer_idx = 0
        self.config = SimpleNamespace(
            num_attention_heads=num_heads, num_key_value_heads=num_kv_heads
        )
        self.q_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        torch.manual_seed(seed)
        with torch.no_grad():
            self.q_proj.weight.normal_()


class _FakeFusedTOVAAttn(nn.Module):
    """Phi3-style module exposing only a fused qkv_proj."""

    def __init__(self, hidden_dim=24, num_heads=4, head_dim=6, num_kv_heads=2, seed=0):
        super().__init__()
        self.head_dim = head_dim
        self.layer_idx = 0
        self.config = SimpleNamespace(
            num_attention_heads=num_heads, num_key_value_heads=num_kv_heads
        )
        out_dim = (num_heads + 2 * num_kv_heads) * head_dim
        self.qkv_proj = nn.Linear(hidden_dim, out_dim, bias=False)
        torch.manual_seed(seed)
        with torch.no_grad():
            self.qkv_proj.weight.normal_()


class _DoubleQNorm(nn.Module):
    def forward(self, x):
        return x * 2.0


class _FakeCacheLayer:
    def __init__(self, keys, values):
        self.keys = keys
        self.values = values


class _FakeCache:
    def __init__(self, layers):
        self.layers = layers


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


def _rope_pos_emb(positions, dim, base=10000.0):
    """Real (cos, sin) of shape [1, S, dim] for the given 1-D positions."""
    half = dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))
    freqs = torch.einsum("s,d->sd", positions.to(torch.float32), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().unsqueeze(0), emb.sin().unsqueeze(0)


def _kvpress_tova_reference(module, hidden_states, keys, cos, sin):
    """Verbatim transcription of kvpress TOVAPress.score's fallback path.

    Mirrors SnapKVPress.compute_window_attention with window_size=1
    (tova_press.py:48-59, snapkv_press.py:42-69), keeping the triu mask the
    production port drops as a no-op.
    """
    bsz, _, k_len, _ = keys.shape
    num_heads = module.config.num_attention_heads
    head_dim = module.head_dim
    num_key_value_groups = num_heads // module.config.num_key_value_heads

    q = module.q_proj(hidden_states[:, -1:])
    q = q.view(bsz, 1, num_heads, head_dim).transpose(1, 2)
    q_norm = getattr(module, "q_norm", None)
    if q_norm is not None:
        q = q_norm(q)

    c, s = cos[:, -1:], sin[:, -1:]
    q = (q * c.unsqueeze(1)) + (_ref_rotate_half(q) * s.unsqueeze(1))

    k = _ref_repeat_kv(keys, num_key_value_groups)
    w = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(head_dim)
    mask = torch.ones_like(w) * float("-inf")
    mask = torch.triu(mask, diagonal=k_len - 1 + 1)
    w = w + mask
    w = nn.functional.softmax(w, dim=-1, dtype=torch.float32).to(q.dtype)
    w = w[..., :-1]

    scores = w.mean(1)
    scores = scores.repeat(1, keys.shape[1], 1)
    scores = F.pad(scores, (0, 1), value=scores.max().item())
    return scores


def _position_keys(B, H_kv, S, D, offset=0.0):
    """Keys whose row s is (s + offset) * ones(D), so kept positions are readable."""
    pos = torch.arange(S, dtype=torch.float32) + offset
    return pos.view(1, 1, S, 1).expand(B, H_kv, S, D).contiguous()


class TestTOVAScore(unittest.TestCase):
    def test_zero_ratio_noop(self):
        sketch = TOVASketch(compression_ratio=0.0)
        module = _FakeTOVAAttn()
        keys = torch.randn(1, 2, 5, 6)
        values = torch.randn(1, 2, 5, 6)
        with patch.object(TOVASketch, "score", side_effect=AssertionError("score called")):
            out_k, out_v = sketch.compress(module, torch.randn(1, 5, 24), keys, values, None, {})
        self.assertIs(out_k, keys)
        self.assertIs(out_v, values)

    def test_eager_attentions_value_pinned(self):
        sketch = TOVASketch(compression_ratio=0.5)
        module = _FakeTOVAAttn(hidden_dim=8, num_heads=2, head_dim=4, num_kv_heads=1)
        B, H_q, S = 1, 2, 4

        attentions = torch.zeros(B, H_q, S, S)
        attentions[0, 0, 3] = torch.tensor([0.1, 0.2, 0.3, 0.4])
        attentions[0, 1, 3] = torch.tensor([0.1, 0.6, 0.2, 0.1])

        keys = _position_keys(B, 1, S, 4)
        values = _position_keys(B, 1, S, 4, offset=10.0)

        scores = sketch.score(module, torch.randn(B, S, 8), keys, values, attentions, {})
        expected = torch.tensor([[[0.10, 0.40, 0.25, 0.40]]])
        torch.testing.assert_close(scores, expected)

        out_k, out_v = sketch.compress(
            module, torch.randn(B, S, 8), keys, values, attentions, {}
        )
        self.assertEqual(out_k.shape, (B, 1, 2, 4))
        self.assertEqual(sorted(out_k[0, 0, :, 0].tolist()), [1.0, 3.0])
        self.assertEqual(sorted(out_v[0, 0, :, 0].tolist()), [11.0, 13.0])

    def test_fallback_matches_kvpress_reference(self):
        torch.manual_seed(7)
        B, S, hidden, H_q, H_kv, D = 1, 7, 24, 4, 2, 6
        module = _FakeTOVAAttn(hidden, H_q, D, H_kv, seed=1)
        hidden_states = torch.randn(B, S, hidden)
        keys = torch.randn(B, H_kv, S, D)
        values = torch.randn(B, H_kv, S, D)
        cos = torch.rand(B, S, D)
        sin = torch.rand(B, S, D)

        scores = TOVASketch(compression_ratio=0.5).score(
            module, hidden_states, keys, values, None, {"position_embeddings": (cos, sin)}
        )
        expected = _kvpress_tova_reference(module, hidden_states, keys, cos, sin)
        self.assertEqual(scores.shape, (B, H_kv, S))
        torch.testing.assert_close(scores, expected, rtol=0.0, atol=1e-6)

    def test_qnorm_duck_typing(self):
        torch.manual_seed(11)
        B, S, hidden, H_q, H_kv, D = 1, 6, 24, 4, 2, 6
        plain = _FakeTOVAAttn(hidden, H_q, D, H_kv, seed=2)
        normed = _FakeTOVAAttn(hidden, H_q, D, H_kv, seed=2)
        normed.q_norm = _DoubleQNorm()
        hidden_states = torch.randn(B, S, hidden)
        keys = torch.randn(B, H_kv, S, D)
        cos, sin = _rope_pos_emb(torch.arange(S), D)
        kwargs = {"position_embeddings": (cos, sin)}
        sketch = TOVASketch(compression_ratio=0.5)

        scores_normed = sketch.score(normed, hidden_states, keys, None, None, kwargs)
        expected = _kvpress_tova_reference(normed, hidden_states, keys, cos, sin)
        torch.testing.assert_close(scores_normed, expected, rtol=0.0, atol=1e-6)

        scores_plain = sketch.score(plain, hidden_states, keys, None, None, kwargs)
        self.assertFalse(torch.allclose(scores_normed, scores_plain))

    def test_phi3_style_fused_qkv_proj(self):
        torch.manual_seed(13)
        B, S, hidden, H_q, H_kv, D = 1, 6, 24, 4, 2, 6
        fused = _FakeFusedTOVAAttn(hidden, H_q, D, H_kv, seed=3)
        split = _FakeTOVAAttn(hidden, H_q, D, H_kv, seed=3)
        with torch.no_grad():
            split.q_proj.weight.copy_(fused.qkv_proj.weight[: H_q * D])
        hidden_states = torch.randn(B, S, hidden)
        keys = torch.randn(B, H_kv, S, D)
        cos, sin = _rope_pos_emb(torch.arange(S), D)
        kwargs = {"position_embeddings": (cos, sin)}
        sketch = TOVASketch(compression_ratio=0.5)

        scores_fused = sketch.score(fused, hidden_states, keys, None, None, kwargs)
        scores_split = sketch.score(split, hidden_states, keys, None, None, kwargs)
        torch.testing.assert_close(scores_fused, scores_split, rtol=0.0, atol=1e-6)

    def test_gqa_head_uniform_scores_and_selection(self):
        torch.manual_seed(17)
        B, S, hidden, H_q, H_kv, D = 1, 8, 24, 4, 2, 6
        module = _FakeTOVAAttn(hidden, H_q, D, H_kv, seed=4)
        hidden_states = torch.randn(B, S, hidden)
        keys = _position_keys(B, H_kv, S, D, offset=1.0)
        values = torch.randn(B, H_kv, S, D)
        cos, sin = _rope_pos_emb(torch.arange(S), D)
        kwargs = {"position_embeddings": (cos, sin)}
        sketch = TOVASketch(compression_ratio=0.5)

        scores = sketch.score(module, hidden_states, keys, values, None, kwargs)
        self.assertEqual(scores.shape, (B, H_kv, S))
        self.assertTrue(torch.equal(scores[:, 0, :], scores[:, 1, :]))

        out_k, out_v = sketch.compress(module, hidden_states, keys, values, None, kwargs)
        self.assertEqual(out_k.shape, (B, H_kv, 4, D))
        self.assertEqual(out_v.shape, (B, H_kv, 4, D))
        kept_h0 = set((out_k[0, 0, :, 0] - 1.0).tolist())
        kept_h1 = set((out_k[0, 1, :, 0] - 1.0).tolist())
        self.assertEqual(kept_h0, kept_h1)
        self.assertEqual(len(kept_h0), 4)

    def test_path_equivalence_eager_vs_recompute(self):
        torch.manual_seed(19)
        B, S, hidden, H_q, H_kv, D = 1, 6, 24, 4, 2, 6
        module = _FakeTOVAAttn(hidden, H_q, D, H_kv, seed=5)
        hidden_states = torch.randn(B, S, hidden)
        cos, sin = _rope_pos_emb(torch.arange(S), D)

        raw_k = torch.randn(B, H_kv, S, D)
        k_rot = (raw_k * cos.unsqueeze(1)) + (_ref_rotate_half(raw_k) * sin.unsqueeze(1))

        q_all = module.q_proj(hidden_states).view(B, S, H_q, D).transpose(1, 2)
        q_rot = (q_all * cos.unsqueeze(1)) + (_ref_rotate_half(q_all) * sin.unsqueeze(1))
        logits = torch.matmul(q_rot, _ref_repeat_kv(k_rot, H_q // H_kv).transpose(2, 3))
        logits = logits / math.sqrt(D)
        causal = torch.triu(torch.full((S, S), float("-inf")), diagonal=1)
        full_attn = nn.functional.softmax(logits + causal, dim=-1, dtype=torch.float32)
        full_attn = full_attn.to(q_rot.dtype)

        values = torch.randn(B, H_kv, S, D)
        sketch = TOVASketch(compression_ratio=0.5)
        scores_eager = sketch.score(module, hidden_states, k_rot, values, full_attn, {})
        scores_recompute = sketch.score(
            module, hidden_states, k_rot, values, None,
            {"position_embeddings": (cos, sin)},
        )
        torch.testing.assert_close(scores_eager, scores_recompute, rtol=0.0, atol=1e-5)

    def test_last_token_always_kept(self):
        torch.manual_seed(23)
        B, H_q, H_kv, S, D = 1, 2, 1, 64, 4
        module = _FakeTOVAAttn(hidden_dim=8, num_heads=H_q, head_dim=D, num_kv_heads=H_kv)
        attentions = torch.rand(B, H_q, S, S)
        keys = _position_keys(B, H_kv, S, D)
        values = torch.randn(B, H_kv, S, D)
        for ratio in (0.25, 0.5, 0.75):
            sketch = TOVASketch(compression_ratio=ratio)
            out_k, _ = sketch.compress(
                module, torch.randn(B, S, 8), keys, values, attentions, {}
            )
            kept = set(out_k[0, 0, :, 0].tolist())
            self.assertEqual(len(kept), int(S * (1 - ratio)))
            self.assertIn(float(S - 1), kept)

    def test_compression_count_invariant_across_layers(self):
        B, hidden, H_q, H_kv, D = 1, 24, 4, 2, 6
        for S, ratio, expected_kept in ((10, 0.3, 7), (5, 0.5, 2)):
            sketch = TOVASketch(compression_ratio=ratio)
            cos, sin = _rope_pos_emb(torch.arange(S), D)
            kwargs = {"position_embeddings": (cos, sin)}
            for seed in (1, 2):
                torch.manual_seed(seed)
                module = _FakeTOVAAttn(hidden, H_q, D, H_kv, seed=seed)
                out_k, out_v = sketch.compress(
                    module,
                    torch.randn(B, S, hidden),
                    torch.randn(B, H_kv, S, D),
                    torch.randn(B, H_kv, S, D),
                    None,
                    kwargs,
                )
                self.assertEqual(out_k.shape, (B, H_kv, expected_kept, D))
                self.assertEqual(out_v.shape, (B, H_kv, expected_kept, D))

    def test_bf16_dtype_robustness(self):
        torch.manual_seed(29)
        B, S, hidden, H_q, H_kv, D = 1, 8, 24, 4, 2, 6
        module = _FakeTOVAAttn(hidden, H_q, D, H_kv, seed=6).to(torch.bfloat16)
        hidden_states = torch.randn(B, S, hidden, dtype=torch.bfloat16)
        keys = torch.randn(B, H_kv, S, D, dtype=torch.bfloat16)
        values = torch.randn(B, H_kv, S, D, dtype=torch.bfloat16)
        cos, sin = _rope_pos_emb(torch.arange(S), D)
        kwargs = {"position_embeddings": (cos.to(torch.bfloat16), sin.to(torch.bfloat16))}
        sketch = TOVASketch(compression_ratio=0.5)

        scores = sketch.score(module, hidden_states, keys, values, None, kwargs)
        self.assertEqual(scores.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(scores).all())

        out_k, out_v = sketch.compress(module, hidden_states, keys, values, None, kwargs)
        self.assertEqual(out_k.shape, (B, H_kv, 4, D))
        self.assertEqual(out_k.dtype, torch.bfloat16)
        self.assertEqual(out_v.dtype, torch.bfloat16)

    def test_single_token_score_raises(self):
        B, hidden, H_q, H_kv, D = 1, 24, 4, 2, 6
        module = _FakeTOVAAttn(hidden, H_q, D, H_kv)
        sketch = TOVASketch(compression_ratio=0.5)
        hidden_states = torch.randn(B, 1, hidden)
        keys = torch.randn(B, H_kv, 1, D)
        values = torch.randn(B, H_kv, 1, D)

        with self.assertRaises(RuntimeError):
            sketch.score(module, hidden_states, keys, values, torch.rand(B, H_q, 1, 1), {})

        cos, sin = _rope_pos_emb(torch.arange(1), D)
        with self.assertRaises(RuntimeError):
            sketch.score(
                module, hidden_states, keys, values, None,
                {"position_embeddings": (cos, sin)},
            )


class TestTOVAForwardHook(unittest.TestCase):
    def _setup(self, S, hidden=24, H_q=4, H_kv=2, D=6, ratio=0.5, seed=31):
        torch.manual_seed(seed)
        module = _FakeTOVAAttn(hidden, H_q, D, H_kv, seed=seed)
        keys = torch.randn(1, H_kv, S, D)
        values = torch.randn(1, H_kv, S, D)
        cache = _FakeCache([_FakeCacheLayer(keys, values)])
        sketch = TOVASketch(compression_ratio=ratio)
        return sketch, module, cache, keys, values

    def test_prefill_hook_compresses_cache(self):
        S = 8
        sketch, module, cache, _, _ = self._setup(S)
        cos, sin = _rope_pos_emb(torch.arange(S), 6)
        kwargs = {
            "hidden_states": torch.randn(1, S, 24),
            "past_key_values": cache,
            "cache_position": torch.arange(S),
            "position_embeddings": (cos, sin),
        }
        output = (torch.randn(1, S, 24), None)
        result = sketch.forward_hook(module, [], kwargs, output)
        self.assertIs(result, output)
        self.assertEqual(cache.layers[0].keys.shape, (1, 2, 4, 6))
        self.assertEqual(cache.layers[0].values.shape, (1, 2, 4, 6))

    def test_decode_step_noop(self):
        S = 8
        sketch, module, cache, keys, values = self._setup(S)
        kwargs = {
            "hidden_states": torch.randn(1, 1, 24),
            "past_key_values": cache,
            "cache_position": torch.tensor([S + 3]),
        }
        output = (torch.randn(1, 1, 24), None)
        result = sketch.forward_hook(module, [], kwargs, output)
        self.assertIs(result, output)
        self.assertIs(cache.layers[0].keys, keys)
        self.assertIs(cache.layers[0].values, values)

    def test_single_token_without_cache_metadata_skips_compress(self):
        sketch, module, cache, keys, values = self._setup(S=1)
        kwargs = {
            "hidden_states": torch.randn(1, 1, 24),
            "past_key_values": cache,
        }
        output = (torch.randn(1, 1, 24), None)
        result = sketch.forward_hook(module, [], kwargs, output)
        self.assertIs(result, output)
        self.assertIs(cache.layers[0].keys, keys)
        self.assertIs(cache.layers[0].values, values)


class TestTOVARegistry(unittest.TestCase):
    def test_registered_as_tova(self):
        from eval_harness.sketch.sketches.registry import (
            available_sketches,
            get_sketch,
            get_sketch_class,
        )

        self.assertIn("tova", available_sketches())
        self.assertIs(get_sketch_class("tova"), TOVASketch)
        sketch = get_sketch("tova", compression_ratio=0.25)
        self.assertIsInstance(sketch, TOVASketch)
        self.assertAlmostEqual(sketch.compression_ratio, 0.25)


if __name__ == "__main__":
    unittest.main()
