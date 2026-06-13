"""Tests for SnapKVSketch (port of kvpress SnapKVPress).

The kvpress math (SnapKVPress.score + SnapKVPress.compute_window_attention,
kvpress/presses/snapkv_press.py) is re-transcribed locally as a reference
oracle; no kvpress import.
"""

import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch import nn

from eval_harness.sketch.sketches.snapkv_sketch import SnapKVSketch


class _FakeSnapKVAttn(nn.Module):
    def __init__(self, hidden_dim=24, num_heads=4, head_dim=6, num_kv_heads=2, seed=0, layer_idx=0):
        super().__init__()
        self.head_dim = head_dim
        self.layer_idx = layer_idx
        self.config = SimpleNamespace(
            num_attention_heads=num_heads, num_key_value_heads=num_kv_heads
        )
        self.q_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        torch.manual_seed(seed)
        with torch.no_grad():
            self.q_proj.weight.normal_()


class _FakeFusedSnapKVAttn(nn.Module):
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


def _identity_pos_emb(B, S, D):
    return torch.ones(B, S, D), torch.zeros(B, S, D)


def _position_values(B, H_kv, S, D, offset=0.0):
    """Values whose row s is (s + offset) * ones(D), so kept positions are readable."""
    pos = torch.arange(S, dtype=torch.float32) + offset
    return pos.view(1, 1, S, 1).expand(B, H_kv, S, D).contiguous()


def _kvpress_window_attention(module, hidden_states, keys, window_size, cos, sin):
    """Verbatim transcription of kvpress SnapKVPress.compute_window_attention
    (snapkv_press.py:41-69), with get_prerope_query_states duck-typed."""
    bsz, _, k_len, _ = keys.shape
    num_heads = module.config.num_attention_heads
    head_dim = module.head_dim
    num_key_value_groups = num_heads // module.config.num_key_value_heads

    q = hidden_states[:, -window_size:]
    qkv_proj = getattr(module, "qkv_proj", None)
    if qkv_proj is not None:
        q = qkv_proj(q)[..., : num_heads * head_dim]
    else:
        q = module.q_proj(q)
    q = q.view(bsz, window_size, num_heads, head_dim).transpose(1, 2)
    q_norm = getattr(module, "q_norm", None)
    if q_norm is not None:
        q = q_norm(q)

    c, s = cos[:, -window_size:], sin[:, -window_size:]
    q = (q * c.unsqueeze(1)) + (_ref_rotate_half(q) * s.unsqueeze(1))

    k = _ref_repeat_kv(keys, num_key_value_groups)
    w = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(head_dim)
    mask = torch.ones_like(w) * float("-inf")
    mask = torch.triu(mask, diagonal=k_len - window_size + 1)
    w = w + mask
    w = nn.functional.softmax(w, dim=-1, dtype=torch.float32).to(q.dtype)
    return w[..., :-window_size]


def _kvpress_snapkv_reference(
    module, hidden_states, keys, window_size, kernel_size, cos, sin, attentions=None
):
    """Verbatim transcription of kvpress SnapKVPress.score (snapkv_press.py:71-105)."""
    bsz, H_kv, k_len, _ = keys.shape
    G = module.config.num_attention_heads // H_kv
    if attentions is not None:
        w = attentions[..., -window_size:, :-window_size]
    else:
        w = _kvpress_window_attention(module, hidden_states, keys, window_size, cos, sin)
    s = w.mean(dim=-2)
    s = F.avg_pool1d(s, kernel_size=kernel_size, padding=kernel_size // 2, stride=1)
    s = s.view(bsz, H_kv, G, k_len - window_size).mean(2)
    s = F.pad(s, (0, window_size), value=s.max().item())
    return s


class TestSnapKVScore(unittest.TestCase):
    def test_zero_ratio_noop(self):
        sketch = SnapKVSketch(compression_ratio=0.0)
        self.assertEqual(sketch.window_size, 64)
        module = _FakeSnapKVAttn()
        keys = torch.randn(1, 2, 4, 6)
        values = torch.randn(1, 2, 4, 6)
        with patch.object(SnapKVSketch, "score", side_effect=AssertionError("score called")):
            out_k, out_v = sketch.compress(module, torch.randn(1, 4, 24), keys, values, None, {})
        self.assertIs(out_k, keys)
        self.assertIs(out_v, values)

    def test_window_force_keep(self):
        module = _FakeSnapKVAttn(hidden_dim=8, num_heads=1, head_dim=4, num_kv_heads=1, seed=0)
        torch.manual_seed(0)
        B, S, W, D = 1, 8, 3, 4
        hidden = torch.randn(B, S, 8)
        keys = torch.randn(B, 1, S, D)
        values = _position_values(B, 1, S, D)
        cos, sin = _rope_pos_emb(torch.arange(S), D)
        kwargs = {"position_embeddings": (cos, sin)}
        sketch = SnapKVSketch(compression_ratio=0.5, window_size=W, kernel_size=3)

        scores = sketch.score(module, hidden, keys, values, None, kwargs)
        self.assertEqual(scores.shape, (1, 1, 8))
        self.assertTrue(torch.all(scores[..., -W:] == scores.max()))

        out_k, out_v = sketch.compress(module, hidden, keys, values, None, kwargs)
        self.assertEqual(out_k.shape, (1, 1, 4, D))
        kept = set(out_v[0, 0, :, 0].tolist())
        self.assertTrue({5.0, 6.0, 7.0} <= kept)
        self.assertEqual(kept, {2.0, 5.0, 6.0, 7.0})

    def test_reference_oracle_gqa_value_pinned(self):
        module = _FakeSnapKVAttn(hidden_dim=16, num_heads=4, head_dim=4, num_kv_heads=2, seed=0)
        torch.manual_seed(0)
        B, S, W, D, H_kv = 1, 8, 3, 4, 2
        hidden = torch.randn(B, S, 16)
        keys = torch.randn(B, H_kv, S, D)
        values = _position_values(B, H_kv, S, D)
        cos, sin = _rope_pos_emb(torch.arange(S), D)
        kwargs = {"position_embeddings": (cos, sin)}
        sketch = SnapKVSketch(compression_ratio=0.5, window_size=W, kernel_size=3)

        scores = sketch.score(module, hidden, keys, values, None, kwargs)
        ref = _kvpress_snapkv_reference(module, hidden, keys, W, 3, cos, sin)
        self.assertEqual(scores.shape, (B, H_kv, S))
        torch.testing.assert_close(scores, ref, rtol=0.0, atol=0.0)

        out_k, out_v = sketch.compress(module, hidden, keys, values, None, kwargs)
        self.assertEqual(out_k.shape, (B, H_kv, 4, D))
        kept_h0 = sorted(out_v[0, 0, :, 0].tolist())
        kept_h1 = sorted(out_v[0, 1, :, 0].tolist())
        self.assertEqual(kept_h0, [1.0, 5.0, 6.0, 7.0])
        self.assertEqual(kept_h1, [3.0, 5.0, 6.0, 7.0])
        for h, kept in ((0, kept_h0), (1, kept_h1)):
            ctx_argmax = float(ref[0, h, : S - W].argmax().item())
            self.assertEqual(set(kept), {5.0, 6.0, 7.0, ctx_argmax})

    def test_avg_pool1d_edge_semantics(self):
        x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        out = F.avg_pool1d(x, kernel_size=3, padding=1, stride=1)
        expected = torch.tensor([[[1.0, 2.0, 3.0, 7.0 / 3.0]]])
        torch.testing.assert_close(out, expected)

    def test_gqa_group_mean(self):
        module = _FakeSnapKVAttn(hidden_dim=16, num_heads=4, head_dim=4, num_kv_heads=2, seed=2)
        with torch.no_grad():
            module.q_proj.weight[4:8] = module.q_proj.weight[0:4]
        torch.manual_seed(2)
        B, S, W, D, H_q, H_kv = 1, 10, 3, 4, 4, 2
        hidden = torch.randn(B, S, 16)
        keys = torch.randn(B, H_kv, S, D)
        values = torch.randn(B, H_kv, S, D)
        cos, sin = _rope_pos_emb(torch.arange(S), D)
        kwargs = {"position_embeddings": (cos, sin)}
        sketch = SnapKVSketch(compression_ratio=0.5, window_size=W, kernel_size=3)

        scores = sketch.score(module, hidden, keys, values, None, kwargs)
        self.assertEqual(scores.shape, (B, H_kv, S))

        q0 = module.q_proj(hidden[:, -W:]).view(B, W, H_q, D).transpose(1, 2)[:, :1]
        c, s = cos[:, -W:], sin[:, -W:]
        q0 = (q0 * c.unsqueeze(1)) + (_ref_rotate_half(q0) * s.unsqueeze(1))
        w = torch.matmul(q0, keys[:, :1].transpose(2, 3)) / math.sqrt(D)
        mask = torch.triu(torch.ones_like(w) * float("-inf"), diagonal=S - W + 1)
        w = nn.functional.softmax(w + mask, dim=-1, dtype=torch.float32).to(q0.dtype)
        w = w[..., :-W]
        single = F.avg_pool1d(w.mean(dim=-2), kernel_size=3, padding=1, stride=1)
        torch.testing.assert_close(scores[:, 0, : S - W], single[:, 0], rtol=0.0, atol=1e-6)

    def test_parity_vs_real_llama_attention(self):
        from transformers import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaModel

        torch.manual_seed(0)
        config = LlamaConfig(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=64,
        )
        config._attn_implementation = "eager"
        model = LlamaModel(config).eval()

        captured = {}

        def pre_hook(mod, args, kwargs):
            hs = kwargs.get("hidden_states")
            if hs is None and args:
                hs = args[0]
            captured["hidden_states"] = hs
            captured["position_embeddings"] = kwargs.get("position_embeddings")

        attn = model.layers[0].self_attn
        handle = attn.register_forward_pre_hook(pre_hook, with_kwargs=True)
        input_ids = torch.randint(0, 128, (1, 12))
        with torch.no_grad():
            out = model(input_ids, output_attentions=True, use_cache=True)
        handle.remove()

        keys = out.past_key_values.layers[0].keys
        window = SnapKVSketch.compute_window_attention(
            attn, captured["hidden_states"], keys, 4, captured["position_embeddings"]
        )
        expected = out.attentions[0][..., -4:, :-4]
        self.assertEqual(window.shape, (1, 4, 4, 8))
        torch.testing.assert_close(window, expected, rtol=1e-5, atol=1e-5)

    def test_window_size_assertion(self):
        module = _FakeSnapKVAttn(hidden_dim=8, num_heads=1, head_dim=4, num_kv_heads=1)
        B, S, D = 1, 8, 4
        hidden = torch.randn(B, S, 8)
        keys = torch.randn(B, 1, S, D)
        values = torch.randn(B, 1, S, D)
        cos, sin = _rope_pos_emb(torch.arange(S), D)
        kwargs = {"position_embeddings": (cos, sin)}
        for window_size in (8, 64):
            sketch = SnapKVSketch(compression_ratio=0.5, window_size=window_size, kernel_size=3)
            with self.assertRaisesRegex(AssertionError, "greater than the window size"):
                sketch.compress(module, hidden, keys, values, None, kwargs)

    def test_even_kernel_size_rejected(self):
        with self.assertRaisesRegex(AssertionError, "odd"):
            SnapKVSketch(compression_ratio=0.5, kernel_size=4)

    def test_dominant_context_selection(self):
        torch.manual_seed(0)
        module = _FakeSnapKVAttn(hidden_dim=8, num_heads=2, head_dim=4, num_kv_heads=2, seed=5)
        B, S, W, D, H_kv = 1, 10, 3, 4, 2
        hidden = torch.randn(B, S, 8)
        shared = torch.randn(8)
        hidden[0, -W:] = shared
        keys = 0.1 * torch.randn(B, H_kv, S, D)
        with torch.no_grad():
            q = module.q_proj(shared.unsqueeze(0)).view(1, H_kv, D)
        for h in range(H_kv):
            qh = q[0, h]
            keys[0, h, 2] = 6.0 * qh / qh.norm()
        values = _position_values(B, H_kv, S, D)
        cos, sin = _identity_pos_emb(B, S, D)
        kwargs = {"position_embeddings": (cos, sin)}
        sketch = SnapKVSketch(compression_ratio=0.6, window_size=W, kernel_size=1)

        out_k, out_v = sketch.compress(module, hidden, keys, values, None, kwargs)
        self.assertEqual(out_k.shape, (B, H_kv, 4, D))
        for h in range(H_kv):
            self.assertEqual(set(out_v[0, h, :, 0].tolist()), {2.0, 7.0, 8.0, 9.0})

    def test_phi3_style_fused_qkv_proj(self):
        torch.manual_seed(13)
        B, S, W, hidden_dim, H_q, H_kv, D = 1, 8, 3, 24, 4, 2, 6
        fused = _FakeFusedSnapKVAttn(hidden_dim, H_q, D, H_kv, seed=3)
        split = _FakeSnapKVAttn(hidden_dim, H_q, D, H_kv, seed=3)
        with torch.no_grad():
            split.q_proj.weight.copy_(fused.qkv_proj.weight[: H_q * D])
        hidden = torch.randn(B, S, hidden_dim)
        keys = torch.randn(B, H_kv, S, D)
        cos, sin = _rope_pos_emb(torch.arange(S), D)
        kwargs = {"position_embeddings": (cos, sin)}
        sketch = SnapKVSketch(compression_ratio=0.5, window_size=W, kernel_size=3)

        scores_fused = sketch.score(fused, hidden, keys, None, None, kwargs)
        scores_split = sketch.score(split, hidden, keys, None, None, kwargs)
        torch.testing.assert_close(scores_fused, scores_split, rtol=0.0, atol=1e-6)

    def test_qnorm_duck_typing(self):
        torch.manual_seed(11)
        B, S, W, hidden_dim, H_q, H_kv, D = 1, 8, 3, 24, 4, 2, 6
        plain = _FakeSnapKVAttn(hidden_dim, H_q, D, H_kv, seed=2)
        normed = _FakeSnapKVAttn(hidden_dim, H_q, D, H_kv, seed=2)
        normed.q_norm = _DoubleQNorm()
        hidden = torch.randn(B, S, hidden_dim)
        keys = torch.randn(B, H_kv, S, D)
        cos, sin = _rope_pos_emb(torch.arange(S), D)
        kwargs = {"position_embeddings": (cos, sin)}
        sketch = SnapKVSketch(compression_ratio=0.5, window_size=W, kernel_size=3)

        scores_normed = sketch.score(normed, hidden, keys, None, None, kwargs)
        ref = _kvpress_snapkv_reference(normed, hidden, keys, W, 3, cos, sin)
        torch.testing.assert_close(scores_normed, ref, rtol=0.0, atol=0.0)

        scores_plain = sketch.score(plain, hidden, keys, None, None, kwargs)
        self.assertFalse(torch.allclose(scores_normed, scores_plain))


class TestSnapKVForwardHook(unittest.TestCase):
    def test_cross_layer_uniformity(self):
        B, S, W, hidden_dim, H_q, H_kv, D = 1, 16, 2, 24, 4, 2, 6
        cos, sin = _rope_pos_emb(torch.arange(S), D)
        sketch = SnapKVSketch(compression_ratio=0.5, window_size=W, kernel_size=3)
        torch.manual_seed(41)
        layers = [
            _FakeCacheLayer(torch.randn(B, H_kv, S, D), torch.randn(B, H_kv, S, D))
            for _ in range(2)
        ]
        cache = _FakeCache(layers)
        for layer_idx in range(2):
            module = _FakeSnapKVAttn(hidden_dim, H_q, D, H_kv, seed=layer_idx, layer_idx=layer_idx)
            kwargs = {
                "hidden_states": torch.randn(B, S, hidden_dim),
                "past_key_values": cache,
                "cache_position": torch.arange(S),
                "position_embeddings": (cos, sin),
            }
            output = (torch.randn(B, S, hidden_dim), None)
            result = sketch.forward_hook(module, [], kwargs, output)
            self.assertIs(result, output)
        for layer_idx in range(2):
            self.assertEqual(cache.layers[layer_idx].keys.shape, (B, H_kv, 8, D))
            self.assertEqual(cache.layers[layer_idx].values.shape, (B, H_kv, 8, D))

    def test_decode_step_noop(self):
        B, S, hidden_dim, H_kv, D = 1, 8, 24, 2, 6
        module = _FakeSnapKVAttn(hidden_dim, 4, D, H_kv, seed=1)
        keys = torch.randn(B, H_kv, S, D)
        values = torch.randn(B, H_kv, S, D)
        cache = _FakeCache([_FakeCacheLayer(keys, values)])
        sketch = SnapKVSketch(compression_ratio=0.5, window_size=2, kernel_size=3)
        kwargs = {
            "hidden_states": torch.randn(B, 1, hidden_dim),
            "past_key_values": cache,
            "cache_position": torch.tensor([S + 3]),
        }
        output = (torch.randn(B, 1, hidden_dim), None)
        result = sketch.forward_hook(module, [], kwargs, output)
        self.assertIs(result, output)
        self.assertIs(cache.layers[0].keys, keys)
        self.assertIs(cache.layers[0].values, values)


class TestSnapKVRegistry(unittest.TestCase):
    def test_registered_as_snapkv(self):
        from eval_harness.sketch.sketches.registry import (
            available_sketches,
            get_sketch,
            get_sketch_class,
        )

        self.assertIn("snapkv", available_sketches())
        self.assertIs(get_sketch_class("snapkv"), SnapKVSketch)
        sketch = get_sketch("snapkv", compression_ratio=0.25, window_size=8, kernel_size=3)
        self.assertIsInstance(sketch, SnapKVSketch)
        self.assertAlmostEqual(sketch.compression_ratio, 0.25)
        self.assertEqual(sketch.window_size, 8)
        self.assertEqual(sketch.kernel_size, 3)


if __name__ == "__main__":
    unittest.main()
