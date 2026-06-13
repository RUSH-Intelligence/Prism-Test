"""Tests for ThinKSketch (kvpress ThinKPress port).

Value-pinned against hand computations and an in-test transcription of the
kvpress ThinKPress math (think_press.py:78-88). No model loading.
"""

from __future__ import annotations

import dataclasses
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from eval_harness.sketch.sketches.think_sketch import ThinKSketch


def _rotate_half_ref(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _identity_pos_emb(B: int, S: int, D: int):
    """RoPE that is a no-op: cos=1, sin=0."""
    return torch.ones(B, S, D), torch.zeros(B, S, D)


def _rope_pos_emb(positions: torch.Tensor, D: int, B: int = 1, base: float = 10000.0):
    """Real Llama-style (cos, sin) of shape [B, S, D] for the given positions."""
    half = D // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))
    freqs = positions.float()[:, None] * inv_freq[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos()[None].expand(B, -1, -1), emb.sin()[None].expand(B, -1, -1)


class _FakeThinKAttn(nn.Module):
    """Minimal Llama-like attention module for ThinK tests."""

    def __init__(self, hidden_dim=8, num_heads=2, head_dim=4, identity_q=False, seed=0, layer_idx=0):
        super().__init__()
        self.head_dim = head_dim
        self.layer_idx = layer_idx
        self.config = SimpleNamespace(num_attention_heads=num_heads)
        self.q_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        if identity_q:
            assert hidden_dim == num_heads * head_dim
            with torch.no_grad():
                self.q_proj.weight.copy_(torch.eye(hidden_dim))
        else:
            torch.manual_seed(seed)
            with torch.no_grad():
                self.q_proj.weight.normal_()


class _FakeFusedQKVAttn(nn.Module):
    """Phi3-style attention module with a fused qkv_proj and no q_proj."""

    def __init__(self, hidden_dim=8, num_heads=2, head_dim=4, num_kv_heads=1, seed=0, layer_idx=0):
        super().__init__()
        self.head_dim = head_dim
        self.layer_idx = layer_idx
        self.config = SimpleNamespace(num_attention_heads=num_heads)
        out_dim = (num_heads + 2 * num_kv_heads) * head_dim
        self.qkv_proj = nn.Linear(hidden_dim, out_dim, bias=False)
        torch.manual_seed(seed)
        with torch.no_grad():
            self.qkv_proj.weight.normal_()


class _FakeCacheLayer:
    def __init__(self, keys: torch.Tensor, values: torch.Tensor):
        self.keys = keys
        self.values = values


class _FakeCache:
    def __init__(self, keys: torch.Tensor, values: torch.Tensor):
        self.layers = [_FakeCacheLayer(keys, values)]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.layers[layer_idx].keys.shape[2]


def _think_reference(module, hidden_states, position_embeddings, keys, ratio, window_size):
    """In-test transcription of kvpress think_press.py:78-88 (clones keys)."""
    keys = keys.clone()
    bsz, num_key_value_heads, k_len, head_dim = keys.shape
    num_key_value_groups = module.config.num_attention_heads // num_key_value_heads

    window = hidden_states[:, -window_size:]
    b, w, _ = window.shape
    q = module.q_proj(window).view(b, w, module.config.num_attention_heads, head_dim).transpose(1, 2)
    cos, sin = position_embeddings
    cos, sin = cos[:, -window_size:], sin[:, -window_size:]
    q = (q * cos.unsqueeze(1)) + (_rotate_half_ref(q) * sin.unsqueeze(1))

    queries_norm = torch.pow(q, 2).mean(dim=2)
    queries_norm = queries_norm.view(bsz, num_key_value_heads, num_key_value_groups, head_dim).mean(2)
    keys_norm = torch.pow(keys, 2).mean(dim=2)
    key_scores = queries_norm * keys_norm

    n_pruned = int(head_dim * ratio)
    indices = key_scores.topk(n_pruned, dim=-1, largest=False).indices
    indices = indices.unsqueeze(2).expand(-1, -1, k_len, -1)
    return keys.scatter_(-1, indices, 0)


class TestThinKSketch(unittest.TestCase):
    def _kwargs(self, hidden, pos_emb):
        return {"hidden_states": hidden, "position_embeddings": pos_emb}

    def test_zero_ratio_returns_same_tensor_objects(self):
        sketch = ThinKSketch(key_channel_compression_ratio=0.0)
        module = _FakeThinKAttn(hidden_dim=8, num_heads=2, head_dim=4)
        keys = torch.randn(1, 2, 8, 4)
        values = torch.randn(1, 2, 8, 4)
        hidden = torch.randn(1, 8, 8)
        keys_out, values_out = sketch.compress(
            module, hidden, keys, values, None, self._kwargs(hidden, _identity_pos_emb(1, 8, 4))
        )
        self.assertIs(keys_out, keys)
        self.assertIs(values_out, values)

    def test_post_init_rejects_out_of_range_ratio(self):
        with self.assertRaises(AssertionError):
            ThinKSketch(key_channel_compression_ratio=1.0)
        with self.assertRaises(AssertionError):
            ThinKSketch(key_channel_compression_ratio=-0.1)

    def test_hand_computed_channel_selection(self):
        sketch = ThinKSketch(key_channel_compression_ratio=0.5, window_size=2)
        module = _FakeThinKAttn(hidden_dim=4, num_heads=1, head_dim=4, identity_q=True)
        hidden = torch.full((1, 4, 4), 99.0)
        hidden[:, -2:] = torch.tensor([[1.0, 2.0, 0.0, 0.0], [1.0, 2.0, 0.0, 0.0]])
        keys = torch.ones(1, 1, 4, 4)
        keys[..., 1] = 2.0
        values = torch.randn(1, 1, 4, 4)
        keys_out, values_out = sketch.compress(
            module, hidden, keys, values, None, self._kwargs(hidden, _identity_pos_emb(1, 4, 4))
        )
        expected = torch.tensor([1.0, 2.0, 0.0, 0.0]).expand(1, 1, 4, 4)
        self.assertTrue(torch.equal(keys_out, expected))
        self.assertIs(values_out, values)

    def test_gqa_group_averaging_per_kv_head_channels(self):
        sketch = ThinKSketch(key_channel_compression_ratio=0.5, window_size=2)
        module = _FakeThinKAttn(hidden_dim=16, num_heads=4, head_dim=4, identity_q=True)
        head_lo = [1.0, 1.0, 0.0, 0.0]
        head_hi = [0.0, 0.0, 1.0, 1.0]
        row = torch.tensor(head_lo + head_lo + head_hi + head_hi)
        hidden = torch.zeros(1, 5, 16)
        hidden[:, -2:] = row
        keys = torch.ones(1, 2, 6, 4)
        values = torch.randn(1, 2, 6, 4)
        keys_out, _ = sketch.compress(
            module, hidden, keys, values, None, self._kwargs(hidden, _identity_pos_emb(1, 5, 4))
        )
        self.assertTrue(torch.equal(keys_out[:, 0, :, :2], torch.ones(1, 6, 2)))
        self.assertTrue(torch.equal(keys_out[:, 0, :, 2:], torch.zeros(1, 6, 2)))
        self.assertTrue(torch.equal(keys_out[:, 1, :, :2], torch.zeros(1, 6, 2)))
        self.assertTrue(torch.equal(keys_out[:, 1, :, 2:], torch.ones(1, 6, 2)))

    def test_reference_transcription_oracle(self):
        torch.manual_seed(7)
        B, H_q, H_kv, D, W, q_len, k_len = 2, 8, 4, 8, 3, 16, 12
        module = _FakeThinKAttn(hidden_dim=H_q * D, num_heads=H_q, head_dim=D, seed=3)
        hidden = torch.randn(B, q_len, H_q * D)
        keys = torch.randn(B, H_kv, k_len, D)
        values = torch.randn(B, H_kv, k_len, D)
        cos, sin = _rope_pos_emb(torch.arange(q_len), D, B=B)
        sketch = ThinKSketch(key_channel_compression_ratio=0.5, window_size=W)
        keys_out, values_out = sketch.compress(
            module, hidden, keys.clone(), values, None, self._kwargs(hidden, (cos, sin))
        )
        oracle = _think_reference(module, hidden, (cos, sin), keys, 0.5, W)
        self.assertTrue(torch.equal(keys_out, oracle))
        self.assertIs(values_out, values)

    def test_window_larger_than_sequence(self):
        torch.manual_seed(1)
        B, H_q, H_kv, D, S = 1, 2, 1, 4, 3
        module = _FakeThinKAttn(hidden_dim=H_q * D, num_heads=H_q, head_dim=D, seed=5)
        hidden = torch.randn(B, S, H_q * D)
        keys = torch.randn(B, H_kv, S, D)
        values = torch.randn(B, H_kv, S, D)
        cos, sin = _rope_pos_emb(torch.arange(S), D, B=B)
        sketch = ThinKSketch(key_channel_compression_ratio=0.5, window_size=32)
        keys_out, _ = sketch.compress(
            module, hidden, keys.clone(), values, None, self._kwargs(hidden, (cos, sin))
        )
        oracle = _think_reference(module, hidden, (cos, sin), keys, 0.5, 32)
        self.assertEqual(keys_out.shape, keys.shape)
        self.assertTrue(torch.equal(keys_out, oracle))

    def test_n_pruned_zero_is_noop(self):
        torch.manual_seed(2)
        module = _FakeThinKAttn(hidden_dim=16, num_heads=2, head_dim=8, seed=2)
        hidden = torch.randn(1, 6, 16)
        keys = torch.randn(1, 2, 6, 8)
        values = torch.randn(1, 2, 6, 8)
        before = keys.clone()
        sketch = ThinKSketch(key_channel_compression_ratio=0.1, window_size=4)
        keys_out, _ = sketch.compress(
            module, hidden, keys, values, None, self._kwargs(hidden, _identity_pos_emb(1, 6, 8))
        )
        self.assertIs(keys_out, keys)
        self.assertTrue(torch.equal(keys_out, before))

    def test_channel_count_invariant(self):
        torch.manual_seed(3)
        module = _FakeThinKAttn(hidden_dim=16, num_heads=2, head_dim=8, seed=4)
        hidden = torch.randn(1, 6, 16)
        keys = torch.rand(1, 2, 6, 8) + 0.5
        values = torch.randn(1, 2, 6, 8)
        sketch = ThinKSketch(key_channel_compression_ratio=0.5, window_size=4)
        keys_out, _ = sketch.compress(
            module, hidden, keys, values, None, self._kwargs(hidden, _identity_pos_emb(1, 6, 8))
        )
        self.assertEqual(keys_out.shape, (1, 2, 6, 8))
        zero_channels = (keys_out == 0).all(dim=2).sum(dim=-1)
        self.assertTrue(torch.equal(zero_channels, torch.full((1, 2), 4)))

    def test_decode_dot_product_equivalence(self):
        torch.manual_seed(4)
        module = _FakeThinKAttn(hidden_dim=8, num_heads=1, head_dim=8, seed=6)
        hidden = torch.randint(-4, 5, (1, 5, 8)).float()
        keys = torch.randint(1, 5, (1, 1, 5, 8)).float()
        values = torch.randn(1, 1, 5, 8)
        sketch = ThinKSketch(key_channel_compression_ratio=0.5, window_size=4)
        keys_out, _ = sketch.compress(
            module, hidden, keys, values, None, self._kwargs(hidden, _identity_pos_emb(1, 5, 8))
        )
        pruned_mask = (keys_out == 0).all(dim=2)[0, 0]
        kept = (~pruned_mask).nonzero(as_tuple=True)[0]
        self.assertEqual(kept.numel(), 4)
        q = torch.randint(-4, 5, (1, 1, 1, 8)).float()
        full = q @ keys_out.transpose(-2, -1)
        kept_only = q[..., kept] @ keys_out[..., kept].transpose(-2, -1)
        self.assertTrue(torch.equal(full, kept_only))

    def test_compression_ratio_property_parity(self):
        sketch = ThinKSketch(key_channel_compression_ratio=0.4)
        self.assertEqual(sketch.compression_ratio, 0.2)
        with self.assertRaises(AttributeError):
            sketch.compression_ratio = 0.5
        field_names = {f.name for f in dataclasses.fields(ThinKSketch)}
        # The KVCompressor base contributes the schedule/operation model; ThinK's
        # OWN fields remain exactly these two, and compression_ratio stays a
        # computed property (never a field).
        base_fields = {"schedule", "operation", "decode_interval"}
        self.assertEqual(field_names - base_fields, {"key_channel_compression_ratio", "window_size"})
        self.assertNotIn("compression_ratio", field_names)

    def test_fused_qkv_proj_path_matches_q_slice(self):
        torch.manual_seed(8)
        fused = _FakeFusedQKVAttn(hidden_dim=8, num_heads=2, head_dim=4, num_kv_heads=1, seed=8)
        plain = _FakeThinKAttn(hidden_dim=8, num_heads=2, head_dim=4, seed=8)
        with torch.no_grad():
            fused.qkv_proj.weight.copy_(torch.randint(-3, 4, fused.qkv_proj.weight.shape).float())
            plain.q_proj.weight.copy_(fused.qkv_proj.weight[: 2 * 4])
        hidden = torch.randint(-3, 4, (1, 5, 8)).float()
        sketch = ThinKSketch(key_channel_compression_ratio=0.5, window_size=3)
        pos_emb = _identity_pos_emb(1, 5, 4)
        q_fused = sketch.compute_window_queries(fused, hidden, pos_emb)
        q_plain = sketch.compute_window_queries(plain, hidden, pos_emb)
        self.assertTrue(torch.equal(q_fused, q_plain))

    def test_q_norm_duck_typing(self):
        class _Double(nn.Module):
            def forward(self, x):
                return x * 2.0

        plain = _FakeThinKAttn(hidden_dim=8, num_heads=2, head_dim=4, seed=9)
        normed = _FakeThinKAttn(hidden_dim=8, num_heads=2, head_dim=4, seed=9)
        with torch.no_grad():
            normed.q_proj.weight.copy_(plain.q_proj.weight)
        normed.q_norm = _Double()
        hidden = torch.randn(1, 5, 8)
        sketch = ThinKSketch(key_channel_compression_ratio=0.5, window_size=3)
        pos_emb = _identity_pos_emb(1, 5, 4)
        q_plain = sketch.compute_window_queries(plain, hidden, pos_emb)
        q_normed = sketch.compute_window_queries(normed, hidden, pos_emb)
        self.assertTrue(torch.equal(q_normed, q_plain * 2.0))


class TestThinKForwardHook(unittest.TestCase):
    def _build(self, S, decode=False):
        module = _FakeThinKAttn(hidden_dim=8, num_heads=2, head_dim=4, seed=10)
        keys = torch.rand(1, 2, S, 4) + 0.5
        values = torch.randn(1, 2, S, 4)
        cache = _FakeCache(keys, values)
        if decode:
            hidden = torch.randn(1, 1, 8)
            cache_position = torch.tensor([S - 1])
        else:
            hidden = torch.randn(1, S, 8)
            cache_position = torch.arange(S)
        kwargs = {
            "hidden_states": hidden,
            "past_key_values": cache,
            "cache_position": cache_position,
            "position_embeddings": _identity_pos_emb(1, hidden.shape[1], 4),
        }
        output = (torch.zeros(1, hidden.shape[1], 8), None)
        return module, cache, kwargs, output

    def test_prefill_compresses_cache_in_place(self):
        sketch = ThinKSketch(key_channel_compression_ratio=0.5, window_size=4)
        module, cache, kwargs, output = self._build(S=6)
        sketch.forward_hook(module, [], kwargs, output)
        keys = cache.layers[0].keys
        self.assertEqual(keys.shape, (1, 2, 6, 4))
        zero_channels = (keys == 0).all(dim=2).sum(dim=-1)
        self.assertTrue(torch.equal(zero_channels, torch.full((1, 2), 2)))

    def test_decode_step_leaves_cache_untouched(self):
        sketch = ThinKSketch(key_channel_compression_ratio=0.5, window_size=4)
        module, cache, kwargs, output = self._build(S=7, decode=True)
        before = cache.layers[0].keys.clone()
        sketch.forward_hook(module, [], kwargs, output)
        self.assertTrue(torch.equal(cache.layers[0].keys, before))


class TestThinKRegistry(unittest.TestCase):
    def test_registered_name_resolves(self):
        from eval_harness.sketch.sketches.registry import get_sketch, get_sketch_class

        self.assertIs(get_sketch_class("think"), ThinKSketch)
        sketch = get_sketch("think", key_channel_compression_ratio=0.5, window_size=2)
        self.assertIsInstance(sketch, ThinKSketch)
        self.assertEqual(sketch.key_channel_compression_ratio, 0.5)
        self.assertEqual(sketch.window_size, 2)

    def test_build_sketch_does_not_inject_adapter_ratio(self):
        from eval_harness.research_adapter import CacheConfig, ResearchAdapter

        cfg = CacheConfig(
            sketch_name="think",
            compression_ratio=0.9,
            sketch_kwargs={"key_channel_compression_ratio": 0.5, "window_size": 2},
        )
        adapter = object.__new__(ResearchAdapter)
        adapter._cache_cfg = cfg
        sketch = adapter._build_sketch(cfg)
        self.assertIsInstance(sketch, ThinKSketch)
        self.assertEqual(sketch.key_channel_compression_ratio, 0.5)
        self.assertEqual(sketch.window_size, 2)
        self.assertEqual(sketch.compression_ratio, 0.25)


if __name__ == "__main__":
    unittest.main()
