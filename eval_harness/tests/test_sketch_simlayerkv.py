"""Tests for SimLayerKVSketch (port of kvpress SimLayerKVPress).

GPU-free: fake attention modules + synthetic tensors; the kvpress math
(SnapKV window attention, lazy-layer decision, lazy truncation) is
re-implemented locally as a reference oracle and value-pinned.
"""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from eval_harness.kv_compression.compressors.simlayerkv_sketch import SimLayerKVSketch


# ======================================================================
# Local fakes and reference oracle (kvpress transcription)
# ======================================================================


class _FakeAttnModule(nn.Module):
    """Minimal Llama-like attention module: q_proj + config + head_dim + layer_idx."""

    def __init__(self, hidden_dim=8, num_heads=2, num_kv_heads=1, head_dim=4, layer_idx=0, seed=0):
        super().__init__()
        self.config = SimpleNamespace(
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
        )
        self.num_heads = num_heads
        self.num_key_value_heads = num_kv_heads
        self.head_dim = head_dim
        self.layer_idx = layer_idx
        self.q_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        torch.manual_seed(seed)
        with torch.no_grad():
            self.q_proj.weight.normal_()


def _identity_pos_emb(B, S, D):
    """RoPE that is a no-op: cos=1, sin=0."""
    return torch.ones(B, S, D), torch.zeros(B, S, D)


def _rope_pos_emb(S, D, B=1, base=10000.0):
    """Real (cos, sin) of shape [B, S, D] for positions 0..S-1 (HF convention)."""
    half = D // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))
    freqs = torch.einsum("s,d->sd", torch.arange(S, dtype=torch.float32), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().unsqueeze(0).expand(B, S, D), emb.sin().unsqueeze(0).expand(B, S, D)


def _rotate_half_ref(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _repeat_kv_ref(hidden, n_rep):
    b, h_kv, s, d = hidden.shape
    if n_rep == 1:
        return hidden
    return hidden[:, :, None, :, :].expand(b, h_kv, n_rep, s, d).reshape(b, h_kv * n_rep, s, d)


def _window_attention_ref(module, hidden_states, keys, window, cos, sin, truncate=True):
    """Independent transcription of kvpress SnapKVPress.compute_window_attention."""
    bsz, _, k_len, _ = keys.shape
    h_q = module.config.num_attention_heads
    d = module.head_dim
    n_rep = h_q // module.config.num_key_value_heads

    q = module.q_proj(hidden_states[:, -window:])
    q = q.view(bsz, window, h_q, d).transpose(1, 2)
    c, s = cos[:, -window:].unsqueeze(1), sin[:, -window:].unsqueeze(1)
    q = q * c + _rotate_half_ref(q) * s

    k = _repeat_kv_ref(keys, n_rep)
    aw = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(d)
    mask = torch.triu(torch.full_like(aw, float("-inf")), diagonal=k_len - window + 1)
    aw = aw + mask
    aw = torch.softmax(aw, dim=-1, dtype=torch.float32).to(q.dtype)
    return aw[..., :-window] if truncate else aw


def _lazy_score_ref(module, hidden_states, keys, cos, sin, n_last, n_initial, n_recent):
    aw = _window_attention_ref(module, hidden_states, keys, n_last, cos, sin)
    v = aw.mean((0, 1, 2))
    return (v[:n_initial].sum() + v[-n_recent:].sum()).item()


def _kv(B, H_kv, S, D, seed=0):
    torch.manual_seed(seed)
    return torch.randn(B, H_kv, S, D), torch.randn(B, H_kv, S, D)


def _position_keys(H_kv, S, D, offset=0.0):
    """keys[0, :, s, :] == s + offset, so kept positions are readable from values."""
    t = torch.arange(S, dtype=torch.float32) + offset
    return t.view(1, 1, S, 1).expand(1, H_kv, S, D).clone()


LOGGER_NAME = "eval_harness.kv_compression.compressors.simlayerkv_sketch"


# ======================================================================
# Defaults, guards, telemetry property
# ======================================================================


class TestDefaultsAndGuards(unittest.TestCase):
    def test_default_threshold_is_noop(self):
        sketch = SimLayerKVSketch()
        self.assertEqual(sketch.lazy_threshold, 1.0)
        module = _FakeAttnModule(num_kv_heads=2, num_heads=2, hidden_dim=8)
        keys, values = _kv(1, 2, 16, 8)
        sketch.is_lazy = lambda *a, **k: self.fail("is_lazy must not be called at threshold 1.0")
        hidden = torch.randn(1, 16, 8)
        kwargs = {"position_embeddings": _identity_pos_emb(1, 16, 4)}
        with self.assertLogs(LOGGER_NAME, level="WARNING"):
            # Default n_recent=1024 also trips the (verbatim) short-sequence warning.
            out_k, out_v = sketch.compress(module, hidden, keys, values, None, kwargs)
        self.assertIs(out_k, keys)
        self.assertIs(out_v, values)
        self.assertEqual(out_k.shape[2], 16)
        self.assertTrue(torch.equal(out_k, keys))
        self.assertEqual(sketch.compression_ratios, [0.0])
        self.assertEqual(sketch.compression_ratio, 0.0)

    def test_threshold_one_short_circuits_before_is_lazy_even_above_min_length(self):
        sketch = SimLayerKVSketch(lazy_threshold=1.0, n_initial=1, n_recent=2, n_last=1)
        sketch.is_lazy = lambda *a, **k: self.fail("is_lazy must not be called at threshold 1.0")
        module = _FakeAttnModule()
        keys, values = _kv(1, 1, 16, 4)
        out_k, _ = sketch.compress(module, torch.randn(1, 16, 8), keys, values, None, {})
        self.assertEqual(out_k.shape[2], 16)
        self.assertEqual(sketch.compression_ratios, [0.0])

    def test_lazy_threshold_validation(self):
        with self.assertRaises(AssertionError):
            SimLayerKVSketch(lazy_threshold=1.5)
        with self.assertRaises(AssertionError):
            SimLayerKVSketch(lazy_threshold=-0.1)

    def test_compression_ratio_property_guards(self):
        sketch = SimLayerKVSketch()
        with self.assertRaisesRegex(ValueError, "Forward pass must be run"):
            _ = sketch.compression_ratio
        with self.assertRaisesRegex(AttributeError, "compression ratio cannot be set"):
            sketch.compression_ratio = 0.5


class TestShortSequenceGuard(unittest.TestCase):
    def test_boundary_k_len_equal_min_length_warns_and_noop(self):
        sketch = SimLayerKVSketch(lazy_threshold=0.0, n_initial=1, n_recent=2, n_last=1)
        sketch.is_lazy = lambda *a, **k: self.fail("is_lazy must not be called below min_length")
        module = _FakeAttnModule()
        keys, values = _kv(1, 1, 4, 4)
        with self.assertLogs(LOGGER_NAME, level="WARNING") as cm:
            out_k, out_v = sketch.compress(module, torch.randn(1, 4, 8), keys, values, None, {})
        self.assertTrue(any("no compression applied" in m for m in cm.output))
        self.assertIs(out_k, keys)
        self.assertIs(out_v, values)
        self.assertEqual(sketch.compression_ratios, [0.0])

    def test_k_len_one_above_min_length_compresses(self):
        sketch = SimLayerKVSketch(lazy_threshold=0.0, n_initial=1, n_recent=2, n_last=1)
        sketch.is_lazy = lambda *a, **k: True
        module = _FakeAttnModule()
        keys = _position_keys(1, 5, 4)
        values = _position_keys(1, 5, 4, offset=100.0)
        kwargs = {"position_embeddings": _identity_pos_emb(1, 5, 4)}
        out_k, out_v = sketch.compress(module, torch.randn(1, 5, 8), keys, values, None, kwargs)
        self.assertEqual(out_k.shape[2], 2)  # n_initial + (n_recent - n_last)
        self.assertEqual(out_k[0, 0, :, 0].tolist(), [0.0, 4.0])
        self.assertEqual(out_v[0, 0, :, 0].tolist(), [100.0, 104.0])
        self.assertAlmostEqual(sketch.compression_ratios[0], (5 - 1 - 2 + 1) / 5)


# ======================================================================
# Lazy truncation: exact kept positions + the verbatim ratio quirk
# ======================================================================


class TestLazySelection(unittest.TestCase):
    def test_exact_selection_positions(self):
        sketch = SimLayerKVSketch(lazy_threshold=0.5, n_initial=2, n_recent=3, n_last=1)
        sketch.is_lazy = lambda *a, **k: True
        module = _FakeAttnModule(num_kv_heads=2)
        keys = _position_keys(2, 10, 4)
        values = _position_keys(2, 10, 4, offset=100.0)
        kwargs = {"position_embeddings": _identity_pos_emb(1, 10, 4)}
        out_k, out_v = sketch.compress(module, torch.randn(1, 10, 8), keys, values, None, kwargs)
        self.assertEqual(out_k.shape, (1, 2, 4, 4))
        for h in range(2):
            self.assertEqual(out_k[0, h, :, 0].tolist(), [0.0, 1.0, 8.0, 9.0])
            self.assertEqual(out_v[0, h, :, 0].tolist(), [100.0, 101.0, 108.0, 109.0])
        self.assertEqual(len(sketch.compression_ratios), 1)
        self.assertAlmostEqual(sketch.compression_ratios[0], (10 - 2 - 3 + 1) / 10)
        self.assertAlmostEqual(sketch.compression_ratio, 0.6)

    def test_n_last_gt1_slice_and_logged_ratio_quirk(self):
        sketch = SimLayerKVSketch(lazy_threshold=0.5, n_initial=1, n_recent=4, n_last=2)
        sketch.is_lazy = lambda *a, **k: True
        module = _FakeAttnModule()
        keys = _position_keys(1, 12, 4)
        values = _position_keys(1, 12, 4, offset=100.0)
        kwargs = {"position_embeddings": _identity_pos_emb(1, 12, 4)}
        out_k, out_v = sketch.compress(module, torch.randn(1, 12, 8), keys, values, None, kwargs)
        self.assertEqual(out_k.shape[2], 3)  # n_initial + n_recent - n_last
        self.assertEqual(out_k[0, 0, :, 0].tolist(), [0.0, 10.0, 11.0])
        self.assertEqual(out_v[0, 0, :, 0].tolist(), [100.0, 110.0, 111.0])
        # Verbatim kvpress quirk: logged ratio hardcodes "+1" (n_last=1), NOT the
        # actually-kept fraction (12 - 3) / 12.
        self.assertAlmostEqual(sketch.compression_ratios[0], (12 - 1 - 4 + 1) / 12)
        self.assertNotAlmostEqual(sketch.compression_ratios[0], (12 - 3) / 12)


# ======================================================================
# is_lazy / window-attention oracle (value-pinned against local kvpress
# transcription)
# ======================================================================


class TestIsLazyOracle(unittest.TestCase):
    def _setup(self, n_last=2, seed=3):
        module = _FakeAttnModule(hidden_dim=8, num_heads=2, num_kv_heads=1, head_dim=4, seed=seed)
        torch.manual_seed(seed + 100)
        hidden = torch.randn(1, 12, 8)
        keys = torch.randn(1, 1, 12, 4)
        cos, sin = _rope_pos_emb(12, 4)
        return module, hidden, keys, cos, sin

    def test_window_attention_matches_reference_real_rope(self):
        n_last = 2
        module, hidden, keys, cos, sin = self._setup(n_last=n_last)
        prod = SimLayerKVSketch.compute_window_attention(module, hidden, keys, n_last, (cos, sin))
        ref = _window_attention_ref(module, hidden, keys, n_last, cos, sin)
        self.assertEqual(prod.shape, (1, 2, n_last, 12 - n_last))
        torch.testing.assert_close(prod, ref, atol=1e-6, rtol=0)

    def test_is_lazy_threshold_boundary(self):
        n_last, n_initial, n_recent = 2, 2, 4
        module, hidden, keys, cos, sin = self._setup(n_last=n_last)
        score = _lazy_score_ref(module, hidden, keys, cos, sin, n_last, n_initial, n_recent)
        self.assertGreater(score, 1e-4)
        self.assertLess(score, 1.0 - 1e-4)

        lo = SimLayerKVSketch(lazy_threshold=score - 1e-4, n_last=n_last, n_recent=n_recent, n_initial=n_initial)
        hi = SimLayerKVSketch(lazy_threshold=score + 1e-4, n_last=n_last, n_recent=n_recent, n_initial=n_initial)
        self.assertTrue(lo.is_lazy(module, hidden, keys, (cos, sin)))
        self.assertFalse(hi.is_lazy(module, hidden, keys, (cos, sin)))

    def test_gqa_window_attention_and_decision(self):
        n_last, n_initial, n_recent = 2, 2, 4
        module = _FakeAttnModule(hidden_dim=16, num_heads=4, num_kv_heads=2, head_dim=4, seed=7)
        torch.manual_seed(11)
        hidden = torch.randn(1, 10, 16)
        keys = torch.empty(1, 2, 10, 4)
        keys[:, 0] = torch.randn(1, 10, 4) * 0.1
        keys[:, 1] = torch.randn(1, 10, 4) * 5.0
        cos, sin = _rope_pos_emb(10, 4)

        prod = SimLayerKVSketch.compute_window_attention(module, hidden, keys, n_last, (cos, sin))
        ref = _window_attention_ref(module, hidden, keys, n_last, cos, sin)
        self.assertEqual(prod.shape, (1, 4, n_last, 10 - n_last))
        torch.testing.assert_close(prod, ref, atol=1e-6, rtol=0)

        score = _lazy_score_ref(module, hidden, keys, cos, sin, n_last, n_initial, n_recent)
        for thr, expected in ((max(score - 1e-4, 0.0), True), (min(score + 1e-4, 1.0), False)):
            sketch = SimLayerKVSketch(
                lazy_threshold=thr, n_last=n_last, n_recent=n_recent, n_initial=n_initial
            )
            self.assertEqual(sketch.is_lazy(module, hidden, keys, (cos, sin)), expected)

    def test_window_causal_mask(self):
        n_last, S = 2, 8
        module = _FakeAttnModule(hidden_dim=8, num_heads=2, num_kv_heads=1, head_dim=4, seed=5)
        torch.manual_seed(13)
        hidden = torch.randn(1, S, 8)
        keys = torch.randn(1, 1, S, 4)
        cos, sin = _identity_pos_emb(1, S, 4)

        full = _window_attention_ref(module, hidden, keys, n_last, cos, sin, truncate=False)
        # Query at position S-2 (first window row) puts exactly 0 mass on key S-1.
        self.assertTrue(torch.all(full[:, :, 0, -1] == 0.0))
        # Query at position S-1 (last row) sees everything.
        self.assertTrue(torch.all(full[:, :, 1, :] > 0.0))
        torch.testing.assert_close(
            full.sum(-1), torch.ones(1, 2, n_last), atol=1e-6, rtol=0
        )
        prod = SimLayerKVSketch.compute_window_attention(module, hidden, keys, n_last, (cos, sin))
        torch.testing.assert_close(prod, full[..., :-n_last], atol=1e-6, rtol=0)


# ======================================================================
# Cross-layer raggedness, telemetry reset, decode gate
# ======================================================================


class TestCrossLayerAndLifecycle(unittest.TestCase):
    def _compress(self, sketch, module, S=6):
        keys = _position_keys(1, S, 4)
        values = _position_keys(1, S, 4, offset=100.0)
        kwargs = {"position_embeddings": _identity_pos_emb(1, S, 4)}
        return sketch.compress(module, torch.randn(1, S, 8), keys, values, None, kwargs)

    def test_cross_layer_raggedness_and_telemetry_reset(self):
        S = 6
        sketch = SimLayerKVSketch(lazy_threshold=0.5, n_initial=1, n_recent=2, n_last=1)
        sketch.is_lazy = lambda module, *a, **k: module.layer_idx == 1
        layer0 = _FakeAttnModule(layer_idx=0)
        layer1 = _FakeAttnModule(layer_idx=1)

        k0, _ = self._compress(sketch, layer0, S)
        k1, _ = self._compress(sketch, layer1, S)
        self.assertEqual(k0.shape[2], S)
        self.assertEqual(k1.shape[2], 1 + 2 - 1)  # n_initial + n_recent - n_last
        expected_ratio = (S - 1 - 2 + 1) / S
        self.assertEqual(len(sketch.compression_ratios), 2)
        self.assertAlmostEqual(sketch.compression_ratios[0], 0.0)
        self.assertAlmostEqual(sketch.compression_ratios[1], expected_ratio)
        self.assertAlmostEqual(sketch.compression_ratio, (0.0 + expected_ratio) / 2)

        # New prefill: layer 0 fires again -> the telemetry list is RESET.
        self._compress(sketch, layer0, S)
        self.assertEqual(len(sketch.compression_ratios), 1)

    def test_reset_on_first_hooked_layer_when_layer0_skipped(self):
        # Prism deviation: on mixed-attention models layer 0 may never be hooked;
        # the reset must still fire on the first hooked layer of each prefill.
        sketch = SimLayerKVSketch(lazy_threshold=0.5, n_initial=1, n_recent=2, n_last=1)
        sketch.is_lazy = lambda *a, **k: False
        layer2 = _FakeAttnModule(layer_idx=2)
        layer3 = _FakeAttnModule(layer_idx=3)

        self._compress(sketch, layer2)
        self._compress(sketch, layer3)
        self.assertEqual(len(sketch.compression_ratios), 2)
        self._compress(sketch, layer2)  # second prefill
        self.assertEqual(len(sketch.compression_ratios), 1)

    def test_decode_step_hook_noop(self):
        S = 8
        sketch = SimLayerKVSketch(lazy_threshold=0.5, n_initial=1, n_recent=2, n_last=1)
        sketch.is_lazy = lambda *a, **k: self.fail("compress must not run on decode steps")
        module = _FakeAttnModule(layer_idx=0)
        keys, values = _kv(1, 1, S, 4)
        cache = SimpleNamespace(layers=[SimpleNamespace(keys=keys, values=values)])
        kwargs = {
            "hidden_states": torch.randn(1, 1, 8),
            "past_key_values": cache,
            "cache_position": torch.tensor([S + 3]),
            "position_embeddings": _identity_pos_emb(1, 1, 4),
        }
        output = (torch.zeros(1, 1, 8), None)
        result = sketch.forward_hook(module, [], kwargs, output)
        self.assertIs(result, output)
        self.assertIs(cache.layers[0].keys, keys)
        self.assertEqual(cache.layers[0].keys.shape[2], S)


# ======================================================================
# Flash-attention assertion (Prism adaptation for ragged decode safety)
# ======================================================================


class TestFlashAssertion(unittest.TestCase):
    @staticmethod
    def _model(attn_implementation):
        return SimpleNamespace(config=SimpleNamespace(_attn_implementation=attn_implementation))

    def test_sdpa_raises_when_compression_enabled(self):
        sketch = SimLayerKVSketch(lazy_threshold=0.5)
        for impl in ("sdpa", "eager", None):
            with self.assertRaisesRegex(ValueError, "flash_attention_2"):
                sketch.post_init_from_model(self._model(impl))

    def test_flash_attention_passes(self):
        sketch = SimLayerKVSketch(lazy_threshold=0.5)
        sketch.post_init_from_model(self._model("flash_attention_2"))

    def test_threshold_one_never_raises(self):
        sketch = SimLayerKVSketch(lazy_threshold=1.0)
        for impl in ("sdpa", "eager", "flash_attention_2", None):
            sketch.post_init_from_model(self._model(impl))


# ======================================================================
# Registry / ResearchAdapter wiring
# ======================================================================


class TestRegistryWiring(unittest.TestCase):
    def test_registry_resolution(self):
        from eval_harness.kv_compression.registry import get_kv_compressor, get_kv_compressor_class

        self.assertIs(get_kv_compressor_class("simlayerkv"), SimLayerKVSketch)
        sketch = get_kv_compressor("simlayerkv", lazy_threshold=0.8, n_recent=128)
        self.assertIsInstance(sketch, SimLayerKVSketch)
        self.assertAlmostEqual(sketch.lazy_threshold, 0.8)
        self.assertEqual(sketch.n_recent, 128)

    def test_build_sketch_does_not_inject_compression_ratio(self):
        from eval_harness.research_adapter import ResearchConfig, ResearchAdapter

        cfg = ResearchConfig(
            kv_compressor="simlayerkv",
            compression_ratio=0.4,
            kv_compressor_kwargs={"lazy_threshold": 0.8},
        )
        adapter = object.__new__(ResearchAdapter)
        adapter._cache_cfg = cfg
        sketch = adapter._build_kv_compressor(cfg)
        self.assertIsInstance(sketch, SimLayerKVSketch)
        self.assertAlmostEqual(sketch.lazy_threshold, 0.8)
        # compression_ratio is a read-only property, not a dataclass field, so
        # the adapter-level ratio must NOT be injected; pre-forward access raises.
        with self.assertRaises(ValueError):
            _ = sketch.compression_ratio


if __name__ == "__main__":
    unittest.main()
