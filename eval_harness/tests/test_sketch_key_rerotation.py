"""Unit tests for KeyRerotationSketch (port of kvpress 0.5.1 KeyRerotationPress).

No model loading; fake attention modules only. The kvpress selection +
delta-rotation math is transcribed inline as a reference oracle (kvpress is not
importable in prism_env), following the prune-raw-then-rotate reference of
kvpress tests/presses/test_key_rerotation_press_rope.py.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import torch

from eval_harness.kv_compression.compressors.key_rerotation_sketch import KeyRerotationSketch
from eval_harness.kv_compression.compressors.knorm_sketch import KnormSketch
from eval_harness.kv_compression.base import ScorerKVCompressor


# ----------------------------------------------------------------------
# Local reference helpers (HF llama RoPE convention: cat-doubled freqs)
# ----------------------------------------------------------------------


def _inv_freq(head_dim: int, base: float = 10000.0) -> torch.Tensor:
    half = head_dim // 2
    return 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))


def _ref_rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _ref_trig(positions: torch.Tensor, inv_freq: torch.Tensor, scale: float = 1.0):
    freqs = positions.to(torch.float32)[:, None] * inv_freq[None, :].float()  # [S, d/2]
    emb = torch.cat((freqs, freqs), dim=-1)  # [S, d]
    return emb.cos() * scale, emb.sin() * scale


def _ref_rotate(x: torch.Tensor, positions: torch.Tensor, inv_freq: torch.Tensor, scale: float = 1.0):
    """Rotate ``x`` [..., S, d] at the given absolute positions; trig amplitude
    ``scale`` mimics HF's baked-in attention_scaling (cos = emb.cos() * s)."""
    cos, sin = _ref_trig(positions, inv_freq, scale)
    return x * cos + _ref_rotate_half(x) * sin


def _ref_rerotated_keys(
    k_raw: torch.Tensor,
    sorted_indices: torch.Tensor,
    inv_freq: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """Prune-raw-then-rotate reference: gather raw keys at the sorted kept
    indices, then rotate at the new contiguous positions 0..n_kept-1 (with the
    same trig amplitude the model bakes into its cache)."""
    B, H, _, D = k_raw.shape
    n_kept = sorted_indices.shape[-1]
    gathered = k_raw.gather(2, sorted_indices.unsqueeze(-1).expand(B, H, n_kept, D))
    return _ref_rotate(gathered, torch.arange(n_kept), inv_freq, scale)


def _kvpress_selection(scores: torch.Tensor, compression_ratio: float) -> torch.Tensor:
    """Verbatim transcription of the kvpress selection
    (key_rerotation_press.py:143-146): floor n_kept, topk, ascending sort."""
    q_len = scores.shape[-1]
    n_kept = int(q_len * (1 - compression_ratio))
    indices = scores.topk(n_kept, dim=-1).indices
    return torch.sort(indices, dim=2).values


# ----------------------------------------------------------------------
# Fakes and stub scorers
# ----------------------------------------------------------------------


def _fake_module(head_dim: int = 4, inv_freq: Optional[torch.Tensor] = None, layer_idx: int = 0, **extra):
    rotary_emb = SimpleNamespace(inv_freq=inv_freq if inv_freq is not None else _inv_freq(head_dim))
    return SimpleNamespace(head_dim=head_dim, rotary_emb=rotary_emb, layer_idx=layer_idx, **extra)


@dataclass
class _FixedScoreSketch(ScorerKVCompressor):
    """Stub scorer returning a fixed [B, H_kv, S] score tensor."""

    fixed_scores: Optional[torch.Tensor] = None

    def score(self, module, hidden_states, keys, values, attentions, kwargs):
        return self.fixed_scores


@dataclass
class _RecordingScorer(ScorerKVCompressor):
    """Stub scorer recording the keys shape it receives; seeded random scores."""

    seed: int = 0
    received_keys_shape: Optional[tuple] = None

    def score(self, module, hidden_states, keys, values, attentions, kwargs):
        self.received_keys_shape = tuple(keys.shape)
        generator = torch.Generator().manual_seed(self.seed)
        return torch.rand(*keys.shape[:-1], generator=generator)


@dataclass
class _PostInitRecorder(ScorerKVCompressor):
    saw_model: Optional[object] = None

    def post_init_from_model(self, model):
        self.saw_model = model


class _FakeCacheLayer:
    def __init__(self, keys, values):
        self.keys = keys
        self.values = values


class _FakeCache:
    def __init__(self, layers):
        self.layers = layers


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


class TestZeroRatioNoOp(unittest.TestCase):
    def test_zero_ratio_returns_identical_objects(self):
        torch.manual_seed(0)
        keys = torch.randn(1, 2, 8, 4)
        values = torch.randn(1, 2, 8, 4)
        sketch = KeyRerotationSketch(press=KnormSketch(compression_ratio=0.0))
        # Module deliberately has NO rotary_emb: the early return at ratio 0
        # must fire before any rerotation machinery is touched.
        module = SimpleNamespace(head_dim=4)
        out_keys, out_values = sketch.compress(module, None, keys, values, None, {})
        self.assertIs(out_keys, keys)
        self.assertIs(out_values, values)


class TestExactMath(unittest.TestCase):
    def test_hand_computed_selection_and_rerotation(self):
        inv_freq = torch.tensor([1.0, 0.01])
        module = _fake_module(head_dim=4, inv_freq=inv_freq)
        k_raw = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4) / 7.0 - 1.0
        values = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4) * 10.0
        keys_in = _ref_rotate(k_raw, torch.arange(4), inv_freq)

        scores = torch.tensor([[[0.0, 10.0, 0.0, 20.0]]])  # topk order [3, 1] -> sorted [1, 3]
        sketch = KeyRerotationSketch(press=_FixedScoreSketch(compression_ratio=0.5, fixed_scores=scores))
        out_keys, out_values = sketch.compress(module, None, keys_in.clone(), values, None, {})

        self.assertEqual(out_keys.shape, (1, 1, 2, 4))
        # Path 1: explicit delta rotation applied to the ROTATED inputs.
        # Kept index 1 -> new pos 0, delta = -1; kept index 3 -> new pos 1, delta = -2.
        row0 = _ref_rotate(keys_in[:, :, 1], torch.tensor([-1.0]), inv_freq)
        row1 = _ref_rotate(keys_in[:, :, 3], torch.tensor([-2.0]), inv_freq)
        expected_delta = torch.stack([row0, row1], dim=2)
        torch.testing.assert_close(out_keys, expected_delta, atol=1e-6, rtol=1e-6)

        # Path 2: independent prune-raw-then-rotate oracle.
        sorted_idx = torch.tensor([[[1, 3]]])
        expected_oracle = _ref_rerotated_keys(k_raw, sorted_idx, inv_freq)
        torch.testing.assert_close(out_keys, expected_oracle, atol=1e-6, rtol=1e-6)

        # New position 0 means R(0) = identity: row 0 recovers the raw key.
        torch.testing.assert_close(out_keys[:, :, 0], k_raw[:, :, 1], atol=1e-6, rtol=1e-6)

        # Values are gathered (never rotated) with the SAME sorted indices.
        self.assertTrue(torch.equal(out_values, values[:, :, [1, 3]]))

    def test_causal_order_sorting_is_replicated(self):
        inv_freq = torch.tensor([1.0, 0.01])
        module = _fake_module(head_dim=4, inv_freq=inv_freq)
        torch.manual_seed(1)
        k_raw = torch.randn(1, 1, 4, 4)
        values = torch.randn(1, 1, 4, 4)
        keys_in = _ref_rotate(k_raw, torch.arange(4), inv_freq)

        # topk order is [2, 0]; the implementation must sort to [0, 2].
        scores = torch.tensor([[[0.5, 0.1, 0.9, 0.2]]])
        sketch = KeyRerotationSketch(press=_FixedScoreSketch(compression_ratio=0.5, fixed_scores=scores))
        out_keys, out_values = sketch.compress(module, None, keys_in.clone(), values, None, {})

        self.assertTrue(torch.equal(out_values, values[:, :, [0, 2]]))
        # Index 0 stays at position 0 (delta 0 -> unchanged); index 2 moves to position 1.
        torch.testing.assert_close(out_keys[:, :, 0], keys_in[:, :, 0], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(
            out_keys[:, :, 1], _ref_rotate(k_raw[:, :, 2], torch.tensor([1.0]), inv_freq),
            atol=1e-6, rtol=1e-6,
        )

    def test_per_head_distinct_selection_uniform_count(self):
        D = 4
        inv_freq = _inv_freq(D)
        module = _fake_module(head_dim=D, inv_freq=inv_freq)
        torch.manual_seed(2)
        k_raw = torch.randn(1, 2, 4, D)
        values = torch.randn(1, 2, 4, D)
        keys_in = _ref_rotate(k_raw, torch.arange(4), inv_freq)

        scores = torch.tensor([[
            [10.0, 0.0, 0.0, 20.0],  # head 0 keeps sorted [0, 3]
            [0.0, 10.0, 20.0, 0.0],  # head 1 keeps sorted [1, 2]
        ]])
        sketch = KeyRerotationSketch(press=_FixedScoreSketch(compression_ratio=0.5, fixed_scores=scores))
        out_keys, out_values = sketch.compress(module, None, keys_in.clone(), values, None, {})

        self.assertEqual(out_keys.shape, (1, 2, 2, D))
        self.assertEqual(out_values.shape, (1, 2, 2, D))
        self.assertTrue(torch.equal(out_values[:, 0], values[:, 0][:, [0, 3]]))
        self.assertTrue(torch.equal(out_values[:, 1], values[:, 1][:, [1, 2]]))

        sorted_idx = torch.tensor([[[0, 3], [1, 2]]])
        expected = _ref_rerotated_keys(k_raw, sorted_idx, inv_freq)
        torch.testing.assert_close(out_keys, expected, atol=1e-6, rtol=1e-6)

    def test_small_seq_single_kept_key(self):
        D = 4
        inv_freq = _inv_freq(D)
        module = _fake_module(head_dim=D, inv_freq=inv_freq)
        torch.manual_seed(3)
        k_raw = torch.randn(1, 1, 3, D)
        values = torch.randn(1, 1, 3, D)
        keys_in = _ref_rotate(k_raw, torch.arange(3), inv_freq)

        scores = torch.tensor([[[0.0, 10.0, 0.0]]])
        sketch = KeyRerotationSketch(press=_FixedScoreSketch(compression_ratio=0.5, fixed_scores=scores))
        out_keys, out_values = sketch.compress(module, None, keys_in.clone(), values, None, {})

        # n_kept = int(3 * 0.5) = 1; the single key is re-rotated to position 0 = raw.
        self.assertEqual(out_keys.shape, (1, 1, 1, D))
        torch.testing.assert_close(out_keys[:, :, 0], k_raw[:, :, 1], atol=1e-6, rtol=1e-6)
        self.assertTrue(torch.equal(out_values, values[:, :, [1]]))

    def test_extreme_ratio_yields_empty_cache(self):
        # kvpress-faithful: int(3 * 0.1) = 0 kept entries, no clamping.
        D = 4
        module = _fake_module(head_dim=D)
        keys_in = torch.randn(1, 1, 3, D)
        values = torch.randn(1, 1, 3, D)
        scores = torch.tensor([[[1.0, 2.0, 3.0]]])
        sketch = KeyRerotationSketch(press=_FixedScoreSketch(compression_ratio=0.9, fixed_scores=scores))
        out_keys, out_values = sketch.compress(module, None, keys_in, values, None, {})
        self.assertEqual(out_keys.shape, (1, 1, 0, D))
        self.assertEqual(out_values.shape, (1, 1, 0, D))


class TestReferenceOracle(unittest.TestCase):
    """Transcription of kvpress tests/presses/test_key_rerotation_press_rope.py:
    press output on rotated keys == gather raw keys at the stored sorted indices,
    then rotate at the new contiguous positions."""

    def _run(self, scale: float, dtype: torch.dtype = torch.float32):
        B, H_kv, S, D = 2, 2, 64, 8
        ratio = 0.5
        inv_freq = _inv_freq(D)
        module = _fake_module(head_dim=D, inv_freq=inv_freq)
        torch.manual_seed(42)
        k_raw = torch.randn(B, H_kv, S, D)
        values = torch.randn(B, H_kv, S, D)
        keys_in = _ref_rotate(k_raw, torch.arange(S), inv_freq, scale)

        scorer = _RecordingScorer(compression_ratio=ratio, seed=7)
        sketch = KeyRerotationSketch(press=scorer)
        out_keys, out_values = sketch.compress(
            module, None, keys_in.clone().to(dtype), values.to(dtype), None, {},
        )

        scores = torch.rand(B, H_kv, S, generator=torch.Generator().manual_seed(7))
        sorted_idx = _kvpress_selection(scores, ratio)
        self.assertEqual(sorted_idx.shape, (B, H_kv, 32))
        expected_keys = _ref_rerotated_keys(k_raw, sorted_idx, inv_freq, scale)
        expected_values = values.gather(2, sorted_idx.unsqueeze(-1).expand(B, H_kv, 32, D))
        return out_keys, out_values, expected_keys, expected_values

    def test_unscaled_rope_matches_prune_raw_then_rotate(self):
        out_keys, out_values, expected_keys, expected_values = self._run(scale=1.0)
        torch.testing.assert_close(out_keys, expected_keys, atol=1e-5, rtol=1e-5)
        self.assertTrue(torch.equal(out_values, expected_values))

    def test_scaled_rope_single_factor_of_s(self):
        """attention_scaling regression guard (yarn-equivalent of the kvpress
        test): cached keys carry s*R(p)*k_raw; the delta rotation is built from
        raw inv_freq (unit trig), so the output must be s*R(new)*k_raw —
        exactly ONE factor of s. Fails on the s^2 undo/redo defect and on any
        erroneous divide-by-s 'normalization'."""
        for scale in (1.31, 0.866):
            with self.subTest(scale=scale):
                out_keys, _, expected_keys, _ = self._run(scale=scale)
                torch.testing.assert_close(out_keys, expected_keys, atol=1e-5, rtol=1e-5)
                # Discrimination: the s^2 (undo/redo with scaled trig) and the
                # unit-amplitude (divide-by-s) variants must NOT match.
                self.assertFalse(
                    torch.allclose(out_keys, expected_keys * scale, atol=1e-3, rtol=1e-3)
                )
                self.assertFalse(
                    torch.allclose(out_keys, expected_keys / scale, atol=1e-3, rtol=1e-3)
                )

    def test_bfloat16_dtype_preserved_and_close_to_fp32_reference(self):
        # Trig is computed in float32 internally and cast back at the end
        # (key_rerotation_press.py:88-96): bf16 output, fp32-reference close.
        out_keys, out_values, expected_keys, expected_values = self._run(
            scale=1.0, dtype=torch.bfloat16,
        )
        self.assertEqual(out_keys.dtype, torch.bfloat16)
        self.assertEqual(out_values.dtype, torch.bfloat16)
        torch.testing.assert_close(
            out_keys.float(), expected_keys, atol=1e-1, rtol=1e-1,
        )
        self.assertTrue(torch.equal(out_values, expected_values.to(torch.bfloat16)))


class TestGQAPassThrough(unittest.TestCase):
    def test_wrapper_operates_on_kv_heads_only(self):
        B, H_kv, S, D = 1, 2, 8, 4
        module = _fake_module(
            head_dim=D,
            config=SimpleNamespace(num_attention_heads=4, num_key_value_heads=H_kv),
        )
        scorer = _RecordingScorer(compression_ratio=0.5, seed=11)
        sketch = KeyRerotationSketch(press=scorer)
        keys = _ref_rotate(torch.randn(B, H_kv, S, D), torch.arange(S), module.rotary_emb.inv_freq)
        values = torch.randn(B, H_kv, S, D)
        out_keys, out_values = sketch.compress(module, None, keys, values, None, {})

        # The wrapped scorer sees the KV-head axis untouched; the wrapper never
        # repeats to query heads (queries are solely the scorer's concern).
        self.assertEqual(scorer.received_keys_shape, (B, H_kv, S, D))
        self.assertEqual(out_keys.shape, (B, H_kv, 4, D))
        self.assertEqual(out_values.shape, (B, H_kv, 4, D))


class TestForwardHookIntegration(unittest.TestCase):
    def _layer_setup(self, S=16, H_kv=2, D=4, ratio=0.25, seed=5):
        torch.manual_seed(seed)
        inv_freq = _inv_freq(D)
        modules = [_fake_module(head_dim=D, inv_freq=inv_freq, layer_idx=i) for i in range(2)]
        layers = [
            _FakeCacheLayer(torch.randn(1, H_kv, S, D), torch.randn(1, H_kv, S, D))
            for _ in range(2)
        ]
        cache = _FakeCache(layers)
        sketch = KeyRerotationSketch(press=KnormSketch(compression_ratio=ratio))
        return sketch, modules, cache

    def test_prefill_hook_compresses_all_layers_uniformly(self):
        S = 16
        sketch, modules, cache = self._layer_setup(S=S)
        for module in modules:
            kwargs = {
                "hidden_states": torch.randn(1, S, 8),
                "past_key_values": cache,
                "cache_position": torch.arange(S),
            }
            output = (torch.randn(1, S, 8), None)
            result = sketch.forward_hook(module, [], kwargs, output)
            self.assertIs(result, output)
        # Rectangularity: every hooked layer holds int(16 * 0.75) = 12 entries.
        for layer in cache.layers:
            self.assertEqual(layer.keys.shape, (1, 2, 12, 4))
            self.assertEqual(layer.values.shape, (1, 2, 12, 4))

    def test_decode_step_is_noop(self):
        S = 16
        sketch, modules, cache = self._layer_setup(S=S)
        for module in modules:
            kwargs = {
                "hidden_states": torch.randn(1, S, 8),
                "past_key_values": cache,
                "cache_position": torch.arange(S),
            }
            sketch.forward_hook(module, [], kwargs, (torch.randn(1, S, 8), None))
        compressed = [(layer.keys, layer.values) for layer in cache.layers]

        for module in modules:
            kwargs = {
                "hidden_states": torch.randn(1, 1, 8),
                "past_key_values": cache,
                "cache_position": torch.tensor([13]),
            }
            output = (torch.randn(1, 1, 8), None)
            result = sketch.forward_hook(module, [], kwargs, output)
            self.assertIs(result, output)
        for layer, (keys, values) in zip(cache.layers, compressed):
            self.assertIs(layer.keys, keys)
            self.assertIs(layer.values, values)


class TestProxyAndRegistry(unittest.TestCase):
    def test_compression_ratio_proxies_to_wrapped_sketch(self):
        inner = KnormSketch(compression_ratio=0.3)
        sketch = KeyRerotationSketch(press=inner)
        self.assertAlmostEqual(sketch.compression_ratio, 0.3)
        sketch.compression_ratio = 0.6
        self.assertAlmostEqual(inner.compression_ratio, 0.6)
        self.assertAlmostEqual(sketch.compression_ratio, 0.6)

    def test_requires_scorer_sketch(self):
        with self.assertRaises(AssertionError):
            KeyRerotationSketch(press=object())

    def test_post_init_from_model_delegates(self):
        inner = _PostInitRecorder(compression_ratio=0.2)
        sketch = KeyRerotationSketch(press=inner)
        sentinel = object()
        sketch.post_init_from_model(sentinel)
        self.assertIs(inner.saw_model, sentinel)

    def test_registered_as_key_rerotation(self):
        from eval_harness.kv_compression.registry import (
            available_kv_compressors,
            get_kv_compressor,
            get_kv_compressor_class,
        )

        self.assertIn("key_rerotation", available_kv_compressors())
        self.assertIs(get_kv_compressor_class("key_rerotation"), KeyRerotationSketch)
        sketch = get_kv_compressor("key_rerotation", press=KnormSketch(compression_ratio=0.4))
        self.assertIsInstance(sketch, KeyRerotationSketch)
        self.assertAlmostEqual(sketch.compression_ratio, 0.4)


if __name__ == "__main__":
    unittest.main()
