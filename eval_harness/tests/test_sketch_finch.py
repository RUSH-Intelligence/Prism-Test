"""Tests for FinchSketch (port of kvpress FinchPress).

Reference oracles are independent transcriptions of the kvpress math
(finch_press.py / snapkv_press.py / key_rerotation_press.py), vendored here
because kvpress is not importable in the test environment.
"""

import logging
import math
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn
from torch.nn import functional as F
from transformers import DynamicCache

from eval_harness.sketch.sketches.finch_sketch import (
    FinchSketch,
    _compute_window_attention,
    _rerotate_keys,
)
from eval_harness.sketch.sketches.registry import (
    available_sketches,
    get_sketch,
    get_sketch_class,
)

_FINCH_LOGGER = "eval_harness.sketch.sketches.finch_sketch"


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _rope_cos_sin(positions, dim, base=10000.0):
    """Real (cos, sin) of shape [1, S, dim] plus the inv_freq used."""
    inv_freq = 1.0 / (base ** (torch.arange(0, dim // 2, dtype=torch.float32) / (dim // 2)))
    freqs = positions.float()[:, None] * inv_freq[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos()[None], emb.sin()[None], inv_freq


def _apply_rope(x, cos, sin):
    """Rotate x of shape [B, H, S, D] with (cos, sin) of shape [B, S, D]."""
    return (x * cos.unsqueeze(1)) + (_rotate_half(x) * sin.unsqueeze(1))


class _FakeAttnModule(nn.Module):
    """Minimal Llama-like attention module for sketch unit tests."""

    def __init__(self, hidden_dim=16, num_heads=4, num_kv_heads=2, head_dim=4, layer_idx=0, seed=0):
        super().__init__()
        self.config = SimpleNamespace(
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
        )
        self.num_key_value_heads = num_kv_heads
        self.head_dim = head_dim
        self.layer_idx = layer_idx
        self.is_sliding = False
        torch.manual_seed(seed)
        self.q_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, num_kv_heads * head_dim, bias=False)
        with torch.no_grad():
            self.q_proj.weight.normal_()
            self.k_proj.weight.normal_()

    def forward(self, hidden_states=None, **kwargs):
        return hidden_states, None


def _reference_window_attention(module, hidden_states, keys, window_size, cos, sin):
    """Transcription of kvpress SnapKVPress.compute_window_attention (snapkv_press.py:41-69)."""
    bsz, num_kv, k_len, head_dim = keys.shape
    num_heads = module.config.num_attention_heads
    n_rep = num_heads // module.config.num_key_value_heads

    q = module.q_proj(hidden_states[:, -window_size:])
    q = q.view(bsz, window_size, num_heads, head_dim).transpose(1, 2)
    cw, sw = cos[:, -window_size:], sin[:, -window_size:]
    q = (q * cw.unsqueeze(1)) + (_rotate_half(q) * sw.unsqueeze(1))

    ks = keys[:, :, None, :, :].expand(bsz, num_kv, n_rep, k_len, head_dim)
    ks = ks.reshape(bsz, num_kv * n_rep, k_len, head_dim)
    attn = torch.matmul(q, ks.transpose(2, 3)) / math.sqrt(head_dim)
    mask = torch.triu(torch.ones_like(attn) * float("-inf"), diagonal=k_len - window_size + 1)
    attn = attn + mask
    attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
    return attn[..., :-window_size]


def _reference_finch_compress(
    module,
    hidden_states,
    keys,
    values,
    kwargs,
    *,
    window_size,
    compression_ratio,
    chunk_length=None,
    normalize_scores=True,
    rerotate=True,
):
    """Transcription of kvpress FinchPress.score/compress (finch_press.py:56-121)
    with KeyRerotationPress.rerotate_keys (key_rerotation_press.py:50-125)."""
    bsz, num_kv, k_len, head_dim = keys.shape
    n_rep = module.config.num_attention_heads // num_kv

    cos, sin = kwargs["position_embeddings"]
    attn = _reference_window_attention(module, hidden_states, keys, window_size, cos, sin)

    if normalize_scores:
        counts = torch.arange(k_len - window_size, k_len)[None, None, :, None].to(attn.device)
        attn = attn * counts
    scores = attn.mean(dim=-2)
    scores = scores.view(bsz, num_kv, n_rep, k_len - window_size).mean(dim=2)
    scores = F.pad(scores, (0, window_size), value=scores.max().item())

    if chunk_length is None:
        n_kept = int(k_len * (1 - compression_ratio))
        indices = scores.topk(n_kept, dim=-1).indices
    else:
        assert chunk_length > window_size / (1 - compression_ratio)
        parts = []
        for i in range(0, k_len, chunk_length):
            cs = scores[:, :, i : i + chunk_length]
            nk = max(1, int(cs.shape[2] * (1 - compression_ratio)))
            parts.append(i + cs.topk(nk, dim=-1).indices)
        indices = torch.cat(parts, dim=-1)

    if rerotate:
        indices = torch.sort(indices, dim=2).values
        inv_freq = module.rotary_emb.inv_freq
        b, h, n_kept_total = indices.shape
        idx = torch.arange(0, n_kept_total, device=indices.device).unsqueeze(0)
        inv = inv_freq[None, None, :, None].float().expand(b, h, -1, 1)
        idxf = idx[:, None, :].float().expand(b, h, n_kept_total)
        delta = (idxf - indices).unsqueeze(2)
        freqs = (delta.float() * inv.float()).transpose(2, 3)
        emb = torch.cat((freqs, freqs), dim=-1)
        new_cos = emb.cos().contiguous().to(dtype=keys.dtype)
        new_sin = emb.sin().contiguous().to(dtype=keys.dtype)
        gidx = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        gk = keys.gather(2, gidx).contiguous()
        keys = (gk * new_cos) + (_rotate_half(gk) * new_sin)
        indices = gidx
    else:
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        keys = keys.gather(2, indices).contiguous()

    values = values.gather(2, indices).contiguous()
    return keys, values


def _tagged(B, H, S, D):
    """K/V tensors whose last-dim entries equal their sequence index, so kept
    positions are observable after gather."""
    t = torch.arange(S, dtype=torch.float32)[None, None, :, None].expand(B, H, S, D)
    return t.contiguous()


class TestFinchRegistry(unittest.TestCase):
    def test_finch_registered_exactly_once(self):
        self.assertIs(get_sketch_class("finch"), FinchSketch)
        names = [n for n in available_sketches() if get_sketch_class(n) is FinchSketch]
        self.assertEqual(names, ["finch"])

    def test_get_sketch_instantiates_with_ratio(self):
        sketch = get_sketch("finch", compression_ratio=0.3)
        self.assertIsInstance(sketch, FinchSketch)
        self.assertAlmostEqual(sketch.compression_ratio, 0.3)
        self.assertIsNone(sketch.window_size)
        self.assertIsNone(sketch.delimiter_token_id)
        self.assertFalse(sketch.rerotate_keys)


class TestFinchWindowDetection(unittest.TestCase):
    def _sketch(self):
        sketch = FinchSketch(compression_ratio=0.5)
        sketch.delimiter_token_id = 99
        return sketch

    def test_detects_window_and_removes_delimiter_row(self):
        sketch = self._sketch()
        ids = torch.randint(0, 90, (1, 10))
        ids[0, 7] = 99
        output = torch.randn(1, 10, 16)
        ret = sketch.embed_token_forward_hook(None, (ids,), output)
        self.assertEqual(sketch.window_size, 2)
        self.assertEqual(ret.shape, (1, 9, 16))
        self.assertTrue(torch.equal(ret, torch.cat([output[:, :7], output[:, 8:]], dim=1)))

    def test_two_delimiters_raise(self):
        sketch = self._sketch()
        ids = torch.randint(0, 90, (1, 10))
        ids[0, 3] = 99
        ids[0, 7] = 99
        with self.assertRaisesRegex(AssertionError, "one delimiter"):
            sketch.embed_token_forward_hook(None, (ids,), torch.randn(1, 10, 16))

    def test_batch_size_two_raises(self):
        sketch = self._sketch()
        ids = torch.randint(0, 90, (2, 10))
        ids[0, 7] = 99
        with self.assertRaisesRegex(AssertionError, "batch size 1"):
            sketch.embed_token_forward_hook(None, (ids,), torch.randn(2, 10, 16))

    def test_decode_step_is_noop(self):
        sketch = self._sketch()
        ids = torch.full((1, 1), 99)
        output = torch.randn(1, 1, 16)
        ret = sketch.embed_token_forward_hook(None, (ids,), output)
        self.assertIs(ret, output)
        self.assertIsNone(sketch.window_size)

    def test_missing_delimiter_is_noop(self):
        sketch = self._sketch()
        ids = torch.randint(0, 90, (1, 10))
        output = torch.randn(1, 10, 16)
        ret = sketch.embed_token_forward_hook(None, (ids,), output)
        self.assertIs(ret, output)
        self.assertIsNone(sketch.window_size)


class TestFinchScoreHandComputed(unittest.TestCase):
    """Hand-computed normalization + selection via the attentions branch.

    Window rows (queries at absolute positions 3, 4 of k_len=5, window=2) over
    3 context keys: [[0.1, 0.2, 0.7], [0.05, 0.15, 0.25]].
    """

    def _inputs(self):
        module = _FakeAttnModule(hidden_dim=4, num_heads=1, num_kv_heads=1, head_dim=4)
        attentions = torch.zeros(1, 1, 5, 5)
        attentions[0, 0, 3, :3] = torch.tensor([0.1, 0.2, 0.7])
        attentions[0, 0, 4, :3] = torch.tensor([0.05, 0.15, 0.25])
        keys = _tagged(1, 1, 5, 4)
        values = _tagged(1, 1, 5, 4)
        hidden = torch.zeros(1, 5, 4)
        return module, attentions, keys, values, hidden

    def test_normalized_scores_pinned(self):
        module, attentions, keys, values, hidden = self._inputs()
        sketch = FinchSketch(compression_ratio=0.4)
        sketch.window_size = 2
        scores = sketch.score(module, hidden, keys, values, attentions, {})
        # rows scaled by arange(3, 5) = [3, 4] -> [[0.3, 0.6, 2.1], [0.2, 0.6, 1.0]]
        expected = torch.tensor([[[0.25, 0.6, 1.55, 1.55, 1.55]]])
        torch.testing.assert_close(scores, expected, atol=1e-6, rtol=1e-6)

    def test_unnormalized_scores_pinned(self):
        module, attentions, keys, values, hidden = self._inputs()
        sketch = FinchSketch(compression_ratio=0.4, normalize_scores=False)
        sketch.window_size = 2
        scores = sketch.score(module, hidden, keys, values, attentions, {})
        expected = torch.tensor([[[0.075, 0.175, 0.475, 0.475, 0.475]]])
        torch.testing.assert_close(scores, expected, atol=1e-6, rtol=1e-6)

    def test_selection_keeps_context_top1_and_window(self):
        module, attentions, keys, values, hidden = self._inputs()
        for normalize in (True, False):
            with self.subTest(normalize_scores=normalize):
                sketch = FinchSketch(compression_ratio=0.4, normalize_scores=normalize)
                sketch.window_size = 2
                out_k, out_v = sketch.compress(module, hidden, keys, values, attentions, {})
                self.assertEqual(out_k.shape, (1, 1, 3, 4))
                self.assertEqual(out_v.shape, (1, 1, 3, 4))
                self.assertEqual(sorted(out_v[0, 0, :, 0].tolist()), [2.0, 3.0, 4.0])
                self.assertTrue(torch.equal(out_k, out_v))


class TestFinchWindowAttentionOracle(unittest.TestCase):
    def _inputs(self):
        module = _FakeAttnModule(hidden_dim=16, num_heads=4, num_kv_heads=2, head_dim=4, seed=7)
        torch.manual_seed(5)
        hidden = torch.randn(1, 6, 16)
        raw_keys = torch.randn(1, 2, 6, 4)
        values = torch.randn(1, 2, 6, 4)
        cos, sin, _ = _rope_cos_sin(torch.arange(6), 4)
        keys = _apply_rope(raw_keys, cos, sin)
        return module, hidden, keys, values, cos, sin

    def test_window_attention_matches_reference(self):
        module, hidden, keys, _, cos, sin = self._inputs()
        out = _compute_window_attention(module, hidden, keys, 2, (cos, sin))
        ref = _reference_window_attention(module, hidden, keys, 2, cos, sin)
        self.assertEqual(out.shape, (1, 4, 2, 4))
        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

    def test_gqa_scores_shape_and_group_mean(self):
        module, hidden, keys, values, cos, sin = self._inputs()
        sketch = FinchSketch(compression_ratio=0.5)
        sketch.window_size = 2
        scores = sketch.score(module, hidden, keys, values, None, {"position_embeddings": (cos, sin)})
        self.assertEqual(scores.shape, (1, 2, 6))

        w = _reference_window_attention(module, hidden, keys, 2, cos, sin)
        w = w * torch.arange(4, 6)[None, None, :, None]
        per_head = w.mean(dim=-2)  # (1, 4, 4)
        grouped = torch.stack(
            [per_head[:, 0:2].mean(dim=1), per_head[:, 2:4].mean(dim=1)], dim=1
        )  # (1, 2, 4)
        expected = F.pad(grouped, (0, 2), value=grouped.max().item())
        torch.testing.assert_close(scores, expected, atol=1e-6, rtol=1e-6)


class TestFinchRerotation(unittest.TestCase):
    def _inputs(self):
        torch.manual_seed(2)
        raw = torch.randn(1, 2, 8, 4)
        cos8, sin8, inv_freq = _rope_cos_sin(torch.arange(8), 4)
        cached = _apply_rope(raw, cos8, sin8)
        indices = torch.tensor([[[0, 2, 3, 7], [1, 2, 5, 6]]])
        module = SimpleNamespace(rotary_emb=SimpleNamespace(inv_freq=inv_freq), head_dim=4)
        return raw, cached, cos8, sin8, indices, module

    def _expected(self, raw, indices):
        gathered = raw.gather(2, indices.unsqueeze(-1).expand(-1, -1, -1, 4))
        cos4, sin4, _ = _rope_cos_sin(torch.arange(4), 4)
        return _apply_rope(gathered, cos4, sin4)

    def test_rerotated_keys_equal_fresh_rotation_at_new_positions(self):
        raw, cached, _, _, indices, module = self._inputs()
        out = _rerotate_keys(module, indices, cached)
        torch.testing.assert_close(out, self._expected(raw, indices), atol=1e-5, rtol=1e-5)

    def test_attention_scaling_commutes_single_factor_survives(self):
        raw, _, cos8, sin8, indices, module = self._inputs()
        s = 1.31
        cached_scaled = _apply_rope(raw, s * cos8, s * sin8)
        out = _rerotate_keys(module, indices, cached_scaled)
        torch.testing.assert_close(out, s * self._expected(raw, indices), atol=1e-5, rtol=1e-5)


class TestFinchChunkedSelection(unittest.TestCase):
    def test_per_chunk_topk_with_window_in_last_chunk(self):
        sketch = FinchSketch(compression_ratio=0.2, chunk_length=5)
        sketch.window_size = 2
        module = _FakeAttnModule(hidden_dim=4, num_heads=1, num_kv_heads=1, head_dim=4)
        keys = _tagged(1, 1, 10, 4)
        values = _tagged(1, 1, 10, 4)
        crafted = torch.tensor([[[0.0, 5.0, 4.0, 3.0, 2.0, 0.1, 0.2, 0.3, 9.9, 9.9]]])
        with mock.patch.object(FinchSketch, "score", return_value=crafted):
            out_k, out_v = sketch.compress(module, torch.zeros(1, 10, 4), keys, values, None, {})
        self.assertEqual(out_v.shape, (1, 1, 8, 4))
        kept = sorted(out_v[0, 0, :, 0].tolist())
        self.assertEqual(kept, [1.0, 2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 9.0])
        self.assertTrue({8.0, 9.0}.issubset(set(kept)))
        self.assertTrue(torch.equal(out_k, out_v))

    def test_chunk_length_constraint_asserted(self):
        sketch = FinchSketch(compression_ratio=0.5, chunk_length=2)
        sketch.window_size = 2
        module = _FakeAttnModule(hidden_dim=4, num_heads=1, num_kv_heads=1, head_dim=4)
        keys = torch.randn(1, 1, 4, 4)
        values = torch.randn(1, 1, 4, 4)
        with mock.patch.object(FinchSketch, "score", return_value=torch.zeros(1, 1, 4)):
            with self.assertRaises(AssertionError) as cm:
                sketch.compress(module, torch.zeros(1, 4, 4), keys, values, None, {})
        self.assertNotIn("No context keys", str(cm.exception))


class TestFinchKvpressParity(unittest.TestCase):
    """Bitwise parity of compress() against an independent kvpress transcription
    for the configs exercised in kvpress tests/presses/test_finch_press.py."""

    def setUp(self):
        logging.disable(logging.WARNING)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def _inputs(self):
        module = _FakeAttnModule(hidden_dim=16, num_heads=4, num_kv_heads=2, head_dim=4, seed=11)
        cos, sin, inv_freq = _rope_cos_sin(torch.arange(10), 4)
        module.rotary_emb = SimpleNamespace(inv_freq=inv_freq)
        torch.manual_seed(3)
        hidden = torch.randn(1, 10, 16)
        keys = _apply_rope(torch.randn(1, 2, 10, 4), cos, sin)
        values = torch.randn(1, 2, 10, 4)
        kwargs = {"position_embeddings": (cos, sin)}
        return module, hidden, keys, values, kwargs

    def test_compress_bitwise_matches_kvpress_reference(self):
        module, hidden, keys, values, kwargs = self._inputs()
        cases = [
            dict(compression_ratio=0.5, chunk_length=None, normalize_scores=True, rerotate=True),
            dict(compression_ratio=0.5, chunk_length=None, normalize_scores=True, rerotate=False),
            dict(compression_ratio=0.5, chunk_length=None, normalize_scores=False, rerotate=True),
            dict(compression_ratio=0.2, chunk_length=5, normalize_scores=True, rerotate=True),
        ]
        for case in cases:
            with self.subTest(**case):
                sketch = FinchSketch(
                    compression_ratio=case["compression_ratio"],
                    chunk_length=case["chunk_length"],
                    normalize_scores=case["normalize_scores"],
                    rerotate_keys=case["rerotate"],
                )
                sketch.window_size = 2
                out_k, out_v = sketch.compress(module, hidden, keys, values, None, kwargs)
                ref_k, ref_v = _reference_finch_compress(
                    module, hidden, keys, values, kwargs, window_size=2, **case
                )
                self.assertTrue(torch.equal(out_k, ref_k))
                self.assertTrue(torch.equal(out_v, ref_v))


class TestFinchEdgeCases(unittest.TestCase):
    def test_zero_ratio_is_noop_without_window(self):
        sketch = FinchSketch(compression_ratio=0.0)
        sketch.delimiter_token_id = 5
        self.assertIsNone(sketch.window_size)
        keys = torch.randn(1, 2, 8, 4)
        values = torch.randn(1, 2, 8, 4)
        out_k, out_v = sketch.compress(SimpleNamespace(), torch.zeros(1, 8, 4), keys, values, None, {})
        self.assertIs(out_k, keys)
        self.assertIs(out_v, values)

    def test_empty_context_raises_clear_assert(self):
        sketch = FinchSketch(compression_ratio=0.5)
        sketch.window_size = 4
        keys = torch.randn(1, 1, 4, 4)
        values = torch.randn(1, 1, 4, 4)
        module = _FakeAttnModule(hidden_dim=4, num_heads=1, num_kv_heads=1, head_dim=4)
        with self.assertRaisesRegex(AssertionError, "No context keys"):
            sketch.compress(module, torch.zeros(1, 4, 4), keys, values, None, {})

    def test_window_partially_dropped_when_n_kept_below_window(self):
        sketch = FinchSketch(compression_ratio=0.7, normalize_scores=False)
        sketch.window_size = 6
        module = _FakeAttnModule(hidden_dim=4, num_heads=1, num_kv_heads=1, head_dim=4)
        attentions = torch.zeros(1, 1, 10, 10)
        attentions[0, 0, -6:, :4] = torch.tensor([0.1, 0.2, 0.3, 0.4]).repeat(6, 1)
        keys = _tagged(1, 1, 10, 4)
        values = _tagged(1, 1, 10, 4)
        out_k, out_v = sketch.compress(module, torch.zeros(1, 10, 4), keys, values, attentions, {})
        self.assertEqual(out_v.shape, (1, 1, 3, 4))
        kept = set(out_v[0, 0, :, 0].tolist())
        # All kept entries are among the tied max scores (context argmax + 6 window pads),
        # and only 3 of the 6 max-padded window slots can survive.
        self.assertTrue(kept.issubset({3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}))
        self.assertLess(len(kept & {4.0, 5.0, 6.0, 7.0, 8.0, 9.0}), 6)

    def test_call_without_delimiter_token_id_raises(self):
        sketch = FinchSketch(compression_ratio=0.5)
        with self.assertRaisesRegex(ValueError, "delimiter token ID"):
            with sketch(mock.MagicMock()):
                pass

    def test_compress_without_window_size_raises(self):
        sketch = FinchSketch(compression_ratio=0.5)
        sketch.delimiter_token_id = 99
        ids = torch.randint(0, 90, (1, 10))
        output = torch.randn(1, 10, 16)
        ret = sketch.embed_token_forward_hook(None, (ids,), output)
        self.assertIs(ret, output)
        keys = torch.randn(1, 1, 10, 4)
        values = torch.randn(1, 1, 10, 4)
        module = _FakeAttnModule(hidden_dim=4, num_heads=1, num_kv_heads=1, head_dim=4)
        with self.assertRaisesRegex(AssertionError, "window_size must be provided"):
            sketch.compress(module, torch.zeros(1, 10, 4), keys, values, None, {})

    def test_rerotate_keys_logs_pipeline_warning(self):
        with self.assertLogs(_FINCH_LOGGER, level="WARNING") as cm:
            FinchSketch(compression_ratio=0.5, rerotate_keys=True)
        self.assertTrue(any("rerotate_keys" in msg for msg in cm.output))


class _FakeDecoderLayer(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.self_attn = _FakeAttnModule(
            hidden_dim=16, num_heads=4, num_kv_heads=2, head_dim=4, layer_idx=layer_idx, seed=layer_idx
        )


class _FakeLanguageModel(nn.Module):
    def __init__(self, inv_freq):
        super().__init__()
        self.embed_tokens = nn.Embedding(120, 16)
        self.layers = nn.ModuleList([_FakeDecoderLayer(0), _FakeDecoderLayer(1)])
        self.rotary_emb = SimpleNamespace(inv_freq=inv_freq)


class _FakeOuterModel(nn.Module):
    def __init__(self, inv_freq):
        super().__init__()
        self.model = _FakeLanguageModel(inv_freq)


class TestFinchHookIntegration(unittest.TestCase):
    """End-to-end hooks: delimiter detection on embed_tokens plus per-layer
    compression, asserting the rectangularity invariant across layers."""

    def setUp(self):
        logging.disable(logging.WARNING)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_two_layer_prefill_compresses_rectangularly(self):
        cos, sin, inv_freq = _rope_cos_sin(torch.arange(8), 4)
        model = _FakeOuterModel(inv_freq)
        sketch = FinchSketch(compression_ratio=0.5)
        sketch.delimiter_token_id = 99

        ids = torch.randint(0, 90, (1, 9))
        ids[0, 6] = 99
        cache = DynamicCache()

        with sketch(model):
            emb = model.model.embed_tokens(ids)
            self.assertEqual(emb.shape, (1, 8, 16))
            self.assertEqual(sketch.window_size, 2)

            torch.manual_seed(0)
            for i, layer in enumerate(model.model.layers):
                keys = torch.randn(1, 2, 8, 4)
                values = torch.randn(1, 2, 8, 4)
                cache.update(keys, values, i)
                layer.self_attn(
                    hidden_states=emb,
                    past_key_values=cache,
                    cache_position=torch.arange(8),
                    position_embeddings=(cos, sin),
                )

        lengths = [cache.layers[i].keys.shape[2] for i in range(2)]
        self.assertEqual(lengths, [4, 4])
        for i in range(2):
            self.assertEqual(cache.layers[i].keys.shape, (1, 2, 4, 4))
            self.assertEqual(cache.layers[i].values.shape, (1, 2, 4, 4))

        # Hooks are removed on context exit: the delimiter row is no longer dropped.
        emb_after = model.model.embed_tokens(ids)
        self.assertEqual(emb_after.shape, (1, 9, 16))


if __name__ == "__main__":
    unittest.main()
