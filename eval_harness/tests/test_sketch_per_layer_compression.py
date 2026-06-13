"""Tests for PerLayerCompressionSketch (port of kvpress PerLayerCompressionPress).

GPU-free: fake attention modules + fake caches; expectations are hand-computed
or transcribed in-test from the kvpress math (ScorerPress.compress + KnormPress
scoring), including the kvpress parity pin from
kvpress tests/test_per_layer_compression_press.py (ratios [0.1, 1.0], S=256).
"""

from __future__ import annotations

import logging
import unittest
from dataclasses import fields
from types import SimpleNamespace

import torch
from torch import nn

from eval_harness.kv_compression.base import KVCompressor
from eval_harness.kv_compression.compressors.knorm_sketch import KnormSketch
from eval_harness.kv_compression.compressors.per_layer_compression_sketch import PerLayerCompressionSketch
from eval_harness.kv_compression.compressors.random_sketch import RandomSketch
from eval_harness.kv_compression.base import ScorerKVCompressor

LOGGER_NAME = "eval_harness.kv_compression.compressors.per_layer_compression_sketch"


# ======================================================================
# Local fakes and helpers
# ======================================================================


class _FakeAttnModule(nn.Module):
    """Minimal attention module: KnormSketch only needs head_dim + layer_idx."""

    def __init__(self, head_dim=6, layer_idx=0):
        super().__init__()
        self.head_dim = head_dim
        self.layer_idx = layer_idx


class _FakeCacheLayer:
    def __init__(self, keys, values):
        self.keys = keys
        self.values = values


class _FakeCache:
    def __init__(self, layers):
        self.layers = layers


class _NoRatioPress(ScorerKVCompressor):
    """ScorerKVCompressor whose own __init__ signature lacks compression_ratio."""

    def __init__(self):
        super().__init__(compression_ratio=0.0)

    def score(self, module, hidden_states, keys, values, attentions, kwargs):
        return -keys.norm(dim=-1)


def _make(press, ratios, **kw):
    """Build the wrapper with the (verbatim-kept) experimental warning silenced."""
    logger = logging.getLogger(LOGGER_NAME)
    prev = logger.disabled
    logger.disabled = True
    try:
        return PerLayerCompressionSketch(press=press, compression_ratios=ratios, **kw)
    finally:
        logger.disabled = prev


def _run_prefill_hook(sketch, module, cache, B, S, hidden=4):
    kwargs = {
        "hidden_states": torch.randn(B, S, hidden),
        "past_key_values": cache,
        "cache_position": torch.arange(S),
    }
    output = (torch.randn(B, S, hidden), None)
    return sketch.forward_hook(module, [], kwargs, output)


def _knorm_reference_gather(keys, values, n_kept, head_dim):
    """In-test transcription of ScorerPress.compress with KnormPress scoring."""
    indices = (-keys.norm(dim=-1)).topk(n_kept, dim=-1).indices
    indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
    return keys.gather(2, indices).contiguous(), values.gather(2, indices).contiguous()


def _fake_model(num_hidden_layers=2, attn_implementation="sdpa"):
    cfg = SimpleNamespace(num_hidden_layers=num_hidden_layers)
    if attn_implementation is not None:
        cfg._attn_implementation = attn_implementation
    return SimpleNamespace(config=cfg)


# ======================================================================
# Construction, validation, properties
# ======================================================================


class TestConstructionAndProperties(unittest.TestCase):
    def test_experimental_warning_logged_at_construction(self):
        with self.assertLogs(LOGGER_NAME, level="WARNING") as cm:
            PerLayerCompressionSketch(press=KnormSketch(), compression_ratios=[0.1])
        joined = "\n".join(cm.output)
        self.assertIn("experimental", joined)
        self.assertIn("flash attention", joined)

    def test_non_scorer_press_rejected(self):
        with self.assertRaises(AssertionError):
            _make(KVCompressor(), [0.1])

    def test_press_without_compression_ratio_in_signature_rejected(self):
        with self.assertRaises(AssertionError):
            _make(_NoRatioPress(), [0.1])

    def test_compression_ratio_is_mean(self):
        sketch = _make(KnormSketch(), [0.2, 0.4])
        self.assertEqual(sketch.compression_ratio, (0.2 + 0.4) / 2)

    def test_compression_ratio_setter_raises(self):
        sketch = _make(KnormSketch(), [0.2, 0.4])
        with self.assertRaisesRegex(AttributeError, "compression ratio cannot be set"):
            sketch.compression_ratio = 0.5

    def test_press_resolved_from_registry_name(self):
        sketch = _make("knorm", [0.5])
        self.assertIsInstance(sketch.press, KnormSketch)

    def test_press_kwargs_forwarded_to_named_press(self):
        sketch = _make("random", [0.5], press_kwargs={"seed": 3})
        self.assertIsInstance(sketch.press, RandomSketch)
        self.assertEqual(sketch.press.seed, 3)

    def test_press_kwargs_with_instance_raises(self):
        with self.assertRaisesRegex(ValueError, "press_kwargs"):
            _make(KnormSketch(), [0.5], press_kwargs={"seed": 1})

    def test_unknown_press_name_raises(self):
        with self.assertRaises(ValueError):
            _make("definitely_not_a_sketch", [0.5])


# ======================================================================
# kvpress parity pin (tests/test_per_layer_compression_press.py)
# ======================================================================


class TestKvpressParityPin(unittest.TestCase):
    def test_ratios_point1_and_1_match_kvpress_shapes_and_content(self):
        torch.manual_seed(0)
        B, H, S, D = 5, 2, 256, 6
        keys = [torch.randn(B, H, S, D) for _ in range(2)]
        values = [torch.randn(B, H, S, D) for _ in range(2)]
        cache = _FakeCache(
            [_FakeCacheLayer(keys[0], values[0]), _FakeCacheLayer(keys[1], values[1])]
        )
        sketch = _make(KnormSketch(), [0.1, 1.0])

        for layer_idx in (0, 1):
            module = _FakeAttnModule(head_dim=D, layer_idx=layer_idx)
            _run_prefill_hook(sketch, module, cache, B=B, S=S)

        # int(256 * (1 - 0.1)) = 230; ratio 1.0 bypasses the <1 assert -> 0 kept.
        self.assertEqual(tuple(cache.layers[0].keys.shape), (5, 2, 230, 6))
        self.assertEqual(tuple(cache.layers[0].values.shape), (5, 2, 230, 6))
        self.assertEqual(tuple(cache.layers[1].keys.shape), (5, 2, 0, 6))
        self.assertEqual(tuple(cache.layers[1].values.shape), (5, 2, 0, 6))

        exp_k, exp_v = _knorm_reference_gather(keys[0], values[0], 230, D)
        self.assertTrue(torch.equal(cache.layers[0].keys, exp_k))
        self.assertTrue(torch.equal(cache.layers[0].values, exp_v))


# ======================================================================
# Selection behavior
# ======================================================================


class TestSelection(unittest.TestCase):
    def test_exact_hand_computed_selection(self):
        keys = torch.tensor([[1.0, 0.0], [0.0, 2.0], [3.0, 0.0], [0.0, 4.0]]).view(1, 1, 4, 2)
        values = torch.tensor([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0], [40.0, 41.0]]).view(1, 1, 4, 2)
        cache = _FakeCache([_FakeCacheLayer(keys, values)])
        sketch = _make(KnormSketch(), [0.5])
        module = _FakeAttnModule(head_dim=2, layer_idx=0)

        _run_prefill_hook(sketch, module, cache, B=1, S=4)

        # Norms 1,2,3,4 -> knorm scores -1 > -2 > -3 > -4; n_kept = int(4*0.5) = 2
        # -> kept positions [0, 1] in topk (descending-score) order.
        expected_keys = torch.tensor([[1.0, 0.0], [0.0, 2.0]]).view(1, 1, 2, 2)
        expected_values = torch.tensor([[10.0, 11.0], [20.0, 21.0]]).view(1, 1, 2, 2)
        self.assertTrue(torch.equal(cache.layers[0].keys, expected_keys))
        self.assertTrue(torch.equal(cache.layers[0].values, expected_values))

    def test_per_layer_ratio_indexed_by_layer_idx_not_call_order(self):
        B, H, S, D = 1, 1, 8, 2
        # keys[..., s, :] = s + 1 -> norms strictly increasing with position,
        # so knorm keeps the first n_kept positions, in ascending order.
        base = (torch.arange(S, dtype=torch.float32) + 1).view(1, 1, S, 1).expand(B, H, S, D).clone()
        keys = [base.clone(), base.clone()]
        values = [base.clone() * 10, base.clone() * 100]
        cache = _FakeCache(
            [_FakeCacheLayer(keys[0], values[0]), _FakeCacheLayer(keys[1], values[1])]
        )
        sketch = _make(KnormSketch(), [0.25, 0.75])

        # layer_idx=1 module first to prove indexing uses module.layer_idx.
        _run_prefill_hook(sketch, _FakeAttnModule(head_dim=D, layer_idx=1), cache, B=B, S=S)
        _run_prefill_hook(sketch, _FakeAttnModule(head_dim=D, layer_idx=0), cache, B=B, S=S)

        self.assertEqual(tuple(cache.layers[0].keys.shape), (B, H, 6, D))  # int(8*0.75)
        self.assertEqual(tuple(cache.layers[1].keys.shape), (B, H, 2, D))  # int(8*0.25)
        self.assertTrue(torch.equal(cache.layers[0].keys, base[:, :, :6]))
        self.assertTrue(torch.equal(cache.layers[0].values, base[:, :, :6] * 10))
        self.assertTrue(torch.equal(cache.layers[1].keys, base[:, :, :2]))
        self.assertTrue(torch.equal(cache.layers[1].values, base[:, :, :2] * 100))

    def test_gqa_per_head_selection_rectangular(self):
        B, H_kv, S, D = 1, 2, 6, 2
        # Head 0: norms ascending with position (keep 0,1,2);
        # head 1: norms descending (keep 5,4,3) -> different indices per head,
        # same n_kept per head (rectangular across heads).
        h0 = (torch.arange(S, dtype=torch.float32) + 1).view(S, 1).expand(S, D)
        h1 = (torch.arange(S, 0, -1, dtype=torch.float32)).view(S, 1).expand(S, D)
        keys = torch.stack([h0, h1]).unsqueeze(0).clone()
        values = torch.arange(B * H_kv * S * D, dtype=torch.float32).view(B, H_kv, S, D)
        cache = _FakeCache([_FakeCacheLayer(keys.clone(), values.clone())])
        sketch = _make(KnormSketch(), [0.5])
        module = _FakeAttnModule(head_dim=D, layer_idx=0)

        _run_prefill_hook(sketch, module, cache, B=B, S=S)

        self.assertEqual(tuple(cache.layers[0].keys.shape), (B, H_kv, 3, D))
        exp_k, exp_v = _knorm_reference_gather(keys, values, 3, D)
        self.assertTrue(torch.equal(cache.layers[0].keys, exp_k))
        self.assertTrue(torch.equal(cache.layers[0].values, exp_v))
        kept0 = set((-keys.norm(dim=-1)).topk(3, dim=-1).indices[0, 0].tolist())
        kept1 = set((-keys.norm(dim=-1)).topk(3, dim=-1).indices[0, 1].tolist())
        self.assertEqual(kept0, {0, 1, 2})
        self.assertEqual(kept1, {3, 4, 5})

    def test_edge_s1_ratio_half_keeps_zero(self):
        keys = torch.randn(1, 1, 1, 2)
        values = torch.randn(1, 1, 1, 2)
        cache = _FakeCache([_FakeCacheLayer(keys, values)])
        sketch = _make(KnormSketch(), [0.5])
        module = _FakeAttnModule(head_dim=2, layer_idx=0)

        _run_prefill_hook(sketch, module, cache, B=1, S=1)

        self.assertEqual(tuple(cache.layers[0].keys.shape), (1, 1, 0, 2))
        self.assertEqual(tuple(cache.layers[0].values.shape), (1, 1, 0, 2))

    def test_mixed_extremes_zero_and_one_in_one_prefill(self):
        B, H, S, D = 1, 1, 4, 2
        keys = [torch.randn(B, H, S, D) for _ in range(2)]
        values = [torch.randn(B, H, S, D) for _ in range(2)]
        cache = _FakeCache(
            [_FakeCacheLayer(keys[0], values[0]), _FakeCacheLayer(keys[1], values[1])]
        )
        sketch = _make(KnormSketch(), [0.0, 1.0])
        self.assertEqual(sketch.compression_ratio, 0.5)

        for layer_idx in (0, 1):
            _run_prefill_hook(sketch, _FakeAttnModule(head_dim=D, layer_idx=layer_idx), cache, B=B, S=S)

        # ratio 0 -> inner compress early-returns the original tensors.
        self.assertIs(cache.layers[0].keys, keys[0])
        self.assertIs(cache.layers[0].values, values[0])
        # ratio 1 -> layer emptied.
        self.assertEqual(tuple(cache.layers[1].keys.shape), (B, H, 0, D))
        self.assertEqual(tuple(cache.layers[1].values.shape), (B, H, 0, D))


# ======================================================================
# No-op paths, ratio restore, decode gate
# ======================================================================


class TestNoopAndRestore(unittest.TestCase):
    def test_zero_ratios_are_noop_and_restore_inner_ratio(self):
        B, H, S, D = 1, 2, 8, 4
        keys = [torch.randn(B, H, S, D) for _ in range(2)]
        values = [torch.randn(B, H, S, D) for _ in range(2)]
        cache = _FakeCache(
            [_FakeCacheLayer(keys[0], values[0]), _FakeCacheLayer(keys[1], values[1])]
        )
        inner = KnormSketch(compression_ratio=0.37)
        sketch = _make(inner, [0.0, 0.0])

        for layer_idx in (0, 1):
            _run_prefill_hook(sketch, _FakeAttnModule(head_dim=D, layer_idx=layer_idx), cache, B=B, S=S)

        for i in range(2):
            self.assertIs(cache.layers[i].keys, keys[i])
            self.assertIs(cache.layers[i].values, values[i])
            self.assertTrue(torch.equal(cache.layers[i].keys, keys[i]))
        self.assertEqual(inner.compression_ratio, 0.37)

    def test_inner_ratio_restored_after_normal_hook(self):
        inner = KnormSketch(compression_ratio=0.0)
        sketch = _make(inner, [0.7])
        keys, values = torch.randn(1, 1, 10, 2), torch.randn(1, 1, 10, 2)
        cache = _FakeCache([_FakeCacheLayer(keys, values)])

        _run_prefill_hook(sketch, _FakeAttnModule(head_dim=2, layer_idx=0), cache, B=1, S=10)

        self.assertEqual(tuple(cache.layers[0].keys.shape), (1, 1, 3, 2))  # int(10*0.3)
        self.assertEqual(inner.compression_ratio, 0.0)

    def test_inner_ratio_restored_when_inner_score_raises(self):
        inner = KnormSketch(compression_ratio=0.0)
        sketch = _make(inner, [0.7])
        keys, values = torch.randn(1, 1, 10, 2), torch.randn(1, 1, 10, 2)
        cache = _FakeCache([_FakeCacheLayer(keys, values)])

        def _boom(*args, **kwargs):
            raise RuntimeError("boom")

        inner.score = _boom
        with self.assertRaisesRegex(RuntimeError, "boom"):
            _run_prefill_hook(sketch, _FakeAttnModule(head_dim=2, layer_idx=0), cache, B=1, S=10)
        # try/finally deviation: inner ratio restored despite the exception.
        self.assertEqual(inner.compression_ratio, 0.0)
        self.assertIs(cache.layers[0].keys, keys)

    def test_decode_step_is_noop_even_with_high_ratio(self):
        keys, values = torch.randn(1, 1, 16, 2), torch.randn(1, 1, 16, 2)
        cache = _FakeCache([_FakeCacheLayer(keys, values)])
        sketch = _make(KnormSketch(), [0.9])
        module = _FakeAttnModule(head_dim=2, layer_idx=0)

        kwargs = {
            "hidden_states": torch.randn(1, 1, 4),
            "past_key_values": cache,
            "cache_position": torch.tensor([16]),  # cache_position[-1] > q_len -> decode
        }
        output = (torch.randn(1, 1, 4), None)
        result = sketch.forward_hook(module, [], kwargs, output)

        self.assertIs(result, output)
        self.assertIs(cache.layers[0].keys, keys)
        self.assertIs(cache.layers[0].values, values)

    def test_multi_token_question_forward_is_noop(self):
        keys, values = torch.randn(1, 1, 20, 2), torch.randn(1, 1, 20, 2)
        cache = _FakeCache([_FakeCacheLayer(keys, values)])
        sketch = _make(KnormSketch(), [0.9])
        module = _FakeAttnModule(head_dim=2, layer_idx=0)

        kwargs = {
            "hidden_states": torch.randn(1, 4, 4),
            "past_key_values": cache,
            "cache_position": torch.arange(16, 20),  # 19 > q_len=4 -> decode
        }
        output = (torch.randn(1, 4, 4), None)
        sketch.forward_hook(module, [], kwargs, output)

        self.assertIs(cache.layers[0].keys, keys)
        self.assertIs(cache.layers[0].values, values)

    def test_ratios_shorter_than_layer_idx_raises_index_error(self):
        sketch = _make(KnormSketch(), [0.5])
        keys, values = torch.randn(1, 1, 4, 2), torch.randn(1, 1, 4, 2)
        cache = _FakeCache([_FakeCacheLayer(keys, values), _FakeCacheLayer(keys, values)])

        with self.assertRaises(IndexError):
            _run_prefill_hook(sketch, _FakeAttnModule(head_dim=2, layer_idx=1), cache, B=1, S=4)
        # IndexError fires before the inner ratio is mutated.
        self.assertEqual(sketch.press.compression_ratio, 0.0)


# ======================================================================
# post_init_from_model: length validation + decode-safety guard
# ======================================================================


class TestPostInitFromModel(unittest.TestCase):
    def test_unequal_ratios_under_sdpa_raises(self):
        sketch = _make(KnormSketch(), [0.1, 0.9])
        with self.assertRaisesRegex(ValueError, "flash_attention_2"):
            sketch.post_init_from_model(_fake_model(2, "sdpa"))

    def test_unequal_ratios_under_eager_raises(self):
        sketch = _make(KnormSketch(), [0.1, 0.9])
        with self.assertRaises(ValueError):
            sketch.post_init_from_model(_fake_model(2, "eager"))

    def test_unequal_ratios_without_attn_implementation_raises(self):
        sketch = _make(KnormSketch(), [0.1, 0.9])
        with self.assertRaises(ValueError):
            sketch.post_init_from_model(_fake_model(2, attn_implementation=None))

    def test_unequal_ratios_under_flash_attention_2_allowed(self):
        sketch = _make(KnormSketch(), [0.1, 0.9])
        sketch.post_init_from_model(_fake_model(2, "flash_attention_2"))

    def test_equal_ratios_under_sdpa_allowed(self):
        sketch = _make(KnormSketch(), [0.3, 0.3])
        sketch.post_init_from_model(_fake_model(2, "sdpa"))

    def test_guard_ignores_entries_beyond_num_hidden_layers(self):
        sketch = _make(KnormSketch(), [0.3, 0.3, 0.9])
        sketch.post_init_from_model(_fake_model(2, "sdpa"))

    def test_too_few_ratios_raises_value_error(self):
        sketch = _make(KnormSketch(), [0.5])
        with self.assertRaisesRegex(ValueError, "entries"):
            sketch.post_init_from_model(_fake_model(2, "flash_attention_2"))

    def test_delegates_to_inner_press(self):
        inner = KnormSketch()
        sketch = _make(inner, [0.3, 0.3])
        calls = []
        inner.post_init_from_model = lambda model: calls.append(model)
        model = _fake_model(2, "sdpa")
        sketch.post_init_from_model(model)
        self.assertEqual(calls, [model])

    def test_num_hidden_layers_from_get_text_config(self):
        text_cfg = SimpleNamespace(num_hidden_layers=2)
        cfg = SimpleNamespace(_attn_implementation="flash_attention_2")
        cfg.get_text_config = lambda: text_cfg
        sketch = _make(KnormSketch(), [0.5])
        with self.assertRaisesRegex(ValueError, "entries"):
            sketch.post_init_from_model(SimpleNamespace(config=cfg))


# ======================================================================
# Registry / adapter wiring
# ======================================================================


class TestRegistryAndWiring(unittest.TestCase):
    def test_registered_name_resolves(self):
        from eval_harness.kv_compression.registry import available_kv_compressors, get_kv_compressor_class

        self.assertIs(get_kv_compressor_class("per_layer_compression"), PerLayerCompressionSketch)
        self.assertIn("per_layer_compression", available_kv_compressors())

    def test_compression_ratio_is_not_a_dataclass_field(self):
        # ResearchAdapter._build_sketch injects cfg.compression_ratio only for
        # dataclass fields, so the adapter-level ratio is ignored here.
        self.assertNotIn("compression_ratio", {f.name for f in fields(PerLayerCompressionSketch)})

    def test_build_sketch_via_adapter_sketch_kwargs(self):
        from eval_harness.research_adapter import ResearchConfig, ResearchAdapter

        cfg = ResearchConfig(
            kv_compressor="per_layer_compression",
            compression_ratio=0.9,  # must be ignored (not a field)
            kv_compressor_kwargs={"press": "knorm", "compression_ratios": [0.1, 0.2]},
        )
        adapter = object.__new__(ResearchAdapter)
        adapter._cache_cfg = cfg
        logger = logging.getLogger(LOGGER_NAME)
        logger.disabled = True
        try:
            sketch = adapter._build_kv_compressor(cfg)
        finally:
            logger.disabled = False

        self.assertIsInstance(sketch, PerLayerCompressionSketch)
        self.assertIsInstance(sketch.press, KnormSketch)
        self.assertEqual(sketch.compression_ratios, [0.1, 0.2])
        self.assertEqual(sketch.compression_ratio, (0.1 + 0.2) / 2)


if __name__ == "__main__":
    unittest.main()
