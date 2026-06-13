"""Tests for ComposedSketch (port of kvpress 0.5.1 ComposedPress).

Reference math (kvpress/presses/composed_press.py:56-62):
    retained_fraction = 1.0
    for press in presses:
        output = press.forward_hook(module, input, kwargs, output)
        retained_fraction *= 1 - press.compression_ratio
    self.compression_ratio = 1 - retained_fraction

No model loading; fake attention modules + DynamicCache only.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass, fields

import torch
from torch import nn
from transformers import DynamicCache

from eval_harness.research_adapter import ResearchConfig, ResearchAdapter
from eval_harness.kv_compression.compressors.composed_sketch import ComposedSketch
from eval_harness.kv_compression.compressors.decoding_sketch import DecodingSketch
from eval_harness.kv_compression.compressors.knorm_sketch import KnormSketch
from eval_harness.kv_compression.compressors.random_sketch import RandomSketch
from eval_harness.kv_compression.registry import (
    available_kv_compressors,
    get_kv_compressor,
    get_kv_compressor_class,
)
from eval_harness.kv_compression.base import ScorerKVCompressor


class _FakeAttnModule(nn.Module):
    """Knorm/Random members only read ``head_dim`` (gather expand) and
    ``layer_idx`` (cache access); GQA-shaped by default."""

    def __init__(self, num_heads=4, num_kv_heads=2, head_dim=4, layer_idx=0):
        super().__init__()
        self.num_heads = num_heads
        self.num_key_value_heads = num_kv_heads
        self.head_dim = head_dim
        self.layer_idx = layer_idx


@dataclass
class _RecordingSketch(ScorerKVCompressor):
    def __post_init__(self):
        super().__post_init__()
        self.seen_models = []

    def post_init_from_model(self, model):
        self.seen_models.append(model)

    def score(self, module, hidden_states, keys, values, attentions, kwargs):
        return torch.zeros(*keys.shape[:-1], device=keys.device)


def _norm_keys(seq: int, head_dim: int = 4, heads: int = 1) -> torch.Tensor:
    """keys[0, h, t, :] = (t + 1) * e0, so ||k_t|| = t + 1 (strictly increasing)."""
    keys = torch.zeros(1, heads, seq, head_dim)
    keys[:, :, :, 0] = torch.arange(1, seq + 1, dtype=torch.float32)
    return keys


def _id_values(seq: int, head_dim: int = 4, heads: int = 1) -> torch.Tensor:
    """values[0, h, t, :] = t for identity tracking."""
    return (
        torch.arange(seq, dtype=torch.float32)
        .view(1, 1, seq, 1)
        .expand(1, heads, seq, head_dim)
        .contiguous()
    )


def _prefill_hook(sketch, module, cache, seq, batch=1, hidden=16):
    hidden_states = torch.randn(batch, seq, hidden)
    output = (torch.randn(batch, seq, hidden), None)
    kwargs = {
        "hidden_states": hidden_states,
        "past_key_values": cache,
        "cache_position": torch.arange(seq),
    }
    result = sketch.forward_hook(module, [], kwargs, output)
    return result, output


class TestComposedRegistry(unittest.TestCase):
    def test_registered_under_composed(self):
        self.assertIn("composed", available_kv_compressors())
        self.assertIs(get_kv_compressor_class("composed"), ComposedSketch)

    def test_get_kv_compressor_resolves_member_specs(self):
        sketch = get_kv_compressor(
            "composed",
            presses=[("knorm", {"compression_ratio": 0.5}), "random"],
        )
        self.assertIsInstance(sketch, ComposedSketch)
        self.assertIsInstance(sketch.presses[0], KnormSketch)
        self.assertAlmostEqual(sketch.presses[0].compression_ratio, 0.5)
        self.assertIsInstance(sketch.presses[1], RandomSketch)
        self.assertEqual(sketch.presses[1].compression_ratio, 0.0)

    def test_instance_members_pass_through_unwrapped(self):
        member = KnormSketch(compression_ratio=0.25)
        sketch = ComposedSketch(presses=[member])
        self.assertIs(sketch.presses[0], member)

    def test_invalid_member_spec_raises_type_error(self):
        with self.assertRaises(TypeError):
            ComposedSketch(presses=[42])

    def test_compression_ratio_is_not_a_dataclass_field(self):
        self.assertNotIn("compression_ratio", {f.name for f in fields(ComposedSketch)})
        self.assertIsNone(ComposedSketch(presses=[]).compression_ratio)

    def test_build_sketch_does_not_inject_adapter_ratio(self):
        adapter = object.__new__(ResearchAdapter)
        cfg = ResearchConfig(
            kv_compressor="composed",
            compression_ratio=0.9,
            kv_compressor_kwargs={"presses": [["knorm", {"compression_ratio": 0.5}]]},
        )
        adapter._cache_cfg = cfg
        sketch = adapter._build_kv_compressor(cfg)
        self.assertIsInstance(sketch, ComposedSketch)
        self.assertIsNone(sketch.compression_ratio)
        self.assertIsInstance(sketch.presses[0], KnormSketch)
        self.assertAlmostEqual(sketch.presses[0].compression_ratio, 0.5)


class TestComposedPostInitFromModel(unittest.TestCase):
    def test_fans_out_to_every_member(self):
        members = [_RecordingSketch(compression_ratio=0.1), _RecordingSketch(compression_ratio=0.2)]
        sketch = ComposedSketch(presses=list(members))
        sentinel = object()
        sketch.post_init_from_model(sentinel)
        for member in members:
            self.assertEqual(member.seen_models, [sentinel])


class TestComposedForwardHook(unittest.TestCase):
    def test_empty_presses_is_noop_with_zero_ratio(self):
        cache = DynamicCache()
        cache.update(torch.randn(1, 2, 8, 4), torch.randn(1, 2, 8, 4), 0)
        keys_before = cache.layers[0].keys
        values_before = cache.layers[0].values
        sketch = ComposedSketch(presses=[])
        result, output = _prefill_hook(sketch, _FakeAttnModule(), cache, 8)
        self.assertIs(result, output)
        self.assertIs(cache.layers[0].keys, keys_before)
        self.assertIs(cache.layers[0].values, values_before)
        self.assertEqual(sketch.compression_ratio, 0.0)

    def test_zero_ratio_members_are_noop(self):
        cache = DynamicCache()
        cache.update(torch.randn(1, 2, 8, 4), torch.randn(1, 2, 8, 4), 0)
        keys_before = cache.layers[0].keys
        values_before = cache.layers[0].values
        sketch = ComposedSketch(
            presses=[KnormSketch(compression_ratio=0.0), RandomSketch(compression_ratio=0.0)]
        )
        result, output = _prefill_hook(sketch, _FakeAttnModule(), cache, 8)
        self.assertIs(result, output)
        self.assertIs(cache.layers[0].keys, keys_before)
        self.assertIs(cache.layers[0].values, values_before)
        self.assertEqual(sketch.compression_ratio, 0.0)

    def test_two_stage_hand_computed_selection(self):
        # Knorm score = -||k|| keeps the SMALLEST norms: stage 1 keeps
        # int(10 * 0.5) = 5 tokens {0..4}, stage 2 keeps int(5 * 0.5) = 2
        # tokens {0, 1} out of the stage-1 survivors.
        cache = DynamicCache()
        cache.update(_norm_keys(10), _id_values(10), 0)
        module = _FakeAttnModule(num_heads=1, num_kv_heads=1)
        sketch = ComposedSketch(
            presses=[KnormSketch(compression_ratio=0.5), KnormSketch(compression_ratio=0.5)]
        )
        result, output = _prefill_hook(sketch, module, cache, 10)
        self.assertIs(result, output)
        self.assertEqual(cache.layers[0].keys.shape, (1, 1, 2, 4))
        self.assertEqual(cache.layers[0].values.shape, (1, 1, 2, 4))
        kept = {int(v) for v in cache.layers[0].values[0, 0, :, 0].tolist()}
        self.assertEqual(kept, {0, 1})
        self.assertEqual(sketch.compression_ratio, 0.75)

    def test_reported_ratio_is_product_formula(self):
        cache = DynamicCache()
        cache.update(_norm_keys(20), _id_values(20), 0)
        module = _FakeAttnModule(num_heads=1, num_kv_heads=1)
        sketch = ComposedSketch(
            presses=[KnormSketch(compression_ratio=0.3), KnormSketch(compression_ratio=0.2)]
        )
        _prefill_hook(sketch, module, cache, 20)
        self.assertAlmostEqual(sketch.compression_ratio, 1 - 0.7 * 0.8, places=12)
        self.assertAlmostEqual(sketch.compression_ratio, 0.44, places=12)
        self.assertEqual(cache.layers[0].keys.shape[2], int(int(20 * 0.7) * 0.8))
        self.assertEqual(cache.layers[0].keys.shape[2], 11)

    def test_truncation_vs_formula_divergence_pinned(self):
        # Realized kept count is the nested truncation
        # int(int(10 * 0.55) * 0.55) = 2, while the reported formula ratio
        # 0.6975 would imply int(10 * 0.3025) = 3.
        cache = DynamicCache()
        cache.update(_norm_keys(10), _id_values(10), 0)
        module = _FakeAttnModule(num_heads=1, num_kv_heads=1)
        sketch = ComposedSketch(
            presses=[KnormSketch(compression_ratio=0.45), KnormSketch(compression_ratio=0.45)]
        )
        _prefill_hook(sketch, module, cache, 10)
        self.assertEqual(cache.layers[0].keys.shape[2], 2)
        self.assertAlmostEqual(sketch.compression_ratio, 0.6975, places=12)
        self.assertEqual(int(10 * (1 - sketch.compression_ratio)), 3)

    def test_decode_step_is_noop(self):
        cache = DynamicCache()
        cache.update(_norm_keys(8), _id_values(8), 0)
        module = _FakeAttnModule(num_heads=1, num_kv_heads=1)
        sketch = ComposedSketch(presses=[KnormSketch(compression_ratio=0.5)])
        _prefill_hook(sketch, module, cache, 8)
        self.assertEqual(cache.layers[0].keys.shape[2], 4)
        keys_after_prefill = cache.layers[0].keys
        values_after_prefill = cache.layers[0].values

        hidden_states = torch.randn(1, 1, 16)
        decode_output = (torch.randn(1, 1, 16), None)
        kwargs = {
            "hidden_states": hidden_states,
            "past_key_values": cache,
            "cache_position": torch.tensor([8]),
        }
        result = sketch.forward_hook(module, [], kwargs, decode_output)
        self.assertIs(result, decode_output)
        self.assertIs(cache.layers[0].keys, keys_after_prefill)
        self.assertIs(cache.layers[0].values, values_after_prefill)
        self.assertEqual(sketch.compression_ratio, 0.5)

    def test_gqa_keeps_kv_heads_rectangular(self):
        cache = DynamicCache()
        keys = torch.randn(1, 2, 12, 4)
        values = torch.randn(1, 2, 12, 4)
        cache.update(keys, values, 0)
        module = _FakeAttnModule(num_heads=8, num_kv_heads=2)
        sketch = ComposedSketch(presses=[KnormSketch(compression_ratio=0.5)])
        _prefill_hook(sketch, module, cache, 12)
        self.assertEqual(cache.layers[0].keys.shape, (1, 2, 6, 4))
        self.assertEqual(cache.layers[0].values.shape, (1, 2, 6, 4))

    def test_nested_composition_reads_inner_ratio_after_delegate(self):
        inner = ComposedSketch(presses=[KnormSketch(compression_ratio=0.5)])
        outer = ComposedSketch(presses=[inner, KnormSketch(compression_ratio=0.5)])
        self.assertIsNone(inner.compression_ratio)
        self.assertIsNone(outer.compression_ratio)

        cache = DynamicCache()
        cache.update(_norm_keys(10), _id_values(10), 0)
        module = _FakeAttnModule(num_heads=1, num_kv_heads=1)
        result, output = _prefill_hook(outer, module, cache, 10)
        self.assertIs(result, output)
        self.assertEqual(inner.compression_ratio, 0.5)
        self.assertEqual(outer.compression_ratio, 0.75)
        self.assertEqual(cache.layers[0].keys.shape[2], 2)

    def test_sequential_pruning_yields_subset_of_stage_one(self):
        seed, seq = 7, 10
        cache = DynamicCache()
        cache.update(torch.randn(1, 1, seq, 4), _id_values(seq), 0)
        module = _FakeAttnModule(num_heads=1, num_kv_heads=1)
        sketch = ComposedSketch(
            presses=[
                RandomSketch(compression_ratio=0.5, seed=seed),
                KnormSketch(compression_ratio=0.5),
            ]
        )
        _prefill_hook(sketch, module, cache, seq)

        generator = torch.Generator()
        generator.manual_seed(seed)
        stage1_scores = torch.rand(1, 1, seq, generator=generator)
        stage1_kept = set(stage1_scores.topk(int(seq * 0.5), dim=-1).indices.flatten().tolist())

        self.assertEqual(cache.layers[0].keys.shape[2], int(int(seq * 0.5) * 0.5))
        survivors = {int(v) for v in cache.layers[0].values[0, 0, :, 0].tolist()}
        self.assertEqual(len(survivors), 2)
        self.assertTrue(survivors.issubset(stage1_kept))

    def test_cross_layer_kept_counts_identical(self):
        torch.manual_seed(0)
        seq = 10
        cache = DynamicCache()
        cache.update(torch.randn(1, 2, seq, 4), torch.randn(1, 2, seq, 4), 0)
        cache.update(torch.randn(1, 2, seq, 4) * 5.0, torch.randn(1, 2, seq, 4), 1)
        sketch = ComposedSketch(
            presses=[KnormSketch(compression_ratio=0.3), KnormSketch(compression_ratio=0.4)]
        )
        for layer_idx in range(2):
            module = _FakeAttnModule(layer_idx=layer_idx)
            _prefill_hook(sketch, module, cache, seq)
        self.assertEqual(cache.layers[0].keys.shape[2], cache.layers[1].keys.shape[2])
        self.assertEqual(cache.layers[0].keys.shape[2], int(int(seq * 0.7) * 0.6))

    def test_member_without_numeric_ratio_raises_at_hook(self):
        # Mirrors kvpress: no construction-time validation; a member lacking
        # a numeric compression_ratio (DecodingSketch) fails when the hook
        # computes 1 - press.compression_ratio.
        sketch = ComposedSketch(
            presses=[DecodingSketch(base_sketch=KnormSketch(compression_ratio=0.5))]
        )
        cache = DynamicCache()
        cache.update(_norm_keys(8), _id_values(8), 0)
        module = _FakeAttnModule(num_heads=1, num_kv_heads=1)
        with self.assertRaises((TypeError, AttributeError)):
            _prefill_hook(sketch, module, cache, 8)


if __name__ == "__main__":
    unittest.main()
