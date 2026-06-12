"""Unit tests for ChunkKVSketch (port of kvpress 0.5.1 ChunkKVPress).

No model loading; fake attention modules only. The kvpress chunk-selection math
(chunkkv_press.py:61-118) is transcribed inline as a reference oracle (kvpress
is not importable in prism_env). The uniform-length padding adaptation is pinned
both by hand-computed cases and by an independent expected-index helper.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from eval_harness.sketch.sketches.chunkkv_sketch import ChunkKVSketch
from eval_harness.sketch.sketches.knorm_sketch import KnormSketch
from eval_harness.sketch.sketches.registry import get_sketch, get_sketch_class


def _fake_module(head_dim: int = 2, **extra) -> SimpleNamespace:
    return SimpleNamespace(head_dim=head_dim, **extra)


def _keys_with_norms(norms, heads: int = 2, head_dim: int = 2) -> torch.Tensor:
    seq = len(norms)
    keys = torch.zeros(1, heads, seq, head_dim)
    keys[..., 0] = torch.tensor(norms, dtype=torch.float32)
    return keys


def _position_values(batch: int = 1, heads: int = 2, seq: int = 8, head_dim: int = 2) -> torch.Tensor:
    pos = torch.arange(seq, dtype=torch.float32).view(1, 1, seq, 1)
    return pos.expand(batch, heads, seq, head_dim).contiguous()


def _kvpress_reference_indices(
    global_scores: torch.Tensor, chunk_length: int, compression_ratio: float
) -> torch.Tensor:
    """Verbatim transcription of kvpress ChunkKVPress.compress chunk selection
    (chunkkv_press.py:79-118), batch element 0; requires >= 1 complete chunk."""
    kv_len = global_scores.shape[-1]
    num_complete_chunks = kv_len // chunk_length
    remaining_tokens = kv_len % chunk_length
    assert num_complete_chunks > 0

    main_scores = global_scores[..., : num_complete_chunks * chunk_length]
    main_chunk_scores = main_scores.sum(dim=1).view(-1, num_complete_chunks, chunk_length).mean(dim=-1)

    if remaining_tokens > 0:
        remaining_scores = global_scores[..., -remaining_tokens:]
        remaining_chunk_score = remaining_scores.sum(dim=1).mean(dim=-1, keepdim=True)
        chunk_scores = torch.cat([main_chunk_scores, remaining_chunk_score], dim=-1)
    else:
        chunk_scores = main_chunk_scores

    n_chunks_kept = max(1, int((num_complete_chunks + (remaining_tokens > 0)) * (1 - compression_ratio)))
    top_chunks = chunk_scores.topk(n_chunks_kept, dim=-1)

    indices = []
    for chunk_idx in top_chunks.indices[0]:
        if chunk_idx < num_complete_chunks:
            start_idx = chunk_idx * chunk_length
            indices.append(torch.arange(start_idx, start_idx + chunk_length))
        else:
            indices.append(torch.arange(num_complete_chunks * chunk_length, kv_len))
    return torch.cat(indices).sort()[0]


def _expected_padded_indices(
    global_scores: torch.Tensor, chunk_length: int, compression_ratio: float
) -> torch.Tensor:
    """kvpress reference indices padded to the spec's deterministic target
    min(n_chunks_kept * chunk_length, kv_len) with the highest head-summed-score
    unselected positions."""
    ref = _kvpress_reference_indices(global_scores, chunk_length, compression_ratio)
    kv_len = global_scores.shape[-1]
    num_complete_chunks = kv_len // chunk_length
    remaining_tokens = kv_len % chunk_length
    n_chunks_kept = max(1, int((num_complete_chunks + (remaining_tokens > 0)) * (1 - compression_ratio)))
    target = min(n_chunks_kept * chunk_length, kv_len)
    n_pad = target - ref.numel()
    if n_pad == 0:
        return ref
    summed = global_scores.sum(dim=1)[0].clone()
    summed[ref] = float("-inf")
    pad = summed.topk(n_pad).indices
    return torch.cat([ref, pad]).sort()[0]


class TestChunkKVCompress(unittest.TestCase):
    def test_zero_ratio_is_noop_identity(self):
        keys = torch.randn(1, 2, 10, 2)
        values = torch.randn(1, 2, 10, 2)
        sketch = ChunkKVSketch(press=KnormSketch(compression_ratio=0.0), chunk_length=4)
        out_keys, out_values = sketch.compress(_fake_module(), None, keys, values, None, {})
        self.assertIs(out_keys, keys)
        self.assertIs(out_values, values)

    def test_zero_ratio_short_circuits_before_attentions_assert(self):
        keys = torch.randn(1, 2, 10, 2)
        values = torch.randn(1, 2, 10, 2)
        sketch = ChunkKVSketch(press=KnormSketch(compression_ratio=0.0), chunk_length=4)
        out_keys, out_values = sketch.compress(
            _fake_module(), None, keys, values, torch.ones(1, 2, 10, 10), {}
        )
        self.assertIs(out_keys, keys)
        self.assertIs(out_values, values)

    def test_attentions_not_supported(self):
        keys = torch.randn(1, 2, 8, 2)
        sketch = ChunkKVSketch(press=KnormSketch(compression_ratio=0.5), chunk_length=4)
        with self.assertRaisesRegex(AssertionError, "ChunkPress does not support attentions"):
            sketch.compress(_fake_module(), None, keys, keys.clone(), torch.ones(1, 2, 8, 8), {})

    def test_exact_divisible_selection_hand_computed(self):
        keys = _keys_with_norms([1, 1, 1, 1, 3, 3, 3, 3])
        values = _position_values(seq=8)
        sketch = ChunkKVSketch(press=KnormSketch(compression_ratio=0.5), chunk_length=4)
        out_keys, out_values = sketch.compress(_fake_module(), None, keys, values, None, {})
        self.assertEqual(out_keys.shape, (1, 2, 4, 2))
        self.assertEqual(out_values.shape, (1, 2, 4, 2))
        self.assertTrue(torch.equal(out_keys, keys[:, :, 0:4, :]))
        self.assertTrue(torch.equal(out_values, values[:, :, 0:4, :]))

    def test_partial_chunk_padded_to_uniform_target(self):
        norms = [1, 1, 9, 9, 9, 9, 9, 9, 0.5, 0.5]
        keys = _keys_with_norms(norms)
        values = _position_values(seq=10)
        global_scores = -keys.norm(dim=-1)

        ref = _kvpress_reference_indices(global_scores, chunk_length=4, compression_ratio=0.5)
        self.assertEqual(ref.tolist(), [8, 9])

        sketch = ChunkKVSketch(press=KnormSketch(compression_ratio=0.5), chunk_length=4)
        out_keys, out_values = sketch.compress(_fake_module(), None, keys, values, None, {})
        self.assertEqual(out_keys.shape, (1, 2, 4, 2))
        for h in range(2):
            kept = out_values[0, h, :, 0].tolist()
            self.assertEqual(kept, [0.0, 1.0, 8.0, 9.0])
            self.assertTrue(set(ref.tolist()).issubset({int(p) for p in kept}))
        self.assertTrue(torch.equal(out_keys, keys[:, :, [0, 1, 8, 9], :]))

    def test_cross_layer_uniform_length(self):
        sketch = ChunkKVSketch(press=KnormSketch(compression_ratio=0.5), chunk_length=4)
        layer_a_keys = _keys_with_norms([1, 1, 9, 9, 9, 9, 9, 9, 0.5, 0.5])
        layer_b_keys = _keys_with_norms([9, 9, 9, 9, 0.5, 0.5, 0.5, 0.5, 9, 9])
        values = _position_values(seq=10)

        out_a, vals_a = sketch.compress(_fake_module(), None, layer_a_keys, values, None, {})
        out_b, vals_b = sketch.compress(_fake_module(), None, layer_b_keys, values, None, {})
        self.assertEqual(out_a.shape[2], 4)
        self.assertEqual(out_b.shape[2], 4)
        self.assertEqual(out_a.shape[2], out_b.shape[2])
        self.assertEqual(vals_a[0, 0, :, 0].tolist(), [0.0, 1.0, 8.0, 9.0])
        self.assertEqual(vals_b[0, 0, :, 0].tolist(), [4.0, 5.0, 6.0, 7.0])

    def test_delegates_to_inner_press_below_chunk_length(self):
        keys = torch.zeros(1, 2, 3, 2)
        keys[0, 0, :, 0] = torch.tensor([5.0, 6.0, 1.0])
        keys[0, 1, :, 0] = torch.tensor([1.0, 6.0, 5.0])
        values = _position_values(seq=3)
        sketch = ChunkKVSketch(press=KnormSketch(compression_ratio=0.5), chunk_length=20)
        out_keys, out_values = sketch.compress(_fake_module(), None, keys, values, None, {})
        self.assertEqual(out_keys.shape, (1, 2, 1, 2))
        self.assertEqual(out_values[0, 0, 0, 0].item(), 2.0)
        self.assertEqual(out_values[0, 1, 0, 0].item(), 0.0)
        self.assertTrue(torch.equal(out_keys[0, 0, 0], keys[0, 0, 2]))
        self.assertTrue(torch.equal(out_keys[0, 1, 0], keys[0, 1, 0]))

    def test_gqa_chunk_selection_uniform_across_kv_heads(self):
        torch.manual_seed(0)
        module = SimpleNamespace(head_dim=4, num_attention_heads=8, num_key_value_heads=2)
        keys = torch.randn(1, 2, 40, 4)
        values = _position_values(heads=2, seq=40, head_dim=4)
        sketch = ChunkKVSketch(press=KnormSketch(compression_ratio=0.5), chunk_length=8)
        out_keys, out_values = sketch.compress(module, None, keys, values, None, {})
        self.assertEqual(out_keys.shape, (1, 2, 16, 4))

        expected = _kvpress_reference_indices(-keys.norm(dim=-1), chunk_length=8, compression_ratio=0.5)
        self.assertTrue(torch.equal(out_keys, keys[:, :, expected, :]))
        self.assertEqual(out_values[0, 0, :, 0].tolist(), out_values[0, 1, :, 0].tolist())

    def test_max_one_chunk_floor(self):
        keys = _keys_with_norms([9, 9, 9, 9, 1, 1, 1, 1])
        values = _position_values(seq=8)
        sketch = ChunkKVSketch(press=KnormSketch(compression_ratio=0.9), chunk_length=4)
        out_keys, out_values = sketch.compress(_fake_module(), None, keys, values, None, {})
        self.assertEqual(out_keys.shape, (1, 2, 4, 2))
        self.assertEqual(out_values[0, 0, :, 0].tolist(), [4.0, 5.0, 6.0, 7.0])

    def test_kvpress_count_parity_s256(self):
        torch.manual_seed(1)
        keys = torch.randn(1, 2, 256, 4)
        values = torch.randn(1, 2, 256, 4)
        for chunk_length in (2, 4, 8, 128):
            with self.subTest(chunk_length=chunk_length):
                sketch = ChunkKVSketch(
                    press=KnormSketch(compression_ratio=0.5), chunk_length=chunk_length
                )
                out_keys, out_values = sketch.compress(
                    _fake_module(head_dim=4), None, keys, values, None, {}
                )
                self.assertEqual(out_keys.shape[2], 128)
                self.assertEqual(out_values.shape[2], 128)

    def test_output_positions_sorted_ascending(self):
        keys = _keys_with_norms([2, 2, 2, 2, 9, 9, 9, 9, 1, 1, 1, 1])
        values = _position_values(seq=12)
        sketch = ChunkKVSketch(press=KnormSketch(compression_ratio=0.3), chunk_length=4)
        out_keys, out_values = sketch.compress(_fake_module(), None, keys, values, None, {})
        self.assertEqual(out_keys.shape, (1, 2, 8, 2))
        for h in range(2):
            self.assertEqual(
                out_values[0, h, :, 0].tolist(), [0.0, 1.0, 2.0, 3.0, 8.0, 9.0, 10.0, 11.0]
            )

    def test_batch_size_two_raises_on_chunk_path(self):
        keys = torch.randn(2, 2, 8, 2)
        sketch = ChunkKVSketch(press=KnormSketch(compression_ratio=0.5), chunk_length=4)
        with self.assertRaisesRegex(AssertionError, "batch size 1"):
            sketch.compress(_fake_module(), None, keys, keys.clone(), None, {})

    def test_batch_size_two_allowed_on_delegate_path(self):
        keys = torch.randn(2, 2, 3, 2)
        values = torch.randn(2, 2, 3, 2)
        sketch = ChunkKVSketch(press=KnormSketch(compression_ratio=0.5), chunk_length=20)
        out_keys, out_values = sketch.compress(_fake_module(), None, keys, values, None, {})
        self.assertEqual(out_keys.shape, (2, 2, 1, 2))
        self.assertEqual(out_values.shape, (2, 2, 1, 2))


class TestChunkKVReferenceOracle(unittest.TestCase):
    def test_matches_kvpress_transcription_plus_pad_rule(self):
        torch.manual_seed(2)
        cases = [
            (64, 8, 0.5),
            (64, 8, 0.25),
            (60, 20, 0.5),
            (40, 4, 0.7),
            (10, 4, 0.5),
            (37, 5, 0.4),
            (26, 8, 0.6),
            (21, 20, 0.3),
        ]
        for seq, chunk_length, ratio in cases:
            with self.subTest(seq=seq, chunk_length=chunk_length, ratio=ratio):
                keys = torch.randn(1, 2, seq, 4)
                values = torch.randn(1, 2, seq, 4)
                global_scores = -keys.norm(dim=-1)
                ref = _kvpress_reference_indices(global_scores, chunk_length, ratio)
                expected = _expected_padded_indices(global_scores, chunk_length, ratio)
                if seq % chunk_length == 0:
                    self.assertTrue(torch.equal(ref, expected))
                self.assertTrue(set(ref.tolist()).issubset(set(expected.tolist())))

                sketch = ChunkKVSketch(
                    press=KnormSketch(compression_ratio=ratio), chunk_length=chunk_length
                )
                out_keys, out_values = sketch.compress(
                    _fake_module(head_dim=4), None, keys, values, None, {}
                )
                self.assertTrue(torch.equal(out_keys, keys[:, :, expected, :]))
                self.assertTrue(torch.equal(out_values, values[:, :, expected, :]))


class TestChunkKVConstruction(unittest.TestCase):
    def test_defaults_mirror_kvpress(self):
        sketch = ChunkKVSketch()
        self.assertEqual(sketch.chunk_length, 20)
        self.assertIsInstance(sketch.press, KnormSketch)
        self.assertEqual(sketch.compression_ratio, 0.0)

    def test_press_must_be_scorer_sketch(self):
        with self.assertRaisesRegex(AssertionError, "requires a ScorerSketch"):
            ChunkKVSketch(press=object())

    def test_compression_ratio_property_delegates_to_inner(self):
        inner = KnormSketch(compression_ratio=0.5)
        sketch = ChunkKVSketch(press=inner, chunk_length=4)
        self.assertAlmostEqual(sketch.compression_ratio, 0.5)
        sketch.compression_ratio = 0.25
        self.assertAlmostEqual(inner.compression_ratio, 0.25)
        self.assertAlmostEqual(sketch.compression_ratio, 0.25)

    def test_ratio_mutation_through_wrapper_drives_compress(self):
        sketch = ChunkKVSketch(press=KnormSketch(compression_ratio=0.0), chunk_length=4)
        keys = _keys_with_norms([1, 1, 1, 1, 3, 3, 3, 3])
        out_keys, _ = sketch.compress(_fake_module(), None, keys, keys.clone(), None, {})
        self.assertIs(out_keys, keys)
        sketch.compression_ratio = 0.5
        out_keys, _ = sketch.compress(_fake_module(), None, keys, keys.clone(), None, {})
        self.assertEqual(out_keys.shape[2], 4)

    def test_constructor_ratio_applies_to_default_press(self):
        sketch = ChunkKVSketch(compression_ratio=0.5)
        self.assertIsInstance(sketch.press, KnormSketch)
        self.assertAlmostEqual(sketch.press.compression_ratio, 0.5)
        self.assertAlmostEqual(sketch.compression_ratio, 0.5)

    def test_constructor_ratio_overrides_explicit_press_ratio(self):
        sketch = ChunkKVSketch(press=KnormSketch(compression_ratio=0.1), compression_ratio=0.5)
        self.assertAlmostEqual(sketch.press.compression_ratio, 0.5)

    def test_post_init_from_model_delegates(self):
        class _RecordingKnorm(KnormSketch):
            def __init__(self, **kw):
                super().__init__(**kw)
                self.seen_models = []

            def post_init_from_model(self, model):
                self.seen_models.append(model)

        inner = _RecordingKnorm(compression_ratio=0.5)
        sketch = ChunkKVSketch(press=inner)
        sentinel = object()
        sketch.post_init_from_model(sentinel)
        self.assertEqual(inner.seen_models, [sentinel])


class TestChunkKVRegistry(unittest.TestCase):
    def test_registry_resolution(self):
        self.assertIs(get_sketch_class("chunkkv"), ChunkKVSketch)

    def test_get_sketch_sets_fields(self):
        sketch = get_sketch("chunkkv", compression_ratio=0.4, chunk_length=8)
        self.assertIsInstance(sketch, ChunkKVSketch)
        self.assertEqual(sketch.chunk_length, 8)
        self.assertIsInstance(sketch.press, KnormSketch)
        self.assertAlmostEqual(sketch.compression_ratio, 0.4)

    def test_build_sketch_injects_adapter_ratio(self):
        from eval_harness.research_adapter import CacheConfig, ResearchAdapter

        adapter = object.__new__(ResearchAdapter)
        cfg = CacheConfig(
            sketch_name="chunkkv", compression_ratio=0.4, sketch_kwargs={"chunk_length": 8}
        )
        sketch = adapter._build_sketch(cfg)
        self.assertIsInstance(sketch, ChunkKVSketch)
        self.assertEqual(sketch.chunk_length, 8)
        self.assertAlmostEqual(sketch.compression_ratio, 0.4)

    def test_build_sketch_kwargs_ratio_overrides_adapter_ratio(self):
        from eval_harness.research_adapter import CacheConfig, ResearchAdapter

        adapter = object.__new__(ResearchAdapter)
        cfg = CacheConfig(
            sketch_name="chunkkv",
            compression_ratio=0.4,
            sketch_kwargs={"compression_ratio": 0.6},
        )
        sketch = adapter._build_sketch(cfg)
        self.assertAlmostEqual(sketch.compression_ratio, 0.6)


if __name__ == "__main__":
    unittest.main()
