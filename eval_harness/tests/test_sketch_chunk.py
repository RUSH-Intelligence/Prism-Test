"""Unit tests for ChunkSketch (port of kvpress 0.5.1 ChunkPress).

No model loading; fake attention modules only. The kvpress chunk loop
(chunk_press.py:60-87) is transcribed inline as a reference oracle (kvpress is
not importable in prism_env).
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from types import SimpleNamespace

import torch

from eval_harness.kv_compression.base import KVCompressor
from eval_harness.kv_compression.compressors.chunk_sketch import ChunkSketch
from eval_harness.kv_compression.compressors.knorm_sketch import KnormSketch
from eval_harness.kv_compression.compressors.random_sketch import RandomSketch
from eval_harness.kv_compression.registry import get_kv_compressor, get_kv_compressor_class
from eval_harness.kv_compression.base import ScorerKVCompressor


def _fake_module(head_dim: int = 4, **extra) -> SimpleNamespace:
    return SimpleNamespace(head_dim=head_dim, **extra)


@dataclass
class _FirstChannelSketch(ScorerKVCompressor):
    """Deterministic test scorer: score = keys[..., 0]."""

    def score(self, module, hidden_states, keys, values, attentions, kwargs):
        return keys[..., 0]


@dataclass
class _RecordingSketch(ScorerKVCompressor):
    """Records post_init_from_model and per-chunk score() call arguments."""

    def __post_init__(self):
        super().__post_init__()
        self.models = []
        self.calls = []

    def post_init_from_model(self, model):
        self.models.append(model)

    def score(self, module, hidden_states, keys, values, attentions, kwargs):
        self.calls.append(
            (tuple(hidden_states.shape), tuple(keys.shape), tuple(values.shape), attentions, kwargs)
        )
        return keys[..., 0]


def _chunk_press_reference(scorer, module, hidden_states, keys, values, chunk_length, kwargs):
    """Verbatim transcription of kvpress ChunkPress.compress (chunk_press.py:60-87)."""
    if scorer.compression_ratio == 0:
        return keys, values
    kv_len = keys.shape[2]
    indices = []
    for i in range(0, kv_len, chunk_length):
        chunk_scores = scorer.score(
            module,
            hidden_states[:, i : i + chunk_length],
            keys[:, :, i : i + chunk_length],
            values[:, :, i : i + chunk_length],
            None,
            kwargs,
        )
        actual_chunk_length = keys[:, :, i : i + chunk_length].shape[2]
        n_kept = max(1, int(actual_chunk_length * (1 - scorer.compression_ratio)))
        indices.append(i + chunk_scores.topk(n_kept, dim=-1).indices)
    indices = torch.cat(indices, dim=-1)
    indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)
    return keys.gather(2, indices).contiguous(), values.gather(2, indices).contiguous()


def _norm_encoded_kv(norms, head_dim):
    """keys[0, 0, t] = [norms[t], 0, ...] so ||k_t|| == norms[t]; values[0, 0, t] = [t, t, ...]."""
    seq = len(norms)
    keys = torch.zeros(1, 1, seq, head_dim)
    keys[0, 0, :, 0] = torch.tensor(norms, dtype=torch.float32)
    values = torch.arange(seq, dtype=torch.float32).view(1, 1, seq, 1).expand(1, 1, seq, head_dim).contiguous()
    return keys, values


class TestChunkSketchConstruction(unittest.TestCase):
    def test_registry_resolution(self):
        self.assertIs(get_kv_compressor_class("chunk"), ChunkSketch)
        sketch = get_kv_compressor("chunk", press=KnormSketch(compression_ratio=0.5), chunk_length=4)
        self.assertIsInstance(sketch, ChunkSketch)
        self.assertEqual(sketch.chunk_length, 4)
        self.assertEqual(sketch.compression_ratio, 0.5)

    def test_default_chunk_length(self):
        self.assertEqual(ChunkSketch(press=KnormSketch()).chunk_length, 1024)

    def test_post_init_rejects_non_scorer_press(self):
        for bad in (None, object(), "knorm", KVCompressor()):
            with self.assertRaises(AssertionError):
                ChunkSketch(press=bad)

    def test_compression_ratio_property_delegates(self):
        press = KnormSketch(compression_ratio=0.25)
        sketch = ChunkSketch(press=press)
        self.assertEqual(sketch.compression_ratio, 0.25)
        sketch.compression_ratio = 0.5
        self.assertEqual(press.compression_ratio, 0.5)
        self.assertEqual(sketch.compression_ratio, 0.5)

    def test_post_init_from_model_forwards_to_wrapped_press(self):
        press = _RecordingSketch(compression_ratio=0.5)
        sketch = ChunkSketch(press=press)
        sentinel = object()
        sketch.post_init_from_model(sentinel)
        self.assertEqual(press.models, [sentinel])


class TestChunkSketchCompress(unittest.TestCase):
    def test_zero_ratio_is_noop_identity(self):
        keys = torch.randn(1, 2, 16, 8)
        values = torch.randn(1, 2, 16, 8)
        hidden = torch.randn(1, 16, 6)
        sketch = ChunkSketch(press=KnormSketch(compression_ratio=0.0), chunk_length=4)
        out_keys, out_values = sketch.compress(_fake_module(head_dim=8), hidden, keys, values, None, {})
        self.assertIs(out_keys, keys)
        self.assertIs(out_values, values)

    def test_zero_ratio_short_circuits_before_attentions_assert(self):
        keys = torch.randn(1, 2, 16, 8)
        values = torch.randn(1, 2, 16, 8)
        hidden = torch.randn(1, 16, 6)
        sketch = ChunkSketch(press=KnormSketch(compression_ratio=0.0), chunk_length=4)
        out_keys, out_values = sketch.compress(
            _fake_module(head_dim=8), hidden, keys, values, torch.ones(1, 2, 16, 16), {}
        )
        self.assertIs(out_keys, keys)
        self.assertIs(out_values, values)

    def test_attentions_assert_raises_when_compressing(self):
        sketch = ChunkSketch(press=KnormSketch(compression_ratio=0.5), chunk_length=4)
        with self.assertRaisesRegex(AssertionError, "attentions"):
            sketch.compress(
                _fake_module(head_dim=8),
                torch.randn(1, 16, 6),
                torch.randn(1, 2, 16, 8),
                torch.randn(1, 2, 16, 8),
                torch.ones(1, 2, 16, 16),
                {},
            )

    def test_hand_computed_knorm_selection(self):
        # norms per position: chunk0 [1,9,2,8] -> knorm keeps {0,2}; chunk1 [3,7,4,6] -> keeps {4,6}.
        # topk descending-score order pins the exact global index order [0, 2, 4, 6].
        keys, values = _norm_encoded_kv([1, 9, 2, 8, 3, 7, 4, 6], head_dim=2)
        hidden = torch.zeros(1, 8, 4)
        sketch = ChunkSketch(press=KnormSketch(compression_ratio=0.5), chunk_length=4)
        out_keys, out_values = sketch.compress(_fake_module(head_dim=2), hidden, keys, values, None, {})
        expected = torch.tensor([0, 2, 4, 6])
        self.assertTrue(torch.equal(out_keys, keys[:, :, expected]))
        self.assertTrue(torch.equal(out_values, values[:, :, expected]))

    def test_per_chunk_floor_and_short_last_chunk(self):
        # S=10, chunk_length=4, ratio=0.9 -> chunks of length 4, 4, 2; n_kept = max(1, 0) = 1 each.
        # Norm minima engineered at positions 3, 5, 9 -> exact kept order [3, 5, 9].
        keys, values = _norm_encoded_kv([5, 6, 7, 1, 5, 1, 6, 7, 5, 1], head_dim=2)
        hidden = torch.zeros(1, 10, 4)
        sketch = ChunkSketch(press=KnormSketch(compression_ratio=0.9), chunk_length=4)
        out_keys, out_values = sketch.compress(_fake_module(head_dim=2), hidden, keys, values, None, {})
        self.assertEqual(out_keys.shape, (1, 1, 3, 2))
        expected = torch.tensor([3, 5, 9])
        self.assertTrue(torch.equal(out_keys, keys[:, :, expected]))
        self.assertTrue(torch.equal(out_values, values[:, :, expected]))
        kept_positions = out_values[0, 0, :, 0].tolist()
        for kept, (lo, hi) in zip(kept_positions, [(0, 4), (4, 8), (8, 10)]):
            self.assertTrue(lo <= kept < hi)

    def test_single_chunk_matches_global_scorer(self):
        gen = torch.Generator().manual_seed(7)
        keys = torch.randn(1, 2, 6, 4, generator=gen)
        values = torch.randn(1, 2, 6, 4, generator=gen)
        hidden = torch.randn(1, 6, 6, generator=gen)
        module = _fake_module(head_dim=4)
        chunk_out = ChunkSketch(press=KnormSketch(compression_ratio=0.5), chunk_length=1024).compress(
            module, hidden, keys, values, None, {}
        )
        global_out = KnormSketch(compression_ratio=0.5).compress(module, hidden, keys, values, None, {})
        self.assertEqual(chunk_out[0].shape, (1, 2, 3, 4))
        self.assertTrue(torch.equal(chunk_out[0], global_out[0]))
        self.assertTrue(torch.equal(chunk_out[1], global_out[1]))

    def test_floor_keeps_one_token_where_global_keeps_zero(self):
        # S=3, ratio=0.9: global int(3 * 0.1) == 0 keeps nothing; ChunkSketch floor keeps max(1, 0) == 1.
        keys, values = _norm_encoded_kv([3, 1, 2], head_dim=2)
        hidden = torch.zeros(1, 3, 4)
        module = _fake_module(head_dim=2)
        chunk_out = ChunkSketch(press=KnormSketch(compression_ratio=0.9), chunk_length=1024).compress(
            module, hidden, keys, values, None, {}
        )
        self.assertEqual(chunk_out[0].shape, (1, 1, 1, 2))
        self.assertTrue(torch.equal(chunk_out[0], keys[:, :, [1]]))
        self.assertTrue(torch.equal(chunk_out[1], values[:, :, [1]]))
        global_out = KnormSketch(compression_ratio=0.9).compress(module, hidden, keys, values, None, {})
        self.assertEqual(global_out[0].shape, (1, 1, 0, 2))

    def test_reference_transcription_oracle(self):
        gen = torch.Generator().manual_seed(1234)
        keys = torch.randn(2, 4, 37, 8, generator=gen)
        values = torch.randn(2, 4, 37, 8, generator=gen)
        hidden = torch.randn(2, 37, 16, generator=gen)
        module = _fake_module(head_dim=8)
        kwargs = {"cache_position": torch.arange(37)}
        scorer = _FirstChannelSketch(compression_ratio=0.25)
        out_keys, out_values = ChunkSketch(press=scorer, chunk_length=8).compress(
            module, hidden, keys, values, None, kwargs
        )
        ref_keys, ref_values = _chunk_press_reference(scorer, module, hidden, keys, values, 8, kwargs)
        # chunks 8,8,8,8,5 -> int(8*0.75)=6 per full chunk, max(1, int(5*0.75))=3 for the tail.
        self.assertEqual(out_keys.shape, (2, 4, 27, 8))
        self.assertTrue(torch.equal(out_keys, ref_keys))
        self.assertTrue(torch.equal(out_values, ref_values))

    def test_wrapped_scorer_sees_sliced_tensors_and_unsliced_kwargs(self):
        scorer = _RecordingSketch(compression_ratio=0.5)
        kwargs = {"cache_position": torch.arange(10)}
        keys = torch.randn(1, 2, 10, 4)
        values = torch.randn(1, 2, 10, 4)
        hidden = torch.randn(1, 10, 6)
        ChunkSketch(press=scorer, chunk_length=4).compress(_fake_module(head_dim=4), hidden, keys, values, None, kwargs)
        self.assertEqual(len(scorer.calls), 3)
        hidden_shapes = [call[0] for call in scorer.calls]
        key_shapes = [call[1] for call in scorer.calls]
        value_shapes = [call[2] for call in scorer.calls]
        self.assertEqual(hidden_shapes, [(1, 4, 6), (1, 4, 6), (1, 2, 6)])
        self.assertEqual(key_shapes, [(1, 2, 4, 4), (1, 2, 4, 4), (1, 2, 2, 4)])
        self.assertEqual(value_shapes, key_shapes)
        for call in scorer.calls:
            self.assertIsNone(call[3])
            self.assertIs(call[4], kwargs)

    def test_per_head_index_freedom_with_rectangular_output(self):
        heads, seq, head_dim = 2, 32, 8
        keys = torch.randn(1, heads, seq, head_dim, generator=torch.Generator().manual_seed(3))
        positions = torch.arange(seq, dtype=torch.float32).view(1, 1, seq, 1)
        head_offsets = torch.tensor([0.0, 1000.0]).view(1, heads, 1, 1)
        values = (positions + head_offsets).expand(1, heads, seq, head_dim).contiguous()
        sketch = ChunkSketch(press=RandomSketch(seed=0, compression_ratio=0.5), chunk_length=8)
        out_keys, out_values = sketch.compress(_fake_module(head_dim=head_dim), torch.zeros(1, seq, 4), keys, values, None, {})
        self.assertEqual(out_keys.shape, (1, heads, 16, head_dim))
        self.assertEqual(out_values.shape, (1, heads, 16, head_dim))
        kept = [
            {int(p) for p in (out_values[0, h, :, 0] - 1000.0 * h).tolist()}
            for h in range(heads)
        ]
        for head_kept in kept:
            self.assertEqual(len(head_kept), 16)
            for lo in range(0, seq, 8):
                self.assertEqual(len([p for p in head_kept if lo <= p < lo + 8]), 4)
        self.assertNotEqual(kept[0], kept[1])

    def test_cross_layer_kept_count_is_deterministic(self):
        sketch = ChunkSketch(press=KnormSketch(compression_ratio=0.25), chunk_length=8)
        lengths = []
        for layer_idx in (0, 1):
            gen = torch.Generator().manual_seed(100 + layer_idx)
            keys = torch.randn(1, 2, 37, 8, generator=gen)
            values = torch.randn(1, 2, 37, 8, generator=gen)
            hidden = torch.randn(1, 37, 6, generator=gen)
            module = _fake_module(head_dim=8, layer_idx=layer_idx)
            out_keys, _ = sketch.compress(module, hidden, keys, values, None, {})
            lengths.append(out_keys.shape[2])
        self.assertEqual(lengths[0], lengths[1])
        self.assertEqual(lengths[0], 4 * 6 + 3)


if __name__ == "__main__":
    unittest.main()
