"""Unit tests for StreamingLLMSketch (port of kvpress 0.5.1 StreamingLLMPress).

No model loading; fake attention modules only. The kvpress score/selection math
is transcribed inline as a reference oracle (kvpress is not importable in
prism_env).
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from eval_harness.sketch.sketches.registry import get_sketch, get_sketch_class
from eval_harness.sketch.sketches.streaming_llm_sketch import StreamingLLMSketch


def _fake_module(head_dim: int = 4, **extra) -> SimpleNamespace:
    return SimpleNamespace(head_dim=head_dim, **extra)


def _position_encoded_kv(batch: int, heads: int, seq: int, head_dim: int):
    positions = torch.arange(seq, dtype=torch.float32).view(1, 1, seq, 1)
    keys = positions.expand(batch, heads, seq, head_dim).contiguous()
    values = (positions + 100.0).expand(batch, heads, seq, head_dim).contiguous()
    return keys, values


def _kvpress_reference(keys: torch.Tensor, compression_ratio: float, n_sink: int):
    """Verbatim transcription of kvpress StreamingLLMPress.score + ScorerPress.compress
    index selection (streaming_llm_press.py:48-54, scorer_press.py:94-95)."""
    k_len = keys.shape[2]
    n_pruned = k_len - int(k_len * (1 - compression_ratio))
    scores = torch.ones_like(keys[..., 0])
    scores[:, :, n_sink : n_sink + n_pruned] = 0
    n_kept = int(k_len * (1 - compression_ratio))
    indices = scores.topk(n_kept, dim=-1).indices
    return scores, indices


class TestStreamingLLMScore(unittest.TestCase):
    def test_pinned_score_tensor(self):
        keys = torch.randn(1, 2, 8, 4)
        sketch = StreamingLLMSketch(compression_ratio=0.25, n_sink=4)
        scores = sketch.score(_fake_module(), None, keys, keys.clone(), None, {})
        expected = torch.tensor([1.0, 1, 1, 1, 0, 0, 1, 1]).view(1, 1, 8).expand(1, 2, 8)
        self.assertTrue(torch.equal(scores, expected))
        self.assertEqual(scores.dtype, keys.dtype)
        self.assertEqual(scores.shape, (1, 2, 8))

    def test_score_dtype_follows_keys_bf16(self):
        keys = torch.randn(1, 2, 8, 4).to(torch.bfloat16)
        sketch = StreamingLLMSketch(compression_ratio=0.25, n_sink=4)
        scores = sketch.score(_fake_module(), None, keys, keys.clone(), None, {})
        self.assertEqual(scores.dtype, torch.bfloat16)
        expected = torch.tensor([1.0, 1, 1, 1, 0, 0, 1, 1], dtype=torch.bfloat16)
        self.assertTrue(torch.equal(scores, expected.view(1, 1, 8).expand(1, 2, 8)))

    def test_scores_independent_of_key_values(self):
        sketch = StreamingLLMSketch(compression_ratio=0.5, n_sink=2)
        a = sketch.score(_fake_module(), None, torch.randn(1, 2, 10, 4), None, None, {})
        b = sketch.score(_fake_module(), None, torch.randn(1, 2, 10, 4) * 1e3, None, None, {})
        self.assertTrue(torch.equal(a, b))


class TestStreamingLLMCompress(unittest.TestCase):
    def test_zero_ratio_is_noop_identity(self):
        keys = torch.randn(1, 2, 10, 4)
        values = torch.randn(1, 2, 10, 4)
        sketch = StreamingLLMSketch(compression_ratio=0.0, n_sink=4)
        out_keys, out_values = sketch.compress(_fake_module(), None, keys, values, None, {})
        self.assertIs(out_keys, keys)
        self.assertIs(out_values, values)

    def test_zero_ratio_short_circuits_before_assert(self):
        keys = torch.randn(1, 2, 3, 4)
        values = torch.randn(1, 2, 3, 4)
        sketch = StreamingLLMSketch(compression_ratio=0.0, n_sink=4)
        out_keys, out_values = sketch.compress(_fake_module(), None, keys, values, None, {})
        self.assertIs(out_keys, keys)
        self.assertIs(out_values, values)

    def test_exact_selection_hand_computed(self):
        keys, values = _position_encoded_kv(1, 2, 10, 4)
        sketch = StreamingLLMSketch(compression_ratio=0.5, n_sink=2)
        out_keys, out_values = sketch.compress(_fake_module(), None, keys, values, None, {})
        self.assertEqual(out_keys.shape, (1, 2, 5, 4))
        self.assertEqual(out_values.shape, (1, 2, 5, 4))
        for h in range(2):
            self.assertEqual(sorted(out_keys[0, h, :, 0].tolist()), [0.0, 1.0, 7.0, 8.0, 9.0])
            self.assertEqual(
                sorted(out_values[0, h, :, 0].tolist()), [100.0, 101.0, 107.0, 108.0, 109.0]
            )

    def test_floor_rounding_n_kept(self):
        keys, values = _position_encoded_kv(1, 2, 7, 4)
        sketch = StreamingLLMSketch(compression_ratio=0.5, n_sink=2)
        out_keys, out_values = sketch.compress(_fake_module(), None, keys, values, None, {})
        self.assertEqual(out_keys.shape, (1, 2, 3, 4))
        for h in range(2):
            self.assertEqual(sorted(out_keys[0, h, :, 0].tolist()), [0.0, 1.0, 6.0])
            self.assertEqual(sorted(out_values[0, h, :, 0].tolist()), [100.0, 101.0, 106.0])

    def test_gqa_never_touches_query_heads(self):
        keys, values = _position_encoded_kv(1, 2, 12, 4)
        module = _fake_module(num_key_value_groups=4)
        sketch = StreamingLLMSketch(compression_ratio=0.5, n_sink=4)
        scores = sketch.score(module, None, keys, values, None, {})
        self.assertEqual(scores.shape, (1, 2, 12))
        out_keys, out_values = sketch.compress(module, None, keys, values, None, {})
        self.assertEqual(out_keys.shape, (1, 2, 6, 4))
        self.assertEqual(out_values.shape, (1, 2, 6, 4))
        for h in range(2):
            self.assertEqual(
                sorted(out_keys[0, h, :, 0].tolist()), [0.0, 1.0, 2.0, 3.0, 10.0, 11.0]
            )

    def test_assert_when_seq_not_longer_than_sinks(self):
        keys, values = _position_encoded_kv(1, 2, 4, 4)
        sketch = StreamingLLMSketch(compression_ratio=0.5, n_sink=4)
        with self.assertRaisesRegex(AssertionError, "n_sink=4"):
            sketch.compress(_fake_module(), None, keys, values, None, {})

    def test_degenerate_high_ratio_keeps_sink_subset(self):
        keys, values = _position_encoded_kv(1, 2, 6, 4)
        sketch = StreamingLLMSketch(compression_ratio=0.7, n_sink=4)
        out_keys, out_values = sketch.compress(_fake_module(), None, keys, values, None, {})
        self.assertEqual(out_keys.shape, (1, 2, 1, 4))
        for h in range(2):
            kept = out_keys[0, h, :, 0].tolist()
            self.assertEqual(len(kept), 1)
            self.assertIn(kept[0], [0.0, 1.0, 2.0, 3.0])

    def test_batch_independence(self):
        keys, values = _position_encoded_kv(2, 2, 10, 4)
        keys = keys + torch.randn(2, 2, 10, 4) * 0.0  # shape sanity, scores ignore values
        sketch = StreamingLLMSketch(compression_ratio=0.5, n_sink=2)
        out_keys, _ = sketch.compress(_fake_module(), None, keys, values, None, {})
        self.assertEqual(out_keys.shape, (2, 2, 5, 4))
        for b in range(2):
            for h in range(2):
                self.assertEqual(
                    sorted(out_keys[b, h, :, 0].tolist()), [0.0, 1.0, 7.0, 8.0, 9.0]
                )


class TestStreamingLLMReferenceOracle(unittest.TestCase):
    def test_matches_kvpress_transcription(self):
        torch.manual_seed(0)
        cases = [(16, 0.5, 4), (10, 0.3, 2), (5, 0.2, 4), (9, 0.85, 3)]
        for seq, ratio, n_sink in cases:
            with self.subTest(seq=seq, ratio=ratio, n_sink=n_sink):
                keys = torch.randn(1, 2, seq, 4)
                values = torch.randn(1, 2, seq, 4)
                values[..., 0] = torch.arange(seq, dtype=values.dtype)
                ref_scores, ref_indices = _kvpress_reference(keys, ratio, n_sink)

                sketch = StreamingLLMSketch(compression_ratio=ratio, n_sink=n_sink)
                scores = sketch.score(_fake_module(), None, keys, values, None, {})
                self.assertTrue(torch.equal(scores, ref_scores))

                _, out_values = sketch.compress(_fake_module(), None, keys, values, None, {})
                for h in range(2):
                    kept = sorted(int(p) for p in out_values[0, h, :, 0].tolist())
                    self.assertEqual(kept, sorted(ref_indices[0, h].tolist()))


class TestStreamingLLMRegistry(unittest.TestCase):
    def test_registry_resolution(self):
        self.assertIs(get_sketch_class("streaming_llm"), StreamingLLMSketch)

    def test_get_sketch_sets_fields(self):
        sketch = get_sketch("streaming_llm", compression_ratio=0.5, n_sink=2)
        self.assertIsInstance(sketch, StreamingLLMSketch)
        self.assertAlmostEqual(sketch.compression_ratio, 0.5)
        self.assertEqual(sketch.n_sink, 2)

    def test_defaults_mirror_kvpress(self):
        sketch = StreamingLLMSketch()
        self.assertEqual(sketch.compression_ratio, 0.0)
        self.assertEqual(sketch.n_sink, 4)


if __name__ == "__main__":
    unittest.main()
