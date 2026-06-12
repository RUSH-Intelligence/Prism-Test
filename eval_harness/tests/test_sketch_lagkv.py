"""Tests for LagKVSketch (port of kvpress LagKVPress).

Reference oracle `_lagkv_reference` is an independent transcription of
kvpress/presses/lagkv_press.py (kvpress 0.5.1); hand-computed fixtures pin
exact values for both `cross_scoring` settings and both score branches.
"""

from __future__ import annotations

import unittest

import torch

from eval_harness.sketch.sketches.lagkv_sketch import LagKVSketch
from eval_harness.sketch.sketches.registry import (
    available_sketches,
    get_sketch,
    get_sketch_class,
)


class _FakeAttnModule:
    def __init__(self, head_dim: int, layer_idx: int = 0):
        self.head_dim = head_dim
        self.layer_idx = layer_idx


def _lagkv_reference(
    keys: torch.Tensor,
    values: torch.Tensor,
    n_sink: int,
    lag_size: int,
    cross_scoring: bool,
) -> torch.Tensor:
    bsz, num_key_value_heads, q_len, d = keys.shape
    if q_len < n_sink + 2 * lag_size:
        score = torch.ones((bsz, num_key_value_heads, q_len), dtype=keys.dtype, device=keys.device)
        if q_len > n_sink:
            score[:, :, n_sink:] = (
                torch.arange(q_len - n_sink, device=keys.device) / (q_len - n_sink)
            ).to(keys.dtype)
        return score

    end_idx = n_sink + ((q_len - n_sink) // lag_size) * lag_size
    tail_len = lag_size + q_len - end_idx

    def states_score(target_v: torch.Tensor) -> torch.Tensor:
        ref = target_v[:, :, 1:, :, :]
        v = target_v[:, :, :-1, :, :]
        min_r = ref.min(dim=-2).values.unsqueeze(-2).expand(-1, -1, -1, lag_size, -1)
        max_r = ref.max(dim=-2).values.unsqueeze(-2).expand(-1, -1, -1, lag_size, -1)
        return ((v - min_r) / (max_r - min_r)).std(dim=-1).softmax(dim=-1)

    key_score = states_score(keys[:, :, n_sink:end_idx].view(bsz, num_key_value_heads, -1, lag_size, d))
    value_score = states_score(values[:, :, n_sink:end_idx].view(bsz, num_key_value_heads, -1, lag_size, d))
    score = (key_score + value_score) / 2
    if not cross_scoring:
        score = (score.argsort(dim=-1).argsort(dim=-1) / lag_size).to(keys.dtype)
    sink_score = torch.ones((bsz, num_key_value_heads, n_sink), dtype=score.dtype, device=score.device)
    tail_score = torch.ones((bsz, num_key_value_heads, tail_len), dtype=score.dtype, device=score.device)
    return torch.cat((sink_score, score.reshape(bsz, num_key_value_heads, -1), tail_score), dim=-1)


def _hand_fixture() -> tuple[torch.Tensor, torch.Tensor]:
    tokens = torch.tensor(
        [
            [9.0, 9.0],
            [0.5, 0.5],
            [0.0, 1.0],
            [0.0, 0.0],
            [1.0, 1.0],
        ]
    )
    keys = tokens.view(1, 1, 5, 2)
    return keys, keys.clone()


def _kept_positions(original: torch.Tensor, compressed: torch.Tensor) -> list[int]:
    positions = []
    for row in compressed:
        matches = (original == row).all(dim=-1).nonzero().flatten().tolist()
        assert len(matches) == 1, f"row matched {len(matches)} original positions"
        positions.append(matches[0])
    return positions


class TestLagKVRegistry(unittest.TestCase):
    def test_registered_name(self):
        self.assertIn("lagkv", available_sketches())
        self.assertIs(get_sketch_class("lagkv"), LagKVSketch)

    def test_get_sketch_instantiates_with_kwargs(self):
        sketch = get_sketch(
            "lagkv", compression_ratio=0.25, n_sink=2, lag_size=16, cross_scoring=True
        )
        self.assertIsInstance(sketch, LagKVSketch)
        self.assertAlmostEqual(sketch.compression_ratio, 0.25)
        self.assertEqual(sketch.n_sink, 2)
        self.assertEqual(sketch.lag_size, 16)
        self.assertTrue(sketch.cross_scoring)

    def test_defaults_match_kvpress(self):
        sketch = LagKVSketch()
        self.assertEqual(sketch.compression_ratio, 0.0)
        self.assertEqual(sketch.n_sink, 4)
        self.assertEqual(sketch.lag_size, 128)
        self.assertFalse(sketch.cross_scoring)


class TestLagKVScore(unittest.TestCase):
    def test_zero_ratio_noop_and_score_never_called(self):
        class _Boom(LagKVSketch):
            def score(self, *args, **kwargs):
                raise AssertionError("score must not be called when compression_ratio == 0")

        sketch = _Boom(compression_ratio=0.0)
        keys = torch.randn(1, 2, 12, 4)
        values = torch.randn(1, 2, 12, 4)
        out_keys, out_values = sketch.compress(
            _FakeAttnModule(head_dim=4), None, keys, values, None, {}
        )
        self.assertIs(out_keys, keys)
        self.assertIs(out_values, values)

    def test_main_branch_hand_computed_rank_scores(self):
        keys, values = _hand_fixture()
        sketch = LagKVSketch(compression_ratio=0.2, n_sink=1, lag_size=2)
        score = sketch.score(None, None, keys, values, None, {})
        expected = torch.tensor([[[1.0, 0.0, 0.5, 1.0, 1.0]]])
        self.assertTrue(torch.equal(score, expected))

        out_keys, out_values = sketch.compress(
            _FakeAttnModule(head_dim=2), None, keys, values, None, {}
        )
        self.assertEqual(tuple(out_keys.shape), (1, 1, 4, 2))
        self.assertEqual(sorted(_kept_positions(keys[0, 0], out_keys[0, 0])), [0, 2, 3, 4])
        self.assertEqual(sorted(_kept_positions(values[0, 0], out_values[0, 0])), [0, 2, 3, 4])

    def test_main_branch_hand_computed_cross_scoring(self):
        keys, values = _hand_fixture()
        sketch = LagKVSketch(compression_ratio=0.2, n_sink=1, lag_size=2, cross_scoring=True)
        score = sketch.score(None, None, keys, values, None, {})
        expected = torch.tensor([[[1.0, 0.33024, 0.66976, 1.0, 1.0]]])
        torch.testing.assert_close(score, expected, atol=1e-4, rtol=0.0)

    def test_fallback_branch_sink_plus_recency_ramp(self):
        torch.manual_seed(0)
        keys = torch.randn(1, 2, 10, 4)
        values = torch.randn(1, 2, 10, 4)
        sketch = LagKVSketch(compression_ratio=0.5)
        score = sketch.score(None, None, keys, values, None, {})
        expected = torch.cat([torch.ones(4), torch.arange(6) / 6]).view(1, 1, 10).expand(1, 2, 10)
        self.assertTrue(torch.equal(score, expected))

        out_keys, _ = sketch.compress(_FakeAttnModule(head_dim=4), None, keys, values, None, {})
        self.assertEqual(tuple(out_keys.shape), (1, 2, 5, 4))
        for h in range(2):
            self.assertEqual(
                sorted(_kept_positions(keys[0, h], out_keys[0, h])), [0, 1, 2, 3, 9]
            )

    def test_fallback_qlen_at_most_nsink_all_ones(self):
        keys = torch.randn(1, 1, 3, 4)
        values = torch.randn(1, 1, 3, 4)
        sketch = LagKVSketch(compression_ratio=0.3)
        score = sketch.score(None, None, keys, values, None, {})
        self.assertTrue(torch.equal(score, torch.ones(1, 1, 3)))

    def test_geometry_remainder_and_rank_property(self):
        torch.manual_seed(1)
        keys = torch.randn(1, 1, 20, 4)
        values = torch.randn(1, 1, 20, 4)
        sketch = LagKVSketch(compression_ratio=0.5, n_sink=2, lag_size=4)
        score = sketch.score(None, None, keys, values, None, {})
        self.assertEqual(tuple(score.shape), (1, 1, 20))
        self.assertTrue(torch.equal(score[..., :2], torch.ones(1, 1, 2)))
        self.assertTrue(torch.equal(score[..., 14:], torch.ones(1, 1, 6)))
        for start in (2, 6, 10):
            partition = score[0, 0, start : start + 4]
            self.assertEqual(sorted(partition.tolist()), [0.0, 0.25, 0.5, 0.75])

    def test_reference_transcription_oracle_both_modes(self):
        torch.manual_seed(0)
        keys = torch.randn(2, 3, 70, 8)
        values = torch.randn(2, 3, 70, 8)
        for cross_scoring in (False, True):
            with self.subTest(cross_scoring=cross_scoring):
                sketch = LagKVSketch(
                    compression_ratio=0.5, n_sink=4, lag_size=16, cross_scoring=cross_scoring
                )
                score = sketch.score(None, None, keys, values, None, {})
                expected = _lagkv_reference(keys, values, 4, 16, cross_scoring)
                self.assertEqual(tuple(score.shape), (2, 3, 70))
                self.assertTrue(torch.equal(score, expected))

    def test_gqa_uniform_count_per_head_selection(self):
        torch.manual_seed(2)
        keys = torch.randn(2, 2, 40, 16)
        values = torch.randn(2, 2, 40, 16)
        sketch = LagKVSketch(compression_ratio=0.5, n_sink=4, lag_size=8)
        score = sketch.score(_FakeAttnModule(head_dim=16), None, keys, values, None, {})
        self.assertEqual(tuple(score.shape), (2, 2, 40))

        out_keys, out_values = sketch.compress(
            _FakeAttnModule(head_dim=16), None, keys, values, None, {}
        )
        self.assertEqual(tuple(out_keys.shape), (2, 2, 20, 16))
        self.assertEqual(tuple(out_values.shape), (2, 2, 20, 16))
        selections = [
            frozenset(_kept_positions(keys[b, h], out_keys[b, h]))
            for b in range(2)
            for h in range(2)
        ]
        self.assertEqual(len(set(selections)), 4)

    def test_sink_and_tail_always_kept(self):
        torch.manual_seed(3)
        keys = torch.randn(1, 2, 40, 8)
        values = torch.randn(1, 2, 40, 8)
        sketch = LagKVSketch(compression_ratio=0.5, n_sink=4, lag_size=8)
        out_keys, _ = sketch.compress(_FakeAttnModule(head_dim=8), None, keys, values, None, {})
        for h in range(2):
            kept = set(_kept_positions(keys[0, h], out_keys[0, h]))
            self.assertTrue(set(range(4)).issubset(kept))
            self.assertTrue(set(range(28, 40)).issubset(kept))

    def test_bfloat16_both_branches(self):
        torch.manual_seed(4)
        for seq_len, kwargs in ((10, {}), (70, {"n_sink": 4, "lag_size": 16})):
            for cross_scoring in (False, True):
                with self.subTest(seq_len=seq_len, cross_scoring=cross_scoring):
                    keys = torch.randn(1, 2, seq_len, 8).to(torch.bfloat16)
                    values = torch.randn(1, 2, seq_len, 8).to(torch.bfloat16)
                    sketch = LagKVSketch(
                        compression_ratio=0.5, cross_scoring=cross_scoring, **kwargs
                    )
                    score = sketch.score(None, None, keys, values, None, {})
                    self.assertEqual(score.dtype, torch.bfloat16)
                    self.assertEqual(tuple(score.shape), (1, 2, seq_len))
                    out_keys, _ = sketch.compress(
                        _FakeAttnModule(head_dim=8), None, keys, values, None, {}
                    )
                    self.assertEqual(tuple(out_keys.shape), (1, 2, seq_len // 2, 8))

    def test_nan_parity_constant_input(self):
        keys = torch.ones(1, 1, 5, 2)
        values = torch.ones(1, 1, 5, 2)

        sketch = LagKVSketch(compression_ratio=0.2, n_sink=1, lag_size=2)
        score = sketch.score(None, None, keys, values, None, {})
        self.assertTrue(torch.isfinite(score).all())
        self.assertTrue(torch.equal(score, torch.tensor([[[1.0, 0.0, 0.5, 1.0, 1.0]]])))

        sketch_cross = LagKVSketch(
            compression_ratio=0.2, n_sink=1, lag_size=2, cross_scoring=True
        )
        score_cross = sketch_cross.score(None, None, keys, values, None, {})
        self.assertTrue(torch.isnan(score_cross[0, 0, 1:3]).all())
        self.assertTrue(torch.equal(score_cross[0, 0, [0, 3, 4]], torch.ones(3)))


if __name__ == "__main__":
    unittest.main()
