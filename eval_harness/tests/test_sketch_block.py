"""Tests for BlockSketch (port of kvpress BlockPress).

Reference loop (kvpress/presses/block_press.py:49-98): streaming/iterative
block-wise top-k — initialize the kept set with the first ``n_kept`` tokens,
then for each subsequent block score (kept ∪ block) with the wrapped scorer
and keep the ``n_kept`` best, mapping back to absolute positions. For
per-token scorers (Knorm) the kept SET equals the global top-k; for
set-dependent scorers (KeyDiff) results depend on ``block_size``.
"""

from __future__ import annotations

import unittest

import torch
from torch import nn
from torch.nn import functional as F
from transformers import DynamicCache

from eval_harness.kv_compression.compressors.block_sketch import BlockSketch
from eval_harness.kv_compression.compressors.knorm_sketch import KnormSketch
from eval_harness.kv_compression.registry import (
    available_kv_compressors,
    get_kv_compressor,
    get_kv_compressor_class,
)
from eval_harness.kv_compression.base import ScorerKVCompressor


class _FakeAttnModule(nn.Module):
    """Minimal attention module: BlockSketch.compress reads nothing from the
    module (head_dim comes from keys.shape), but the plain ScorerKVCompressor path
    uses ``head_dim`` and the forward hook uses ``layer_idx``."""

    def __init__(self, num_heads=4, num_kv_heads=2, head_dim=8, layer_idx=0):
        super().__init__()
        self.num_heads = num_heads
        self.num_key_value_heads = num_kv_heads
        self.head_dim = head_dim
        self.layer_idx = layer_idx


class _KeyDiffScorer(ScorerKVCompressor):
    """In-test transcription of kvpress KeyDiffPress.score
    (keydiff_press.py:46-47) — a set-dependent scorer."""

    def score(self, module, hidden_states, keys, values, attentions, kwargs):
        anchor = F.normalize(keys, p=2, dim=-1).mean(dim=2, keepdim=True)
        return -F.cosine_similarity(keys, anchor, dim=-1)


class _QueueScorer(ScorerKVCompressor):
    """Probe scorer: captures every score() input and returns pre-rigged
    score tensors from a queue (raises IndexError if called too often)."""

    def __post_init__(self):
        super().__post_init__()
        self.queue = []
        self.captured = []

    def score(self, module, hidden_states, keys, values, attentions, kwargs):
        self.captured.append(
            (
                hidden_states.detach().clone(),
                keys.detach().clone(),
                values.detach().clone(),
            )
        )
        return self.queue.pop(0)


class _RecordingScorer(ScorerKVCompressor):
    def post_init_from_model(self, model):
        self.seen_model = model

    def score(self, module, hidden_states, keys, values, attentions, kwargs):
        return -keys.norm(dim=-1)


def _block_press_oracle(score_fn, keys, values, n_kept, block_size):
    """Plain-Python per-(batch, head) reimplementation of the BlockPress loop
    (block_press.py:65-96). ``score_fn(sub_keys, sub_values)`` operates on
    [1, 1, n, d] tensors. Returns (keys, values, kept_position_lists)."""
    bsz, num_kv_heads, k_len, head_dim = keys.shape
    out_keys = torch.empty(bsz, num_kv_heads, n_kept, head_dim, dtype=keys.dtype)
    out_values = torch.empty_like(out_keys)
    kept_positions = [[None] * num_kv_heads for _ in range(bsz)]
    for b in range(bsz):
        for h in range(num_kv_heads):
            kept = list(range(n_kept))
            for i in range(n_kept, k_len, block_size):
                end = min(i + block_size, k_len)
                cand = kept + list(range(i, end))
                idx = torch.tensor(cand, dtype=torch.long)
                sub_keys = keys[b : b + 1, h : h + 1].index_select(2, idx)
                sub_values = values[b : b + 1, h : h + 1].index_select(2, idx)
                scores = score_fn(sub_keys, sub_values)[0, 0]
                top = scores.topk(n_kept).indices.tolist()
                kept = [cand[j] for j in top]
            idx = torch.tensor(kept, dtype=torch.long)
            out_keys[b, h] = keys[b, h].index_select(0, idx)
            out_values[b, h] = values[b, h].index_select(0, idx)
            kept_positions[b][h] = kept
    return out_keys, out_values, kept_positions


def _recover_indices(full, subset):
    """Map each subset row back to its (unique) position in ``full``.

    full: [B, H, S, D]; subset: [B, H, n, D] of exact row copies -> [B, H, n].
    """
    eq = (subset.unsqueeze(-2) == full.unsqueeze(-3)).all(-1)
    assert (eq.sum(-1) == 1).all(), "subset rows must match exactly one input row"
    return eq.float().argmax(-1)


class TestBlockSketchRegistry(unittest.TestCase):
    def test_registered_name(self):
        self.assertIn("block", available_kv_compressors())
        self.assertIs(get_kv_compressor_class("block"), BlockSketch)

    def test_get_kv_compressor_instantiates_with_kwargs(self):
        sketch = get_kv_compressor(
            "block", sketch=KnormSketch(compression_ratio=0.5), block_size=4
        )
        self.assertIsInstance(sketch, BlockSketch)
        self.assertEqual(sketch.block_size, 4)
        self.assertAlmostEqual(sketch.compression_ratio, 0.5)


class TestConstructionAndDelegation(unittest.TestCase):
    def test_requires_scorer_sketch(self):
        with self.assertRaises(AssertionError):
            BlockSketch(sketch=object())

    def test_block_size_zero_raises_at_construction(self):
        # Deviation pin: upstream does not validate block_size (UB for <= 0).
        with self.assertRaises(AssertionError):
            BlockSketch(sketch=KnormSketch(compression_ratio=0.5), block_size=0)

    def test_compression_ratio_property_delegates(self):
        inner = KnormSketch(compression_ratio=0.3)
        sketch = BlockSketch(sketch=inner)
        self.assertAlmostEqual(sketch.compression_ratio, 0.3)
        sketch.compression_ratio = 0.7
        self.assertAlmostEqual(inner.compression_ratio, 0.7)

    def test_post_init_from_model_delegates(self):
        inner = _RecordingScorer(compression_ratio=0.5)
        sentinel = object()
        BlockSketch(sketch=inner).post_init_from_model(sentinel)
        self.assertIs(inner.seen_model, sentinel)


class TestZeroRatioNoop(unittest.TestCase):
    def test_returns_same_tensor_objects_without_scoring(self):
        inner = _QueueScorer(compression_ratio=0.0)  # empty queue: score() would raise
        sketch = BlockSketch(sketch=inner, block_size=4)
        module = _FakeAttnModule(num_kv_heads=2, head_dim=8)
        keys = torch.randn(1, 2, 16, 8)
        values = torch.randn(1, 2, 16, 8)
        hidden = torch.randn(1, 16, 16)
        out_keys, out_values = sketch.compress(module, hidden, keys, values, None, {})
        self.assertIs(out_keys, keys)
        self.assertIs(out_values, values)
        self.assertEqual(inner.captured, [])


class TestHandComputedKnormTrace(unittest.TestCase):
    def test_exact_streaming_selection_and_order(self):
        # bsz=1, H_kv=1, k_len=8, head_dim=2, block_size=2, ratio=0.5 -> n_kept=4.
        # keys[0, 0, p] = [8 - p, 0] -> norms [8,7,6,5,4,3,2,1], knorm score = -norm.
        # Trace: kept=[0,1,2,3];
        #   i=4: cand [0,1,2,3,4,5], scores [-8,-7,-6,-5,-4,-3] -> kept [5,4,3,2];
        #   i=6: cand [5,4,3,2,6,7], scores [-3,-4,-5,-6,-2,-1] -> kept [7,6,5,4].
        k_len = 8
        keys = torch.zeros(1, 1, k_len, 2)
        keys[0, 0, :, 0] = torch.arange(k_len, 0, -1, dtype=torch.float32)
        values = torch.stack(
            [torch.arange(k_len, dtype=torch.float32)] * 2, dim=-1
        ).reshape(1, 1, k_len, 2)
        hidden = torch.zeros(1, k_len, 4)
        module = _FakeAttnModule(num_kv_heads=1, head_dim=2)

        sketch = BlockSketch(sketch=KnormSketch(compression_ratio=0.5), block_size=2)
        out_keys, out_values = sketch.compress(module, hidden, keys, values, None, {})

        expected_order = torch.tensor([7, 6, 5, 4])
        self.assertTrue(torch.equal(out_keys, keys[:, :, expected_order]))
        self.assertTrue(torch.equal(out_values, values[:, :, expected_order]))
        # Kept SET equals the global top-4 of the per-token score.
        self.assertEqual(
            set(_recover_indices(keys, out_keys)[0, 0].tolist()), {4, 5, 6, 7}
        )


class TestStreamingTopKEquivalence(unittest.TestCase):
    def test_matches_plain_scorer_for_all_block_sizes(self):
        # Transcription of kvpress tests/presses/test_block_press.py:30-63
        # (order-insensitive sum hash vs the plain press), plus an exact
        # kept-index SET pin against the global top-k.
        torch.manual_seed(0)
        keys = torch.randn(1, 2, 256, 8)
        values = torch.randn(1, 2, 256, 8)
        hidden = torch.randn(1, 256, 16)
        module = _FakeAttnModule(num_kv_heads=2, head_dim=8)

        plain = KnormSketch(compression_ratio=0.5)
        plain_keys, plain_values = plain.compress(module, hidden, keys, values, None, {})
        self.assertEqual(plain_keys.shape, (1, 2, 128, 8))
        expected_set = (-keys.norm(dim=-1)).topk(128, dim=-1).indices.sort(-1).values

        for block_size in (2, 4, 8, 128, 256):
            sketch = BlockSketch(
                sketch=KnormSketch(compression_ratio=0.5), block_size=block_size
            )
            out_keys, out_values = sketch.compress(
                module, hidden, keys, values, None, {}
            )
            self.assertEqual(out_keys.shape, (1, 2, 128, 8))
            self.assertEqual(out_values.shape, (1, 2, 128, 8))
            torch.testing.assert_close(out_keys.sum(), plain_keys.sum())
            torch.testing.assert_close(out_values.sum(), plain_values.sum())
            recovered = _recover_indices(keys, out_keys).sort(-1).values
            self.assertTrue(torch.equal(recovered, expected_set))


class TestKeyDiffOracle(unittest.TestCase):
    def test_bitwise_match_with_partial_final_block(self):
        # bsz=2, H_kv=2, k_len=13, head_dim=4, block_size=4, ratio=0.4 ->
        # n_kept = int(13 * 0.6) = 7; iterations i=7 (block [7,11)) and
        # i=11 (partial block [11,13)).
        torch.manual_seed(3)
        keys = torch.randn(2, 2, 13, 4)
        values = torch.randn(2, 2, 13, 4)
        hidden = torch.randn(2, 13, 8)
        module = _FakeAttnModule(num_kv_heads=2, head_dim=4)

        sketch = BlockSketch(sketch=_KeyDiffScorer(compression_ratio=0.4), block_size=4)
        out_keys, out_values = sketch.compress(module, hidden, keys, values, None, {})

        def keydiff_fn(sub_keys, sub_values):
            anchor = F.normalize(sub_keys, p=2, dim=-1).mean(dim=2, keepdim=True)
            return -F.cosine_similarity(sub_keys, anchor, dim=-1)

        ref_keys, ref_values, kept = _block_press_oracle(
            keydiff_fn, keys, values, n_kept=7, block_size=4
        )
        self.assertTrue(torch.equal(out_keys, ref_keys))
        self.assertTrue(torch.equal(out_values, ref_values))

        # Per-head selection: equal counts, divergent position sets.
        for b in range(2):
            self.assertEqual(len(kept[b][0]), 7)
            self.assertEqual(len(kept[b][1]), 7)
            self.assertNotEqual(set(kept[b][0]), set(kept[b][1]))


class TestGQAHiddenStatesGather(unittest.TestCase):
    def test_current_states_mix_heads_exactly_as_upstream(self):
        # H_q=4, H_kv=2, head_dim=4 -> hidden_dim=16, per-KV-head slice = 8.
        # k_len=6, ratio=2/3 -> n_kept=2, block_size=2 -> iterations i=2, i=4.
        # Rigged scores make head 0 keep {0,2} and head 1 keep {1,3} after
        # iteration 1, so iteration 2's candidate indices diverge per head.
        k_len, hidden_dim = 6, 16
        hidden = (
            1000.0 * torch.arange(k_len, dtype=torch.float32)[:, None]
            + torch.arange(hidden_dim, dtype=torch.float32)[None, :]
        ).unsqueeze(0)
        torch.manual_seed(5)
        keys = torch.randn(1, 2, k_len, 4)
        values = torch.randn(1, 2, k_len, 4)
        module = _FakeAttnModule(num_heads=4, num_kv_heads=2, head_dim=4)

        probe = _QueueScorer(compression_ratio=2.0 / 3.0)
        probe.queue = [
            torch.tensor([[[4.0, 1.0, 3.0, 0.5], [1.0, 4.0, 0.5, 3.0]]]),
            torch.tensor([[[9.0, 8.0, 1.0, 0.0], [9.0, 8.0, 1.0, 0.0]]]),
        ]
        sketch = BlockSketch(sketch=probe, block_size=2)
        out_keys, out_values = sketch.compress(module, hidden, keys, values, None, {})

        self.assertEqual(len(probe.captured), 2)

        # Iteration 1: per-head indices identical ([0,1,2,3]) -> rows unmixed.
        first_hidden = probe.captured[0][0]
        self.assertTrue(torch.equal(first_hidden, hidden[:, :4]))

        # Iteration 2: candidates head0=[0,2,4,5], head1=[1,3,4,5]
        # (kept_indices in descending-score order, then the block).
        cand0, cand1 = [0, 2, 4, 5], [1, 3, 4, 5]
        second_hidden = probe.captured[1][0]
        self.assertEqual(second_hidden.shape, (1, 4, hidden_dim))
        for j in range(4):
            self.assertTrue(
                torch.equal(second_hidden[0, j, 0:8], hidden[0, cand0[j], 0:8])
            )
            self.assertTrue(
                torch.equal(second_hidden[0, j, 8:16], hidden[0, cand1[j], 8:16])
            )
        # Row 0 is a cross-head "frankenstein" token (t=0 and t=1 halves): it
        # matches no original hidden row.
        self.assertFalse((second_hidden[0, 0] == hidden[0]).all(-1).any())

        # Gathered keys follow the per-head candidate indices exactly.
        second_keys = probe.captured[1][1]
        self.assertTrue(torch.equal(second_keys[0, 0], keys[0, 0, cand0]))
        self.assertTrue(torch.equal(second_keys[0, 1], keys[0, 1, cand1]))

        # Final selection: rectangular output, divergent per-head positions.
        self.assertEqual(out_keys.shape, (1, 2, 2, 4))
        self.assertTrue(torch.equal(out_keys[0, 0], keys[0, 0, [0, 2]]))
        self.assertTrue(torch.equal(out_keys[0, 1], keys[0, 1, [1, 3]]))
        self.assertTrue(torch.equal(out_values[0, 0], values[0, 0, [0, 2]]))
        self.assertTrue(torch.equal(out_values[0, 1], values[0, 1, [1, 3]]))

    def test_hidden_dim_not_divisible_by_kv_heads_raises(self):
        module = _FakeAttnModule(num_kv_heads=2, head_dim=2)
        keys = torch.randn(1, 2, 8, 2)
        values = torch.randn(1, 2, 8, 2)
        hidden = torch.randn(1, 8, 15)
        sketch = BlockSketch(sketch=KnormSketch(compression_ratio=0.5), block_size=2)
        with self.assertRaises(RuntimeError):
            sketch.compress(module, hidden, keys, values, None, {})


class TestAttentionsGuard(unittest.TestCase):
    def test_non_none_attentions_raise(self):
        module = _FakeAttnModule(num_kv_heads=2, head_dim=8)
        keys = torch.randn(1, 2, 8, 8)
        values = torch.randn(1, 2, 8, 8)
        hidden = torch.randn(1, 8, 16)
        sketch = BlockSketch(sketch=KnormSketch(compression_ratio=0.5), block_size=2)
        with self.assertRaisesRegex(
            AssertionError, "BlockPress does not support attentions."
        ):
            sketch.compress(module, hidden, keys, values, torch.ones(1, 4, 8, 8), {})


class TestDegenerateBlockSize(unittest.TestCase):
    def test_block_size_ge_k_len_equals_plain_scorer_bitwise(self):
        torch.manual_seed(1)
        keys = torch.randn(1, 2, 64, 8)
        values = torch.randn(1, 2, 64, 8)
        hidden = torch.randn(1, 64, 16)
        module = _FakeAttnModule(num_kv_heads=2, head_dim=8)

        plain_keys, plain_values = KnormSketch(compression_ratio=0.5).compress(
            module, hidden, keys, values, None, {}
        )
        for block_size in (10_000, 64):
            sketch = BlockSketch(
                sketch=KnormSketch(compression_ratio=0.5), block_size=block_size
            )
            out_keys, out_values = sketch.compress(
                module, hidden, keys, values, None, {}
            )
            self.assertTrue(torch.equal(out_keys, plain_keys))
            self.assertTrue(torch.equal(out_values, plain_values))


class TestNKeptFloorEdge(unittest.TestCase):
    def test_n_kept_one_streams_the_single_winner(self):
        # k_len=10, ratio=0.85 -> n_kept = int(1.5...) = 1; block_size=3 ->
        # iterations over blocks [1,4), [4,7), [7,10). keys[0,0,p]=[10-p, 0]
        # (descending norms), knorm score=-norm: kept walks 0 -> 3 -> 6 -> 9.
        k_len = 10
        keys = torch.zeros(1, 1, k_len, 2)
        keys[0, 0, :, 0] = torch.arange(k_len, 0, -1, dtype=torch.float32)
        values = torch.randn(1, 1, k_len, 2)
        hidden = torch.zeros(1, k_len, 4)
        module = _FakeAttnModule(num_kv_heads=1, head_dim=2)

        sketch = BlockSketch(sketch=KnormSketch(compression_ratio=0.85), block_size=3)
        out_keys, out_values = sketch.compress(module, hidden, keys, values, None, {})
        self.assertEqual(out_keys.shape, (1, 1, 1, 2))
        self.assertTrue(torch.equal(out_keys, keys[:, :, [9]]))
        self.assertTrue(torch.equal(out_values, values[:, :, [9]]))


class TestForwardHookIntegration(unittest.TestCase):
    def _hook_kwargs(self, hidden_states, cache, cache_position):
        return {
            "hidden_states": hidden_states,
            "past_key_values": cache,
            "cache_position": cache_position,
        }

    def test_prefill_step_replaces_cache_with_compressed_kv(self):
        torch.manual_seed(2)
        B, H_kv, S, D = 1, 2, 32, 8
        keys = torch.randn(B, H_kv, S, D)
        values = torch.randn(B, H_kv, S, D)
        hidden = torch.randn(B, S, 16)
        cache = DynamicCache()
        cache.update(keys.clone(), values.clone(), 0)

        module = _FakeAttnModule(num_heads=4, num_kv_heads=H_kv, head_dim=D, layer_idx=0)
        sketch = BlockSketch(sketch=KnormSketch(compression_ratio=0.5), block_size=8)
        output = (hidden, None)
        result = sketch.forward_hook(
            module, [], self._hook_kwargs(hidden, cache, torch.arange(S)), output
        )
        self.assertIs(result, output)

        kwargs = self._hook_kwargs(hidden, cache, torch.arange(S))
        expected_keys, expected_values = sketch.compress(
            module, hidden, keys, values, None, kwargs
        )
        self.assertEqual(cache.layers[0].keys.shape, (B, H_kv, 16, D))
        self.assertTrue(torch.equal(cache.layers[0].keys, expected_keys))
        self.assertTrue(torch.equal(cache.layers[0].values, expected_values))

    def test_decode_step_leaves_cache_untouched(self):
        torch.manual_seed(4)
        B, H_kv, S, D = 1, 2, 32, 8
        cache = DynamicCache()
        cache.update(torch.randn(B, H_kv, S, D), torch.randn(B, H_kv, S, D), 0)
        cache.update(torch.randn(B, H_kv, 1, D), torch.randn(B, H_kv, 1, D), 0)
        keys_before = cache.layers[0].keys.clone()
        values_before = cache.layers[0].values.clone()

        module = _FakeAttnModule(num_heads=4, num_kv_heads=H_kv, head_dim=D, layer_idx=0)
        sketch = BlockSketch(sketch=KnormSketch(compression_ratio=0.5), block_size=8)
        hidden = torch.randn(B, 1, 16)
        output = (hidden, None)
        # cache_position[-1] = S > q_len = 1 -> _is_decoding_step gates a no-op.
        sketch.forward_hook(
            module, [], self._hook_kwargs(hidden, cache, torch.tensor([S])), output
        )
        self.assertTrue(torch.equal(cache.layers[0].keys, keys_before))
        self.assertTrue(torch.equal(cache.layers[0].values, values_before))


if __name__ == "__main__":
    unittest.main()
