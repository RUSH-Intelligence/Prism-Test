"""Tests for CURSketch (port of kvpress 0.5.1 CURPress, CurDKV arXiv:2509.15038)."""

from __future__ import annotations

import math
import sys
import unittest

import torch
import torch.nn.functional as F
from torch import nn

from eval_harness.kv_compression.compressors.cur_sketch import CURSketch
from eval_harness.kv_compression.registry import available_kv_compressors, get_kv_compressor, get_kv_compressor_class

_KVPRESS_ROOT = "/scratch/sj157/kvpress"
try:
    sys.path.append(_KVPRESS_ROOT)
    from kvpress.presses.cur_press import CURPress as _KvpressCURPress
except Exception:
    _KvpressCURPress = None
finally:
    if _KVPRESS_ROOT in sys.path:
        sys.path.remove(_KVPRESS_ROOT)


class _FakeAttnModule(nn.Module):
    def __init__(self, head_dim: int):
        super().__init__()
        self.head_dim = head_dim
        self.layer_idx = 0


def _cur_reference(
    keys: torch.Tensor,
    values: torch.Tensor,
    num_sinks: int = 4,
    leverage_type: str = "kv_product",
    use_random_leverage: bool = False,
    use_local_approximation: bool = True,
    local_window_size: int = 16,
) -> torch.Tensor:
    """Verbatim transcription of kvpress 0.5.1 cur_press.py:34-67 (score body)."""
    if use_random_leverage:
        r = 20
        G = torch.randn(keys.shape[-1], r, device=keys.device) / math.sqrt(r)
        keys = keys @ G
        values = values @ G

    k2 = (keys**2).sum(dim=-1)
    v2 = (values**2).sum(dim=-1)

    if use_local_approximation:
        b, h, n = k2.shape
        w = local_window_size
        k2 = F.pad(k2, (0, (w - n % w) % w)).reshape(b, h, -1, w)
        k2 = (k2 / k2.sum(dim=-1, keepdim=True)).reshape(b, h, -1)[:, :, :n]
        v2 = F.pad(v2, (0, (w - n % w) % w)).reshape(b, h, -1, w)
        v2 = (v2 / v2.sum(dim=-1, keepdim=True)).reshape(b, h, -1)[:, :, :n]

    if leverage_type == "key":
        scores = k2
    elif leverage_type == "value":
        scores = v2
    elif leverage_type == "kv_avg":
        scores = (k2 + v2) / 2
    elif leverage_type == "kv_product":
        scores = k2 * v2
    else:
        raise ValueError("Unknown leverage type")

    scores = scores / scores.sum(dim=-1, keepdim=True)
    scores[:, :, :num_sinks] = 1.0
    return scores


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_random_rope(keys: torch.Tensor, seed: int = 11) -> torch.Tensor:
    s, d = keys.shape[-2], keys.shape[-1]
    g = torch.Generator().manual_seed(seed)
    angles = torch.rand(s, d // 2, generator=g) * 2 * math.pi
    cos = torch.cat([angles.cos(), angles.cos()], dim=-1)
    sin = torch.cat([angles.sin(), angles.sin()], dim=-1)
    return keys * cos + _rotate_half(keys) * sin


def _random_kv(b, h, s, d, seed=0):
    g = torch.Generator().manual_seed(seed)
    keys = torch.randn(b, h, s, d, generator=g)
    values = torch.randn(b, h, s, d, generator=g)
    return keys, values


class TestCURRegistry(unittest.TestCase):
    def test_registered_under_cur(self):
        self.assertIn("cur", available_kv_compressors())
        self.assertIs(get_kv_compressor_class("cur"), CURSketch)

    def test_get_kv_compressor_instantiates_with_kwargs(self):
        sketch = get_kv_compressor(
            "cur",
            compression_ratio=0.3,
            num_sinks=2,
            leverage_type="key",
            use_local_approximation=False,
            local_window_size=8,
        )
        self.assertIsInstance(sketch, CURSketch)
        self.assertAlmostEqual(sketch.compression_ratio, 0.3)
        self.assertEqual(sketch.num_sinks, 2)
        self.assertEqual(sketch.leverage_type, "key")
        self.assertFalse(sketch.use_local_approximation)
        self.assertEqual(sketch.local_window_size, 8)

    def test_defaults_match_kvpress(self):
        sketch = CURSketch()
        self.assertEqual(sketch.compression_ratio, 0.0)
        self.assertEqual(sketch.num_sinks, 4)
        self.assertEqual(sketch.leverage_type, "kv_product")
        self.assertFalse(sketch.use_random_leverage)
        self.assertTrue(sketch.use_local_approximation)
        self.assertEqual(sketch.local_window_size, 16)


class _RaisingScoreCUR(CURSketch):
    def score(self, module, hidden_states, keys, values, attentions, kwargs):
        raise AssertionError("score() must not be called when compression_ratio == 0")


class TestCURZeroRatio(unittest.TestCase):
    def test_zero_ratio_is_noop_and_score_never_called(self):
        keys, values = _random_kv(1, 2, 10, 4, seed=1)
        sketch = _RaisingScoreCUR(compression_ratio=0.0)
        out_keys, out_values = sketch.compress(_FakeAttnModule(4), None, keys, values, None, {})
        self.assertIs(out_keys, keys)
        self.assertIs(out_values, values)


class TestCURHandComputed(unittest.TestCase):
    def setUp(self):
        self.keys = torch.tensor(
            [[[[1.0, 0.0], [2.0, 0.0], [0.0, 3.0], [1.0, 1.0], [0.0, 0.0], [5.0, 0.0]]]]
        )
        self.values = torch.ones(1, 1, 6, 2)

    def test_exact_scores_no_local_approx(self):
        sketch = CURSketch(
            compression_ratio=0.5,
            num_sinks=1,
            leverage_type="kv_product",
            use_local_approximation=False,
        )
        scores = sketch.score(None, None, self.keys, self.values, None, {})
        expected = torch.tensor([[[1.0, 8 / 82, 18 / 82, 4 / 82, 0.0, 50 / 82]]])
        torch.testing.assert_close(scores, expected)

    def test_compress_keeps_expected_index_set(self):
        sketch = CURSketch(
            compression_ratio=0.5,
            num_sinks=1,
            leverage_type="kv_product",
            use_local_approximation=False,
        )
        out_keys, out_values = sketch.compress(
            _FakeAttnModule(2), None, self.keys, self.values, None, {}
        )
        self.assertEqual(tuple(out_keys.shape), (1, 1, 3, 2))
        self.assertEqual(tuple(out_values.shape), (1, 1, 3, 2))
        kept_indices = []
        for row in out_keys[0, 0]:
            matches = (self.keys[0, 0] == row).all(dim=-1).nonzero().flatten().tolist()
            kept_indices.append(matches[0])
        self.assertEqual(sorted(kept_indices), [0, 2, 5])
        self.assertTrue(torch.equal(out_values, torch.ones(1, 1, 3, 2)))


class TestCURLocalApproximationOracle(unittest.TestCase):
    def _check(self, s, w, seed):
        g = torch.Generator().manual_seed(seed)
        keys = torch.rand(2, 3, s, 5, generator=g) + 0.1
        values = torch.rand(2, 3, s, 5, generator=g) + 0.1
        sketch = CURSketch(
            num_sinks=1, use_local_approximation=True, local_window_size=w
        )
        got = sketch.score(None, None, keys, values, None, {})
        expected = _cur_reference(
            keys, values, num_sinks=1, use_local_approximation=True, local_window_size=w
        )
        self.assertEqual(tuple(got.shape), (2, 3, s))
        torch.testing.assert_close(got, expected, rtol=1e-6, atol=1e-6)

    def test_partial_pad_window(self):
        self._check(s=10, w=4, seed=3)

    def test_exact_multiple_window(self):
        self._check(s=8, w=4, seed=4)

    def test_seq_shorter_than_window(self):
        self._check(s=3, w=4, seed=5)


class TestCURLeverageTypes(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(6)
        self.keys = torch.rand(2, 2, 10, 4, generator=g) + 0.1
        self.values = torch.rand(2, 2, 10, 4, generator=g) + 0.1

    def test_each_leverage_type_matches_reference(self):
        for leverage_type in ("key", "value", "kv_avg", "kv_product"):
            sketch = CURSketch(
                num_sinks=1, leverage_type=leverage_type, local_window_size=4
            )
            got = sketch.score(None, None, self.keys, self.values, None, {})
            expected = _cur_reference(
                self.keys,
                self.values,
                num_sinks=1,
                leverage_type=leverage_type,
                local_window_size=4,
            )
            torch.testing.assert_close(got, expected, rtol=1e-6, atol=1e-6, msg=leverage_type)

    def test_key_and_value_paths_differ(self):
        key_scores = CURSketch(num_sinks=0, leverage_type="key", local_window_size=4).score(
            None, None, self.keys, self.values, None, {}
        )
        value_scores = CURSketch(num_sinks=0, leverage_type="value", local_window_size=4).score(
            None, None, self.keys, self.values, None, {}
        )
        self.assertFalse(torch.allclose(key_scores, value_scores))

    def test_unknown_leverage_type_raises(self):
        sketch = CURSketch(leverage_type="bogus")
        with self.assertRaisesRegex(ValueError, "Unknown leverage type"):
            sketch.score(None, None, self.keys, self.values, None, {})


class TestCURSinks(unittest.TestCase):
    def test_sinks_always_kept(self):
        keys, values = _random_kv(2, 3, 64, 8, seed=7)
        sketch = CURSketch(compression_ratio=0.75, num_sinks=4)
        scores = sketch.score(None, None, keys, values, None, {})
        self.assertTrue(torch.all(scores[:, :, :4] == 1.0))
        n_kept = int(64 * (1 - 0.75))
        self.assertEqual(n_kept, 16)
        indices = scores.topk(n_kept, dim=-1).indices
        for b in range(2):
            for h in range(3):
                kept = set(indices[b, h].tolist())
                self.assertTrue({0, 1, 2, 3}.issubset(kept), f"sinks missing at b={b}, h={h}")

    def test_no_sinks_low_norm_first_token_dropped(self):
        keys, values = _random_kv(1, 2, 64, 8, seed=8)
        keys[:, :, 0, :] = 0.0
        sketch = CURSketch(
            compression_ratio=0.75, num_sinks=0, use_local_approximation=False
        )
        scores = sketch.score(None, None, keys, values, None, {})
        indices = scores.topk(16, dim=-1).indices
        for h in range(2):
            self.assertNotIn(0, indices[0, h].tolist())


class TestCURGQASelection(unittest.TestCase):
    def test_per_kv_head_selection_and_shapes(self):
        b, h_kv, s, d = 2, 4, 40, 16
        g = torch.Generator().manual_seed(9)
        keys = 0.01 * torch.randn(b, h_kv, s, d, generator=g)
        keys[:, 0, 20:40, :] += 1.0
        keys[:, 1, 10:30, :] += 1.0
        values = torch.ones(b, h_kv, s, d)
        sketch = CURSketch(
            compression_ratio=0.5,
            num_sinks=0,
            use_local_approximation=False,
            leverage_type="kv_product",
        )
        scores = sketch.score(None, None, keys, values, None, {})
        self.assertEqual(tuple(scores.shape), (b, h_kv, s))
        indices = scores.topk(20, dim=-1).indices
        for batch in range(b):
            head0 = set(indices[batch, 0].tolist())
            head1 = set(indices[batch, 1].tolist())
            self.assertEqual(head0, set(range(20, 40)))
            self.assertEqual(head1, set(range(10, 30)))
            self.assertNotEqual(head0, head1)
        out_keys, out_values = sketch.compress(
            _FakeAttnModule(d), None, keys, values, None, {}
        )
        self.assertEqual(tuple(out_keys.shape), (b, h_kv, 20, d))
        self.assertEqual(tuple(out_values.shape), (b, h_kv, 20, d))


class TestCURRopeInvariance(unittest.TestCase):
    def test_key_scores_invariant_to_rope_rotation(self):
        keys, values = _random_kv(1, 2, 12, 8, seed=10)
        rotated_keys = _apply_random_rope(keys)
        self.assertFalse(torch.allclose(keys, rotated_keys))
        sketch = CURSketch(
            num_sinks=0, leverage_type="key", use_local_approximation=False
        )
        plain = sketch.score(None, None, keys, values, None, {})
        rotated = sketch.score(None, None, rotated_keys, values, None, {})
        torch.testing.assert_close(plain, rotated, rtol=1e-5, atol=1e-5)


class TestCURRandomLeverage(unittest.TestCase):
    def setUp(self):
        self.keys, self.values = _random_kv(1, 2, 12, 6, seed=12)
        self.sketch = CURSketch(use_random_leverage=True)

    def test_seeded_determinism_and_finiteness(self):
        torch.manual_seed(0)
        first = self.sketch.score(None, None, self.keys, self.values, None, {})
        self.assertEqual(tuple(first.shape), (1, 2, 12))
        self.assertTrue(torch.isfinite(first).all())
        torch.manual_seed(0)
        second = self.sketch.score(None, None, self.keys, self.values, None, {})
        self.assertTrue(torch.equal(first, second))

    def test_unseeded_resampled_per_call(self):
        torch.manual_seed(0)
        first = self.sketch.score(None, None, self.keys, self.values, None, {})
        second = self.sketch.score(None, None, self.keys, self.values, None, {})
        self.assertFalse(torch.equal(first, second))

    def test_float32_bitwise_matches_kvpress_rng_stream(self):
        torch.manual_seed(7)
        expected = _cur_reference(self.keys, self.values, use_random_leverage=True)
        torch.manual_seed(7)
        got = self.sketch.score(None, None, self.keys, self.values, None, {})
        self.assertTrue(torch.equal(got, expected))

    def test_bf16_runs_without_error_dtype_deviation(self):
        keys = self.keys.bfloat16()
        values = self.values.bfloat16()
        torch.manual_seed(0)
        scores = self.sketch.score(None, None, keys, values, None, {})
        self.assertEqual(tuple(scores.shape), (1, 2, 12))
        self.assertEqual(scores.dtype, torch.bfloat16)


class TestCURRounding(unittest.TestCase):
    def test_n_kept_floor(self):
        keys, values = _random_kv(1, 2, 7, 4, seed=13)
        sketch = CURSketch(compression_ratio=0.5)
        out_keys, out_values = sketch.compress(
            _FakeAttnModule(4), None, keys, values, None, {}
        )
        self.assertEqual(out_keys.shape[2], 3)
        self.assertEqual(out_values.shape[2], 3)

    def test_single_token_half_ratio_empties_cache(self):
        keys, values = _random_kv(1, 2, 1, 4, seed=14)
        sketch = CURSketch(compression_ratio=0.5)
        out_keys, out_values = sketch.compress(
            _FakeAttnModule(4), None, keys, values, None, {}
        )
        self.assertEqual(out_keys.shape[2], 0)
        self.assertEqual(out_values.shape[2], 0)


class TestCURNaNCharacterization(unittest.TestCase):
    def test_zero_keys_with_local_approximation_nan(self):
        keys = torch.zeros(1, 1, 8, 4)
        values = torch.randn(1, 1, 8, 4, generator=torch.Generator().manual_seed(15))
        sketch = CURSketch(num_sinks=0, use_local_approximation=True)
        scores = sketch.score(None, None, keys, values, None, {})
        self.assertTrue(torch.isnan(scores).any())

    def test_zero_keys_and_values_global_normalize_nan(self):
        keys = torch.zeros(1, 1, 8, 4)
        values = torch.zeros(1, 1, 8, 4)
        sketch = CURSketch(num_sinks=0, use_local_approximation=False)
        scores = sketch.score(None, None, keys, values, None, {})
        self.assertTrue(torch.isnan(scores).all())


@unittest.skipUnless(_KvpressCURPress is not None, "kvpress not importable")
class TestCURKvpressLiveParity(unittest.TestCase):
    def setUp(self):
        self.keys, self.values = _random_kv(2, 3, 37, 8, seed=42)

    def _assert_parity(self, **cfg):
        press = _KvpressCURPress(compression_ratio=0.5, **cfg)
        sketch = CURSketch(compression_ratio=0.5, **cfg)
        expected = press.score(None, None, self.keys, self.values, None, {})
        got = sketch.score(None, None, self.keys, self.values, None, {})
        self.assertTrue(torch.equal(got, expected), f"parity failed for {cfg}")

    def test_defaults(self):
        self._assert_parity()

    def test_no_local_approximation(self):
        self._assert_parity(use_local_approximation=False)

    def test_leverage_types(self):
        for leverage_type in ("key", "value", "kv_avg", "kv_product"):
            self._assert_parity(leverage_type=leverage_type)

    def test_no_sinks(self):
        self._assert_parity(num_sinks=0)

    def test_random_leverage_same_seed(self):
        press = _KvpressCURPress(compression_ratio=0.5, use_random_leverage=True)
        sketch = CURSketch(compression_ratio=0.5, use_random_leverage=True)
        torch.manual_seed(123)
        expected = press.score(None, None, self.keys, self.values, None, {})
        torch.manual_seed(123)
        got = sketch.score(None, None, self.keys, self.values, None, {})
        self.assertTrue(torch.equal(got, expected))


if __name__ == "__main__":
    unittest.main()
