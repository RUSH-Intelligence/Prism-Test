"""Unit tests for DMSSketch (port of kvpress 0.5.1 DMSPress).

No model loading; fake attention modules and caches only. Eviction indices,
shift arithmetic and compression-ratio bookkeeping are pinned against
hand-computed values from the kvpress math (kvpress is not importable in
prism_env). The masked-key application path (attention_patch.py) is exercised
end-to-end with an in-test transcription of the search_hyperplane
post-condition, since DMS is the first sketch relying on it.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from eval_harness.research_adapter import CacheConfig, ResearchAdapter
from eval_harness.sketch.attention_patch import attention_patch
from eval_harness.sketch.sketches.base_sketch import BaseSketch
from eval_harness.sketch.sketches.dms_sketch import DMSSketch
from eval_harness.sketch.sketches.knorm_sketch import KnormSketch
from eval_harness.sketch.sketches.random_sketch import RandomSketch
from eval_harness.sketch.sketches.registry import (
    available_sketches,
    get_sketch,
    get_sketch_class,
)


class _FakeCacheLayer:
    def __init__(self, keys: torch.Tensor, values: torch.Tensor):
        self.keys = keys
        self.values = values


class _FakeCache:
    def __init__(self, layer_kvs: list[tuple[torch.Tensor, torch.Tensor]]):
        self.layers = [_FakeCacheLayer(k, v) for k, v in layer_kvs]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.layers[layer_idx].keys.shape[2]


def _fake_module(layer_idx: int = 0, head_dim: int = 4) -> SimpleNamespace:
    # The patched attention wrapper pre-creates masked_key_indices on every
    # full prefill forward; fakes replicate that contract.
    return SimpleNamespace(layer_idx=layer_idx, head_dim=head_dim, masked_key_indices=None)


def _keys_from_norms(norms: list[list[float]], head_dim: int = 4) -> torch.Tensor:
    """[1, H, S, head_dim] keys where ||k[0, h, s]|| == norms[h][s] exactly."""
    keys = torch.zeros(1, len(norms), len(norms[0]), head_dim)
    keys[..., 0] = torch.tensor(norms, dtype=torch.float32)
    return keys


def _single_layer_cache(keys: torch.Tensor) -> _FakeCache:
    return _FakeCache([(keys, torch.zeros_like(keys))])


def _append_token(cache: _FakeCache, layer_idx: int, norm: float, head_dim: int = 4) -> None:
    layer = cache.layers[layer_idx]
    new = torch.zeros(1, layer.keys.shape[1], 1, head_dim)
    new[..., 0] = norm
    layer.keys = torch.cat([layer.keys, new], dim=2)
    layer.values = torch.cat([layer.values, torch.zeros_like(new)], dim=2)


def _run_hook(sketch, module, cache, q_len, cache_position):
    kwargs = {
        "hidden_states": torch.zeros(1, q_len, 8),
        "past_key_values": cache,
    }
    if cache_position is not None:
        kwargs["cache_position"] = cache_position
    output = ("attn_out", None)
    result = sketch.forward_hook(module, (), kwargs, output)
    return result, output


def _assert_masked_equal(test, masked, batch, head, token):
    test.assertIsNotNone(masked)
    test.assertEqual(len(masked), 3)
    test.assertTrue(torch.equal(masked[0], torch.tensor(batch)))
    test.assertTrue(torch.equal(masked[1], torch.tensor(head)))
    test.assertTrue(torch.equal(masked[2], torch.tensor(token)))


class TestDMSConstruction(unittest.TestCase):
    def test_threshold_none_fails_fast(self):
        with self.assertRaises(AssertionError):
            DMSSketch(press=KnormSketch())

    def test_press_resolved_from_registry_name(self):
        sketch = DMSSketch(press="knorm", threshold=-2.0, sliding_window_size=64)
        self.assertIsInstance(sketch.press, KnormSketch)
        self.assertEqual(sketch.sliding_window_size, 64)

    def test_press_must_be_scorer(self):
        with self.assertRaises(AssertionError):
            DMSSketch(press=BaseSketch(), threshold=0.0)

    def test_decoding_true_warns_about_pipeline_gate(self):
        with self.assertLogs("eval_harness.sketch.sketches.dms_sketch", level="WARNING") as cm:
            DMSSketch(press=KnormSketch(), threshold=0.0, decoding=True)
        self.assertTrue(any("decode" in line.lower() for line in cm.output))

    def test_compression_ratio_asserts_before_forward(self):
        sketch = DMSSketch(press=KnormSketch(), threshold=0.0)
        with self.assertRaises(AssertionError):
            _ = sketch.compression_ratio

    def test_compression_ratio_setter_raises(self):
        sketch = DMSSketch(press=KnormSketch(), threshold=0.0)
        with self.assertRaises(AttributeError):
            sketch.compression_ratio = 0.5

    def test_defaults_mirror_kvpress(self):
        sketch = DMSSketch(press=KnormSketch(), threshold=0.0)
        self.assertEqual(sketch.sliding_window_size, 128)
        self.assertFalse(sketch.decoding)
        self.assertEqual(sketch.scores_buffer, {})
        self.assertEqual(sketch.compression_ratios, {})


class TestDMSPrefill(unittest.TestCase):
    def test_no_eviction_noop_when_window_covers_sequence(self):
        torch.manual_seed(0)
        keys = torch.randn(1, 2, 10, 4)
        keys_clone = keys.clone()
        cache = _single_layer_cache(keys)
        values_ref = cache.layers[0].values
        module = _fake_module()
        sketch = DMSSketch(press=KnormSketch(), threshold=-2.0, sliding_window_size=16)

        result, output = _run_hook(sketch, module, cache, q_len=10, cache_position=torch.arange(10))

        self.assertIs(result, output)
        self.assertIsNone(module.masked_key_indices)
        self.assertEqual(sketch.compression_ratios, {0: 0})
        self.assertEqual(tuple(sketch.scores_buffer[0].shape), (1, 2, 10))
        self.assertIs(cache.layers[0].keys, keys)
        self.assertIs(cache.layers[0].values, values_ref)
        self.assertTrue(torch.equal(cache.layers[0].keys, keys_clone))

    def test_value_pinned_prefill_eviction(self):
        keys = _keys_from_norms([[1, 3, 1, 3, 1, 1], [3, 1, 3, 1, 1, 1]])
        cache = _single_layer_cache(keys)
        module = _fake_module()
        sketch = DMSSketch(press=KnormSketch(), threshold=-2.0, sliding_window_size=2)

        _run_hook(sketch, module, cache, q_len=6, cache_position=torch.arange(6))

        # Positions 0..3 evaluated (n_to_evict = 6 - 2 = 4); knorm score -norm < -2
        # masks norm-3 keys: head 0 at t in {1, 3}, head 1 at t in {0, 2}; shift = 0.
        _assert_masked_equal(self, module.masked_key_indices, [0, 0, 0, 0], [0, 0, 1, 1], [1, 3, 0, 2])
        self.assertEqual(sketch.compression_ratios, {0: 4 / 12})
        expected_buffer = torch.full((1, 2, 2), -1.0)
        self.assertTrue(torch.equal(sketch.scores_buffer[0], expected_buffer))
        self.assertIs(cache.layers[0].keys, keys)

    def test_prefill_without_cache_position_uses_seq_length_fallback(self):
        keys = _keys_from_norms([[1, 3, 1, 3, 1, 1], [3, 1, 3, 1, 1, 1]])
        cache = _single_layer_cache(keys)
        module = _fake_module()
        sketch = DMSSketch(press=KnormSketch(), threshold=-2.0, sliding_window_size=2)

        _run_hook(sketch, module, cache, q_len=6, cache_position=None)

        _assert_masked_equal(self, module.masked_key_indices, [0, 0, 0, 0], [0, 0, 1, 1], [1, 3, 0, 2])
        self.assertEqual(sketch.compression_ratios, {0: 4 / 12})

    def test_layer_zero_prefill_resets_buffers(self):
        sketch = DMSSketch(press=KnormSketch(), threshold=-2.0, sliding_window_size=16)
        sketch.scores_buffer[5] = torch.zeros(1, 2, 3)
        sketch.compression_ratios[5] = 0.9
        cache = _single_layer_cache(torch.randn(1, 2, 4, 4))
        module = _fake_module(layer_idx=0)

        _run_hook(sketch, module, cache, q_len=4, cache_position=torch.arange(4))

        self.assertNotIn(5, sketch.scores_buffer)
        self.assertNotIn(5, sketch.compression_ratios)
        self.assertIn(0, sketch.scores_buffer)
        self.assertEqual(sketch.compression_ratios, {0: 0})

    def test_non_zero_layer_prefill_does_not_reset(self):
        sketch = DMSSketch(press=KnormSketch(), threshold=-2.0, sliding_window_size=16)
        junk = torch.zeros(1, 2, 3)
        sketch.scores_buffer[5] = junk
        sketch.compression_ratios[5] = 0.9
        kv = torch.randn(1, 2, 4, 4)
        cache = _FakeCache([(kv, kv.clone()), (kv.clone(), kv.clone())])
        module = _fake_module(layer_idx=1)

        _run_hook(sketch, module, cache, q_len=4, cache_position=torch.arange(4))

        self.assertIs(sketch.scores_buffer[5], junk)
        self.assertEqual(sketch.compression_ratios[5], 0.9)
        self.assertIn(1, sketch.scores_buffer)


class TestDMSDecode(unittest.TestCase):
    def _prefilled(self, decoding=True):
        keys = _keys_from_norms([[1, 1, 5, 7]])
        cache = _single_layer_cache(keys)
        module = _fake_module()
        sketch = DMSSketch(press=KnormSketch(), threshold=-2.0, sliding_window_size=2, decoding=decoding)
        _run_hook(sketch, module, cache, q_len=4, cache_position=torch.arange(4))
        # Positions 0, 1 (norm 1, score -1 > -2) already aged out unmasked;
        # buffer holds scores of positions 2, 3 = [-5, -7].
        self.assertIsNone(module.masked_key_indices)
        self.assertTrue(torch.equal(sketch.scores_buffer[0], torch.tensor([[[-5.0, -7.0]]])))
        return sketch, module, cache

    def test_decode_shift_arithmetic_value_pinned(self):
        sketch, module, cache = self._prefilled()

        _append_token(cache, 0, norm=1.0)
        _run_hook(sketch, module, cache, q_len=1, cache_position=torch.tensor([4]))
        # Buffer [-5, -7, -1] exceeds window 2 -> evict s2; shift = 5 - 1 - 2 = 2.
        _assert_masked_equal(self, module.masked_key_indices, [0], [0], [2])
        self.assertEqual(sketch.compression_ratios, {0: 1 / 5})
        self.assertTrue(torch.equal(sketch.scores_buffer[0], torch.tensor([[[-7.0, -1.0]]])))

        _append_token(cache, 0, norm=1.0)
        _run_hook(sketch, module, cache, q_len=1, cache_position=torch.tensor([5]))
        # Evict s3; shift = 6 - 1 - 2 = 3; merge preserves entry order [2, 3].
        _assert_masked_equal(self, module.masked_key_indices, [0, 0], [0, 0], [2, 3])
        self.assertEqual(sketch.compression_ratios, {0: 2 / 6})

    def test_decode_without_cache_position_uses_seq_length_fallback(self):
        sketch, module, cache = self._prefilled()
        _append_token(cache, 0, norm=1.0)
        _run_hook(sketch, module, cache, q_len=1, cache_position=None)
        _assert_masked_equal(self, module.masked_key_indices, [0], [0], [2])

    def test_decoding_false_gate_is_noop(self):
        sketch, module, cache = self._prefilled(decoding=False)
        ratios_before = dict(sketch.compression_ratios)

        _append_token(cache, 0, norm=100.0)
        result, output = _run_hook(sketch, module, cache, q_len=1, cache_position=torch.tensor([4]))

        self.assertIs(result, output)
        self.assertIsNone(module.masked_key_indices)
        self.assertEqual(sketch.scores_buffer[0].shape[-1], 2)
        self.assertEqual(sketch.compression_ratios, ratios_before)


class TestDMSEdgeCases(unittest.TestCase):
    def test_window_larger_than_sequence_never_masks(self):
        keys = _keys_from_norms([[9.0] * 10])
        cache = _single_layer_cache(keys)
        module = _fake_module()
        sketch = DMSSketch(press=KnormSketch(), threshold=-2.0, sliding_window_size=32, decoding=True)
        _run_hook(sketch, module, cache, q_len=10, cache_position=torch.arange(10))
        for step in range(2):
            _append_token(cache, 0, norm=9.0)
            _run_hook(sketch, module, cache, q_len=1, cache_position=torch.tensor([10 + step]))

        self.assertIsNone(module.masked_key_indices)
        self.assertEqual(sketch.compression_ratio, 0)

    def test_threshold_plus_inf_masks_every_out_of_window_position(self):
        keys = _keys_from_norms([[1.0] * 10, [1.0] * 10])
        cache = _single_layer_cache(keys)
        module = _fake_module()
        sketch = DMSSketch(press=KnormSketch(), threshold=float("inf"), sliding_window_size=2)
        _run_hook(sketch, module, cache, q_len=10, cache_position=torch.arange(10))

        _assert_masked_equal(
            self,
            module.masked_key_indices,
            [0] * 16,
            [0] * 8 + [1] * 8,
            list(range(8)) + list(range(8)),
        )
        self.assertEqual(sketch.compression_ratios, {0: (10 - 2) / 10})

    def test_threshold_minus_inf_masks_nothing(self):
        keys = _keys_from_norms([[1.0] * 10])
        cache = _single_layer_cache(keys)
        module = _fake_module()
        sketch = DMSSketch(press=KnormSketch(), threshold=float("-inf"), sliding_window_size=2)
        _run_hook(sketch, module, cache, q_len=10, cache_position=torch.arange(10))

        self.assertIsNone(module.masked_key_indices)
        self.assertEqual(sketch.compression_ratios, {0: 0})

    def test_zero_window_evaluates_all_positions_with_zero_shift(self):
        keys = _keys_from_norms([[1.0] * 6])
        cache = _single_layer_cache(keys)
        module = _fake_module()
        sketch = DMSSketch(press=KnormSketch(), threshold=float("inf"), sliding_window_size=0)
        _run_hook(sketch, module, cache, q_len=6, cache_position=torch.arange(6))

        _assert_masked_equal(self, module.masked_key_indices, [0] * 6, [0] * 6, [0, 1, 2, 3, 4, 5])
        self.assertEqual(sketch.scores_buffer[0].shape[-1], 0)


class TestDMSCompressionRatioBookkeeping(unittest.TestCase):
    def test_bookkeeping_matches_independent_recount(self):
        """Transcribes the kvpress oracle (tests/presses/test_head_compression.py:62-89):
        the masked fraction recomputed from module.masked_key_indices at the final
        cache length, averaged over layers, equals press.compression_ratio exactly."""
        torch.manual_seed(11)
        n_layers, heads, seq, head_dim, decode_steps = 2, 2, 16, 4, 3
        cache = _FakeCache(
            [(torch.randn(1, heads, seq, head_dim), torch.randn(1, heads, seq, head_dim)) for _ in range(n_layers)]
        )
        modules = [_fake_module(layer_idx=i, head_dim=head_dim) for i in range(n_layers)]
        sketch = DMSSketch(press=RandomSketch(), threshold=0.5, sliding_window_size=0, decoding=True)

        for module in modules:
            _run_hook(sketch, module, cache, q_len=seq, cache_position=torch.arange(seq))
        for step in range(decode_steps):
            for layer_idx in range(n_layers):
                _append_token(cache, layer_idx, norm=1.0, head_dim=head_dim)
            for module in modules:
                _run_hook(sketch, module, cache, q_len=1, cache_position=torch.tensor([seq + step]))

        final_len = seq + decode_steps
        recounted = []
        for module in modules:
            n_masked = 0 if module.masked_key_indices is None else len(module.masked_key_indices[0])
            recounted.append(n_masked / (1 * heads * final_len))
        self.assertGreater(sum(recounted), 0)
        self.assertEqual(sketch.compression_ratio, sum(recounted) / len(recounted))

    def test_random_press_threshold_statistical_sanity(self):
        """kvpress usage pattern (RandomPress, threshold=0.5, sliding_window_size=0):
        the content-adaptive masked fraction lands within 4 sigma of 0.5."""
        torch.manual_seed(123)
        seq = 4096
        cache = _single_layer_cache(torch.randn(1, 1, seq, 4))
        module = _fake_module()
        sketch = DMSSketch(press=RandomSketch(), threshold=0.5, sliding_window_size=0)
        _run_hook(sketch, module, cache, q_len=seq, cache_position=torch.arange(seq))

        fraction = len(module.masked_key_indices[0]) / seq
        self.assertLess(abs(fraction - 0.5), 4 * 0.0078125)
        self.assertEqual(sketch.compression_ratios[0], fraction)
        self.assertEqual(sketch.compression_ratio, fraction)


class TestAttentionPatchMasking(unittest.TestCase):
    """End-to-end check of the masked-key application path DMS relies on."""

    def test_gqa_fake_keys_replace_exactly_the_masked_rows(self):
        torch.manual_seed(0)
        query = torch.randn(1, 4, 3, 8)
        query[..., 0] = 5.0 + query[..., 0].abs()
        key = torch.randn(1, 2, 8, 8)
        value = torch.randn(1, 2, 8, 8)
        key_before = key.clone()
        module = SimpleNamespace(
            masked_key_indices=(torch.tensor([0, 0]), torch.tensor([0, 1]), torch.tensor([2, 5]))
        )
        calls = []

        def stub(mod, q, k, v, mask, dropout, **kw):
            calls.append((q, k, v))
            return "attn-result"

        wrapped = attention_patch(stub)
        result = wrapped(module, query, key, value, None, 0.0)

        self.assertEqual(result, "attn-result")
        self.assertIs(calls[0][1], key)
        changed = (key != key_before).any(dim=-1)
        expected = torch.zeros(1, 2, 8, dtype=torch.bool)
        expected[0, 0, 2] = True
        expected[0, 1, 5] = True
        self.assertTrue(torch.equal(changed, expected))

        # search_hyperplane post-condition: exp(<q, k_fake>) == 0 for every
        # grouped query of the masked KV head.
        grouped_q = query.view(1, 2, 2, 3, 8).reshape(2, 6, 8)
        for head, token in ((0, 2), (1, 5)):
            logits = grouped_q[head] @ key[0, head, token]
            self.assertLess(logits.max().item(), -50.0)
            self.assertTrue((torch.exp(logits) < 1e-20).all())

    def test_full_prefill_forward_resets_masked_indices(self):
        query = torch.randn(1, 4, 8, 8)
        key = torch.randn(1, 2, 8, 8)
        key_before = key.clone()
        module = SimpleNamespace(
            masked_key_indices=(torch.tensor([0]), torch.tensor([0]), torch.tensor([1]))
        )
        wrapped = attention_patch(lambda *args, **kw: "dense")
        result = wrapped(module, query, key, key.clone(), None, 0.0)

        self.assertEqual(result, "dense")
        self.assertIsNone(module.masked_key_indices)
        self.assertTrue(torch.equal(key, key_before))

    def test_cu_seq_lens_k_fixup(self):
        query = torch.randn(1, 4, 8, 8)
        key = torch.randn(1, 2, 8, 8)
        module = SimpleNamespace(masked_key_indices=None)
        wrapped = attention_patch(lambda *args, **kw: None)
        cu = torch.tensor([0, 999])
        wrapped(module, query, key, key.clone(), None, 0.0, cu_seq_lens_k=cu)
        self.assertEqual(cu[-1].item(), 8)


class _CtxAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_idx = 0
        self.head_dim = 4


class _CtxLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _CtxAttn()


class _CtxLanguageModel(nn.Module):
    def __init__(self, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([_CtxLayer() for _ in range(n_layers)])
        self.rotary_emb = nn.Identity()


class _CtxModel(nn.Module):
    def __init__(self, attn_implementation: str = "sdpa", n_layers: int = 2):
        super().__init__()
        self.model = _CtxLanguageModel(n_layers)
        self.config = SimpleNamespace(_attn_implementation=attn_implementation)


class TestDMSContextManager(unittest.TestCase):
    def _sketch(self, **kwargs):
        kwargs.setdefault("press", KnormSketch())
        kwargs.setdefault("threshold", 0.0)
        return DMSSketch(**kwargs)

    def test_hooks_registered_and_removed(self):
        model = _CtxModel()
        attns = [layer.self_attn for layer in model.model.layers]
        with self._sketch()(model):
            for attn in attns:
                self.assertEqual(len(attn._forward_hooks), 1)
                self.assertIs(attn.rotary_emb, model.model.rotary_emb)
        for attn in attns:
            self.assertEqual(len(attn._forward_hooks), 0)

    def test_eager_attention_refused(self):
        model = _CtxModel(attn_implementation="eager")
        with self.assertRaisesRegex(ValueError, "eager"):
            with self._sketch()(model):
                pass

    def test_replaced_attention_forward_refused(self):
        model = _CtxModel()
        model.model.layers[0].self_attn.forward = lambda *args, **kw: None
        with self.assertRaisesRegex(ValueError, "self_attn.forward"):
            with self._sketch()(model):
                pass

    def test_preexisting_prune_hook_refused(self):
        model = _CtxModel()
        model.model.layers[1].self_attn.register_forward_hook(lambda mod, inp, out: None)
        with self.assertRaisesRegex(ValueError, "forward hooks"):
            with self._sketch()(model):
                pass


class TestDMSRegistryAndBuild(unittest.TestCase):
    def test_registry_resolution(self):
        self.assertIn("dms", available_sketches())
        self.assertIs(get_sketch_class("dms"), DMSSketch)

    def test_get_sketch_with_kwargs(self):
        sketch = get_sketch("dms", press="knorm", threshold=-2.0, sliding_window_size=64)
        self.assertIsInstance(sketch, DMSSketch)
        self.assertIsInstance(sketch.press, KnormSketch)
        self.assertEqual(sketch.threshold, -2.0)

    def test_build_sketch_does_not_inject_adapter_compression_ratio(self):
        cfg = CacheConfig(
            sketch_name="dms",
            compression_ratio=0.4,
            sketch_kwargs={"press": "knorm", "threshold": -2.0, "sliding_window_size": 64},
        )
        adapter = object.__new__(ResearchAdapter)
        adapter._cache_cfg = cfg
        sketch = adapter._build_sketch(cfg)
        self.assertIsInstance(sketch, DMSSketch)
        self.assertIsInstance(sketch.press, KnormSketch)
        self.assertEqual(sketch.sliding_window_size, 64)
        with self.assertRaises(AssertionError):
            _ = sketch.compression_ratio


if __name__ == "__main__":
    unittest.main()
