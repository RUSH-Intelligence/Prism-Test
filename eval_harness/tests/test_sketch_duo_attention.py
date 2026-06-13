"""Tests for DuoAttentionSketch (port of kvpress DuoAttentionPress).

DuoAttention performs no physical pruning: compression is virtual via
``module.masked_key_indices`` consumed by the globally patched attention
functions (``eval_harness/kv_compression/attention_patch.py``). This sketch is the
first consumer of that machinery in Prism-Test, so the wrapper itself is
tested here too (kvpress math re-implemented inline as reference oracles,
plus a CPU equivalence oracle on a tiny config-built Llama — no hub access).
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from eval_harness.kv_compression.attention_patch import attention_patch, search_hyperplane
from eval_harness.kv_compression.compressors.duo_attention_sketch import PATTERNS_DICT, DuoAttentionSketch


# ======================================================================
# Local fakes and helpers
# ======================================================================


class _FakeAttnModule(nn.Module):
    def __init__(self, layer_idx=0, attn_implementation="sdpa", head_dim=4):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = head_dim
        self.config = SimpleNamespace(_attn_implementation=attn_implementation)


def _fake_model(name="fake/model", layers=None):
    model = SimpleNamespace(
        config=SimpleNamespace(name_or_path=name),
        device=torch.device("cpu"),
    )
    if layers is not None:
        model.model = SimpleNamespace(layers=layers, rotary_emb=object())
    return model


def _make_sketch(streaming_mask, sink, recent, ratio=0.0):
    sketch = DuoAttentionSketch(head_compression_ratio=ratio)
    sketch.streaming_mask = torch.as_tensor(streaming_mask, dtype=torch.bool)
    sketch.sink_size = sink
    sketch.recent_size = recent
    return sketch


def _repeat_kv_ref(hidden, n_rep):
    b, h_kv, s, d = hidden.shape
    if n_rep == 1:
        return hidden
    return hidden[:, :, None, :, :].expand(b, h_kv, n_rep, s, d).reshape(b, h_kv * n_rep, s, d)


def _kv(B, H_kv, S, D, seed=0):
    torch.manual_seed(seed)
    return torch.randn(B, H_kv, S, D), torch.randn(B, H_kv, S, D)


# ======================================================================
# compress: exact mask construction, branches, guards (value-pinned)
# ======================================================================


class TestCompress(unittest.TestCase):
    def test_exact_mask_construction_value_pinned(self):
        sketch = _make_sketch([[True, False]], sink=2, recent=3, ratio=0.5)
        module = _FakeAttnModule(layer_idx=0)
        keys, values = _kv(1, 2, 10, 4)

        out_k, out_v = sketch.compress(module, torch.randn(1, 10, 8), keys, values, None, {})

        # Virtual eviction: returned tensors are the SAME objects, no copy.
        self.assertIs(out_k, keys)
        self.assertIs(out_v, values)

        expected = (
            torch.zeros(5, dtype=torch.long),
            torch.zeros(5, dtype=torch.long),
            torch.arange(2, 7),
        )
        self.assertEqual(len(module.masked_key_indices), 3)
        for got, exp in zip(module.masked_key_indices, expected):
            self.assertTrue(torch.equal(got, exp))

        # 0.5 streaming fraction * (1 - (2+3)/10) == 0.25 exactly.
        self.assertEqual(sketch.compression_ratio_, 0.25)
        self.assertEqual(sketch.compression_ratio, 0.25)

    def test_zero_ratio_short_sequence_indices_never_set(self):
        sketch = _make_sketch([[False, False]], sink=2, recent=3, ratio=0.0)
        module = _FakeAttnModule()
        keys, values = _kv(1, 2, 4, 4)
        out_k, out_v = sketch.compress(module, torch.randn(1, 4, 8), keys, values, None, {})
        self.assertFalse(hasattr(module, "masked_key_indices"))
        self.assertIs(out_k, keys)
        self.assertIs(out_v, values)
        self.assertEqual(sketch.compression_ratio_, 0.0)

    def test_zero_ratio_long_sequence_empty_indices(self):
        sketch = _make_sketch([[False, False]], sink=2, recent=3, ratio=0.0)
        module = _FakeAttnModule()
        keys, values = _kv(1, 2, 10, 4)
        sketch.compress(module, torch.randn(1, 10, 8), keys, values, None, {})
        self.assertEqual(len(module.masked_key_indices), 3)
        for t in module.masked_key_indices:
            self.assertEqual(t.numel(), 0)
        self.assertEqual(sketch.compression_ratio_, 0.0)

    def test_negative_compression_ratio_upstream_artifact(self):
        # All-streaming, S < sink + recent: slice [2:-3] of a length-4 axis is
        # empty -> three empty index tensors; the reported ratio goes NEGATIVE
        # (kept verbatim from kvpress, not clamped).
        sketch = _make_sketch([[True, True]], sink=2, recent=3, ratio=1.0)
        module = _FakeAttnModule()
        keys, values = _kv(1, 2, 4, 4)
        sketch.compress(module, torch.randn(1, 4, 8), keys, values, None, {})
        self.assertEqual(len(module.masked_key_indices), 3)
        for t in module.masked_key_indices:
            self.assertEqual(t.numel(), 0)
        self.assertEqual(sketch.compression_ratio_, 1.0 * (1 - 5 / 4))
        self.assertEqual(sketch.compression_ratio_, -0.25)

    def test_eager_attention_raises(self):
        sketch = _make_sketch([[True]], sink=2, recent=3)
        module = _FakeAttnModule(attn_implementation="eager")
        keys, values = _kv(1, 1, 10, 4)
        with self.assertRaisesRegex(AssertionError, "eager mode not supported"):
            sketch.compress(module, torch.randn(1, 10, 8), keys, values, None, {})

    def test_uninitialized_streaming_mask_raises(self):
        sketch = DuoAttentionSketch(head_compression_ratio=0.5)
        module = _FakeAttnModule()
        keys, values = _kv(1, 1, 10, 4)
        with self.assertRaisesRegex(ValueError, "Streaming mask not initialized"):
            sketch.compress(module, torch.randn(1, 10, 8), keys, values, None, {})


class TestCompressReferenceOracle(unittest.TestCase):
    def test_matches_inline_kvpress_transcription(self):
        B, H_kv, S, D, sink, recent = 2, 4, 37, 4, 5, 7
        streaming_mask = torch.tensor(
            [[False, True, False, False], [True, False, True, True]], dtype=torch.bool
        )
        layer_idx = 1
        sketch = _make_sketch(streaming_mask, sink=sink, recent=recent, ratio=0.3)
        module = _FakeAttnModule(layer_idx=layer_idx)
        keys, values = _kv(B, H_kv, S, D, seed=11)

        sketch.compress(module, torch.randn(B, S, 8), keys, values, None, {})

        # Inline transcription of kvpress duo_attention_press.py:110-116.
        ref_mask = torch.zeros_like(keys[..., 0], dtype=torch.bool)
        ref_mask[:, streaming_mask[layer_idx], sink:-recent] = True
        ref_indices = torch.nonzero(ref_mask, as_tuple=True)
        ref_ratio = streaming_mask.float().mean().item()
        ref_ratio *= 1 - (sink + recent) / S

        self.assertEqual(len(module.masked_key_indices), 3)
        for got, exp in zip(module.masked_key_indices, ref_indices):
            self.assertTrue(torch.equal(got, exp))
        self.assertEqual(sketch.compression_ratio_, ref_ratio)


# ======================================================================
# post_init_from_model: global streaming-head selection (hand-computed)
# ======================================================================


class TestPostInitSelection(unittest.TestCase):
    SCORES = np.array([[0.9, 0.1], [0.5, 0.3]])

    def _sketch_with_loader(self, ratio, scores=None, sink=2, recent=3):
        sketch = DuoAttentionSketch(head_compression_ratio=ratio)
        sketch.load_attention_pattern = lambda model: (sink, recent, np.asarray(self.SCORES if scores is None else scores))
        return sketch

    def test_half_ratio_selects_two_globally_lowest(self):
        sketch = self._sketch_with_loader(0.5)
        sketch.post_init_from_model(_fake_model())
        self.assertEqual(sketch.sink_size, 2)
        self.assertEqual(sketch.recent_size, 3)
        self.assertEqual(sketch.streaming_mask.dtype, torch.bool)
        self.assertEqual(
            sketch.streaming_mask.tolist(), [[False, True], [False, True]]
        )

    def test_rounding_pins(self):
        # round(4 * 0.4) == 2 heads, round(4 * 0.2) == 1 head (the 0.1 score).
        sketch = self._sketch_with_loader(0.4)
        sketch.post_init_from_model(_fake_model())
        self.assertEqual(sketch.streaming_mask.tolist(), [[False, True], [False, True]])

        sketch = self._sketch_with_loader(0.2)
        sketch.post_init_from_model(_fake_model())
        self.assertEqual(sketch.streaming_mask.tolist(), [[False, True], [False, False]])

    def test_zero_ratio_all_retrieval(self):
        sketch = self._sketch_with_loader(0.0)
        sketch.post_init_from_model(_fake_model())
        self.assertFalse(sketch.streaming_mask.any().item())

    def test_global_argsort_gives_uneven_per_layer_budgets(self):
        sketch = self._sketch_with_loader(0.5, scores=[[0.1, 0.2], [0.8, 0.9]])
        sketch.post_init_from_model(_fake_model())
        self.assertEqual(sketch.streaming_mask.tolist(), [[True, True], [False, False]])

    def test_attention_pattern_injection_bypasses_loader(self):
        sketch = DuoAttentionSketch(
            head_compression_ratio=0.5,
            attention_pattern=(2, 3, [[0.9, 0.1], [0.5, 0.3]]),
        )
        sketch.load_attention_pattern = lambda model: self.fail("loader must not run with injected pattern")
        sketch.post_init_from_model(_fake_model())
        self.assertEqual(sketch.sink_size, 2)
        self.assertEqual(sketch.recent_size, 3)
        self.assertEqual(sketch.streaming_mask.tolist(), [[False, True], [False, True]])

    def test_recent_size_zero_asserts(self):
        sketch = DuoAttentionSketch(attention_pattern=(2, 0, [[0.5, 0.5]]))
        with self.assertRaisesRegex(AssertionError, "recent_size"):
            sketch.post_init_from_model(_fake_model())


class TestMemoization(unittest.TestCase):
    def _layer(self):
        return SimpleNamespace(self_attn=_FakeAttnModule())

    def test_loader_runs_once_across_context_entries(self):
        model = _fake_model(layers=[self._layer()])
        sketch = DuoAttentionSketch(head_compression_ratio=0.5)
        calls = []

        def loader(m):
            calls.append(m)
            return 2, 3, np.array([[0.9, 0.1]])

        sketch.load_attention_pattern = loader
        with sketch(model):
            pass
        mask_first = sketch.streaming_mask
        self.assertIsNotNone(mask_first)
        with sketch(model):
            pass
        self.assertEqual(len(calls), 1)
        self.assertIs(sketch.streaming_mask, mask_first)

    def test_reinitializes_for_a_different_model_name(self):
        sketch = DuoAttentionSketch(head_compression_ratio=0.5)
        calls = []

        def loader(m):
            calls.append(m)
            return 2, 3, np.array([[0.9, 0.1]])

        sketch.load_attention_pattern = loader
        with sketch(_fake_model(name="model/a", layers=[self._layer()])):
            pass
        with sketch(_fake_model(name="model/b", layers=[self._layer()])):
            pass
        self.assertEqual(len(calls), 2)


# ======================================================================
# attention_patch wrapper (first Prism consumer — tested directly)
# ======================================================================


class TestSearchHyperplaneOracle(unittest.TestCase):
    def test_fake_keys_zero_out_exp(self):
        # Transcribed from kvpress tests/test_attention_patch.py.
        torch.manual_seed(0)
        X = torch.rand(4, 64, 16)
        Y = search_hyperplane(X)
        self.assertEqual(Y.shape, (4, 16))
        self.assertEqual(torch.exp(torch.bmm(X, Y.unsqueeze(-1))).max().item(), 0.0)


class TestAttentionPatchWrapper(unittest.TestCase):
    def _wrap_probe(self):
        calls = {}

        def probe(module, query, key, value, attention_mask, dropout, **kwargs):
            calls["key"] = key
            return "probe-ret"

        return attention_patch(probe), calls

    def test_gqa_decode_masks_exactly_the_indexed_positions(self):
        wrapped, calls = self._wrap_probe()
        module = SimpleNamespace()
        module.masked_key_indices = (
            torch.zeros(3, dtype=torch.long),
            torch.zeros(3, dtype=torch.long),
            torch.tensor([2, 3, 4]),
        )
        torch.manual_seed(1)
        query = torch.rand(1, 4, 1, 8)  # all-positive queries: robust exp underflow
        key = torch.randn(1, 2, 8, 8)
        value = torch.randn(1, 2, 8, 8)
        key_orig = key.clone()

        ret = wrapped(module, query, key, value, None, 0.0)
        self.assertEqual(ret, "probe-ret")
        self.assertIs(calls["key"], key)

        weights = torch.softmax(
            torch.matmul(query, _repeat_kv_ref(key, 2).transpose(2, 3)) / math.sqrt(8), dim=-1
        )
        for q_head in (0, 1):  # query heads of KV head 0 (the masked one)
            self.assertTrue(torch.all(weights[0, q_head, 0, 2:5] == 0.0))
            self.assertTrue(torch.all(weights[0, q_head, 0, :2] > 0.0))
            self.assertTrue(torch.all(weights[0, q_head, 0, 5:] > 0.0))
        for q_head in (2, 3):  # query heads of KV head 1 (untouched)
            self.assertTrue(torch.all(weights[0, q_head, 0] > 0.0))

        self.assertTrue(torch.equal(key[:, 1], key_orig[:, 1]))
        self.assertTrue(torch.equal(key[0, 0, :2], key_orig[0, 0, :2]))
        self.assertTrue(torch.equal(key[0, 0, 5:], key_orig[0, 0, 5:]))
        self.assertFalse(torch.equal(key[0, 0, 2:5], key_orig[0, 0, 2:5]))

    def test_prefill_resets_masked_key_indices(self):
        wrapped, calls = self._wrap_probe()
        module = SimpleNamespace()
        module.masked_key_indices = (
            torch.zeros(1, dtype=torch.long),
            torch.zeros(1, dtype=torch.long),
            torch.tensor([3]),
        )
        torch.manual_seed(2)
        query = torch.rand(1, 4, 8, 8)  # q_len == k_len -> prefill
        key = torch.randn(1, 2, 8, 8)
        key_orig = key.clone()
        wrapped(module, query, key, torch.randn(1, 2, 8, 8), None, 0.0)
        self.assertIsNone(module.masked_key_indices)
        self.assertTrue(torch.equal(calls["key"], key_orig))

    def test_empty_indices_decode_is_a_noop(self):
        wrapped, calls = self._wrap_probe()
        module = SimpleNamespace()
        empty = torch.zeros(0, dtype=torch.long)
        module.masked_key_indices = (empty, empty, empty)
        torch.manual_seed(3)
        query = torch.rand(1, 4, 1, 8)
        key = torch.randn(1, 2, 8, 8)
        key_orig = key.clone()
        wrapped(module, query, key, torch.randn(1, 2, 8, 8), None, 0.0)
        self.assertTrue(torch.equal(key, key_orig))

    def test_cu_seq_lens_k_fixup(self):
        wrapped, _ = self._wrap_probe()
        module = SimpleNamespace()
        cu = torch.tensor([0, 5])
        wrapped(
            module,
            torch.rand(1, 4, 8, 8),
            torch.randn(1, 2, 8, 8),
            torch.randn(1, 2, 8, 8),
            None,
            0.0,
            cu_seq_lens_k=cu,
        )
        self.assertEqual(cu[-1].item(), 8)


# ======================================================================
# forward_hook level: virtual eviction leaves the cache full-length
# ======================================================================


class TestForwardHookVirtualEviction(unittest.TestCase):
    def test_prefill_sets_indices_and_cache_stays_full_length(self):
        S = 10
        sketch = _make_sketch([[True, False]], sink=2, recent=3, ratio=0.5)
        module = _FakeAttnModule(layer_idx=0)
        keys, values = _kv(1, 2, S, 4)
        cache = SimpleNamespace(layers=[SimpleNamespace(keys=keys, values=values)])
        kwargs = {
            "hidden_states": torch.randn(1, S, 8),
            "past_key_values": cache,
            "cache_position": torch.arange(S),
        }
        output = (torch.zeros(1, S, 8), None)

        result = sketch.forward_hook(module, [], kwargs, output)
        self.assertIs(result, output)

        expected_seq = torch.arange(2, 7)
        self.assertTrue(torch.equal(module.masked_key_indices[2], expected_seq))
        self.assertTrue(torch.equal(module.masked_key_indices[0], torch.zeros(5, dtype=torch.long)))
        self.assertTrue(torch.equal(module.masked_key_indices[1], torch.zeros(5, dtype=torch.long)))

        # The cache is NOT pruned: same objects, full length (masking is virtual).
        self.assertIs(cache.layers[0].keys, keys)
        self.assertIs(cache.layers[0].values, values)
        self.assertEqual(cache.layers[0].keys.shape[2], S)

    def test_decode_step_is_a_noop(self):
        S = 10
        sketch = _make_sketch([[True, False]], sink=2, recent=3, ratio=0.5)
        sketch.compress = lambda *a, **k: self.fail("compress must not run on decode steps")
        module = _FakeAttnModule(layer_idx=0)
        keys, values = _kv(1, 2, S, 4)
        cache = SimpleNamespace(layers=[SimpleNamespace(keys=keys, values=values)])
        kwargs = {
            "hidden_states": torch.randn(1, 1, 8),
            "past_key_values": cache,
            "cache_position": torch.tensor([S + 3]),
        }
        output = (torch.zeros(1, 1, 8), None)
        result = sketch.forward_hook(module, [], kwargs, output)
        self.assertIs(result, output)
        self.assertFalse(hasattr(module, "masked_key_indices"))


# ======================================================================
# Equivalence oracle: all-heads-streaming == physical sink+recent slicing
# ======================================================================


def _build_tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=256,
        max_position_embeddings=4096,
        rope_theta=10000.0,
        attn_implementation="sdpa",
    )
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg).eval()


class TestEquivalenceOracle(unittest.TestCase):
    def test_all_streaming_equals_physical_truncation(self):
        from transformers import DynamicCache

        model = _build_tiny_llama()
        S, sink, recent = 32, 4, 4
        torch.manual_seed(1)
        ctx = torch.randint(0, 256, (1, S))
        next_tok = torch.randint(0, 256, (1, 1))

        sketch = DuoAttentionSketch(
            head_compression_ratio=1.0,
            attention_pattern=(sink, recent, np.full((2, 2), 0.5)),
        )

        # (A) virtual eviction: sketch prefill + masked decode step.
        cache_a = DynamicCache()
        with torch.no_grad():
            with sketch(model):
                model.model(input_ids=ctx, past_key_values=cache_a, use_cache=True)
            for layer in model.model.layers:
                self.assertIsNotNone(layer.self_attn.masked_key_indices)
                self.assertEqual(
                    layer.self_attn.masked_key_indices[2].numel(), 2 * (S - sink - recent)
                )
                self.assertEqual(cache_a.layers[layer.self_attn.layer_idx].keys.shape[2], S)
            logits_a = model(next_tok, past_key_values=cache_a, use_cache=True).logits
        self.assertEqual(sketch.compression_ratio_, 1.0 * (1 - (sink + recent) / S))

        # (B) physical eviction: uncompressed prefill, slice to sink+recent,
        # decode with the question position continuing at the ORIGINAL position.
        cache_b = DynamicCache()
        with torch.no_grad():
            model.model(input_ids=ctx, past_key_values=cache_b, use_cache=True)
            for layer in model.model.layers:
                self.assertIsNone(layer.self_attn.masked_key_indices)
            for layer_cache in cache_b.layers:
                layer_cache.keys = torch.cat(
                    [layer_cache.keys[:, :, :sink], layer_cache.keys[:, :, S - recent :]], dim=2
                )
                layer_cache.values = torch.cat(
                    [layer_cache.values[:, :, :sink], layer_cache.values[:, :, S - recent :]], dim=2
                )
            logits_b = model(
                next_tok,
                past_key_values=cache_b,
                position_ids=torch.tensor([[S]]),
                use_cache=True,
            ).logits

        torch.testing.assert_close(logits_a, logits_b, atol=1e-4, rtol=1e-4)

        # (C) discriminating control: the uncompressed decode must differ.
        cache_c = DynamicCache()
        with torch.no_grad():
            model.model(input_ids=ctx, past_key_values=cache_c, use_cache=True)
            logits_c = model(next_tok, past_key_values=cache_c, use_cache=True).logits
        self.assertFalse(torch.allclose(logits_a, logits_c, atol=1e-4, rtol=1e-4))


# ======================================================================
# Pattern loading: local dir, alias pin, optional network smoke test
# ======================================================================


class TestPatternLoading(unittest.TestCase):
    def test_local_pattern_dir_loads_and_clips(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "config.json").write_text(json.dumps({"sink_size": 2, "recent_size": 3}))
            Path(d, "full_attention_heads.tsv").write_text("1.7\t0.1\n-0.5\t0.9\n")

            sink, recent, scores = DuoAttentionSketch(pattern_dir=d).load_attention_pattern(
                _fake_model(name="unlisted/model")
            )
            self.assertEqual((sink, recent), (2, 3))
            np.testing.assert_array_equal(scores, np.array([[1.0, 0.1], [0.0, 0.9]]))

            sketch = DuoAttentionSketch(head_compression_ratio=0.5, pattern_dir=d)
            sketch.post_init_from_model(_fake_model(name="unlisted/model"))
            self.assertEqual(sketch.sink_size, 2)
            self.assertEqual(sketch.recent_size, 3)
            # Two lowest clipped scores: 0.0 at (1,0) and 0.1 at (0,1).
            self.assertEqual(sketch.streaming_mask.tolist(), [[False, True], [True, False]])

    def test_llama31_rename_alias_pinned(self):
        path = "Meta-Llama-3.1-8B-Instruct/lr=0.02-reg=0.05-ctx=1000_128000-multi_passkey10"
        self.assertEqual(PATTERNS_DICT["meta-llama/Llama-3.1-8B-Instruct"], path)
        self.assertEqual(PATTERNS_DICT["meta-llama/Meta-Llama-3.1-8B-Instruct"], path)

    def test_unknown_checkpoint_asserts(self):
        with self.assertRaisesRegex(AssertionError, "not in"):
            DuoAttentionSketch().load_attention_pattern(_fake_model(name="meta-llama/Meta-Llama-3-8B"))

    @unittest.skipIf(os.environ.get("HF_HUB_OFFLINE") == "1", "offline test environment")
    def test_aliased_llama31_pattern_download_smoke(self):
        try:
            sink, recent, scores = DuoAttentionSketch().load_attention_pattern(
                _fake_model(name="meta-llama/Llama-3.1-8B-Instruct")
            )
        except Exception as exc:  # no network on compute nodes
            self.skipTest(f"network unavailable: {exc}")
        self.assertIsInstance(sink, int)
        self.assertIsInstance(recent, int)
        self.assertEqual(scores.shape, (32, 8))
        self.assertGreaterEqual(scores.min(), 0.0)
        self.assertLessEqual(scores.max(), 1.0)


# ======================================================================
# Registry / ResearchAdapter wiring
# ======================================================================


class TestRegistryWiring(unittest.TestCase):
    def test_registry_resolution(self):
        from eval_harness.kv_compression.registry import get_kv_compressor, get_kv_compressor_class

        self.assertIs(get_kv_compressor_class("duo_attention"), DuoAttentionSketch)
        sketch = get_kv_compressor("duo_attention", head_compression_ratio=0.5)
        self.assertIsInstance(sketch, DuoAttentionSketch)
        self.assertAlmostEqual(sketch.head_compression_ratio, 0.5)

    def test_compression_ratio_is_not_a_dataclass_field(self):
        field_names = {f.name for f in dataclasses.fields(DuoAttentionSketch)}
        self.assertNotIn("compression_ratio", field_names)
        self.assertIn("compression_ratio_", field_names)

    def test_compression_ratio_property_guards(self):
        sketch = DuoAttentionSketch(head_compression_ratio=0.5)
        with self.assertRaisesRegex(AssertionError, "Forward pass must be run"):
            _ = sketch.compression_ratio
        with self.assertRaisesRegex(AttributeError, "compression ratio cannot be set"):
            sketch.compression_ratio = 0.3

    def test_build_sketch_ignores_adapter_compression_ratio(self):
        from eval_harness.research_adapter import ResearchConfig, ResearchAdapter

        cfg = ResearchConfig(
            kv_compressor="duo_attention",
            compression_ratio=0.4,
            kv_compressor_kwargs={"head_compression_ratio": 0.5},
        )
        adapter = object.__new__(ResearchAdapter)
        adapter._cache_cfg = cfg
        sketch = adapter._build_kv_compressor(cfg)
        self.assertIsInstance(sketch, DuoAttentionSketch)
        self.assertAlmostEqual(sketch.head_compression_ratio, 0.5)
        with self.assertRaises(AssertionError):
            _ = sketch.compression_ratio


if __name__ == "__main__":
    unittest.main()
