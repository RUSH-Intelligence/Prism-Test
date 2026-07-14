"""Regression test for the ``compression_schedule`` coercion contract.

``KVCompressor.__post_init__`` (kv_compression/base.py) coerces a user-supplied
``schedule`` (a string or list from YAML) into a ``frozenset`` of
``CompressionSchedule`` members.  Subclasses that override ``__post_init__`` must
chain ``super().__post_init__()`` or the raw value is never coerced and the first
``fires_on_prefill``/``fires_on_decode`` access crashes with
``TypeError: unsupported operand type(s) for &: 'str' and 'set'``.

This guards every compressor that overrides ``__post_init__`` against that
regression by constructing each with ``schedule`` as both a ``str`` and a
``list``.  ``TestPostPrefillFiresBeforeQuestion`` additionally pins the
end-to-end POST_PREFILL ordering contract for the ridge/compactor fairness
pair: the hook compresses exactly once per layer, AFTER the full context
prefill and BEFORE any question token, to exactly ``int(T * (1 - ratio))``
tokens per head.  No model loading.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from eval_harness.kv_compression.base import CompressionSchedule
from eval_harness.kv_compression.compressors.adakv_sketch import AdaKVSketch
from eval_harness.kv_compression.compressors.block_sketch import BlockSketch
from eval_harness.kv_compression.compressors.chunk_sketch import ChunkSketch
from eval_harness.kv_compression.compressors.chunkkv_sketch import ChunkKVSketch
from eval_harness.kv_compression.compressors.compactor_sketch import CompactorSketch
from eval_harness.kv_compression.compressors.composed_sketch import ComposedSketch
from eval_harness.kv_compression.compressors.criticalkv_sketch import CriticalAdaKVSketch
from eval_harness.kv_compression.compressors.decoding_sketch import DecodingSketch
from eval_harness.kv_compression.compressors.dms_sketch import DMSSketch
from eval_harness.kv_compression.compressors.fastkvzip_sketch import FastKVzipSketch
from eval_harness.kv_compression.compressors.finch_sketch import FinchSketch
from eval_harness.kv_compression.compressors.key_rerotation_sketch import KeyRerotationSketch
from eval_harness.kv_compression.compressors.knorm_sketch import KnormSketch
from eval_harness.kv_compression.compressors.kvzip_sketch import KVzipSketch
from eval_harness.kv_compression.compressors.per_layer_compression_sketch import (
    PerLayerCompressionSketch,
)
from eval_harness.kv_compression.compressors.ridge_sketch import RidgeSketch
from eval_harness.kv_compression.compressors.simlayerkv_sketch import SimLayerKVSketch
from eval_harness.kv_compression.compressors.snapkv_sketch import SnapKVSketch
from eval_harness.kv_compression.compressors.think_sketch import ThinKSketch
from eval_harness.tests.test_sketch_ridge import _FakeAttnModule as _RidgeFakeAttnModule


class _CompactorFakeAttnModule(nn.Module):
    """Llama-shaped attention stub for CompactorSketch (defined locally rather
    than imported from test_sketch_compactor): exposes exactly what the
    sketch's pre-RoPE re-projections read — ``config.num_attention_heads``,
    ``head_dim``, ``q_proj``, ``k_proj``."""

    def __init__(self, hidden_dim=32, num_heads=4, num_kv_heads=2, head_dim=8, seed=0):
        super().__init__()
        self.config = SimpleNamespace(num_attention_heads=num_heads)
        self.num_key_value_heads = num_kv_heads
        self.head_dim = head_dim
        self.layer_idx = 0
        self.q_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, num_kv_heads * head_dim, bias=False)
        torch.manual_seed(seed)
        with torch.no_grad():
            self.q_proj.weight.normal_()
            self.k_proj.weight.normal_()


def _rope_pos_emb(positions, D, base=10000.0):
    """Llama-style ``(cos, sin)`` of shape ``[1, S, D]`` (duplicated halves), fp32."""
    half = D // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))
    freqs = positions.to(torch.float32)[:, None] * inv_freq
    emb = torch.cat([freqs, freqs], dim=-1).unsqueeze(0)
    return emb.cos(), emb.sin()


def _scorer():
    return KnormSketch(compression_ratio=0.5)


# label -> (class, factory for the minimal valid non-schedule kwargs).  Covers
# every compressor that overrides __post_init__: the 15 that previously skipped
# super().__post_init__() plus snapkv/finch as already-correct positive controls,
# and compactor to pin the ridge-vs-compactor fairness pair to the same default
# POST_PREFILL contract.
CASES = {
    "adakv": (AdaKVSketch, lambda: {"press": _scorer()}),
    "block": (BlockSketch, lambda: {"sketch": _scorer()}),
    "chunk": (ChunkSketch, lambda: {"press": _scorer()}),
    "chunkkv": (ChunkKVSketch, lambda: {}),
    "compactor": (CompactorSketch, lambda: {}),
    "composed": (ComposedSketch, lambda: {"presses": ["knorm"]}),
    "criticalkv": (CriticalAdaKVSketch, lambda: {"press": _scorer()}),
    "decoding": (DecodingSketch, lambda: {"base_sketch": _scorer()}),
    "dms": (DMSSketch, lambda: {"press": _scorer(), "threshold": 0.0}),
    "fastkvzip": (FastKVzipSketch, lambda: {}),
    "key_rerotation": (KeyRerotationSketch, lambda: {"press": _scorer()}),
    "kvzip": (KVzipSketch, lambda: {}),
    "per_layer_compression": (
        PerLayerCompressionSketch,
        lambda: {"press": _scorer(), "compression_ratios": [0.5]},
    ),
    "ridge": (RidgeSketch, lambda: {}),
    "simlayerkv": (SimLayerKVSketch, lambda: {}),
    "think": (ThinKSketch, lambda: {}),
    # positive controls — already chained super() before this fix.
    "snapkv": (SnapKVSketch, lambda: {}),
    "finch": (FinchSketch, lambda: {}),
}


class TestCompressionScheduleCoercion(unittest.TestCase):
    def test_string_schedule_is_coerced(self):
        for label, (cls, kwargs) in CASES.items():
            with self.subTest(compressor=label):
                obj = cls(**kwargs(), schedule="decode")
                self.assertIsInstance(obj.schedule, frozenset)
                self.assertEqual(obj.schedule, frozenset({CompressionSchedule.DECODE}))
                # The properties below are exactly what crashed on the raw str.
                self.assertTrue(obj.fires_on_decode)
                self.assertFalse(obj.fires_on_prefill)

    def test_list_schedule_is_coerced(self):
        for label, (cls, kwargs) in CASES.items():
            with self.subTest(compressor=label):
                obj = cls(**kwargs(), schedule=["decode", "streaming"])
                self.assertIsInstance(obj.schedule, frozenset)
                self.assertEqual(
                    obj.schedule,
                    frozenset({CompressionSchedule.DECODE, CompressionSchedule.STREAMING}),
                )
                self.assertTrue(obj.fires_on_decode)
                self.assertTrue(obj.fires_on_prefill)

    def test_default_schedule_is_post_prefill(self):
        for label, (cls, kwargs) in CASES.items():
            with self.subTest(compressor=label):
                obj = cls(**kwargs())
                self.assertEqual(
                    obj.schedule, frozenset({CompressionSchedule.POST_PREFILL})
                )
                self.assertTrue(obj.fires_on_prefill)
                self.assertFalse(obj.fires_on_decode)


class TestPostPrefillFiresBeforeQuestion(unittest.TestCase):
    """End-to-end ordering pin for the ridge-vs-compactor fairness pair.

    Drives ``KVCompressor.forward_hook`` with a real ``DynamicCache`` through
    the pipeline sequence — context prefill (``set_phase("prefill")``), then a
    multi-token question forward and single-token decode steps
    (``set_phase("decode")``) — and pins that the default POST_PREFILL
    schedule invokes ``compress`` exactly once per layer, AFTER the full
    context prefill (compress sees all T context tokens) and BEFORE any
    question token is processed, shrinking every head to exactly
    ``int(T * (1 - ratio))``.  Fake attention modules are Llama-shaped stubs
    (ridge's from its sketch test file, compactor's defined above); no model
    loading.
    """

    B, H_KV, D, HIDDEN_DIM = 1, 2, 8, 32
    # T large enough that (a) RidgeSketch's min_tokens_to_compress default (64)
    # passes and (b) compactor's protected spans (16 + 64 = 80 at defaults) do
    # not consume the whole n_kept budget, so both selections stay score-driven.
    T = 192
    RATIO = 0.5
    N_LAYERS = 2
    N_QUESTION = 5  # q_len > 1 "question" forward after prefill

    def _make_modules(self, module_cls):
        modules = []
        for layer_idx in range(self.N_LAYERS):
            module = module_cls(hidden_dim=self.HIDDEN_DIM, num_heads=4,
                                num_kv_heads=self.H_KV, head_dim=self.D,
                                seed=layer_idx)
            module.layer_idx = layer_idx
            modules.append(module)
        return modules

    def _assert_compresses_once_after_prefill_before_question(
        self, compressor, module_cls
    ):
        from transformers import DynamicCache

        torch.manual_seed(0)
        modules = self._make_modules(module_cls)
        cache = DynamicCache()
        for layer_idx in range(self.N_LAYERS):
            cache.update(torch.randn(self.B, self.H_KV, self.T, self.D),
                         torch.randn(self.B, self.H_KV, self.T, self.D), layer_idx)
        n_kept = int(self.T * (1 - self.RATIO))
        cos_sin = _rope_pos_emb(torch.arange(self.T), self.D)

        with mock.patch.object(compressor, "compress",
                               wraps=compressor.compress) as spy:
            # (a) Context prefill: POST_PREFILL fires once per layer's hook.
            compressor.set_phase("prefill")
            hidden = torch.randn(self.B, self.T, self.HIDDEN_DIM)
            for module in modules:
                kwargs = {
                    "hidden_states": hidden,
                    "past_key_values": cache,
                    "cache_position": torch.arange(self.T),
                    "position_embeddings": cos_sin,
                }
                output = (torch.randn(self.B, self.T, self.HIDDEN_DIM), None)
                self.assertIs(compressor.forward_hook(module, [], kwargs, output), output)

            self.assertEqual(spy.call_count, self.N_LAYERS)  # exactly once per layer
            for call in spy.call_args_list:
                # compress saw the FULL context (all T tokens) and nothing
                # else: it ran after prefill, before any question token existed.
                self.assertEqual(call.args[2].shape[2], self.T)  # keys
                self.assertEqual(call.args[3].shape[2], self.T)  # values
            kept = []
            for layer_idx in range(self.N_LAYERS):
                layer = cache.layers[layer_idx]
                self.assertEqual(tuple(layer.keys.shape), (self.B, self.H_KV, n_kept, self.D))
                self.assertEqual(tuple(layer.values.shape), (self.B, self.H_KV, n_kept, self.D))
                kept.append((layer.keys.clone(), layer.values.clone()))

            # (b) Question forward + decode steps: the hook must stay a pure
            # no-op on the cache (POST_PREFILL never fires on decode).
            compressor.set_phase("decode")
            appended = 0
            for q_len in (self.N_QUESTION, 1, 1):
                pos = self.T + appended
                q_hidden = torch.randn(self.B, q_len, self.HIDDEN_DIM)
                for layer_idx, module in enumerate(modules):
                    cache.update(torch.randn(self.B, self.H_KV, q_len, self.D),
                                 torch.randn(self.B, self.H_KV, q_len, self.D), layer_idx)
                    kwargs = {
                        "hidden_states": q_hidden,
                        "past_key_values": cache,
                        "cache_position": torch.arange(pos, pos + q_len),
                    }
                    out = (torch.randn(self.B, q_len, self.HIDDEN_DIM), None)
                    self.assertIs(compressor.forward_hook(module, [], kwargs, out), out)
                appended += q_len

            self.assertEqual(spy.call_count, self.N_LAYERS)  # never fired again
            for layer_idx in range(self.N_LAYERS):
                layer = cache.layers[layer_idx]
                # Grew only by the appended question/decode tokens ...
                self.assertEqual(layer.keys.shape[2], n_kept + appended)
                self.assertEqual(layer.values.shape[2], n_kept + appended)
                # ... with the compressed prefix left byte-identical.
                self.assertTrue(torch.equal(layer.keys[:, :, :n_kept], kept[layer_idx][0]))
                self.assertTrue(torch.equal(layer.values[:, :, :n_kept], kept[layer_idx][1]))

    def test_ridge_fires_once_after_prefill_before_question(self):
        # min_tokens_to_compress default (64) < T=192 so the gate passes; the
        # 8/64 default windows give keep_mid = 96 - 72 = 24 > 0, so the output
        # hits int(T * (1 - r)) exactly.
        self._assert_compresses_once_after_prefill_before_question(
            RidgeSketch(compression_ratio=self.RATIO), _RidgeFakeAttnModule,
        )

    def test_compactor_fires_once_after_prefill_before_question(self):
        # Deterministic injected phi ([head_dim, k], used verbatim — overrides
        # the shared seeded PHI). Default sinks protect 16 + 64 = 80 tokens
        # (< T=192, so the whole-prompt no-op guard doesn't bite); they score
        # +inf and consume 80 of the n_kept = 96 slots, leaving 16 interior
        # tokens genuinely score-selected.
        torch.manual_seed(42)
        self._assert_compresses_once_after_prefill_before_question(
            CompactorSketch(compression_ratio=self.RATIO, phi=torch.randn(self.D, 4) * 0.5),
            _CompactorFakeAttnModule,
        )


if __name__ == "__main__":
    unittest.main()
