"""Equivalence guard for chunked prefill (Step 2).

Drives the real ``SketchTextGenerationPipeline._run_prefill`` / ``_forward`` on
a TINY config-built Llama (CPU, eager, random weights — same fake pattern as
``test_prefill_integration.py``).  The contract:

* ``prefill_chunk_size=None`` takes the original single-pass code path verbatim.
* A chunked prefill with correct absolute positions produces a **byte-identical**
  post-prefill cache and identical decoded answers vs. the single pass, because
  a plain causal forward is invariant to how the context is chunked.
"""

from __future__ import annotations

import unittest

import torch
from transformers import DynamicCache, LlamaConfig, LlamaForCausalLM

from eval_harness.prefill_methods.base import PrefillMethod
from eval_harness.sketch.cache_adapter import create_cache_adapter
from eval_harness.sketch.pipeline import SketchTextGenerationPipeline


class _StubTokenizer:
    model_max_length = 8192

    def decode(self, ids, skip_special_tokens=True):  # noqa: ARG002
        return "x" * len(ids)


def _build_model(num_hidden_layers: int = 2) -> LlamaForCausalLM:
    cfg = LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=256,
        max_position_embeddings=8192,
        rope_theta=10000.0,
        attn_implementation="eager",
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg).eval()
    if model.generation_config.eos_token_id is None:
        model.generation_config.eos_token_id = 2
    return model


def _make_pipeline(model: LlamaForCausalLM) -> SketchTextGenerationPipeline:
    pipe = object.__new__(SketchTextGenerationPipeline)
    pipe.model = model
    pipe.tokenizer = _StubTokenizer()
    return pipe


def _cache_kv(cache) -> list:
    return [(layer.keys.clone(), layer.values.clone()) for layer in cache.layers]


class TestChunkedPrefillEquivalence(unittest.TestCase):
    CONTEXT_LEN = 200

    def _prefill(self, pipe, context_ids, chunk_size):
        cache = DynamicCache()
        with torch.no_grad():
            pipe._run_prefill(
                context_ids=context_ids, cache=cache, prefill_chunk_size=chunk_size,
            )
        return cache

    def test_none_chunk_size_is_single_pass(self):
        """``prefill_chunk_size=None`` and ``>= length`` both take the single
        full-context pass and yield the same cache."""
        model = _build_model()
        pipe = _make_pipeline(model)
        torch.manual_seed(1)
        ctx = torch.randint(0, 256, (1, self.CONTEXT_LEN))

        single = self._prefill(pipe, ctx, None)
        oversized = self._prefill(pipe, ctx, self.CONTEXT_LEN + 50)

        for (k1, v1), (k2, v2) in zip(_cache_kv(single), _cache_kv(oversized)):
            self.assertTrue(torch.equal(k1, k2))
            self.assertTrue(torch.equal(v1, v2))

    def test_chunked_cache_matches_single_pass(self):
        """A multi-chunk prefill produces a cache numerically equal to the
        single pass (chunk boundaries land both on and off divisors)."""
        model = _build_model()
        pipe = _make_pipeline(model)
        torch.manual_seed(2)
        ctx = torch.randint(0, 256, (1, self.CONTEXT_LEN))

        single = _cache_kv(self._prefill(pipe, ctx, None))
        for chunk_size in (1, 17, 64, 100):
            chunked = _cache_kv(self._prefill(pipe, ctx, chunk_size))
            self.assertEqual(len(single), len(chunked))
            for li, ((k1, v1), (k2, v2)) in enumerate(zip(single, chunked)):
                self.assertEqual(k1.shape, k2.shape, f"chunk={chunk_size} layer={li}")
                self.assertTrue(
                    torch.allclose(k1, k2, atol=1e-6, rtol=1e-5),
                    f"keys differ chunk={chunk_size} layer={li} "
                    f"max|Δ|={(k1 - k2).abs().max().item():.2e}",
                )
                self.assertTrue(
                    torch.allclose(v1, v2, atol=1e-6, rtol=1e-5),
                    f"values differ chunk={chunk_size} layer={li}",
                )

    def test_forward_answers_match_single_pass(self):
        """End-to-end through ``_forward``: chunked prefill yields the same
        decoded answer as the single pass (no doors installed)."""
        model = _build_model()
        pipe = _make_pipeline(model)
        torch.manual_seed(3)
        inputs = {
            "context_ids": torch.randint(0, 256, (1, self.CONTEXT_LEN)),
            "questions_ids": [torch.randint(0, 256, (1, 8))],
        }

        def run(chunk_size):
            adapter = create_cache_adapter(model)
            with torch.no_grad():
                return pipe._forward(
                    inputs,
                    max_new_tokens=5,
                    sketch=None,
                    prefill_method=PrefillMethod(),
                    prefill_chunk_size=chunk_size,
                    cache=adapter.initialize_cache(None),
                    cache_adapter=adapter,
                )

        baseline = run(None)
        self.assertEqual(run(48), baseline)
        self.assertEqual(run(1), baseline)

    def test_zero_chunk_size_rejected(self):
        model = _build_model()
        pipe = _make_pipeline(model)
        ctx = torch.randint(0, 256, (1, 16))
        with self.assertRaises(ValueError):
            pipe._run_prefill(context_ids=ctx, cache=DynamicCache(), prefill_chunk_size=0)


if __name__ == "__main__":
    unittest.main()
