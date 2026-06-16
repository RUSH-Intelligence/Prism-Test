"""Equivalence guard for chunked prefill (Step 2).

Drives the real ``ResearchGenerationPipeline._run_prefill`` / ``_forward`` on
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

from eval_harness.attention_methods._method_base import PrefillMethod
from eval_harness.kv_compression import KnormSketch
from eval_harness.kv_compression.cache_adapter import create_cache_adapter
from eval_harness.research_pipeline import ResearchGenerationPipeline


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


def _make_pipeline(model: LlamaForCausalLM) -> ResearchGenerationPipeline:
    pipe = object.__new__(ResearchGenerationPipeline)
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
                    kv_compressor=None,
                    attention_method=PrefillMethod(),
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


class TestStreamingCompressionBounded(unittest.TestCase):
    """``streaming`` KV compression must fire after EVERY prefill chunk.

    Regression for the cache_position heuristic that labelled every chunk past
    the first as decode — a ``streaming`` compressor (``fires_on_decode=False``)
    was then skipped on all but the first chunk, so the cache grew unbounded.
    The pipeline now declares the phase explicitly and feeds each chunk physical
    ``cache_position`` derived from the (post-eviction) cache length.
    """

    CONTEXT_LEN = 200
    CHUNK = 40

    def _post_prefill_seq_len(self, sketch, chunk_size):
        model = _build_model()
        pipe = _make_pipeline(model)
        torch.manual_seed(4)
        inputs = {
            "context_ids": torch.randint(0, 256, (1, self.CONTEXT_LEN)),
            "questions_ids": [torch.randint(0, 256, (1, 6))],
        }
        adapter = create_cache_adapter(model)
        cache = adapter.initialize_cache(None)
        with torch.no_grad():
            answers = pipe._forward(
                inputs,
                max_new_tokens=3,
                kv_compressor=sketch,
                attention_method=None,
                positional_method=None,
                prefill_chunk_size=chunk_size,
                cache=cache,
                cache_adapter=adapter,
            )
        # The decode loop checkpoints/restores the cache per question, so after
        # _forward the cache is back at its post-prefill (compressed) length.
        return adapter.get_seq_length(cache), answers

    def test_streaming_bounds_cache_across_chunks(self):
        """50% streaming eviction with 5 chunks keeps the cache near the chunk
        size (~38), NOT the ~180 the single-fire bug would leave."""
        seq_len, answers = self._post_prefill_seq_len(
            KnormSketch(compression_ratio=0.5, schedule=["streaming"]),
            chunk_size=self.CHUNK,
        )
        # Each fire halves a cache that never exceeds prev+CHUNK <= 2*CHUNK, so
        # the post-prefill length is bounded by CHUNK and converges up toward it
        # (20 -> 30 -> 35 -> 37 -> 38 here).  The single-fire bug would instead
        # leave 0.5*CHUNK + 4*CHUNK = 180.  Bound tightly to [CHUNK//2, CHUNK].
        self.assertLessEqual(seq_len, self.CHUNK)
        self.assertGreater(seq_len, self.CHUNK // 2)
        # Decode still completes (positions stay consistent after eviction).
        self.assertIsInstance(answers[0], str)

    def test_streaming_single_pass_fires_once(self):
        """With no chunking (single pass) streaming coincides with a single
        post_prefill fire: 50% of 200 == ~100."""
        seq_len, _ = self._post_prefill_seq_len(
            KnormSketch(compression_ratio=0.5, schedule=["streaming"]),
            chunk_size=None,
        )
        self.assertEqual(seq_len, self.CONTEXT_LEN // 2)


class TestPostPrefillFinalChunkOnly(unittest.TestCase):
    """``POST_PREFILL`` must fire ONLY after the full prefill cache exists.

    With chunked prefill the hook is invoked after every layer of every chunk.
    Without gating, a POST_PREFILL compressor would prune incrementally — each
    chunk seeing a cache already shrunk by prior chunks, scoring against fewer
    keys, and producing a final cache that differs from the single-pass result.
    The pipeline's per-chunk ``set_prefill_is_final`` toggle restores byte
    identity by suppressing POST_PREFILL on every chunk except the last.
    """

    CONTEXT_LEN = 200

    def _post_prefill_kv(self, sketch, chunk_size):
        model = _build_model()
        pipe = _make_pipeline(model)
        torch.manual_seed(5)
        inputs = {
            "context_ids": torch.randint(0, 256, (1, self.CONTEXT_LEN)),
            "questions_ids": [torch.randint(0, 256, (1, 6))],
        }
        adapter = create_cache_adapter(model)
        cache = adapter.initialize_cache(None)
        with torch.no_grad():
            pipe._forward(
                inputs,
                max_new_tokens=3,
                kv_compressor=sketch,
                attention_method=None,
                positional_method=None,
                prefill_chunk_size=chunk_size,
                cache=cache,
                cache_adapter=adapter,
            )
        return adapter.get_seq_length(cache), _cache_kv(cache)

    def test_post_prefill_chunked_matches_single_pass(self):
        """Chunked POST_PREFILL must produce the same cache as single-pass."""
        single_len, single_kv = self._post_prefill_kv(
            KnormSketch(compression_ratio=0.4, schedule=["post_prefill"]),
            chunk_size=None,
        )
        self.assertEqual(single_len, int(self.CONTEXT_LEN * 0.6))

        for chunk_size in (37, 64, 100):
            chunked_len, chunked_kv = self._post_prefill_kv(
                KnormSketch(compression_ratio=0.4, schedule=["post_prefill"]),
                chunk_size=chunk_size,
            )
            self.assertEqual(
                chunked_len, single_len,
                f"post-prefill length differs at chunk={chunk_size}: "
                f"{chunked_len} vs {single_len}",
            )
            for li, ((k1, v1), (k2, v2)) in enumerate(zip(single_kv, chunked_kv)):
                self.assertEqual(k1.shape, k2.shape, f"chunk={chunk_size} layer={li}")
                self.assertTrue(
                    torch.allclose(k1, k2, atol=1e-6, rtol=1e-5),
                    f"keys differ chunk={chunk_size} layer={li}",
                )
                self.assertTrue(
                    torch.allclose(v1, v2, atol=1e-6, rtol=1e-5),
                    f"values differ chunk={chunk_size} layer={li}",
                )

    def test_streaming_still_fires_per_chunk(self):
        """Sanity: STREAMING must NOT be affected by the POST_PREFILL gate —
        it still evicts after every chunk, leaving a much smaller final cache
        than the equivalent POST_PREFILL run."""
        # streaming at 50% across 5 chunks of 40 converges near CHUNK (per the
        # existing test); post_prefill at 50% leaves exactly CONTEXT/2 = 100.
        streaming_len, _ = self._post_prefill_kv(
            KnormSketch(compression_ratio=0.5, schedule=["streaming"]),
            chunk_size=40,
        )
        post_len, _ = self._post_prefill_kv(
            KnormSketch(compression_ratio=0.5, schedule=["post_prefill"]),
            chunk_size=40,
        )
        self.assertLessEqual(streaming_len, 40)
        self.assertEqual(post_len, self.CONTEXT_LEN // 2)


if __name__ == "__main__":
    unittest.main()
