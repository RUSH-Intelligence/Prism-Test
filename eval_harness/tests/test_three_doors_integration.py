"""Integration of the three doors through the real adapter + pipeline (Step 8).

* The adapter resolves all three doors from a ``ResearchConfig`` (positional via
  the positional registry, attention via the unified resolver, kv via the
  kv-compression registry).
* The pipeline composes a positional method (door 1) with a KV compressor
  (door 3) end-to-end on a tiny config-built Llama: the run completes and the
  compressor actually prunes the post-prefill cache.
"""

from __future__ import annotations

import unittest

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from eval_harness.kv_compression import KnormSketch
from eval_harness.kv_compression.cache_adapter import create_cache_adapter
from eval_harness.positional_methods import PositionalMethod
from eval_harness.positional_methods.linear_pi import LinearPIMethod
from eval_harness.positional_methods.yarn import YaRNMethod
from eval_harness.research_adapter import ResearchAdapter, ResearchConfig
from eval_harness.research_pipeline import SketchTextGenerationPipeline


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


def _make_pipeline(model) -> SketchTextGenerationPipeline:
    pipe = object.__new__(SketchTextGenerationPipeline)
    pipe.model = model
    pipe.tokenizer = _StubTokenizer()
    return pipe


def _inputs(context_len=200, q_len=8):
    torch.manual_seed(1)
    return {
        "context_ids": torch.randint(0, 256, (1, context_len)),
        "questions_ids": [torch.randint(0, 256, (1, q_len))],
    }


class TestAdapterBuildsThreeDoors(unittest.TestCase):
    def test_build_each_door_from_config(self):
        cfg = ResearchConfig(
            positional_method="yarn",
            positional_method_kwargs={"factor": 4.0, "original_max_position_embeddings": 2048},
            attention_method="none",
            kv_compressor="knorm",
            compression_ratio=0.5,
        )
        self.assertIsInstance(ResearchAdapter._build_positional_method(cfg), YaRNMethod)
        self.assertIsNone(ResearchAdapter._build_attention_method(cfg))
        kv = self.__class__._adapter_kv(cfg)
        self.assertIsInstance(kv, KnormSketch)
        self.assertAlmostEqual(kv.compression_ratio, 0.5)

    @staticmethod
    def _adapter_kv(cfg):
        shell = object.__new__(ResearchAdapter)
        return shell._build_kv_compressor(cfg)

    def test_positional_none_is_none(self):
        cfg = ResearchConfig(positional_method="none")
        self.assertIsNone(ResearchAdapter._build_positional_method(cfg))

    def test_compression_schedule_forwarded(self):
        cfg = ResearchConfig(kv_compressor="knorm", compression_ratio=0.3,
                             compression_schedule=["decode"])
        kv = self._adapter_kv(cfg)
        self.assertTrue(kv.fires_on_decode)
        self.assertFalse(kv.fires_on_prefill)


class TestPipelineComposesDoors(unittest.TestCase):
    def _run(self, positional_method, sketch):
        model = _build_model()
        pipe = _make_pipeline(model)
        adapter = create_cache_adapter(model)
        cache = adapter.initialize_cache(None)
        with torch.no_grad():
            answers = pipe._forward(
                _inputs(),
                max_new_tokens=4,
                sketch=sketch,
                prefill_method=None,
                positional_method=positional_method,
                cache=cache,
                cache_adapter=adapter,
            )
        return answers, cache, adapter

    def test_positional_plus_kv_compressor_compose(self):
        """Door 1 (Linear-PI) + Door 3 (knorm prune) run together; the run
        completes and the cache is pruned below the context length."""
        answers, cache, adapter = self._run(
            LinearPIMethod(factor=2.0), KnormSketch(compression_ratio=0.5),
        )
        self.assertEqual(len(answers), 1)
        self.assertIsInstance(answers[0], str)
        # knorm at ratio 0.5 prunes ~half the 200-token prefill cache.
        self.assertLess(adapter.get_seq_length(cache), 200)

    def test_identity_positional_noop_matches_plain(self):
        """The base (identity) PositionalMethod composes without changing the
        no-compressor answer vs. no positional door."""
        plain, _, _ = self._run(None, None)
        with_identity, _, _ = self._run(PositionalMethod(), None)
        self.assertEqual(plain, with_identity)


if __name__ == "__main__":
    unittest.main()
