"""The drift guard: the benchmark must measure what the eval harness runs.

``eval_harness/profiling/runner.py`` deliberately does NOT re-implement the
generation path -- it instruments the shipped one from outside.  These tests pin
that:

* instrumenting must not change the generated tokens (byte-identical output),
* disabling EOS must yield an exact, method-independent step count,
* the compressor must still fire once per layer with the hook wrapper installed,
* and the post-prefill cache must land on ``int(T*(1-r))`` for every method.

If someone later reimplements the decode loop inside the benchmark, the first
test fails.  Tiny config-built Llama on CPU, no weights downloaded -- the
``test_chunked_prefill.py`` idiom.
"""

from __future__ import annotations

import unittest

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from eval_harness.profiling.audit import expected_budget
from eval_harness.profiling.cell import PerfCell
from eval_harness.profiling.runner import BenchRuntime, time_cell
from eval_harness.research_pipeline import ResearchGenerationPipeline

METHODS = ["none", "knorm", "keydiff", "cur", "snapkv", "streaming_llm"]
CTX, RATIO, STEPS = 256, 0.9, 6


class _Tokenizer:
    """Deterministic word-level stand-in; no download."""

    model_max_length = 4096

    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        ids = [(sum(ord(c) for c in w) % 200) + 3 for w in text.split()]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}

    def decode(self, ids, skip_special_tokens=True):
        return ",".join(str(int(i)) for i in ids)


def _build_model(layers: int = 2) -> LlamaForCausalLM:
    cfg = LlamaConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=layers,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=256,
        max_position_embeddings=4096, rope_theta=10000.0, attn_implementation="eager",
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg).eval()
    model.generation_config.eos_token_id = 2
    return model


def _pipeline(model):
    pipe = object.__new__(ResearchGenerationPipeline)
    pipe.model = model
    pipe.tokenizer = _Tokenizer()
    return pipe


def _runtime(model, pipe):
    return BenchRuntime(
        model=model, tokenizer=_Tokenizer(), pipe=pipe, adapter=None, device="cpu",
        hf_model="tiny", attn_impl_actual="eager", dtype="float32",
        weights_bytes=sum(p.numel() * p.element_size() for p in model.parameters()),
        max_position_embeddings=4096, n_layers=model.config.num_hidden_layers,
    )


def _cell(method, **kw):
    params = dict(
        model_key="tiny", hf_model="tiny", method=method,
        compression_ratio=0.0 if method == "none" else RATIO,
        context_tokens=CTX, attn_impl="eager", dtype="float32",
        decode_steps=STEPS, warmup_repeats=0, repeats=1,
    )
    params.update(kw)
    return PerfCell(**params)


def _compressor(method, ratio):
    from eval_harness.kv_compression import get_kv_compressor
    return None if method == "none" else get_kv_compressor(method, compression_ratio=ratio)


class TestInstrumentationIsTransparent(unittest.TestCase):
    """Instrumenting must not perturb the computation."""

    def test_generated_tokens_match_uninstrumented_forward(self):
        for method in METHODS:
            with self.subTest(method=method):
                from eval_harness.kv_compression.cache_adapter import create_cache_adapter

                model = _build_model()
                pipe = _pipeline(model)
                ratio = 0.0 if method == "none" else RATIO
                ctx = torch.randint(0, 200, (1, CTX))
                q = _Tokenizer()("a b c d")["input_ids"]

                # Reference: the shipped path, no instrumentation, EOS disabled so
                # both sides run the same number of steps.
                saved = model.generation_config.eos_token_id
                model.generation_config.eos_token_id = [-1]
                ca = create_cache_adapter(model)
                cache = ca.initialize_cache(None)
                with torch.no_grad():
                    ref = pipe._forward({"context_ids": ctx, "questions_ids": [q]},
                                        max_new_tokens=STEPS + 1,
                                        kv_compressor=_compressor(method, ratio),
                                        cache=cache, cache_adapter=ca)
                model.generation_config.eos_token_id = saved
                ref_cache_len = ca.get_seq_length(cache)

                # Measured: the same computation, wrapped in timers.
                from eval_harness.profiling import runner as R
                orig = R.build_exact_prompt
                R.build_exact_prompt = lambda tok, n, **kw: ("", ctx)
                try:
                    payload = time_cell(_runtime(model, pipe), _cell(method),
                                        question="a b c d", measure_compression=True)
                finally:
                    R.build_exact_prompt = orig

                self.assertEqual(payload["decode"]["answer_head"], (ref[0] or "")[:60],
                                 f"{method}: instrumentation changed the generated tokens")
                self.assertEqual(payload["kv_cache"]["seq_len_max"], ref_cache_len)


class TestBudgetAndStepCount(unittest.TestCase):
    def test_cache_lands_on_the_exact_budget(self):
        for method in METHODS:
            with self.subTest(method=method):
                model = _build_model()
                payload = time_cell(_runtime(model, _pipeline(model)), _cell(method))
                want = CTX if method == "none" else expected_budget(CTX, RATIO)
                self.assertEqual(payload["kv_cache"]["seq_len_max"], want)

    def test_eos_does_not_truncate_the_decode_loop(self):
        """The shipped loop breaks on EOS (research_pipeline.py:477).

        The benchmark disables it via generation_config so every cell contributes
        an identical number of samples to the tok/s denominator.
        """
        model = _build_model()
        # Make EOS the token the model actually emits first, so an enabled EOS
        # would certainly stop the loop.
        pipe = _pipeline(model)
        payload = time_cell(_runtime(model, pipe), _cell("knorm", repeats=2))
        # N decode forwards yield N-1 token-to-token periods per repeat.
        self.assertEqual(payload["decode"]["n_samples"], (STEPS - 1) * 2)
        self.assertTrue(payload["decode"]["eos_disabled"])
        # generation_config restored afterwards
        self.assertEqual(model.generation_config.eos_token_id, 2)

    def test_repeats_and_warmup_shape(self):
        model = _build_model()
        payload = time_cell(_runtime(model, _pipeline(model)),
                            _cell("knorm", repeats=3, warmup_repeats=1))
        self.assertEqual(len(payload["decode"]["raw_ms"]), 3)
        for arr in payload["decode"]["raw_ms"]:
            self.assertEqual(len(arr), STEPS - 1)   # warmup repeats excluded
        self.assertEqual(payload["decode"]["n_samples"], (STEPS - 1) * 3)


class TestLatencyDefinition(unittest.TestCase):
    """The headline latency must be the token-to-token PERIOD, not the forward span.

    research_pipeline.py:475-477 does `logits.argmax()` then a blocking
    `new_id.item()` between two forwards. Timing only the forward would drop that
    host-side work from the inter-token latency.
    """

    def test_period_is_at_least_the_forward_span(self):
        model = _build_model()
        p = time_cell(_runtime(model, _pipeline(model)), _cell("knorm", repeats=2))
        period = p["decode"]["per_step"]["median"]
        forward = p["decode"]["forward_device_ms"]["median"]
        self.assertGreaterEqual(period, forward)
        self.assertIn("token-to-token", p["decode"]["latency_definition"])

    def test_forward_series_is_aligned_with_the_period_series(self):
        """Both series must cover the same forwards, or period - forward is meaningless."""
        model = _build_model()
        p = time_cell(_runtime(model, _pipeline(model)), _cell("knorm", repeats=1))
        self.assertEqual(p["decode"]["forward_device_ms"]["n"], p["decode"]["per_step"]["n"])


class TestCompressionStage(unittest.TestCase):
    def test_hook_fires_once_per_layer(self):
        """Per-layer F6 timing, via an instance-level forward_hook wrap."""
        model = _build_model(layers=3)
        payload = time_cell(_runtime(model, _pipeline(model)),
                            _cell("knorm"), measure_compression=True)
        self.assertEqual(payload["compression_stage"]["n_calls"], 3)

    def test_anchor_has_no_compression_stage(self):
        model = _build_model()
        payload = time_cell(_runtime(model, _pipeline(model)),
                            _cell("none"), measure_compression=True)
        self.assertNotIn("compression_stage", payload)


class TestPayloadShape(unittest.TestCase):
    def test_required_fields_present_and_jsonable(self):
        import json
        from eval_harness.profiling.cell import _jsonable

        model = _build_model()
        payload = time_cell(_runtime(model, _pipeline(model)), _cell("knorm"))
        for path in (("prefill", "summary", "median"), ("ttft", "ttft_ms", "median"),
                     ("decode", "per_step", "median"), ("decode", "throughput_tok_s"),
                     ("kv_cache", "seq_len_max"), ("memory", "weights_bytes")):
            node = payload
            for k in path:
                self.assertIn(k, node, f"missing {'.'.join(path)}")
                node = node[k]
            self.assertIsNotNone(node)
        json.dumps(_jsonable(payload))

    def test_ttft_is_prefill_plus_question_block(self):
        model = _build_model()
        p = time_cell(_runtime(model, _pipeline(model)), _cell("knorm"))
        self.assertAlmostEqual(
            p["ttft"]["ttft_ms"]["median"],
            p["prefill"]["summary"]["median"] + p["ttft"]["question_block_ms"]["median"],
            places=5)


if __name__ == "__main__":
    unittest.main()
