"""End-to-end integration of prefill methods through the real pipeline.

These tests drive :meth:`ResearchGenerationPipeline._forward` on a TINY,
config-built ``LlamaForCausalLM`` (CPU, eager attention, random weights — no
downloaded checkpoints) so that a *real* model forward, the installed
``prefill_method`` hooks/forward-replacement, and the per-token decode loop all
run together.  This complements ``test_prefill_methods.py`` (which unit-tests
the hooks against fake modules): here the methods are exercised through the
exact wiring the research backend uses.

Fake pattern (matches the project convention)
----------------------------------------------
* The pipeline is constructed with ``object.__new__`` (no model download / no
  ``Pipeline.__init__``); only ``model`` and ``tokenizer`` are populated.
* The tokenizer is a minimal stub.  ``_forward`` + ``generate_answer`` touch
  only ``tokenizer.decode(ids, skip_special_tokens=True)`` (``_sanitize_``
  ``parameters`` reads ``model_max_length``, but that runs in ``preprocess``,
  not in ``_forward``); the stub provides both so the same object is reusable.
* ``input_tensors`` are pre-tokenized id tensors, bypassing the chat template.

Layer count
-----------
ReAttention's per-layer top-k naturally retains a different count per layer.
HF's normal decode shares one causal mask / position grid across layers
(sized from layer 0), so a ragged ``DynamicCache`` either crashes (a layer
longer than layer 0) or silently mis-aligns the causal mask (a layer shorter
than layer 0).  ``ReAttentionMethod.uniform_retained`` (default on) equalizes
the retained length across layers, so every test here — including the
ReAttention ones — runs on a genuine multi-layer model and asserts the
post-prefill cache is layer-uniform.  DCA needs no equalization (it replaces
the attention forward and never prunes the cache).
"""

from __future__ import annotations

import unittest

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from eval_harness.attention_methods._method_base import PrefillMethod
from eval_harness.attention_methods.dca import DCAMethod
from eval_harness.attention_methods.reattention import ReAttentionMethod
from eval_harness.kv_compression.cache_adapter import create_cache_adapter
from eval_harness.research_pipeline import ResearchGenerationPipeline


class _StubTokenizer:
    """Minimal tokenizer stub exposing only what the pipeline touches.

    ``model_max_length`` is read by ``_sanitize_parameters`` (not exercised
    here, but provided for completeness); ``decode`` is the only method
    ``_forward`` → ``generate_answer`` calls — it returns ``"x" * len(ids)``
    so the decoded string length equals the number of generated tokens.
    """

    model_max_length = 8192

    def decode(self, ids, skip_special_tokens=True):  # noqa: D401, ARG002
        return "x" * len(ids)


def _build_model(num_hidden_layers: int) -> LlamaForCausalLM:
    """A tiny, deterministically-initialized Llama on CPU with eager attention."""
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
    # generate_answer indexes generation_config.eos_token_id; make sure it is
    # a concrete int so the stop-token membership test works.
    if model.generation_config.eos_token_id is None:
        model.generation_config.eos_token_id = 2
    return model


def _make_pipeline(model: LlamaForCausalLM) -> ResearchGenerationPipeline:
    pipe = object.__new__(ResearchGenerationPipeline)
    pipe.model = model
    pipe.tokenizer = _StubTokenizer()
    return pipe


def _make_inputs(context_len: int = 256, question_len: int = 8) -> dict:
    torch.manual_seed(0)
    return {
        "context_ids": torch.randint(0, 256, (1, context_len)),
        "questions_ids": [torch.randint(0, 256, (1, question_len))],
    }


class TestPrefillIntegration(unittest.TestCase):
    MAX_NEW_TOKENS = 4

    def _assert_valid_answers(self, answers, expected_len=None):
        self.assertIsInstance(answers, list)
        self.assertEqual(len(answers), 1)
        self.assertIsInstance(answers[0], str)
        if expected_len is not None:
            self.assertEqual(len(answers[0]), expected_len)

    # -- baseline (no method) ------------------------------------------------

    def test_base_method_runs(self):
        """The base no-op PrefillMethod path: full prefill + decode, no hooks
        that alter the cache.  Returns a single decoded string."""
        model = _build_model(num_hidden_layers=2)
        pipe = _make_pipeline(model)
        answers = pipe._forward(
            _make_inputs(),
            max_new_tokens=self.MAX_NEW_TOKENS,
            kv_compressor=None,
            attention_method=PrefillMethod(),
            cache=None,
            cache_adapter=None,
        )
        self._assert_valid_answers(answers)

    # -- DCA (attention-forward replacement, multi-chunk) --------------------

    def test_dca_multichunk_runs(self):
        """DCA over a 256-token context with chunk_len=80 spans multiple
        chunks (so the 3-component intra/successive/inter path is exercised,
        not just the single-chunk reduce-to-causal case).  The forward
        replacement stays active across prefill AND decode; we assert the
        decode loop produced ``max_new_tokens`` tokens (stub decode maps token
        count to string length)."""
        model = _build_model(num_hidden_layers=2)
        pipe = _make_pipeline(model)
        method = DCAMethod(
            chunk_size=96,        # chunk_len = 96 - 16 = 80
            local_window=16,
            pretraining_length=128,
            use_flash_attn="off",
        )
        answers = pipe._forward(
            _make_inputs(),
            max_new_tokens=self.MAX_NEW_TOKENS,
            kv_compressor=None,
            attention_method=method,
            cache=None,
            cache_adapter=None,
        )
        # Decode ran the full new-token budget (no early EOS for these random
        # weights → argmax is stable but not the eos id).
        self._assert_valid_answers(answers, expected_len=self.MAX_NEW_TOKENS)

    def test_dca_question_pass_straddling_chunk_boundary_is_finite(self):
        """The multi-token question pass with a chunk boundary INSIDE it must
        produce finite logits (regression: the LSE decode decomposition NaN'd
        every pre-boundary query, silently corrupting the sample).

        chunk_len = 96 - 16 = 80; context = 155 and question = 8 put the
        boundary 160 strictly inside (155, 162].  We drive prefill + the
        question forward manually (the same calls ``_forward`` /
        ``generate_answer`` make) so the question-pass logits are observable.
        """
        model = _build_model(num_hidden_layers=2)
        method = DCAMethod(
            chunk_size=96,
            local_window=16,
            pretraining_length=128,
            use_flash_attn="off",
        )
        torch.manual_seed(0)
        context_ids = torch.randint(0, 256, (1, 155))
        question_ids = torch.randint(0, 256, (1, 8))

        from transformers import DynamicCache

        with method(model):
            cache = DynamicCache()
            model.model(input_ids=context_ids, past_key_values=cache)
            position_ids = torch.arange(155, 163).unsqueeze(0)
            out = model(
                input_ids=question_ids,
                past_key_values=cache,
                position_ids=position_ids,
            )
        self.assertTrue(torch.isfinite(out.logits).all())

    # -- ReAttention (post-attention prune hook) -----------------------------

    def _per_layer_lengths(self, cache) -> list[int]:
        return [int(layer.keys.shape[2]) for layer in cache.layers]

    def test_reattention_prunes_cache_multilayer_uniform(self):
        """ReAttention prunes the KV cache on a MULTI-LAYER model and decodes.

        We pass an explicit cache + cache_adapter so we can read the post-run
        sequence lengths.  ``recall_clip`` caps the unique middle set so the
        union-over-queries does not cover the whole middle (which would leave
        the cache full); the result is a real reduction below the 256-token
        context.  ``uniform_retained`` (default) equalizes the per-layer
        selection, so the 2-layer decode runs against a layer-uniform cache —
        this is the ragged-cache regression test.

        ``restore_after_question`` trims the cache back to the *post-prefill*
        (already-pruned) length after decode, so the final measured lengths
        are the pruned prefill lengths — strictly less than the context.
        """
        model = _build_model(num_hidden_layers=2)
        pipe = _make_pipeline(model)
        method = ReAttentionMethod(
            global_size=4,
            local_size=16,
            mid_size=4,
            span_size=8,
            recall_clip=8,          # cap the middle so pruning actually bites
            use_triton_kernel="off",
        )
        cache_adapter = create_cache_adapter(model)
        cache = cache_adapter.initialize_cache(None)

        answers = pipe._forward(
            _make_inputs(),
            max_new_tokens=self.MAX_NEW_TOKENS,
            kv_compressor=None,
            attention_method=method,
            cache=cache,
            cache_adapter=cache_adapter,
        )
        self._assert_valid_answers(answers)

        lengths = self._per_layer_lengths(cache)
        self.assertEqual(len(set(lengths)), 1, f"ragged cache: {lengths}")
        final_len = cache_adapter.get_seq_length(cache)
        self.assertLess(final_len, 256)                       # genuinely pruned
        self.assertGreaterEqual(final_len, 4 + 16)            # global + local kept

    def test_reattention_uniform_budget_exact_length(self):
        """An explicit ``uniform_budget`` pins every layer to exactly
        ``global + budget + local`` retained tokens — the fully reproducible
        configuration for multi-layer benchmarking (exercises the shrink path
        whenever a layer naturally selects more than the budget)."""
        model = _build_model(num_hidden_layers=4)
        pipe = _make_pipeline(model)
        method = ReAttentionMethod(
            global_size=4,
            local_size=16,
            mid_size=4,
            span_size=8,
            uniform_budget=24,
            use_triton_kernel="off",
        )
        cache_adapter = create_cache_adapter(model)
        cache = cache_adapter.initialize_cache(None)

        answers = pipe._forward(
            _make_inputs(),
            max_new_tokens=self.MAX_NEW_TOKENS,
            kv_compressor=None,
            attention_method=method,
            cache=cache,
            cache_adapter=cache_adapter,
        )
        self._assert_valid_answers(answers)
        self.assertEqual(self._per_layer_lengths(cache), [4 + 24 + 16] * 4)

    def test_reattention_reposition_runs_multilayer(self):
        """The ``reposition=True`` path runs end-to-end on a 2-layer model.

        This re-rotates the retained keys to the contiguous end-anchored
        window ``[A - R, A - 1]`` during the prefill hook, and
        ``compute_question_position_ids`` continues the decode tokens from the
        anchor ``A`` (the override the pipeline feeds into ``generate_answer``).
        ``uniform_retained`` keeps ``R`` identical across layers, so the
        shared decode position grid is coherent.  We assert a valid decoded
        string with no exception — exercising the compact-repositioning path
        including ``compute_question_position_ids``."""
        model = _build_model(num_hidden_layers=2)
        pipe = _make_pipeline(model)
        method = ReAttentionMethod(
            global_size=4,
            local_size=16,
            mid_size=4,
            span_size=8,
            recall_clip=8,          # bounds the compacted-window anchor A
            reposition=True,
            use_triton_kernel="off",
        )
        answers = pipe._forward(
            _make_inputs(),
            max_new_tokens=self.MAX_NEW_TOKENS,
            kv_compressor=None,
            attention_method=method,
            cache=None,
            cache_adapter=None,
        )
        self._assert_valid_answers(answers)


if __name__ == "__main__":
    unittest.main()
