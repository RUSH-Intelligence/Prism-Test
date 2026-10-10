"""Teacher / student execution on a tiny config-built Llama through the REAL
ResearchGenerationPipeline prefill and the knorm compressor (CPU, fp32, eager)."""
from __future__ import annotations

import copy
import unittest

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from eval_harness.kv_compression import KnormSketch
from eval_harness.kv_compression.cache_adapter import create_cache_adapter
from eval_harness.kv_recovery.alignment import hidden_loss, position_index
from eval_harness.kv_recovery.config import PositionsCfg, RecoveryConfig
from eval_harness.kv_recovery.hidden_states import FINAL_NORM_KEY, capture_layer_outputs, gather_positions
from eval_harness.kv_recovery.model_spec import inspect_model
from eval_harness.kv_recovery.student import (
    Example,
    assert_budget,
    assert_no_hooks,
    build_compressor,
    expected_budget,
    prefill_context,
    probe_block_continuation,
    resolve_segment_mode,
    run_student,
    run_teacher,
    segment_forward,
)
from eval_harness.research_adapter import ResearchAdapter
from eval_harness.research_pipeline import ResearchGenerationPipeline

try:
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    HAS_QWEN35 = True
except Exception:  # pragma: no cover
    HAS_QWEN35 = False


class _StubTokenizer:
    model_max_length = 8192
    bos_token = None

    def decode(self, ids, skip_special_tokens=True):  # noqa: ARG002
        return "x" * len(ids)


def _tiny_llama(num_hidden_layers: int = 2) -> LlamaForCausalLM:
    cfg = LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=num_hidden_layers,
                      num_attention_heads=4, num_key_value_heads=2, vocab_size=256,
                      max_position_embeddings=8192, rope_theta=10000.0, attn_implementation="eager")
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg).eval()
    with torch.no_grad():   # scale-diverse keys so the compressor has something to rank
        for p in model.parameters():
            p.mul_(4.0 if p.dim() > 1 else 1.0)
    model.requires_grad_(False)
    if model.generation_config.eos_token_id is None:
        model.generation_config.eos_token_id = 2
    return model


def _shell(model) -> ResearchAdapter:
    """A ResearchAdapter without weights loading (the repo's object.__new__ test idiom)."""
    adapter = object.__new__(ResearchAdapter)
    adapter._model = model
    adapter._tokenizer = _StubTokenizer()
    pipe = object.__new__(ResearchGenerationPipeline)
    pipe.model = model
    pipe.tokenizer = adapter._tokenizer
    adapter._pipe = pipe
    adapter._cache_adapter = create_cache_adapter(model)
    return adapter


def _example(T=40, L=8, seed=1) -> Example:
    g = torch.Generator().manual_seed(seed)
    return Example(id=f"ex{seed}", ctx_ids=torch.randint(0, 256, (1, T), generator=g),
                   suffix_ids=torch.randint(0, 256, (1, L), generator=g))


class TestPrefillAndBudget(unittest.TestCase):
    def setUp(self):
        self.model = _tiny_llama()
        self.adapter = _shell(self.model)
        self.spec = inspect_model(self.model)
        self.ex = _example()

    def test_dense_prefill_keeps_full_cache(self):
        cache = prefill_context(self.adapter, self.ex.ctx_ids, None)
        self.assertEqual(self.adapter._cache_adapter.get_seq_length(cache), 40)
        assert_budget(cache, self.spec, 40, 0.0)
        assert_no_hooks(self.model)

    def test_compressed_prefill_hits_budget_and_removes_hooks(self):
        comp = KnormSketch(compression_ratio=0.5)
        cache = prefill_context(self.adapter, self.ex.ctx_ids, comp)
        self.assertEqual(expected_budget(40, 0.5), 20)
        assert_budget(cache, self.spec, 40, 0.5)
        assert_no_hooks(self.model)              # context manager removed the hooks
        with self.assertRaises(AssertionError):
            assert_budget(cache, self.spec, 40, 0.0)
        # The suffix appends L entries to every layer.
        segment_forward(self.model, cache, self.ex.suffix_ids, 40, logits_to_keep=1)
        assert_budget(cache, self.spec, 40, 0.5, suffix_appended=8)

    def test_build_compressor_from_config(self):
        cfg = RecoveryConfig.from_dict({"kv_compression": {"kv_compressor": "knorm", "compression_ratio": 0.5}})
        comp = build_compressor(cfg)
        self.assertIsInstance(comp, KnormSketch)
        self.assertAlmostEqual(comp.compression_ratio, 0.5)
        dense = RecoveryConfig.from_dict({"kv_compression": {"kv_compressor": "none"}})
        self.assertIsNone(build_compressor(dense))
        decode_only = RecoveryConfig.from_dict({"kv_compression": {"kv_compressor": "knorm", "compression_ratio": 0.5,
                                                                  "compression_schedule": ["decode"]}})
        with self.assertRaises(ValueError):
            build_compressor(decode_only)

    def test_segment_mode_resolution(self):
        from eval_harness.kv_recovery.config import StudentCfg
        self.assertEqual(resolve_segment_mode(self.model, StudentCfg(segment_mode="auto")), "block")
        self.assertEqual(resolve_segment_mode(self.model, StudentCfg(segment_mode="token_by_token")), "token_by_token")


class TestSegmentForward(unittest.TestCase):
    def setUp(self):
        self.model = _tiny_llama()
        self.adapter = _shell(self.model)
        self.ex = _example(T=40, L=8)

    def test_block_equals_full_sequence_and_token_by_token(self):
        probe = probe_block_continuation(self.model, self.adapter._cache_adapter, T=40, L=8, rtol=1e-4)
        self.assertTrue(probe["block_ok"], probe)
        self.assertTrue(probe["token_by_token_ok"], probe)

    def test_positions_are_absolute_and_cache_position_absent(self):
        seen = {}
        orig = self.model.forward

        def spy(*args, **kwargs):
            seen.update({k: v for k, v in kwargs.items() if k in ("position_ids", "cache_position", "logits_to_keep")})
            return orig(*args, **kwargs)

        self.model.forward = spy
        try:
            cache = prefill_context(self.adapter, self.ex.ctx_ids, KnormSketch(compression_ratio=0.5))
            logits = segment_forward(self.model, cache, self.ex.suffix_ids, 40, logits_to_keep=3)
        finally:
            del self.model.forward
        self.assertEqual(seen["position_ids"].tolist(), [list(range(40, 48))])
        self.assertNotIn("cache_position", seen)
        self.assertEqual(tuple(logits.shape), (1, 3, 256))

    def test_token_by_token_logits_match_block(self):
        cache_a = prefill_context(self.adapter, self.ex.ctx_ids, None)
        cache_b = prefill_context(self.adapter, self.ex.ctx_ids, None)
        with torch.no_grad():
            a = segment_forward(self.model, cache_a, self.ex.suffix_ids, 40, logits_to_keep=0, mode="block")
            b = segment_forward(self.model, cache_b, self.ex.suffix_ids, 40, logits_to_keep=0, mode="token_by_token")
        self.assertEqual(tuple(a.shape), (1, 8, 256))
        self.assertTrue(torch.allclose(a, b, atol=1e-4), float((a - b).abs().max()))


class TestTeacherStudent(unittest.TestCase):
    def setUp(self):
        self.teacher_model = _tiny_llama()
        self.student_model = copy.deepcopy(self.teacher_model)
        self.teacher = _shell(self.teacher_model)
        self.student = _shell(self.student_model)
        self.spec = inspect_model(self.student_model)
        self.ex = _example(T=40, L=8)
        self.keys = [0, 1]

    def test_same_model_without_compression_is_bitwise_identical(self):
        t = run_teacher(self.teacher, self.ex, self.keys, include_final_norm=True, want_logits=True)
        s = run_student(self.student, self.ex, None, self.keys, include_final_norm=True, grad=False,
                        want_logits=True, spec=self.spec)
        for k in [0, 1, FINAL_NORM_KEY]:
            self.assertTrue(torch.equal(t.states[k], s.states[k]), k)
            self.assertFalse(t.states[k].requires_grad)
        self.assertTrue(torch.equal(t.logits, s.logits))
        pos = position_index(PositionsCfg("all"), 8)
        loss, _, _ = hidden_loss(gather_positions(s.states, pos), gather_positions(t.states, pos), "normalized_mse",
                                 [0, 1, FINAL_NORM_KEY])
        self.assertEqual(float(loss), 0.0)
        self.assertEqual(t.cache_len_after_prefill, 40)
        self.assertEqual(t.cache_len_after_segment, 48)

    def test_compression_increases_divergence_and_student_keeps_grad(self):
        comp = KnormSketch(compression_ratio=0.5)
        p = self.student_model.model.layers[1].self_attn.o_proj.weight
        p.requires_grad_(True)
        t = run_teacher(self.teacher, self.ex, self.keys, include_final_norm=True, want_logits=False)
        s = run_student(self.student, self.ex, comp, self.keys, include_final_norm=True, grad=True,
                        want_logits=False, spec=self.spec, compression_ratio=0.5)
        self.assertEqual(s.cache_len_after_prefill, 20)
        self.assertEqual(s.per_layer_cache_len, {0: 28, 1: 28})
        self.assertTrue(s.states[1].requires_grad)
        self.assertTrue(s.states[FINAL_NORM_KEY].requires_grad)
        self.assertFalse(s.states[0].requires_grad)     # upstream of the trainable parameter
        pos = position_index(PositionsCfg("all"), 8)
        loss, per_layer, _ = hidden_loss(gather_positions(s.states, pos), gather_positions(t.states, pos),
                                         "normalized_mse", [0, 1, FINAL_NORM_KEY])
        self.assertGreater(float(loss.detach()), 0.0)
        self.assertGreater(per_layer["1"], 0.0)
        loss.backward()
        self.assertIsNotNone(p.grad)
        self.assertTrue(torch.isfinite(p.grad).all())
        stray = [n for n, q in self.student_model.named_parameters() if q.grad is not None and q is not p]
        self.assertEqual(stray, [])
        # the teacher never saw a hook and keeps a full cache
        assert_no_hooks(self.teacher_model)
        self.assertEqual(t.cache_len_after_prefill, 40)

    def test_more_compression_more_divergence(self):
        t = run_teacher(self.teacher, self.ex, self.keys, include_final_norm=False, want_logits=False)
        pos = position_index(PositionsCfg("all"), 8)
        losses = []
        for r in (0.25, 0.75):
            s = run_student(self.student, self.ex, KnormSketch(compression_ratio=r), self.keys,
                            include_final_norm=False, grad=False, want_logits=False, spec=self.spec, compression_ratio=r)
            loss, _, _ = hidden_loss(gather_positions(s.states, pos), gather_positions(t.states, pos), "cosine", self.keys)
            losses.append(float(loss))
        self.assertGreater(losses[1], losses[0])

    def test_prefill_grad_flag_lets_gradient_reach_the_write_path(self):
        comp = KnormSketch(compression_ratio=0.5)
        k = self.student_model.model.layers[1].self_attn.k_proj.weight
        k.requires_grad_(True)
        t = run_teacher(self.teacher, self.ex, [1], include_final_norm=False, want_logits=False)
        pos = position_index(PositionsCfg("all"), 8)
        grads = []
        for prefill_grad in (False, True):
            k.grad = None
            s = run_student(self.student, self.ex, comp, [1], include_final_norm=False, grad=True, want_logits=False,
                            prefill_grad=prefill_grad, spec=self.spec, compression_ratio=0.5)
            loss, _, _ = hidden_loss(gather_positions(s.states, pos), gather_positions(t.states, pos), "cosine", [1])
            loss.backward()
            grads.append(k.grad.detach().clone())
        self.assertFalse(torch.equal(grads[0], grads[1]))   # the context K/V contribute only with prefill_grad


@unittest.skipUnless(HAS_QWEN35, "transformers build lacks Qwen3.5")
class TestTinyQwen35Continuation(unittest.TestCase):
    def test_block_continuation_matches_full_forward(self):
        cfg = Qwen3_5TextConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
                                num_key_value_heads=2, head_dim=16, vocab_size=512, max_position_embeddings=512,
                                layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
                                linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=16,
                                linear_value_head_dim=16, linear_conv_kernel_dim=4, tie_word_embeddings=True)
        cfg._attn_implementation = "eager"
        torch.manual_seed(0)
        model = Qwen3_5ForCausalLM(cfg).eval()
        model.requires_grad_(False)
        probe = probe_block_continuation(model, create_cache_adapter(model), T=24, L=6, rtol=1e-3)
        self.assertTrue(probe["block_ok"], probe)
        self.assertTrue(probe["token_by_token_ok"], probe)
        from eval_harness.kv_recovery.config import StudentCfg
        self.assertEqual(resolve_segment_mode(model, StudentCfg()), "block")


if __name__ == "__main__":
    unittest.main()
