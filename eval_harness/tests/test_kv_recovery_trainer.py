"""Training loop on a tiny Llama (CPU): loss decreases, frozen weights untouched, checks, offline store."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from eval_harness.kv_compression import KnormSketch
from eval_harness.kv_compression.cache_adapter import create_cache_adapter
from eval_harness.kv_recovery.alignment import alignment_keys_for
from eval_harness.kv_recovery.config import RecoveryConfig
from eval_harness.kv_recovery.model_spec import inspect_model
from eval_harness.kv_recovery.student import Example
from eval_harness.kv_recovery.trainable import (
    assert_trainable,
    changed_parameters,
    first_trainable_layer,
    freeze_all_but,
    select_trainable,
    snapshot_parameters,
    trainable_parameters,
)
from eval_harness.kv_recovery.trainer import (
    AlignmentSetup,
    TeacherStore,
    TrainingUnstable,
    check_same_model,
    teacher_digest,
    teacher_states_for,
    train,
    weight_update_norms,
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


def _tiny_llama(dtype=torch.float32):
    cfg = LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, vocab_size=256, max_position_embeddings=8192, rope_theta=10000.0,
                      attn_implementation="eager")
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(4.0 if p.dim() > 1 else 1.0)
    model.requires_grad_(False)
    model.generation_config.eos_token_id = 2
    return model.to(dtype)


def _shell(model):
    adapter = object.__new__(ResearchAdapter)
    adapter._model = model
    adapter._tokenizer = _StubTokenizer()
    pipe = object.__new__(ResearchGenerationPipeline)
    pipe.model = model
    pipe.tokenizer = adapter._tokenizer
    adapter._pipe = pipe
    adapter._cache_adapter = create_cache_adapter(model)
    return adapter


def _examples(n, T=40, L=8, seed=0):
    out = []
    for i in range(n):
        g = torch.Generator().manual_seed(seed + i)
        out.append(Example(id=f"ex{seed}_{i}", ctx_ids=torch.randint(0, 256, (1, T), generator=g),
                           suffix_ids=torch.randint(0, 256, (1, L), generator=g)))
    return out


def _setup(cfg, model, names):
    spec = inspect_model(model)
    keys = alignment_keys_for(cfg.alignment, spec.n_layers, first_trainable_layer=first_trainable_layer(names, spec))
    return AlignmentSetup(keys=keys, layer_indices=[k for k in keys if isinstance(k, int)],
                          include_final_norm=cfg.alignment.include_final_norm, positions_cfg=cfg.alignment.positions,
                          loss_name=cfg.alignment.loss, layer_weights=None, loss_cfg=cfg.loss,
                          want_logits=cfg.loss.kl_weight > 0, mode="block", spec=spec,
                          compression_ratio=cfg.kv_compression.compression_ratio, prefill_chunk_size=None,
                          prefill_grad=cfg.student.prefill_grad)


BASE = {"kv_compression": {"kv_compressor": "knorm", "compression_ratio": 0.5},
        "trainable": {"strategy": "last_n_blocks", "n": 1},
        "alignment": {"layers": {"strategy": "from_first_trainable"}, "include_final_norm": True},
        "optim": {"learning_rate": 1e-3, "grad_accum": 2, "epochs": 1, "val_every_steps": 1, "master_weights_fp32": True},
        "data": {"max_length": 48, "suffix_length": 8}}


class TestTrainLoop(unittest.TestCase):
    def setUp(self):
        self.teacher_model = _tiny_llama()
        self.student_model = copy.deepcopy(self.teacher_model)
        self.teacher, self.student = _shell(self.teacher_model), _shell(self.student_model)
        self.cfg = RecoveryConfig.from_dict(BASE)
        self.spec = inspect_model(self.student_model)
        self.names = select_trainable(self.student_model, self.spec, self.cfg.trainable)
        self.expected = freeze_all_but(self.student_model, self.names)
        self.trainable = trainable_parameters(self.student_model, self.names)
        self.comp = KnormSketch(compression_ratio=0.5)
        self.setup = _setup(self.cfg, self.student_model, self.names)
        self.train_ex, self.val_ex = _examples(6, seed=0), _examples(2, seed=100)
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_same_model_check_passes_before_training(self):
        res = check_same_model(self.teacher, self.student, self.train_ex[:2], self.setup)
        self.assertTrue(res["passed"], res)
        self.assertTrue(all(r["bitwise_equal"] for r in res["examples"]))

    def test_loss_decreases_and_only_trainable_change(self):
        # Optimisation sanity (spec §22): on a FIXED batch of two windows the alignment loss
        # decreases over a few steps (validation = the same windows under new ids).
        cfg = RecoveryConfig.from_dict({**BASE, "optim": {**BASE["optim"], "grad_accum": 1, "epochs": 4,
                                                         "val_every_steps": 8}})
        fixed = self.train_ex[:2]
        val = [Example(id="v" + e.id, ctx_ids=e.ctx_ids, suffix_ids=e.suffix_ids) for e in fixed]
        snap_all = snapshot_parameters(self.student_model)
        log_path = Path(self.tmp.name) / "train_metrics.jsonl"
        state = train(self.teacher, self.student, self.comp, self.trainable, fixed, val, cfg,
                      self.setup, log_path=log_path, print_fn=lambda *a, **k: None)
        self.assertEqual(state.optimizer_steps, 8)
        self.assertTrue(state.check3["passed"], state.check3)
        self.assertEqual(state.check3["zero_grad_trainable"], [])
        self.assertLess(state.step_logs[-1]["loss"], state.step_logs[0]["loss"])
        self.assertLess(state.val_logs[-1]["val_loss"], state.val_logs[0]["val_loss"])
        rows = [json.loads(l) for l in log_path.read_text().splitlines()]
        self.assertEqual(len(rows), 8)
        self.assertEqual(set(rows[0]) >= {"step", "lr", "loss", "hidden_loss", "kl_loss", "per_layer", "per_bucket",
                                          "grad_norm_pre_clip", "bf16_elements_changed", "examples_seen"}, True)
        # frozen-weight sanity: nothing outside the trainable subset moved; trainable did
        changed = changed_parameters(self.student_model, snap_all)
        self.assertTrue(set(changed) <= set(self.names), set(changed) - set(self.names))
        self.assertTrue(changed)
        assert_trainable(self.student_model, self.expected, self.spec)
        # masters were copied back: parameters equal masters rounded to the param dtype
        for n in self.names:
            self.assertTrue(torch.equal(self.trainable[n].detach(), state.masters[n].to(self.trainable[n].dtype)))
        rows_norm = weight_update_norms(snapshot_parameters(self.student_model, self.names), self.trainable, state.masters,
                                        state.optimizer_steps, cfg.optim.learning_rate, self.spec)
        self.assertEqual({r["parameter"] for r in rows_norm}, set(self.names))

    def test_max_steps_and_bf16_masters(self):
        cfg = RecoveryConfig.from_dict({**BASE, "optim": {**BASE["optim"], "max_steps": 1}})
        tm, sm = _tiny_llama(torch.bfloat16), None
        sm = copy.deepcopy(tm)
        teacher, student = _shell(tm), _shell(sm)
        names = select_trainable(sm, inspect_model(sm), cfg.trainable)
        freeze_all_but(sm, names)
        state = train(teacher, student, KnormSketch(compression_ratio=0.5), trainable_parameters(sm, names),
                      _examples(4), _examples(1, seed=50), cfg, _setup(cfg, sm, names),
                      log_path=Path(self.tmp.name) / "m.jsonl", print_fn=lambda *a, **k: None)
        self.assertEqual(state.optimizer_steps, 1)
        self.assertEqual(next(iter(state.masters.values())).dtype, torch.float32)
        self.assertGreater(state.step_logs[0]["bf16_elements_changed"], 0)

    def test_instability_rule_raises(self):
        cfg = RecoveryConfig.from_dict({**BASE, "optim": {**BASE["optim"], "grad_accum": 1, "learning_rate": 1e-3,
                                                         "instability": {"grad_norm_factor": 1e-9, "consecutive_steps": 1}}})
        with self.assertRaises(TrainingUnstable):
            train(self.teacher, self.student, self.comp, self.trainable, self.train_ex, self.val_ex, cfg, self.setup,
                  log_path=Path(self.tmp.name) / "u.jsonl", print_fn=lambda *a, **k: None)

    def test_offline_teacher_store_round_trip(self):
        store_dir = Path(self.tmp.name) / "states"
        store_dir.mkdir()
        for ex in self.train_ex + self.val_ex:
            TeacherStore.save(store_dir, ex.id, teacher_states_for(self.teacher, ex, self.setup), dtype=torch.float32)
        (store_dir / "manifest.json").write_text(json.dumps({"teacher_digest": teacher_digest(self.cfg),
                                                             "keys": [str(k) for k in self.setup.keys]}))
        store = TeacherStore(store_dir)
        store.check(self.cfg, self.setup.keys)
        ts_on = teacher_states_for(self.teacher, self.train_ex[0], self.setup)
        ts_off = teacher_states_for(None, self.train_ex[0], self.setup, store=store)
        for k in self.setup.keys:
            self.assertTrue(torch.equal(ts_on.gathered[k], ts_off.gathered[k]))
        state = train(None, self.student, self.comp, self.trainable, self.train_ex, self.val_ex, self.cfg, self.setup,
                      log_path=Path(self.tmp.name) / "o.jsonl", store=store, print_fn=lambda *a, **k: None)
        self.assertEqual(state.optimizer_steps, 3)
        other = RecoveryConfig.from_dict({**BASE, "data": {**BASE["data"], "suffix_length": 4}})
        with self.assertRaises(ValueError):
            store.check(other, self.setup.keys)

    def test_kl_term_trains(self):
        cfg = RecoveryConfig.from_dict({**BASE, "loss": {"hidden_weight": 1.0, "kl_weight": 0.5, "temperature": 2.0}})
        setup = _setup(cfg, self.student_model, self.names)
        state = train(self.teacher, self.student, self.comp, self.trainable, self.train_ex[:2], self.val_ex[:1], cfg,
                      setup, log_path=Path(self.tmp.name) / "k.jsonl", print_fn=lambda *a, **k: None)
        self.assertGreater(state.step_logs[0]["kl_loss"], 0.0)


@unittest.skipUnless(HAS_QWEN35, "transformers build lacks Qwen3.5")
class TestHybridBackwardThroughLinearAttention(unittest.TestCase):
    """Regression: gradients must flow THROUGH a Qwen3.5 linear-attention block that sits between a
    trainable parameter and the aligned layer (the cache layer's in-place state update used to
    invalidate the kernel's saved initial state)."""

    def _tiny(self):
        cfg = Qwen3_5TextConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
                                num_key_value_heads=2, head_dim=16, vocab_size=512, max_position_embeddings=512,
                                layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
                                linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=16,
                                linear_value_head_dim=16, linear_conv_kernel_dim=4, tie_word_embeddings=True)
        cfg._attn_implementation = "eager"
        torch.manual_seed(0)
        m = Qwen3_5ForCausalLM(cfg).eval()
        with torch.no_grad():
            for p in m.parameters():
                p.mul_(4.0 if p.dim() > 1 else 1.0)
        m.requires_grad_(False)
        m.generation_config.eos_token_id = 2
        return m

    def test_last_two_blocks_train_one_step(self):
        tm = self._tiny(); sm = copy.deepcopy(tm)
        teacher, student = _shell(tm), _shell(sm)
        cfg = RecoveryConfig.from_dict({**BASE, "trainable": {"strategy": "last_n_blocks", "n": 2},
                                        "optim": {**BASE["optim"], "grad_accum": 1, "max_steps": 2}})
        names = select_trainable(sm, inspect_model(sm), cfg.trainable)
        self.assertTrue(any(".linear_attn." in n for n in names))
        freeze_all_but(sm, names)
        g = torch.Generator().manual_seed(3)
        ex = [Example(id=f"q{i}", ctx_ids=torch.randint(0, 512, (1, 40), generator=g),
                      suffix_ids=torch.randint(0, 512, (1, 8), generator=g)) for i in range(2)]
        state = train(teacher, student, KnormSketch(compression_ratio=0.5), trainable_parameters(sm, names), ex, ex[:1],
                      cfg, _setup(cfg, sm, names), log_path=Path(tempfile.mkdtemp()) / "q.jsonl", print_fn=lambda *a, **k: None)
        self.assertEqual(state.optimizer_steps, 2)
        self.assertTrue(state.check3["passed"], state.check3)
        self.assertEqual(state.check3["zero_grad_trainable"], [])


if __name__ == "__main__":
    unittest.main()
