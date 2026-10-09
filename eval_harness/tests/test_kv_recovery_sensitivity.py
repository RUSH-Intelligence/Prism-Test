"""Compression-sensitivity layer selection (``trainable.layers: sensitivity``): the E_l formula,
aggregation / ranking / top-k, the measurement through the real teacher/student path on tiny
config-built models (CPU, fp32, eager), the calibration split, and an end-to-end training step."""
from __future__ import annotations

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from eval_harness.kv_compression import KnormSketch
from eval_harness.kv_compression.cache_adapter import create_cache_adapter
from eval_harness.kv_recovery.alignment import alignment_keys_for
from eval_harness.kv_recovery.config import DataCfg, RecoveryConfig, TrainableCfg
from eval_harness.kv_recovery.data import assert_disjoint, build_examples, load_split
from eval_harness.kv_recovery.model_spec import inspect_model
from eval_harness.kv_recovery.sensitivity import (
    SELECTOR,
    SensitivityReport,
    aggregate_scores,
    candidate_layers,
    layer_sensitivity,
    measure_layer_sensitivity,
    rank_layers,
    select_layers_by_sensitivity,
    select_top_k,
)
from eval_harness.kv_recovery.student import Example
from eval_harness.kv_recovery.trainable import (
    first_trainable_layer,
    freeze_all_but,
    select_layers,
    select_trainable,
    snapshot_parameters,
    changed_parameters,
    trainable_parameters,
)
from eval_harness.kv_recovery.trainer import AlignmentSetup, train
from eval_harness.research_adapter import ResearchAdapter
from eval_harness.research_pipeline import ResearchGenerationPipeline

try:
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    HAS_QWEN35 = True
except Exception:  # pragma: no cover
    HAS_QWEN35 = False


# ---------------------------------------------------------------------------
# helpers (the repo's object.__new__ shell idiom, tiny models, synthetic windows)
# ---------------------------------------------------------------------------
class _StubTokenizer:
    model_max_length = 8192
    bos_token = None

    def decode(self, ids, skip_special_tokens=True):  # noqa: ARG002
        return "x" * len(ids)


class _WordTokenizer:
    bos_token_id = 1

    def encode(self, text, add_special_tokens=False):  # noqa: ARG002
        return [2 + (hash(w) % 1000) for w in text.split()]


def _tiny_llama(num_hidden_layers=3):
    cfg = LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=num_hidden_layers, num_attention_heads=4,
                      num_key_value_heads=2, vocab_size=256, max_position_embeddings=8192, rope_theta=10000.0,
                      attn_implementation="eager")
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(4.0 if p.dim() > 1 else 1.0)
    model.requires_grad_(False)
    model.generation_config.eos_token_id = 2
    return model


def _tiny_qwen35():
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


def _examples(n, T=40, L=8, seed=0, vocab=256):
    out = []
    for i in range(n):
        g = torch.Generator().manual_seed(seed + i)
        out.append(Example(id=f"c{seed}_{i}", ctx_ids=torch.randint(0, vocab, (1, T), generator=g),
                           suffix_ids=torch.randint(0, vocab, (1, L), generator=g)))
    return out


def _cfg(**over):
    base = {"kv_compression": {"kv_compressor": "knorm", "compression_ratio": 0.5},
            "trainable": {"strategy": "attention_projections", "modules": ["q_proj", "o_proj"], "layers": "sensitivity",
                          "sensitivity": {"top_k": 1, "num_examples": 2}},
            "alignment": {"layers": {"strategy": "from_first_trainable"}, "include_final_norm": True},
            "optim": {"learning_rate": 1e-3, "grad_accum": 1, "epochs": 1, "val_every_steps": 1, "max_steps": 2},
            "data": {"max_length": 48, "suffix_length": 8}}
    for k, v in over.items():
        base[k] = {**base.get(k, {}), **v} if isinstance(v, dict) else v
    return RecoveryConfig.from_dict(base)


# ---------------------------------------------------------------------------
# the formula
# ---------------------------------------------------------------------------
class TestFormula(unittest.TestCase):
    def test_identical_states_give_exactly_zero(self):
        t = {0: torch.randn(5, 7), 1: torch.randn(5, 7), "norm": torch.randn(5, 7)}
        e = layer_sensitivity(t, {k: v.clone() for k, v in t.items()}, eps=1e-6)
        self.assertEqual(e, {0: 0.0, 1: 0.0})           # the final norm is not a layer

    def test_matches_the_definition(self):
        g = torch.Generator().manual_seed(0)
        t = {0: torch.randn(6, 4, generator=g), 3: torch.randn(6, 4, generator=g)}
        s = {0: t[0] * 1.5, 3: t[3] + 1.0}
        e = layer_sensitivity(t, s, eps=0.0)
        self.assertAlmostEqual(e[0], 0.5, places=6)        # ||0.5 T|| / ||T||
        ref3 = float(torch.linalg.norm(t[3] - s[3]) / torch.linalg.norm(t[3]))
        self.assertAlmostEqual(e[3], ref3, places=6)
        # eps only matters for a vanishing teacher norm, where it keeps the ratio finite
        zero_t = {0: torch.zeros(3, 2)}
        self.assertAlmostEqual(layer_sensitivity(zero_t, {0: torch.ones(3, 2)}, eps=1.0)[0], math.sqrt(6.0), places=5)
        self.assertTrue(math.isinf(layer_sensitivity(zero_t, {0: torch.ones(3, 2)}, eps=0.0)[0]))

    def test_scale_invariant_and_fp32(self):
        g = torch.Generator().manual_seed(1)
        t = {2: torch.randn(8, 16, generator=g)}
        s = {2: t[2] + 0.1 * torch.randn(8, 16, generator=g)}
        a = layer_sensitivity(t, s, eps=0.0)[2]
        b = layer_sensitivity({2: t[2] * 1000}, {2: s[2] * 1000}, eps=0.0)[2]
        self.assertAlmostEqual(a, b, places=5)
        c = layer_sensitivity({2: t[2].to(torch.bfloat16)}, {2: s[2].to(torch.bfloat16)}, eps=0.0)[2]
        self.assertAlmostEqual(a, c, places=2)
        with self.assertRaises(ValueError):
            layer_sensitivity(t, {2: torch.zeros(7, 16)}, eps=0.0)


class TestAggregateRankSelect(unittest.TestCase):
    PER_EX = {"a": {0: 0.10, 1: 0.30, 2: 0.20, 3: 0.30}, "b": {0: 0.20, 1: 0.10, 2: 0.60, 3: 0.30}}

    def test_mean_median_std(self):
        mean, std = aggregate_scores(self.PER_EX, "mean")
        self.assertEqual(sorted(mean), [0, 1, 2, 3])
        for l, v in {0: 0.15, 1: 0.2, 2: 0.4, 3: 0.3}.items():
            self.assertAlmostEqual(mean[l], v, places=12)
        self.assertAlmostEqual(std[2], 0.2)
        self.assertEqual(std[3], 0.0)
        median, _ = aggregate_scores(self.PER_EX, "median")
        self.assertEqual(median[2], 0.4)
        with self.assertRaises(ValueError):
            aggregate_scores(self.PER_EX, "max")
        with self.assertRaises(ValueError):
            aggregate_scores({}, "mean")
        with self.assertRaises(ValueError):
            aggregate_scores({"a": {0: 1.0}, "b": {1: 1.0}}, "mean")

    def test_ranking_descending_with_deeper_tie_break(self):
        self.assertEqual(rank_layers({0: 0.15, 1: 0.2, 2: 0.4, 3: 0.3}), [2, 3, 1, 0])
        self.assertEqual(rank_layers({0: 0.5, 1: 0.5, 2: 0.1}), [1, 0, 2])      # tie -> deeper layer first
        with self.assertRaises(ValueError):
            rank_layers({0: float("nan"), 1: 0.1})

    def test_top_k_among_candidates(self):
        ranking = [2, 3, 1, 0]
        self.assertEqual(select_top_k(ranking, [0, 1, 2, 3], 2), [2, 3])
        self.assertEqual(select_top_k(ranking, [0, 1], 1), [1])                 # ineligible layers are skipped
        self.assertEqual(select_top_k(ranking, [3], 5), [3])                    # k capped at the candidate count
        self.assertEqual(select_top_k(ranking, [0, 1, 2, 3], 3), [1, 2, 3])     # ascending layer order
        with self.assertRaises(ValueError):
            select_top_k(ranking, [], 1)
        with self.assertRaises(ValueError):
            select_top_k(ranking, [0], 0)
        with self.assertRaises(ValueError):
            select_top_k(ranking, [7], 1)                                        # unmeasured candidate

    def test_candidate_pools(self):
        spec = inspect_model(_tiny_llama(3))
        self.assertEqual(candidate_layers("attention_projections", spec), [0, 1, 2])
        self.assertEqual(candidate_layers("blocks", spec), [0, 1, 2])
        self.assertEqual(candidate_layers("mlp", spec), [0, 1, 2])
        with self.assertRaises(ValueError):
            candidate_layers("last_n_blocks", spec)

    def test_report_serialises(self):
        rep = SensitivityReport(method=SELECTOR, layers=[0, 1], per_example={"a": {0: 0.1, 1: 0.2}}, aggregate="mean",
                                scores={0: 0.1, 1: 0.2}, std={0: 0.0, 1: 0.0}, ranking=[1, 0], candidates=[0, 1],
                                selected=[1], top_k=1, eps=1e-6, positions={"strategy": "all", "n": 128},
                                calibration_ids=["a"], n_examples=1, compressor="knorm", compression_ratio=0.5,
                                hooked_layers=[0, 1], seconds=0.1)
        d = json.loads(json.dumps(rep.to_dict()))
        self.assertEqual(d["selected"], [1])
        self.assertEqual(d["scores"], {"0": 0.1, "1": 0.2})
        self.assertIn("selected (top-1 of 2 eligible): [1]", rep.table())
        self.assertEqual([r["rank"] for r in rep.rows()], [2, 1])


class TestTrainableIntegration(unittest.TestCase):
    def test_selector_must_be_resolved(self):
        model = _tiny_llama(3)
        spec = inspect_model(model)
        tcfg = TrainableCfg(strategy="attention_projections", modules=["q_proj", "o_proj"], layers="sensitivity")
        with self.assertRaises(ValueError):
            select_layers("sensitivity", [0, 1, 2])
        with self.assertRaises(ValueError):
            select_trainable(model, spec, tcfg)
        names = select_trainable(model, spec, tcfg, resolved_layers=[2, 0])
        self.assertEqual(names, sorted(f"model.layers.{i}.self_attn.{m}.weight" for i in (0, 2) for m in ("o_proj", "q_proj")))
        self.assertEqual(first_trainable_layer(names, spec), 0)
        blocks = select_trainable(model, spec, TrainableCfg(strategy="blocks", layers="sensitivity"), resolved_layers=[1])
        self.assertTrue(blocks and all(n.startswith("model.layers.1.") for n in blocks))
        self.assertIn("model.layers.1.mlp.down_proj.weight", blocks)
        # resolved_layers is ignored for static selectors
        static = select_trainable(model, spec, TrainableCfg(strategy="attention_projections", modules=["q_proj"], layers="last_n:1"),
                                  resolved_layers=[0])
        self.assertEqual(static, ["model.layers.2.self_attn.q_proj.weight"])
        with self.assertRaises(ValueError):
            select_trainable(model, spec, tcfg, resolved_layers=[])

    def test_blocks_strategy_static_selectors(self):
        model = _tiny_llama(3)
        spec = inspect_model(model)
        last = select_trainable(model, spec, TrainableCfg(strategy="blocks", layers="last_n:2"))
        self.assertEqual(last, select_trainable(model, spec, TrainableCfg(strategy="last_n_blocks", n=2)))
        explicit = select_trainable(model, spec, TrainableCfg(strategy="blocks", layers=[0]))
        self.assertTrue(all(n.startswith("model.layers.0.") for n in explicit))


class TestConfigRules(unittest.TestCase):
    def test_validation(self):
        ok = _cfg()
        self.assertEqual(ok.trainable.layers, "sensitivity")
        self.assertEqual(ok.trainable.sensitivity.top_k, 1)
        bad = [
            {"trainable": {"strategy": "last_n_blocks", "layers": "sensitivity"}},
            {"trainable": {"strategy": "full", "layers": "sensitivity"}},
            {"trainable": {"strategy": "attention_projections", "layers": "sensitivity", "sensitivity": {"top_k": 0}}},
            {"trainable": {"strategy": "attention_projections", "layers": "sensitivity", "sensitivity": {"num_examples": 0}}},
            {"trainable": {"strategy": "attention_projections", "layers": "sensitivity", "sensitivity": {"split": "test"}}},
            {"trainable": {"strategy": "attention_projections", "layers": "sensitivity", "sensitivity": {"aggregate": "max"}}},
            {"trainable": {"strategy": "attention_projections", "layers": "sensitivity", "sensitivity": {"eps": -1.0}}},
            {"trainable": {"strategy": "attention_projections", "layers": "sensitivity",
                           "sensitivity": {"positions": {"strategy": "first_k", "n": 0}}}},
            {"trainable": {"strategy": "attention_projections", "layers": "sensitivity"}, "data": {"val_path": None}},
            {"trainable": {"strategy": "attention_projections", "layers": "sensitivity"},
             "teacher": {"mode": "offline", "states_dir": "x"}},
            {"trainable": {"sensitivity": {"top_kk": 3}}},
        ]
        for d in bad:
            with self.subTest(d=d):
                with self.assertRaises((ValueError, KeyError)):
                    RecoveryConfig.from_dict(d)
        # static selectors never look at the sensitivity block, so an odd block is harmless there
        RecoveryConfig.from_dict({"trainable": {"strategy": "last_n_blocks", "layers": "all", "sensitivity": {"top_k": 99}}})
        # round trip keeps the nested block
        again = RecoveryConfig.from_dict(ok.to_dict())
        self.assertEqual(again.to_dict(), ok.to_dict())
        self.assertNotEqual(_cfg().digest(), _cfg(trainable={"sensitivity": {"top_k": 2, "num_examples": 2}}).digest())

    def test_dotted_overrides_reach_the_nested_block(self):
        from eval_harness.kv_recovery.config import apply_overrides
        d = apply_overrides({}, overrides=["trainable.layers=sensitivity", "trainable.sensitivity.top_k=3",
                                           "trainable.strategy=attention_projections"])
        cfg = RecoveryConfig.from_dict(d)
        self.assertEqual(cfg.trainable.sensitivity.top_k, 3)


# ---------------------------------------------------------------------------
# measurement on tiny models through the real prefill / segment path
# ---------------------------------------------------------------------------
class TestTinyLlamaMeasurement(unittest.TestCase):
    def setUp(self):
        self.teacher_model = _tiny_llama(3)
        self.student_model = copy.deepcopy(self.teacher_model)
        self.teacher, self.student = _shell(self.teacher_model), _shell(self.student_model)
        self.spec = inspect_model(self.student_model)
        self.calib = _examples(2, seed=300)

    def test_compressed_positive_dense_zero_deterministic(self):
        comp = KnormSketch(compression_ratio=0.5)
        per_ex = measure_layer_sensitivity(self.teacher, self.student, comp, self.calib, spec=self.spec, compression_ratio=0.5)
        self.assertEqual(set(per_ex), {e.id for e in self.calib})
        for scores in per_ex.values():
            self.assertEqual(sorted(scores), [0, 1, 2])
            self.assertTrue(all(math.isfinite(v) and v > 0.0 for v in scores.values()), scores)
        again = measure_layer_sensitivity(self.teacher, self.student, comp, self.calib, spec=self.spec, compression_ratio=0.5)
        self.assertEqual(per_ex, again)
        zero = measure_layer_sensitivity(self.teacher, self.student, None, self.calib, spec=self.spec)
        self.assertTrue(all(v == 0.0 for sc in zero.values() for v in sc.values()), zero)
        # one model instance can serve both passes (the standalone script does this)
        same = measure_layer_sensitivity(self.student, self.student, comp, self.calib, spec=self.spec, compression_ratio=0.5)
        self.assertEqual(same, per_ex)
        # no hooks left behind, weights untouched
        from eval_harness.kv_recovery.student import assert_no_hooks
        assert_no_hooks(self.student_model); assert_no_hooks(self.teacher_model)
        tp = dict(self.teacher_model.named_parameters())
        self.assertTrue(all(torch.equal(p, tp[n]) for n, p in self.student_model.named_parameters()))

    def test_more_compression_more_sensitivity(self):
        agg = {}
        for r in (0.25, 0.75):
            per_ex = measure_layer_sensitivity(self.teacher, self.student, KnormSketch(compression_ratio=r), self.calib,
                                               spec=self.spec, compression_ratio=r)
            scores, _ = aggregate_scores(per_ex, "mean")
            agg[r] = sum(scores.values()) / len(scores)
        self.assertGreater(agg[0.75], agg[0.25])

    def test_positions_subset_and_selection_report(self):
        cfg = _cfg(trainable={"sensitivity": {"top_k": 2, "num_examples": 2, "positions": {"strategy": "first_k", "n": 3}}})
        comp = KnormSketch(compression_ratio=0.5)
        rep = select_layers_by_sensitivity(cfg, self.teacher, self.student, comp, self.calib, spec=self.spec)
        self.assertEqual(rep.candidates, [0, 1, 2])
        self.assertEqual(len(rep.selected), 2)
        self.assertEqual(rep.selected, sorted(rep.ranking[:2]))
        self.assertEqual(rep.ranking, rank_layers(rep.scores))
        self.assertEqual(rep.n_examples, 2)
        self.assertEqual(rep.positions, {"strategy": "first_k", "n": 3})
        self.assertEqual(rep.calibration_ids, [e.id for e in self.calib])
        # positions matter: the full-suffix measurement differs from the first-3 one
        full = select_layers_by_sensitivity(_cfg(trainable={"sensitivity": {"top_k": 2, "num_examples": 2}}), self.teacher,
                                            self.student, comp, self.calib, spec=self.spec)
        self.assertNotEqual(full.scores, rep.scores)
        # the per-example numbers aggregate to the reported scores
        mean, _ = aggregate_scores(rep.per_example, "mean")
        self.assertEqual(mean, rep.scores)
        with self.assertRaises(ValueError):
            select_layers_by_sensitivity(cfg, self.teacher, self.student, None, self.calib, spec=self.spec)
        with self.assertRaises(ValueError):
            select_layers_by_sensitivity(cfg, self.teacher, self.student, comp, [], spec=self.spec)

    def test_top_k_beyond_candidates_is_noted(self):
        cfg = _cfg(trainable={"sensitivity": {"top_k": 9, "num_examples": 2}})
        rep = select_layers_by_sensitivity(cfg, self.teacher, self.student, KnormSketch(compression_ratio=0.5), self.calib,
                                           spec=self.spec)
        self.assertEqual(rep.selected, [0, 1, 2])
        self.assertTrue(any("equals 'all'" in n for n in rep.notes), rep.notes)


@unittest.skipUnless(HAS_QWEN35, "transformers build lacks Qwen3.5")
class TestTinyQwen35Hybrid(unittest.TestCase):
    def test_linear_layers_before_the_first_kv_layer_have_zero_sensitivity(self):
        tm = _tiny_qwen35(); sm = copy.deepcopy(tm)
        teacher, student = _shell(tm), _shell(sm)
        spec = inspect_model(sm)
        self.assertEqual(spec.full_attention_layers, (3,))
        calib = _examples(2, seed=7, vocab=512)
        cfg = _cfg(trainable={"sensitivity": {"top_k": 1, "num_examples": 2}})
        rep = select_layers_by_sensitivity(cfg, teacher, student, KnormSketch(compression_ratio=0.5), calib, spec=spec)
        self.assertEqual(rep.layers, [0, 1, 2, 3])
        for l in (0, 1, 2):
            self.assertEqual(rep.scores[l], 0.0, l)              # K/V pruning cannot reach them
        self.assertGreater(rep.scores[3], 0.0)
        self.assertEqual(rep.candidates, [3])
        self.assertEqual(rep.selected, [3])
        self.assertEqual(rep.ranking[0], 3)
        self.assertTrue(any("hybrid" in n for n in rep.notes), rep.notes)
        names = select_trainable(sm, spec, cfg.trainable, resolved_layers=rep.selected)
        self.assertEqual(names, ["model.layers.3.self_attn.o_proj.weight", "model.layers.3.self_attn.q_proj.weight"])


# ---------------------------------------------------------------------------
# calibration split
# ---------------------------------------------------------------------------
class TestCalibrationSplit(unittest.TestCase):
    def test_disjoint_from_train_and_val(self):
        tok = _WordTokenizer()
        with tempfile.TemporaryDirectory() as tmp:
            train_p, val_p = Path(tmp) / "train.jsonl", Path(tmp) / "val.jsonl"
            train_p.write_text("\n".join(json.dumps({"id": f"t{i}", "text": " ".join(f"a{i}_{j}" for j in range(30))}) for i in range(5)) + "\n")
            val_p.write_text("\n".join(json.dumps({"id": f"v{i}", "text": " ".join(f"b{i}_{j}" for j in range(30))}) for i in range(6)) + "\n")
            cfg = RecoveryConfig.from_dict({"data": {"path": str(train_p), "val_path": str(val_p), "max_length": 16, "suffix_length": 4,
                                                     "num_train_examples": 3, "num_val_examples": 2},
                                            "trainable": {"strategy": "attention_projections", "layers": "sensitivity",
                                                          "sensitivity": {"top_k": 1, "num_examples": 3}}})
            train, _ = load_split(cfg, tok, "train")
            val, _ = load_split(cfg, tok, "val")
            used = {e.id for e in train} | {e.id for e in val}
            calib, stats = load_split(cfg, tok, "calibration", exclude_ids=used)
            self.assertEqual(len(calib), 3)
            self.assertTrue(all(e.id.startswith("v") for e in calib))
            self.assertFalse({e.id for e in calib} & used)
            self.assertLessEqual(stats.n_skipped_excluded, 2)   # excluded rows met before the quota was reached
            assert_disjoint(train, calib); assert_disjoint(val, calib)
            again, _ = load_split(cfg, tok, "calibration", exclude_ids=used)
            self.assertEqual([e.id for e in calib], [e.id for e in again])        # seeded, reproducible
            # a different seed than the validation draw: calibration is not simply "the first val rows"
            cfg4 = RecoveryConfig.from_dict({**cfg.to_dict(), "trainable": {**cfg.to_dict()["trainable"],
                                                                            "sensitivity": {"top_k": 1, "num_examples": 5}}})
            with self.assertRaises(ValueError) as ctx:
                load_split(cfg4, tok, "calibration", exclude_ids=used)
            self.assertIn("sensitivity", str(ctx.exception))
            # split: train draws from the training file
            cfg_tr = RecoveryConfig.from_dict({**cfg.to_dict(), "trainable": {**cfg.to_dict()["trainable"],
                                                                              "sensitivity": {"top_k": 1, "num_examples": 2, "split": "train"}}})
            calib_tr, _ = load_split(cfg_tr, tok, "calibration", exclude_ids={e.id for e in train})
            self.assertTrue(all(e.id.startswith("t") for e in calib_tr))
            self.assertFalse({e.id for e in calib_tr} & {e.id for e in train})

    def test_build_examples_exclude(self):
        tok = _WordTokenizer()
        rows = [{"id": f"r{i}", "text": " ".join(f"w{i}_{j}" for j in range(30))} for i in range(4)]
        dcfg = DataCfg(max_length=16, suffix_length=4)
        ex, stats = build_examples(rows, tok, dcfg, n_examples=2, seed=0, exclude_ids={"r0", "r1"})
        self.assertEqual({e.id for e in ex}, {"r2", "r3"})
        self.assertLessEqual(stats.n_skipped_excluded, 2)
        with self.assertRaises(ValueError):                 # quota beyond the non-excluded rows -> loud failure
            build_examples(rows, tok, dcfg, n_examples=3, seed=0, exclude_ids={"r0", "r1"})


# ---------------------------------------------------------------------------
# end to end: select, then train with the same alignment loss
# ---------------------------------------------------------------------------
class TestEndToEnd(unittest.TestCase):
    def test_select_then_train_touches_only_selected_layers(self):
        tm = _tiny_llama(3); sm = copy.deepcopy(tm)
        teacher, student = _shell(tm), _shell(sm)
        spec = inspect_model(sm)
        cfg = _cfg(trainable={"sensitivity": {"top_k": 1, "num_examples": 2}})
        comp = KnormSketch(compression_ratio=0.5)
        train_ex, val_ex, calib = _examples(4, seed=0), _examples(1, seed=100), _examples(2, seed=300)
        rep = select_layers_by_sensitivity(cfg, teacher, student, comp, calib, spec=spec)
        self.assertEqual(len(rep.selected), 1)
        names = select_trainable(sm, spec, cfg.trainable, resolved_layers=rep.selected)
        self.assertEqual(len(names), 2)
        self.assertTrue(all(f".layers.{rep.selected[0]}." in n for n in names))
        freeze_all_but(sm, names)
        first_tl = first_trainable_layer(names, spec)
        keys = alignment_keys_for(cfg.alignment, spec.n_layers, first_trainable_layer=first_tl)
        self.assertEqual(keys, list(range(first_tl, spec.n_layers)) + ["norm"])
        setup = AlignmentSetup(keys=keys, layer_indices=[k for k in keys if isinstance(k, int)], include_final_norm=True,
                               positions_cfg=cfg.alignment.positions, loss_name=cfg.alignment.loss, layer_weights=None,
                               loss_cfg=cfg.loss, want_logits=False, mode="block", spec=spec, compression_ratio=0.5,
                               prefill_chunk_size=None, prefill_grad=False)
        snap = snapshot_parameters(sm)
        with tempfile.TemporaryDirectory() as tmp:
            state = train(teacher, student, comp, trainable_parameters(sm, names), train_ex, val_ex, cfg, setup,
                          log_path=Path(tmp) / "t.jsonl", print_fn=lambda *a, **k: None)
        self.assertEqual(state.optimizer_steps, 2)
        self.assertTrue(state.check3["passed"], state.check3)
        changed = changed_parameters(sm, snap)
        self.assertEqual(sorted(changed), sorted(names))
        # the selection was made with the original weights and the report survives JSON
        json.dumps(rep.to_dict())


if __name__ == "__main__":
    unittest.main()
