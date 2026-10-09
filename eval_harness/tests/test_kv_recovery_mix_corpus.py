"""Mixed distillation corpus: builder helpers (row ranges, sampling, packing, qa rows), qa rows through the
data module (windows, caps, stats, disjointness) and the matrix presets — CPU, no downloads."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
import torch
import yaml

from eval_harness.kv_recovery.config import RecoveryConfig
from eval_harness.kv_recovery.data import assert_disjoint, build_examples, load_split

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts import prepare_kv_recovery_mix as mix  # noqa: E402
from scripts import kv_recovery_matrix as matrix  # noqa: E402


class _WordTokenizer:
    bos_token_id = 1

    def encode(self, text, add_special_tokens=False, return_tensors=None):  # noqa: ARG002
        ids = [2 + (hash(w) % 1000) for w in text.split()]
        return torch.tensor([ids]) if return_tensors == "pt" else ids


class _FakePipeline:
    def __init__(self):
        self.tokenizer = _WordTokenizer()

    def preprocess(self, context, questions, answer_prefix, max_context_length, use_chat_template=True,
                   strip_auto_system_block=False, **kwargs):  # noqa: ARG002
        ctx = self.tokenizer.encode("<s> " + context, return_tensors="pt")
        qs = [self.tokenizer.encode(q + " \n " + answer_prefix, return_tensors="pt") for q in questions]
        return {"context_ids": ctx, "questions_ids": qs}


def _qa(i, task="qa_1", n_ctx=40, source="ruler16k"):
    return {"id": f"{source}/{task}/row{i}", "kind": "qa", "source": source, "task": task, "row": i,
            "context": " ".join(f"{task}_c{i}_w{j}" for j in range(n_ctx)), "question": f"what is item {i} ?",
            "answer_prefix": "The answer is", "answer": [f"{1000 + i}"]}


def _text(i, n_words=60, source="pg19"):
    return {"id": f"{source}-{i}", "kind": "text", "source": source, "text": " ".join(f"{source}{i}_w{j}" for j in range(n_words))}


class TestBuilderHelpers(unittest.TestCase):
    def test_ranges_and_sampling(self):
        self.assertEqual(mix.parse_range("120-199"), (120, 200))
        self.assertEqual(mix.parse_range("137"), (137, 138))
        with self.assertRaises(ValueError):
            mix.parse_range("50-40")
        mix.check_row_ranges((120, 200), (100, 120))
        with self.assertRaises(ValueError):
            mix.check_row_ranges((90, 200), (100, 120))          # overlaps the evaluated pool
        with self.assertRaises(ValueError):
            mix.check_row_ranges((100, 200), (150, 170))         # train / val overlap
        rows = mix.sample_rows(200, 120, 200, 60, seed="s")
        self.assertEqual(len(rows), 60)
        self.assertTrue(all(120 <= r < 200 for r in rows))
        self.assertEqual(rows, mix.sample_rows(200, 120, 200, 60, seed="s"))
        self.assertEqual(mix.sample_rows(150, 100, 200, 99, seed="s"), list(range(100, 150)))   # capped by availability

    def test_packing(self):
        docs = [(f"d{i}", " ".join(["word"] * 100)) for i in range(10)] + [("tiny", "too short")]
        rows = list(mix.pack_documents(docs, target_chars=1200, min_doc_chars=200))
        self.assertEqual(len(rows), 3)                         # 3 x ~500 chars per row -> 3 rows from 9-10 docs
        self.assertTrue(all(r["chars"] >= 1200 for r in rows))
        self.assertTrue(all(r["n_docs"] == len(r["doc_ids"]) for r in rows))
        self.assertNotIn("tiny", [d for r in rows for d in r["doc_ids"]])

    def test_qa_row_normalises_answers(self):
        r = mix.qa_row("longbench", "qasper", 123, {"context": "c", "question": "q", "answers": "['a', 'b']", "answer_prefix": "Answer:"})
        self.assertEqual(r["answer"], ["a", "b"])
        self.assertEqual(r["id"], "longbench/qasper/row123")
        r2 = mix.qa_row("ruler16k", "qa_1", 5, {"context": "c", "question": "q", "answer": ["x"]})
        self.assertEqual((r2["answer"], r2["answer_prefix"]), (["x"], ""))

    def test_benchmark_rows_respect_range_and_cap(self):
        class _B:
            def load(self, subsets=None):
                frames = [pd.DataFrame({"context": ["x " * (10 * (i % 3 + 1)) for i in range(200)], "question": ["q"] * 200,
                                        "answer_prefix": ["A:"] * 200, "answer": [["1"]] * 200, "task": [t] * 200})
                          for t in (subsets or ["qa_1"])]
                return pd.concat(frames, ignore_index=True)
        with mock.patch("eval_harness.benchmarks.registry.get_benchmark", return_value=_B()):
            rows, rep = mix.benchmark_qa_rows("ruler16k", ["qa_1", "cwe"], (120, 200), 10, seed=0, max_chars=None, tag="t")
            self.assertEqual(len(rows), 20)
            self.assertTrue(all(120 <= r["row"] < 200 for r in rows))
            self.assertEqual(sorted(rep), ["cwe", "qa_1"])
            capped, rep2 = mix.benchmark_qa_rows("longbench", ["qa_1"], (100, 200), 10, seed=0, max_chars=25, tag="t")
            self.assertTrue(all(len(r["context"]) <= 25 for r in capped))
            self.assertGreater(rep2["qa_1"]["skipped_long"], 0)


class TestMixedWindows(unittest.TestCase):
    def test_qa_and_text_rows_become_windows(self):
        pipe = _FakePipeline()
        rows = [_qa(i) for i in range(120, 126)] + [_text(i) for i in range(4)]
        cfg = RecoveryConfig.from_dict({"data": {"max_length": 16, "suffix_length": 4}})
        ex, stats = build_examples(rows, pipe.tokenizer, cfg.data, n_examples=10, seed=0, pipeline=pipe)
        self.assertEqual(len(ex), 10)
        self.assertEqual(stats.by_kind, {"qa": 6, "text": 4})
        self.assertEqual(stats.by_source, {"pg19": 4, "ruler16k": 6})
        qa = [e for e in ex if e.meta["kind"] == "qa"][0]
        self.assertEqual(qa.id, "ruler16k/qa_1/row120" if qa.meta["row"] == 120 else qa.id)
        self.assertGreater(qa.meta["n_answer_tokens"], 0)
        self.assertEqual(qa.context_len, 41)                                   # <s> + 40 words, no window cut for qa rows
        txt = [e for e in ex if e.meta["kind"] == "text"][0]
        self.assertEqual((txt.context_len, txt.suffix_len), (12, 4))         # text rows keep the [T | L] window
        # the context cap skips long qa rows and counts them
        cfg2 = RecoveryConfig.from_dict({"data": {"max_length": 16, "suffix_length": 4, "max_context_tokens": 30}})
        ex2, stats2 = build_examples(rows, pipe.tokenizer, cfg2.data, n_examples=4, seed=0, pipeline=pipe)
        self.assertEqual(stats2.by_kind, {"text": 4})
        self.assertEqual(len(ex2), 4)
        with self.assertRaises(ValueError) as ctx:                               # asking for more than the cap leaves
            build_examples(rows, pipe.tokenizer, cfg2.data, n_examples=10, seed=0, pipeline=pipe)
        self.assertIn("long=6", str(ctx.exception))
        # qa rows need the pipeline
        with self.assertRaises(ValueError):
            build_examples([_qa(1)], pipe.tokenizer, cfg.data, n_examples=1, seed=0, pipeline=None)
        with self.assertRaises(ValueError):
            build_examples([{"id": "z", "kind": "video", "text": "x"}], pipe.tokenizer, cfg.data, n_examples=1, seed=0, pipeline=pipe)
        # question-only region
        cfg3 = RecoveryConfig.from_dict({"data": {"max_length": 16, "suffix_length": 4, "qa_region": "question"}})
        ex3, _ = build_examples([_qa(7)], pipe.tokenizer, cfg3.data, n_examples=1, seed=0, pipeline=pipe)
        self.assertEqual(ex3[0].meta["n_answer_tokens"], 0)

    def test_load_split_and_disjointness_on_a_mixed_corpus(self):
        pipe = _FakePipeline()
        with tempfile.TemporaryDirectory() as tmp:
            tr = Path(tmp) / "mix_train.jsonl"; va = Path(tmp) / "mix_val.jsonl"
            tr.write_text("\n".join(json.dumps(r) for r in [_qa(i) for i in range(120, 130)] + [_text(i) for i in range(10)]) + "\n")
            va.write_text("\n".join(json.dumps(r) for r in [_qa(i) for i in range(100, 106)] + [_text(i + 50) for i in range(6)]) + "\n")
            cfg = RecoveryConfig.from_dict({"data": {"path": str(tr), "val_path": str(va), "max_length": 16, "suffix_length": 4,
                                                     "num_train_examples": 12, "num_val_examples": 4},
                                            "trainable": {"strategy": "attention_projections", "layers": "sensitivity",
                                                          "sensitivity": {"top_k": 2, "num_examples": 4}}})
            train, ts = load_split(cfg, pipe.tokenizer, "train", pipeline=pipe)
            val, vs = load_split(cfg, pipe.tokenizer, "val", pipeline=pipe)
            calib, cs = load_split(cfg, pipe.tokenizer, "calibration", pipeline=pipe, exclude_ids={e.id for e in train} | {e.id for e in val})
            self.assertEqual((len(train), len(val), len(calib)), (12, 4, 4))
            assert_disjoint(train, val); assert_disjoint(train, calib); assert_disjoint(val, calib)
            self.assertTrue(set(ts.by_kind) == {"qa", "text"})
            # RULER rows of one task share their instruction prefix: that must NOT count as a duplicate context
            same_task_train = [e for e in train if e.meta["kind"] == "qa"]
            same_task_val = [e for e in val if e.meta["kind"] == "qa"]
            self.assertTrue(same_task_train and same_task_val)
            assert_disjoint(same_task_train, same_task_val)
            # but an identical context is caught
            with self.assertRaises(AssertionError):
                assert_disjoint(train, [train[0]])


class TestMatrixPresets(unittest.TestCase):
    def setUp(self):
        self.m = yaml.safe_load((REPO / "configs" / "kv_recovery" / "matrix.yaml").read_text())

    def test_base_entries_contexts_and_presets(self):
        self.assertEqual(matrix.base_entry(self.m, "ministral_3b"), ("configs/kv_recovery/ministral_3b.yaml", ["16k", "32k"]))
        self.assertEqual(matrix.base_entry(self.m, "ministral_3b_mix")[1], ["mix16k", "mix32k"])
        cells = matrix.expand(self.m, preset="ablation_topk")
        self.assertEqual(len(cells), 40)
        names = {c.run_name for c in cells}
        self.assertIn("ministral_3b_mix16k_knorm_r075_qo_sens8", names)
        self.assertIn("qwen35_4b_mix32k_cur_r075_kv_sens8", names)
        self.assertNotIn("qwen35_4b_mix16k_knorm_r075_qo_sens16", names)          # skipped: 8 K/V layers only
        self.assertTrue(all(c.base_config.endswith("_mix.yaml") for c in cells))
        self.assertTrue(all(c.overrides["data.num_train_examples"] == 1024 for c in cells))
        self.assertTrue(all(c.overrides["data.path"].endswith(f"{c.context}_train.jsonl") for c in cells))
        for c in cells:
            RecoveryConfig.from_dict(matrix.yaml.safe_load(open(REPO / c.base_config)) if False else
                                     __import__("eval_harness.kv_recovery.config", fromlist=["apply_overrides"]).apply_overrides(
                                         yaml.safe_load(open(REPO / c.base_config)), shortcuts=c.overrides))
        prim = matrix.expand(self.m, primary=True)
        self.assertEqual(len(prim), 24)
        self.assertTrue(all(c.context == "16k" and not c.base_config.endswith("_mix.yaml") for c in prim))
        with self.assertRaises(ValueError):
            matrix.expand(self.m, preset="nope")
        # the mix cards load and evaluate LongBench on rows 0-99 (rows 100-199 are corpus rows)
        for card in ("ministral_3b_mix.yaml", "qwen35_4b_mix.yaml"):
            cfg = RecoveryConfig.from_dict(yaml.safe_load(open(REPO / "configs" / "kv_recovery" / card)))
            lb = [b for b in cfg.eval.benchmarks if b.benchmark == "longbench"][0]
            self.assertEqual(lb.max_requests, 100)
            self.assertEqual(cfg.data.max_context_tokens, 16384)
            self.assertEqual(cfg.data.num_train_examples, 1024)


if __name__ == "__main__":
    unittest.main()
