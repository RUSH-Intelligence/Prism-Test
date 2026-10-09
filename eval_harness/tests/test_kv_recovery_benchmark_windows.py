"""Benchmark-context windows for the layer-wise analysis (``data.benchmark_examples``) and the
figure script's data handling — CPU, no datasets download (the benchmark registry is stubbed)."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
import torch

from eval_harness.kv_recovery import data as kvdata
from eval_harness.kv_recovery.data import benchmark_example, benchmark_examples, benchmark_rows

try:
    import matplotlib  # noqa: F401
    HAS_MPL = True
except Exception:  # pragma: no cover
    HAS_MPL = False


class _WordTokenizer:
    def encode(self, text, add_special_tokens=False, return_tensors=None):  # noqa: ARG002
        ids = [2 + (hash(w) % 1000) for w in text.split()]
        return torch.tensor([ids]) if return_tensors == "pt" else ids


class _FakePipeline:
    """Mimics ResearchGenerationPipeline.preprocess for the raw path: context ids + [question + suffix + prefix] ids."""

    def __init__(self):
        self.tokenizer = _WordTokenizer()
        self.calls = []

    def preprocess(self, context, questions, answer_prefix, max_context_length, use_chat_template=True,
                   strip_auto_system_block=False, **kwargs):
        self.calls.append({"context": context, "questions": questions, "answer_prefix": answer_prefix,
                           "use_chat_template": use_chat_template, "strip": strip_auto_system_block})
        ctx = self.tokenizer.encode("<s> " + context, return_tensors="pt")
        qs = [self.tokenizer.encode(q + " \n " + answer_prefix, return_tensors="pt") for q in questions]
        return {"context_ids": ctx, "questions_ids": qs}


def _fake_frame(task: str, n: int) -> pd.DataFrame:
    return pd.DataFrame({
        "context": [" ".join(f"{task}_ctx{i}_w{j}" for j in range(40)) for i in range(n)],
        "question": [f"What is the magic number for {task} item {i} ?" for i in range(n)],
        "answer_prefix": ["The answer is"] * n,
        "answer": [[f"{1000 + i}", f"{2000 + i}"] if task.startswith("niah_multi") else [f"{1000 + i}"] for i in range(n)],
        "task": [task] * n,
    })


class _FakeBenchmark:
    def __init__(self, tasks):
        self.tasks = tasks

    def load(self, subsets=None):
        frames = [_fake_frame(t, 120) for t in (subsets or self.tasks)]
        return pd.concat(frames, ignore_index=True)


class TestBenchmarkWindows(unittest.TestCase):
    def setUp(self):
        self.bench = _FakeBenchmark(["niah_single_1", "niah_multivalue", "qa_1"])
        self.patcher = mock.patch("eval_harness.benchmarks.registry.get_benchmark", return_value=self.bench)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def test_rows_are_sampled_from_the_evaluation_pool_and_seeded(self):
        rows = benchmark_rows("ruler16k", None, pool_rows=100, rows_per_task=2, seed=0)
        self.assertEqual(len(rows), 6)
        self.assertEqual([r["_task"] for r in rows], ["niah_single_1"] * 2 + ["niah_multivalue"] * 2 + ["qa_1"] * 2)
        self.assertTrue(all(r["_row"] < 100 for r in rows))                    # never beyond the scored pool
        again = benchmark_rows("ruler16k", None, pool_rows=100, rows_per_task=2, seed=0)
        self.assertEqual([r["_row"] for r in rows], [r["_row"] for r in again])
        other = benchmark_rows("ruler16k", None, pool_rows=100, rows_per_task=2, seed=1)
        self.assertNotEqual([r["_row"] for r in rows], [r["_row"] for r in other])
        sub = benchmark_rows("ruler16k", ["qa_1"], pool_rows=5, rows_per_task=9, seed=0)
        self.assertEqual(len(sub), 5)                                           # capped at the pool
        self.assertTrue(all(r["_row"] < 5 for r in sub))

    def test_example_shaping_matches_the_evaluation_prompt(self):
        pipe = _FakePipeline()
        row = {**_fake_frame("qa_1", 1).iloc[0].to_dict(), "_task": "qa_1", "_row": 7}
        ex = benchmark_example(row, pipe.tokenizer, pipeline=pipe, bench_name="ruler16k", region="question_answer")
        self.assertEqual(ex.id, "ruler16k/qa_1/row7")
        call = pipe.calls[-1]
        self.assertEqual(call["questions"], [row["question"]])
        self.assertEqual(call["answer_prefix"], "The answer is")
        self.assertTrue(call["use_chat_template"] and call["strip"])            # the evaluation defaults
        self.assertNotIn(row["question"], call["context"])                      # query_aware: false
        q_only = benchmark_example(row, pipe.tokenizer, pipeline=pipe, bench_name="ruler16k", region="question")
        self.assertEqual(q_only.suffix_len, ex.meta["n_question_tokens"])
        self.assertEqual(ex.suffix_len, ex.meta["n_question_tokens"] + ex.meta["n_answer_tokens"])
        self.assertGreater(ex.meta["n_answer_tokens"], 0)
        self.assertEqual(ex.meta["context_tokens"], ex.context_len)
        self.assertEqual(ex.ctx_ids.dtype, torch.long)
        with self.assertRaises(ValueError):
            benchmark_example(row, pipe.tokenizer, pipeline=pipe, bench_name="ruler16k", region="answers")

    def test_multi_answer_tasks_join_their_gold_answers(self):
        pipe = _FakePipeline()
        row = {**_fake_frame("niah_multivalue", 1).iloc[0].to_dict(), "_task": "niah_multivalue", "_row": 0}
        ex = benchmark_example(row, pipe.tokenizer, pipeline=pipe, bench_name="ruler16k")
        self.assertEqual(ex.meta["n_answer_tokens"], 2)                          # "1000, 2000" -> two words
        row_qa = {**_fake_frame("qa_1", 1).iloc[0].to_dict(), "_task": "qa_1", "_row": 0}
        row_qa["answer"] = ["France", "France"]                                  # synonyms -> first only
        ex_qa = benchmark_example(row_qa, pipe.tokenizer, pipeline=pipe, bench_name="ruler16k")
        self.assertEqual(ex_qa.meta["n_answer_tokens"], 1)

    def test_examples_and_stats(self):
        pipe = _FakePipeline()
        exs, stats = benchmark_examples("ruler32k", None, pipe.tokenizer, pipeline=pipe, rows_per_task=1, pool_rows=10, seed=3)
        self.assertEqual(len(exs), 3)
        self.assertEqual(stats.n_used, 3)
        self.assertEqual(stats.format, "benchmark")
        self.assertEqual(stats.suffix_mode, "question_answer")
        self.assertTrue(all(e.id.startswith("ruler32k/") for e in exs))
        self.assertEqual(stats.context_tokens, sum(e.context_len for e in exs))


@unittest.skipUnless(HAS_MPL, "matplotlib not installed")
class TestFigureScript(unittest.TestCase):
    def _fake_measurement(self, model, source, comp, ratio, layers, hooked, tasks):
        import random
        rng = random.Random(f"{source}{comp}{ratio}")
        per_example = {}
        for t in tasks:
            for r in range(2):
                per_example[f"{source}/{t}/row{r}"] = {str(l): (0.1 + 0.02 * l + rng.random() * 0.02 if l in hooked else 0.0)
                                                      for l in layers}
        scores = {str(l): sum(v[str(l)] for v in per_example.values()) / len(per_example) for l in layers}
        return {"schema_version": 2, "model": model, "source": source, "compressor": comp, "compression_ratio": ratio,
                "protocol": {"n_examples": len(per_example), "rows_per_task": 2},
                "report": {"layers": layers, "hooked_layers": hooked, "scores": scores, "std": {str(l): 0.01 for l in layers},
                           "per_example": per_example, "ranking": sorted(layers, key=lambda l: -scores[str(l)]),
                           "candidates": hooked, "selected": hooked[:2], "top_k": 2}}

    def test_profiles_and_task_figures_are_written(self):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
        import plot_layer_sensitivity as pls

        layers, hooked = list(range(8)), [3, 7]
        tasks = ["niah_single_1", "qa_1", "cwe"]
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "in"; src.mkdir()
            for source in ("pg19", "ruler16k", "ruler32k"):
                for comp in ("knorm", "cur"):
                    for ratio in (0.75, 0.5):
                        d = self._fake_measurement("org/Tiny.5", source, comp, ratio, layers, hooked, tasks if source != "pg19" else ["all"])
                        (src / f"org--Tiny.5__{source}__{comp}_r{int(ratio*100):03d}.json").write_text(json.dumps(d))
            (src / "summary.json").write_text("[]")
            meas = pls.load_measurements(str(src))
            self.assertEqual(len(meas), 12)
            curves = pls.per_task_curves([m for m in meas if m["source"] == "ruler16k"][0]["report"])
            self.assertEqual(set(curves), set(tasks))
            self.assertEqual(curves["qa_1"][1][0], 0.0)                          # layer 0 carries no K/V -> 0
            out = Path(tmp) / "fig"
            rc = pls.main(["--inputs", str(src), "--out-dir", str(out), "--formats", "png"])
            self.assertEqual(rc, 0)
            names = sorted(p.name for p in out.iterdir())
            self.assertIn("org--Tiny.5__profiles.png", names)
            self.assertIn("org--Tiny.5__tasks__knorm_r075.png", names)
            self.assertIn("org--Tiny.5__tasks__cur_r075.png", names)


if __name__ == "__main__":
    unittest.main()
