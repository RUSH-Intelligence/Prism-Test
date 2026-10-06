"""Recovery metrics, per-example re-scoring, paired bootstrap, representation metrics, rendering."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from eval_harness.benchmarks.registry import get_benchmark
from eval_harness.kv_recovery.metrics import (
    align_conditions,
    benchmark_report,
    consistency_check,
    paired_bootstrap,
    per_example_scores,
    recovery_metrics,
    render_markdown,
    representation_metrics,
    stat_fraction,
    stat_recovery,
)


class TestRecoveryMetrics(unittest.TestCase):
    def test_example_from_spec(self):
        m = recovery_metrics(90.0, 75.0, 84.0)
        self.assertAlmostEqual(m["compression_drop"], 15.0)
        self.assertAlmostEqual(m["recovery"], 9.0)
        self.assertAlmostEqual(m["recovery_fraction"], 0.6)
        self.assertTrue(m["recovery_fraction_defined"])
        self.assertEqual(m["flags"], [])

    def test_zero_and_negative_drop(self):
        z = recovery_metrics(80.0, 80.0, 85.0)
        self.assertIsNone(z["recovery_fraction"]); self.assertFalse(z["recovery_fraction_defined"])
        self.assertEqual(z["undefined_reason"], "no_compression_drop"); self.assertIn("undefined_gap", z["flags"])
        n = recovery_metrics(80.0, 82.0, 85.0)
        self.assertEqual(n["undefined_reason"], "negative_compression_drop")
        u = recovery_metrics(80.0, 78.0, 79.0)
        self.assertIn("unstable_gap", u["flags"]); self.assertAlmostEqual(u["recovery_fraction"], 0.5)
        d = recovery_metrics(80.0, 60.0, 70.0, dense_recovered=79.0)
        self.assertAlmostEqual(d["dense_regression"], -1.0); self.assertAlmostEqual(d["did"], 11.0)


def _ruler_csv(path: Path, preds):
    rows = []
    for task, answer, pred in preds:
        rows.append({"question": f"q-{task}-{answer}", "answer_prefix": "", "answer": f"['{answer}']", "task": task,
                     "max_new_tokens": 32, "context_length": 16384, "predicted_answer": pred})
    pd.DataFrame(rows).to_csv(path, index=False)


class TestPerExampleScores(unittest.TestCase):
    def test_ruler_rows_match_benchmark_scorer(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "predictions.csv"
            _ruler_csv(p, [("niah_single_1", "123", "the number is 123"), ("niah_single_1", "456", "no idea"),
                           ("qa_1", "Paris", "Answer: paris."), ("qa_1", "Rome", "Berlin")])
            df = per_example_scores("ruler16k", p)
            self.assertEqual(df["score"].tolist(), [100.0, 0.0, 100.0, 0.0])
            self.assertEqual(df["ordinal"].tolist(), [0, 1, 0, 1])
            full = get_benchmark("ruler16k").score(pd.read_csv(p, keep_default_na=False))
            cc = consistency_check(df, full)
            self.assertTrue(cc["ok"], cc)

    def test_longbench_trec_all_classes_from_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "predictions.csv"
            pd.DataFrame([{"question": "q", "answers": "['Location']", "task": "trec", "all_classes": "['Location' 'Number' 'Person']",
                           "predicted_answer": "Location\nmore", "max_new_tokens": 64},
                          {"question": "q2", "answers": "['the red door']", "task": "qasper", "all_classes": "",
                           "predicted_answer": "the red door", "max_new_tokens": 128}]).to_csv(p, index=False)
            df = per_example_scores("longbench", p)
            self.assertEqual(df["score"].tolist(), [100.0, 100.0])


class TestBootstrap(unittest.TestCase):
    def _frames(self, seed=0):
        rng = np.random.default_rng(seed)
        rows = []
        for task in ("a", "b"):
            for i in range(30):
                rows.append({"task": task, "ordinal": i, "row_id": f"{task}{i}"})
        base = pd.DataFrame(rows)
        dense = base.assign(score=rng.integers(0, 2, len(base)) * 100.0 * 0 + 100.0)
        comp = base.assign(score=(rng.random(len(base)) < 0.4) * 100.0)
        rec = base.assign(score=np.maximum(comp["score"].to_numpy(), (rng.random(len(base)) < 0.5) * 100.0))
        return {"dense": dense, "compressed": comp, "compressed_recovered": rec}

    def test_align_and_identical_conditions_zero_width(self):
        frames = self._frames()
        cells = align_conditions(frames)
        self.assertEqual(set(cells), {"a", "b"})
        same = {"x": frames["dense"], "y": frames["dense"].copy()}
        cells2 = align_conditions(same)
        res = paired_bootstrap(cells2, lambda m: m["x"] - m["y"], 200, 0)
        self.assertEqual(res["estimate"], 0.0); self.assertEqual(res["ci_low"], 0.0); self.assertEqual(res["ci_high"], 0.0)
        bad = {"dense": frames["dense"], "compressed": frames["compressed"].assign(row_id="zzz")}
        with self.assertRaises(ValueError):
            align_conditions(bad)

    def test_report_structure_and_reproducibility(self):
        frames = self._frames()
        r1 = benchmark_report(frames, n_resamples=300, seed=1)
        r2 = benchmark_report(frames, n_resamples=300, seed=1)
        self.assertEqual(r1["overall"]["ci"], r2["overall"]["ci"])
        o = r1["overall"]
        self.assertEqual(o["n_examples"], 60)
        for k in ("compression_drop", "recovery", "recovery_fraction"):
            ci = o["ci"][k]
            self.assertLessEqual(ci["ci_low"], ci["estimate"]); self.assertGreaterEqual(ci["ci_high"], ci["estimate"])
        self.assertEqual(set(r1["tasks"]), {"a", "b"})
        self.assertAlmostEqual(o["recovery_fraction"], stat_fraction({"dense": np.array([o["dense"]]),
                               "compressed": np.array([o["compressed"]]), "compressed_recovered": np.array([o["compressed_recovered"]])})[0])
        md = render_markdown({"run_name": "t", "model": {"name": "m"}, "kv_compression": {"kv_compressor": "knorm", "compression_ratio": 0.75},
                              "checkpoint": {"sha256": "abc"}, "benchmarks": {"ruler16k": r1}})
        self.assertIn("| ruler16k | **macro** |", md)
        self.assertEqual(stat_recovery({"compressed_recovered": np.array([3.0]), "compressed": np.array([1.0])})[0], 2.0)


class TestRepresentationMetrics(unittest.TestCase):
    def test_identical_and_perturbed(self):
        torch.manual_seed(0)
        t = {0: torch.randn(5, 8), "norm": torch.randn(5, 8)}
        same = representation_metrics(t, t)
        self.assertAlmostEqual(same["0"]["cosine"], 1.0, places=5)
        self.assertEqual(same["0"]["normalized_mse"], 0.0)
        self.assertEqual(same["norm"]["relative_error"], 0.0)
        s = {k: v + 0.5 * torch.randn_like(v) for k, v in t.items()}
        pert = representation_metrics(t, s)
        self.assertLess(pert["0"]["cosine"], 1.0); self.assertGreater(pert["0"]["normalized_mse"], 0.0)
        self.assertEqual(pert["0"]["n_positions"], 5)


if __name__ == "__main__":
    unittest.main()
