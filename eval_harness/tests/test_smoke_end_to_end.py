"""Smoke test: full eval pipeline runs end-to-end without a real model.

Uses the in-memory `mock_benchmark` and a fake adapter that returns a constant
answer. Catches wiring breaks between config -> runner -> benchmark -> adapter
-> scoring -> file output that the per-piece unit tests don't see.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from typing import List
from unittest.mock import patch

from eval_harness.config import EvalConfig
from eval_harness.runner import EvalRunner


class _FakeAdapter:
    """Returns a constant answer for any prompt. No GPU, no model load."""

    def generate(self, prompts: List[str], gen_cfg) -> List[str]:
        return ["Paris"] * len(prompts)


class TestSmokeEndToEnd(unittest.TestCase):
    def test_pipeline_runs_and_writes_outputs(self):
        tmpdir = tempfile.mkdtemp(prefix="smoke_eval_")
        self.addCleanup(shutil.rmtree, tmpdir, ignore_errors=True)

        config = EvalConfig(
            benchmark="mock_benchmark",
            backend="hf",
            model="fake-model-not-loaded",
            max_new_tokens=4,
            output_dir=tmpdir,
        )

        def _fake_setup(runner_self):
            runner_self.adapter = _FakeAdapter()

        with patch.object(EvalRunner, "_setup_adapter", _fake_setup):
            runner = EvalRunner(config)
            run_dir = runner.run()

        self.assertTrue((run_dir / "predictions.csv").exists())
        self.assertTrue((run_dir / "metrics.json").exists())
        self.assertTrue((run_dir / "config.yaml").exists())

        with (run_dir / "metrics.json").open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        self.assertIn("overall_score", metrics)
        self.assertEqual(metrics["total_samples"], 2)


if __name__ == "__main__":
    unittest.main()
