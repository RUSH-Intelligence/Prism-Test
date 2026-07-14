"""Tests for EvalRunner._apply_max_requests offset/cap slicing.

The request_offset knob exists so a tuning split and an evaluation split can
be carved deterministically from one dataset: eval = rows [0:100]
(offset 0, max_requests 100), tune = rows [100:105] (offset 100,
max_requests 5). These tests pin the per-subset [offset : offset+limit]
slicing semantics without loading any model or dataset.
"""

from __future__ import annotations

import unittest

import pandas as pd

from eval_harness.config import EvalConfig
from eval_harness.runner import EvalRunner


def _df_with_tasks() -> pd.DataFrame:
    # Task A: 10 rows (q A0..A9), task B: 3 rows (q B0..B2).
    rows = [{"task": "A", "question": f"A{i}"} for i in range(10)]
    rows += [{"task": "B", "question": f"B{i}"} for i in range(3)]
    return pd.DataFrame(rows)


def _questions(df: pd.DataFrame, task: str) -> list[str]:
    return df[df["task"] == task]["question"].tolist()


class TestApplyMaxRequestsOffset(unittest.TestCase):
    def test_offset_zero_matches_head_semantics(self):
        df = _df_with_tasks()
        out = EvalRunner._apply_max_requests(df, 4, None, 0)
        self.assertEqual(_questions(out, "A"), ["A0", "A1", "A2", "A3"])
        self.assertEqual(_questions(out, "B"), ["B0", "B1", "B2"])

    def test_identity_fast_path(self):
        df = _df_with_tasks()
        out = EvalRunner._apply_max_requests(df, None, None, 0)
        self.assertIs(out, df)

    def test_offset_slices_per_subset(self):
        df = _df_with_tasks()
        out = EvalRunner._apply_max_requests(df, 3, None, 5)
        self.assertEqual(_questions(out, "A"), ["A5", "A6", "A7"])
        # Task B has only 3 rows; offset 5 leaves nothing.
        self.assertEqual(_questions(out, "B"), [])

    def test_offset_disjoint_from_head_slice(self):
        df = _df_with_tasks()
        eval_split = EvalRunner._apply_max_requests(df, 5, None, 0)
        tune_split = EvalRunner._apply_max_requests(df, 5, None, 5)
        eval_q = set(eval_split["question"])
        tune_q = set(tune_split["question"])
        self.assertEqual(eval_q & tune_q, set())
        self.assertEqual(_questions(eval_split, "A") + _questions(tune_split, "A"),
                         [f"A{i}" for i in range(10)])

    def test_offset_with_no_limit_drops_prefix(self):
        df = _df_with_tasks()
        out = EvalRunner._apply_max_requests(df, None, None, 8)
        self.assertEqual(_questions(out, "A"), ["A8", "A9"])
        self.assertEqual(_questions(out, "B"), [])

    def test_offset_composes_with_per_subset_limits(self):
        df = _df_with_tasks()
        out = EvalRunner._apply_max_requests(df, 2, {"A": 3, "B": 0}, 1)
        self.assertEqual(_questions(out, "A"), ["A1", "A2", "A3"])
        self.assertEqual(_questions(out, "B"), [])

    def test_offset_beyond_subset_length_yields_empty(self):
        df = _df_with_tasks()
        out = EvalRunner._apply_max_requests(df, 5, None, 100)
        self.assertEqual(len(out), 0)

    def test_no_task_column_offset_and_cap(self):
        df = pd.DataFrame({"question": [f"q{i}" for i in range(10)]})
        out = EvalRunner._apply_max_requests(df, 3, None, 4)
        self.assertEqual(out["question"].tolist(), ["q4", "q5", "q6"])
        out_nolimit = EvalRunner._apply_max_requests(df, None, None, 7)
        self.assertEqual(out_nolimit["question"].tolist(), ["q7", "q8", "q9"])

    def test_global_zero_limit_still_empties(self):
        df = _df_with_tasks()
        out = EvalRunner._apply_max_requests(df, 0, None, 5)
        self.assertEqual(len(out), 0)


class TestEvalConfigRequestOffset(unittest.TestCase):
    def test_default_zero(self):
        self.assertEqual(EvalConfig(backend="hf").request_offset, 0)

    def test_negative_rejected(self):
        with self.assertRaises(ValueError):
            EvalConfig(backend="hf", request_offset=-1)

    def test_offset_with_fraction_rejected(self):
        # fraction sampling reshuffles rows before slicing; the combination
        # would silently break the disjoint tuning/eval split guarantee.
        with self.assertRaises(ValueError):
            EvalConfig(backend="hf", request_offset=100, fraction=0.5)
        # offset 0 composes with fraction fine (old behavior).
        EvalConfig(backend="hf", request_offset=0, fraction=0.5)

    def test_round_trips_through_dict(self):
        cfg = EvalConfig(backend="hf", request_offset=100)
        self.assertEqual(cfg.to_dict()["request_offset"], 100)
