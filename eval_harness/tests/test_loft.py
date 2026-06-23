"""Tests for LoftBenchmark — prediction extraction + RAG scoring.

Pins the upstream-parity behavior (google-deepmind/loft) that loft.py implements:
  * _extract_prediction commits to the FIRST bracketed line (break OUTSIDE the
    try) and has NO answer_prefix-split / raw-text fallback (non-bracketed output
    scores 0, like upstream);
  * overall_score headline == subspan_em (LOFT's primary metric), not em;
  * single-value (nq/hotpot/musique) reports em/subspan_em/f1; multi-value
    (qampari/quest) reports em/coverage/subspan_em (no f1), and coverage omits
    empty predictions from its denominator (upstream MultiValueRagEvaluation).

No model loading (object.__new__). The multi-value subspan metric needs scipy,
so that test is skipped where scipy is absent (mirrors the suite's other skips).
"""
from __future__ import annotations

import importlib.util
import unittest

import pandas as pd

from eval_harness.benchmarks.loft import LoftBenchmark, _extract_prediction

_HAS_SCIPY = importlib.util.find_spec("scipy") is not None


def _score(rows):
    df = pd.DataFrame(rows)
    if "answer_prefix" not in df.columns:
        df["answer_prefix"] = "Final Answer: "
    bench = object.__new__(LoftBenchmark)  # bypass __init__: score() uses only df
    return bench.score(df)


class TestLoftExtractPrediction(unittest.TestCase):
    def test_wellformed_list(self):
        self.assertEqual(_extract_prediction("Final Answer: ['Madrid']"), ["Madrid"])

    def test_no_brackets_returns_empty(self):
        # upstream returns [] (scores 0); NO answer_prefix or raw-text fallback
        self.assertEqual(_extract_prediction("The answer is Madrid"), [])
        self.assertEqual(_extract_prediction("Final Answer: Madrid"), [])

    def test_commits_to_first_bracket_line(self):
        # first bracket line wins even when a later line looks better
        self.assertEqual(
            _extract_prediction("x ['foo']\nFinal Answer: ['Madrid']"), ["foo"]
        )

    def test_first_bracket_parsefail_returns_empty(self):
        # first bracket line fails to parse -> break with [] (does NOT scan onward)
        self.assertEqual(
            _extract_prediction("Note [x y z]\nFinal Answer: ['Madrid']"), []
        )

    def test_multi_value_list(self):
        self.assertEqual(
            _extract_prediction("Final Answer: ['a', 'b', 'c']"), ["a", "b", "c"]
        )

    def test_apostrophe_escaped(self):
        self.assertEqual(
            _extract_prediction("['child bride', \"the devil's sleep\"]"),
            ["child bride", "the devil's sleep"],
        )

    def test_markdown_stars_and_backticks_stripped(self):
        self.assertEqual(_extract_prediction("**Final Answer:** `['Paris']`"), ["Paris"])

    def test_int_list_coerced_to_str(self):
        self.assertEqual(_extract_prediction("Final Answer: [1, 2]"), ["1", "2"])


class TestLoftScoreSingleValue(unittest.TestCase):
    def test_headline_is_subspan_em_not_em(self):
        # one exact hit + one subspan-only hit ("the city of Madrid" contains "madrid")
        res = _score([
            {"task": "nq_32k", "answers": ["Madrid"],
             "predicted_answer": "Final Answer: ['Madrid']"},
            {"task": "nq_32k", "answers": ["Madrid"],
             "predicted_answer": "Final Answer: ['the city of Madrid']"},
        ])
        ts = res["task_scores"]["nq_32k"]
        self.assertEqual(ts["em"], 50.0)        # only the exact one
        self.assertEqual(ts["subspan_em"], 100.0)  # both substring-match
        self.assertIn("f1", ts)
        # headline must be subspan_em (LOFT primary metric), not em
        self.assertEqual(res["overall_score"], res["overall_metrics"]["subspan_em"])
        self.assertEqual(res["overall_score"], 100.0)

    def test_non_bracketed_prediction_scores_zero(self):
        res = _score([
            {"task": "hotpotqa_32k", "answers": ["Madrid"],
             "predicted_answer": "Madrid is the answer"},
        ])
        ts = res["task_scores"]["hotpotqa_32k"]
        self.assertEqual((ts["em"], ts["subspan_em"], ts["f1"]), (0.0, 0.0, 0.0))


@unittest.skipUnless(_HAS_SCIPY, "multi-value subspan metric requires scipy")
class TestLoftScoreMultiValue(unittest.TestCase):
    def test_metrics_keys_and_coverage_omits_empty(self):
        # qampari: one perfect set match + one EMPTY (non-bracketed) prediction
        res = _score([
            {"task": "qampari_32k", "answers": ["Spain", "France"],
             "predicted_answer": "Final Answer: ['Spain', 'France']"},
            {"task": "qampari_32k", "answers": ["Spain", "France"],
             "predicted_answer": "no list here"},
        ])
        ts = res["task_scores"]["qampari_32k"]
        self.assertIn("coverage", ts)
        self.assertNotIn("f1", ts)              # multi-value has no f1
        self.assertEqual(ts["em"], 50.0)        # (1 + 0)/2
        self.assertEqual(ts["subspan_em"], 50.0)
        # coverage omits the empty prediction from the denominator -> 1.0 over 1 = 100
        self.assertEqual(ts["coverage"], 100.0)
        self.assertEqual(res["overall_score"], ts["subspan_em"])

    def test_partial_coverage_normalized(self):
        res = _score([
            {"task": "quest_32k", "answers": ["spain", "france"],
             "predicted_answer": "Final Answer: ['Spain', 'Xyz']"},
        ])
        ts = res["task_scores"]["quest_32k"]
        self.assertEqual(ts["coverage"], 50.0)  # spain matches (case-normalized), xyz/france miss
        self.assertEqual(ts["em"], 0.0)         # sets differ


if __name__ == "__main__":
    unittest.main()
