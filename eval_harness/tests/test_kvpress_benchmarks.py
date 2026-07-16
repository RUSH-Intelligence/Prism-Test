"""Tests for the kvpress-evaluation benchmark ports: math500, aime25, needle_in_haystack.

Differential parity: the reference scorers are loaded straight from the kvpress
checkout (``/scratch/sj157/kvpress/evaluation/benchmarks/*/calculate_metrics.py``)
and the ported extract/score functions are asserted equal to them on a synthetic
battery. Quirks pinned on purpose (they are faithful ports):
  * math500 ``extract_boxed_first``: FIRST ``boxed{``, cut at FIRST ``}`` — no
    brace balancing (``\\boxed{\\frac{14}{3}}`` -> ``\\frac{14``), no
    normalization (``boxed{ 7}`` -> ``" 7"`` != ``"7"``), None when no box;
  * aime25 ``extract_boxed_last``: LAST ``boxed{`` — ``split("boxed{")[-1]``
    NEVER raises, so a box-free prediction "extracts" itself up to its first
    ``}`` (a non-None string that just fails the equality check);
  * needle: pltrdy-rouge with the reference's swapped argument order.

No network, no model loading: ``object.__new__(Benchmark)`` everywhere, and the
loader tests ``mock.patch`` each module's ``load_dataset`` symbol.
"""
from __future__ import annotations

import importlib.util
import os
import unittest
from unittest import mock

import pandas as pd

from eval_harness.benchmarks.aime25 import (
    AIME25_DATASET,
    Aime25Benchmark,
    extract_boxed_last,
)
from eval_harness.benchmarks.math500 import (
    MATH500_DATASET,
    Math500Benchmark,
    extract_boxed_first,
)
from eval_harness.benchmarks.needle_in_haystack import (
    CONTEXT_WRAPPER,
    DEFAULT_DEPTHS,
    NeedleInHaystackBenchmark,
    insert_needle_at_depth,
)

_HAS_ROUGE = importlib.util.find_spec("rouge") is not None

_KVPRESS_BENCH_DIR = "/scratch/sj157/kvpress/evaluation/benchmarks"
_HAS_KVPRESS_REF = os.path.isfile(
    os.path.join(_KVPRESS_BENCH_DIR, "math500", "calculate_metrics.py")
)


def _load_reference(benchmark: str):
    """Import a kvpress calculate_metrics.py under a unique module name."""
    path = os.path.join(_KVPRESS_BENCH_DIR, benchmark, "calculate_metrics.py")
    spec = importlib.util.spec_from_file_location(f"kvpress_ref_{benchmark}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_REF_MATH500 = _load_reference("math500") if _HAS_KVPRESS_REF else None
_REF_AIME25 = _load_reference("aime25") if _HAS_KVPRESS_REF else None
# The needle reference imports `rouge` at module top: gate on both.
_REF_NEEDLE = (
    _load_reference("needle_in_haystack") if (_HAS_KVPRESS_REF and _HAS_ROUGE) else None
)


def _score(benchmark_class, df: pd.DataFrame) -> dict:
    bench = object.__new__(benchmark_class)  # bypass __init__: score() uses only df
    return bench.score(df)


# ---------------------------------------------------------------------------
# Extraction battery — covers every case required for differential parity.
# ---------------------------------------------------------------------------
EXTRACTION_BATTERY = [
    "The answer is 42, with no box at all",           # no boxed{ anywhere
    "so the result is \\boxed{7}",                     # one boxed simple integer
    "first \\boxed{3} then more \\boxed{9} end",       # multiple: first vs last differ
    "\\boxed{\\frac{14}{3}}",                          # brace-nested answer
    "boxed{5} leads the string",                       # boxed at string start, no backslash
    "trailing text \\boxed{11}",                       # boxed at string end
    "",                                                # empty prediction
    "boxed{7}",                                        # prediction IS bare boxed{7}
    "boxed{ 7}",                                       # whitespace inside braces
    "答案是 \\boxed{７} です",                          # unicode (fullwidth digit) value
    "no box but a closing } brace",                    # aime fallback truncates at }
    "no box and no closing brace",                     # aime fallback = whole string
    "\\boxed{}",                                       # empty box
    "prefix \\boxed{-14}. suffix",                     # negative integer
]

# (predicted_answer, answer) rows for full score() parity.
SCORE_BATTERY = [
    ("The final answer is \\boxed{7}.", "7"),          # correct for both
    ("I think \\boxed{8}", "7"),                        # answered, wrong
    ("some text, no box", "7"),                         # unanswered
    ("first \\boxed{7} then \\boxed{9}", "7"),          # math500 hit, aime25 miss
    ("\\boxed{\\frac{14}{3}}", "\\frac{14}{3}"),       # nested braces can never match
    ("boxed{ 7}", "7"),                                 # NO normalization -> miss
    ("boxed{7}", 7),                                    # no backslash + int answer -> hit
    ("", "7"),                                          # empty prediction
    ("trailing \\boxed{042}", "42"),                    # no numeric normalization -> miss
]
# Hand-computed truths for SCORE_BATTERY (also cross-checked vs the reference):
#   math500: correct rows 0,3,6 -> 3; aime25: correct rows 0,6 -> 2
#   answered (contains "boxed{") rows 0,1,3,4,5,6,8 -> 7; total 9.
_BATTERY_MATH500_CORRECT = 3
_BATTERY_AIME25_CORRECT = 2
_BATTERY_ANSWERED = 7


def _battery_df() -> pd.DataFrame:
    return pd.DataFrame(SCORE_BATTERY, columns=["predicted_answer", "answer"])


# ---------------------------------------------------------------------------
# 1a. Differential parity — extraction functions vs kvpress reference.
# ---------------------------------------------------------------------------
@unittest.skipUnless(_HAS_KVPRESS_REF, "kvpress reference checkout not available")
class TestExtractionParityWithKvpress(unittest.TestCase):
    def test_math500_extraction_matches_reference_on_battery(self):
        for pred in EXTRACTION_BATTERY:
            with self.subTest(pred=pred):
                self.assertEqual(
                    extract_boxed_first(pred), _REF_MATH500.extract_boxed(pred)
                )

    def test_aime25_extraction_matches_reference_on_battery(self):
        for pred in EXTRACTION_BATTERY:
            with self.subTest(pred=pred):
                self.assertEqual(
                    extract_boxed_last(pred), _REF_AIME25.extract_boxed(pred)
                )

    def test_reference_first_vs_last_disagree_where_expected(self):
        # Sanity that the two reference scorers really differ (guards against
        # accidentally loading the same module twice).
        pred = "first \\boxed{3} then more \\boxed{9} end"
        self.assertEqual(_REF_MATH500.extract_boxed(pred), "3")
        self.assertEqual(_REF_AIME25.extract_boxed(pred), "9")


# ---------------------------------------------------------------------------
# 1b. Differential parity — full score() vs reference calculate_metrics().
# ---------------------------------------------------------------------------
@unittest.skipUnless(_HAS_KVPRESS_REF, "kvpress reference checkout not available")
class TestScoreParityWithKvpress(unittest.TestCase):
    def _assert_parity(self, benchmark_class, ref_module, expected_correct):
        df = _battery_df()
        ref = ref_module.calculate_metrics(df)
        res = _score(benchmark_class, df)
        for key in ("correct", "answered", "accuracy", "total"):
            with self.subTest(key=key):
                self.assertEqual(res[key], ref[key])
        # Lock the absolute values too (guards ref and port drifting together).
        self.assertEqual(ref["correct"], expected_correct)
        self.assertEqual(ref["answered"], _BATTERY_ANSWERED)
        self.assertEqual(ref["total"], len(SCORE_BATTERY))
        self.assertEqual(res["overall_score"], round(100.0 * ref["accuracy"], 2))

    def test_math500_score_matches_reference(self):
        self._assert_parity(Math500Benchmark, _REF_MATH500, _BATTERY_MATH500_CORRECT)

    def test_aime25_score_matches_reference(self):
        self._assert_parity(Aime25Benchmark, _REF_AIME25, _BATTERY_AIME25_CORRECT)

    def test_first_vs_last_divergence_on_single_row(self):
        # The one behavioral difference between the two benchmarks.
        df = pd.DataFrame(
            [{"predicted_answer": "steps \\boxed{9} final \\boxed{7}", "answer": "7"}]
        )
        self.assertEqual(_REF_MATH500.calculate_metrics(df)["correct"], 0)
        self.assertEqual(_REF_AIME25.calculate_metrics(df)["correct"], 1)
        self.assertEqual(_score(Math500Benchmark, df)["correct"], 0)
        self.assertEqual(_score(Aime25Benchmark, df)["correct"], 1)


# ---------------------------------------------------------------------------
# Pinned extraction quirks (independent of the reference checkout).
# ---------------------------------------------------------------------------
class TestExtractionQuirks(unittest.TestCase):
    def test_no_boxed_math500_returns_none(self):
        self.assertIsNone(extract_boxed_first("The answer is 42, with no box at all"))
        self.assertIsNone(extract_boxed_first(""))

    def test_no_boxed_aime25_returns_non_none_string(self):
        # split("boxed{")[-1] never raises: box-free predictions "extract" as
        # themselves truncated at the first "}" (then fail equality).
        self.assertEqual(
            extract_boxed_last("no box but a closing } brace"), "no box but a closing "
        )
        self.assertEqual(
            extract_boxed_last("no box and no closing brace"),
            "no box and no closing brace",
        )
        self.assertEqual(extract_boxed_last(""), "")  # empty in, empty out, no raise

    def test_aime25_never_raises_on_battery(self):
        for pred in EXTRACTION_BATTERY:
            with self.subTest(pred=pred):
                self.assertIsInstance(extract_boxed_last(pred), str)

    def test_single_boxed_integer(self):
        self.assertEqual(extract_boxed_first("so the result is \\boxed{7}"), "7")
        self.assertEqual(extract_boxed_last("so the result is \\boxed{7}"), "7")

    def test_multiple_boxed_first_vs_last(self):
        pred = "first \\boxed{3} then more \\boxed{9} end"
        self.assertEqual(extract_boxed_first(pred), "3")
        self.assertEqual(extract_boxed_last(pred), "9")

    def test_brace_nested_extracts_truncated_fragment(self):
        # No brace balancing: cut at the FIRST "}".
        self.assertEqual(extract_boxed_first("\\boxed{\\frac{14}{3}}"), "\\frac{14")
        self.assertEqual(extract_boxed_last("\\boxed{\\frac{14}{3}}"), "\\frac{14")

    def test_boxed_at_string_start_and_end(self):
        self.assertEqual(extract_boxed_first("boxed{5} leads the string"), "5")
        self.assertEqual(extract_boxed_first("trailing text \\boxed{11}"), "11")
        self.assertEqual(extract_boxed_last("trailing text \\boxed{11}"), "11")

    def test_bare_boxed_without_backslash_matches(self):
        # "boxed{" is the split token — the backslash is not required.
        self.assertEqual(extract_boxed_first("boxed{7}"), "7")
        self.assertEqual(extract_boxed_last("boxed{7}"), "7")

    def test_no_whitespace_or_unicode_normalization(self):
        self.assertEqual(extract_boxed_first("boxed{ 7}"), " 7")
        self.assertNotEqual(extract_boxed_first("boxed{ 7}"), "7")
        self.assertEqual(extract_boxed_first("答案是 \\boxed{７} です"), "７")
        self.assertNotEqual(extract_boxed_first("答案是 \\boxed{７} です"), "7")

    def test_empty_box(self):
        self.assertEqual(extract_boxed_first("\\boxed{}"), "")
        self.assertEqual(extract_boxed_last("\\boxed{}"), "")


# ---------------------------------------------------------------------------
# 2 + 3. Scorer edge cases and metrics shape (both boxed benchmarks).
# ---------------------------------------------------------------------------
class TestBoxedScorerEdgeCases(unittest.TestCase):
    BENCHMARKS = ((Math500Benchmark, "math500"), (Aime25Benchmark, "aime25"))

    def test_none_predicted_answer_does_not_crash(self):
        df = pd.DataFrame([{"predicted_answer": None, "answer": "7"}])
        self.assertIsNone(df.iloc[0]["predicted_answer"])  # object dtype keeps None
        for benchmark_class, _ in self.BENCHMARKS:
            with self.subTest(benchmark=benchmark_class.__name__):
                res = _score(benchmark_class, df)
                self.assertEqual(res["correct"], 0)
                self.assertEqual(res["answered"], 0)
                self.assertEqual(res["total"], 1)

    def test_missing_predicted_answer_column_does_not_crash(self):
        df = pd.DataFrame({"answer": ["7", "8"]})
        for benchmark_class, _ in self.BENCHMARKS:
            with self.subTest(benchmark=benchmark_class.__name__):
                res = _score(benchmark_class, df)
                self.assertEqual(res["correct"], 0)
                self.assertEqual(res["answered"], 0)
                self.assertEqual(res["total"], 2)
                self.assertEqual(res["overall_score"], 0.0)

    def test_empty_df_returns_zero_metrics_dict(self):
        expected = {
            "overall_score": 0.0,
            "correct": 0,
            "answered": 0,
            "accuracy": 0.0,
            "total": 0,
            "task_scores": {},
        }
        for benchmark_class, _ in self.BENCHMARKS:
            with self.subTest(benchmark=benchmark_class.__name__):
                self.assertEqual(_score(benchmark_class, pd.DataFrame()), expected)

    def test_metrics_shape_and_types(self):
        df = _battery_df()
        for benchmark_class, task_name in self.BENCHMARKS:
            with self.subTest(benchmark=benchmark_class.__name__):
                res = _score(benchmark_class, df)
                # kvpress-verbatim keys, correct types (int not bool).
                self.assertIsInstance(res["correct"], int)
                self.assertNotIsInstance(res["correct"], bool)
                self.assertIsInstance(res["answered"], int)
                self.assertNotIsInstance(res["answered"], bool)
                self.assertIsInstance(res["accuracy"], float)
                self.assertIsInstance(res["total"], int)
                # Prism aggregates.
                self.assertEqual(
                    res["overall_score"], round(100.0 * res["accuracy"], 2)
                )
                self.assertEqual(
                    res["task_scores"],
                    {task_name: {"accuracy": res["overall_score"]}},
                )
                self.assertEqual(res["total"], len(df))

    def test_battery_absolute_counts(self):
        # Independent of the reference checkout: pin the exact battery counts.
        df = _battery_df()
        res_math = _score(Math500Benchmark, df)
        res_aime = _score(Aime25Benchmark, df)
        self.assertEqual(res_math["correct"], _BATTERY_MATH500_CORRECT)
        self.assertEqual(res_aime["correct"], _BATTERY_AIME25_CORRECT)
        self.assertEqual(res_math["answered"], _BATTERY_ANSWERED)
        self.assertEqual(res_aime["answered"], _BATTERY_ANSWERED)
        self.assertEqual(res_math["overall_score"], round(100.0 * 3 / 9, 2))
        self.assertEqual(res_aime["overall_score"], round(100.0 * 2 / 9, 2))


# ---------------------------------------------------------------------------
# 4. insert_needle_at_depth properties (pure function, no rouge needed).
# ---------------------------------------------------------------------------
_WRAP_PREFIX = "This is a very long story book: <book> "
_WRAP_SUFFIX = " </book>."
NEEDLE = "The best thing to do in San Francisco is eat a sandwich in Dolores Park."


class TestNeedleInsert(unittest.TestCase):
    HAY = " ".join(f"word{i:03d}" for i in range(200))  # 200 words, 1599 chars

    def _content(self, wrapped: str) -> str:
        self.assertTrue(wrapped.startswith(_WRAP_PREFIX))
        self.assertTrue(wrapped.endswith(_WRAP_SUFFIX))
        return wrapped[len(_WRAP_PREFIX) : -len(_WRAP_SUFFIX)]

    def test_wrapper_applied_exactly(self):
        out = insert_needle_at_depth(self.HAY, NEEDLE, 50, 10_000)
        self.assertTrue(out.startswith("This is a very long story book: <book> "))
        self.assertTrue(out.endswith(" </book>."))
        # And the module constant matches the kvpress default wrapper verbatim.
        self.assertEqual(
            CONTEXT_WRAPPER, "This is a very long story book: <book> {context} </book>."
        )

    def test_depth_100_needle_at_end_before_wrapper_suffix(self):
        out = insert_needle_at_depth(self.HAY, NEEDLE, 100, 10_000)
        self.assertEqual(self._content(out), self.HAY + NEEDLE)
        self.assertTrue(out.endswith(NEEDLE + _WRAP_SUFFIX))

    def test_depth_0_word_leading_haystack_snaps_past_first_word(self):
        # Pinned quirk: the insertion index snaps FORWARD to the next
        # whitespace, so with a word-leading haystack depth 0 places the
        # needle glued after the FIRST word — not at the very start (the
        # kvpress reference inserts at raw token index 0, i.e. truly first).
        out = insert_needle_at_depth("alpha beta gamma", NEEDLE, 0, 10_000)
        self.assertEqual(self._content(out), "alpha" + NEEDLE + " beta gamma")

    def test_depth_0_whitespace_leading_haystack_puts_needle_at_start(self):
        out = insert_needle_at_depth(" alpha beta", NEEDLE, 0, 10_000)
        self.assertEqual(self._content(out), NEEDLE + " alpha beta")

    def test_depth_50_single_occurrence_near_midpoint(self):
        out = insert_needle_at_depth(self.HAY, NEEDLE, 50, 10_000)
        content = self._content(out)
        self.assertEqual(content.count(NEEDLE), 1)
        pos = content.find(NEEDLE)
        raw_idx = int(len(self.HAY) * 50 / 100.0)
        max_word_len = max(len(w) for w in self.HAY.split())
        # Forward snap: midpoint <= insertion point <= midpoint + one word.
        self.assertGreaterEqual(pos, raw_idx)
        self.assertLessEqual(pos, raw_idx + max_word_len)
        # Text before the needle is the untouched haystack prefix.
        self.assertEqual(content[:pos], self.HAY[:pos])

    def test_word_boundary_invariant_across_depths(self):
        # The implementation guarantees the char at the insertion index is
        # whitespace (or the index is 0 / end-of-haystack): the needle never
        # lands mid-word — the char FOLLOWING it is whitespace or content end.
        for depth in range(0, 101, 5):
            with self.subTest(depth=depth):
                out = insert_needle_at_depth(self.HAY, NEEDLE, depth, 10_000)
                content = self._content(out)
                self.assertEqual(content.count(NEEDLE), 1)
                pos = content.find(NEEDLE)
                end = pos + len(NEEDLE)
                self.assertTrue(
                    pos == 0 or end == len(content) or content[end].isspace(),
                    f"needle splits a word at depth {depth}: {content[end - 3:end + 3]!r}",
                )

    def test_char_budget_respected(self):
        budget = 100
        out = insert_needle_at_depth(self.HAY, NEEDLE, 50, budget)
        content = self._content(out)
        # Truncation is exact: hay[:budget] plus the needle, nothing more.
        self.assertEqual(len(content), budget + len(NEEDLE))
        self.assertLessEqual(len(content), budget + len(NEEDLE))
        # Content is exactly the truncated haystack with the needle spliced in.
        self.assertEqual(content.replace(NEEDLE, "", 1), self.HAY[:budget])
        # Budget larger than the haystack: full haystack + needle.
        big = insert_needle_at_depth(self.HAY, NEEDLE, 50, 10 * len(self.HAY))
        self.assertEqual(len(self._content(big)), len(self.HAY) + len(NEEDLE))

    def test_zero_and_negative_budget_yield_needle_only(self):
        for budget in (0, -5):
            with self.subTest(budget=budget):
                out = insert_needle_at_depth(self.HAY, NEEDLE, 50, budget)
                self.assertEqual(self._content(out), NEEDLE)


# ---------------------------------------------------------------------------
# 6. _parse_depths.
# ---------------------------------------------------------------------------
class TestParseDepths(unittest.TestCase):
    def test_prefixed_and_bare_forms_parse(self):
        self.assertEqual(
            NeedleInHaystackBenchmark._parse_depths(["depth_50", "50"]), [50, 50]
        )
        self.assertEqual(
            NeedleInHaystackBenchmark._parse_depths(["Depth_10", " depth_0 ", "100"]),
            [10, 0, 100],
        )

    def test_out_of_range_raises_value_error(self):
        for bad in (["depth_101"], ["101"], ["-1"], ["depth_150"]):
            with self.subTest(subset=bad):
                with self.assertRaises(ValueError):
                    NeedleInHaystackBenchmark._parse_depths(bad)


# ---------------------------------------------------------------------------
# 5. Needle score() — rouge-dependent.
# ---------------------------------------------------------------------------
@unittest.skipUnless(_HAS_ROUGE, "needle scoring requires the pltrdy rouge package")
class TestNeedleScore(unittest.TestCase):
    def _df(self, rows):
        return pd.DataFrame(rows)

    def test_perfect_prediction_scores_100(self):
        res = _score(
            NeedleInHaystackBenchmark,
            self._df([{"needle": NEEDLE, "predicted_answer": NEEDLE, "task": "depth_50"}]),
        )
        # pltrdy rouge F is 2pr/(p+r+1e-8): identical text gives 0.999999995,
        # which rounds to a headline of exactly 100.0.
        self.assertAlmostEqual(
            res["kvpress_per_sample"][0]["rouge-l"]["f"], 1.0, places=6
        )
        self.assertEqual(res["overall_score"], 100.0)
        self.assertEqual(res["task_scores"]["depth_50"]["rougeL_f"], 100.0)
        self.assertEqual(res["total_samples"], 1)

    def test_unrelated_prediction_scores_low(self):
        res = _score(
            NeedleInHaystackBenchmark,
            self._df(
                [{"needle": NEEDLE, "predicted_answer": "purple monkey quantum flute",
                  "task": "depth_0"}]
            ),
        )
        self.assertEqual(res["kvpress_per_sample"][0]["rouge-l"]["f"], 0.0)
        self.assertEqual(res["overall_score"], 0.0)

    def test_empty_and_none_predictions_score_zero_without_raising(self):
        for pred in ("", None, "   "):
            with self.subTest(pred=pred):
                res = _score(
                    NeedleInHaystackBenchmark,
                    self._df(
                        [{"needle": NEEDLE, "predicted_answer": pred, "task": "depth_0"}]
                    ),
                )
                self.assertEqual(res["overall_score"], 0.0)
                self.assertEqual(
                    res["kvpress_per_sample"][0]["rouge-l"],
                    {"r": 0.0, "p": 0.0, "f": 0.0},
                )

    def test_none_prediction_takes_zero_path_not_nan_string(self):
        # Regression: pandas>=3 iterrows coerces None to NaN; str(nan) is the
        # literal "nan", which would be rouge-scored against the needle. Use a
        # needle containing "nan" as a token so that path would score > 0.
        res = _score(
            NeedleInHaystackBenchmark,
            self._df(
                [{"needle": "banana nan split", "predicted_answer": None,
                  "task": "depth_0"}]
            ),
        )
        self.assertEqual(
            res["kvpress_per_sample"][0]["rouge-1"], {"r": 0.0, "p": 0.0, "f": 0.0}
        )
        self.assertEqual(res["overall_score"], 0.0)

    def test_mixed_rows_average_and_per_depth_task_scores(self):
        df = self._df(
            [
                {"needle": NEEDLE, "predicted_answer": NEEDLE, "task": "depth_0"},
                {"needle": NEEDLE, "predicted_answer": "", "task": "depth_100"},
            ]
        )
        res = _score(NeedleInHaystackBenchmark, df)
        self.assertEqual(res["overall_score"], 50.0)  # (≈1.0 + 0.0)/2 -> 50.0
        self.assertEqual(set(res["task_scores"]), {"depth_0", "depth_100"})
        for metrics in res["task_scores"].values():
            self.assertEqual(set(metrics), {"rouge1_f", "rouge2_f", "rougeL_f"})
        self.assertEqual(len(res["kvpress_per_sample"]), len(df))

    def test_missing_task_column_defaults_to_needle_key(self):
        res = _score(
            NeedleInHaystackBenchmark,
            self._df([{"needle": NEEDLE, "predicted_answer": NEEDLE}]),
        )
        self.assertEqual(list(res["task_scores"]), ["needle"])

    def test_empty_df_returns_zero_metrics(self):
        res = _score(NeedleInHaystackBenchmark, pd.DataFrame())
        self.assertEqual(
            res, {"overall_score": 0.0, "task_scores": {}, "kvpress_per_sample": []}
        )


@unittest.skipUnless(
    _HAS_ROUGE and _HAS_KVPRESS_REF,
    "needle differential parity requires rouge and the kvpress checkout",
)
class TestNeedleScoreParityWithKvpress(unittest.TestCase):
    def test_per_row_dicts_match_reference(self):
        df = pd.DataFrame(
            [
                {"needle": NEEDLE, "predicted_answer": NEEDLE, "task": "depth_0"},
                {
                    "needle": NEEDLE,
                    "predicted_answer": "The best thing to do in San Francisco is "
                    "to walk across the Golden Gate bridge",
                    "task": "depth_50",
                },
                {
                    "needle": NEEDLE,
                    "predicted_answer": "completely unrelated words entirely",
                    "task": "depth_100",
                },
            ]
        )
        ref_scores = _REF_NEEDLE.calculate_metrics(df)
        res = _score(NeedleInHaystackBenchmark, df)
        # Same rouge lib, same swapped (needle, prediction) argument order.
        self.assertEqual(res["kvpress_per_sample"], ref_scores)
        expected_overall = round(
            100.0 * sum(s["rouge-l"]["f"] for s in ref_scores) / len(ref_scores), 2
        )
        self.assertEqual(res["overall_score"], expected_overall)


# ---------------------------------------------------------------------------
# 7. Loader column mapping — load_dataset patched, no network.
# ---------------------------------------------------------------------------
class TestLoaderColumnMapping(unittest.TestCase):
    def test_math500_injects_defensive_defaults(self):
        raw = pd.DataFrame({"question": ["q1", "q2"], "answer": ["1", "2"]})
        with mock.patch("eval_harness.benchmarks.math500.load_dataset") as ld:
            ld.return_value.to_pandas.return_value = raw
            df = object.__new__(Math500Benchmark).load()
        ld.assert_called_once_with(MATH500_DATASET, split="test")
        self.assertEqual(len(df), 2)
        self.assertTrue((df["context"] == " ").all())
        self.assertTrue((df["answer_prefix"] == "").all())
        self.assertTrue((df["max_new_tokens"] == 4096).all())
        self.assertTrue((df["task"] == "math500").all())
        self.assertEqual(list(df["question"]), ["q1", "q2"])
        self.assertEqual(list(df["answer"]), ["1", "2"])

    def test_math500_does_not_overwrite_shipped_columns(self):
        # Defaults are only-if-absent; task is stamped unconditionally.
        raw = pd.DataFrame(
            {
                "question": ["q1"],
                "answer": ["1"],
                "context": ["shipped"],
                "answer_prefix": ["Answer: "],
                "max_new_tokens": [123],
                "task": ["stale"],
            }
        )
        with mock.patch("eval_harness.benchmarks.math500.load_dataset") as ld:
            ld.return_value.to_pandas.return_value = raw
            df = object.__new__(Math500Benchmark).load()
        self.assertEqual(df.iloc[0]["context"], "shipped")
        self.assertEqual(df.iloc[0]["answer_prefix"], "Answer: ")
        self.assertEqual(df.iloc[0]["max_new_tokens"], 123)
        self.assertEqual(df.iloc[0]["task"], "math500")  # always overwritten

    def test_aime25_injects_defensive_defaults(self):
        raw = pd.DataFrame({"question": ["p1"], "answer": ["70"]})
        with mock.patch("eval_harness.benchmarks.aime25.load_dataset") as ld:
            ld.return_value.to_pandas.return_value = raw
            df = object.__new__(Aime25Benchmark).load()
        ld.assert_called_once_with(AIME25_DATASET, split="test")
        self.assertTrue((df["context"] == " ").all())
        self.assertTrue((df["answer_prefix"] == "").all())
        self.assertTrue((df["max_new_tokens"] == 32000).all())
        self.assertTrue((df["task"] == "aime25").all())

    @staticmethod
    def _needle_seed(**overrides) -> pd.DataFrame:
        row = {
            "context": " ".join(f"word{i:03d}" for i in range(300)),
            "needle": NEEDLE,
            "question": "What is the best thing to do in San Francisco?",
            "answer_prefix": "The best thing to do in San Francisco is",
            "max_new_tokens": 40,
        }
        row.update(overrides)
        return pd.DataFrame([row])

    def test_needle_loader_one_row_per_depth(self):
        seed = self._needle_seed()
        with mock.patch(
            "eval_harness.benchmarks.needle_in_haystack.load_dataset"
        ) as ld:
            ld.return_value.to_pandas.return_value = seed
            df = object.__new__(NeedleInHaystackBenchmark).load()
        ld.assert_called_once_with("alessiodevoto/paul_graham_essays", split="test")
        self.assertEqual(len(df), len(DEFAULT_DEPTHS))
        self.assertEqual(list(df["task"]), [f"depth_{d}" for d in DEFAULT_DEPTHS])
        self.assertEqual(list(df["needle_depth"]), DEFAULT_DEPTHS)
        self.assertTrue((df["answer"] == NEEDLE).all())
        self.assertTrue((df["needle"] == NEEDLE).all())
        self.assertTrue((df["max_new_tokens"] == 40).all())
        self.assertTrue(
            (df["question"] == "What is the best thing to do in San Francisco?").all()
        )
        for context in df["context"]:
            self.assertEqual(context.count(NEEDLE), 1)
            self.assertTrue(context.startswith(_WRAP_PREFIX))
            self.assertTrue(context.endswith(_WRAP_SUFFIX))

    def test_needle_loader_subset_selection(self):
        with mock.patch(
            "eval_harness.benchmarks.needle_in_haystack.load_dataset"
        ) as ld:
            ld.return_value.to_pandas.return_value = self._needle_seed()
            df = object.__new__(NeedleInHaystackBenchmark).load(["depth_50"])
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["task"], "depth_50")
        self.assertEqual(df.iloc[0]["needle_depth"], 50)

    def test_needle_loader_defaults_max_new_tokens_to_40(self):
        seed = self._needle_seed().drop(columns=["max_new_tokens"])
        with mock.patch(
            "eval_harness.benchmarks.needle_in_haystack.load_dataset"
        ) as ld:
            ld.return_value.to_pandas.return_value = seed
            df = object.__new__(NeedleInHaystackBenchmark).load(["depth_0"])
        self.assertEqual(df.iloc[0]["max_new_tokens"], 40)


# ---------------------------------------------------------------------------
# 8. Registry resolution (import-time only; load() never called).
# ---------------------------------------------------------------------------
class TestRegistryResolution(unittest.TestCase):
    def test_new_benchmarks_resolve(self):
        from eval_harness.benchmarks.registry import get_benchmark

        self.assertIsInstance(get_benchmark("math500"), Math500Benchmark)
        self.assertIsInstance(get_benchmark("aime25"), Aime25Benchmark)
        self.assertIsInstance(
            get_benchmark("needle_in_haystack"), NeedleInHaystackBenchmark
        )
        self.assertIsInstance(get_benchmark("niah_pg"), NeedleInHaystackBenchmark)


if __name__ == "__main__":
    unittest.main()
