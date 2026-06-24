"""Unit tests for LongBench / LongBench-v2 scoring.

Validates the official per-task metric routing (token-F1, classification,
retrieval, count, code similarity, first-line trimming) and the LongBench-v2
answer extraction. ROUGE / Chinese (jieba) paths are exercised only when those
optional libraries are installed.
"""
from __future__ import annotations

import importlib.util
import unittest

import pandas as pd

from eval_harness.benchmarks.common import (
    extract_longbench_v2_answer,
    extract_option_letter,
)
from eval_harness.benchmarks.longbench import LongBenchBenchmark
from eval_harness.benchmarks.longbench_metrics import (
    classification_score,
    code_sim_score,
    count_score,
    metric_for_task,
    qa_f1_score,
    retrieval_score,
)
from eval_harness.benchmarks.longbenchv2 import LongBenchV2Benchmark

_HAS_ROUGE = importlib.util.find_spec("rouge") is not None
_HAS_JIEBA = importlib.util.find_spec("jieba") is not None


class TestMetricRouting(unittest.TestCase):
    def test_each_subset_routes_to_a_metric(self):
        for task in [
            "narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh",
            "hotpotqa", "2wikimqa", "musique", "dureader", "gov_report",
            "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum",
            "lsht", "passage_count", "passage_retrieval_en",
            "passage_retrieval_zh", "lcc", "repobench-p",
        ]:
            self.assertIsNotNone(metric_for_task(task), task)

    def test_e_suffix_routes_like_base_task(self):
        self.assertIs(metric_for_task("gov_report_e"), metric_for_task("gov_report"))
        self.assertIs(metric_for_task("trec_e"), metric_for_task("trec"))


class TestIndividualMetrics(unittest.TestCase):
    def test_qa_f1_exact_match(self):
        self.assertAlmostEqual(qa_f1_score("the Eiffel Tower", "Eiffel Tower"), 1.0)

    def test_qa_f1_partial(self):
        # pred tokens {paris france}, gt tokens {paris}: P=1/2, R=1 -> F1=2/3
        self.assertAlmostEqual(qa_f1_score("paris france", "paris"), 2 / 3)

    def test_qa_f1_no_overlap(self):
        self.assertEqual(qa_f1_score("dog", "cat"), 0.0)

    def test_count_score(self):
        self.assertEqual(count_score("the answer is 5", "5"), 1.0)
        self.assertEqual(count_score("maybe 3 or 5", "5"), 0.5)
        self.assertEqual(count_score("no numbers", "5"), 0.0)

    def test_retrieval_score(self):
        self.assertEqual(retrieval_score("Paragraph 7", "Paragraph 7 is correct"), 1.0)
        self.assertEqual(retrieval_score("It is 3", "Paragraph 7 is correct"), 0.0)

    def test_classification_score(self):
        classes = ["Sports", "Politics", "Technology"]
        self.assertEqual(
            classification_score("This is Politics", "Politics", all_classes=classes), 1.0
        )
        # Two classes matched -> 1/2 credit.
        self.assertEqual(
            classification_score("Politics and Sports", "Politics", all_classes=classes), 0.5
        )
        self.assertEqual(
            classification_score("Technology", "Politics", all_classes=classes), 0.0
        )

    def test_code_sim_identical(self):
        self.assertEqual(code_sim_score("return x + 1", "return x + 1"), 1.0)

    def test_code_sim_skips_comment_lines(self):
        # First non-comment line is compared.
        score = code_sim_score("# a comment\nreturn x + 1", "return x + 1")
        self.assertEqual(score, 1.0)


class TestLongBenchScorer(unittest.TestCase):
    def _bench(self) -> LongBenchBenchmark:
        return LongBenchBenchmark()

    def test_first_line_trim_for_classification(self):
        # trec uses classification_score AND first-line trimming. The bogus
        # second line must be ignored.
        df = pd.DataFrame([
            {
                "task": "trec",
                "predicted_answer": "Sports\nPolitics Technology",
                "answers": ["Sports"],
                "all_classes": ["Sports", "Politics", "Technology"],
            }
        ])
        out = self._bench().score(df)
        self.assertEqual(out["task_scores"]["trec"], 100.0)

    def test_mixed_tasks_use_distinct_metrics(self):
        df = pd.DataFrame([
            {"task": "passage_count", "predicted_answer": "5", "answers": ["5"], "all_classes": None},
            {"task": "hotpotqa", "predicted_answer": "Barack Obama", "answers": ["Barack Obama"], "all_classes": None},
        ])
        out = self._bench().score(df)
        self.assertEqual(out["task_scores"]["passage_count"], 100.0)
        self.assertEqual(out["task_scores"]["hotpotqa"], 100.0)
        self.assertEqual(out["total_samples"], 2)

    def test_length_buckets_only_for_e_subsets(self):
        df = pd.DataFrame([
            {"task": "hotpotqa_e", "predicted_answer": "x", "answers": ["x"], "all_classes": None, "length": 1000},
            {"task": "hotpotqa", "predicted_answer": "x", "answers": ["x"], "all_classes": None, "length": 1000},
        ])
        out = self._bench().score(df)
        self.assertIn("task_scores_by_length", out)
        self.assertIn("hotpotqa_e", out["task_scores_by_length"])
        self.assertNotIn("hotpotqa", out["task_scores_by_length"])

    def test_empty(self):
        out = self._bench().score(pd.DataFrame())
        self.assertEqual(out["overall_score"], 0.0)
        self.assertEqual(out["total_samples"], 0)

    def test_code_task_preserves_double_asterisk(self):
        # lcc / repobench-p use code-similarity scoring against unstripped gold.
        # A blanket ``**`` strip on the prediction would corrupt valid Python
        # (``x ** 2``) and silently depress code-task scores.
        df = pd.DataFrame([
            {"task": "lcc", "predicted_answer": "return x ** 2",
             "answers": ["return x ** 2"], "all_classes": None},
            {"task": "repobench-p", "predicted_answer": "y ** 0.5",
             "answers": ["y ** 0.5"], "all_classes": None},
        ])
        out = self._bench().score(df)
        self.assertEqual(out["task_scores"]["lcc"], 100.0)
        self.assertEqual(out["task_scores"]["repobench-p"], 100.0)

    def test_prose_task_still_strips_double_asterisk(self):
        # Prose tasks (token-overlap / ROUGE) should still strip ``**`` so
        # Mistral's markdown-bold wrapper doesn't tank F1 against plain gold.
        df = pd.DataFrame([
            {"task": "hotpotqa", "predicted_answer": "**Barack Obama**",
             "answers": ["Barack Obama"], "all_classes": None},
        ])
        out = self._bench().score(df)
        self.assertEqual(out["task_scores"]["hotpotqa"], 100.0)

    @unittest.skipUnless(_HAS_ROUGE, "rouge not installed")
    def test_rouge_task_runs(self):
        df = pd.DataFrame([
            {"task": "gov_report", "predicted_answer": "the cat sat on the mat",
             "answers": ["the cat sat on the mat"], "all_classes": None},
        ])
        out = self._bench().score(df)
        self.assertEqual(out["task_scores"]["gov_report"], 100.0)

    @unittest.skipUnless(_HAS_JIEBA and _HAS_ROUGE, "jieba/rouge not installed")
    def test_chinese_task_runs(self):
        df = pd.DataFrame([
            {"task": "multifieldqa_zh", "predicted_answer": "北京是中国的首都",
             "answers": ["北京是中国的首都"], "all_classes": None},
        ])
        out = self._bench().score(df)
        self.assertEqual(out["task_scores"]["multifieldqa_zh"], 100.0)


class TestLongBenchV2Extraction(unittest.TestCase):
    def test_official_template_parenthesized(self):
        self.assertEqual(
            extract_longbench_v2_answer("The correct answer is (C)."), "C"
        )

    def test_official_template_bare(self):
        self.assertEqual(
            extract_longbench_v2_answer("The correct answer is B"), "B"
        )

    def test_strips_markdown_emphasis(self):
        self.assertEqual(
            extract_longbench_v2_answer("**The correct answer is (A)**"), "A"
        )

    def test_cot_picks_final_not_first(self):
        # Reasoning mentions A and C, final answer is D. The official template
        # wins over stray letters in the reasoning.
        resp = "Option A seems plausible, C is wrong. The correct answer is (D)."
        self.assertEqual(extract_longbench_v2_answer(resp), "D")

    def test_fallback_last_letter(self):
        # No template -> fall back to last standalone letter.
        self.assertEqual(extract_longbench_v2_answer("I think it is B"), "B")

    def test_no_letter(self):
        self.assertEqual(extract_longbench_v2_answer("no idea"), "")


class TestLongBenchV2Scorer(unittest.TestCase):
    def test_accuracy(self):
        df = pd.DataFrame([
            {"task": "0shot", "predicted_answer": "The correct answer is (A)", "answer": "A"},
            {"task": "0shot", "predicted_answer": "The correct answer is (B)", "answer": "C"},
        ])
        out = LongBenchV2Benchmark().score(df)
        self.assertEqual(out["task_scores"]["0shot"]["accuracy"], 50.0)
        self.assertEqual(out["total_samples"], 2)


if __name__ == "__main__":
    unittest.main()
