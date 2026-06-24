from __future__ import annotations

from typing import Dict, List

import pandas as pd

from .base import Benchmark, BenchmarkInfo
from .common import parse_answers
from .longbench_metrics import (
    FIRST_LINE_TASKS,
    base_task_name,
    metric_for_task,
)
from .registry import register_benchmark


LONG_BENCH_SUBSETS = [
    "narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique",
    "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht",
    "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p",
    "qasper_e", "multifieldqa_en_e", "hotpotqa_e", "2wikimqa_e", "gov_report_e", "multi_news_e",
    "trec_e", "triviaqa_e", "samsum_e", "passage_count_e", "passage_retrieval_en_e", "lcc_e", "repobench-p_e",
]

# Official LongBench pred.py skips chat-template wrapping for these few-shot /
# completion-style datasets: "chat models are better off without build prompts
# on these tasks." Everything else gets the chat wrapper. The runner reads the
# per-row `use_chat_template` column populated by `load` and overrides the
# adapter's config-level default per generate_for_context call.
CHAT_TEMPLATE_SKIP_TASKS = frozenset({
    "trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p",
})

# Code-completion tasks scored by code-similarity (difflib/fuzzywuzzy). Stripping
# ``**`` from predictions would corrupt legitimate Python (e.g. ``x ** 2``) since
# gold answers are not stripped — the markdown-bold strip only helps the
# token-overlap and ROUGE-L tasks (Mistral wraps prose answers in ``**...**``).
_CODE_TASKS = frozenset({"lcc", "repobench-p"})


@register_benchmark("longbench")
class LongBenchBenchmark(Benchmark):
    @property
    def info(self) -> BenchmarkInfo:
        return BenchmarkInfo(
            name="longbench",
            description="LongBench multi-task long-context benchmark",
            default_subsets=LONG_BENCH_SUBSETS,
        )

    def load(self, subsets: List[str] | None = None) -> pd.DataFrame:
        subsets = self.resolve_subsets(subsets)
        frames: List[pd.DataFrame] = []
        for subset in subsets:
            from datasets import load_dataset

            ds = load_dataset("Xnhyacinth/LongBench", subset, split="test")
            sdf = ds.to_pandas()
            sdf["task"] = subset
            if "answer_prefix" not in sdf.columns:
                sdf["answer_prefix"] = ""
            if "max_new_tokens" not in sdf.columns:
                sdf["max_new_tokens"] = 128
            sdf["use_chat_template"] = base_task_name(subset) not in CHAT_TEMPLATE_SKIP_TASKS
            # LongBench parity (official pred.py): when the chat template is
            # applied, the auto-injected system header (Llama "Cutting Knowledge
            # Date" / Mistral "[SYSTEM_PROMPT]") is stripped so prompts match
            # published baselines. No-op on the skip-list tasks (they bypass the
            # chat path entirely).
            sdf["strip_auto_system_block"] = True
            # Official LongBench pred.py middle-truncates (first half + last
            # half) when the context overflows the model window. Opt in
            # per-row so other benchmarks (RULER NIAH) keep head-truncation.
            sdf["middle_truncation"] = True
            frames.append(sdf)
        return pd.concat(frames, ignore_index=True)

    # Upstream LongBench-E length buckets: <4k, 4k-8k, >=8k.
    _LENGTH_BUCKETS = [(0, 4000, "0-4k"), (4000, 8000, "4-8k"), (8000, float("inf"), "8k+")]

    @staticmethod
    def _row_all_classes(row) -> List[str]:
        ac = row.get("all_classes", None)
        if ac is None:
            return []
        # pandas may store this as an ndarray; coerce to a plain list of str.
        if hasattr(ac, "tolist"):
            ac = ac.tolist()
        if isinstance(ac, (list, tuple)):
            return [str(c) for c in ac]
        return []

    def _score_row(self, row) -> float:
        """Official per-task scoring: max over ground truths, first-line trim."""
        task = base_task_name(str(row.get("task", "")))
        metric = metric_for_task(task)
        if metric is None:
            return 0.0

        prediction = str(row.get("predicted_answer", "") or "")
        # Mistral-instruct family wraps prose answers in ``**bold**`` markdown
        # by default. LongBench's prose metrics are token-overlap based, so the
        # asterisks tokenize as junk that lowers F1 vs gold ~1-3 pts/task.
        # Llama-3.1 emits plain text and is unaffected. Strip only the markers
        # — and only on non-code tasks (lcc/repobench-p use code-similarity
        # against unstripped gold, so a strip would corrupt ``x ** 2`` etc.).
        # Predictions.csv keeps the raw model output for transparency.
        if task not in _CODE_TASKS:
            prediction = prediction.replace("**", "")
        if task in FIRST_LINE_TASKS:
            prediction = prediction.lstrip("\n").split("\n")[0]

        ground_truths = parse_answers(row.get("answers", row.get("answer", [])))
        all_classes = self._row_all_classes(row)

        score = 0.0
        for gt in ground_truths:
            score = max(score, metric(prediction, gt, all_classes=all_classes))
        return score

    def _score_rows(self, rows: pd.DataFrame) -> float:
        vals = [self._score_row(row) for _, row in rows.iterrows()]
        return round((sum(vals) / len(vals)) * 100, 2) if vals else 0.0

    def _length_bucket_scores(self, tdf: pd.DataFrame) -> Dict[str, float]:
        bucket_scores: Dict[str, float] = {}
        for lo, hi, label in self._LENGTH_BUCKETS:
            subset = tdf[(tdf["length"] >= lo) & (tdf["length"] < hi)]
            if len(subset) > 0:
                bucket_scores[label] = self._score_rows(subset)
        return bucket_scores

    def score(self, df: pd.DataFrame) -> Dict[str, object]:
        if len(df) == 0:
            return {"overall_score": 0.0, "task_scores": {}, "total_samples": 0}

        task_scores: Dict[str, float] = {}
        task_scores_by_length: Dict[str, Dict[str, float]] = {}

        for task, tdf in df.groupby("task"):
            task = str(task)
            task_scores[task] = self._score_rows(tdf)

            # Length-bucketed reporting is only meaningful for LongBench-E subsets.
            if task.endswith("_e") and "length" in tdf.columns:
                bucket_scores = self._length_bucket_scores(tdf)
                if bucket_scores:
                    task_scores_by_length[task] = bucket_scores

        overall = sum(task_scores.values()) / len(task_scores) if task_scores else 0.0
        result: Dict[str, object] = {
            "overall_score": round(overall, 2),
            "task_scores": task_scores,
            "total_samples": int(len(df)),
        }
        if task_scores_by_length:
            result["task_scores_by_length"] = task_scores_by_length
        return result
