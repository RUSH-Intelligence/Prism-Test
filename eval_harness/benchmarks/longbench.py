from __future__ import annotations

from typing import Dict, List

import pandas as pd

from .base import Benchmark, BenchmarkInfo
from .common import parse_answers, token_f1_any
from .registry import register_benchmark


LONG_BENCH_SUBSETS = [
    "narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique",
    "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht",
    "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p",
    "qasper_e", "multifieldqa_en_e", "hotpotqa_e", "2wikimqa_e", "gov_report_e", "multi_news_e",
    "trec_e", "triviaqa_e", "samsum_e", "passage_count_e", "passage_retrieval_en_e", "lcc_e", "repobench-p_e",
]


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
            frames.append(sdf)
        return pd.concat(frames, ignore_index=True)

    _LENGTH_BUCKETS = [(0, 4000, "0-4k"), (4000, 8000, "4-8k"), (8000, float("inf"), "8k+")]

    def _score_rows(self, rows: pd.DataFrame) -> float:
        vals = [
            token_f1_any(row.get("predicted_answer", ""), parse_answers(row.get("answers", row.get("answer", []))))
            for _, row in rows.iterrows()
        ]
        return round((sum(vals) / len(vals)) * 100, 2) if vals else 0.0

    def score(self, df: pd.DataFrame) -> Dict[str, object]:
        if len(df) == 0:
            return {"overall_score": 0.0, "task_scores": {}, "total_samples": 0}

        task_scores: Dict[str, float] = {}
        task_scores_by_length: Dict[str, Dict[str, float]] = {}

        for task, tdf in df.groupby("task"):
            task = str(task)
            task_scores[task] = self._score_rows(tdf)

            if "length" in tdf.columns:
                bucket_scores: Dict[str, float] = {}
                for lo, hi, label in self._LENGTH_BUCKETS:
                    subset = tdf[(tdf["length"] > lo) & (tdf["length"] <= hi)]
                    if len(subset) > 0:
                        bucket_scores[label] = self._score_rows(subset)
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
