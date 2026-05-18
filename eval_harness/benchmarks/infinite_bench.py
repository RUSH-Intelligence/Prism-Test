from __future__ import annotations

from typing import Dict, List

import pandas as pd

from .base import Benchmark, BenchmarkInfo
from .common import parse_answers, substring_match_any
from .registry import register_benchmark


INFINITE_BENCH_SUBSETS = [
    "passkey",
    "kv_retrieval",
    "number_string",
    "code_run",
    "code_debug",
    "math_find",
    "longbook_qa_eng",
    "longdialogue_qa_eng",
    "longbook_choice_eng",
]


@register_benchmark("infinite_bench")
class InfiniteBenchBenchmark(Benchmark):
    @property
    def info(self) -> BenchmarkInfo:
        return BenchmarkInfo(
            name="infinite_bench",
            description="InfiniteBench long-context benchmark",
            default_subsets=INFINITE_BENCH_SUBSETS,
        )

    def load(self, subsets: List[str] | None = None) -> pd.DataFrame:
        subsets = self.resolve_subsets(subsets)
        frames: List[pd.DataFrame] = []

        for subset in subsets:
            from datasets import load_dataset

            ds = load_dataset("MaxJeblick/InfiniteBench", subset, split="test")
            sdf = ds.to_pandas()
            sdf["task"] = subset
            if "answer" not in sdf.columns and "answers" in sdf.columns:
                sdf["answer"] = sdf["answers"]
            if "answer_prefix" not in sdf.columns:
                sdf["answer_prefix"] = ""
            if "max_new_tokens" not in sdf.columns:
                sdf["max_new_tokens"] = 64
            frames.append(sdf)

        return pd.concat(frames, ignore_index=True)

    def score(self, df: pd.DataFrame) -> Dict[str, object]:
        if len(df) == 0:
            return {"overall_score": 0.0, "task_scores": {}, "total_samples": 0}

        task_scores: Dict[str, float] = {}
        for task, tdf in df.groupby("task"):
            vals = []
            for _, row in tdf.iterrows():
                refs = parse_answers(row.get("answer", row.get("answers", [])))
                vals.append(substring_match_any(row.get("predicted_answer", ""), refs))
            task_scores[str(task)] = round((sum(vals) / len(vals)) * 100, 2) if vals else 0.0

        overall = sum(task_scores.values()) / len(task_scores) if task_scores else 0.0
        return {
            "overall_score": round(overall, 2),
            "task_scores": {k: {"string_match": v} for k, v in task_scores.items()},
            "total_samples": int(len(df)),
        }
