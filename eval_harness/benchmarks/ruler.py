from __future__ import annotations

from typing import Dict, List

import pandas as pd

from .base import Benchmark, BenchmarkInfo
from .common import parse_answers, substring_match_any
from .registry import register_benchmark


@register_benchmark("ruler")
class RulerBenchmark(Benchmark):
    @property
    def info(self) -> BenchmarkInfo:
        return BenchmarkInfo(
            name="ruler",
            description="Ruler benchmark across context-length subsets",
            default_subsets=["4096", "8192", "16384", "32768"],
        )

    def load(self, subsets: List[str] | None = None) -> pd.DataFrame:
        subsets = self.resolve_subsets(subsets)
        frames: List[pd.DataFrame] = []

        for subset in subsets:
            from datasets import load_dataset

            ds = load_dataset("simonjegou/ruler", subset, split="test")
            sdf = ds.to_pandas()
            if "task" not in sdf.columns:
                sdf["task"] = "ruler"
            sdf["context_length"] = int(subset)
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
