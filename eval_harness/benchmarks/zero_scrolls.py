from __future__ import annotations

from typing import Dict, List

import pandas as pd

from .base import Benchmark, BenchmarkInfo
from .registry import register_benchmark


@register_benchmark("zero_scrolls")
class ZeroScrollsBenchmark(Benchmark):
    @property
    def info(self) -> BenchmarkInfo:
        return BenchmarkInfo(
            name="zero_scrolls",
            description="ZeroScrolls long-document benchmark",
            default_subsets=["default"],
        )

    def load(self, subsets: List[str] | None = None) -> pd.DataFrame:
        subsets = self.resolve_subsets(subsets)
        frames: List[pd.DataFrame] = []

        for subset in subsets:
            from datasets import load_dataset

            ds = load_dataset("simonjegou/zero_scrolls", subset, split="test")
            sdf = ds.to_pandas()
            sdf["context"] = sdf.apply(lambda x: x["input"][: x["document_end_index"]], axis=1)
            sdf["question"] = sdf.apply(
                lambda x: x["input"][x["document_end_index"] : x["query_end_index"]], axis=1
            )
            sdf["answer_prefix"] = (
                sdf.apply(lambda x: x["input"][x["query_end_index"] :], axis=1).astype(str).str.strip()
            )
            sdf["answer"] = ""
            sdf["task"] = subset
            if "max_new_tokens" not in sdf.columns:
                sdf["max_new_tokens"] = 512
            frames.append(sdf)

        return pd.concat(frames, ignore_index=True)

    def score(self, df: pd.DataFrame) -> Dict[str, object]:
        if len(df) == 0:
            return {"overall_score": 0.0, "task_scores": {}, "total_samples": 0}

        task_scores: Dict[str, Dict[str, float]] = {}
        response_lengths: List[float] = []

        for task, tdf in df.groupby("task"):
            avg_len = tdf["predicted_answer"].fillna("").astype(str).str.len().mean()
            response_lengths.append(float(avg_len))
            task_scores[str(task)] = {
                "avg_response_length": round(float(avg_len), 2),
                "num_samples": float(len(tdf)),
            }

        overall_avg = sum(response_lengths) / len(response_lengths) if response_lengths else 0.0
        return {
            "overall_score": 0.0,
            "task_scores": task_scores,
            "avg_response_length": round(overall_avg, 2),
            "total_samples": int(len(df)),
            "note": "ZeroScrolls has no gold answers in this harness, so score is placeholder.",
        }
