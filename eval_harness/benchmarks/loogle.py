from __future__ import annotations

from typing import Dict, List

import pandas as pd

from .base import Benchmark, BenchmarkInfo
from .common import exact_match_any, parse_answers, token_f1_any
from .registry import register_benchmark


LOOGLE_SUBSETS = [
    "shortdep_qa",
    "longdep_qa",
    "shortdep_cloze",
    "longdep_summarization",
]


@register_benchmark("loogle")
class LoogleBenchmark(Benchmark):
    @property
    def info(self) -> BenchmarkInfo:
        return BenchmarkInfo(
            name="loogle",
            description="Loogle long-context benchmark",
            default_subsets=LOOGLE_SUBSETS,
        )

    def load(self, subsets: List[str] | None = None) -> pd.DataFrame:
        subsets = self.resolve_subsets(subsets)
        frames: List[pd.DataFrame] = []

        for subset in subsets:
            from datasets import load_dataset

            ds = load_dataset("simonjegou/loogle", subset, split="test")
            sdf = ds.to_pandas()
            sdf["task"] = subset
            if "answer_prefix" not in sdf.columns:
                sdf["answer_prefix"] = ""
            if "max_new_tokens" not in sdf.columns:
                sdf["max_new_tokens"] = 128
            frames.append(sdf)

        return pd.concat(frames, ignore_index=True)

    def score(self, df: pd.DataFrame) -> Dict[str, object]:
        if len(df) == 0:
            return {"overall_score": 0.0, "task_scores": {}, "total_samples": 0}

        task_scores: Dict[str, Dict[str, float]] = {}
        overall_vals: List[float] = []

        for task, tdf in df.groupby("task"):
            vals: List[float] = []
            for _, row in tdf.iterrows():
                refs = parse_answers(row.get("answer", row.get("answers", [])))
                pred = row.get("predicted_answer", "")
                if str(task) == "shortdep_cloze":
                    vals.append(exact_match_any(pred, refs))
                else:
                    vals.append(token_f1_any(pred, refs))

            metric_name = "exact_match" if str(task) == "shortdep_cloze" else "token_f1"
            score = (sum(vals) / len(vals) * 100) if vals else 0.0
            task_scores[str(task)] = {metric_name: round(score, 2)}
            overall_vals.append(score)

        overall = sum(overall_vals) / len(overall_vals) if overall_vals else 0.0
        return {
            "overall_score": round(overall, 2),
            "task_scores": task_scores,
            "total_samples": int(len(df)),
        }
