from __future__ import annotations

import re
from typing import Dict, List

import pandas as pd

from .base import Benchmark, BenchmarkInfo
from .common import parse_answers
from .registry import register_benchmark
from .ruler64k import RULER_SUBSETS, _collect_rows_for_context_length


@register_benchmark("ruler128k")
class Ruler128KBenchmark(Benchmark):
    @property
    def info(self) -> BenchmarkInfo:
        return BenchmarkInfo(
            name="ruler128k",
            description="RULER 128k benchmark from tonychenxyz/ruler-full",
            default_subsets=RULER_SUBSETS,
        )

    def load(self, subsets: List[str] | None = None) -> pd.DataFrame:
        subsets = self.resolve_subsets(subsets)
        return _collect_rows_for_context_length(subsets, context_length=131072)

    def score(self, df: pd.DataFrame) -> Dict[str, object]:
        if len(df) == 0:
            return {
                "overall_score": 0.0,
                "task_scores": {},
                "context_length_scores": {},
                "summary": {
                    "total_tasks": 0,
                    "total_samples": 0,
                    "context_lengths": [],
                },
            }

        np_pattern = re.compile(r"[\x00-\x1f]")

        def _clean_prediction(text) -> str:
            return np_pattern.sub("", str(text).strip()).strip().lower()

        def _string_match_part(pred: str, refs: List[str]) -> float:
            if not refs:
                return 0.0
            return max(1.0 if r and r in pred else 0.0 for r in refs)

        def _string_match_all(pred: str, refs: List[str]) -> float:
            if not refs:
                return 0.0
            return sum(1.0 if r and r in pred else 0.0 for r in refs) / len(refs)

        task_scores: Dict[str, float] = {}
        for task, tdf in df.groupby("task"):
            task_category = str(task).split("_")[0]
            metric_fn = _string_match_part if task_category == "qa" else _string_match_all
            vals = []
            for _, row in tdf.iterrows():
                pred = _clean_prediction(row.get("predicted_answer", ""))
                refs = [
                    str(r).strip().lower()
                    for r in parse_answers(row.get("answer", row.get("answers", [])))
                ]
                vals.append(metric_fn(pred, refs))
            task_scores[str(task)] = round((sum(vals) / len(vals)) * 100, 2) if vals else 0.0

        overall = sum(task_scores.values()) / len(task_scores) if task_scores else 0.0

        context_length_scores: Dict[str, float] = {}
        if "context_length" in df.columns:
            for context_length, cdf in df.groupby("context_length"):
                c_task_scores: Dict[str, float] = {}
                for task, tdf in cdf.groupby("task"):
                    task_category = str(task).split("_")[0]
                    metric_fn = _string_match_part if task_category == "qa" else _string_match_all
                    vals = []
                    for _, row in tdf.iterrows():
                        pred = _clean_prediction(row.get("predicted_answer", ""))
                        refs = [
                            str(r).strip().lower()
                            for r in parse_answers(row.get("answer", row.get("answers", [])))
                        ]
                        vals.append(metric_fn(pred, refs))
                    c_task_scores[str(task)] = (sum(vals) / len(vals)) * 100 if vals else 0.0
                c_overall = sum(c_task_scores.values()) / len(c_task_scores) if c_task_scores else 0.0
                context_length_scores[str(context_length)] = round(c_overall, 2)

        return {
            "overall_score": round(overall, 2),
            "task_scores": {k: {"string_match": v} for k, v in task_scores.items()},
            "context_length_scores": context_length_scores,
            "summary": {
                "total_tasks": len(task_scores),
                "total_samples": int(len(df)),
                "context_lengths": list(context_length_scores.keys()),
            },
        }
