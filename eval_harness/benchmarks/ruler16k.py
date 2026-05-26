from __future__ import annotations

import re
from typing import Dict, List

import pandas as pd

from .base import Benchmark, BenchmarkInfo
from .common import parse_answers
from .registry import register_benchmark


RULER_TASK_SUBSETS = [
    "cwe",
    "fwe",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multiquery",
    "niah_multivalue",
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "qa_1",
    "qa_2",
    "vt",
]


@register_benchmark("ruler16k")
class Ruler16KBenchmark(Benchmark):
    @property
    def info(self) -> BenchmarkInfo:
        return BenchmarkInfo(
            name="ruler16k",
            description="Ruler 16k benchmark",
            default_subsets=RULER_TASK_SUBSETS,
        )

    def load(self, subsets: List[str] | None = None) -> pd.DataFrame:
        subsets = self.resolve_subsets(subsets)
        frames: List[pd.DataFrame] = []

        for subset in subsets:
            from datasets import load_dataset

            ds = load_dataset("xAlg-AI/att-hub-ruler-16k", subset, split=subset)
            sdf = ds.to_pandas()
            if "task" not in sdf.columns:
                sdf["task"] = subset
            if "context_length" not in sdf.columns:
                sdf["context_length"] = 16384
            if "answer_prefix" not in sdf.columns:
                sdf["answer_prefix"] = ""
            if "max_new_tokens" not in sdf.columns:
                sdf["max_new_tokens"] = 64
            frames.append(sdf)

        return pd.concat(frames, ignore_index=True)

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
