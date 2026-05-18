from __future__ import annotations

from typing import Dict, List

import pandas as pd

from .base import Benchmark, BenchmarkInfo
from .common import parse_answers, substring_match_any
from .registry import register_benchmark


RULER_SUBSETS = [
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


def _collect_rows_for_context_length(subsets: List[str], context_length: int) -> pd.DataFrame:
    from datasets import load_dataset

    wanted = set(subsets)
    rows: List[Dict[str, object]] = []

    for variant in ["plain", "memwrap"]:
        ds = load_dataset("tonychenxyz/ruler-full", variant, split="validation", streaming=True)
        for sample in ds:
            category = str(sample.get("category", ""))
            marker = f"_{context_length}"
            if marker not in category:
                continue

            extra = sample.get("extra_info") or {}
            task = str(extra.get("ruler_task", ""))
            if task not in wanted:
                suffix = category.split("/")[-1]
                if suffix.endswith(marker):
                    task = suffix[: -len(marker)]
            if task not in wanted:
                continue

            ground_truth = extra.get("ground_truth") or {}
            if isinstance(ground_truth, dict):
                answer = ground_truth.get("answers", "")
            else:
                answer = ground_truth
            rows.append(
                {
                    "context": str(sample.get("prompt", "")),
                    "question": "",
                    "answer": answer,
                    "task": task,
                    "context_length": context_length,
                    "answer_prefix": "",
                    "max_new_tokens": 64,
                    "variant": variant,
                    "category": category,
                }
            )

    if not rows:
        raise ValueError(
            f"No rows found in tonychenxyz/ruler-full for context_length={context_length} and subsets={subsets}."
        )

    return pd.DataFrame(rows)


@register_benchmark("ruler64k")
class Ruler64KBenchmark(Benchmark):
    @property
    def info(self) -> BenchmarkInfo:
        return BenchmarkInfo(
            name="ruler64k",
            description="RULER 64k benchmark from tonychenxyz/ruler-full",
            default_subsets=RULER_SUBSETS,
        )

    def load(self, subsets: List[str] | None = None) -> pd.DataFrame:
        subsets = self.resolve_subsets(subsets)
        return _collect_rows_for_context_length(subsets, context_length=65536)

    def score(self, df: pd.DataFrame) -> Dict[str, object]:
        if len(df) == 0:
            return {"overall_score": 0.0, "task_scores": {}, "total_samples": 0}

        task_scores: Dict[str, float] = {}
        for task, tdf in df.groupby("task"):
            vals = []
            for _, row in tdf.iterrows():
                vals.append(substring_match_any(row.get("predicted_answer", ""), parse_answers(row.get("answer", []))))
            task_scores[str(task)] = round((sum(vals) / len(vals)) * 100, 2) if vals else 0.0

        overall = sum(task_scores.values()) / len(task_scores) if task_scores else 0.0
        return {
            "overall_score": round(overall, 2),
            "task_scores": {k: {"string_match": v} for k, v in task_scores.items()},
            "total_samples": int(len(df)),
        }
