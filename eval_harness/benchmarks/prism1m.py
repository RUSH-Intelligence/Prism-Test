from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from .base import Benchmark, BenchmarkInfo
from .common import parse_answers, substring_match_any
from .registry import register_benchmark


PRISM1M_DEFAULT_PATH = "/scratch/sj157/Prism-Test/datasets/Prism-Data/1M/qa_1.jsonl"


@register_benchmark("prism1m")
class Prism1MBenchmark(Benchmark):
    @property
    def info(self) -> BenchmarkInfo:
        return BenchmarkInfo(
            name="prism1m",
            description="Local Prism 1M-context JSONL QA benchmark",
            default_subsets=[PRISM1M_DEFAULT_PATH],
        )

    def load(self, subsets: List[str] | None = None) -> pd.DataFrame:
        subsets = self.resolve_subsets(subsets)
        frames: List[pd.DataFrame] = []

        for subset in subsets:
            jsonl_path = Path(subset)
            if not jsonl_path.exists() or not jsonl_path.is_file() or jsonl_path.suffix != ".jsonl":
                raise ValueError(f"prism1m expects a local .jsonl file path, got: {subset}")

            # Read local jsonl directly to avoid HF datasets cache writes.
            sdf = pd.read_json(jsonl_path, lines=True)

            # Support legacy/raw Prism files that use input/outputs instead of
            # context/question/answer.
            if "context" not in sdf.columns and "input" in sdf.columns:
                sdf["context"] = sdf["input"]
            if "question" not in sdf.columns:
                sdf["question"] = ""
            if "answer" not in sdf.columns and "outputs" in sdf.columns:
                sdf["answer"] = sdf["outputs"]

            for col in ["context", "question", "answer"]:
                if col not in sdf.columns:
                    raise ValueError(f"prism1m dataset is missing required column: {col}")

            if "task" not in sdf.columns:
                sdf["task"] = jsonl_path.stem
            if "context_length" not in sdf.columns:
                sdf["context_length"] = 1_000_000
            if "answer_prefix" not in sdf.columns:
                sdf["answer_prefix"] = ""
            if "max_new_tokens" not in sdf.columns:
                sdf["max_new_tokens"] = 512

            frames.append(sdf)

        return pd.concat(frames, ignore_index=True)

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
