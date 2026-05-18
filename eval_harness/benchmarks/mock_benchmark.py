from __future__ import annotations

from typing import Dict, List

import pandas as pd

from .base import Benchmark, BenchmarkInfo
from .common import parse_answers, substring_match_any
from .registry import register_benchmark


@register_benchmark("mock_benchmark")
class MockBenchmark(Benchmark):
    @property
    def info(self) -> BenchmarkInfo:
        return BenchmarkInfo(
            name="mock_benchmark",
            description="Small local benchmark for smoke testing",
            default_subsets=["reading_comprehension"],
        )

    def load(self, subsets: List[str] | None = None) -> pd.DataFrame:
        del subsets
        rows = [
            {
                "context": "The capital of France is Paris.",
                "question": "What is the capital of France?",
                "answer": ["Paris"],
                "task": "reading_comprehension",
                "answer_prefix": "",
                "max_new_tokens": 32,
            },
            {
                "context": "Water freezes at 0 degrees Celsius.",
                "question": "At what temperature does water freeze in Celsius?",
                "answer": ["0", "0 degrees Celsius"],
                "task": "reading_comprehension",
                "answer_prefix": "",
                "max_new_tokens": 32,
            },
        ]
        return pd.DataFrame(rows)

    def score(self, df: pd.DataFrame) -> Dict[str, object]:
        if len(df) == 0:
            return {"overall_score": 0.0, "task_scores": {}, "total_samples": 0}

        vals = []
        for _, row in df.iterrows():
            vals.append(substring_match_any(row.get("predicted_answer", ""), parse_answers(row.get("answer", []))))

        acc = (sum(vals) / len(vals) * 100) if vals else 0.0
        return {
            "overall_score": round(acc, 2),
            "task_scores": {"reading_comprehension": {"string_match": round(acc, 2)}},
            "total_samples": int(len(df)),
        }
