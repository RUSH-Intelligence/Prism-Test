from __future__ import annotations

from typing import Dict, List

import pandas as pd

from .base import Benchmark, BenchmarkInfo
from .common import extract_int_0_999, parse_answers
from .registry import register_benchmark


@register_benchmark("aime2025")
class AIME2025Benchmark(Benchmark):
    @property
    def info(self) -> BenchmarkInfo:
        return BenchmarkInfo(
            name="aime2025",
            description="AIME 2025 math reasoning benchmark",
            default_subsets=["aime2025"],
        )

    def load(self, subsets: List[str] | None = None) -> pd.DataFrame:
        del subsets
        from datasets import load_dataset

        df = load_dataset("xAlg-AI/att-hub-aime2025", split="test").to_pandas()
        df["task"] = "aime2025"
        if "answer_prefix" not in df.columns:
            df["answer_prefix"] = ""
        if "max_new_tokens" not in df.columns:
            df["max_new_tokens"] = 512
        return df

    def score(self, df: pd.DataFrame) -> Dict[str, object]:
        total = len(df)
        if total == 0:
            return {"overall_score": 0.0, "exact_match": 0.0, "total_samples": 0}

        correct = 0
        extracted = 0
        for _, row in df.iterrows():
            pred = extract_int_0_999(row.get("predicted_answer", ""))
            refs = [extract_int_0_999(a) for a in parse_answers(row.get("answer", []))]
            refs = [r for r in refs if r]
            if pred:
                extracted += 1
            if pred and pred in refs:
                correct += 1

        exact = correct / total
        extraction_rate = extracted / total
        return {
            "overall_score": round(exact * 100, 2),
            "exact_match": round(exact * 100, 2),
            "extraction_rate": round(extraction_rate * 100, 2),
            "total_samples": total,
            "task_scores": {"aime2025": {"exact_match": round(exact * 100, 2)}},
        }
