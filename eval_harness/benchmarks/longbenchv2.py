from __future__ import annotations

from typing import Dict, List

import pandas as pd

from .base import Benchmark, BenchmarkInfo
from .common import extract_longbench_v2_answer, extract_option_letter
from .registry import register_benchmark


@register_benchmark("longbenchv2")
class LongBenchV2Benchmark(Benchmark):
    @property
    def info(self) -> BenchmarkInfo:
        return BenchmarkInfo(
            name="longbenchv2",
            description="LongBench-v2 multiple-choice long-context benchmark",
            default_subsets=["0shot", "cot"],
        )

    def load(self, subsets: List[str] | None = None) -> pd.DataFrame:
        subsets = self.resolve_subsets(subsets)
        frames: List[pd.DataFrame] = []
        for subset in subsets:
            from datasets import load_dataset

            ds = load_dataset("Xnhyacinth/LongBench-v2", subset, split="test")
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
        all_acc = []
        for task, tdf in df.groupby("task"):
            n = len(tdf)
            correct = 0
            for _, row in tdf.iterrows():
                pred = extract_longbench_v2_answer(row.get("predicted_answer", ""))
                # Gold ``answer`` is already a bare A-D letter in the dataset.
                gold = extract_option_letter(row.get("answer", ""))
                if pred and gold and pred == gold:
                    correct += 1
            acc = (correct / n) * 100 if n else 0.0
            task_scores[str(task)] = {"accuracy": round(acc, 2)}
            all_acc.append(acc)

        overall = sum(all_acc) / len(all_acc) if all_acc else 0.0
        return {
            "overall_score": round(overall, 2),
            "task_scores": task_scores,
            "total_samples": int(len(df)),
        }
