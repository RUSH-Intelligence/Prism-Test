"""MATH-500 benchmark (kvpress-evaluation port).

Port of ``/scratch/sj157/kvpress/evaluation/benchmarks/math500`` (dataset
``alessiodevoto/math500``, 500 problems adapted from HuggingFaceH4/MATH-500).

Row shape mirrors the reference exactly: ``context`` is the constant single
space ``" "`` for every row (the harness prefills one trivial context group
and greedy-decodes each problem in the assistant region), ``question`` is the
raw problem statement, ``answer_prefix`` is empty, ``max_new_tokens`` is 4096.
No system prompt and no "put your answer in \\boxed{}" instruction is added —
the model must emit ``\\boxed{}`` on its own.

Because the prefill is trivial, post-prefill KV compression is a no-op here
(min_tokens_to_compress gates it); this benchmark exercises DECODE-time
compression (e.g. ``streaming_ridge``) over the growing reasoning cache.

Scoring quirks replicated faithfully from kvpress (documented, on purpose):
- ``extract_boxed`` splits on ``"boxed{"`` and takes element [1] — the FIRST
  boxed occurrence — then truncates at the FIRST ``"}"``. No brace balancing:
  ``\\boxed{\\frac{14}{3}}`` extracts ``"\\frac{14"`` and can never match a
  brace-containing reference. Exact-match is therefore only reliable for
  brace-free answers (397/500 references contain no ``{``).
- Comparison is exact string equality against ``str(answer)`` — no math
  equivalence, no normalization.
- ``answered`` counts rows whose prediction contains the substring
  ``"boxed{"``.
- Scoring assumes non-thinking generation (the research pipeline renders chat
  templates with ``enable_thinking=False``, matching the kvpress pipeline); a
  thinking trace containing an intermediate ``\\boxed{}`` would corrupt the
  first-boxed extraction.

Deviation from kvpress: the returned metrics add the Prism-conventional
``overall_score`` (accuracy x 100) and ``task_scores`` alongside the verbatim
kvpress keys (``correct``, ``answered``, ``accuracy``, ``total``).
"""

from typing import Dict, List, Optional

import pandas as pd

from eval_harness.benchmarks.base import Benchmark, BenchmarkInfo
from eval_harness.benchmarks.registry import register_benchmark

MATH500_DATASET = "alessiodevoto/math500"


def extract_boxed_first(pred_answer: str) -> Optional[str]:
    """kvpress math500 extraction: FIRST ``boxed{``, up to the FIRST ``}``.

    Returns None when no ``boxed{`` is present (the [1] index raises).
    """
    try:
        return str(pred_answer.split("boxed{")[1].split("}")[0])
    except IndexError:
        return None


@register_benchmark("math500")
class Math500Benchmark(Benchmark):
    """MATH-500 (kvpress port): exact-match on the first boxed answer."""

    @property
    def info(self) -> BenchmarkInfo:
        return BenchmarkInfo(
            name="math500",
            description="MATH-500 (kvpress port): 500 competition math problems, "
            "first-boxed exact-match scoring",
            default_subsets=["math500"],
        )

    def load(self, subsets: Optional[List[str]] = None) -> pd.DataFrame:
        # Lazy: CI installs no `datasets`; registry auto-discovery
        # imports every benchmark module (repo convention).
        from datasets import load_dataset

        del subsets  # single-subset benchmark
        df = load_dataset(MATH500_DATASET, split="test").to_pandas()
        # The dataset ships context/answer_prefix/max_new_tokens; default them
        # defensively so the runner contract holds even if the schema drifts.
        if "context" not in df.columns:
            df["context"] = " "
        if "answer_prefix" not in df.columns:
            df["answer_prefix"] = ""
        if "max_new_tokens" not in df.columns:
            df["max_new_tokens"] = 4096
        df["task"] = "math500"
        return df

    def score(self, df: pd.DataFrame) -> Dict[str, object]:
        if df.empty:
            return {
                "overall_score": 0.0,
                "correct": 0,
                "answered": 0,
                "accuracy": 0.0,
                "total": 0,
                "task_scores": {},
            }

        correct = 0
        answered = 0
        for _, row in df.iterrows():
            pred = row.get("predicted_answer")
            if not isinstance(pred, str):
                pred = ""  # None/NaN-safe (pandas>=3 iterrows coerces None to NaN)
            correct += extract_boxed_first(pred) == str(row["answer"])
            answered += "boxed{" in pred

        total = len(df)
        accuracy = correct / total
        return {
            "overall_score": round(100.0 * accuracy, 2),
            # kvpress-verbatim keys:
            "correct": int(correct),
            "answered": int(answered),
            "accuracy": accuracy,
            "total": total,
            "task_scores": {"math500": {"accuracy": round(100.0 * accuracy, 2)}},
        }
