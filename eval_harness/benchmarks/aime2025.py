"""AIME-25 benchmark (kvpress-evaluation port).

Port of ``/scratch/sj157/kvpress/evaluation/benchmarks/aime25`` (dataset
``alessiodevoto/aime25``, the 30 problems of AIME 2025 I & II).

REPLACES the previous ``aime2025`` benchmark (``xAlg-AI/att-hub-aime2025``
with integer-in-[0,999] extraction and a 512-token budget — too small for
reasoning models, which emit a long chain of thought before the final
``\\boxed{}``). This is the kvpress-faithful variant: last-boxed string
extraction and the dataset's 32000-token budget. Registered as ``aime2025``
with alias ``aime25``; the xAlg dataset remains reachable via the ``aime``
benchmark.

Row shape mirrors the reference exactly: ``context`` is the constant single
space ``" "``, ``question`` is the raw problem, ``answer_prefix`` is empty,
``max_new_tokens`` is 32000, and no boxing instruction is injected. As with
math500, the prefill is trivial, so this benchmark exercises DECODE-time KV
compression (e.g. ``streaming_ridge``) rather than post-prefill compression.

Scoring quirks replicated faithfully from kvpress (documented, on purpose):
- ``extract_boxed`` splits on ``"boxed{"`` and takes element [-1] — the LAST
  boxed occurrence (suits models that box intermediate results and the final
  answer last) — truncated at the FIRST ``"}"``. Unlike math500's [1]
  indexing, [-1] NEVER raises: with no ``boxed{`` present the "extraction" is
  the prediction itself up to its first ``}`` (or the whole prediction), which
  simply fails the equality check in practice.
- Comparison is exact string equality vs ``str(answer)``; AIME answers are
  brace-free integer strings, so the first-``}`` truncation is harmless here.
- ``answered`` counts rows containing the substring ``"boxed{"``.

Deviation from kvpress: metrics add the Prism-conventional ``overall_score``
(accuracy x 100) and ``task_scores`` alongside the verbatim kvpress keys.
"""

from typing import Dict, List, Optional

import pandas as pd
from datasets import load_dataset

from eval_harness.benchmarks.base import Benchmark, BenchmarkInfo
from eval_harness.benchmarks.registry import register_benchmark

AIME25_DATASET = "alessiodevoto/aime25"


def extract_boxed_last(pred_answer: str) -> Optional[str]:
    """kvpress aime25 extraction: LAST ``boxed{``, up to the FIRST ``}``.

    Faithfully never returns None on missing ``boxed{`` (split()[-1] is the
    whole string); kept identical so scored behavior matches the reference.
    """
    try:
        return str(pred_answer.split("boxed{")[-1].split("}")[0])
    except IndexError:
        return None


@register_benchmark("aime2025", aliases=["aime25"])
class Aime2025Benchmark(Benchmark):
    """AIME 2025 (kvpress port): exact-match on the last boxed answer."""

    @property
    def info(self) -> BenchmarkInfo:
        return BenchmarkInfo(
            name="aime2025",
            description="AIME 2025 I & II (kvpress port): 30 problems, "
            "last-boxed exact-match scoring, 32k-token reasoning budget",
            default_subsets=["aime2025"],
        )

    def load(self, subsets: Optional[List[str]] = None) -> pd.DataFrame:
        del subsets  # single-subset benchmark
        df = load_dataset(AIME25_DATASET, split="test").to_pandas()
        if "context" not in df.columns:
            df["context"] = " "
        if "answer_prefix" not in df.columns:
            df["answer_prefix"] = ""
        if "max_new_tokens" not in df.columns:
            df["max_new_tokens"] = 32000
        df["task"] = "aime2025"
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
            correct += extract_boxed_last(pred) == str(row["answer"])
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
            "task_scores": {"aime2025": {"accuracy": round(100.0 * accuracy, 2)}},
        }
