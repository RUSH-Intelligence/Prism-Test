"""Needle-in-a-haystack benchmark (kvpress-evaluation port).

Port of ``/scratch/sj157/kvpress/evaluation/benchmarks/needle_in_haystack``
(dataset ``alessiodevoto/paul_graham_essays`` — one row holding a ~3M-char
Paul Graham essay blob plus the canonical needle/question/answer_prefix:
the needle is the "best thing to do in San Francisco" sentence).

Haystack construction follows the reference algorithm: truncate the essay
blob to a context budget, splice the needle at ``depth%`` of the truncated
length, wrap with ``"This is a very long story book: <book> {context}
</book>."``. One row is produced per depth; depths are the benchmark's
subsets (``depth_0`` .. ``depth_100`` in steps of 10 by default, selectable
via ``--subsets depth_50,...``).

Deviations from kvpress (documented):
- The reference truncates/splices by TOKENS using the run's tokenizer
  (``max_context_length - needle_tokens - 150``); a Prism benchmark has no
  tokenizer at load time, so this port budgets by CHARACTERS
  (``chars_per_token`` x token budget, default 4 chars/token ~= Llama/Qwen
  English text) and snaps the insertion point to the next word boundary.
  Depth semantics (fraction of the haystack before the needle) are preserved;
  absolute context length is approximate. Set the class attribute
  ``context_length_tokens`` (subclass or edit) for other lengths.
- ROUGE is computed with the same pltrdy-``rouge`` call and the reference's
  swapped argument order — ``get_scores(needle, prediction)`` passes the
  needle as "hypothesis" and the prediction as "reference". ROUGE F1 is
  symmetric under the swap; precision/recall are transposed relative to
  convention. Kept verbatim for score parity.
- The reference returns an UNAGGREGATED per-row list; this port additionally
  emits Prism-conventional aggregates (``overall_score`` = mean ROUGE-L F x
  100, per-depth ``task_scores``) and keeps the raw per-row dicts under
  ``kvpress_per_sample``.
- Empty predictions score 0 instead of raising (pltrdy rouge raises
  ValueError on empty input; a truncated/failed generation should not crash
  scoring).

The ``rouge`` package is imported lazily inside ``score`` so benchmark
auto-discovery does not require it (install with ``pip install rouge``).
"""

from typing import Dict, List, Optional

import pandas as pd

from eval_harness.benchmarks.base import Benchmark, BenchmarkInfo
from eval_harness.benchmarks.registry import register_benchmark

def load_dataset(*args, **kwargs):
    """Lazy proxy for :func:`datasets.load_dataset`.

    CI installs no ``datasets`` package and registry auto-discovery imports
    every benchmark module, so the import must not happen at module level;
    tests also patch this module-level name.
    """
    from datasets import load_dataset as _load_dataset

    return _load_dataset(*args, **kwargs)


NEEDLE_DATASET = "alessiodevoto/paul_graham_essays"
CONTEXT_WRAPPER = "This is a very long story book: <book> {context} </book>."
DEFAULT_DEPTHS = list(range(0, 101, 10))
# The reference reserves 150 tokens of headroom below max_context_length.
TOKEN_HEADROOM = 150


def insert_needle_at_depth(
    haystack: str,
    needle: str,
    depth_percent: float,
    char_budget: int,
    wrapper: str = CONTEXT_WRAPPER,
) -> str:
    """Truncate ``haystack`` to ``char_budget`` chars, splice ``needle`` at
    ``depth_percent`` of the truncated length (snapped forward to the next
    word boundary), and wrap. Pure function for unit testing."""
    hay = haystack[: max(char_budget, 0)]
    idx = int(len(hay) * depth_percent / 100.0)
    idx = max(0, min(idx, len(hay)))
    # Snap forward to whitespace so the needle never splits a word.
    while idx < len(hay) and not hay[idx].isspace():
        idx += 1
    spliced = hay[:idx] + needle + hay[idx:]
    return wrapper.format(context=spliced)


@register_benchmark("needle_in_haystack", aliases=["niah_pg"])
class NeedleInHaystackBenchmark(Benchmark):
    """Paul-Graham-essays needle test (kvpress port), ROUGE-scored."""

    # Approximate token budget of the haystack (see module docstring).
    context_length_tokens: int = 16384
    chars_per_token: int = 4

    @property
    def info(self) -> BenchmarkInfo:
        return BenchmarkInfo(
            name="needle_in_haystack",
            description="Needle-in-a-haystack over Paul Graham essays "
            "(kvpress port); one row per insertion depth; ROUGE scoring",
            default_subsets=[f"depth_{d}" for d in DEFAULT_DEPTHS],
        )

    @staticmethod
    def _parse_depths(subsets: List[str]) -> List[int]:
        depths = []
        for name in subsets:
            token = str(name).strip().lower()
            if token.startswith("depth_"):
                token = token[len("depth_"):]
            depth = int(token)
            if not 0 <= depth <= 100:
                raise ValueError(f"needle depth must be in [0, 100], got {depth}")
            depths.append(depth)
        return depths

    def load(self, subsets: Optional[List[str]] = None) -> pd.DataFrame:
        depths = self._parse_depths(self.resolve_subsets(subsets))
        seed = load_dataset(NEEDLE_DATASET, split="test").to_pandas().iloc[0]

        needle = str(seed["needle"])
        question = str(seed["question"])
        answer_prefix = str(seed["answer_prefix"])
        max_new_tokens = int(seed.get("max_new_tokens", 40))

        needle_tokens_est = max(len(needle) // self.chars_per_token, 1)
        char_budget = (
            self.context_length_tokens - needle_tokens_est - TOKEN_HEADROOM
        ) * self.chars_per_token

        rows = []
        for depth in depths:
            rows.append(
                {
                    "context": insert_needle_at_depth(
                        str(seed["context"]), needle, depth, char_budget,
                    ),
                    "question": question,
                    "answer_prefix": answer_prefix,
                    "max_new_tokens": max_new_tokens,
                    "needle": needle,
                    "answer": needle,
                    "needle_depth": depth,
                    "task": f"depth_{depth}",
                }
            )
        return pd.DataFrame(rows)

    def score(self, df: pd.DataFrame) -> Dict[str, object]:
        if df.empty:
            return {"overall_score": 0.0, "task_scores": {}, "kvpress_per_sample": []}

        from rouge import Rouge  # lazy: keep auto-discovery dependency-free

        scorer = Rouge()
        zero = {
            "rouge-1": {"r": 0.0, "p": 0.0, "f": 0.0},
            "rouge-2": {"r": 0.0, "p": 0.0, "f": 0.0},
            "rouge-l": {"r": 0.0, "p": 0.0, "f": 0.0},
        }

        per_sample = []
        task_scores: Dict[str, Dict[str, float]] = {}
        rouge_l_fs = []
        for _, row in df.iterrows():
            needle = str(row["needle"]).strip()
            pred_val = row.get("predicted_answer")
            # None/NaN-safe: pandas>=3 iterrows coerces None to NaN, and
            # str(nan) would otherwise rouge-score the literal string "nan".
            pred = pred_val.strip() if isinstance(pred_val, str) else ""
            if needle and pred:
                # kvpress argument order kept verbatim (needle as "hypothesis").
                score = scorer.get_scores(needle, pred)[0]
            else:
                score = zero
            per_sample.append(score)
            rouge_l_f = float(score["rouge-l"]["f"])
            rouge_l_fs.append(rouge_l_f)
            task_scores[str(row.get("task", "needle"))] = {
                "rouge1_f": round(100.0 * float(score["rouge-1"]["f"]), 2),
                "rouge2_f": round(100.0 * float(score["rouge-2"]["f"]), 2),
                "rougeL_f": round(100.0 * rouge_l_f, 2),
            }

        overall = 100.0 * sum(rouge_l_fs) / len(rouge_l_fs)
        return {
            "overall_score": round(overall, 2),
            "task_scores": task_scores,
            "total_samples": len(df),
            # kvpress-verbatim per-row ROUGE dicts (their metrics.json payload).
            "kvpress_per_sample": per_sample,
        }
