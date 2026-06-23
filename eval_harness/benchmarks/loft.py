from __future__ import annotations

import ast
import collections
import re
import string
import unicodedata
from typing import Dict, List

import numpy as np
import pandas as pd

from .base import Benchmark, BenchmarkInfo
from .registry import register_benchmark


LOFT_SUBSETS = [
    "nq_32k", "nq_128k", "nq_1m",
    "hotpotqa_32k", "hotpotqa_128k", "hotpotqa_1m",
    "musique_32k", "musique_128k", "musique_1m",
    "qampari_32k", "qampari_128k", "qampari_1m",
    "quest_32k", "quest_128k", "quest_1m",
]


def _normalize_answer(text: str) -> str:
    text = unicodedata.normalize("NFD", str(text))

    def _remove_articles(value: str) -> str:
        return re.sub(re.compile(r"\b(a|an|the)\b", re.UNICODE), " ", value)

    def _white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def _remove_punc(value: str) -> str:
        return "".join(ch for ch in value if ch not in set(string.punctuation))

    return _white_space_fix(_remove_articles(_remove_punc(text.lower())))


def _normalize_answers(values: List[str]) -> List[str]:
    return [_normalize_answer(v) for v in values]


def _get_tokens(text: str) -> List[str]:
    if not text:
        return []
    return _normalize_answer(text).split()


def _compute_em(gold_answers: List[str], pred_answer: str) -> float:
    return max(float(ga == pred_answer) for ga in gold_answers)


def _compute_subspan_em(gold_answers: List[str], pred_answer: str) -> float:
    return max(1.0 if ga in pred_answer else 0.0 for ga in gold_answers)


def _compute_f1(gold_answers: List[str], pred_answer: str) -> float:
    pred_toks = _get_tokens(pred_answer)
    f1_scores: List[float] = []
    for ga in gold_answers:
        gold_toks = _get_tokens(ga)
        common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
        num_same = sum(common.values())

        if num_same == 0:
            f1_scores.append(0.0)
            continue

        if not gold_toks or not pred_toks:
            f1_scores.append(float(gold_toks == pred_toks))
            continue

        precision = num_same / len(pred_toks)
        recall = num_same / len(gold_toks)
        f1_scores.append((2 * precision * recall) / (precision + recall))

    return max(f1_scores) if f1_scores else 0.0


def _compute_em_multi_value(gold_answers: List[str], pred_answers: List[str]) -> float:
    return float(set(gold_answers) == set(pred_answers))


def _compute_coverage(gold_answers: List[str], pred_answers: List[str]) -> float:
    if not gold_answers:
        return 0.0
    return len(set(pred_answers).intersection(set(gold_answers))) / float(len(gold_answers))


def _compute_multi_value_subspan_em(gold_answers: List[str], pred_answers: List[str]) -> float:
    if not gold_answers:
        return 0.0
    if not pred_answers:
        return 0.0

    import scipy.optimize  # lazy: keeps module import (registry autoload) scipy-free

    scores = np.zeros([len(gold_answers), len(pred_answers)])
    for gold_index, gold_item in enumerate(gold_answers):
        for pred_index, pred_item in enumerate(pred_answers):
            if gold_item in pred_item or pred_item in gold_item:
                scores[gold_index, pred_index] = 1

    row_ind, col_ind = scipy.optimize.linear_sum_assignment(-scores)
    aligned_scores = np.zeros(len(gold_answers))
    for r, c in zip(row_ind, col_ind):
        aligned_scores[r] = scores[r, c]
    return float(all(aligned_scores))


def _extract_prediction(model_output: str, answer_prefix: str = "final answer") -> List[str]:
    """Faithful port of upstream LOFT ``extract_prediction`` (loft_upstream/utils.py).

    Commits to the FIRST line containing both ``[`` and ``]`` — the ``break`` fires for
    that line whether or not it parses (so a non-parseable first bracket line yields no
    prediction rather than scanning on). The bracketed slice is parsed as a Python
    literal; if no line contains a bracket, ``[]`` is returned. There is deliberately NO
    answer_prefix-split or raw-text fallback: upstream returns ``[]`` for non-bracketed
    output (scoring 0), and adding fallbacks inflates em/subspan/f1/coverage above
    upstream. ``answer_prefix`` is advisory only (upstream uses it for a warning).

    One intentional, score-neutral deviation: upstream assigns ``preds = literal_eval(...)``
    raw; we coerce to a list of ``str`` so a non-list literal does not poison downstream
    list handling (well-formed ``[...]`` list outputs are unchanged).
    """
    def _escape_single_quotes(text: str) -> str:
        return re.sub(r"([a-zA-Z0-9])'([a-zA-Z0-9])", r"\1\\'\2", text)

    cleaned = str(model_output).replace("*", "").replace("`", "")
    preds: List[str] = []
    for line in cleaned.strip().split("\n"):
        if "[" in line and "]" in line:
            pred_start_index = line.find("[")
            pred_end_index = line.rfind("]") + 1
            pred_as_str = line[pred_start_index:pred_end_index].strip()
            try:
                parsed = ast.literal_eval(_escape_single_quotes(pred_as_str))
                preds = (
                    [str(p) for p in parsed]
                    if isinstance(parsed, (list, tuple))
                    else [str(parsed)]
                )
            except Exception:
                pass
            break
    return preds


@register_benchmark("loft", aliases=["loft_rag"])
class LoftBenchmark(Benchmark):
    @property
    def info(self) -> BenchmarkInfo:
        return BenchmarkInfo(
            name="loft",
            description="LOFT RAG long-context benchmark",
            default_subsets=LOFT_SUBSETS,
        )

    def load(self, subsets: List[str] | None = None) -> pd.DataFrame:
        subsets = self.resolve_subsets(subsets)
        frames: List[pd.DataFrame] = []

        for subset in subsets:
            parts = subset.split("_")
            if len(parts) < 2:
                raise ValueError(f"Invalid LOFT subset '{subset}'. Use format like 'nq_32k'.")
            ds_name = "_".join(parts[:-1])
            length = parts[-1]
            hf_id = f"f20180301/loft-rag-{ds_name}-{length}"
            from datasets import load_dataset

            dsd = load_dataset(hf_id)
            subframes = []
            for split in ["dev", "test"]:
                if split in dsd:
                    sdf = dsd[split].to_pandas()
                    sdf["split"] = split
                    subframes.append(sdf)
            if not subframes:
                raise ValueError(f"No splits found for {subset} ({hf_id})")
            out = pd.concat(subframes, ignore_index=True)
            out["task"] = subset
            if "answer_prefix" not in out.columns:
                out["answer_prefix"] = "Final Answer: "
            if "max_new_tokens" not in out.columns:
                out["max_new_tokens"] = 128
            frames.append(out)

        if not frames:
            raise ValueError("No LOFT subsets were loaded.")

        combined = pd.concat(frames, ignore_index=True)
        required_columns = [
            "context",
            "question",
            "answers",
            "task",
            "answer_prefix",
            "max_new_tokens",
        ]
        missing_columns = [col for col in required_columns if col not in combined.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        return combined

    def score(self, df: pd.DataFrame) -> Dict[str, object]:
        if len(df) == 0:
            return {"overall_score": 0.0, "task_scores": {}, "total_samples": 0}

        task_scores: Dict[str, Dict[str, float]] = {}
        overall_em_vals: List[float] = []
        overall_subspan_vals: List[float] = []
        overall_f1_vals: List[float] = []
        overall_coverage_vals: List[float] = []

        for task, tdf in df.groupby("task"):
            is_multi = str(task).startswith("qampari") or str(task).startswith("quest")
            em_vals: List[float] = []
            subspan_vals: List[float] = []
            f1_vals: List[float] = []
            cover_vals: List[float] = []

            answer_prefix = (
                str(tdf["answer_prefix"].iloc[0])
                if "answer_prefix" in tdf.columns and len(tdf) > 0
                else "Final Answer: "
            )

            for _, row in tdf.iterrows():
                gold_answers = row.get("answers", [])
                if hasattr(gold_answers, "tolist") and not isinstance(gold_answers, (str, bytes)):
                    try:
                        gold_answers = gold_answers.tolist()
                    except Exception:
                        pass
                if isinstance(gold_answers, tuple):
                    gold_answers = list(gold_answers)
                if not isinstance(gold_answers, list):
                    gold_answers = [] if pd.isna(gold_answers) else [gold_answers]

                gold_answers = [ga for ga in gold_answers if pd.notna(ga)]
                refs = _normalize_answers([str(ga) for ga in gold_answers])

                if not refs:
                    em_vals.append(0.0)
                    subspan_vals.append(0.0)
                    if is_multi:
                        cover_vals.append(0.0)
                    else:
                        f1_vals.append(0.0)
                    continue

                predicted_output = str(row.get("predicted_answer", "")) if pd.notna(row.get("predicted_answer", "")) else ""
                pred_answers_raw = _extract_prediction(predicted_output, answer_prefix.lower())

                if not pred_answers_raw:
                    em_vals.append(0.0)
                    subspan_vals.append(0.0)
                    if not is_multi:
                        f1_vals.append(0.0)
                    # Upstream MultiValueRagEvaluation omits coverage for empty
                    # predictions (it is only recorded on the non-empty branch),
                    # so it must NOT enter the coverage denominator here.
                    continue

                preds = _normalize_answers(pred_answers_raw)

                if is_multi:
                    em_vals.append(_compute_em_multi_value(refs, preds))
                    subspan_vals.append(_compute_multi_value_subspan_em(refs, preds))
                    cover_vals.append(_compute_coverage(refs, preds))
                else:
                    pred_answer = preds[0]
                    em_vals.append(_compute_em(refs, pred_answer))
                    subspan_vals.append(_compute_subspan_em(refs, pred_answer))
                    f1_vals.append(_compute_f1(refs, pred_answer))

            em_score = (sum(em_vals) / len(em_vals) * 100) if em_vals else 0.0
            subspan_score = (sum(subspan_vals) / len(subspan_vals) * 100) if subspan_vals else 0.0
            item = {"em": round(em_score, 2), "subspan_em": round(subspan_score, 2)}
            if is_multi:
                cov = (sum(cover_vals) / len(cover_vals) * 100) if cover_vals else 0.0
                item["coverage"] = round(cov, 2)
                overall_coverage_vals.append(cov)
            else:
                f1_score = (sum(f1_vals) / len(f1_vals) * 100) if f1_vals else 0.0
                item["f1"] = round(f1_score, 2)
                overall_f1_vals.append(f1_score)

            overall_em_vals.append(em_score)
            overall_subspan_vals.append(subspan_score)
            task_scores[str(task)] = item

        overall_em = sum(overall_em_vals) / len(overall_em_vals) if overall_em_vals else 0.0
        overall_subspan = sum(overall_subspan_vals) / len(overall_subspan_vals) if overall_subspan_vals else 0.0
        overall_metrics: Dict[str, float] = {
            "em": round(overall_em, 2),
            "subspan_em": round(overall_subspan, 2),
        }
        if overall_f1_vals:
            overall_metrics["f1"] = round(sum(overall_f1_vals) / len(overall_f1_vals), 2)
        if overall_coverage_vals:
            overall_metrics["coverage"] = round(sum(overall_coverage_vals) / len(overall_coverage_vals), 2)

        return {
            # LOFT's paper/leaderboard primary metric for all 5 RAG datasets is
            # subspan_em (README "Primary Metric" column), not exact match.
            "overall_score": round(overall_subspan, 2),
            "overall_metrics": overall_metrics,
            "task_scores": task_scores,
            "total_samples": int(len(df)),
        }
