from __future__ import annotations

import re
from typing import Dict, Iterable, List

import pandas as pd

from .base import Benchmark, BenchmarkInfo
from .common import normalize_text, parse_answers
from .registry import register_benchmark


DATASET_HARD = "InfiniAILab/gsm_infinite_hard_128k"
DATASET_MEDIUM = "InfiniAILab/gsm_infinite_medium_128k"
DATASET_SYMBOLIC = "InfiniAILab/gsm_infinite_symbolic_128k"


def _select_split(ds_dict) -> str:
    if "train" in ds_dict:
        return "train"
    return next(iter(ds_dict.keys()))


def _first_non_empty_str(row: pd.Series, keys: Iterable[str]) -> str:
    for key in keys:
        if key in row and pd.notna(row[key]):
            value = str(row[key]).strip()
            if value:
                return value
    return ""


def _extract_last_number(text: str) -> str:
    matches = re.findall(r"-?\d+(?:\.\d+)?", str(text))
    return matches[-1] if matches else ""


def _build_answer_candidates(row: pd.Series) -> List[str]:
    candidates: List[str] = []

    if "answer_q" in row and pd.notna(row["answer_q"]):
        candidates.append(str(row["answer_q"]))

    if "answer_list" in row and pd.notna(row["answer_list"]):
        parsed = parse_answers(row["answer_list"])
        candidates.extend(parsed)

    if "solution" in row and pd.notna(row["solution"]):
        solution = str(row["solution"])
        candidates.append(solution)
        number = _extract_last_number(solution)
        if number:
            candidates.append(number)

    answer_text = _first_non_empty_str(row, ["answer", "output", "outputs"])
    if answer_text:
        candidates.extend(parse_answers(answer_text))

    deduped: List[str] = []
    seen = set()
    for candidate in candidates:
        c = str(candidate).strip()
        if not c:
            continue
        if c not in seen:
            deduped.append(c)
            seen.add(c)
    return deduped


def _normalize_for_prompt(df: pd.DataFrame, task_name: str) -> pd.DataFrame:
    out = df.copy()

    contexts: List[str] = []
    questions: List[str] = []
    answers: List[List[str]] = []
    context_lengths: List[int] = []

    for _, row in out.iterrows():
        problem = _first_non_empty_str(row, ["problem", "context", "input", "prompt"])
        question = _first_non_empty_str(row, ["question"])

        context = problem
        if not context and question:
            context = question

        contexts.append(context)
        questions.append(question)
        answers.append(_build_answer_candidates(row))

        if "length" in row and pd.notna(row["length"]):
            try:
                context_lengths.append(int(row["length"]))
            except Exception:
                context_lengths.append(128000)
        else:
            context_lengths.append(128000)

    out["context"] = contexts
    out["question"] = questions
    out["answer"] = answers
    out["task"] = task_name
    out["context_length"] = context_lengths

    if "answer_prefix" not in out.columns:
        out["answer_prefix"] = ""
    if "max_new_tokens" not in out.columns:
        out["max_new_tokens"] = 512

    return out


def _row_correct(prediction: str, refs: List[str]) -> float:
    pred = str(prediction).strip()
    if not pred:
        return 0.0

    pred_num = _extract_last_number(pred)
    if pred_num:
        for ref in refs:
            ref_num = _extract_last_number(ref)
            if ref_num and pred_num == ref_num:
                return 1.0

    pred_norm = normalize_text(pred)
    ref_norms = [normalize_text(r) for r in refs if str(r).strip()]
    return 1.0 if any(r and (r in pred_norm or pred_norm in r) for r in ref_norms) else 0.0


class _GSMInfinite128KBase(Benchmark):
    dataset_id: str = ""
    benchmark_name: str = ""
    benchmark_description: str = ""

    @property
    def info(self) -> BenchmarkInfo:
        return BenchmarkInfo(
            name=self.benchmark_name,
            description=self.benchmark_description,
            default_subsets=["default"],
        )

    def load(self, subsets: List[str] | None = None) -> pd.DataFrame:
        from datasets import load_dataset

        ds_dict = load_dataset(self.dataset_id)
        split = _select_split(ds_dict)
        sdf = ds_dict[split].to_pandas()
        return _normalize_for_prompt(sdf, task_name=self.benchmark_name)

    def score(self, df: pd.DataFrame) -> Dict[str, object]:
        if len(df) == 0:
            return {"overall_score": 0.0, "task_scores": {}, "total_samples": 0}

        vals = []
        for _, row in df.iterrows():
            refs = parse_answers(row.get("answer", []))
            vals.append(_row_correct(row.get("predicted_answer", ""), refs))

        score = round((sum(vals) / len(vals)) * 100, 2) if vals else 0.0
        return {
            "overall_score": score,
            "task_scores": {self.benchmark_name: {"string_match": score}},
            "total_samples": int(len(df)),
        }


@register_benchmark("gsm_infinite_hard_128k")
class GSMInfiniteHard128KBenchmark(_GSMInfinite128KBase):
    dataset_id = DATASET_HARD
    benchmark_name = "gsm_infinite_hard_128k"
    benchmark_description = "GSM-Infinite Hard 128k from InfiniAILab"


@register_benchmark("gsm_infinite_medium_128k")
class GSMInfiniteMedium128KBenchmark(_GSMInfinite128KBase):
    dataset_id = DATASET_MEDIUM
    benchmark_name = "gsm_infinite_medium_128k"
    benchmark_description = "GSM-Infinite Medium 128k from InfiniAILab"


@register_benchmark("gsm_infinite_symbolic_128k")
class GSMInfiniteSymbolic128KBenchmark(_GSMInfinite128KBase):
    dataset_id = DATASET_SYMBOLIC
    benchmark_name = "gsm_infinite_symbolic_128k"
    benchmark_description = "GSM-Infinite Symbolic 128k from InfiniAILab"
