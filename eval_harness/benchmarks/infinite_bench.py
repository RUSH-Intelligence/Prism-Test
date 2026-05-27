from __future__ import annotations

import re
import string
from collections import Counter
from typing import Dict, List

import pandas as pd

from .base import Benchmark, BenchmarkInfo
from .common import substring_match_any
from .registry import register_benchmark


INFINITE_BENCH_SUBSETS = [
    "passkey",
    "kv_retrieval",
    "number_string",
    "code_run",
    "code_debug",
    "math_find",
    "longbook_qa_eng",
    "longdialogue_qa_eng",
    "longbook_choice_eng",
]


def _first_int_match(text: str) -> str:
    for item in re.split(r"[^0-9]", str(text)):
        if item:
            return item
    return ""


def _normalize_answer(text: str) -> str:
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _qa_f1_score(pred: str, labels: List[str]) -> float:
    pred_tokens = _normalize_answer(pred).split()
    if not pred_tokens:
        return 0.0

    best = 0.0
    for label in labels:
        label_tokens = _normalize_answer(label).split()
        if not label_tokens:
            continue
        common = Counter(pred_tokens) & Counter(label_tokens)
        same = sum(common.values())
        if same == 0:
            continue
        precision = same / len(pred_tokens)
        recall = same / len(label_tokens)
        best = max(best, (2 * precision * recall) / (precision + recall))
    return best


def _score_kv_retrieval(pred: str, label) -> float:
    key = label[0] if isinstance(label, list) and label else label
    cleaned = str(pred)
    for ch in ["\n", ":", '"', "'", ".", ",", "?", "!", "{", "}"]:
        cleaned = cleaned.replace(ch, " ")
    return 1.0 if str(key) in cleaned.split() else 0.0


def _score_passkey_like(pred: str, label) -> float:
    key = label[0] if isinstance(label, list) and label else label
    return 1.0 if str(key) == _first_int_match(pred) else 0.0


def _score_code_run(pred: str, label) -> float:
    key = label[0] if isinstance(label, list) and label else label
    cleaned = str(pred).strip()
    for ch in ["\n", ".", "`", "'", '"', ":"]:
        cleaned = cleaned.replace(ch, " ")
    words = cleaned.split()
    if not words:
        return 0.0
    try:
        return 1.0 if int(words[-1]) == int(key) else 0.0
    except Exception:
        return 0.0


def _score_code_debug(pred: str, label) -> float:
    if not isinstance(label, list) or len(label) < 2:
        return 0.0

    fn_name = str(label[0])
    option = str(label[1])
    raw = str(pred).strip()

    match = re.search(r"\b[A-J]\b(?!.*\b[A-J]\b)", raw)
    if match and match.group(0) == option:
        return 1.0

    cleaned = raw
    for ch in ["\n", "`", "'", '"', "-", "*", "Option", "option"]:
        cleaned = cleaned.replace(ch, " ")
    while "  " in cleaned:
        cleaned = cleaned.replace("  ", " ")

    if cleaned.startswith(option) or cleaned.startswith(fn_name):
        return 1.0

    for prefix in ["answer is:", "is:", "answer:", "correct option is:"]:
        idx = cleaned.find(prefix)
        if idx == -1:
            continue
        if len(cleaned) < idx + len(prefix) + 1:
            return 0.0
        after = cleaned[idx + len(prefix) + 1 :]
        if after.startswith(option) or after.startswith(fn_name):
            return 1.0
        return 0.0

    return 0.0


def _score_math_find(pred: str, label) -> float:
    key = label[0] if isinstance(label, list) and label else label
    if isinstance(key, int):
        found = re.search(r"\d+\.\d+|\d+", str(pred))
        if found is None:
            return 0.0
        return 1.0 if int(float(found.group(0).strip())) == key else 0.0
    if isinstance(key, float):
        found = re.search(r"\d+\.\d+|\d+", str(pred))
        if found is None:
            return 0.0
        return 1.0 if float(found.group(0).strip()) == key else 0.0
    return 0.0


def _score_longdialogue_qa_eng(pred: str, label) -> float:
    labels = label if isinstance(label, list) else [label]
    upred = str(pred).upper()
    for item in labels:
        if str(item).upper() in upred:
            return 1.0
    return 0.0


def _score_longbook_choice_eng(pred: str, label) -> float:
    labels = [str(x) for x in (label if isinstance(label, list) else [label])]
    raw = str(pred).strip()

    match = re.search(r"\b[A-D]\b(?!.*\b[A-D]\b)", raw)
    if match and match.group(0) in labels:
        return 1.0

    if not raw:
        return 0.0
    if raw[0] in "ABCD" and raw[0] in labels:
        return 1.0
    if raw in labels:
        return 1.0

    cleaned = raw
    for ch in ["\n", '"', "'", ".", ",", "?", "!", "{", "}"]:
        cleaned = cleaned.replace(ch, " ")
    while "  " in cleaned:
        cleaned = cleaned.replace("  ", " ")

    for prefix in ["answer is:", "answer:", "answer is", "option is"]:
        idx = cleaned.find(prefix)
        if idx == -1:
            continue
        if len(cleaned) < idx + len(prefix) + 1:
            return 0.0
        after = cleaned[idx + len(prefix) + 1 :]
        for item in labels:
            if after.startswith(item):
                return 1.0
        return 0.0

    for word in cleaned.split():
        if word in "ABCD" and word in labels:
            return 1.0
    return 0.0


def _score_longbook_qa_eng(pred: str, label) -> float:
    labels = [str(x) for x in (label if isinstance(label, list) else [label])]
    return _qa_f1_score(pred, labels)


def _score_one(task: str, pred: str, label) -> float:
    task_name = str(task)
    if task_name == "kv_retrieval":
        return _score_kv_retrieval(pred, label)
    if task_name in {"passkey", "number_string"}:
        return _score_passkey_like(pred, label)
    if task_name == "code_run":
        return _score_code_run(pred, label)
    if task_name == "code_debug":
        return _score_code_debug(pred, label)
    if task_name == "math_find":
        return _score_math_find(pred, label)
    if task_name == "longdialogue_qa_eng":
        return _score_longdialogue_qa_eng(pred, label)
    if task_name == "longbook_choice_eng":
        return _score_longbook_choice_eng(pred, label)
    if task_name == "longbook_qa_eng":
        return _score_longbook_qa_eng(pred, label)

    # Fallback for any unexpected task while keeping benchmark usable.
    labels = label if isinstance(label, list) else [label]
    return substring_match_any(pred, [str(x) for x in labels])


@register_benchmark("infinite_bench")
class InfiniteBenchBenchmark(Benchmark):
    @property
    def info(self) -> BenchmarkInfo:
        return BenchmarkInfo(
            name="infinite_bench",
            description="InfiniteBench long-context benchmark",
            default_subsets=INFINITE_BENCH_SUBSETS,
        )

    def load(self, subsets: List[str] | None = None) -> pd.DataFrame:
        subsets = self.resolve_subsets(subsets)
        frames: List[pd.DataFrame] = []

        for subset in subsets:
            from datasets import load_dataset

            ds = load_dataset("MaxJeblick/InfiniteBench", subset, split="test")
            sdf = ds.to_pandas()
            sdf["task"] = subset
            if "answer" not in sdf.columns and "answers" in sdf.columns:
                sdf["answer"] = sdf["answers"]
            if "answer_prefix" not in sdf.columns:
                sdf["answer_prefix"] = ""
            if "max_new_tokens" not in sdf.columns:
                sdf["max_new_tokens"] = 64
            frames.append(sdf)

        return pd.concat(frames, ignore_index=True)

    def score(self, df: pd.DataFrame) -> Dict[str, object]:
        if len(df) == 0:
            return {"overall_score": 0.0, "task_scores": {}, "total_samples": 0}

        task_scores: Dict[str, float] = {}
        for task, tdf in df.groupby("task"):
            vals = []
            for _, row in tdf.iterrows():
                label = row.get("answer")
                if label is None and "answers" in tdf.columns:
                    label = row.get("answers")
                vals.append(_score_one(str(task), str(row.get("predicted_answer", "")), label))
            task_scores[str(task)] = round((sum(vals) / len(vals)) * 100, 2) if vals else 0.0

        overall = sum(task_scores.values()) / len(task_scores) if task_scores else 0.0
        return {
            "overall_score": round(overall, 2),
            "task_scores": task_scores,
            "total_samples": int(len(df)),
        }
