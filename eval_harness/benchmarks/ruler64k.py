from __future__ import annotations

import logging
import re
from typing import Dict, List

import pandas as pd

from .base import Benchmark, BenchmarkInfo
from .common import parse_answers
from .registry import register_benchmark

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt decomposition (query-AGNOSTIC parity with att-hub ruler16k/32k).
#
# tonychenxyz/ruler-full ships each sample as a FULL Qwen chat-templated string
# with the question baked into the user turn at the very end:
#   <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n[CONTEXT]...[QUESTION][ANSWER_PREFIX]<|im_end|>\n<|im_start|>assistant\n
# Previously the whole prompt was dumped into the `context` column (question
# included), so the KV compressor — which compresses ONLY `context`, before the
# question is processed — saw the query inside the compressed region (query-AWARE),
# inflating observation-attention methods (snapkv/pyramidkv). att-hub ruler16k/32k
# instead ship SEPARATE context/question/answer_prefix columns (query-agnostic).
# We replicate that: strip the chat template and split the body into
# [context | question | answer_prefix]. (att-hub-ruler-64k/128k do not exist on HF.)
# ---------------------------------------------------------------------------
_USER_HEAD_RE = re.compile(r"<\|im_start\|>user\n")
_ASSISTANT_TAIL = "<|im_end|>\n<|im_start|>assistant\n"

# Phrase that begins the trailing question, per RULER task. `rfind` (LAST match)
# is load-bearing: cwe/qa/vt repeat the instruction once as a preamble (stays in
# context) and once at the end (the real question).
QUESTION_ANCHORS = {
    "niah_single_1": "What is the special magic number for",
    "niah_single_2": "What is the special magic number for",
    "niah_multikey_1": "What is the special magic number for",
    "niah_multikey_2": "What is the special magic number for",
    "niah_single_3": "What is the special magic uuid for",
    "niah_multikey_3": "What is the special magic uuid for",
    "niah_multiquery": "What are all the special magic numbers for",
    "niah_multivalue": "What are all the special magic numbers for",
    "qa_1": "Answer the question based on the given documents",
    "qa_2": "Answer the question based on the given documents",
    "vt": "Question: Find all variables that are assigned the value",
    "cwe": "Question: What are the 10 most common words",
    "fwe": "Question: Do not provide any explanation",
}

# Per-task generation budgets matching att-hub ruler16k (overridden when the run
# config sets max_new_tokens, e.g. the multi-ctx fleet's 128).
_MAX_NEW_TOKENS = {"qa_1": 32, "qa_2": 32, "vt": 30, "cwe": 120, "fwe": 50}


def _split_prompt(prompt: str, task: str):
    """Split a ruler-full chat-templated prompt into (context, question, answer_prefix).

    Returns ``None`` if the prompt cannot be split (unknown task / missing
    anchor) so the caller can fall back to legacy whole-prompt-as-context
    behavior rather than crash. Validated on all 13 RULER tasks (both variants,
    all context lengths) with exact reconstruction.
    """
    body = prompt
    heads = list(_USER_HEAD_RE.finditer(body))
    if heads:
        body = body[heads[-1].end():]
    tail = body.rfind(_ASSISTANT_TAIL)
    if tail != -1:
        body = body[:tail]
    else:
        body = body.rstrip()
        if body.endswith("<|im_end|>"):
            body = body[: -len("<|im_end|>")]

    anchor = QUESTION_ANCHORS.get(task)
    if not anchor or anchor not in body:
        return None
    qi = body.rfind(anchor)
    context, q_block = body[:qi], body[qi:]

    sp = q_block.find("? ")
    if sp != -1:
        question, answer_prefix = q_block[: sp + 2], q_block[sp + 2:]
    else:  # vt-style: no "?", split at the trailing "Answer:" cue
        ap = q_block.find("Answer:")
        if ap == -1:
            return None
        question, answer_prefix = q_block[:ap], q_block[ap:]
    return context, question, answer_prefix


RULER_SUBSETS = [
    "cwe",
    "fwe",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multiquery",
    "niah_multivalue",
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "qa_1",
    "qa_2",
    "vt",
]


def _collect_rows_for_context_length(subsets: List[str], context_length: int) -> pd.DataFrame:
    from datasets import load_dataset

    wanted = set(subsets)
    rows: List[Dict[str, object]] = []

    for variant in ["plain", "memwrap"]:
        ds = load_dataset("tonychenxyz/ruler-full", variant, split="validation", streaming=True)
        for sample in ds:
            category = str(sample.get("category", ""))
            marker = f"_{context_length}"
            if marker not in category:
                continue

            extra = sample.get("extra_info") or {}
            task = str(extra.get("ruler_task", ""))
            if task not in wanted:
                suffix = category.split("/")[-1]
                if suffix.endswith(marker):
                    task = suffix[: -len(marker)]
            if task not in wanted:
                continue

            ground_truth = extra.get("ground_truth") or {}
            if isinstance(ground_truth, dict):
                answer = ground_truth.get("answers", "")
            else:
                answer = ground_truth
            prompt = str(sample.get("prompt", ""))
            split = _split_prompt(prompt, task)
            if split is not None:
                context, question, answer_prefix = split
            else:
                logger.warning(
                    "ruler-full: could not split prompt for task=%r (cl=%s); falling back "
                    "to whole-prompt context (query-AWARE for this row).",
                    task, context_length,
                )
                context, question, answer_prefix = prompt, "", ""
            rows.append(
                {
                    "context": context,
                    "question": question,
                    "answer": answer,
                    "task": task,
                    "context_length": context_length,
                    "answer_prefix": answer_prefix,
                    "max_new_tokens": _MAX_NEW_TOKENS.get(task, 128),
                    "variant": variant,
                    "category": category,
                }
            )

    if not rows:
        raise ValueError(
            f"No rows found in tonychenxyz/ruler-full for context_length={context_length} and subsets={subsets}."
        )

    return pd.DataFrame(rows)


@register_benchmark("ruler64k")
class Ruler64KBenchmark(Benchmark):
    @property
    def info(self) -> BenchmarkInfo:
        return BenchmarkInfo(
            name="ruler64k",
            description="RULER 64k benchmark from tonychenxyz/ruler-full",
            default_subsets=RULER_SUBSETS,
        )

    def load(self, subsets: List[str] | None = None) -> pd.DataFrame:
        subsets = self.resolve_subsets(subsets)
        return _collect_rows_for_context_length(subsets, context_length=65536)

    def score(self, df: pd.DataFrame) -> Dict[str, object]:
        if len(df) == 0:
            return {
                "overall_score": 0.0,
                "task_scores": {},
                "context_length_scores": {},
                "summary": {
                    "total_tasks": 0,
                    "total_samples": 0,
                    "context_lengths": [],
                },
            }

        np_pattern = re.compile(r"[\x00-\x1f]")

        def _clean_prediction(text) -> str:
            return np_pattern.sub("", str(text).strip()).strip().lower()

        def _string_match_part(pred: str, refs: List[str]) -> float:
            if not refs:
                return 0.0
            return max(1.0 if r and r in pred else 0.0 for r in refs)

        def _string_match_all(pred: str, refs: List[str]) -> float:
            if not refs:
                return 0.0
            return sum(1.0 if r and r in pred else 0.0 for r in refs) / len(refs)

        task_scores: Dict[str, float] = {}
        for task, tdf in df.groupby("task"):
            task_category = str(task).split("_")[0]
            metric_fn = _string_match_part if task_category == "qa" else _string_match_all
            vals = []
            for _, row in tdf.iterrows():
                pred = _clean_prediction(row.get("predicted_answer", ""))
                refs = [
                    str(r).strip().lower()
                    for r in parse_answers(row.get("answer", row.get("answers", [])))
                ]
                vals.append(metric_fn(pred, refs))
            task_scores[str(task)] = round((sum(vals) / len(vals)) * 100, 2) if vals else 0.0

        overall = sum(task_scores.values()) / len(task_scores) if task_scores else 0.0

        context_length_scores: Dict[str, float] = {}
        if "context_length" in df.columns:
            for context_length, cdf in df.groupby("context_length"):
                c_task_scores: Dict[str, float] = {}
                for task, tdf in cdf.groupby("task"):
                    task_category = str(task).split("_")[0]
                    metric_fn = _string_match_part if task_category == "qa" else _string_match_all
                    vals = []
                    for _, row in tdf.iterrows():
                        pred = _clean_prediction(row.get("predicted_answer", ""))
                        refs = [
                            str(r).strip().lower()
                            for r in parse_answers(row.get("answer", row.get("answers", [])))
                        ]
                        vals.append(metric_fn(pred, refs))
                    c_task_scores[str(task)] = (sum(vals) / len(vals)) * 100 if vals else 0.0
                c_overall = sum(c_task_scores.values()) / len(c_task_scores) if c_task_scores else 0.0
                context_length_scores[str(context_length)] = round(c_overall, 2)

        return {
            "overall_score": round(overall, 2),
            "task_scores": {k: {"string_match": v} for k, v in task_scores.items()},
            "context_length_scores": context_length_scores,
            "summary": {
                "total_tasks": len(task_scores),
                "total_samples": int(len(df)),
                "context_lengths": list(context_length_scores.keys()),
            },
        }
