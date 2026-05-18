from __future__ import annotations

import ast
from typing import Dict, List

import pandas as pd

from .base import Benchmark, BenchmarkInfo
from .common import normalize_text, parse_answers, substring_match_any
from .registry import register_benchmark


LOFT_SUBSETS = [
    "nq_32k", "nq_128k", "nq_1m",
    "hotpotqa_32k", "hotpotqa_128k", "hotpotqa_1m",
    "musique_32k", "musique_128k", "musique_1m",
    "qampari_32k", "qampari_128k", "qampari_1m",
    "quest_32k", "quest_128k", "quest_1m",
]


def _extract_list_like_prediction(text: str) -> List[str]:
    raw = str(text).strip()
    if "[" in raw and "]" in raw:
        start = raw.find("[")
        end = raw.rfind("]") + 1
        chunk = raw[start:end]
        try:
            val = ast.literal_eval(chunk)
            if isinstance(val, (list, tuple)):
                return [str(v) for v in val]
        except Exception:
            pass
    return [raw]


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
                continue
            out = pd.concat(subframes, ignore_index=True)
            out["task"] = subset
            if "answer_prefix" not in out.columns:
                out["answer_prefix"] = "Final Answer: "
            if "max_new_tokens" not in out.columns:
                out["max_new_tokens"] = 128
            frames.append(out)

        if not frames:
            raise ValueError("No LOFT subsets were loaded.")
        return pd.concat(frames, ignore_index=True)

    def score(self, df: pd.DataFrame) -> Dict[str, object]:
        if len(df) == 0:
            return {"overall_score": 0.0, "task_scores": {}, "total_samples": 0}

        task_scores: Dict[str, Dict[str, float]] = {}
        overall_vals = []

        for task, tdf in df.groupby("task"):
            is_multi = str(task).startswith("qampari") or str(task).startswith("quest")
            em_vals = []
            cover_vals = []
            for _, row in tdf.iterrows():
                refs = [normalize_text(x) for x in parse_answers(row.get("answers", []))]
                preds = [normalize_text(x) for x in _extract_list_like_prediction(row.get("predicted_answer", ""))]
                if not refs:
                    em_vals.append(0.0)
                    if is_multi:
                        cover_vals.append(0.0)
                    continue

                if is_multi:
                    ref_set = set([r for r in refs if r])
                    pred_set = set([p for p in preds if p])
                    em = 1.0 if pred_set == ref_set and ref_set else 0.0
                    coverage = (len(ref_set.intersection(pred_set)) / len(ref_set)) if ref_set else 0.0
                    em_vals.append(em)
                    cover_vals.append(coverage)
                else:
                    em_vals.append(substring_match_any(" ".join(preds), refs))

            em_score = (sum(em_vals) / len(em_vals) * 100) if em_vals else 0.0
            item = {"em": round(em_score, 2)}
            if is_multi:
                cov = (sum(cover_vals) / len(cover_vals) * 100) if cover_vals else 0.0
                item["coverage"] = round(cov, 2)
                overall_vals.append((em_score + cov) / 2)
            else:
                overall_vals.append(em_score)
            task_scores[str(task)] = item

        overall = sum(overall_vals) / len(overall_vals) if overall_vals else 0.0
        return {
            "overall_score": round(overall, 2),
            "task_scores": task_scores,
            "total_samples": int(len(df)),
        }
