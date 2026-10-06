"""Recovery metrics (spec §15), paired bootstrap, per-example re-scoring and representation
metrics (spec §16).

* ``compression_drop = dense - compressed``; ``recovery = recovered - compressed``;
  ``recovery_fraction = recovery / compression_drop`` — ``None`` (flag ``undefined_gap``) when the
  drop is <= 0, flagged ``unstable_gap`` when the drop is small (< 3 points on a benchmark macro,
  < 10 on a single task) because the fraction is then dominated by noise.
* ``predictions.csv`` carries no per-example score, so rows are re-scored through the benchmark's
  own scorer (RULER: ``Benchmark.score`` on one-row frames; LongBench: ``_score_row``) and paired
  across arms by (task, row ordinal) after verifying row identity (task, question, answer).
* Paired, task-stratified bootstrap ported from kv_compression_adaptation/src/analysis/stats.py.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

UNSTABLE_GAP_MACRO = 3.0
UNSTABLE_GAP_TASK = 10.0


# ---------------------------------------------------------------------------
# recovery metrics
# ---------------------------------------------------------------------------
def recovery_metrics(dense: float, compressed: float, recovered: float, dense_recovered: Optional[float] = None,
                     *, unstable_gap: float = UNSTABLE_GAP_MACRO, eps: float = 1e-9) -> Dict[str, Any]:
    drop = float(dense) - float(compressed)
    rec = float(recovered) - float(compressed)
    out: Dict[str, Any] = {
        "dense": float(dense), "compressed": float(compressed), "compressed_recovered": float(recovered),
        "compression_drop": drop, "recovery": rec,
        "recovery_fraction": None, "recovery_fraction_defined": False, "undefined_reason": None, "flags": [],
    }
    if drop > eps:
        out["recovery_fraction"] = rec / drop
        out["recovery_fraction_defined"] = True
    else:
        out["undefined_reason"] = "no_compression_drop" if abs(drop) <= eps else "negative_compression_drop"
        out["flags"].append("undefined_gap")
    if abs(drop) < unstable_gap:
        out["flags"].append("unstable_gap")
    if dense_recovered is not None:
        out["dense_recovered"] = float(dense_recovered)
        out["dense_regression"] = float(dense_recovered) - float(dense)
        out["did"] = rec - out["dense_regression"]
    return out


# ---------------------------------------------------------------------------
# per-example re-scoring
# ---------------------------------------------------------------------------
def _row_identity(task: str, question: Any, answer: Any) -> str:
    return hashlib.sha256(f"{task}\x1f{question}\x1f{answer}".encode("utf-8", "replace")).hexdigest()[:16]


def per_example_scores(benchmark_name: str, predictions_csv) -> "pd.DataFrame":
    """``task, ordinal, row_id, score`` (0-100) for every row of a run's predictions.csv."""
    import pandas as pd

    from eval_harness.benchmarks.common import parse_answers
    from eval_harness.benchmarks.registry import get_benchmark

    df = pd.read_csv(predictions_csv, keep_default_na=False)
    bench = get_benchmark(benchmark_name)
    rows: List[dict] = []
    ordinal: Dict[str, int] = {}
    for i in range(len(df)):
        row = df.iloc[i]
        task = str(row.get("task", ""))
        ordinal[task] = ordinal.get(task, -1) + 1
        answer = row.get("answer", row.get("answers", ""))
        if benchmark_name.lower().startswith("ruler"):
            s = bench.score(df.iloc[[i]])["task_scores"][task]["string_match"]
        elif benchmark_name.lower().startswith("longbench"):
            r = row.to_dict()
            r["all_classes"] = parse_answers(r.get("all_classes")) if r.get("all_classes", "") != "" else []
            s = bench._score_row(r) * 100.0
        else:
            s = bench.score(df.iloc[[i]])["overall_score"]
        rows.append({"task": task, "ordinal": ordinal[task], "row_id": _row_identity(task, row.get("question", ""), answer),
                     "score": float(s)})
    return pd.DataFrame(rows)


def consistency_check(per_example: "pd.DataFrame", metrics_json: Dict[str, Any], *, tol: float = 0.011) -> Dict[str, Any]:
    """Per-task means of the re-scored rows must reproduce the run's own task scores."""
    out: Dict[str, Any] = {"ok": True, "tasks": {}}
    for task, g in per_example.groupby("task"):
        ref = metrics_json.get("task_scores", {}).get(task)
        if isinstance(ref, dict):
            ref = ref.get("string_match")
        if ref is None:
            continue
        mine = round(float(g["score"].mean()), 2)
        ok = abs(mine - float(ref)) <= tol
        out["tasks"][task] = {"rescored": mine, "metrics_json": float(ref), "ok": ok}
        out["ok"] = out["ok"] and ok
    return out


# ---------------------------------------------------------------------------
# pairing + bootstrap
# ---------------------------------------------------------------------------
def align_conditions(frames: Dict[str, "pd.DataFrame"]) -> Dict[str, Dict[str, np.ndarray]]:
    """``{condition: per-example df}`` -> ``{task: {condition: scores[n]}}`` with identical rows."""
    conds = list(frames)
    ref = frames[conds[0]]
    cells: Dict[str, Dict[str, np.ndarray]] = {}
    for task, g in ref.groupby("task"):
        g = g.sort_values("ordinal")
        ids = g["row_id"].tolist()
        cells[task] = {}
        for c in conds:
            h = frames[c][frames[c]["task"] == task].sort_values("ordinal")
            if h["row_id"].tolist() != ids:
                raise ValueError(f"task {task}: condition {c!r} rows differ from {conds[0]!r} (different questions/answers "
                                 f"or ordering) — the arms are not paired")
            cells[task][c] = h["score"].to_numpy(dtype=float)
    return cells


def macro(cells: Dict[str, Dict[str, np.ndarray]], cond: str) -> float:
    return float(np.mean([v[cond].mean() for v in cells.values()]))


def paired_bootstrap(cells: Dict[str, Dict[str, np.ndarray]], statistic: Callable[[Dict[str, np.ndarray]], np.ndarray],
                     n_resamples: int, seed: int, alpha: float = 0.05) -> Dict[str, Any]:
    """Resample rows WITHIN each task with the same indices for every condition (paired),
    compute the macro over tasks per replicate, apply ``statistic`` to the ``{cond: [B]}`` arrays."""
    rng = np.random.default_rng(seed)
    conds = list(next(iter(cells.values())))
    boot = {c: np.zeros(n_resamples) for c in conds}
    for v in cells.values():
        n = len(next(iter(v.values())))
        idx = rng.integers(0, n, size=(n_resamples, n))
        for c in conds:
            boot[c] += v[c][idx].mean(axis=1)
    for c in conds:
        boot[c] /= len(cells)
    point = float(statistic({c: np.array([macro(cells, c)]) for c in conds})[0])
    samples = statistic(boot)
    samples = samples[np.isfinite(samples)]
    if len(samples):
        lo, hi = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    else:
        lo, hi = float("nan"), float("nan")
    return {"estimate": point if math.isfinite(point) else None, "ci_low": float(lo) if math.isfinite(lo) else None,
            "ci_high": float(hi) if math.isfinite(hi) else None, "n_finite": int(len(samples)), "n_resamples": n_resamples}


def stat_drop(m):
    return m["dense"] - m["compressed"]


def stat_recovery(m):
    return m["compressed_recovered"] - m["compressed"]


def stat_fraction(m):
    gap = m["dense"] - m["compressed"]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(gap > 0, (m["compressed_recovered"] - m["compressed"]) / gap, np.nan)


def stat_dense_regression(m):
    return m["dense_recovered"] - m["dense"]


def stat_did(m):
    return (m["compressed_recovered"] - m["compressed"]) - (m["dense_recovered"] - m["dense"])


def _ci_block(cells, has_dense_recovered: bool, n_resamples: int, seed: int, alpha: float) -> Dict[str, Any]:
    stats = {"compression_drop": stat_drop, "recovery": stat_recovery, "recovery_fraction": stat_fraction}
    if has_dense_recovered:
        stats.update({"dense_regression": stat_dense_regression, "did": stat_did})
    return {name: paired_bootstrap(cells, fn, n_resamples, seed, alpha) for name, fn in stats.items()}


def benchmark_report(frames: Dict[str, "pd.DataFrame"], *, n_resamples: int = 10000, seed: int = 0,
                     alpha: float = 0.05) -> Dict[str, Any]:
    """Overall (macro over tasks) and per-task recovery metrics with paired bootstrap CIs."""
    cells = align_conditions(frames)
    has_dr = "dense_recovered" in frames
    overall_vals = {c: macro(cells, c) for c in frames}
    overall = recovery_metrics(overall_vals["dense"], overall_vals["compressed"], overall_vals["compressed_recovered"],
                               overall_vals.get("dense_recovered"), unstable_gap=UNSTABLE_GAP_MACRO)
    overall["n_examples"] = int(sum(len(next(iter(v.values()))) for v in cells.values()))
    overall["ci"] = _ci_block(cells, has_dr, n_resamples, seed, alpha)
    tasks: Dict[str, Any] = {}
    for task, v in cells.items():
        vals = {c: float(v[c].mean()) for c in v}
        t = recovery_metrics(vals["dense"], vals["compressed"], vals["compressed_recovered"], vals.get("dense_recovered"),
                             unstable_gap=UNSTABLE_GAP_TASK)
        t["n_examples"] = int(len(next(iter(v.values()))))
        t["ci"] = _ci_block({task: v}, has_dr, n_resamples, seed, alpha)
        tasks[task] = t
    return {"overall": overall, "tasks": tasks,
            "bootstrap": {"n_resamples": n_resamples, "seed": seed, "alpha": alpha, "paired": True, "stratified_by": "task"}}


# ---------------------------------------------------------------------------
# representation metrics (spec §16)
# ---------------------------------------------------------------------------
def representation_metrics(teacher: Dict[Any, "torch.Tensor"], student: Dict[Any, "torch.Tensor"]) -> Dict[str, Dict[str, float]]:
    """Per key: cosine mean/std, normalized MSE (sum over hidden of the unit-vector difference,
    the training loss), relative error ||S-T||_F / ||T||_F; inputs are ``[P, H]`` tensors."""
    import torch
    import torch.nn.functional as F

    out: Dict[str, Dict[str, float]] = {}
    for key, t in teacher.items():
        s = student[key]
        tf, sf = t.float(), s.float()
        cos = F.cosine_similarity(sf, tf, dim=-1)
        nmse = (F.normalize(sf, dim=-1) - F.normalize(tf, dim=-1)).pow(2).sum(-1)
        rel = float(torch.linalg.norm(sf - tf) / torch.linalg.norm(tf).clamp_min(1e-12))
        out[str(key)] = {"cosine": float(cos.mean()), "cosine_std": float(cos.std(unbiased=False)) if cos.numel() > 1 else 0.0,
                         "cosine_min": float(cos.min()), "normalized_mse": float(nmse.mean()), "relative_error": rel,
                         "n_positions": int(tf.shape[0])}
    return out


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def _fmt(x: Optional[float], nd: int = 1) -> str:
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{nd}f}"


def _ci(block: Optional[Dict[str, Any]], scale: float = 1.0, nd: int = 1) -> str:
    if not block or block.get("ci_low") is None:
        return ""
    return f" [{_fmt(block['ci_low'] * scale, nd)}, {_fmt(block['ci_high'] * scale, nd)}]"


def render_markdown(results: Dict[str, Any]) -> str:
    lines = [f"# KV recovery — {results.get('run_name', '')}", "",
             f"model `{results.get('model', {}).get('name')}` · compressor "
             f"`{(results.get('kv_compression') or {}).get('kv_compressor')}` @ ratio "
             f"{(results.get('kv_compression') or {}).get('compression_ratio')} · checkpoint "
             f"`{(results.get('checkpoint') or {}).get('sha256', '')[:12]}`", "",
             "| Benchmark | Task | n | Dense | Compressed | Recovered | Drop [CI] | Recovery [CI] | Recovery fraction [CI] | flags |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for bench, rep in (results.get("benchmarks") or {}).items():
        o = rep["overall"]
        frac = f"{_fmt(o['recovery_fraction'] * 100 if o['recovery_fraction'] is not None else None)}%"
        lines.append(f"| {bench} | **macro** | {o['n_examples']} | {_fmt(o['dense'])} | {_fmt(o['compressed'])} | "
                     f"{_fmt(o['compressed_recovered'])} | {_fmt(o['compression_drop'])}{_ci(o['ci'].get('compression_drop'))} | "
                     f"{_fmt(o['recovery'])}{_ci(o['ci'].get('recovery'))} | {frac}{_ci(o['ci'].get('recovery_fraction'), 100.0, 0)} | "
                     f"{', '.join(o['flags'])} |")
        for task, t in rep["tasks"].items():
            frac = f"{_fmt(t['recovery_fraction'] * 100 if t['recovery_fraction'] is not None else None)}%"
            lines.append(f"| {bench} | {task} | {t['n_examples']} | {_fmt(t['dense'])} | {_fmt(t['compressed'])} | "
                         f"{_fmt(t['compressed_recovered'])} | {_fmt(t['compression_drop'])}{_ci(t['ci'].get('compression_drop'))} | "
                         f"{_fmt(t['recovery'])}{_ci(t['ci'].get('recovery'))} | {frac}{_ci(t['ci'].get('recovery_fraction'), 100.0, 0)} | "
                         f"{', '.join(t['flags'])} |")
    agg = results.get("aggregate", {}).get("macro_over_benchmarks")
    if agg:
        lines += ["", f"Macro over benchmarks: dense {_fmt(agg['dense'])}, compressed {_fmt(agg['compressed'])}, "
                      f"recovered {_fmt(agg['compressed_recovered'])}, recovery fraction "
                      f"{_fmt(agg['recovery_fraction'] * 100 if agg['recovery_fraction'] is not None else None)}%"]
    return "\n".join(lines) + "\n"
