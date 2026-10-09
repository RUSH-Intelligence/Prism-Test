#!/usr/bin/env python
"""Figures for the layer-wise compression sensitivity E_l (``scripts/measure_layer_sensitivity.py`` output).

  python scripts/plot_layer_sensitivity.py --inputs outputs/kv_recovery/sensitivity --out-dir docs/figures \
      [--models mistralai/Ministral-3-3B-Instruct-2512,Qwen/Qwen3.5-4B] [--formats png,svg] [--dpi 160]

Reads every ``*.json`` the measurement script wrote (schema 1 = PG-19 calibration windows, schema 2 =
``source`` field: pg19 | ruler16k | ruler32k | longbench) and draws, per model:

* ``<model>__profiles``   — E_l vs layer, one panel per source (PG-19 16K reference, RULER-16K, RULER-32K),
                            hue = compressor (knorm blue, cur orange), line style = ratio (0.75 solid, 0.5 dashed),
                            shaded ± std across windows, non-K/V layers (hybrids) greyed, direct end labels.
* ``<model>__tasks__<compressor>_r<ratio>`` — small multiples, one panel per RULER task, RULER-16K vs RULER-32K.

Static matplotlib figures (no torch); text uses neutral ink, series colours only on marks.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# palette (validated reference instance: categorical slots in fixed order; light surface)
# ---------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK, INK2, INK3 = "#1a1a19", "#5c5b55", "#8a897f"
GRID = "#e6e5e0"
SHADE = "#efeeea"
SERIES = {"knorm": "#2a78d6", "cur": "#eb6834"}            # slot 1 blue, slot 2 orange (hue follows the compressor)
LENGTH_SERIES = {"ruler16k": "#2a78d6", "ruler32k": "#eb6834", "pg19": "#1baf7a"}   # slots 1, 2, 3
STYLE = {0.75: "-", 0.5: "--"}                              # ratio = line style (secondary, non-colour channel)
SOURCE_TITLE = {"pg19": "PG-19 held-out windows (16K, continuation)", "ruler16k": "RULER-16K (question + answer)",
                "ruler32k": "RULER-32K (question + answer)", "longbench": "LongBench-16 (question + answer)"}
SOURCE_ORDER = ["pg19", "ruler16k", "ruler32k", "longbench"]
TASK_ORDER = ["niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
              "niah_multiquery", "niah_multivalue", "cwe", "fwe", "vt", "qa_1", "qa_2"]


def _slug(model: str) -> str:
    return model.replace("/", "--")


def _ratio_tag(r: float) -> str:
    return f"r{int(round(float(r) * 100)):03d}"


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load_measurements(inputs: str) -> List[Dict[str, Any]]:
    paths = sorted(glob.glob(os.path.join(inputs, "**", "*.json"), recursive=True)) if os.path.isdir(inputs) else sorted(glob.glob(inputs))
    out = []
    for p in paths:
        if os.path.basename(p).startswith("summary"):
            continue
        try:
            d = json.load(open(p))
        except Exception:
            continue
        if "report" not in d or "compressor" not in d:
            continue
        d["source"] = d.get("source") or d.get("protocol", {}).get("source") or "pg19"
        d["_path"] = p
        out.append(d)
    return out


def per_task_curves(rep: Dict[str, Any]) -> Dict[str, Tuple[List[int], List[float]]]:
    """``{task: (layers, mean E_l over that task's windows)}`` from ``per_example`` (ids ``bench/task/rowN``)."""
    layers = [int(l) for l in rep["layers"]]
    acc: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    for ex_id, scores in rep["per_example"].items():
        parts = ex_id.split("/")
        task = parts[1] if len(parts) >= 3 else "all"
        acc[task].append({int(k): float(v) for k, v in scores.items()})
    out = {}
    for task, rows in acc.items():
        out[task] = (layers, [sum(r[l] for r in rows) / len(rows) for l in layers])
    return out


# ---------------------------------------------------------------------------
# drawing helpers
# ---------------------------------------------------------------------------
def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(1.0)
    ax.grid(True, axis="y", color=GRID, linewidth=1.0, linestyle="-")
    ax.grid(False, axis="x")
    ax.tick_params(colors=INK2, labelsize=8.5, length=0)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK2)


def _shade_non_kv(ax, layers: List[int], hooked: List[int]):
    """Grey bands over layers that carry no K/V cache (hybrid models) — pruning cannot touch them directly."""
    hooked_set = set(hooked)
    runs: List[Tuple[int, int]] = []
    for l in layers:
        if l in hooked_set:
            continue
        if runs and runs[-1][1] == l - 1:
            runs[-1] = (runs[-1][0], l)
        else:
            runs.append((l, l))
    for a, b in runs:
        ax.axvspan(a - 0.5, b + 0.5, color=SHADE, zorder=0, linewidth=0)
    return bool(runs)


def _end_label(ax, x: float, y: float, text: str, color: str, dy: float = 0.0):
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points", fontsize=8, color=INK2, va="center", ha="left",
                bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.85))


def _spread_labels(items: List[Tuple[float, str, str]], min_gap: float) -> List[float]:
    """Nudge end-label y positions apart (data units) so neighbouring labels do not collide."""
    order = sorted(range(len(items)), key=lambda i: items[i][0])
    ys = [items[i][0] for i in order]
    for k in range(1, len(ys)):
        if ys[k] - ys[k - 1] < min_gap:
            ys[k] = ys[k - 1] + min_gap
    out = [0.0] * len(items)
    for k, i in enumerate(order):
        out[i] = ys[k]
    return out


# ---------------------------------------------------------------------------
# figure 1: profiles per source
# ---------------------------------------------------------------------------
def plot_profiles(model: str, meas: List[Dict[str, Any]], out_dir: Path, formats: List[str], dpi: int) -> Optional[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for d in meas:
        by_source[d["source"]].append(d)
    sources = [s for s in SOURCE_ORDER if s in by_source]
    if not sources:
        return None
    n = len(sources)
    fig, axes = plt.subplots(1, n, figsize=(5.6 * n, 4.6), sharey=True, dpi=dpi)
    axes = list(axes) if n > 1 else [axes]
    fig.patch.set_facecolor(SURFACE)
    ymax = 0.0
    for ax, source in zip(axes, sources):
        _style_axes(ax)
        items = sorted(by_source[source], key=lambda d: (d["compressor"], -float(d["compression_ratio"])))
        rep0 = items[0]["report"]
        layers = [int(l) for l in rep0["layers"]]
        shaded = _shade_non_kv(ax, layers, [int(l) for l in rep0.get("hooked_layers", layers)])
        labels: List[Tuple[float, str, str]] = []
        for d in items:
            rep = d["report"]
            comp, ratio = d["compressor"], float(d["compression_ratio"])
            xs = [int(l) for l in rep["layers"]]
            ys = [float(rep["scores"][str(l)]) for l in xs]
            sd = [float(rep["std"].get(str(l), 0.0)) for l in xs]
            color = SERIES.get(comp, INK3)
            ax.fill_between(xs, [y - s for y, s in zip(ys, sd)], [y + s for y, s in zip(ys, sd)], color=color,
                            alpha=0.10 if ratio == 0.75 else 0.06, linewidth=0, zorder=1)
            ax.plot(xs, ys, STYLE.get(ratio, "-"), color=color, linewidth=2, solid_capstyle="round", zorder=3,
                    marker="o", markersize=3.6, markeredgecolor=SURFACE, markeredgewidth=0.8)
            ymax = max(ymax, max(y + s for y, s in zip(ys, sd)))
            labels.append((ys[-1], f"{comp} @ {ratio:g}", color))
        n_ex = items[0].get("protocol", {}).get("n_examples", "?")
        ax.set_title(f"{SOURCE_TITLE.get(source, source)} — {n_ex} windows", fontsize=10, color=INK, loc="left", pad=8)
        ax.set_xlabel("decoder layer", fontsize=9, color=INK2)
        ax.set_xlim(min(layers) - 0.6, max(layers) + 0.6)
        step = 2 if len(layers) <= 32 else 4
        ax.set_xticks([l for l in layers if l % step == 0])
        if labels:
            ys_adj = _spread_labels(labels, min_gap=0.075 * max(ymax, 1e-6))
            for (y, text, color), ya in zip(labels, ys_adj):
                ax.annotate(text, (layers[-1], y), xytext=(layers[-1] + 0.4, ya), textcoords="data", fontsize=8, color=INK2,
                            va="center", ha="left", arrowprops=dict(arrowstyle="-", color=GRID, lw=0.8) if abs(ya - y) > 1e-9 else None)
        if shaded:
            ax.text(0.99, 0.02, "grey: no K/V cache (linear attention)", transform=ax.transAxes, fontsize=7.5, color=INK3,
                    ha="right", va="bottom")
    axes[0].set_ylabel("E_l = ‖H_dense − H_comp‖_F / (‖H_dense‖_F + ε)", fontsize=9, color=INK2)
    for ax in axes:
        ax.set_ylim(0, ymax * 1.12 if ymax > 0 else 1)
        ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1] + 4.0)      # room for the end labels
    handles = [Line2D([0], [0], color=SERIES["knorm"], lw=2, label="knorm"), Line2D([0], [0], color=SERIES["cur"], lw=2, label="cur"),
               Line2D([0], [0], color=INK2, lw=2, ls="-", label="ratio 0.75 (keep 25 %)"),
               Line2D([0], [0], color=INK2, lw=2, ls="--", label="ratio 0.5 (keep 50 %)")]
    fig.legend(handles=handles, loc="upper right", ncol=4, frameon=False, fontsize=8.5, labelcolor=INK2, bbox_to_anchor=(0.99, 0.975))
    fig.suptitle(f"Hidden-state misalignment per layer — {model}", fontsize=12, color=INK, x=0.01, ha="left", y=0.985)
    fig.text(0.01, 0.915, "mean over windows, band = ±1 std across windows; measured on the tokens processed after compression",
             fontsize=8.5, color=INK3, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    base = out_dir / f"{_slug(model)}__profiles"
    for fmt in formats:
        fig.savefig(f"{base}.{fmt}", facecolor=SURFACE)
    plt.close(fig)
    return Path(f"{base}.{formats[0]}")


# ---------------------------------------------------------------------------
# figure 2: per-task small multiples, RULER-16K vs RULER-32K
# ---------------------------------------------------------------------------
def plot_tasks(model: str, meas: List[Dict[str, Any]], compressor: str, ratio: float, out_dir: Path, formats: List[str],
               dpi: int) -> Optional[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    sel = {d["source"]: d for d in meas if d["compressor"] == compressor and abs(float(d["compression_ratio"]) - ratio) < 1e-9
           and d["source"] in ("ruler16k", "ruler32k")}
    if not sel:
        return None
    curves = {src: per_task_curves(d["report"]) for src, d in sel.items()}
    tasks = [t for t in TASK_ORDER if any(t in c for c in curves.values())]
    tasks += sorted({t for c in curves.values() for t in c} - set(tasks))
    if not tasks:
        return None
    ncol = 5 if len(tasks) > 12 else 4
    nrow = math.ceil(len(tasks) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 2.7 * nrow + 1.0), sharex=True, sharey=True, dpi=dpi)
    fig.patch.set_facecolor(SURFACE)
    axes_flat = [a for row in (axes if nrow > 1 else [axes]) for a in (row if ncol > 1 else [row])]
    rep0 = next(iter(sel.values()))["report"]
    layers = [int(l) for l in rep0["layers"]]
    hooked = [int(l) for l in rep0.get("hooked_layers", layers)]
    ymax = 0.0
    for ax, task in zip(axes_flat, tasks):
        _style_axes(ax)
        _shade_non_kv(ax, layers, hooked)
        for src in ("ruler16k", "ruler32k"):
            if src in curves and task in curves[src]:
                xs, ys = curves[src][task]
                ax.plot(xs, ys, "-", color=LENGTH_SERIES[src], linewidth=2, solid_capstyle="round", zorder=3,
                        marker="o", markersize=2.8, markeredgecolor=SURFACE, markeredgewidth=0.6)
                ymax = max(ymax, max(ys))
        ax.set_title(task, fontsize=9, color=INK, loc="left", pad=4)
        step = 4 if len(layers) <= 32 else 8
        ax.set_xticks([l for l in layers if l % step == 0])
    for ax in axes_flat[len(tasks):]:
        ax.axis("off")
    for ax in axes_flat[:len(tasks)]:
        ax.set_ylim(0, ymax * 1.08 if ymax > 0 else 1)
    for ax in (axes_flat[i] for i in range(len(tasks)) if i // ncol == nrow - 1 or i + ncol >= len(tasks)):
        ax.set_xlabel("layer", fontsize=8.5, color=INK2)
    for r in range(nrow):
        axes_flat[r * ncol].set_ylabel("E_l", fontsize=8.5, color=INK2)
    n16 = sel.get("ruler16k", {}).get("protocol", {}).get("rows_per_task", "?")
    handles = [Line2D([0], [0], color=LENGTH_SERIES["ruler16k"], lw=2, label="RULER-16K"),
               Line2D([0], [0], color=LENGTH_SERIES["ruler32k"], lw=2, label="RULER-32K")]
    fig.legend(handles=handles, loc="upper right", ncol=2, frameon=False, fontsize=9, labelcolor=INK2, bbox_to_anchor=(0.99, 0.985))
    fig.suptitle(f"Per-task misalignment — {model} · {compressor} @ ratio {ratio:g}", fontsize=12, color=INK, x=0.01, ha="left", y=0.99)
    fig.text(0.01, 0.945, f"mean over {n16} rows per task; measured on question + answer tokens (evaluation prompt shaping)",
             fontsize=8.5, color=INK3, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    base = out_dir / f"{_slug(model)}__tasks__{compressor}_{_ratio_tag(ratio)}"
    for fmt in formats:
        fig.savefig(f"{base}.{fmt}", facecolor=SURFACE)
    plt.close(fig)
    return Path(f"{base}.{formats[0]}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", default="outputs/kv_recovery/sensitivity", help="directory (searched recursively) or glob of measurement JSONs")
    ap.add_argument("--out-dir", default="docs/figures")
    ap.add_argument("--models", help="comma list of model names (default: every model found)")
    ap.add_argument("--formats", default="png,svg")
    ap.add_argument("--dpi", type=int, default=160)
    ap.add_argument("--task-ratio", type=float, default=0.75, help="ratio for the per-task figures")
    args = ap.parse_args(argv)
    meas = load_measurements(args.inputs)
    if not meas:
        raise SystemExit(f"no measurement JSONs under {args.inputs}")
    models = [m.strip() for m in args.models.split(",")] if args.models else sorted({d["model"] for d in meas})
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    written: List[Path] = []
    for model in models:
        mm = [d for d in meas if d["model"] == model]
        if not mm:
            print(f"{model}: no measurements")
            continue
        p = plot_profiles(model, mm, out_dir, formats, args.dpi)
        if p:
            written.append(p)
        for comp in sorted({d["compressor"] for d in mm}):
            p = plot_tasks(model, mm, comp, args.task_ratio, out_dir, formats, args.dpi)
            if p:
                written.append(p)
    for p in written:
        print("wrote", p)
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
