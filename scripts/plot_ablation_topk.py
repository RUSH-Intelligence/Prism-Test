#!/usr/bin/env python
"""Figures for the layer-count ablation (``--preset ablation_topk``): recovery versus the number of
calibrated layers, and the per-task breakdown of the best ladders.

  python scripts/plot_ablation_topk.py [--runs 'outputs/kv_recovery/*_mix*'] [--out-dir outputs/kv_recovery/figures] \
      [--formats png,svg] [--dpi 160]

Reads every run's ``eval_results.json`` (written by ``eval_kv_recovery.py report``) and draws

* ``ablation_topk__ladders``      — small multiples: rows = model × training corpus, columns = benchmark;
                                    recovery (recovered − compressed, points) vs k ∈ {4, 8, 16} with paired-bootstrap
                                    95 % CIs; hue = compressor (knorm blue, cur orange), line style = projections
                                    (q+o solid, k+v dashed).
* ``ablation_topk__tasks__<model>`` — per-task recovery on RULER-16K for that model's best ladder (the
                                    compressor × projections × corpus with the largest top-k recovery), grouped
                                    bars for k = 4 / 8 / 16 (one-hue ordinal ramp), CI whiskers.

Static matplotlib figures, no torch; figures are written outside the repository by default.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SURFACE = "#fcfcfb"
INK, INK2, INK3 = "#1a1a19", "#5c5b55", "#8a897f"
GRID = "#e6e5e0"
SERIES = {"knorm": "#2a78d6", "cur": "#eb6834"}
STYLE = {"qo": "-", "kv": "--"}
FAMILY = {"qo": "q+o", "kv": "k+v"}
RAMP = {4: "#86b6ef", 8: "#2a78d6", 16: "#104281"}            # ordinal blue ramp, steps 250 / 450 / 650
MODEL_NAME = {"ministral_3b": "Ministral-3-3B", "qwen35_4b": "Qwen3.5-4B"}
BENCH_NAME = {"ruler16k": "RULER-16K", "ruler32k": "RULER-32K", "longbench": "LongBench (rows 0–99)"}
BENCHES = ["ruler16k", "ruler32k", "longbench"]
TASK_ORDER = ["niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
              "niah_multiquery", "niah_multivalue", "cwe", "fwe", "vt", "qa_1", "qa_2"]
RUN_RE = re.compile(r"(ministral_3b|qwen35_4b)_mix(16k|32k)_(knorm|cur)_r075_(qo|kv)_sens(\d+)$")


def load_runs(pattern: str) -> List[Dict[str, Any]]:
    out = []
    for d in sorted(glob.glob(pattern)):
        name = os.path.basename(d)
        m = RUN_RE.match(name)
        p = os.path.join(d, "eval_results.json")
        if not m or not os.path.exists(p):
            continue
        model, ctx, comp, fam, k = m.groups()
        res = json.load(open(p))
        out.append({"run": name, "model": model, "corpus": f"mix{ctx}", "compressor": comp, "family": fam, "k": int(k),
                    "benchmarks": res.get("benchmarks", {})})
    return out


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(True, axis="y", color=GRID, linewidth=1.0)
    ax.grid(False, axis="x")
    ax.tick_params(colors=INK2, labelsize=8.5, length=0)
    ax.axhline(0, color=INK3, linewidth=1.0, zorder=1)


def _save(fig, path: str) -> None:
    fig.savefig(path, facecolor=SURFACE)
    if path.endswith(".svg"):
        text = Path(path).read_text(encoding="utf-8")
        Path(path).write_text("\n".join(line.rstrip() for line in text.splitlines()) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# figure 1: ladders
# ---------------------------------------------------------------------------
def plot_ladders(runs: List[Dict[str, Any]], out_dir: Path, formats: List[str], dpi: int) -> Optional[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in runs:
        groups[(r["model"], r["corpus"])].append(r)
    rows = [(m, c) for m in ("ministral_3b", "qwen35_4b") for c in ("mix16k", "mix32k") if (m, c) in groups]
    if not rows:
        return None
    fig, axes = plt.subplots(len(rows), len(BENCHES), figsize=(4.6 * len(BENCHES), 2.9 * len(rows) + 0.8), dpi=dpi, sharex=True)
    fig.patch.set_facecolor(SURFACE)
    axes = [list(a) for a in axes] if len(rows) > 1 else [list(axes)]
    for i, (model, corpus) in enumerate(rows):
        for j, bench in enumerate(BENCHES):
            ax = axes[i][j]
            _style_axes(ax)
            lo_all, hi_all = 0.0, 0.0
            for comp in ("knorm", "cur"):
                for fam in ("qo", "kv"):
                    pts = sorted([r for r in groups[(model, corpus)] if r["compressor"] == comp and r["family"] == fam and bench in r["benchmarks"]],
                                 key=lambda r: r["k"])
                    if not pts:
                        continue
                    ks = [r["k"] for r in pts]
                    rec = [r["benchmarks"][bench]["overall"]["recovery"] for r in pts]
                    ci = [r["benchmarks"][bench]["overall"]["ci"]["recovery"] for r in pts]
                    lo = [c["ci_low"] for c in ci]; hi = [c["ci_high"] for c in ci]
                    lo_all, hi_all = min(lo_all, min(lo)), max(hi_all, max(hi))
                    color = SERIES[comp]
                    ax.errorbar(ks, rec, yerr=[[a - b for a, b in zip(rec, lo)], [b - a for a, b in zip(rec, hi)]],
                                fmt=STYLE[fam], color=color, linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE,
                                markeredgewidth=0.8, capsize=2.5, elinewidth=1.0, ecolor=color, alpha=0.95, zorder=3)
            ax.set_xticks([4, 8, 16]); ax.set_xscale("log", base=2); ax.set_xticks([4, 8, 16]); ax.set_xticklabels(["top-4", "top-8", "top-16"])
            ax.minorticks_off()
            pad = 0.08 * max(hi_all - lo_all, 1.0)
            ax.set_ylim(min(lo_all, 0) - pad, hi_all + pad)
            if i == 0:
                ax.set_title(BENCH_NAME[bench], fontsize=10, color=INK, loc="left", pad=6)
            if j == 0:
                ax.set_ylabel(f"{MODEL_NAME[model]}\ntrained on {corpus}\nrecovery (points)", fontsize=8.5, color=INK2)
            for lbl in ax.get_xticklabels() + ax.get_yticklabels():
                lbl.set_color(INK2)
    handles = [Line2D([0], [0], color=SERIES["knorm"], lw=2, label="knorm"), Line2D([0], [0], color=SERIES["cur"], lw=2, label="cur"),
               Line2D([0], [0], color=INK2, lw=2, ls="-", label="q_proj + o_proj"), Line2D([0], [0], color=INK2, lw=2, ls="--", label="k_proj + v_proj")]
    fig.legend(handles=handles, loc="upper right", ncol=4, frameon=False, fontsize=8.5, labelcolor=INK2, bbox_to_anchor=(0.99, 0.985))
    fig.suptitle("Recovery vs number of calibrated layers (top-k most compression-sensitive)", fontsize=12, color=INK, x=0.01, ha="left", y=0.99)
    fig.text(0.01, 0.95, "recovered − compressed, points; whiskers = paired bootstrap 95 % CI; ratio 0.75; 256 steps on 1 024 mixed windows; "
                         "Qwen3.5 has 8 K/V layers (top-8 = all)", fontsize=8.5, color=INK3, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    base = out_dir / "ablation_topk__ladders"
    for fmt in formats:
        _save(fig, f"{base}.{fmt}")
    plt.close(fig)
    return Path(f"{base}.{formats[0]}")


# ---------------------------------------------------------------------------
# figure 2: per-task breakdown of the best ladder per model
# ---------------------------------------------------------------------------
def best_ladder(runs: List[Dict[str, Any]], model: str, bench: str = "ruler16k") -> Optional[Tuple[str, str, str]]:
    best, key = None, None
    for r in runs:
        if r["model"] != model or bench not in r["benchmarks"]:
            continue
        rec = r["benchmarks"][bench]["overall"]["recovery"]
        if best is None or rec > best:
            best, key = rec, (r["corpus"], r["compressor"], r["family"])
    return key


def plot_tasks(runs: List[Dict[str, Any]], model: str, out_dir: Path, formats: List[str], dpi: int, bench: str = "ruler16k") -> Optional[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    key = best_ladder(runs, model, bench)
    if key is None:
        return None
    corpus, comp, fam = key
    ladder = sorted([r for r in runs if r["model"] == model and (r["corpus"], r["compressor"], r["family"]) == key and bench in r["benchmarks"]],
                    key=lambda r: r["k"])
    tasks_all = ladder[0]["benchmarks"][bench]["tasks"]
    tasks = [t for t in TASK_ORDER if t in tasks_all] + sorted(set(tasks_all) - set(TASK_ORDER))
    fig, ax = plt.subplots(figsize=(13, 4.8), dpi=dpi)
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax)
    n = len(ladder)
    width = 0.8 / n
    xs = list(range(len(tasks)))
    ymin, ymax = 0.0, 0.0
    for idx, r in enumerate(ladder):
        tb = r["benchmarks"][bench]["tasks"]
        rec = [tb[t]["recovery"] for t in tasks]
        lo = [tb[t]["ci"]["recovery"]["ci_low"] for t in tasks]
        hi = [tb[t]["ci"]["recovery"]["ci_high"] for t in tasks]
        ymin, ymax = min(ymin, min(lo)), max(ymax, max(hi))
        pos = [x - 0.4 + width * (idx + 0.5) for x in xs]
        ax.bar(pos, rec, width=width * 0.92, color=RAMP.get(r["k"], SERIES["knorm"]), edgecolor=SURFACE, linewidth=0.8, zorder=3,
               label=f"top-{r['k']}")
        ax.errorbar(pos, rec, yerr=[[a - b for a, b in zip(rec, lo)], [b - a for a, b in zip(rec, hi)]], fmt="none", ecolor=INK2,
                    elinewidth=0.8, capsize=1.5, zorder=4)
    comp_line = ladder[0]["benchmarks"][bench]["tasks"]
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{t.replace('niah_', '')}\n({comp_line[t]['compressed']:.0f} → {comp_line[t]['dense']:.0f})" for t in tasks],
                       fontsize=8, color=INK2)
    ax.text(0.0, -0.17, "niah_* tasks shown without the prefix", transform=ax.transAxes, fontsize=7.5, color=INK3, ha="left")
    ax.set_ylabel("recovery (points)", fontsize=9, color=INK2)
    pad = 0.08 * max(ymax - ymin, 1.0)
    ax.set_ylim(ymin - pad, ymax + pad)
    handles = [Patch(color=RAMP.get(r["k"], SERIES["knorm"]), label=f"top-{r['k']} ({r['benchmarks'][bench]['overall']['recovery']:+.1f} macro)") for r in ladder]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=8.5, labelcolor=INK2, ncol=n)
    fig.suptitle(f"Per-task recovery on {BENCH_NAME[bench]} — {MODEL_NAME[model]} · {comp} · {FAMILY[fam]} · trained on {corpus}",
                 fontsize=12, color=INK, x=0.01, ha="left", y=0.99)
    fig.text(0.01, 0.92, "bars = recovered − compressed per task (whiskers: paired bootstrap 95 % CI); x labels show compressed → dense accuracy",
             fontsize=8.5, color=INK3, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    base = out_dir / f"ablation_topk__tasks__{model}"
    for fmt in formats:
        _save(fig, f"{base}.{fmt}")
    plt.close(fig)
    return Path(f"{base}.{formats[0]}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="outputs/kv_recovery/*_mix*", help="glob of run directories holding eval_results.json")
    ap.add_argument("--out-dir", default="outputs/kv_recovery/figures")
    ap.add_argument("--formats", default="png,svg")
    ap.add_argument("--dpi", type=int, default=160)
    args = ap.parse_args(argv)
    runs = load_runs(args.runs)
    if not runs:
        raise SystemExit(f"no evaluated ablation runs under {args.runs}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    written = [p for p in [plot_ladders(runs, out_dir, formats, args.dpi)] if p]
    for model in sorted({r["model"] for r in runs}):
        p = plot_tasks(runs, model, out_dir, formats, args.dpi)
        if p:
            written.append(p)
    for p in written:
        print("wrote", p)
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
