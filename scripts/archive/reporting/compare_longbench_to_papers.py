"""Compare LongBench sweep results against published reference numbers.

Standalone read-only script. Does NOT import from eval_harness and does NOT
write anywhere the sweep reads from. Safe to run while the sweep is live.

Usage:
    python scripts/compare_longbench_to_papers.py \
        --sweep-dir /fs/nexus-scratch/aysingha/workspace/Prism-Test/results/longbench_sweep/meta-llama--Llama-3.1-8B-Instruct

    # Only show Full vs published baseline:
    python scripts/compare_longbench_to_papers.py --sweep-dir ... --cells Full

    # Flag drift greater than N points:
    python scripts/compare_longbench_to_papers.py --sweep-dir ... --tolerance 3.0

Reference numbers live in REFERENCE_SCORES below. The Full baseline numbers
should be filled in from one (or both) of:
    - LongBench paper / THUDM/LongBench leaderboard (Llama-3-8B-Instruct row,
      closest published proxy for 3.1)
    - kvpress README leaderboard (NVIDIA/kvpress) — the most relevant source,
      since the sweep's compressors are kvpress 0.5.1 ports

Leave a value as None for any task/cell you don't have a reference for; the
script will just skip the comparison for that cell.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Reference scores (fill in from papers / kvpress leaderboard).
#
# Schema:
#   REFERENCE_SCORES[cell_name] = {
#       "source": "<where this came from>",
#       "task_scores": {task_name: score_or_None, ...},
#       "overall": float or None,
#   }
#
# Task names must match the keys in metrics.json exactly. The sweep uses these
# 16 LongBench subsets (English-only):
TASK_NAMES = [
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "gov_report",
    "qmsum",
    "multi_news",
    "trec",
    "triviaqa",
    "samsum",
    "passage_count",
    "passage_retrieval_en",
    "lcc",
    "repobench-p",
]

# TODO(user): fill these in. Numbers below are placeholders set to None so
# unfilled cells skip cleanly. Pull from:
#   - kvpress README leaderboard (Llama-3.1-8B-Instruct, no compression): Full
#   - kvpress README leaderboard at the matching compression ratio: each method
#   - LongBench paper Table 4 (Llama-3-8B-Instruct) as a cross-check for Full
REFERENCE_SCORES: dict[str, dict] = {
    "Full": {
        "source": "kvpress README, Llama-3.1-8B-Instruct, no compression (FILL IN)",
        "overall": None,
        "task_scores": {t: None for t in TASK_NAMES},
    },
    # Per-method references (compression_ratio = fraction dropped).
    # Add entries you can find published numbers for; leave others out.
    # Example shape:
    # "Knorm__r0.6": {
    #     "source": "kvpress leaderboard, knorm @ 0.6 compression",
    #     "overall": None,
    #     "task_scores": {t: None for t in TASK_NAMES},
    # },
}


# ---------------------------------------------------------------------------
def find_metrics(cell_dir: Path) -> Optional[Path]:
    """Return the metrics.json inside the single longbench__* run dir, or None."""
    runs = sorted(cell_dir.glob("longbench__*/metrics.json"))
    if not runs:
        return None
    if len(runs) > 1:
        print(f"  warning: {cell_dir.name} has {len(runs)} run dirs; using newest",
              file=sys.stderr)
        runs.sort(key=lambda p: p.stat().st_mtime)
    return runs[-1]


def load_metrics(metrics_path: Path) -> dict:
    with metrics_path.open() as f:
        return json.load(f)


def fmt_delta(measured: Optional[float], reference: Optional[float],
              tolerance: float) -> str:
    if measured is None or reference is None:
        return ""
    delta = measured - reference
    flag = "  !!" if abs(delta) > tolerance else ""
    sign = "+" if delta >= 0 else ""
    return f"  ({sign}{delta:.2f}){flag}"


def compare_cell(cell_dir: Path, tolerance: float) -> None:
    name = cell_dir.name
    metrics_path = find_metrics(cell_dir)
    if metrics_path is None:
        print(f"\n[{name}] no metrics.json yet (cell incomplete)")
        return

    metrics = load_metrics(metrics_path)
    measured_overall = metrics.get("overall_score")
    measured_tasks = metrics.get("task_scores", {})

    ref = REFERENCE_SCORES.get(name)
    if ref is None:
        print(f"\n[{name}] no reference numbers in REFERENCE_SCORES — measured only")
        print(f"  overall_score: {measured_overall}")
        for task in TASK_NAMES:
            print(f"    {task:<24} {measured_tasks.get(task, '—')}")
        return

    print(f"\n[{name}] source: {ref['source']}")
    print(f"  overall: measured={measured_overall}  ref={ref.get('overall')}"
          f"{fmt_delta(measured_overall, ref.get('overall'), tolerance)}")
    ref_tasks = ref.get("task_scores", {})
    for task in TASK_NAMES:
        m = measured_tasks.get(task)
        r = ref_tasks.get(task)
        line = f"    {task:<24} measured={m}  ref={r}"
        line += fmt_delta(m, r, tolerance)
        print(line)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", required=True, type=Path,
                        help="Per-model sweep root, e.g. results/longbench_sweep/meta-llama--Llama-3.1-8B-Instruct")
    parser.add_argument("--cells", nargs="*", default=None,
                        help="Specific cell names to compare; defaults to all")
    parser.add_argument("--tolerance", type=float, default=3.0,
                        help="Flag (!!) deltas larger than this many points")
    args = parser.parse_args()

    if not args.sweep_dir.is_dir():
        print(f"error: sweep dir not found: {args.sweep_dir}", file=sys.stderr)
        return 2

    if args.cells:
        cell_dirs = [args.sweep_dir / c for c in args.cells]
    else:
        cell_dirs = [p for p in sorted(args.sweep_dir.iterdir())
                     if p.is_dir() and not p.name.startswith("_")
                     and p.name != "manifest.cells"]

    print(f"Comparing {len(cell_dirs)} cell(s) under {args.sweep_dir}")
    print(f"Tolerance for flagging drift: ±{args.tolerance} pts")
    for cell_dir in cell_dirs:
        if not cell_dir.is_dir():
            print(f"\n[{cell_dir.name}] missing dir; skipping")
            continue
        compare_cell(cell_dir, args.tolerance)

    return 0


if __name__ == "__main__":
    sys.exit(main())
