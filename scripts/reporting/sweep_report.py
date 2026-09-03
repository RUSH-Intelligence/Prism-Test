#!/usr/bin/env python
"""Generic sweep reporter — one command for any sweep, any model/benchmark.

Reads the per-cell manifest fragments a sweep writes
(``<root>/<model>/<benchmark>/manifest.cells/cell_*.json``), loads each cell's
``metrics.json`` (whose path is recorded IN the fragment), and emits a table of
scores. It never parses folder names, so the barcode-suffixed flat layout
(``Ridge_g2__r0.9__<fingerprint>/``) works out of the box.

Usage::

    python scripts/reporting/sweep_report.py                       # results/sweep -> stdout
    python scripts/reporting/sweep_report.py results/sweep
    python scripts/reporting/sweep_report.py --csv out.csv --xlsx out.xlsx

Rows = cells (method + ratio); columns = per-subset scores + overall, grouped by
(model, benchmark). One xlsx sheet per (model, benchmark).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

_META_COLS = ["model", "benchmark", "label", "ratio", "fingerprint",
              "decision", "ok", "total_samples", "overall"]


def _load_metrics(frag: dict) -> dict | None:
    """metrics.json for a cell — from the path in the fragment, else by searching
    the cell folder (robust to the flat/nested layout)."""
    p = frag.get("metrics")
    if p and Path(p).exists():
        try:
            return json.loads(Path(p).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
    return None


def _scalar(v):
    """A per-task score as one number, across benchmark shapes.

    LongBench stores ``task_scores = {task: float}``; RULER stores
    ``{task: {"string_match": float}}``. Flat numbers pass through; a
    ``{metric: value}`` dict collapses to the mean of its numeric values (a
    single-metric dict — RULER's — just yields that value). Anything with no
    numeric content returns None so the cell renders blank instead of a raw dict
    (which crashes openpyxl and garbles the CSV)."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return v
    if isinstance(v, dict):
        nums = [x for x in v.values() if isinstance(x, (int, float))
                and not isinstance(x, bool)]
        return round(sum(nums) / len(nums), 4) if nums else None
    return None


def _total_samples(metrics: dict, frag: dict):
    """total_samples, tolerating both layouts: top level (LongBench) or nested
    under ``summary`` (RULER); falls back to the manifest fragment."""
    if metrics.get("total_samples") is not None:
        return metrics["total_samples"]
    summ = metrics.get("summary") or {}
    if summ.get("total_samples") is not None:
        return summ["total_samples"]
    return frag.get("total_samples")


def collect(root: Path) -> list[dict]:
    """One record per cell fragment, with its scores loaded."""
    records: list[dict] = []
    for frag_path in sorted(root.rglob("manifest.cells/cell_*.json")):
        try:
            frag = json.loads(frag_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        metrics = _load_metrics(frag) or {}
        scores = metrics.get("task_scores") or {}
        records.append({
            "model": frag.get("model"),
            "benchmark": frag.get("benchmark"),
            "label": frag.get("cell_id") or frag.get("label"),
            "ratio": frag.get("ratio"),
            "fingerprint": frag.get("fingerprint"),
            "decision": frag.get("decision"),
            "ok": frag.get("ok"),
            "total_samples": _total_samples(metrics, frag),
            "overall": metrics.get("overall_score"),
            "scores": {str(k): _scalar(v) for k, v in scores.items()},
        })
    return records


def _sorted_key(r: dict):
    return (str(r["model"]), str(r["benchmark"]), str(r["label"]),
            r["ratio"] if r["ratio"] is not None else -1.0)


def write_csv(records: list[dict], path: Path) -> None:
    all_subsets = sorted({s for r in records for s in r["scores"]})
    header = _META_COLS + all_subsets
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in sorted(records, key=_sorted_key):
            row = [r.get(c) for c in _META_COLS]
            row += [r["scores"].get(s, "") for s in all_subsets]
            w.writerow(row)


def write_xlsx(records: list[dict], path: Path) -> None:
    from openpyxl import Workbook

    wb = Workbook()
    wb.remove(wb.active)
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        groups.setdefault((r["model"], r["benchmark"]), []).append(r)

    used_names: set[str] = set()
    for (model, benchmark), rows in sorted(groups.items(), key=lambda kv: str(kv[0])):
        subsets = sorted({s for r in rows for s in r["scores"]})
        name = f"{benchmark}-{str(model).split('/')[-1]}"[:31]
        base, n = name, 1
        while name in used_names:      # xlsx sheet names must be unique
            name = f"{base[:28]}_{n}"
            n += 1
        used_names.add(name)
        ws = wb.create_sheet(title=name)
        ws.append(["label", "ratio", "overall"] + subsets)
        for r in sorted(rows, key=_sorted_key):
            ws.append([r["label"], r["ratio"], r["overall"]]
                      + [r["scores"].get(s, "") for s in subsets])
    wb.save(path)


def print_table(records: list[dict]) -> None:
    if not records:
        print("No cells found.")
        return
    cur = None
    for r in sorted(records, key=_sorted_key):
        key = (r["model"], r["benchmark"])
        if key != cur:
            cur = key
            print(f"\n=== {r['model']}  /  {r['benchmark']} ===")
            print(f"  {'cell':40s} {'ratio':>6} {'overall':>8}  decision")
        ratio = "-" if r["ratio"] is None else f"{r['ratio']:.2f}"
        overall = "-" if r["overall"] is None else f"{r['overall']:.2f}"
        print(f"  {str(r['label']):40s} {ratio:>6} {overall:>8}  {r['decision']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", default=str(REPO_ROOT / "results" / "sweep"),
                    help="Sweep results root (default: results/sweep)")
    ap.add_argument("--csv", default=None, help="Write a flat CSV table here")
    ap.add_argument("--xlsx", default=None, help="Write an xlsx (one sheet per model/benchmark)")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        sys.exit(f"Sweep root not found: {root}")

    records = collect(root)
    print_table(records)
    if args.csv:
        write_csv(records, Path(args.csv))
        print(f"\nWrote CSV:  {args.csv}  ({len(records)} cells)")
    if args.xlsx:
        write_xlsx(records, Path(args.xlsx))
        print(f"Wrote xlsx: {args.xlsx}")


if __name__ == "__main__":
    main()
