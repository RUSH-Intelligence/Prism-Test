#!/usr/bin/env python
"""Fill a per-model RULER16k Ridge-improved sheet in ``Ridge Press.xlsx``.

The Ridge-improved RULER sweep (``scripts/slurm/ruler_ridge_improved_sweep.sbatch``)
runs the Track-3 winner — RidgeSketch with ``normalize_keys_for_tau=True`` and the
upstream-reference ``sink=4`` / ``local=28`` windows — over a γ × ratio grid, plus a
same-framework Compactor control at the same 4 ratios. This is the RULER analog of
the LongBench ``50_ridgeimp_longbench_*`` sheets.

Unlike ``ruler_ridge_ablation_to_xlsx.py`` (which reads ``manifest.cells`` fragments),
this script walks the results tree directly:

    results/ruler16k_ridge_improved/<model-slug>/
        Compactor__r<r>/ .../metrics.json
        Ridge_g<γ>_rqF_nkT_sk4_lo28__r<r>/ .../metrics.json

so it works even when the sweep's manifest was never merged, and it tolerates the
numbered resume-dedup subdirs (``.../<n>/metrics.json``) the runner writes when an
output dir already exists. Missing cells are left blank.

Sheet layout (mirrors ``50_ridgeimp_longbench_*``): one block per ratio, each with a
Compactor row then the Ridge γ rows, RULER subset headers + Avg.

Usage
-----
    python scripts/ruler_ridge_improved_to_xlsx.py
    python scripts/ruler_ridge_improved_to_xlsx.py \
        --results-dir results/ruler16k_ridge_improved/Qwen--Qwen3.5-9B \
        --sheet 32_ridgeimp_ruler_qwen
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import openpyxl
from openpyxl.styles import Font

REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # scripts/reporting/ -> repo root

# RULER subset name -> short display header (same order as the ablation sheets).
SUBSET_TO_HEADER = {
    "cwe": "CWE",
    "fwe": "FWE",
    "niah_multikey_1": "NIAH-MK1",
    "niah_multikey_2": "NIAH-MK2",
    "niah_multikey_3": "NIAH-MK3",
    "niah_multiquery": "NIAH-MQ",
    "niah_multivalue": "NIAH-MV",
    "niah_single_1": "NIAH-S1",
    "niah_single_2": "NIAH-S2",
    "niah_single_3": "NIAH-S3",
    "qa_1": "QA1",
    "qa_2": "QA2",
    "vt": "VT",
}
SUBSET_ORDER = list(SUBSET_TO_HEADER.keys())

RATIO_RE = re.compile(r"__r([0-9.]+)$")
GAMMA_RE = re.compile(r"^Ridge_g([0-9.]+)_")


def _load_metrics(cell_dir: Path) -> dict | None:
    """Return the single metrics.json's task_scores for a cell, or None.

    The runner writes either ``<cell>/ruler16k__.../metrics.json`` or, on resume,
    ``<cell>/ruler16k__.../<n>/metrics.json``. There is exactly one per cell; if
    more than one is found we take the most recently modified.
    """
    hits = sorted(glob.glob(str(cell_dir / "**" / "metrics.json"), recursive=True),
                  key=lambda p: Path(p).stat().st_mtime)
    if not hits:
        return None
    data = json.loads(Path(hits[-1]).read_text(encoding="utf-8"))
    return {sub: s["string_match"]
            for sub, s in data.get("task_scores", {}).items()
            if "string_match" in s}


def load_results(results_dir: Path) -> tuple[dict, list[float], list[float]]:
    """Return ({(method_or_gamma, ratio): scores}, ratios, gammas).

    Keys are ("Compactor", ratio) or (gamma, ratio).
    """
    out: dict = {}
    ratios, gammas = set(), set()
    for cell_dir in sorted(results_dir.iterdir()):
        if not cell_dir.is_dir() or cell_dir.name == "manifest.cells":
            continue
        m = RATIO_RE.search(cell_dir.name)
        if not m:
            continue
        ratio = float(m.group(1))
        ratios.add(ratio)
        scores = _load_metrics(cell_dir)
        if scores is None:
            continue
        if cell_dir.name.startswith("Compactor__"):
            out[("Compactor", ratio)] = scores
        else:
            g = GAMMA_RE.match(cell_dir.name)
            if not g:
                continue
            gamma = float(g.group(1))
            gammas.add(gamma)
            out[(gamma, ratio)] = scores
    return out, sorted(ratios), sorted(gammas)


def _avg(values: list) -> float | None:
    nums = [v for v in values if isinstance(v, (int, float))]
    return round(sum(nums) / len(nums), 2) if nums else None


def _write_row(ws, row: int, label: str, scores: dict, avg_col: int) -> None:
    ws.cell(row, 1, label)
    if not scores:
        return
    vals = []
    for c, sub in enumerate(SUBSET_ORDER, start=2):
        v = scores.get(sub)
        vals.append(v)
        if v is not None:
            cell = ws.cell(row, c, round(float(v), 2))
            cell.number_format = "0.00"
    a = _avg(vals)
    if a is not None:
        cell = ws.cell(row, avg_col, a)
        cell.number_format = "0.00"


def write_sheet(wb, sheet_name: str, model: str,
                ratios: list[float], gammas: list[float], results: dict) -> int:
    if sheet_name in wb.sheetnames:
        del wb[sheet_name]
    ws = wb.create_sheet(sheet_name)
    bold = Font(bold=True)
    headers = [SUBSET_TO_HEADER[s] for s in SUBSET_ORDER]
    avg_col = len(SUBSET_ORDER) + 2  # label + 13 subsets, Avg in col 15

    ws.cell(1, 1, "RULER16k").font = bold
    ws.cell(1, 2, model).font = bold

    row = 3
    written = 0
    for ratio in ratios:
        ws.cell(row, 1, f"{ratio:g}").font = bold
        for c, h in enumerate(headers, start=2):
            ws.cell(row, c, h).font = bold
        ws.cell(row, avg_col, "Avg").font = bold
        row += 1

        comp = results.get(("Compactor", ratio), {})
        _write_row(ws, row, "Compactor", comp, avg_col)
        written += bool(comp)
        row += 1

        for gamma in gammas:
            scores = results.get((gamma, ratio), {})
            _write_row(ws, row, f"Ridge_g{gamma:g}", scores, avg_col)
            written += bool(scores)
            row += 1
        row += 1  # blank separator
    return written


def _infer_model(results_dir: Path) -> str:
    return results_dir.name.replace("--", "/", 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir",
                    default=str(REPO_ROOT / "results" / "ruler16k_ridge_improved"
                                / "Qwen--Qwen3.5-9B"),
                    help="Path to results/ruler16k_ridge_improved/<model-slug>/")
    ap.add_argument("--xlsx", default=str(REPO_ROOT / "Ridge Press.xlsx"))
    ap.add_argument("--sheet", default="32_ridgeimp_ruler_qwen")
    ap.add_argument("--position", type=int, default=None,
                    help="0-based tab index to move the sheet to (default: leave at end)")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    model = _infer_model(results_dir)
    results, ratios, gammas = load_results(results_dir)
    if not results:
        raise SystemExit(f"No completed cells with metrics under {results_dir}.")

    xlsx = Path(args.xlsx)
    wb = openpyxl.load_workbook(xlsx)
    written = write_sheet(wb, args.sheet, model, ratios, gammas, results)
    if args.position is not None:
        idx = wb.sheetnames.index(args.sheet)
        wb.move_sheet(args.sheet, offset=args.position - idx)
    wb.save(xlsx)

    total = (len(gammas) + 1) * len(ratios)  # +1 Compactor row per ratio
    print(f"Wrote sheet '{args.sheet}' to {xlsx} "
          f"({written}/{total} cells with data; ratios={ratios}, gammas={gammas}).")
    missing = [f"Ridge_g{g:g}__r{r:g}" for r in ratios for g in gammas
               if (g, r) not in results]
    missing += [f"Compactor__r{r:g}" for r in ratios if ("Compactor", r) not in results]
    if missing:
        print(f"Missing cells (left blank): {', '.join(missing)}")


if __name__ == "__main__":
    main()
