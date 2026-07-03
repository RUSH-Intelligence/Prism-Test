#!/usr/bin/env python
"""Fill a per-model RULER16k Ridge gamma x lambda ablation sheet in ``Ridge Press.xlsx``.

Reads the per-cell manifest fragments written by ``scripts/longbench_sweep.py``
when run with ``--benchmark ruler16k --methods ridge --cell-index ...`` (the
SLURM array path in ``scripts/slurm/ruler_ridge_sweep.sbatch``), follows each
cell's ``metrics.json`` (produced by ``RULERBenchmark.score``), and writes a
new sheet laid out like the LongBench tabs:

    RULER16k  <model>  ratio=<r>
    <r> | CWE FWE NIAH-MK1 ... VT | Avg
      Ridge_g0.0_l0.1   ...
      Ridge_g0.0_l0.01  ...
      ... (44 rows)

Existing sheets (the LongBench tabs) are left untouched.

Usage
-----
    python scripts/ruler_ridge_ablation_to_xlsx.py
    python scripts/ruler_ridge_ablation_to_xlsx.py \
        --cells-dir results/ruler16k_sweep/<model>/manifest.cells
    python scripts/ruler_ridge_ablation_to_xlsx.py \
        --xlsx "Ridge Press.xlsx" --sheet RULER16k-Ridge-Ablation-Llama-3.1-8B
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import openpyxl
from openpyxl.styles import Font

REPO_ROOT = Path(__file__).resolve().parent.parent

# RULER subset name -> short display header.
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


def _cell_axes(cell: dict) -> tuple[float, float, float]:
    """Return (gamma, lambda, ratio) for a Ridge ablation cell."""
    kw = cell.get("kv_compressor_kwargs", {})
    return float(kw["envelope_gamma"]), float(kw["ridge_lambda"]), float(cell["ratio"])


def _cell_axes_2x2(cell: dict) -> tuple[float, bool, bool, float]:
    """Return (gamma, rotate_queries, normalize_keys_for_tau, ratio) for a 2x2 cell."""
    kw = cell.get("kv_compressor_kwargs", {})
    return (
        float(kw["envelope_gamma"]),
        bool(kw.get("rotate_queries", False)),
        bool(kw.get("normalize_keys_for_tau", False)),
        float(cell["ratio"]),
    )


def _detect_layout(cells_dir: Path) -> str:
    """Return 'lambda' if cells carry ridge_lambda, '2x2' for the rotate/normalize axes."""
    for frag in sorted(cells_dir.glob("cell_*.json")):
        cell = json.loads(frag.read_text(encoding="utf-8"))
        kw = cell.get("kv_compressor_kwargs", {})
        if "ridge_lambda" in kw:
            return "lambda"
        if "rotate_queries" in kw or "normalize_keys_for_tau" in kw:
            return "2x2"
    raise SystemExit(f"Could not detect layout from cells in {cells_dir}.")


def load_results(cells_dir: Path) -> tuple[dict, list[float], list[float], list[float]]:
    """Return ({(gamma, lambda, ratio): {subset: score}}, gammas, lambdas, ratios)."""
    out: dict = {}
    gammas, lambdas, ratios = set(), set(), set()
    for frag in sorted(cells_dir.glob("cell_*.json")):
        cell = json.loads(frag.read_text(encoding="utf-8"))
        if not cell.get("ok") or not cell.get("metrics"):
            continue
        try:
            data = json.loads(Path(cell["metrics"]).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        gamma, lam, ratio = _cell_axes(cell)
        gammas.add(gamma)
        lambdas.add(lam)
        ratios.add(ratio)
        task_scores = {sub: s["string_match"] for sub, s in data.get("task_scores", {}).items()
                       if "string_match" in s}
        out[(gamma, lam, ratio)] = task_scores
    return out, sorted(gammas), sorted(lambdas, reverse=True), sorted(ratios)


def load_results_2x2(cells_dir: Path) -> tuple[dict, list[float], list[float]]:
    """Return ({(gamma, rotate, normalize, ratio): {subset: score}}, gammas, ratios)."""
    out: dict = {}
    gammas, ratios = set(), set()
    for frag in sorted(cells_dir.glob("cell_*.json")):
        cell = json.loads(frag.read_text(encoding="utf-8"))
        if not cell.get("ok") or not cell.get("metrics"):
            continue
        try:
            data = json.loads(Path(cell["metrics"]).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        gamma, rot, norm, ratio = _cell_axes_2x2(cell)
        gammas.add(gamma)
        ratios.add(ratio)
        task_scores = {sub: s["string_match"] for sub, s in data.get("task_scores", {}).items()
                       if "string_match" in s}
        out[(gamma, rot, norm, ratio)] = task_scores
    return out, sorted(gammas), sorted(ratios)


def _avg(values: list) -> float | None:
    nums = [v for v in values if isinstance(v, (int, float))]
    return round(sum(nums) / len(nums), 2) if nums else None


def write_sheet(wb, sheet_name: str, model: str,
                gammas: list[float], lambdas: list[float], ratios: list[float],
                results: dict) -> None:
    if sheet_name in wb.sheetnames:
        del wb[sheet_name]
    ws = wb.create_sheet(sheet_name)
    bold = Font(bold=True)
    headers = [SUBSET_TO_HEADER[s] for s in SUBSET_ORDER]
    avg_col = len(SUBSET_ORDER) + 2

    def write_cell_row(r: int, gamma: float, lam: float, ratio: float) -> None:
        label = f"Ridge_g{gamma}_l{_fmt_lam(lam)}"
        scores = results.get((gamma, lam, ratio), {})
        ws.cell(r, 1, label)
        vals = []
        for c, sub in enumerate(SUBSET_ORDER, start=2):
            v = scores.get(sub)
            vals.append(v)
            if v is not None:
                cell = ws.cell(r, c, round(float(v), 2))
                # Force 2-decimal display so 0/1 render as 0.00/1.00 instead of
                # bare integers (openpyxl collapses whole-number floats to int,
                # which some grid renderers display as blank).
                cell.number_format = "0.00"
        a = _avg(vals)
        if a is not None:
            cell = ws.cell(r, avg_col, a)
            cell.number_format = "0.00"

    row = 1
    ws.cell(row, 1, "RULER16k").font = bold
    ws.cell(row, 2, model).font = bold
    row += 2

    for ratio in ratios:
        ws.cell(row, 1, ratio).font = bold
        for c, h in enumerate(headers, start=2):
            ws.cell(row, c, h).font = bold
        ws.cell(row, avg_col, "Avg").font = bold
        row += 1
        for gamma in gammas:
            for lam in lambdas:
                write_cell_row(row, gamma, lam, ratio)
                row += 1
        row += 1  # blank separator between ratio blocks


CORNERS_2x2: list[tuple[bool, bool]] = [
    (False, False),
    (False, True),
    (True, False),
    (True, True),
]


def _corner_label(rotate: bool, normalize: bool) -> str:
    return f"rq{'T' if rotate else 'F'}_nk{'T' if normalize else 'F'}"


def _fmt_gamma(gamma: float) -> str:
    """Match cell_id formatting: integers as '0', '1'; halves as '0.5', '1.5'."""
    return f"{gamma:g}"


def write_sheet_2x2(wb, sheet_name: str, model: str,
                    gammas: list[float], ratio: float,
                    results: dict) -> None:
    """Write one sheet for a single ratio with 4 corner blocks stacked."""
    if sheet_name in wb.sheetnames:
        del wb[sheet_name]
    ws = wb.create_sheet(sheet_name)
    bold = Font(bold=True)
    headers = [SUBSET_TO_HEADER[s] for s in SUBSET_ORDER]
    avg_col = len(SUBSET_ORDER) + 2

    row = 1
    ws.cell(row, 1, "RULER16k").font = bold
    ws.cell(row, 2, model).font = bold
    ws.cell(row, 3, f"ratio={ratio}").font = bold
    row += 2

    for rotate, normalize in CORNERS_2x2:
        corner = _corner_label(rotate, normalize)
        ws.cell(row, 1, corner).font = bold
        for c, h in enumerate(headers, start=2):
            ws.cell(row, c, h).font = bold
        ws.cell(row, avg_col, "Avg").font = bold
        row += 1
        for gamma in gammas:
            label = f"Ridge_g{_fmt_gamma(gamma)}_{corner}"
            scores = results.get((gamma, rotate, normalize, ratio), {})
            ws.cell(row, 1, label)
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
            row += 1
        row += 1  # blank separator between corner blocks


def _fmt_lam(lam: float) -> str:
    """Match the cell_id lambda formatting (0.1, 0.01, 0.0001, 1e-05)."""
    if lam >= 1e-4:
        return f"{lam:g}"
    return f"{lam:.0e}".replace("e-0", "e-").replace("e-", "e-0")  # 1e-05 not 1e-5


def _pick_cells_dir() -> Path:
    roots = [REPO_ROOT / "results" / "ruler16k_sweep_2x2",
             REPO_ROOT / "results" / "ruler16k_sweep"]
    hits: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        hits.extend(p for p in root.glob("*/manifest.cells")
                    if any(p.glob("cell_*.json")))
    if not hits:
        raise SystemExit("No populated manifest.cells/ found under results/ruler16k_sweep{,_2x2} — run the sweep first.")
    return max(hits, key=lambda p: p.stat().st_mtime)


def _infer_model(cells_dir: Path) -> str:
    """The model slug is the parent dir; turn 'meta-llama--Llama-3.1-8B-Instruct' back into 'meta-llama/Llama-3.1-8B-Instruct'."""
    slug = cells_dir.parent.name
    return slug.replace("--", "/", 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells-dir", default=None,
                    help="Path to <model>/manifest.cells/ dir (default: newest under results/ruler16k_sweep)")
    ap.add_argument("--xlsx", default=str(REPO_ROOT / "Ridge Press.xlsx"))
    ap.add_argument("--sheet", default=None,
                    help="Sheet name (default: RULER16k-Ridge-Ablation-<model-slug>)")
    args = ap.parse_args()

    cells_dir = Path(args.cells_dir) if args.cells_dir else _pick_cells_dir()
    model = _infer_model(cells_dir)
    layout = _detect_layout(cells_dir)

    # Keep <=31 chars to satisfy Excel's sheet-name limit. Strip the
    # "-Instruct"/"-Instruct-<date>" suffix from the model slug to make room.
    short = model.split("/")[-1]
    for suffix in ("-Instruct-2512", "-Instruct"):
        if short.endswith(suffix):
            short = short[: -len(suffix)]
            break

    xlsx = Path(args.xlsx)
    if xlsx.exists():
        wb = openpyxl.load_workbook(xlsx)
        action = "updated"
    else:
        wb = openpyxl.Workbook()
        default = wb.active
        if default is not None and default.max_row == 1 and default.max_column == 1 \
                and default.cell(1, 1).value is None:
            wb.remove(default)
        action = "created"

    if layout == "lambda":
        results, gammas, lambdas, ratios = load_results(cells_dir)
        if not results:
            raise SystemExit(f"No completed cells with metrics under {cells_dir}.")
        sheet = args.sheet or f"RULER16k-Ridge-{short}"
        write_sheet(wb, sheet, model, gammas, lambdas, ratios, results)
        xlsx.parent.mkdir(parents=True, exist_ok=True)
        wb.save(xlsx)
        total = len(gammas) * len(lambdas) * len(ratios)
        print(f"Wrote sheet '{sheet}' to {xlsx} ({action}; {len(results)}/{total} cells with data).")
        missing = [f"g{g}_l{_fmt_lam(l)}__r{r}" for r in ratios for g in gammas for l in lambdas
                   if (g, l, r) not in results]
        if missing:
            print(f"Missing cells (left blank): {', '.join(missing)}")
        return

    # 2x2 layout: one sheet per ratio, four corner blocks per sheet.
    results, gammas, ratios = load_results_2x2(cells_dir)
    if not results:
        raise SystemExit(f"No completed cells with metrics under {cells_dir}.")
    written: list[str] = []
    for ratio in ratios:
        ratio_tag = f"r{ratio:g}".replace(".", "p")
        # Excel caps sheet names at 31 chars; "Ridge2x2-Ministral-3-3B-r0p6" = 28.
        sheet = args.sheet or f"Ridge2x2-{short}-{ratio_tag}"
        if args.sheet and len(ratios) > 1:
            sheet = f"{args.sheet}-{ratio_tag}"
        write_sheet_2x2(wb, sheet, model, gammas, ratio, results)
        written.append(sheet)
    xlsx.parent.mkdir(parents=True, exist_ok=True)
    wb.save(xlsx)
    total = len(gammas) * len(CORNERS_2x2) * len(ratios)
    print(f"Wrote {len(written)} sheet(s) {written} to {xlsx} "
          f"({action}; {len(results)}/{total} cells with data).")
    missing = [f"g{_fmt_gamma(g)}_{_corner_label(r, n)}__r{ratio}"
               for ratio in ratios for r, n in CORNERS_2x2 for g in gammas
               if (g, r, n, ratio) not in results]
    if missing:
        print(f"Missing cells (left blank): {', '.join(missing)}")


if __name__ == "__main__":
    main()
