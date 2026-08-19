#!/usr/bin/env python
"""Fill the ``Verified`` (random-sampling + Ridge) sheets in ``Ridge Press.xlsx``.

The verified sweep runs ``VerifiedSketch`` — a deterministic ridge "head" of size
``det_fraction`` of the per-head budget plus a uniform-random tail filling the rest
(seed 42; inner=ridge, rqF/nkT/sink4/local28) — over a
``det_fraction × envelope_gamma`` grid at three ratios (0.6/0.8/0.9), on both
RULER-16K and LongBench for Llama-3.1-8B.

Because ``det_fraction`` spans the full range, each sheet self-contains its
anchors: ``d1`` (det_fraction=1.0) is pure Ridge (no random tail) and ``d0``
(det_fraction=0.0) is a pure uniform-random tail (gamma is irrelevant, so it is a
single row).

The grid is read from the per-model ``manifest.cells/`` fragments (which carry the
exact ``det_fraction`` and ``envelope_gamma`` floats), and each cell's single
``metrics.json`` is loaded from its result dir (tolerating the numbered
resume-dedup subdirs). Missing cells are left blank.

Sheet layout (one sheet per benchmark): one block per ratio — a bold ratio header
row with the subset headers + Avg, then the config rows grouped by det_fraction
then gamma, labelled ``V_d<det>_g<gamma>`` (``V_d0`` for the pure-random row).

Usage
-----
    python scripts/verified_to_xlsx.py                 # both benchmarks + index
    python scripts/verified_to_xlsx.py --only ruler
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import openpyxl
from openpyxl.styles import Font

REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # scripts/reporting/ -> repo root

# --- per-benchmark subset order + display headers --------------------------------
RULER_SUBSETS = {
    "cwe": "CWE", "fwe": "FWE",
    "niah_multikey_1": "NIAH-MK1", "niah_multikey_2": "NIAH-MK2",
    "niah_multikey_3": "NIAH-MK3", "niah_multiquery": "NIAH-MQ",
    "niah_multivalue": "NIAH-MV", "niah_single_1": "NIAH-S1",
    "niah_single_2": "NIAH-S2", "niah_single_3": "NIAH-S3",
    "qa_1": "QA1", "qa_2": "QA2", "vt": "VT",
}
LONGBENCH_SUBSETS = {
    "narrativeqa": "NrtvQA", "qasper": "Qasper", "multifieldqa_en": "MF-en",
    "hotpotqa": "HotpotQA", "2wikimqa": "2WikiMQA", "musique": "Musique",
    "gov_report": "GovReport", "qmsum": "QMSum", "multi_news": "MultiNews",
    "trec": "TREC", "triviaqa": "TriviaQA", "samsum": "SAMSum",
    "passage_count": "PCount", "passage_retrieval_en": "PRe",
    "lcc": "LCC", "repobench-p": "RB-P",
}

BENCHMARKS = {
    "ruler": dict(
        results="ruler16k_verified", sheet="60_verified_ruler_llama",
        title="RULER16k", subsets=RULER_SUBSETS,
    ),
    "longbench": dict(
        results="longbench_verified", sheet="61_verified_longbench_llama",
        title="LongBench", subsets=LONGBENCH_SUBSETS,
    ),
}

INDEX_SHEET = "00_INDEX"
INDEX_ROWS = {  # sheet_name -> full index row (matches 00_INDEX columns)
    "60_verified_ruler_llama": [
        "60_verified_ruler_llama", "Verified", "Llama-3.1-8B", "RULER-16K",
        "0.6/0.8/0.9", "det_fraction × γ",
        "Random-sampling + Ridge: deterministic ridge head (det_fraction) + "
        "uniform-random tail, seed42, rqF/nkT/sink4/local28; d1=pure Ridge, d0=pure random",
    ],
    "61_verified_longbench_llama": [
        "61_verified_longbench_llama", "Verified", "Llama-3.1-8B", "LongBench",
        "0.6/0.8/0.9", "det_fraction × γ",
        "Same VerifiedSketch grid applied to LongBench; d1=pure Ridge, d0=pure random tail",
    ],
}


def _score(entry) -> float | None:
    """A task_scores entry is either a float (LongBench) or {'string_match': f} (RULER)."""
    if isinstance(entry, dict):
        entry = entry.get("string_match")
    return float(entry) if isinstance(entry, (int, float)) else None


def _load_metrics(cell_id_dir: Path) -> dict:
    """{subset: score} for a cell, taking the newest metrics.json under it (empty if none)."""
    hits = sorted(glob.glob(str(cell_id_dir / "**" / "metrics.json"), recursive=True),
                  key=lambda p: Path(p).stat().st_mtime)
    if not hits:
        return {}
    data = json.loads(Path(hits[-1]).read_text(encoding="utf-8"))
    out = {}
    for sub, entry in data.get("task_scores", {}).items():
        s = _score(entry)
        if s is not None:
            out[sub] = s
    return out


def load_grid(model_dir: Path) -> tuple[dict, list, list, list]:
    """Return ({(ratio, det, gamma): scores}, ratios, dets, gammas) from manifest.cells."""
    results: dict = {}
    ratios, dets, gammas = set(), set(), set()
    for frag in sorted((model_dir / "manifest.cells").glob("cell_*.json")):
        cell = json.loads(frag.read_text(encoding="utf-8"))
        kw = cell.get("kv_compressor_kwargs", {})
        ratio = float(cell["ratio"])
        det = float(kw.get("det_fraction"))
        gamma = float(kw.get("inner_kwargs", {}).get("envelope_gamma", 0.0))
        ratios.add(ratio)
        dets.add(det)
        if det > 0.0:  # gamma only meaningful when there is a deterministic head
            gammas.add(gamma)
        scores = _load_metrics(model_dir / cell["cell_id"])
        if scores:
            results[(ratio, det, gamma)] = scores
    return results, sorted(ratios), sorted(dets), sorted(gammas)


def _avg(values: list) -> float | None:
    nums = [v for v in values if isinstance(v, (int, float))]
    return round(sum(nums) / len(nums), 2) if nums else None


def _write_row(ws, row: int, label: str, scores: dict, subsets: list, avg_col: int) -> bool:
    ws.cell(row, 1, label)
    vals = []
    for c, sub in enumerate(subsets, start=2):
        v = scores.get(sub)
        vals.append(v)
        if v is not None:
            cell = ws.cell(row, c, round(float(v), 2))
            cell.number_format = "0.00"
    a = _avg(vals)
    if a is not None:
        cell = ws.cell(row, avg_col, a)
        cell.number_format = "0.00"
    return bool(scores)


def write_sheet(wb, spec: dict, model: str, results: dict,
                ratios: list, dets: list, gammas: list) -> tuple[int, list]:
    name = spec["sheet"]
    if name in wb.sheetnames:
        del wb[name]
    ws = wb.create_sheet(name)
    bold = Font(bold=True)
    subsets = list(spec["subsets"].keys())
    headers = list(spec["subsets"].values())
    avg_col = len(subsets) + 2

    ws.cell(1, 1, spec["title"]).font = bold
    ws.cell(1, 2, model).font = bold
    ws.cell(1, 3, "VerifiedSketch: ridge head (det_fraction) + uniform-random tail, "
                  "seed42, rqF/nkT/sink4/local28").font = bold

    row, written, missing = 3, 0, []
    for ratio in ratios:
        ws.cell(row, 1, f"{ratio:g}").font = bold
        for c, h in enumerate(headers, start=2):
            ws.cell(row, c, h).font = bold
        ws.cell(row, avg_col, "Avg").font = bold
        row += 1
        for det in dets:
            det_gammas = [0.0] if det == 0.0 else gammas
            for gamma in det_gammas:
                label = f"V_d{det:g}" if det == 0.0 else f"V_d{det:g}_g{gamma:g}"
                scores = results.get((ratio, det, gamma), {})
                if _write_row(ws, row, label, scores, subsets, avg_col):
                    written += 1
                else:
                    missing.append(f"{label}__r{ratio:g}")
                row += 1
        row += 1  # blank separator between ratio blocks
    return written, missing


def update_index(wb, sheet_name: str) -> None:
    """Add/refresh the 00_INDEX row for ``sheet_name`` (keeping the # column sequential)."""
    ws = wb[INDEX_SHEET]
    want = INDEX_ROWS[sheet_name]
    # Find an existing row for this sheet (col 2 == sheet name), else append.
    target = None
    max_num = 0
    for r in range(2, ws.max_row + 1):
        num = ws.cell(r, 1).value
        if isinstance(num, (int, float)):
            max_num = max(max_num, int(num))
        if ws.cell(r, 2).value == sheet_name:
            target = r
    if target is None:
        target = ws.max_row + 1
        ws.cell(target, 1, max_num + 1)
    for c, val in enumerate(want, start=2):  # cols 2..8 (Sheet..Notes)
        ws.cell(target, c, val)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=list(BENCHMARKS), default=None,
                    help="Only write one benchmark's sheet (default: both).")
    ap.add_argument("--model-slug", default="meta-llama--Llama-3.1-8B-Instruct")
    ap.add_argument("--xlsx", default=str(REPO_ROOT / "Ridge Press.xlsx"))
    ap.add_argument("--no-index", action="store_true", help="Skip 00_INDEX update.")
    args = ap.parse_args()

    xlsx = Path(args.xlsx)
    wb = openpyxl.load_workbook(xlsx)
    model = args.model_slug.replace("--", "/", 1)
    names = [args.only] if args.only else list(BENCHMARKS)

    for key in names:
        spec = BENCHMARKS[key]
        model_dir = REPO_ROOT / "results" / spec["results"] / args.model_slug
        if not (model_dir / "manifest.cells").is_dir():
            raise SystemExit(f"No manifest.cells under {model_dir}")
        results, ratios, dets, gammas = load_grid(model_dir)
        if not results:
            raise SystemExit(f"No completed cells with metrics under {model_dir}")
        written, missing = write_sheet(wb, spec, model, results, ratios, dets, gammas)
        if not args.no_index:
            update_index(wb, spec["sheet"])
        total = sum(len([0.0] if d == 0.0 else gammas) for d in dets) * len(ratios)
        print(f"Wrote '{spec['sheet']}' ({written}/{total} cells; "
              f"ratios={ratios}, dets={dets}, gammas={gammas}).")
        if missing:
            print(f"  Missing (left blank): {', '.join(missing)}")

    wb.save(xlsx)
    print(f"Saved {xlsx}")


if __name__ == "__main__":
    main()
