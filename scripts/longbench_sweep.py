#!/usr/bin/env python
"""Orchestrate the LongBench KV-compression sweep for one model.

Runs every (method, compression-ratio) cell of the LongBench scoreboard through
the existing research backend and records where each ``metrics.json`` landed, so
``scripts/longbench_to_xlsx.py`` can fill ``Ridge Press.xlsx``.

Method rows = the superset of the PyramidKV paper (arXiv:2406.02069) and the
team's RULER sheet:
    Full (no compression baseline, once) +
    Knorm, CurDkv, PyramidKv, SnapKv, Expected_attn, Key_Diff,
    StreamingLLM, Compactor     (each at ratios 0.6 / 0.9 / 0.95)
    Ridge                       (gamma sweep: envelope_gamma ∈
                                 {0, 0.5, 1, 1.5, 2, 2.5, 3} × ratios
                                 0.6 / 0.9 / 0.95 = 21 cells)

H2O is intentionally excluded: the in-tree H2OSketch only sums attention from
the LAST prefill chunk's queries (no running accumulator), so it is not a
faithful port of the paper's heavy-hitter score, and on top of that it requires
eager attention which OOMs at LongBench-length contexts. Bring it back only
after the scoring is fixed.

Columns = the paper's 16 English LongBench tasks (Table 1).

The CLI only overrides top-level config fields, not the nested ``research_config``
(``kv_compressor`` etc.), so each cell is run from a generated temp YAML. Each run
gets its own ``output_dir`` (the auto-named run dir does NOT encode the
compressor, so distinct dirs are required to keep cells apart). Runs are launched
as subprocesses so CUDA memory is fully reclaimed between the ~37 model loads.

Usage
-----
    python scripts/longbench_sweep.py                    # full sweep, Llama-3.1-8B
    python scripts/longbench_sweep.py --max-requests 5   # quick smoke pass
    python scripts/longbench_sweep.py --methods knorm,h2o --ratios 0.2
    python scripts/longbench_sweep.py --dry-run          # print the plan only
    python scripts/longbench_sweep.py --cell-index 7     # run only cell 7 (for SLURM arrays)

Writes a manifest at ``<out-root>/manifest.json`` consumed by the fill script.
With ``--cell-index N`` the run instead writes ``<out-root>/manifest.cells/cell_NN.json``
to avoid races between concurrent array tasks; merge them later by re-running
without ``--cell-index`` in ``--resume`` mode (it rebuilds the unified manifest
from existing per-cell metrics).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

# The paper's 16 English LongBench tasks, in Table 1 column order. The display
# header is what the xlsx column will be labelled (see longbench_to_xlsx.py).
LONGBENCH_16 = [
    "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique",
    "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum",
    "passage_count", "passage_retrieval_en", "lcc", "repobench-p",
]

# Sheet row label -> kv_compression registry key. "Full" is the no-compression
# baseline (kv_compressor=none); it is run once, not per ratio.
# Ridge is sweep separately as a (gamma, ratio) grid — see RIDGE_GAMMA_GRID
# below — so it is intentionally absent from this dict.
METHODS: dict[str, str] = {
    "Knorm": "knorm",
    "CurDkv": "cur",
    "PyramidKv": "pyramidkv",
    "SnapKv": "snapkv",
    "Expected_attn": "expected_attention",
    "Key_Diff": "keydiff",
    "StreamingLLM": "streaming_llm",
    "Compactor": "compactor",
}

DEFAULT_RATIOS = [0.6, 0.9, 0.95]

# Ridge gamma sweep. envelope_gamma is the dial on the query-side signal in
# Ridge's default fixed_envelope combine_mode:
#   score_i = max(p_ridge_i, envelope_gamma * p_query_i)
# 0 = pure ridge (query muted); 1 = balanced (the press's default);
# >1 tilts toward the query side. Each gamma runs at every ratio.
RIDGE_GAMMAS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

# PyramidKV's per-layer pyramid budgets leave the cache cross-layer ragged.
# Under transformers 5.x sdpa/eager the single decode mask is sized from layer 0
# and would broadcast-error on shorter layers, so pyramidkv_sketch.py raises
# unless attn_implementation == "flash_attention_2" (or uniform_budget=True,
# which degenerates the method to SnapKV). Use flash-attn for that one cell.
FLASH_ATTN_METHODS = {"pyramidkv"}

# Compressors that draw random Gaussian sketches at score-time. kvpress leaves
# these unseeded, so two reruns of the same cell can differ by ~0.5 points.
# Threading a seed through kv_compressor_kwargs makes the random projection
# reproducible across reruns (still differs per layer because the seeded
# generator advances naturally between calls). CUR only draws when
# ``use_random_leverage=True`` (off by default); Compactor draws on every
# call.
SEEDED_METHODS = {"cur", "compactor"}
SWEEP_SEED = 42


def _extended(path: Path) -> str:
    r"""Return an extended-length (``\\?\``) absolute path string on Windows so
    the long 16-subset run-dir name can be created past the 260-char MAX_PATH."""
    ap = path.resolve()
    if sys.platform == "win32":
        return "\\\\?\\" + str(ap).replace("/", "\\")
    return str(ap)


def _slug(model: str) -> str:
    return model.replace("/", "--")


def build_config(base: dict, *, model: str, kv_compressor: str, ratio: float,
                 subsets: list[str], out_root: Path, cell_id: str,
                 max_requests: int | None, max_model_len: int | None,
                 extra_kv_kwargs: dict | None = None) -> dict:
    """Construct the full EvalConfig dict for one sweep cell."""
    cfg = copy.deepcopy(base)
    cfg["benchmark"] = "longbench"
    cfg["subsets"] = ",".join(subsets)
    cfg["backend"] = "research"
    cfg["model"] = model
    # Sweep cells need run-to-run reproducibility AND a fixed SDPA backend so
    # numbers are comparable to the kvpress leaderboard (which assumes flash /
    # math, not mem-efficient). Off by default at the EvalConfig level — opt
    # in here for every LongBench sweep cell.
    cfg["deterministic"] = True
    # Let the per-task value from the LongBench dataset win (128 for QA tasks,
    # 512 for summarization: gov_report/qmsum/multi_news, 64 for code-completion).
    # A global cap here would silently chop summaries off at 128 tokens.
    cfg["max_new_tokens"] = None
    cfg["temperature"] = 0.0
    cfg["top_p"] = 1.0
    if max_model_len is not None:
        cfg["max_model_len"] = max_model_len
    if max_requests is not None:
        cfg["max_requests"] = max_requests
    # Unique per-cell output dir (extended-length so the long auto-named leaf fits).
    cfg["output_dir"] = _extended(out_root / cell_id)

    llm = dict(cfg.get("llm_kwargs") or {})
    llm["attn_implementation"] = "flash_attention_2" if kv_compressor in FLASH_ATTN_METHODS else "sdpa"
    rc = dict(llm.get("research_config") or {})
    rc["kv_compressor"] = kv_compressor
    rc["compression_ratio"] = 0.0 if kv_compressor == "none" else float(ratio)
    rc["attention_method"] = "none"
    rc["attention_method_kwargs"] = {}
    # Leave use_chat_template at its config default (True) — LongBench's load()
    # tags each row with a `use_chat_template` column derived from the official
    # pred.py skip-list (trec/triviaqa/samsum/lsht/lcc/repobench-p go raw, the
    # rest get the chat wrapper), and the runner threads the per-row value to
    # the adapter as a per-call override. A blanket False here would break the
    # QA-style tasks (hotpotqa/2wikimqa/qasper/multifieldqa_en/musique) that
    # Llama-3.1-Instruct cannot answer without its native chat format.
    # Seed the random sketches in cur/compactor so reruns are reproducible,
    # and merge in any per-cell extras (e.g. Ridge's envelope_gamma).
    kv_kwargs = dict(rc.get("kv_compressor_kwargs") or {})
    if kv_compressor in SEEDED_METHODS:
        kv_kwargs.setdefault("seed", SWEEP_SEED)
    if extra_kv_kwargs:
        kv_kwargs.update(extra_kv_kwargs)
    if kv_kwargs:
        rc["kv_compressor_kwargs"] = kv_kwargs
    rc.pop("prefill_chunk_size", None)
    llm["research_config"] = rc
    cfg["llm_kwargs"] = llm
    return cfg


def find_metrics(out_dir: Path) -> Path | None:
    """Locate the metrics.json the harness wrote under this cell's output dir."""
    hits = sorted(out_dir.rglob("metrics.json"), key=lambda p: p.stat().st_mtime)
    return hits[-1] if hits else None


def _total_samples(metrics_path: Path | None) -> int | None:
    if not metrics_path or not Path(metrics_path).exists():
        return None
    try:
        return int(json.loads(Path(metrics_path).read_text(encoding="utf-8")).get("total_samples", 0)) or None
    except (OSError, ValueError):  # ValueError covers json.JSONDecodeError
        return None


def run_cell(cfg: dict, tmp_yaml: Path) -> int:
    tmp_yaml.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "eval_harness.cli", "run", "--config_file", str(tmp_yaml)],
        cwd=str(REPO_ROOT),
    )
    return proc.returncode


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--template", default=str(REPO_ROOT / "evaluate" / "evaluate_kv.yaml"),
                    help="Base config YAML to clone per run")
    ap.add_argument("--out-root", default=None,
                    help="Where run dirs + manifest land (default: results/longbench_sweep/<model>)")
    ap.add_argument("--methods", default=None,
                    help="Comma list of registry keys to include (default: all 9)")
    ap.add_argument("--ratios", default=None,
                    help="Comma list of compression ratios (default: 0.2,0.4,0.6,0.8)")
    ap.add_argument("--tasks", default=None, help="Comma list of subsets (default: the 16)")
    ap.add_argument("--max-requests", type=int, default=None, help="Cap rows per subset")
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--skip-full", action="store_true", help="Do not run the Full baseline")
    ap.add_argument("--resume", action="store_true", help="Skip cells whose metrics.json already exists")
    ap.add_argument("--dry-run", action="store_true", help="Print the plan and exit")
    ap.add_argument("--cell-index", type=int, default=None,
                    help="Run only the Nth cell (0-indexed) and exit. Intended for SLURM job arrays")
    args = ap.parse_args()

    base = yaml.safe_load(Path(args.template).read_text(encoding="utf-8")) or {}
    subsets = [t.strip() for t in args.tasks.split(",")] if args.tasks else LONGBENCH_16
    ratios = [float(r) for r in args.ratios.split(",")] if args.ratios else DEFAULT_RATIOS

    if args.methods:
        wanted = {m.strip() for m in args.methods.split(",")}
        methods = {lbl: key for lbl, key in METHODS.items() if key in wanted}
    else:
        methods = dict(METHODS)

    out_root = Path(args.out_root) if args.out_root else (
        REPO_ROOT / "results" / "longbench_sweep" / _slug(args.model))
    out_root.mkdir(parents=True, exist_ok=True)

    # Build the list of cells: (label, kv_key, ratio_or_None, cell_id, extra_kwargs).
    cells: list[tuple[str, str, float | None, str, dict]] = []
    if not args.skip_full:
        cells.append(("Full", "none", None, "Full", {}))
    for label, key in methods.items():
        for ratio in ratios:
            cells.append((label, key, ratio, f"{label}__r{ratio}", {}))
    # Ridge gamma sweep: envelope_gamma ∈ RIDGE_GAMMAS × each ratio.
    # combine_mode is fixed_envelope by default so envelope_gamma is live;
    # pin it here to be explicit/robust against upstream default changes.
    for gamma in RIDGE_GAMMAS:
        for ratio in ratios:
            cells.append((
                "Ridge", "ridge", ratio,
                f"Ridge_g{gamma}__r{ratio}",
                {"envelope_gamma": float(gamma), "combine_mode": "fixed_envelope"},
            ))

    print(f"Model:   {args.model}")
    print(f"Tasks:   {len(subsets)} subsets")
    print(f"Cells:   {len(cells)} runs (out-root: {out_root})")
    for idx, (label, key, ratio, cell_id, extras) in enumerate(cells):
        flag = " [flash_attn_2]" if key in FLASH_ATTN_METHODS else ""
        extra = f" extras={extras}" if extras else ""
        print(f"  [{idx:2d}] {cell_id:28s} kv_compressor={key} ratio={ratio}{flag}{extra}")
    if args.dry_run:
        return

    if args.cell_index is not None:
        if not 0 <= args.cell_index < len(cells):
            sys.exit(f"--cell-index {args.cell_index} out of range [0, {len(cells)})")
        selected = [(args.cell_index, cells[args.cell_index])]
        # Per-cell manifest fragment under manifest.cells/ — avoids races between
        # concurrent SLURM array tasks all trying to write the same manifest.json.
        (out_root / "manifest.cells").mkdir(parents=True, exist_ok=True)
        per_cell_manifest = out_root / "manifest.cells" / f"cell_{args.cell_index:02d}.json"
    else:
        selected = list(enumerate(cells))
        per_cell_manifest = None

    # Per-process tmp YAML so concurrent array tasks don't trample each other.
    tmp_yaml_name = f"_cell_config_{os.getpid()}.yaml"
    tmp_yaml = out_root / tmp_yaml_name
    manifest = {"model": args.model, "subsets": subsets, "ratios": ratios, "cells": []}
    n_selected = len(selected)
    for run_pos, (idx, (label, key, ratio, cell_id, extras)) in enumerate(selected, 1):
        prefix = f"cell {idx}" if args.cell_index is not None else f"{run_pos}/{n_selected}"
        cell_dir = out_root / cell_id
        existing = find_metrics(cell_dir) if cell_dir.exists() else None
        elapsed = None
        if args.resume and existing is not None:
            print(f"[{prefix}] {cell_id}: resume — found {existing}")
            metrics_path, rc = existing, 0
        else:
            print(f"\n[{prefix}] {cell_id}: running…", flush=True)
            cfg = build_config(base, model=args.model, kv_compressor=key, ratio=ratio or 0.0,
                               subsets=subsets, out_root=out_root, cell_id=cell_id,
                               max_requests=args.max_requests, max_model_len=args.max_model_len,
                               extra_kv_kwargs=extras or None)
            t0 = time.time()
            rc = run_cell(cfg, tmp_yaml)
            elapsed = time.time() - t0
            metrics_path = find_metrics(cell_dir)
            samples = _total_samples(metrics_path)
            per = f"{elapsed / samples:.2f}s/sample" if samples else "n/a"
            print(f"    -> {elapsed:.1f}s wall, {samples or '?'} samples ({per})", flush=True)
        cell_record = {
            "index": idx,
            "label": label, "kv_compressor": key, "ratio": ratio, "cell_id": cell_id,
            "kv_compressor_kwargs": extras or None,
            "returncode": rc, "elapsed_sec": round(elapsed, 1) if elapsed else None,
            "total_samples": _total_samples(metrics_path),
            "metrics": str(metrics_path) if metrics_path else None,
            "ok": rc == 0 and metrics_path is not None,
        }
        manifest["cells"].append(cell_record)
        if per_cell_manifest is not None:
            # SLURM-array mode: write only the per-cell fragment; the merged
            # manifest.json is rebuilt by a follow-up resume run.
            per_cell_manifest.write_text(json.dumps(cell_record, indent=2), encoding="utf-8")
        else:
            (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Best-effort cleanup of this process's tmp YAML.
    try:
        tmp_yaml.unlink()
    except FileNotFoundError:
        pass

    ok = sum(1 for c in manifest["cells"] if c["ok"])
    print(f"\nDone: {ok}/{n_selected} cells succeeded.")
    if args.cell_index is None:
        print(f"Manifest: {out_root / 'manifest.json'}")
    if ok < n_selected:
        sys.exit(1)


if __name__ == "__main__":
    main()
