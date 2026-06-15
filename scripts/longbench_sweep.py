#!/usr/bin/env python
"""Orchestrate the LongBench KV-compression sweep for one model.

Runs every (method, compression-ratio) cell of the LongBench scoreboard through
the existing research backend and records where each ``metrics.json`` landed, so
``scripts/longbench_to_xlsx.py`` can fill ``Ridge Press.xlsx``.

Method rows = the superset of the PyramidKV paper (arXiv:2406.02069) and the
team's RULER sheet:
    Full (no compression baseline, once) +
    Knorm, CurDkv, PyramidKv, SnapKv, Expected_attn, Key_Diff, Ridge,
    StreamingLLM, H2O           (each at ratios 0.2 / 0.4 / 0.6 / 0.8)

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

Writes a manifest at ``<out-root>/manifest.json`` consumed by the fill script.
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
METHODS: dict[str, str] = {
    "Knorm": "knorm",
    "CurDkv": "cur",
    "PyramidKv": "pyramidkv",
    "SnapKv": "snapkv",
    "Expected_attn": "expected_attention",
    "Key_Diff": "keydiff",
    "Ridge": "ridge",
    "StreamingLLM": "streaming_llm",
    "H2O": "h2o",
}

DEFAULT_RATIOS = [0.2, 0.4, 0.6, 0.8]

# Only H2O / observed_attention need eager attention (to receive attention probs
# in the hook). Everything else uses sdpa, the validated parity path.
EAGER_METHODS = {"h2o"}


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
                 max_requests: int | None, max_model_len: int | None) -> dict:
    """Construct the full EvalConfig dict for one sweep cell."""
    cfg = copy.deepcopy(base)
    cfg["benchmark"] = "longbench"
    cfg["subsets"] = ",".join(subsets)
    cfg["backend"] = "research"
    cfg["model"] = model
    cfg["max_new_tokens"] = 128
    cfg["temperature"] = 0.0
    cfg["top_p"] = 1.0
    if max_model_len is not None:
        cfg["max_model_len"] = max_model_len
    if max_requests is not None:
        cfg["max_requests"] = max_requests
    # Unique per-cell output dir (extended-length so the long auto-named leaf fits).
    cfg["output_dir"] = _extended(out_root / cell_id)

    llm = dict(cfg.get("llm_kwargs") or {})
    llm["attn_implementation"] = "eager" if kv_compressor in EAGER_METHODS else "sdpa"
    rc = dict(llm.get("research_config") or {})
    rc["kv_compressor"] = kv_compressor
    rc["compression_ratio"] = 0.0 if kv_compressor == "none" else float(ratio)
    rc["attention_method"] = "none"
    rc["attention_method_kwargs"] = {}
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

    # Build the list of cells: (label, kv_key, ratio_or_None, cell_id).
    cells: list[tuple[str, str, float | None, str]] = []
    if not args.skip_full:
        cells.append(("Full", "none", None, "Full"))
    for label, key in methods.items():
        for ratio in ratios:
            cells.append((label, key, ratio, f"{label}__r{ratio}"))

    print(f"Model:   {args.model}")
    print(f"Tasks:   {len(subsets)} subsets")
    print(f"Cells:   {len(cells)} runs (out-root: {out_root})")
    for label, key, ratio, cell_id in cells:
        eager = " [eager]" if key in EAGER_METHODS else ""
        print(f"  - {cell_id:24s} kv_compressor={key} ratio={ratio}{eager}")
    if args.dry_run:
        return

    tmp_yaml = out_root / "_cell_config.yaml"
    manifest = {"model": args.model, "subsets": subsets, "ratios": ratios, "cells": []}
    for i, (label, key, ratio, cell_id) in enumerate(cells, 1):
        cell_dir = out_root / cell_id
        existing = find_metrics(cell_dir) if cell_dir.exists() else None
        elapsed = None
        if args.resume and existing is not None:
            print(f"[{i}/{len(cells)}] {cell_id}: resume — found {existing}")
            metrics_path, rc = existing, 0
        else:
            print(f"\n[{i}/{len(cells)}] {cell_id}: running…", flush=True)
            cfg = build_config(base, model=args.model, kv_compressor=key, ratio=ratio or 0.0,
                               subsets=subsets, out_root=out_root, cell_id=cell_id,
                               max_requests=args.max_requests, max_model_len=args.max_model_len)
            t0 = time.time()
            rc = run_cell(cfg, tmp_yaml)
            elapsed = time.time() - t0
            metrics_path = find_metrics(cell_dir)
            samples = _total_samples(metrics_path)
            per = f"{elapsed / samples:.2f}s/sample" if samples else "n/a"
            print(f"    -> {elapsed:.1f}s wall, {samples or '?'} samples ({per})", flush=True)
        manifest["cells"].append({
            "label": label, "kv_compressor": key, "ratio": ratio, "cell_id": cell_id,
            "returncode": rc, "elapsed_sec": round(elapsed, 1) if elapsed else None,
            "total_samples": _total_samples(metrics_path),
            "metrics": str(metrics_path) if metrics_path else None,
            "ok": rc == 0 and metrics_path is not None,
        })
        # Persist manifest after every cell so a crash mid-sweep is recoverable.
        (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    ok = sum(1 for c in manifest["cells"] if c["ok"])
    print(f"\nDone: {ok}/{len(cells)} cells succeeded. Manifest: {out_root / 'manifest.json'}")
    if ok < len(cells):
        sys.exit(1)


if __name__ == "__main__":
    main()
