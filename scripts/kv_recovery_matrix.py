#!/usr/bin/env python
"""Expand configs/kv_recovery/matrix.yaml into training runs and (optionally) submit them.

  python scripts/kv_recovery_matrix.py --dry-run                 # print every cell + sbatch command
  python scripts/kv_recovery_matrix.py --primary --submit        # the pilot preset (16K, ratio 0.75)
  python scripts/kv_recovery_matrix.py --models ministral_3b --compressors cur --trainable qo_last4 --submit
  python scripts/kv_recovery_matrix.py --ablations --dry-run     # pilot-cell ablations

Each cell = <base config> + dotted overrides (contexts x compressors x ratios x trainable
subsets [+ model overrides] + the pre-registered hyperparameters); run_name =
<model>_<context>_<compressor>_r<ratio>_<trainable>. Idempotent: a cell whose
<output>/<run_name>/checkpoint/metadata.json exists (or whose job is in the queue) is skipped.
A manifest TSV is appended under <out-root>/matrix_manifest.tsv.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SBATCH = REPO_ROOT / "scripts" / "slurm" / "kv_recovery_train.sbatch"


@dataclass
class MatrixCell:
    run_name: str
    model_key: str
    base_config: str
    context: str
    compressor: str
    ratio: float
    trainable: str
    overrides: Dict[str, Any] = field(default_factory=dict)
    ablation: Optional[str] = None

    def set_args(self) -> List[str]:
        return [f"{k}={json.dumps(v)}" for k, v in self.overrides.items()]


def load_matrix(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text()) or {}


def _ratio_tag(r: float) -> str:
    return f"r{int(round(float(r) * 100)):03d}"


def expand(matrix: dict, *, models: Optional[List[str]] = None, contexts: Optional[List[str]] = None,
           compressors: Optional[List[str]] = None, ratios: Optional[List[float]] = None,
           trainables: Optional[List[str]] = None, primary: bool = False, ablations: bool = False) -> List[MatrixCell]:
    base = matrix["base_configs"]
    ctxs = matrix["contexts"]
    hp = matrix.get("hyperparameters") or {}
    model_ov = matrix.get("model_overrides") or {}
    prim = matrix.get("primary") or {}
    sel_models = models or list(base)
    sel_ctx = contexts or (prim.get("contexts") if primary else None) or list(ctxs)
    sel_comp = compressors or matrix["compressors"]
    sel_ratios = ratios or (prim.get("ratios") if primary else None) or matrix["ratios"]
    sel_train = trainables or list(matrix["trainable"])
    cells: List[MatrixCell] = []

    def make(mk, ctx, comp, r, tk, extra=None, abl=None):
        ov: Dict[str, Any] = dict(hp)
        ov.update(ctxs[ctx])
        ov["kv_compression.kv_compressor"] = comp
        ov["kv_compression.compression_ratio"] = float(r)
        ov.update(matrix["trainable"][tk])
        ov.update((model_ov.get(mk) or {}).get(tk) or {})
        if extra:
            ov.update(extra)
        name = f"{mk}_{ctx}_{comp}_{_ratio_tag(r)}_{tk}" + (f"_{abl}" if abl else "")
        ov["run_name"] = name
        return MatrixCell(run_name=name, model_key=mk, base_config=base[mk], context=ctx, compressor=comp,
                          ratio=float(r), trainable=tk, overrides=ov, ablation=abl)

    if ablations:
        pilot = matrix.get("pilot") or {}
        mk, ctx, comp, r, tk = (pilot.get("model", sel_models[0]), pilot.get("context", sel_ctx[0]),
                                pilot.get("compressor", sel_comp[0]), pilot.get("ratio", sel_ratios[0]),
                                pilot.get("trainable", sel_train[0]))
        for abl, extra in (matrix.get("ablations") or {}).items():
            cells.append(make(mk, ctx, comp, r, tk, extra=extra, abl=abl))
        return cells
    for mk in sel_models:
        for ctx in sel_ctx:
            for comp in sel_comp:
                for r in sel_ratios:
                    for tk in sel_train:
                        cells.append(make(mk, ctx, comp, r, tk))
    return cells


def cell_status(cell: MatrixCell, out_root: Path) -> str:
    if (out_root / cell.run_name / "checkpoint" / "metadata.json").exists():
        return "done"
    try:
        out = subprocess.run(["squeue", "--noheader", "--format=%i %j", "--name", f"kvrec_train_{cell.run_name}",
                              "-u", os.environ.get("USER", "")], capture_output=True, text=True, timeout=20)
        if out.stdout.strip():
            return "in_flight"
    except Exception:
        pass
    return "pending"


def sbatch_command(cell: MatrixCell, *, time_limit: str) -> List[str]:
    return ["sbatch", "--parsable", f"--job-name=kvrec_train_{cell.run_name}", f"--time={time_limit}", "--export=ALL", str(SBATCH)]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix", default=str(REPO_ROOT / "configs" / "kv_recovery" / "matrix.yaml"))
    ap.add_argument("--out-root", default=str(REPO_ROOT / "outputs" / "kv_recovery"))
    ap.add_argument("--models"); ap.add_argument("--contexts"); ap.add_argument("--compressors")
    ap.add_argument("--ratios"); ap.add_argument("--trainable")
    ap.add_argument("--primary", action="store_true")
    ap.add_argument("--ablations", action="store_true")
    ap.add_argument("--time", default="6:00:00")
    ap.add_argument("--max-jobs", type=int, default=16)
    ap.add_argument("--stagger", type=float, default=3.0)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--submit", action="store_true")
    g.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    split = lambda s: [x.strip() for x in s.split(",") if x.strip()] if s else None  # noqa: E731
    matrix = load_matrix(Path(args.matrix))
    cells = expand(matrix, models=split(args.models), contexts=split(args.contexts), compressors=split(args.compressors),
                   ratios=[float(x) for x in split(args.ratios)] if args.ratios else None, trainables=split(args.trainable),
                   primary=args.primary, ablations=args.ablations)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = out_root / "matrix_manifest.tsv"
    submitted = 0
    for cell in cells:
        status = cell_status(cell, out_root)
        cmd = sbatch_command(cell, time_limit=args.time)
        line = f"{status:9s} {cell.run_name:60s} {cell.base_config}"
        if args.submit and status == "pending" and submitted < args.max_jobs:
            env = {**os.environ, "CONFIG": cell.base_config, "RUN_NAME": cell.run_name,
                   "SET": "\n".join(a for a in cell.set_args() if not a.startswith("run_name="))}
            out = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=str(REPO_ROOT))
            if out.returncode != 0:
                raise SystemExit(f"sbatch failed for {cell.run_name}: {out.stderr}")
            job = out.stdout.strip().split(";")[0]
            with manifest.open("a") as f:
                f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\t{job}\t{cell.run_name}\t{cell.base_config}\t{json.dumps(cell.overrides)}\n")
            line += f"  -> job {job}"
            submitted += 1
            time.sleep(args.stagger)
        elif not args.submit:
            line += "\n    " + " ".join(cmd) + "\n    SET=" + " ".join(cell.set_args())
        print(line)
    print(f"{len(cells)} cells; submitted {submitted}" if args.submit else f"{len(cells)} cells (dry run; pass --submit)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
