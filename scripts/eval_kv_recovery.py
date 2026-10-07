#!/usr/bin/env python
"""Three-way evaluation driver (spec §13-§15): dense / compressed / compressed_recovered
(+ optional dense_recovered) through the EXISTING eval harness, then the recovery report.

  python scripts/eval_kv_recovery.py run --run-name <trained run> [--submit]      # uses <run>/config.yaml
  python scripts/eval_kv_recovery.py run --config configs/kv_recovery/ministral_3b.yaml --run-name demo \
      [--checkpoint DIR] [--conditions dense,compressed,compressed_recovered[,dense_recovered]] \
      [--benchmarks ruler16k,ruler32k,longbench] [--dense-dir ruler16k=/path/to/dense/cell ...] \
      [--submit | --local | --dry-run] [--allow-compression-mismatch] [--allow-dense-mismatch]
  python scripts/eval_kv_recovery.py report --config ... --run-name demo [--n-resamples 10000 --seed 0]

Every arm is built from the ONE RecoveryConfig (eval_configs.build_cells): identical model
flags, prompt shaping, subsets, max_requests, seed, deterministic mode and query_aware=false;
compressed and compressed_recovered differ only in llm_kwargs.weight_delta (asserted). Cells are
barcode-named under <output.root>/eval/<model>/<benchmark>/ and shared across runs; cells with a
matching DONE.json are skipped. `report` re-scores predictions.csv rows through the benchmark's
own scorer, verifies that the arms are paired row-by-row and that their run_spec receipts agree,
and writes eval_results.{json,md} with paired, task-stratified bootstrap CIs.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

from eval_harness.kv_recovery.config import RecoveryConfig, load_config  # noqa: E402
from eval_harness.kv_recovery.eval_configs import (  # noqa: E402
    DEFAULT_CONDITIONS,
    RECOVERED_CONDITIONS,
    Cell,
    assert_delta_matches_config,
    build_cells,
    find_reusable_dense,
    results_root,
)

logger = logging.getLogger("kv_recovery.eval")
SBATCH = REPO_ROOT / "scripts" / "slurm" / "kv_recovery_eval.sbatch"


def _common(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--config", help="RecoveryConfig YAML; defaults to the run's own config.yaml when the run dir exists")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--run-name")
    ap.add_argument("--run-dir", help="explicit run directory (default <output.root>/<run_name>)")
    ap.add_argument("--output-root", default="outputs/kv_recovery", help="where <run_name> dirs live when --config is not given")


def _load(args) -> tuple[RecoveryConfig, Path]:
    """The evaluation must use the EXACT config the delta was trained with, so when the run
    directory already holds a config.yaml (written by train_kv_recovery.py) that file wins;
    ``--config`` is the fallback for runs that have not been trained yet (dense/compressed-only
    plans) and ``--set`` overrides apply on top of either (e.g. eval-only changes)."""
    shortcuts = {"run_name": args.run_name} if args.run_name else {}
    run_dir = Path(args.run_dir) if args.run_dir else None
    if run_dir is None and args.run_name:
        if args.config:
            base = load_config(args.config, shortcuts=shortcuts)
            run_dir = base.run_dir
        else:
            run_dir = Path(args.output_root) / args.run_name
    saved = (run_dir / "config.yaml") if run_dir is not None else None
    if saved is not None and saved.exists():
        if args.config:
            logger.info("using the run's own %s (the --config card is ignored for the training identity)", saved)
        cfg = load_config(saved, overrides=args.set, shortcuts=shortcuts)
    elif args.config:
        cfg = load_config(args.config, overrides=args.set, shortcuts=shortcuts)
    else:
        raise SystemExit(f"no config: pass --config, or --run-name/--run-dir of a trained run ({saved} not found)")
    run_dir = run_dir or cfg.run_dir
    return cfg, run_dir


def done_fingerprint(run_dir: Path) -> Optional[str]:
    p = run_dir / "DONE.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text()).get("fingerprint")
    except Exception:
        return None


def in_flight(job_name: str) -> Optional[str]:
    try:
        out = subprocess.run(["squeue", "--noheader", "--format=%i %j", "--name", job_name, "-u", os.environ.get("USER", "")],
                             capture_output=True, text=True, timeout=20)
        for line in out.stdout.splitlines():
            if line.strip():
                return line.split()[0]
    except Exception:
        pass
    return None


def cell_status(cell: Cell, *, force: bool) -> str:
    if not force and done_fingerprint(cell.run_dir) == cell.barcode:
        return "done"
    if in_flight(cell.job_name):
        return "in_flight"
    return "pending"


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
def cmd_run(args) -> int:
    cfg, run_dir = _load(args)
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    benches = [b.strip() for b in args.benchmarks.split(",")] if args.benchmarks else None
    ckpt = Path(args.checkpoint) if args.checkpoint else run_dir / "checkpoint"
    sha = None
    mismatch: Dict[str, Any] = {}
    if any(c in RECOVERED_CONDITIONS for c in conditions):
        from eval_harness.kv_recovery.checkpoint import checkpoint_digest, load_metadata

        if not (ckpt / "metadata.json").exists():
            raise SystemExit(f"no checkpoint at {ckpt}")
        load_metadata(ckpt)
        sha = checkpoint_digest(ckpt)
        mismatch = assert_delta_matches_config(ckpt, cfg, allow_mismatch=args.allow_compression_mismatch)
        if mismatch:
            logger.warning("compression/prompt mismatch between delta and eval (override active): %s", mismatch)
    cells = build_cells(cfg, checkpoint_dir=ckpt if sha else None, checkpoint_sha256=sha, conditions=conditions,
                        benchmarks=benches)

    # Reuse existing dense cells (fingerprint-checked unless overridden).
    dense_dirs = dict(cfg.eval.dense_dir or {})
    for item in args.dense_dir or []:
        b, p = item.split("=", 1)
        dense_dirs[b.strip()] = p.strip()
    reused: Dict[str, str] = {}
    for cell in cells:
        if cell.condition == "dense" and cell.benchmark in dense_dirs:
            cand = Path(dense_dirs[cell.benchmark])
            ok = find_reusable_dense(cand, cell.config)
            if ok is not None or args.allow_dense_mismatch:
                cell.run_dir = cand
                reused[cell.benchmark] = str(cand) + ("" if ok is not None else " (UNVERIFIED: dense mismatch override)")
            else:
                raise SystemExit(f"--dense-dir {cand} is not a completed dense cell with fingerprint {cell.barcode}; "
                                 "pass --allow-dense-mismatch to use it anyway (recorded)")

    cfg_dir = results_root(cfg) / "_configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "eval_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"cells": {}}
    manifest.update({"run_name": cfg.run_name, "config_digest": cfg.digest(), "model": cfg.model.name,
                     "checkpoint": {"path": str(ckpt), "sha256": sha} if sha else None,
                     "compression_mismatch_override": mismatch, "dense_reused": reused,
                     "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    walltimes = {b.benchmark: b.walltime for b in cfg.eval.benchmarks}
    for item in args.walltime or []:
        b, t = item.split("=", 1)
        walltimes[b.strip()] = t.strip()
    run_dir.mkdir(parents=True, exist_ok=True)
    plan: List[str] = []
    for cell in cells:
        yaml_path = cfg_dir / f"{cell.benchmark}__{cell.condition}__{cell.barcode}.yaml"
        yaml_path.write_text(yaml.safe_dump(cell.config, sort_keys=False))
        status = "reused" if (cell.condition == "dense" and cell.benchmark in reused) else cell_status(cell, force=args.force)
        entry = manifest["cells"].setdefault(cell.benchmark, {}).setdefault(cell.condition, {})
        entry.update({"run_dir": str(cell.run_dir), "barcode": cell.barcode, "config_file": str(yaml_path), "status": status})
        plan.append(f"{cell.benchmark:10s} {cell.condition:22s} {status:9s} {cell.run_dir}")
        if status != "pending":
            continue
        if args.submit:
            env = {**os.environ, "CONFIG_FILE": str(yaml_path)}
            cmd = ["sbatch", "--parsable", f"--job-name={cell.job_name}", f"--time={walltimes.get(cell.benchmark, '4:00:00')}",
                   "--export=ALL", str(SBATCH)]
            if args.force:
                cell.config["resume"] = False
                yaml_path.write_text(yaml.safe_dump(cell.config, sort_keys=False))
            out = subprocess.run(cmd, env=env, capture_output=True, text=True)
            if out.returncode != 0:
                raise SystemExit(f"sbatch failed for {cell.job_name}: {out.stderr}")
            entry["job_id"] = out.stdout.strip().split(";")[0]
            entry["status"] = "submitted"
            plan[-1] += f"  -> job {entry['job_id']}"
            time.sleep(float(args.stagger))
        elif args.local:
            from eval_harness.cli import CliEntryPoint

            CliEntryPoint().run(config_file=str(yaml_path), resume=False if args.force else None)
            entry["status"] = "done" if done_fingerprint(cell.run_dir) == cell.barcode else "failed"
            plan[-1] += f"  -> {entry['status']}"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    mode = "SUBMITTED" if args.submit else ("RAN LOCALLY" if args.local else "DRY RUN (pass --submit or --local)")
    print(f"=== kv recovery eval plan [{mode}] — manifest {manifest_path}")
    print("\n".join(plan))
    return 0


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def cmd_report(args) -> int:
    from eval_harness.kv_recovery.metrics import (benchmark_report, consistency_check, per_example_scores,
                                                  recovery_metrics, render_markdown, run_spec_comparability)
    from eval_harness.kv_recovery.provenance import provenance

    cfg, run_dir = _load(args)
    manifest_path = Path(args.manifest) if args.manifest else run_dir / "eval_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    results: Dict[str, Any] = {
        "schema_version": 1, "run_name": cfg.run_name, "model": {"name": cfg.model.name, "revision": cfg.model.revision},
        "kv_compression": cfg.kv_compression.__dict__, "checkpoint": manifest.get("checkpoint"),
        "compression_mismatch_override": manifest.get("compression_mismatch_override") or {},
        "dense_reused": manifest.get("dense_reused") or {}, "benchmarks": {}, "skipped": {},
        "protocol": {"bootstrap": {"n_resamples": args.n_resamples, "seed": args.seed, "alpha": args.alpha,
                                   "paired": True, "stratified_by": "task"},
                     "pairing": "task + row ordinal; row identity (task, question, answer) verified across arms",
                     "scoring": "re-scored from predictions.csv through the benchmark's own scorer",
                     "query_aware": False, "deterministic": True, "attn_implementation": cfg.model.attn_implementation,
                     "strip_auto_system_block": cfg.eval.strip_auto_system_block},
    }
    required = ["dense", "compressed", "compressed_recovered"]
    for bench, conds in manifest.get("cells", {}).items():
        have = {c: Path(e["run_dir"]) for c, e in conds.items()}
        missing = [c for c in required if c not in have or not (have[c] / "DONE.json").exists()]
        if missing:
            results["skipped"][bench] = f"incomplete arms: {missing}"
            continue
        arms = [c for c in required + ["dense_recovered"] if c in have and (have[c] / "DONE.json").exists()]
        specs = {c: json.loads((have[c] / "run_spec.json").read_text()) for c in arms}
        comp = run_spec_comparability(specs)
        if not comp["ok"] and not args.ignore_comparability:
            raise SystemExit(f"{bench}: arms are not comparable: {comp['problems']} (pass --ignore-comparability to record and continue)")
        frames, consistency = {}, {}
        for c in arms:
            frames[c] = per_example_scores(bench, have[c] / "predictions.csv")
            consistency[c] = consistency_check(frames[c], json.loads((have[c] / "metrics.json").read_text()))
        rep = benchmark_report(frames, n_resamples=args.n_resamples, seed=args.seed, alpha=args.alpha)
        rep["cells"] = {c: {"run_dir": str(have[c]), "fingerprint": specs[c].get("fingerprint")} for c in arms}
        rep["comparability"] = comp
        rep["consistency"] = consistency
        results["benchmarks"][bench] = rep
    if results["benchmarks"]:
        ov = [r["overall"] for r in results["benchmarks"].values()]
        n = len(ov)
        agg = recovery_metrics(sum(o["dense"] for o in ov) / n, sum(o["compressed"] for o in ov) / n,
                               sum(o["compressed_recovered"] for o in ov) / n,
                               (sum(o["dense_recovered"] for o in ov) / n) if all("dense_recovered" in o for o in ov) else None)
        agg["note"] = "mean of per-benchmark macro scores (RULER string-match and LongBench F1/ROUGE are different scales)"
        results["aggregate"] = {"macro_over_benchmarks": agg}
    results["provenance"] = provenance(REPO_ROOT)
    out = Path(args.out) if args.out else run_dir / "eval_results.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    md = render_markdown(results)
    out.with_suffix(".md").write_text(md)
    print(md)
    if results["skipped"]:
        print("skipped:", json.dumps(results["skipped"]))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    _common(r)
    r.add_argument("--checkpoint")
    r.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS))
    r.add_argument("--benchmarks")
    r.add_argument("--dense-dir", action="append", metavar="BENCH=PATH")
    r.add_argument("--walltime", action="append", metavar="BENCH=HH:MM:SS")
    g = r.add_mutually_exclusive_group()
    g.add_argument("--submit", action="store_true")
    g.add_argument("--local", action="store_true")
    g.add_argument("--dry-run", action="store_true")
    r.add_argument("--force", action="store_true", help="re-run cells even if a matching DONE.json exists")
    r.add_argument("--stagger", default="2", help="seconds between sbatch submissions")
    r.add_argument("--allow-compression-mismatch", action="store_true")
    r.add_argument("--allow-dense-mismatch", action="store_true")
    r.set_defaults(func=cmd_run)
    p = sub.add_parser("report")
    _common(p)
    p.add_argument("--manifest")
    p.add_argument("--out")
    p.add_argument("--n-resamples", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--ignore-comparability", action="store_true")
    p.set_defaults(func=cmd_report)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
