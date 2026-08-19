#!/usr/bin/env python
"""Generic KV-compression sweep engine driven by a single sweep.yaml.

This is benchmark-agnostic and model-agnostic: it reads ONE config file that
lists the dials as lists (models, benchmarks, methods, ratios) plus optional
per-method sweep grids (Ridge / Verified), and runs every combination through
the normal research backend.

Two things it does, selected by mode:

  * ``--submit``      Read the config and submit one SLURM array job per
                      (model x benchmark) pair. This is what the launcher
                      ``scripts/submit_sweep.sh`` calls. Add ``--dry-run`` to
                      print the plan (and the exact sbatch commands) without
                      submitting.

  * ``--cell-index N``  Run exactly the Nth cell of ONE (model, benchmark)
                      pair (``--model`` / ``--benchmark`` required) and exit.
                      This is what each array task runs inside sweep.sbatch.
                      Writes a per-cell manifest fragment so concurrent array
                      tasks never race on one file; ``--resume`` skips a cell
                      whose metrics.json already exists.

Cell grid for one (model, benchmark) pair, in order:

    Full baseline (kv_compressor=none, once; toggled by full_baseline)
  + every ORDINARY method x every ratio
  + Ridge:    gamma x lambda x rotate_queries x ratio    (from the `ridge:` block)
  + Verified: det_fraction x (Ridge inner grid) x ratio  (from the `verified:` block)

`--cell-index` indexing is a pure function of (config, model, benchmark), so the
launcher's cell count and the array task's cell selection always agree.

This engine deliberately does NOT import the legacy per-benchmark sweep scripts
(longbench_sweep.py, ...). It is the standalone replacement for them.
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

# --- Method quirks carried over from the legacy sweep (documented in CLAUDE.md).
# pyramidkv leaves the cache cross-layer ragged -> needs flash_attention_2.
FLASH_ATTN_METHODS = {"pyramidkv"}
# Compressors that draw random Gaussian sketches at score-time; seed them so
# reruns of the same cell are reproducible.
SEEDED_METHODS = {"cur", "compactor"}
SWEEP_SEED = 42

# Methods with their own dedicated sweep block (handled specially below); every
# other listed method is an "ordinary" method x ratio cell.
SPECIAL_METHODS = {"ridge", "verified"}

DEFAULT_TEMPLATE = "evaluate/sweep_base.yaml"
DEFAULT_RATIOS = [0.6, 0.9, 0.95]


# ============================================================================
# Config loading + small helpers
# ============================================================================
def _slug(model: str) -> str:
    return model.replace("/", "--")


def _extended(path: Path) -> str:
    r"""Extended-length (``\\?\``) path on Windows so long run-dir names fit."""
    ap = path.resolve()
    if sys.platform == "win32":
        return "\\\\?\\" + str(ap).replace("/", "\\")
    return str(ap)


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not cfg.get("models"):
        sys.exit(f"{path}: `models:` is required (a non-empty list).")
    if not cfg.get("benchmarks"):
        sys.exit(f"{path}: `benchmarks:` is required (a non-empty list).")
    return cfg


def _as_list(v) -> list:
    if v is None:
        return []
    return list(v) if isinstance(v, (list, tuple)) else [v]


def _bool_list(v) -> list:
    """A special-block list that may be absent. Returns [None] when absent so
    the cell loop leaves the compressor's own default in place."""
    lst = _as_list(v)
    return lst if lst else [None]


def resolve_model_settings(cfg: dict, model: str) -> dict:
    """Template + max_model_len + per-model llm_kwargs (model side-table wins).

    `llm_kwargs` is the escape hatch for a model's special load flags — e.g.
    Ministral needs `dequantize_fp8: true` — so no model ever needs its own
    template file; a couple of side-table lines suffice."""
    ov = (cfg.get("model_overrides") or {}).get(model, {}) or {}
    return {
        "template": ov.get("template", cfg.get("template", DEFAULT_TEMPLATE)),
        "max_model_len": ov.get("max_model_len", cfg.get("max_model_len")),
        "llm_kwargs": ov.get("llm_kwargs") or {},
    }


def resolve_benchmark_settings(cfg: dict, benchmark: str) -> dict:
    """Subsets + optional max_model_len for one benchmark (side-table wins)."""
    ov = (cfg.get("benchmark_overrides") or {}).get(benchmark, {}) or {}
    subsets = ov.get("subsets", cfg.get("subsets"))
    return {
        "subsets": _as_list(subsets) or None,   # None -> benchmark's own defaults
        "max_model_len": ov.get("max_model_len"),
    }


# ============================================================================
# Cell grid — pure function of (config, model, benchmark)
# ============================================================================
def build_cells(cfg: dict, model: str, benchmark: str) -> list[tuple]:
    """Return an ordered list of cells: (label, kv_key, ratio|None, cell_id, extras)."""
    ratios = [float(r) for r in _as_list(cfg.get("ratios")) or DEFAULT_RATIOS]
    methods = _as_list(cfg.get("methods"))
    ordinary = [m for m in methods if m not in SPECIAL_METHODS]

    ridge = cfg.get("ridge") or {}
    verified = cfg.get("verified") or {}

    # Ridge dials (absent list -> [None] = single cell at the press default).
    ridge_gammas = _bool_list(ridge.get("gammas"))
    ridge_lambdas = _bool_list(ridge.get("lambdas"))
    ridge_rq = _bool_list(ridge.get("rotate_queries"))
    ridge_sink = ridge.get("sink_size")
    ridge_local = ridge.get("local_size")

    cells: list[tuple] = []

    # 1. Full baseline (once).
    if cfg.get("full_baseline", True):
        cells.append(("Full", "none", None, "Full", {}))

    # 2. Ordinary methods x ratio.
    for key in ordinary:
        for ratio in ratios:
            cells.append((key, key, ratio, f"{key}__r{ratio}", {}))

    def _ridge_extras(gamma, lam, rq) -> tuple[dict, str]:
        extras: dict = {}
        tag = ""
        if gamma is not None:
            extras["envelope_gamma"] = float(gamma)
            tag += f"_g{gamma:g}"
        if lam is not None:
            extras["ridge_lambda"] = float(lam)
            tag += f"_l{lam:g}"
        if rq is not None:
            extras["rotate_queries"] = bool(rq)
            tag += f"_rq{'T' if rq else 'F'}"
        if ridge_sink is not None:
            extras["sink_size"] = int(ridge_sink)
            tag += f"_sk{int(ridge_sink)}"
        if ridge_local is not None:
            extras["local_size"] = int(ridge_local)
            tag += f"_lo{int(ridge_local)}"
        return extras, tag

    # 3. Ridge: gamma x lambda x rotate_queries x ratio.
    if "ridge" in methods:
        for gamma in ridge_gammas:
            for lam in ridge_lambdas:
                for rq in ridge_rq:
                    extras, tag = _ridge_extras(gamma, lam, rq)
                    for ratio in ratios:
                        cell_id = f"Ridge{tag}__r{ratio}"
                        cells.append(("Ridge", "ridge", ratio, cell_id, dict(extras)))

    # 4. Verified: det_fraction x (Ridge inner grid) x ratio.
    if "verified" in methods:
        inner = verified.get("inner", "ridge")
        det_fractions = _as_list(verified.get("det_fractions")) or [None]
        for det in det_fractions:
            # det == 0 mutes the inner's scoring -> a single gamma-free anchor.
            gammas = ridge_gammas if (det is None or float(det) > 0.0) else [None]
            for gamma in gammas:
                for lam in ridge_lambdas:
                    for rq in ridge_rq:
                        inner_kwargs, tag = _ridge_extras(gamma, lam, rq)
                        extras = {"inner": inner, "sample_seed": SWEEP_SEED,
                                  "inner_kwargs": inner_kwargs}
                        det_tag = "" if det is None else f"_d{float(det):g}".replace(".", "p")
                        if det is not None:
                            extras["det_fraction"] = float(det)
                        for ratio in ratios:
                            cell_id = f"Verified{det_tag}{tag}__r{ratio}"
                            cells.append(("Verified", "verified", ratio, cell_id, copy.deepcopy(extras)))

    return cells


# ============================================================================
# Running one cell (build EvalConfig -> subprocess the CLI)
# ============================================================================
def build_run_config(base: dict, *, model: str, benchmark: str, kv_compressor: str,
                     ratio: float, subsets: list[str] | None, out_root: Path,
                     cell_id: str, max_requests: int | None,
                     max_model_len: int | None, extra_kv_kwargs: dict | None,
                     extra_llm_kwargs: dict | None = None) -> dict:
    """Construct the full EvalConfig dict for one cell (ported from the legacy
    sweep's build_config, generalized over benchmark)."""
    c = copy.deepcopy(base)
    c["benchmark"] = benchmark
    if subsets:
        c["subsets"] = ",".join(subsets)
    else:
        c.pop("subsets", None)          # benchmark's own default_subsets win
    c["backend"] = "research"
    c["model"] = model
    c["deterministic"] = True           # comparable, reproducible numbers
    c["max_new_tokens"] = None          # per-task benchmark value wins
    c["temperature"] = 0.0
    c["top_p"] = 1.0
    if max_model_len is not None:
        c["max_model_len"] = max_model_len
    if max_requests is not None:
        c["max_requests"] = max_requests
    c["output_dir"] = _extended(out_root / cell_id)

    llm = dict(c.get("llm_kwargs") or {})
    rc = dict(llm.get("research_config") or {})
    # Per-model llm_kwargs from the model side-table (e.g. dequantize_fp8 for
    # Ministral). A nested research_config here is merged, not replaced.
    if extra_llm_kwargs:
        ov = dict(extra_llm_kwargs)
        ov_rc = ov.pop("research_config", None)
        if isinstance(ov_rc, dict):
            rc.update(ov_rc)
        llm.update(ov)
    # pyramidkv needs flash-attn; otherwise keep the base/override choice (sdpa default).
    llm["attn_implementation"] = ("flash_attention_2" if kv_compressor in FLASH_ATTN_METHODS
                                  else llm.get("attn_implementation", "sdpa"))
    rc["kv_compressor"] = kv_compressor
    rc["compression_ratio"] = 0.0 if kv_compressor == "none" else float(ratio)
    rc["attention_method"] = "none"
    rc["attention_method_kwargs"] = {}
    kv_kwargs = dict(rc.get("kv_compressor_kwargs") or {})
    if kv_compressor in SEEDED_METHODS:
        kv_kwargs.setdefault("seed", SWEEP_SEED)
    if extra_kv_kwargs:
        kv_kwargs.update(extra_kv_kwargs)
    if kv_kwargs:
        rc["kv_compressor_kwargs"] = kv_kwargs
    rc.pop("prefill_chunk_size", None)
    llm["research_config"] = rc
    c["llm_kwargs"] = llm
    return c


def find_metrics(out_dir: Path) -> Path | None:
    hits = sorted(out_dir.rglob("metrics.json"), key=lambda p: p.stat().st_mtime)
    return hits[-1] if hits else None


def _total_samples(metrics_path: Path | None) -> int | None:
    if not metrics_path or not Path(metrics_path).exists():
        return None
    try:
        return int(json.loads(Path(metrics_path).read_text(encoding="utf-8")).get("total_samples", 0)) or None
    except (OSError, ValueError):
        return None


def out_root_for(cfg: dict, model: str, benchmark: str, cli_out_root: str | None) -> Path:
    if cli_out_root:
        base = Path(cli_out_root)
    else:
        base = REPO_ROOT / "results" / str(cfg.get("sweep_name", "sweep"))
    return base / _slug(model) / benchmark


# ============================================================================
# Mode: run one cell
# ============================================================================
def run_one_cell(cfg: dict, cfg_path: Path, model: str, benchmark: str,
                 cell_index: int, resume: bool, cli_out_root: str | None) -> int:
    cells = build_cells(cfg, model, benchmark)
    if not 0 <= cell_index < len(cells):
        sys.exit(f"--cell-index {cell_index} out of range [0, {len(cells)}) for "
                 f"{model} / {benchmark}")
    label, key, ratio, cell_id, extras = cells[cell_index]

    mset = resolve_model_settings(cfg, model)
    bset = resolve_benchmark_settings(cfg, benchmark)
    max_model_len = bset["max_model_len"] or mset["max_model_len"]
    base = yaml.safe_load((REPO_ROOT / mset["template"]).read_text(encoding="utf-8")) or {}

    out_root = out_root_for(cfg, model, benchmark, cli_out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "manifest.cells").mkdir(parents=True, exist_ok=True)
    frag = out_root / "manifest.cells" / f"cell_{cell_index:03d}.json"
    cell_dir = out_root / cell_id

    existing = find_metrics(cell_dir) if cell_dir.exists() else None
    elapsed = None
    if resume and existing is not None:
        print(f"[cell {cell_index}] {cell_id}: resume — found {existing}")
        metrics_path, rc = existing, 0
    else:
        print(f"[cell {cell_index}] {cell_id}: running  ({model} / {benchmark})", flush=True)
        run_cfg = build_run_config(
            base, model=model, benchmark=benchmark, kv_compressor=key,
            ratio=ratio or 0.0, subsets=bset["subsets"], out_root=out_root,
            cell_id=cell_id, max_requests=cfg.get("max_requests"),
            max_model_len=max_model_len, extra_kv_kwargs=extras or None,
            extra_llm_kwargs=mset["llm_kwargs"] or None)
        tmp_yaml = out_root / f"_cell_config_{os.getpid()}.yaml"
        tmp_yaml.write_text(yaml.safe_dump(run_cfg, sort_keys=False), encoding="utf-8")
        t0 = time.time()
        rc = subprocess.run(
            [sys.executable, "-m", "eval_harness.cli", "run", "--config_file", str(tmp_yaml)],
            cwd=str(REPO_ROOT)).returncode
        elapsed = time.time() - t0
        try:
            tmp_yaml.unlink()
        except FileNotFoundError:
            pass
        metrics_path = find_metrics(cell_dir)
        samples = _total_samples(metrics_path)
        per = f"{elapsed / samples:.2f}s/sample" if samples else "n/a"
        print(f"    -> {elapsed:.1f}s wall, {samples or '?'} samples ({per})", flush=True)

    record = {
        "index": cell_index, "model": model, "benchmark": benchmark,
        "label": label, "kv_compressor": key, "ratio": ratio, "cell_id": cell_id,
        "kv_compressor_kwargs": extras or None, "returncode": rc,
        "elapsed_sec": round(elapsed, 1) if elapsed else None,
        "total_samples": _total_samples(metrics_path),
        "metrics": str(metrics_path) if metrics_path else None,
        "ok": rc == 0 and metrics_path is not None,
    }
    frag.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return 0 if record["ok"] else 1


# ============================================================================
# Mode: submit (one SLURM array per model x benchmark)
# ============================================================================
def _sbatch_command(cfg: dict, cfg_path: Path, model: str, benchmark: str,
                    n_cells: int) -> list[str]:
    s = cfg.get("slurm") or {}
    job_name = f"sweep-{_slug(model)[:20]}-{benchmark}"
    # Flat log filenames under logs/sweep/ (which the launcher creates) — SLURM
    # opens these before the job body runs, so the directory must already exist.
    log_stem = "logs/sweep/%x-%A_%a"
    cmd = [
        "sbatch",
        f"--job-name={job_name}",
        f"--array=0-{n_cells - 1}",
        f"--gres=gpu:{s.get('gpu', 'rtxa6000')}:1",
        f"--cpus-per-task={s.get('cpus', 8)}",
        f"--mem={s.get('mem', '64G')}",
        f"--time={s.get('time', '12:00:00')}",
        f"--output={log_stem}.out",
        f"--error={log_stem}.err",
    ]
    for key in ("partition", "qos", "account"):
        if s.get(key):
            cmd.append(f"--{key}={s[key]}")
    export = f"ALL,SWEEP_CONFIG={cfg_path},SWEEP_MODEL={model},SWEEP_BENCHMARK={benchmark}"
    cmd += [f"--export={export}", "scripts/sweep.sbatch"]
    return cmd


def submit(cfg: dict, cfg_path: Path, dry_run: bool, cli_out_root: str | None) -> int:
    models = _as_list(cfg["models"])
    benchmarks = _as_list(cfg["benchmarks"])
    print(f"Sweep config: {cfg_path}")
    print(f"Models:     {models}")
    print(f"Benchmarks: {benchmarks}")
    print(f"Methods:    {_as_list(cfg.get('methods'))}  ratios={_as_list(cfg.get('ratios')) or DEFAULT_RATIOS}"
          f"  full_baseline={cfg.get('full_baseline', True)}")
    print("-" * 72)

    total_jobs = total_cells = 0
    for model in models:
        for benchmark in benchmarks:
            cells = build_cells(cfg, model, benchmark)
            if not cells:
                print(f"  {model} / {benchmark}: 0 cells — skipped (empty methods + no full_baseline)")
                continue
            total_jobs += 1
            total_cells += len(cells)
            out_root = out_root_for(cfg, model, benchmark, cli_out_root)
            cmd = _sbatch_command(cfg, cfg_path, model, benchmark, len(cells))
            print(f"  {model} / {benchmark}: {len(cells)} cells -> {out_root}")
            if dry_run:
                print(f"      $ {' '.join(cmd)}")
                # Show the cell map so the array indices are greppable.
                for i, (label, key, ratio, cell_id, extras) in enumerate(cells):
                    flag = " [flash_attn_2]" if key in FLASH_ATTN_METHODS else ""
                    ex = f" extras={extras}" if extras else ""
                    print(f"        [{i:2d}] {cell_id:30s} kv={key} ratio={ratio}{flag}{ex}")
            else:
                Path("logs/sweep").mkdir(parents=True, exist_ok=True)
                res = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
                if res.returncode != 0:
                    print(f"      sbatch FAILED: {res.stderr.strip()}")
                    return 1
                print(f"      submitted: {res.stdout.strip()}")

    print("-" * 72)
    verb = "would submit" if dry_run else "submitted"
    print(f"{verb} {total_jobs} array job(s), {total_cells} cells total.")
    if not dry_run and total_jobs:
        print("Per-cell results: results/sweep/<model>/<benchmark>/ (metrics under each cell "
              "dir; per-cell manifest fragments in manifest.cells/). Re-run "
              "./scripts/submit_sweep.sh anytime — finished cells resume/skip automatically.")
    return 0


# ============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="sweep.yaml", help="Sweep config YAML (default: ./sweep.yaml)")
    ap.add_argument("--out-root", default=None, help="Override results root (default: results/sweep)")
    # submit mode
    ap.add_argument("--submit", action="store_true", help="Submit SLURM array jobs")
    ap.add_argument("--dry-run", action="store_true", help="Print the plan (and sbatch commands); submit nothing")
    # cell mode (inside the array)
    ap.add_argument("--model", default=None, help="cell mode: which model")
    ap.add_argument("--benchmark", default=None, help="cell mode: which benchmark")
    ap.add_argument("--cell-index", type=int, default=None, help="cell mode: run the Nth cell and exit")
    ap.add_argument("--resume", action="store_true", help="cell mode: skip a cell whose metrics.json exists")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (REPO_ROOT / cfg_path).resolve() if not cfg_path.exists() else cfg_path.resolve()
    if not cfg_path.exists():
        sys.exit(f"Config not found: {cfg_path}")
    cfg = load_config(cfg_path)

    if args.cell_index is not None:
        if not (args.model and args.benchmark):
            sys.exit("--cell-index requires --model and --benchmark")
        sys.exit(run_one_cell(cfg, cfg_path, args.model, args.benchmark,
                              args.cell_index, args.resume, args.out_root))

    # default (and --submit) -> submit / plan
    sys.exit(submit(cfg, cfg_path, dry_run=args.dry_run or not args.submit,
                    cli_out_root=args.out_root))


if __name__ == "__main__":
    main()
