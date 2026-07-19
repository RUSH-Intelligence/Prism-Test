#!/usr/bin/env python3
"""Aggregate and orchestrate the ridge envelope_gamma RULER sweep.

Pipeline (see scripts/slurm/launch_ridge_gamma_tune.sh for the tuning phase):

  1. tune        collect tuning metrics (5 samples/subset, request_offset=100)
                 -> full grid CSV + winners.json with the best gamma per
                 (bench, ratio, subset). Exits nonzero while cells are missing,
                 so it doubles as a completeness check.
  2. launch-eval group subsets by winning gamma within each (bench, ratio) and
                 print (or --execute) the 100-sample eval sbatch commands
                 (request_offset=0, disjoint from the tuning rows).
  3. report      final markdown: per length, dataset x ratio tables of
                 (eval score, best gamma), plus gamma-preference
                 characterization by task family and tuning-curve summaries.

Tie-break rule for "best gamma" (5-sample string_match quantizes to multiples
of 20, so ties are common): highest tune score, then gamma closest to 1.0
(the neutral envelope), then the smaller gamma. Ties are reported.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

GAMMAS = ["0", "0.5", "1", "1.5", "2", "2.5", "3", "3.5", "4"]
RATIOS = ["0.2", "0.4", "0.6", "0.8"]
BENCHES = ["ruler16k", "ruler32k", "ruler64k", "ruler128k"]
SUBSETS = [
    "cwe", "fwe",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multiquery", "niah_multivalue",
    "niah_single_1", "niah_single_2", "niah_single_3",
    "qa_1", "qa_2", "vt",
]
FAMILY = {
    **{s: "retrieval (niah)" for s in SUBSETS if s.startswith("niah_")},
    "cwe": "aggregation", "fwe": "aggregation",
    "qa_1": "qa", "qa_2": "qa",
    "vt": "variable-tracking",
}

REPO = Path("/scratch/sj157/Prism-Test")
SBATCH_SCRIPT = REPO / "scripts/slurm/compactor_ruler.sbatch"
DEFAULT_ROOT = Path("/scratch/sj157/results_ridge_gamma")


def newest_metrics(cell_dir: Path) -> Path | None:
    """Newest metrics.json under a cell dir (re-runs nest /1, /2, ...)."""
    if not cell_dir.is_dir():
        return None
    candidates = list(cell_dir.rglob("metrics.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_task_scores(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text())
    out: dict[str, float] = {}
    for subset, metrics in (data.get("task_scores") or {}).items():
        if isinstance(metrics, dict):
            val = metrics.get("string_match")
        else:
            val = metrics
        if val is not None:
            out[str(subset)] = float(val)
    return out


def pick_best_gamma(scores_by_gamma: dict[str, float]) -> tuple[str, float, list[str]]:
    """(best_gamma, best_score, tied_gammas) under the documented tie-break."""
    best_score = max(scores_by_gamma.values())
    tied = [g for g, s in scores_by_gamma.items() if s == best_score]
    best = min(tied, key=lambda g: (abs(float(g) - 1.0), float(g)))
    return best, best_score, sorted(tied, key=float)


# ---------------------------------------------------------------- tune ----

def cmd_tune(args: argparse.Namespace) -> int:
    tune_root = Path(args.tune_root)
    grid_rows: list[dict] = []
    missing: list[str] = []
    scores: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    # (bench, ratio, subset) -> {gamma: score}

    for bench in BENCHES:
        for g in GAMMAS:
            for r in RATIOS:
                cell = tune_root / f"{bench}_g{g}_r{r}"
                mpath = newest_metrics(cell)
                if mpath is None:
                    missing.append(cell.name)
                    continue
                task_scores = load_task_scores(mpath)
                for subset in SUBSETS:
                    if subset not in task_scores:
                        missing.append(f"{cell.name}:{subset}")
                        continue
                    score = task_scores[subset]
                    scores[(bench, r, subset)][g] = score
                    grid_rows.append({
                        "bench": bench, "ratio": r, "subset": subset,
                        "gamma": g, "tune_score": score,
                    })

    grid_csv = Path(args.out_csv)
    with grid_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["bench", "ratio", "subset", "gamma", "tune_score"])
        writer.writeheader()
        writer.writerows(grid_rows)
    print(f"wrote {grid_csv} ({len(grid_rows)} rows)")

    winners: dict = defaultdict(lambda: defaultdict(dict))
    n_tied = 0
    for (bench, r, subset), by_gamma in sorted(scores.items()):
        if len(by_gamma) < len(GAMMAS):
            continue  # incomplete cell — reported via `missing`
        best, best_score, tied = pick_best_gamma(by_gamma)
        if len(tied) > 1:
            n_tied += 1
        winners[bench][r][subset] = {
            "gamma": best, "tune_score": best_score, "tied_gammas": tied,
        }

    winners_path = Path(args.out_json)
    winners_path.write_text(json.dumps(winners, indent=2, sort_keys=True))
    n_cells = sum(len(rd) for bd in winners.values() for rd in bd.values())
    print(f"wrote {winners_path} ({n_cells} (bench,ratio,subset) winners; "
          f"{n_tied} with tied best gammas)")

    if missing:
        print(f"\nINCOMPLETE — {len(missing)} missing tune cells/subsets:", file=sys.stderr)
        for name in missing[:40]:
            print(f"  {name}", file=sys.stderr)
        if len(missing) > 40:
            print(f"  ... and {len(missing) - 40} more", file=sys.stderr)
        return 1
    print("tuning grid complete.")
    return 0


# ---------------------------------------------------------- launch-eval ----

def _eval_cell_done(eval_dir: Path, subsets: list[str]) -> bool:
    mpath = newest_metrics(eval_dir)
    if mpath is None:
        return False
    task_scores = load_task_scores(mpath)
    return all(s in task_scores for s in subsets)


def _job_in_flight(jobname: str) -> bool:
    """True if a SLURM job with this name is already queued/running.

    metrics.json only appears when a run completes, so the done-check alone
    would double-submit cells whose jobs are still in the queue.
    """
    try:
        out = subprocess.run(
            ["squeue", "--noheader", f"--name={jobname}", "--format=%i"],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return bool(out.stdout.strip())


def cmd_launch_eval(args: argparse.Namespace) -> int:
    winners = json.loads(Path(args.winners).read_text())
    eval_root = Path(args.eval_root)
    n_launched = 0
    n_skipped = 0

    for bench in BENCHES:
        for r in RATIOS:
            cell = winners.get(bench, {}).get(r)
            if not cell:
                print(f"WARN: no winners for {bench} r={r}; run `tune` first", file=sys.stderr)
                continue
            groups: dict[str, list[str]] = defaultdict(list)
            for subset in SUBSETS:
                if subset in cell:
                    groups[cell[subset]["gamma"]].append(subset)
            for g, subsets in sorted(groups.items(), key=lambda kv: float(kv[0])):
                outdir = eval_root / f"{bench}_r{r}_g{g}"
                jobname = f"rge_{bench}_r{r}_g{g}"
                if _eval_cell_done(outdir, subsets):
                    n_skipped += 1
                    continue
                if args.execute and _job_in_flight(jobname):
                    print(f"in-flight, not resubmitting: {jobname}")
                    n_skipped += 1
                    continue
                # Vars go into the submitting environment + plain --export=ALL:
                # `--export=ALL,VAR=val` splits on commas and would mangle the
                # comma-separated SUBSETS list.
                cell_env = {
                    "BENCH": bench, "METHOD": "ridge", "RATIO": r,
                    "MAXLEN": "131072", "OUTDIR": str(outdir),
                    "SUBSETS": ",".join(subsets), "MAXREQ": "100", "OFFSET": "0",
                    "KWARGS": f"{{envelope_gamma: {g}}}",
                }
                if args.model:
                    cell_env["MODEL"] = args.model
                cmd = [
                    "sbatch", "--parsable",
                    f"--job-name={jobname}",
                    "--export=ALL",
                    str(SBATCH_SCRIPT),
                ]
                if args.execute:
                    outdir.mkdir(parents=True, exist_ok=True)
                    jobid = subprocess.run(
                        cmd, check=True, capture_output=True, text=True,
                        env={**os.environ, **cell_env},
                    ).stdout.strip()
                    print(f"submitted {jobid}  {jobname}  ({len(subsets)} subsets)")
                else:
                    env_prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in cell_env.items())
                    print(f"env {env_prefix} " + " ".join(shlex.quote(c) for c in cmd))
                n_launched += 1

    print(f"----\n{'submitted' if args.execute else 'printed'}={n_launched} skipped(done)={n_skipped}")
    return 0


# -------------------------------------------------------------- report ----

def cmd_report(args: argparse.Namespace) -> int:
    winners = json.loads(Path(args.winners).read_text())
    eval_root = Path(args.eval_root)

    # eval score lookup: (bench, ratio, subset) -> score at its winning gamma
    eval_scores: dict[tuple[str, str, str], float] = {}
    missing_eval: list[str] = []
    for bench in BENCHES:
        for r in RATIOS:
            cell = winners.get(bench, {}).get(r, {})
            groups: dict[str, list[str]] = defaultdict(list)
            for subset, info in cell.items():
                groups[info["gamma"]].append(subset)
            for g, subsets in groups.items():
                mpath = newest_metrics(eval_root / f"{bench}_r{r}_g{g}")
                task_scores = load_task_scores(mpath) if mpath else {}
                for subset in subsets:
                    if subset in task_scores:
                        eval_scores[(bench, r, subset)] = task_scores[subset]
                    else:
                        missing_eval.append(f"{bench}_r{r}_g{g}:{subset}")

    lines: list[str] = []
    lines.append("# Ridge envelope_gamma on RULER — tuned results")
    lines.append("")
    lines.append("Model: meta-llama/Llama-3.1-8B-Instruct, backend research, "
                 "kv_compressor ridge (key-normalized leverage, fixed-envelope), "
                 "attention_method none, single-pass prefill.")
    lines.append("")
    lines.append("Protocol: gamma tuned per (dataset, ratio, length) on 5 samples "
                 "(rows 100-104); reported scores are the disjoint 100-sample eval "
                 "split (rows 0-99) at the winning gamma. Ratio = fraction of KV "
                 "pruned. Tie-break: max tune score, then gamma closest to 1.0, "
                 "then smaller gamma — 5-sample scores quantize to multiples of 20, "
                 "so tuned gammas are coarse estimates.")
    lines.append("")

    for bench in BENCHES:
        lines.append(f"## {bench}")
        lines.append("")
        header = "| dataset | " + " | ".join(f"r={r}" for r in RATIOS) + " |"
        sep = "|---" * (len(RATIOS) + 1) + "|"
        lines.append(header)
        lines.append(sep)
        col_scores: dict[str, list[float]] = defaultdict(list)
        for subset in SUBSETS:
            cells = []
            for r in RATIOS:
                info = winners.get(bench, {}).get(r, {}).get(subset)
                score = eval_scores.get((bench, r, subset))
                if info is None or score is None:
                    cells.append("—")
                    continue
                tie_mark = "*" if len(info.get("tied_gammas", [])) > 1 else ""
                cells.append(f"{score:.1f} (γ={info['gamma']}{tie_mark})")
                col_scores[r].append(score)
            lines.append(f"| {subset} | " + " | ".join(cells) + " |")
        mean_cells = []
        for r in RATIOS:
            vals = col_scores[r]
            mean_cells.append(f"**{sum(vals) / len(vals):.1f}**" if vals else "—")
        lines.append("| **mean** | " + " | ".join(mean_cells) + " |")
        lines.append("")
        lines.append("`*` = tuned gamma had ties at the 5-sample tune score; "
                     "tie-break favored the gamma closest to 1.0.")
        lines.append("")

    # Characterization: winning-gamma tendencies by task family.
    lines.append("## Gamma characterization by task family")
    lines.append("")
    lines.append("| family | " + " | ".join(f"r={r}" for r in RATIOS) + " | overall |")
    lines.append("|---" * (len(RATIOS) + 2) + "|")
    fam_gammas_all: dict[str, list[float]] = defaultdict(list)
    fam_ratio_gammas: dict[tuple[str, str], list[float]] = defaultdict(list)
    for bench in BENCHES:
        for r in RATIOS:
            for subset, info in winners.get(bench, {}).get(r, {}).items():
                fam = FAMILY[subset]
                fam_ratio_gammas[(fam, r)].append(float(info["gamma"]))
                fam_gammas_all[fam].append(float(info["gamma"]))
    for fam in sorted(set(FAMILY.values())):
        cells = []
        for r in RATIOS:
            vals = fam_ratio_gammas.get((fam, r), [])
            cells.append(f"{sum(vals) / len(vals):.2f}" if vals else "—")
        overall = fam_gammas_all.get(fam, [])
        cells.append(f"{sum(overall) / len(overall):.2f}" if overall else "—")
        lines.append(f"| {fam} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("(cells are the mean winning gamma across the family's datasets "
                 "and the four context lengths; higher = the query-side omega "
                 "signal helps, lower = pure ridge diversity suffices)")
    lines.append("")

    # Tuning-curve summary from the grid CSV, if available.
    grid_path = Path(args.tune_grid) if args.tune_grid else None
    if grid_path and grid_path.exists():
        by_fam_gamma: dict[tuple[str, str], list[float]] = defaultdict(list)
        with grid_path.open() as fh:
            for row in csv.DictReader(fh):
                fam = FAMILY.get(row["subset"])
                if fam:
                    by_fam_gamma[(fam, row["gamma"])].append(float(row["tune_score"]))
        lines.append("## Mean tune-score vs gamma (all lengths/ratios pooled)")
        lines.append("")
        lines.append("| family | " + " | ".join(f"γ={g}" for g in GAMMAS) + " |")
        lines.append("|---" * (len(GAMMAS) + 1) + "|")
        for fam in sorted(set(FAMILY.values())):
            cells = []
            for g in GAMMAS:
                vals = by_fam_gamma.get((fam, g), [])
                cells.append(f"{sum(vals) / len(vals):.1f}" if vals else "—")
            lines.append(f"| {fam} | " + " | ".join(cells) + " |")
        lines.append("")

    if missing_eval:
        lines.append(f"## MISSING EVAL CELLS ({len(missing_eval)})")
        lines.extend(f"- {m}" for m in missing_eval[:40])
        lines.append("")

    out = Path(args.out)
    out.write_text("\n".join(lines))
    print(f"wrote {out}")
    if missing_eval:
        print(f"WARNING: {len(missing_eval)} missing eval cells — rerun launch-eval",
              file=sys.stderr)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_tune = sub.add_parser("tune", help="aggregate tuning cells -> winners.json + grid CSV")
    p_tune.add_argument("--tune-root", default=str(DEFAULT_ROOT / "tune"))
    p_tune.add_argument("--out-json", default=str(DEFAULT_ROOT / "winners.json"))
    p_tune.add_argument("--out-csv", default=str(DEFAULT_ROOT / "tune_grid.csv"))

    p_launch = sub.add_parser("launch-eval", help="print/submit eval jobs at winning gammas")
    p_launch.add_argument("--winners", default=str(DEFAULT_ROOT / "winners.json"))
    p_launch.add_argument("--eval-root", default=str(DEFAULT_ROOT / "eval"))
    p_launch.add_argument("--model", default=None, help="override MODEL for the sbatch")
    p_launch.add_argument("--execute", action="store_true", help="submit instead of print")

    p_report = sub.add_parser("report", help="final markdown report from eval results")
    p_report.add_argument("--winners", default=str(DEFAULT_ROOT / "winners.json"))
    p_report.add_argument("--eval-root", default=str(DEFAULT_ROOT / "eval"))
    p_report.add_argument("--tune-grid", default=str(DEFAULT_ROOT / "tune_grid.csv"))
    p_report.add_argument("--out", default=str(DEFAULT_ROOT / "ridge_gamma_ruler.md"))

    args = parser.parse_args()
    if args.cmd == "tune":
        return cmd_tune(args)
    if args.cmd == "launch-eval":
        return cmd_launch_eval(args)
    if args.cmd == "report":
        return cmd_report(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
