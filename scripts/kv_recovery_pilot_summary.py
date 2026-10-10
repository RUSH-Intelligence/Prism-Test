#!/usr/bin/env python
"""Aggregate eval_results.json across KV-recovery runs into one table.

  python scripts/kv_recovery_pilot_summary.py [--glob 'outputs/kv_recovery/*_16k_*_r075_*'] [--out outputs/kv_recovery/pilot_summary.md]

One row per (run, benchmark): dense / compressed / recovered macro scores, compression drop,
recovery with its paired-bootstrap CI, recovery fraction with CI, and flags. Also writes a CSV.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path


def rows_for(run_dir: Path):
    p = run_dir / "eval_results.json"
    if not p.exists():
        return []
    r = json.loads(p.read_text())
    out = []
    for bench, rep in (r.get("benchmarks") or {}).items():
        o = rep["overall"]
        ci_r, ci_f = o["ci"]["recovery"], o["ci"]["recovery_fraction"]
        out.append({
            "run": run_dir.name, "benchmark": bench, "n": o["n_examples"],
            "dense": o["dense"], "compressed": o["compressed"], "recovered": o["compressed_recovered"],
            "drop": o["compression_drop"], "recovery": o["recovery"],
            "recovery_ci_low": ci_r["ci_low"], "recovery_ci_high": ci_r["ci_high"],
            "fraction": o["recovery_fraction"], "fraction_ci_low": ci_f["ci_low"], "fraction_ci_high": ci_f["ci_high"],
            "flags": ",".join(o["flags"]), "comparable": rep.get("comparability", {}).get("ok"),
        })
    return out


def fmt(x, nd=1):
    return "n/a" if x is None else f"{x:.{nd}f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", default="outputs/kv_recovery/*_16k_*_r075_*")
    ap.add_argument("--out", default="outputs/kv_recovery/pilot_summary.md")
    args = ap.parse_args(argv)
    rows = []
    for d in sorted(glob.glob(args.glob)):
        rows += rows_for(Path(d))
    if not rows:
        print("no eval_results.json found")
        return 1
    lines = ["| run | benchmark | n | dense | compressed | recovered | drop | recovery [CI] | recovery fraction [CI] | flags |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["benchmark"], r["run"])):
        frac = f"{fmt(r['fraction'] * 100 if r['fraction'] is not None else None, 1)}%"
        frac_ci = (f" [{fmt(r['fraction_ci_low'] * 100, 0)}, {fmt(r['fraction_ci_high'] * 100, 0)}]"
                   if r["fraction_ci_low"] is not None else "")
        lines.append(f"| `{r['run']}` | {r['benchmark']} | {r['n']} | {fmt(r['dense'])} | {fmt(r['compressed'])} | "
                     f"{fmt(r['recovered'])} | {fmt(r['drop'])} | {fmt(r['recovery'], 1)} [{fmt(r['recovery_ci_low'])}, "
                     f"{fmt(r['recovery_ci_high'])}] | {frac}{frac_ci} | {r['flags']} |")
    md = "\n".join(lines) + "\n"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    with out.with_suffix(".csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(md)
    print(f"wrote {out} and {out.with_suffix('.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
