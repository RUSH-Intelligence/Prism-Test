#!/usr/bin/env python
"""Aggregate KV-compression performance cells into tables.

    python scripts/kv_perf_report.py status --root /scratch/sj157/kv_perf
    python scripts/kv_perf_report.py report --root /scratch/sj157/kv_perf --out-dir /scratch/sj157/kv_perf/report

``status`` exits 0 only when every discovered cell has a perf.json, so
``until ... status; do sleep 300; done`` works.  ``report`` exits 1 on audit
problems unless --allow-problems.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from eval_harness.profiling.audit import (  # noqa: E402
    H200_PEAK_BW_GBPS, achieved_bandwidth_gbps, audit_cell, expected_budget, roofline_step_ms,
)
from eval_harness.profiling.cell import PerfCell as _PerfCell  # noqa: E402
from eval_harness.profiling.cell import PERF_FILENAME  # noqa: E402
from eval_harness.profiling.stats import speedup  # noqa: E402

DASH = "—"          # expected but absent
MARK_ATTN = "‡"     # attn deviates from the anchor's


def load_cells(root: Path):
    """Newest perf.json per cell dir (re-runs nest /1, /2, ...)."""
    by_dir = {}
    for p in sorted(root.rglob(PERF_FILENAME)):
        cell_dir = p.parent
        while cell_dir.name.isdigit():
            cell_dir = cell_dir.parent
        prev = by_dir.get(cell_dir)
        if prev is None or p.stat().st_mtime > prev.stat().st_mtime:
            by_dir[cell_dir] = p
    out = []
    for cell_dir, p in sorted(by_dir.items()):
        try:
            out.append((cell_dir, json.loads(p.read_text())))
        except Exception as exc:                       # noqa: BLE001
            print(f"WARNING: unreadable {p}: {exc}", file=sys.stderr)
    return out


def _g(d, *path, default=None):
    for k in path:
        if not isinstance(d, dict) or k not in d or d[k] is None:
            return default
        d = d[k]
    return d


def fmt(v, spec=".2f"):
    return DASH if v is None else format(v, spec)


def _reaudit(p):
    """Re-run the audit from the stored payload.

    The gate is re-evaluated at report time rather than trusted from the artifact,
    so tightening or relaxing a threshold never requires re-running a GPU sweep.
    """
    c = p.get("cell", {})
    try:
        cell = _PerfCell(
            model_key=c.get("model_key", ""), hf_model=c.get("hf_model", ""),
            method=c.get("method", "none"),
            compression_ratio=float(c.get("compression_ratio") or 0.0),
            context_tokens=int(c.get("context_tokens") or 0),
            attn_impl=c.get("attn_impl", ""), dtype=c.get("dtype", ""),
            decode_steps=int(c.get("decode_steps") or 0),
            repeats=int(c.get("repeats") or 0))
        rule = _g(p, "config", "budget_rule", default="strict")
        a = audit_cell(cell, p, budget_rule=rule)
        return {"problems": a["problems"], "warnings": a["warnings"]}
    except Exception as exc:                                   # noqa: BLE001
        return {"problems": [f"re-audit failed: {exc}"], "warnings": []}


def rows_from(payloads):
    """Flatten to per-cell rows, resolving each cell's anchor."""
    anchors = {}
    for _, p in payloads:
        c = p.get("cell", {})
        if c.get("is_anchor"):
            anchors[c["anchor_key"]] = p
    rows = []
    for cell_dir, p in payloads:
        c = p.get("cell", {})
        a = anchors.get(c.get("anchor_key"))
        step = _g(p, "decode", "per_step", "median")
        a_step = _g(a, "decode", "per_step", "median") if a else None
        pre = _g(p, "prefill", "summary", "median")
        a_pre = _g(a, "prefill", "summary", "median") if a else None
        kvb = _g(p, "kv_cache", "bytes_total")
        a_kvb = _g(a, "kv_cache", "bytes_total") if a else None
        ctx = _g(p, "prefill", "tokens", default=c.get("context_tokens"))
        s_mean = ((_g(p, "decode", "cache_len_start", default=0)
                   + _g(p, "decode", "cache_len_end", default=0)) / 2)
        w = _g(p, "memory", "weights_bytes", default=0)
        bpt = _g(p, "kv_cache", "bytes_per_token", default=0) * _g(p, "kv_cache", "layers_with_kv", default=0)
        bw = achieved_bandwidth_gbps(w, bpt, s_mean, step) if (step and bpt) else None
        rl = roofline_step_ms(w, bpt, s_mean) if bpt else None            # this harness
        rl_ideal = roofline_step_ms(w, bpt, s_mean, dyncache=False) if bpt else None
        rows.append({
            "run_dir": str(cell_dir),
            "model_key": c.get("model_key"), "label": c.get("label"),
            "hf_model": c.get("hf_model"), "context_tokens": ctx,
            "method": c.get("method"), "ratio": c.get("compression_ratio"),
            "is_anchor": c.get("is_anchor"), "attn_impl": c.get("attn_impl"),
            "dtype": c.get("dtype"), "anchor_key": c.get("anchor_key"),
            "prefill_ms_median": pre, "prefill_ms_p95": _g(p, "prefill", "summary", "p95"),
            "prefill_tok_s": _g(p, "prefill", "throughput_tok_s"),
            "prefill_host_gap_ms": _g(p, "prefill", "host_gap_ms"),
            "ttft_ms_median": _g(p, "ttft", "ttft_ms", "median"),
            "question_block_ms_median": _g(p, "ttft", "question_block_ms", "median"),
            "step_ms_median": step, "step_ms_p90": _g(p, "decode", "per_step", "p90"),
            "step_ms_p99": _g(p, "decode", "per_step", "p99"),
            "step_ms_cv": _g(p, "decode", "per_step", "cv"),
            "decode_tok_s": _g(p, "decode", "throughput_tok_s"),
            "decode_tok_s_median_based": _g(p, "decode", "throughput_tok_s_median_based"),
            "repeat_spread": _g(p, "decode", "repeat_spread"),
            "kv_seq_post": _g(p, "kv_cache", "seq_len_max"),
            "kv_seq_expected": (ctx if c.get("is_anchor")
                                else expected_budget(ctx or 0, c.get("compression_ratio") or 0)),
            "kv_bytes_post": kvb, "kv_ragged": _g(p, "kv_cache", "ragged"),
            "peak_alloc_decode_bytes": _g(p, "memory", "peak_alloc_decode_bytes"),
            "peak_reserved_bytes": _g(p, "memory", "peak_reserved_bytes"),
            "decode_speedup": speedup(a_step, step),
            "prefill_overhead_pct": ((pre / a_pre - 1) * 100 if (pre and a_pre) else None),
            "kv_bytes_reduction_x": (a_kvb / kvb if (a_kvb and kvb) else None),
            "achieved_bw_GBps": bw,
            "bw_util_pct": (100 * bw / H200_PEAK_BW_GBPS if bw else None),
            "roofline_step_ms": rl, "roofline_step_ms_ideal": rl_ideal,
            "compress_ms_total": _g(p, "compression_stage", "total_ms"),
            "compress_frac_of_prefill_pct": _g(p, "compression_stage", "frac_of_prefill_pct"),
            "eos_disabled": _g(p, "decode", "eos_disabled"),
            "repeats": c.get("repeats"), "decode_steps": c.get("decode_steps"),
            "gpu_name": _g(p, "environment", "gpu_name"),
            "git_sha": (_g(p, "environment", "git_sha") or "")[:8],
            "status": _g(p, "timing", "status"),
            **_reaudit(p),
            "has_anchor": a is not None,
        })
    return rows


def grid(rows, value_key, spec=".2f", model=None):
    """method x context markdown grid, anchor row first."""
    rs = [r for r in rows if model is None or r["model_key"] == model]
    ctxs = sorted({r["context_tokens"] for r in rs})
    methods, seen = [], set()
    for r in sorted(rs, key=lambda r: (not r["is_anchor"], r["method"], r["ratio"])):
        name = "full KV" if r["is_anchor"] else f'{r["method"]} r{r["ratio"]:g}'
        if name not in seen:
            seen.add(name)
            methods.append(name)
    out = ["| method | " + " | ".join(f"{c//1024}K" for c in ctxs) + " |",
           "|---" * (len(ctxs) + 1) + "|"]
    for name in methods:
        cells = []
        for c in ctxs:
            m = [r for r in rs if r["context_tokens"] == c
                 and (("full KV" if r["is_anchor"] else f'{r["method"]} r{r["ratio"]:g}') == name)]
            cells.append(fmt(m[0][value_key], spec) if m else DASH)
        bold = "**" if name == "full KV" else ""
        out.append(f"| {bold}{name}{bold} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def cmd_status(args) -> int:
    rows = rows_from(load_cells(Path(args.root)))
    if not rows:
        print(f"no cells under {args.root}")
        return 1
    bad = [r for r in rows if r["problems"]]
    print(f"{len(rows)} cells; {len(bad)} with audit problems")
    for r in sorted(rows, key=lambda r: (r["model_key"] or "", r["context_tokens"] or 0, r["method"])):
        flag = "FAIL" if r["problems"] else ("warn" if r["warnings"] else "ok")
        print(f"  {r['model_key']:>10} ctx{r['context_tokens']:<7} {r['method']:>14} "
              f"r{r['ratio']:<4g} {flag:>4}  {fmt(r['step_ms_median'])} ms/step  "
              f"{fmt(r['decode_tok_s'],'.1f')} tok/s")
    return 1 if bad else 0


def cmd_report(args) -> int:
    payloads = load_cells(Path(args.root))
    rows = rows_from(payloads)
    if not rows:
        print(f"no cells under {args.root}", file=sys.stderr)
        return 1
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = next((p.get("environment", {}) for _, p in payloads), {})
    models = sorted({r["model_key"] for r in rows})

    md = ["# KV-compression performance", "",
          "## Environment", "",
          f"- GPU: **{env.get('gpu_name')}** (driver {env.get('cuda_driver_version')})",
          f"- torch {env.get('torch_version')} / transformers {env.get('transformers_version')} "
          f"/ flash-attn {env.get('flash_attn_version')}",
          f"- git `{(env.get('git_sha') or '')[:8]}`" + (" **(dirty worktree)**" if env.get("git_dirty") else ""),
          "", "## Protocol", "",
          "- Backend `research` (HF eager loop), **batch = 1**, greedy, bf16, single GPU.",
          "- EOS disabled during measurement, so every cell contributes an identical",
          "  number of samples to the tok/s denominator. Generated text is not scored.",
          "- `decode_tok_s` = total tokens / total time (= 1000/mean step). The",
          "  median-based figure is reported separately; it hides the tail.",
          "- Per-head budget is `int(T*(1-r))`; the audit gate fails any cell that misses it.", ""]

    for m in models:
        rs = [r for r in rows if r["model_key"] == m]
        label = rs[0]["label"] or m
        md += [f"## {label}", "",
               "### Decode throughput (tok/s)", "", grid(rs, "decode_tok_s", ".1f"), "",
               "### Per-step decode latency (ms, median)", "", grid(rs, "step_ms_median", ".2f"), "",
               "### Per-step decode latency (ms, p99)", "", grid(rs, "step_ms_p99", ".2f"), "",
               "### Decode speedup vs full KV", "", grid(rs, "decode_speedup", ".2f"), "",
               "### Prefill wall (ms, median)", "", grid(rs, "prefill_ms_median", ".1f"), "",
               "### Prefill overhead vs full KV (%)", "", grid(rs, "prefill_overhead_pct", "+.1f"), "",
               "### Time to first token (ms, median)", "", grid(rs, "ttft_ms_median", ".1f"), "",
               "### KV cache after compression (MiB)", "",
               grid([{**r, "_kv_mib": (r["kv_bytes_post"] / 2**20 if r["kv_bytes_post"] else None)}
                     for r in rs], "_kv_mib", ".1f"), "",
               "### Peak allocated during decode (GiB)", "",
               grid([{**r, "_pk": (r["peak_alloc_decode_bytes"] / 2**30
                                   if r["peak_alloc_decode_bytes"] else None)} for r in rs], "_pk", ".2f"), "",
               "### Achieved memory bandwidth (% of peak)", "", grid(rs, "bw_util_pct", ".1f"), "",
               "### Roofline check — measured vs HBM-bandwidth floor (ms/step)", "",
               "Floor for THIS harness (DynamicCache re-cats every step, ~3x KV traffic):", "",
               grid(rs, "roofline_step_ms", ".2f"), "",
               "Floor for an ideal pre-allocated/paged engine:", "",
               grid(rs, "roofline_step_ms_ideal", ".2f"), "",
               "A measured value far above both floors means the loop is **launch-bound**",
               "(Python + ~350 kernel launches per decode step), not bandwidth-bound, so the",
               "KV savings cannot show up as latency. Compare against the bandwidth table above.", ""]
        comp = [r for r in rs if r["compress_ms_total"] is not None]
        if comp:
            md += ["### Compression (scoring) stage time (ms) — figure F6", "",
                   grid(comp, "compress_ms_total", ".2f"), ""]

    md += ["## Cell inventory", "",
           "| cell | steps | repeats | cache | expected | jitter p99/med | audit |",
           "|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["model_key"] or "", r["context_tokens"] or 0, r["method"])):
        j = (r["step_ms_p99"] / r["step_ms_median"]) if (r["step_ms_p99"] and r["step_ms_median"]) else None
        verdict = "FAIL" if r["problems"] else ("warn" if r["warnings"] else "ok")
        md.append(f"| {r['model_key']}/ctx{r['context_tokens']}/{r['method']} r{r['ratio']:g} | "
                  f"{r['decode_steps']} | {r['repeats']} | {r['kv_seq_post']} | "
                  f"{r['kv_seq_expected']} | {fmt(j)} | {verdict} |")

    probs = [(r, p) for r in rows for p in r["problems"]]
    warns = [(r, w) for r in rows for w in r["warnings"]]
    md += ["", "## Warnings", ""]
    if not probs and not warns:
        md.append("None — every cell present, budget-audited and internally consistent.")
    for r, p in probs:
        md.append(f"- **PROBLEM** `{r['model_key']}/ctx{r['context_tokens']}/{r['method']}`: {p}")
    for r, w in warns:
        md.append(f"- warn `{r['model_key']}/ctx{r['context_tokens']}/{r['method']}`: {w}")

    md += ["", "## How to read these numbers", "",
           "- **Batch = 1 means weights dominate.** This model reads ~15 GB of weights per",
           "  decode step versus ~1 GB of KV at 8K, so a small speedup at short context is",
           "  correct physics, not a broken measurement. Compression's benefit grows with",
           "  context and with batch size; these are the pessimistic end of that curve.",
           "- **Prefill gets slower, not faster.** The compressor hook fires per layer",
           "  *after* that layer's full-length attention has already run, so eviction cannot",
           "  reduce any earlier work. TTFT therefore worsens; the win is in decode and memory.",
           "- **This harness re-copies the KV cache every decode step.** transformers'",
           "  `DynamicCache` grows by `torch.cat` (cache_utils.py:143-144) with no",
           "  pre-allocation, so KV traffic is ~3x the naive model and compression looks",
           "  *better* here than it would in a paged engine (vLLM/TRT-LLM). Do not",
           "  extrapolate these ratios to such an engine.",
           "- Top-k eviction leaves the cache in score order, not positional order, so these",
           "  runs are not comparable to prefix-caching or paged-block setups.", ""]

    md_path = out_dir / "kv_perf.md"
    md_path.write_text("\n".join(md) + "\n")
    print(f"wrote {md_path}")

    csv_path = out_dir / "kv_perf_cells.csv"
    from eval_harness.profiling.cell import CSV_COLUMNS
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(CSV_COLUMNS)
        for r in rows:
            w.writerow(["" if r.get(c) is None else
                        (f"{r[c]:.4f}" if isinstance(r.get(c), float) else
                         (str(r[c]).lower() if isinstance(r.get(c), bool) else r.get(c)))
                        for c in CSV_COLUMNS[:-1]] + [len(r["problems"])])
    print(f"wrote {csv_path}")
    return 1 if (probs and not args.allow_problems) else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="/scratch/sj157/kv_perf")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("status"); s.set_defaults(func=cmd_status)
    r = sub.add_parser("report")
    r.add_argument("--out-dir", default=None)
    r.add_argument("--allow-problems", action="store_true")
    r.set_defaults(func=cmd_report)
    args = p.parse_args(argv)
    if args.cmd == "report" and args.out_dir is None:
        args.out_dir = str(Path(args.root) / "report")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
