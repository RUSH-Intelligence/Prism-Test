#!/usr/bin/env python
"""Measure decode throughput, per-step latency, prefill wall and TTFT for KV compressors.

Batch=1 latency protocol.  One cell = (model, method, ratio, context length).
Every compressed cell is paired with a full-KV anchor at the same
(model, context, attn, dtype); without that denominator a speedup means nothing.

    python scripts/bench_kv_perf.py \
        --model meta-llama/Llama-3.1-8B-Instruct --model-key llama8b \
        --methods knorm,cur,keydiff,snapkv,streaming_llm --ratios 0.9 \
        --context-lengths 8192,16384,32768,65536,131072 \
        --attn-impl flash_attention_2 --dtype bfloat16 \
        --decode-steps 128 --repeats 5 --out-dir /scratch/sj157/kv_perf

Run under the interpreter that owns the GPU env:
    /scratch/sj157/prism_env/bin/python scripts/bench_kv_perf.py ...
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from eval_harness.profiling.audit import audit_cell  # noqa: E402
from eval_harness.profiling.cell import PerfCell, newest_perf, write_perf  # noqa: E402
from eval_harness.profiling.environment import capture_environment, gpu_processes  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--model-key", default=None, help="short key for paths (default: model tail)")
    p.add_argument("--label", default="")
    p.add_argument("--methods", default="knorm,cur,keydiff,snapkv,streaming_llm")
    p.add_argument("--ratios", default="0.9")
    p.add_argument("--context-lengths", default="8192,16384,32768,65536,131072")
    p.add_argument("--skip-anchor", action="store_true",
                   help="do NOT inject the full-KV cell (disables every speedup column)")
    p.add_argument("--decode-steps", type=int, default=128)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmup-repeats", type=int, default=2)
    p.add_argument("--attn-impl", default="flash_attention_2")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--trust-remote-code", action="store_true", default=True)
    p.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    p.add_argument("--dequantize-fp8", action="store_true")
    p.add_argument("--kv-kwargs", default="{}", help="JSON dict merged into kv_compressor_kwargs")
    p.add_argument("--method-variants", default=None,
                   help='JSON list (or @file.json) of extra cells that parameterise one method, '
                        'e.g. \'[{"label":"p6l80","method":"rarekv",'
                        '"kwargs":{"n_planes":6,"n_tables":80}}]\'. Each becomes its own cell '
                        'and its own result dir; --kv-kwargs is merged in underneath.')
    p.add_argument("--budget-rule", default="strict", choices=["strict", "ragged_mean"])
    p.add_argument("--prompt-seed", type=int, default=42)
    p.add_argument("--no-compression-stage", dest="measure_compression",
                   action="store_false", default=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--isolation", default="cell", choices=["cell", "none"],
                   help="'cell' = one subprocess per cell (default; total allocator/model "
                        "isolation). 'none' = all cells in this process.")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--tag", default="")
    p.add_argument("--allow-busy-gpu", action="store_true",
                   help="do not abort when another process holds the GPU")
    p.add_argument("--only-cell", default=None, help=argparse.SUPPRESS)   # internal
    return p.parse_args(argv)


def expand(args):
    """Grid -> cells, anchor first at each context so it is measured on a clean pool."""
    key = args.model_key or args.model.rstrip("/").split("/")[-1]
    methods = [m.strip() for m in args.methods.split(",") if m.strip() and m.strip() != "none"]
    ratios = [float(r) for r in args.ratios.split(",") if r.strip()]
    ctxs = [int(c) for c in args.context_lengths.split(",") if c.strip()]
    kwargs = json.loads(args.kv_kwargs) if args.kv_kwargs else {}

    variants = []
    if args.method_variants:
        spec = args.method_variants
        if spec.startswith("@"):
            spec = Path(spec[1:]).read_text()
        for v in json.loads(spec):
            if "method" not in v:
                raise SystemExit(f"--method-variants entry missing 'method': {v}")
            variants.append((v["method"], v.get("label", ""), dict(v.get("kwargs", {}))))

    cells = []
    for ctx in ctxs:
        common = dict(model_key=key, hf_model=args.model, context_tokens=ctx,
                      attn_impl=args.attn_impl, dtype=args.dtype,
                      decode_steps=args.decode_steps, repeats=args.repeats,
                      warmup_repeats=args.warmup_repeats, label=args.label or key)
        if not args.skip_anchor:
            cells.append(PerfCell(method="none", compression_ratio=0.0, **common))
        for m in methods:
            for r in ratios:
                cells.append(PerfCell(method=m, compression_ratio=r,
                                      kv_compressor_kwargs=dict(kwargs), **common))
        for m, vlabel, vkwargs in variants:
            for r in ratios:
                merged = dict(kwargs)
                merged.update(vkwargs)
                cells.append(PerfCell(method=m, compression_ratio=r, variant=vlabel,
                                      kv_compressor_kwargs=merged, **common))
    return cells


def run_one(args, cell) -> int:
    """Measure a single cell in THIS process and write perf.json."""
    import torch
    from eval_harness.profiling.runner import load_runtime, time_cell

    torch.manual_seed(42)
    run_dir = Path(args.out_dir) / cell.cell_id
    t0 = time.time()
    rt = load_runtime(cell.hf_model, dtype=cell.dtype, attn_impl=cell.attn_impl,
                      trust_remote_code=args.trust_remote_code,
                      max_model_len=args.max_model_len, dequantize_fp8=args.dequantize_fp8)
    load_s = time.time() - t0
    payload = time_cell(rt, cell, prompt_seed=args.prompt_seed,
                        measure_compression=args.measure_compression)
    payload.update(
        schema_version=1, artifact="kv_perf_cell",
        cell=cell.to_dict(),
        config={"model": cell.hf_model, "dtype": cell.dtype,
                "attn_implementation_requested": cell.attn_impl,
                "attn_implementation_actual": rt.attn_impl_actual,
                "trust_remote_code": args.trust_remote_code,
                "dequantize_fp8": args.dequantize_fp8,
                "max_model_len": args.max_model_len,
                "budget_rule": args.budget_rule, "prompt_seed": args.prompt_seed,
                "kv_compressor": cell.method, "compression_ratio": cell.compression_ratio,
                "kv_compressor_kwargs": cell.kv_compressor_kwargs,
                "positional_method": "none", "attention_method": "none",
                "prefill_chunk_size": None},
        environment=capture_environment(args.tag),
        timing={"model_load_s": load_s, "elapsed_wall_s": time.time() - t0, "status": "ok"},
    )
    payload["audit"] = audit_cell(cell, payload, budget_rule=args.budget_rule)
    path = write_perf(run_dir, payload)
    probs = payload["audit"]["problems"]
    st = payload["decode"]["per_step"]
    print(f"wrote {path}")
    print(f"  {cell.cell_id}: prefill {payload['prefill']['summary']['median']:.1f} ms | "
          f"TTFT {payload['ttft']['ttft_ms']['median']:.1f} ms | "
          f"step {st['median']:.2f} ms (p99 {st['p99']:.2f}) | "
          f"{payload['decode']['throughput_tok_s']:.1f} tok/s | "
          f"cache {payload['kv_cache']['seq_len_max']}")
    for w in payload["audit"]["warnings"]:
        print(f"  WARN: {w}")
    for pr in probs:
        print(f"  PROBLEM: {pr}", file=sys.stderr)
    return 1 if probs else 0


def main(argv=None) -> int:
    args = parse_args(argv)
    cells = expand(args)
    out = Path(args.out_dir)

    if args.only_cell:                       # internal: subprocess worker
        cell = next((c for c in cells if c.cell_id == args.only_cell), None)
        if cell is None:
            print(f"no such cell: {args.only_cell}", file=sys.stderr)
            return 2
        return run_one(args, cell)

    print(f"# {len(cells)} cells -> {out}")
    for i, c in enumerate(cells):
        print(f"  [{i:2d}] {c.cell_id}")
    if args.dry_run:
        return 0

    busy = gpu_processes()
    if busy and not args.allow_busy_gpu:
        print(f"FATAL: another process holds this GPU; timing would be invalid:\n{busy}",
              file=sys.stderr)
        return 3

    out.mkdir(parents=True, exist_ok=True)
    rc = 0
    for i, cell in enumerate(cells):
        cell_dir = out / cell.cell_id
        if args.resume and newest_perf(cell_dir):
            print(f"[{i}/{len(cells)}] skip (done): {cell.cell_id}")
            continue
        print(f"[{i}/{len(cells)}] {cell.cell_id}", flush=True)
        if args.isolation == "none":
            rc |= run_one(args, cell)
        else:
            # One process per cell: the caching allocator's pool, cuBLASLt
            # heuristics and the never-restored attn_module.rotary_emb graft
            # (kv_compression/base.py:474-475) all persist for a process lifetime.
            cmd = [sys.executable, str(Path(__file__).resolve())] + \
                  [a for a in (sys.argv[1:]) if a != "--dry-run"] + \
                  ["--only-cell", cell.cell_id, "--isolation", "none"]
            r = subprocess.run(cmd, cwd=str(REPO), env=os.environ.copy())
            rc |= (1 if r.returncode else 0)
    return rc


if __name__ == "__main__":
    sys.exit(main())
