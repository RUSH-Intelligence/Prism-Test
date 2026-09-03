#!/usr/bin/env python3
"""How fast -- and how exact -- are the RareKV bucket-packing kernels?

Three paths produce the same [BH, L, T] bucket-id tensor:

  torch   the reference: cuBLAS GEMM, then `sign -> int32 -> *powers -> sum ->
          permute` in ATen. Five passes over a [B*H*T, L*P] intermediate.
  pack    TIER 1: the SAME cuBLAS GEMM, then ONE kernel that reads `proj` once
          and writes uint8/int16 ids straight into the L-major layout.
          Bit-identical by construction -- `pack_exact` below asserts it.
  fused   TIER 2: the GEMM on tensor cores inside the packing kernel, so `proj`
          never materialises. NOT bit-identical to cuBLAS; bit-identical to a
          declared reference (see csrc/rarekv_fused.cu). This script MEASURES the
          divergence rather than claiming it away.

What each column settles
------------------------
  t_gemm_ms          `k2 @ planes` ALONE. This is ~40% of the Tier-1 budget and
                     the single largest uncertainty in the traffic model.
                     Measure it first; believe nothing until it lands.
  t_pack_torch_ms    the four ATen ops + the transposing clone
  t_pack_cuda_ms     the Tier-1 kernel
  t_fused_ms         the Tier-2 kernel (GEMM included)
  t_collide_ms       the collision kernel on the L-major ids
  t_score_*_ms       the headline: the whole RareKVSketch.score()
  t_keydiff_ms       the baseline rarekv is judged against
  peak_*_mib         ABSOLUTE allocator highwater during the arm
  transient_*_mib    peak minus the allocator baseline at the start of the arm --
                     this is the column to compare against the 5*M /
                     (2+b/P)*M / (b/P)*M bytes-per-element predictions. The
                     absolute peak carries ~512 MiB of resident keys+values at
                     T=131072 and a per-arm baseline that differs between arms,
                     so it cannot check them.
  pack_exact         torch.equal(pack, reference)                    -- MUST be True
  fused_exact_intops GATE A at this cell: with operands drawn from {-1,0,+1}
                     every product is an exact small integer and every partial
                     sum is <= D, so EVERY accumulation order agrees bitwise.
                     A False here is an indexing bug, not a float difference.
  fused_flip_rate    fraction of the B*H*T*L*P projection signs that differ
                     between `fused` and `cublas` on real-valued keys
  fused_retained_dl  |retained_fused symmetric-difference retained_cublas| / budget
  gemm_chunk_exact   torch.equal(mm_chunked, mm_full) -- gates Tier 1c's default
  redprec_noop       torch.equal(proj_flag_on, proj_flag_off) ON THIS CELL'S OWN
                     operands -- decides whether
                     allow_bf16_reduced_precision_reduction can be pinned False.
                     Per cell, not once globally: split-k selection is a function
                     of M, N, K, and M spans 64K..2M across the sweep.
  gbps_*             achieved bandwidth per stage, so a missed prediction is
                     attributable to a stage rather than to "the kernel was slow".
                     The pack arms time GEMM+kernel, so `t_pack_kernel_ms` (=
                     t_pack_cuda_ms - t_gemm_ms) and `gbps_pack_kernel` isolate the
                     stage the model's 3.6 TB/s constant actually predicts.
                     `tflops_fused` rather than gbps_fused is the meaningful one
                     for Tier 2: it is compute/SMEM bound, and its HBM traffic is
                     only N*D*2 + N*L*b.

Calibration for the divergence columns, from the class docstring's own
yardsticks: a bf16-vs-fp32 GEMM flips ~5e-4 of signs and moves ~2% of the
retained set; changing `seed` moves ~56%. A Tier-2 claim without this table is
not a claim.

Run on one H200:  python scripts/bench_rarekv_kernel.py
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval_harness.kernels import rarekv_lsh                              # noqa: E402
from eval_harness.kernels.rarekv_lsh import (                            # noqa: E402
    bucket_dtype, buckets_from_proj_torch, collision_sums_lmajor, lsh_buckets,
    planes_to_planes_t,
)
from eval_harness.kv_compression.compressors.rarekv_sketch import RareKVSketch   # noqa: E402
from eval_harness.kv_compression.compressors.keydiff_sketch import KeyDiffSketch  # noqa: E402
from eval_harness.profiling.environment import capture_environment       # noqa: E402


class FakeMod(torch.nn.Module):
    layer_idx = 0


def timeit(fn, warmup=1, iters=3):
    """(median ms, peak MiB, transient MiB) or (None, "OOM", None).

    `max_memory_allocated` is ABSOLUTE: it includes everything resident at the
    time, which during the score arms is keys+values (512 MiB at T=131072, B=1,
    H=8, D=128, bf16) plus whatever else the cell is holding -- and the arms do
    not all hold the same things. The per-element transient predictions
    (5*M / (2+b/P)*M / (b/P)*M) can only be checked against the DELTA, so record
    the baseline immediately before the reset and report both.
    """
    try:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        peak = torch.cuda.max_memory_allocated()
        times.sort()
        return (times[len(times) // 2], peak / 2 ** 20, (peak - base) / 2 ** 20)
    except torch.cuda.OutOfMemoryError:
        return None, "OOM", None
    finally:
        gc.collect()
        torch.cuda.empty_cache()


def _arm(out, name, fn):
    """Time one arm and record its ms / absolute peak / transient under `name`."""
    ms, peak, trans = timeit(fn)
    out[f"t_{name}_ms"] = ms
    out[f"peak_{name}_mib"] = peak
    out[f"transient_{name}_mib"] = trans
    return ms


def gbps(nbytes, ms):
    return None if not ms else nbytes / (ms * 1e-3) / 1e9


def retained(scores, ratio):
    """The base class's selection: top int(T*(1-ratio)) per (b, h)."""
    T = scores.shape[-1]
    k = max(int(T * (1.0 - ratio)), 1)
    return torch.topk(scores, k, dim=-1).indices


def _int_keys(shape, seed, dtype, device):
    """Gate-A operands: {-1, 0, +1}, exactly representable, partial sums <= D."""
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randint(-1, 2, shape, generator=g, device=device).to(dtype)


def redprec_equal(k2, planes):
    """Is `allow_bf16_reduced_precision_reduction` a measured no-op on THESE operands?

    Run per cell, on the cell's own shapes: cuBLAS's split-k decision is a
    function of M, N and K, and M spans 65,536 to 2,097,152 across this sweep, so
    a single probe on one small shape cannot license pinning the flag. If the flag
    is a no-op everywhere it ships, it can be pinned False in a follow-up commit
    documented as a measured no-op; flipping it BEFORE measuring would silently
    move every already-published rarekv number.
    """
    flag = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    try:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        on = (k2 @ planes).clone()
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        off = k2 @ planes
        eq = bool(torch.equal(on, off))
        flips = 0.0 if eq else float(((on > 0) != (off > 0)).float().mean())
        return {"redprec_noop": eq, "redprec_sign_flip_rate": flips}
    except torch.cuda.OutOfMemoryError:
        return {"redprec_noop": None, "redprec_sign_flip_rate": None}
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = flag
        gc.collect(); torch.cuda.empty_cache()


def run_cell(T, P, L, B=1, H=8, D=128, device="cuda", dtype=torch.bfloat16,
             block_m=0, ratio=0.9, chunk_rows=(1 << 15, 1 << 17, 1 << 18)):
    R = 1 << P
    BH, N = B * H, B * H * T
    M = N * L * P                          # elements of the projection
    b = torch.tensor([], dtype=bucket_dtype(P)).element_size()
    # The fused kernel instantiates BM = 128 only, and refuses anything else with a
    # TORCH_CHECK. Forwarding --block-m 256 to it killed the whole sweep on the
    # first cell; the BM autotune arm applies to `pack`, so record the two
    # separately rather than losing the run.
    fused_bm = block_m if block_m in (0, 128) else 0
    out = {"T": T, "P": P, "L": L, "R": R, "N": N, "D": D, "block_m": block_m,
           "fused_block_m": fused_bm,
           "dtype": str(dtype), "bucket_bytes": b,
           "proj_gib": M * 2 / 2 ** 30, "bucket_gib": N * L * b / 2 ** 30,
           "fused_extension_built": rarekv_lsh._FUSED_EXT is not None}

    keys = torch.randn(B, H, T, D, device=device, dtype=dtype)
    values = torch.randn(B, H, T, D, device=device, dtype=dtype)
    mod = FakeMod()
    sk = RareKVSketch(compression_ratio=ratio, n_planes=P, n_tables=L, seed=42)
    planes = sk._planes(mod, D, torch.device(device), dtype)
    planes_t = planes_to_planes_t(planes, L, P)
    k2 = keys.reshape(-1, D)

    # ---- 0. is the unpinned reduced-precision flag a no-op on THIS shape? ----
    out.update(redprec_equal(k2, planes))

    # ---- 1. the GEMM alone: the assumed 2.8 TB/s, and 40% of the Tier-1 budget
    t_gemm = _arm(out, "gemm", lambda: k2 @ planes)
    out["gbps_gemm"] = gbps(N * D * 2 + D * L * P * 2 + M * 2, t_gemm)

    # ---- 2. bucket ids, three ways -----------------------------------------
    proj = k2 @ planes
    ref = buckets_from_proj_torch(proj.clone(), BH, T, L, P)
    del proj
    torch.cuda.empty_cache()

    def _torch_path():
        return buckets_from_proj_torch(k2 @ planes, BH, T, L, P)

    t_pt = _arm(out, "pack_torch", _torch_path)
    # 22 B per projection element: proj write 2 + read 2, bool write 1 + read 1,
    # int32 cast write 4, mul_ read 4 + write 4, sum read 4. Then 16 + b bytes per
    # (row, table): the int32 sum's write 4, the permute clone's read 4 + write 4,
    # and the dtype cast's read 4 + write b. Plus the GEMM's read of K.
    out["gbps_pack_torch"] = gbps(N * D * 2 + M * 22 + N * L * (16 + b), t_pt)

    got_pack, path = lsh_buckets(keys, planes, None, L, P, mode="pack", block_m=block_m)
    out["pack_path"] = path
    out["pack_exact"] = bool(torch.equal(got_pack, ref))
    del got_pack
    t_pc = _arm(out, "pack_cuda",
                lambda: lsh_buckets(keys, planes, None, L, P, mode="pack",
                                    block_m=block_m))
    # This arm times the GEMM (reads K, writes proj) AND the kernel (reads proj,
    # writes the ids), so the projection is touched FOUR bytes' worth per element,
    # not two. Counting it once under-reported achieved bandwidth by ~42%.
    out["gbps_pack_cuda"] = gbps(N * D * 2 + M * 4 + N * L * b, t_pc)
    # The kernel's own stage, isolated: this is what the model's 3.6 TB/s
    # streaming-pack constant predicts, and the only number that can falsify it.
    if t_pc and t_gemm and t_pc > t_gemm:
        out["t_pack_kernel_ms"] = t_pc - t_gemm
        out["gbps_pack_kernel"] = gbps(M * 2 + N * L * b, t_pc - t_gemm)
    else:
        out["t_pack_kernel_ms"] = out["gbps_pack_kernel"] = None

    # Tier 1c: chunk the GEMM along M. Gates whether the default can flip on.
    out["gemm_chunk_exact"] = {}
    for cr in chunk_rows:
        if cr >= N:
            continue
        try:
            ch, _ = lsh_buckets(keys, planes, None, L, P, mode="pack",
                                block_m=block_m, gemm_chunk_rows=cr)
            out["gemm_chunk_exact"][str(cr)] = bool(torch.equal(ch, ref))
            del ch
        except torch.cuda.OutOfMemoryError:
            out["gemm_chunk_exact"][str(cr)] = None
        gc.collect(); torch.cuda.empty_cache()

    # Tier 2. `fused` silently degrades to `pack` whenever a precondition fails --
    # including a failed Tier-2 nvcc build. Recording pack timings under a fused
    # key would publish a PERFECT divergence table (flip rate 0.0, retained delta
    # 0.0) for a path that never ran, which is the exact failure the dispatcher's
    # returned path exists to prevent. So: check the resolved path, and null every
    # fused column if it is not "fused".
    fused_ok = (P <= rarekv_lsh.KERNEL_MAX_PLANES and D in rarekv_lsh.FUSED_HEAD_DIMS
                and N >= rarekv_lsh.FUSED_MIN_ROWS)
    fused_cols = ("t_fused_ms", "peak_fused_mib", "transient_fused_mib", "gbps_fused",
                  "tflops_fused", "fused_equals_cublas", "fused_exact_intops",
                  "fused_flip_rate", "fused_bucket_diff_rate", "fused_retained_overlap",
                  "fused_retained_dl")
    if fused_ok:
        got_fused, fpath = lsh_buckets(keys, planes, planes_t, L, P, mode="fused",
                                       block_m=fused_bm)
        out["fused_path"] = fpath
        if fpath != "fused":
            fused_ok = False
            out["fused_skipped"] = f"dispatcher fell back to {fpath!r}"
        else:
            out["fused_equals_cublas"] = bool(torch.equal(got_fused, ref))
        del got_fused
    else:
        out["fused_path"] = None
        out["fused_skipped"] = "preconditions (P / head_dim / rows) not met"
    if not fused_ok:
        for c in fused_cols:
            out.setdefault(c, None)

    if fused_ok:
        t_f = _arm(out, "fused",
                   lambda: lsh_buckets(keys, planes, planes_t, L, P, mode="fused",
                                       block_m=fused_bm))
        # HBM traffic only; the fused kernel is compute/shared-memory bound, so an
        # achieved-GB/s figure reads as ~12% of roofline and misleads. The tensor
        # core number is the one to compare against the ~54% SMEM-bound ceiling
        # documented in csrc/rarekv_fused.cu.
        out["gbps_fused"] = gbps(N * D * 2 + N * L * b, t_f)
        out["tflops_fused"] = None if not t_f else 2.0 * N * D * L * P / (t_f * 1e-3) / 1e12

    # ---- 3. GATE A: integer operands, every order agrees bitwise ------------
    if fused_ok:
        ik = _int_keys((B, H, min(T, 8192), D), 67, dtype, device)
        ip = _int_keys((D, L * P), 71, dtype, device)
        ipt = planes_to_planes_t(ip, L, P)
        want = buckets_from_proj_torch(ik.reshape(-1, D).float() @ ip.float(),
                                       BH, ik.shape[2], L, P)
        gate, gpath = lsh_buckets(ik, ip, ipt, L, P, mode="fused", block_m=fused_bm)
        out["fused_exact_intops"] = bool(torch.equal(gate, want)) if gpath == "fused" else None
        del ik, ip, ipt, want, gate
        gc.collect(); torch.cuda.empty_cache()

    # ---- 4. divergence on REAL keys ----------------------------------------
    # Only the SIGN matters, so a cuBLAS-vs-R difference moves a bucket id only
    # when a projection sits within an ulp of zero.
    if fused_ok:
        f_b, _ = lsh_buckets(keys, planes, planes_t, L, P, mode="fused", block_m=fused_bm)
        diff_bits = 0
        # An id differs iff at least one of its P signs did; count sign flips by
        # XOR-ing the ids and popcounting, which is exact and needs no second GEMM.
        x = (f_b.to(torch.int32) ^ ref.to(torch.int32))
        for bit in range(P):
            diff_bits += int(((x >> bit) & 1).sum())
        out["fused_flip_rate"] = diff_bits / float(N * L * P)
        out["fused_bucket_diff_rate"] = float((x != 0).float().mean())
        del x

        csum_f = collision_sums_lmajor(f_b, R)
        csum_c = collision_sums_lmajor(ref, R)
        del f_b
        vn = torch.linalg.vector_norm(values, dim=-1).float()
        def _scores(csum):
            d = (csum.float() / L - 1.0) / float(max(T - 1, 1))
            return (sk.eps + d).pow(-sk.alpha).view(B, H, T) * vn
        r_f = retained(_scores(csum_f), ratio)
        r_c = retained(_scores(csum_c), ratio)
        k = r_f.shape[-1]
        inter = 0
        for bi in range(B):
            for hi in range(H):
                a = set(r_f[bi, hi].tolist()); c = set(r_c[bi, hi].tolist())
                inter += len(a & c)
        out["fused_retained_overlap"] = inter / float(B * H * k)
        out["fused_retained_dl"] = 2.0 * (1.0 - inter / float(B * H * k))
        del csum_f, csum_c, r_f, r_c, vn
        gc.collect(); torch.cuda.empty_cache()

    # ---- 5. collision kernel on the L-major ids ----------------------------
    t_col = _arm(out, "collide", lambda: collision_sums_lmajor(ref, R))
    out["gbps_collide"] = gbps(N * L * b * 2, t_col)
    del ref
    gc.collect(); torch.cuda.empty_cache()

    # ---- 6. the headline: whole-score timings ------------------------------
    for mode in ("torch", "pack", "fused"):
        if mode == "fused" and not fused_ok:
            out["t_score_fused_ms"] = out["peak_score_fused_mib"] = None
            out["transient_score_fused_mib"] = None
            continue
        s = RareKVSketch(compression_ratio=ratio, n_planes=P, n_tables=L, seed=42,
                         lsh_mode=mode, block_m=(fused_bm if mode == "fused" else block_m))
        _arm(out, f"score_{mode}", lambda s=s: s.score(mod, None, keys, values, None, {}))
        out[f"score_{mode}_path"] = list(s.lsh_paths)
    kd = KeyDiffSketch(compression_ratio=ratio)
    _arm(out, "keydiff", lambda: kd.score(mod, None, keys, values, None, {}))

    for a, bkey in (("t_score_pack_ms", "pack"), ("t_score_fused_ms", "fused")):
        if out.get(a) and out.get("t_score_torch_ms"):
            out[f"speedup_{bkey}_vs_torch"] = out["t_score_torch_ms"] / out[a]
        if out.get(a) and out.get("t_keydiff_ms"):
            out[f"speedup_{bkey}_vs_keydiff"] = out["t_keydiff_ms"] / out[a]

    del keys, values, k2
    gc.collect(); torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/scratch/sj157/kv_perf/rarekv_kernel_bench.json")
    ap.add_argument("--T", type=int, nargs="+", default=[32768, 65536, 131072, 262144])
    ap.add_argument("--L", type=int, nargs="+", default=[70, 100])
    ap.add_argument("--P", type=int, nargs="+", default=[5, 6, 7, 8, 9])
    ap.add_argument("--block-m", type=int, nargs="+", default=[0])
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--ratio", type=float, default=0.9)
    ap.add_argument("--reference-cells", action="store_true",
                    help="also run the historical (P,L) cells at T=131072 so the new "
                         "numbers are directly comparable to rarekv_dense_bench.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("this benchmark needs a CUDA device")
    ext = rarekv_lsh._kernel_ext()
    env = capture_environment(tag="rarekv_kernel_bench")
    env["rarekv_extension_built"] = ext is not None
    # Tier 2 lives in a second, lazily built extension; force it here so its
    # (long) nvcc time lands OUTSIDE every timed cell.
    env["rarekv_fused_extension_built"] = rarekv_lsh._fused_ext() is not None
    env["rarekv_cuda_cflags"] = list(rarekv_lsh._CUDA_CFLAGS)
    env["backend_flags_redprec"] = {
        "matmul_allow_bf16_reduced_precision_reduction":
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "matmul_allow_fp16_reduced_precision_reduction":
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
    }
    print(json.dumps(env, indent=2, default=str))
    if ext is None:
        print("WARNING: the CUDA extension did not build; only the torch path is real")

    cells = [(T, P, L, bm) for T in args.T for L in args.L for P in args.P
             for bm in args.block_m]
    if args.reference_cells:
        cells += [(131072, P, L, 0) for P, L in ((2, 40), (3, 50), (8, 50), (6, 80), (10, 60))]

    rows = []
    for T, P, L, bm in cells:
        try:
            row = run_cell(T, P, L, H=args.heads, D=args.head_dim, block_m=bm,
                           ratio=args.ratio)
        except torch.cuda.OutOfMemoryError:
            row = {"T": T, "P": P, "L": L, "block_m": bm, "error": "OOM"}
            gc.collect(); torch.cuda.empty_cache()
        except (RuntimeError, ValueError) as exc:    # TORCH_CHECK, max_bucket_slots, ...
            # One bad cell must degrade to an error ROW, never take the sweep --
            # and with it the warm extension cache -- down with it.
            row = {"T": T, "P": P, "L": L, "block_m": bm, "error": str(exc)}
            gc.collect(); torch.cuda.empty_cache()
        rows.append(row)
        print(json.dumps(row, default=str), flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"environment": env, "cells": rows},
                                         indent=2, default=str))
    print(f"wrote {args.out}")

    # An OOM at the top of the sweep is an expected, tolerated outcome and stays a
    # zero exit; anything else (a TORCH_CHECK, a bad config) is a real failure.
    bad = [r for r in rows if r.get("pack_exact") is False
           or r.get("fused_exact_intops") is False
           or (r.get("error") and r.get("error") != "OOM")]
    if bad:
        print(f"EXACTNESS/RUN FAILURES in {len(bad)} cells -- do not believe any timing above")
        return 1
    # A silent Tier-2 fallback is not an exactness failure, but it makes every
    # fused column meaningless, so it must not exit 0 unnoticed either.
    fell_back = [r for r in rows if r.get("fused_skipped", "").startswith("dispatcher")]
    if fell_back:
        print(f"TIER 2 NEVER RAN in {len(fell_back)} cells (dispatcher fell back); "
              f"their fused columns are null by construction")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
