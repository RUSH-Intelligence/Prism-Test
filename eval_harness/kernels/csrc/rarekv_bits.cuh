// RareKV bit-layout contract, shared by the Tier-1 pack kernel and the Tier-2
// fused kernel. This header holds the ONLY copy of the sign predicate, the
// shared-memory sign bitmap layout, and the bucket-id extractor, so all of the
// fiddly indexing is written once and validated once.
//
// Nothing here is torch-aware on purpose: it compiles into any .cu.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

// ------------------------------------------------------------- the predicate
// EXACTNESS: `> 0.0f` on a widening convert. bf16/fp16 -> f32 is exact, so this
// agrees with torch's `proj > 0` on every float class: +0.0 false, -0.0 false,
// NaN false (IEEE unordered), +inf true, -inf false, positive subnormal true.
//
// Do NOT "optimise" this into a sign-bit test: `(u & 0x8000) == 0` disagrees
// with torch on NaN and on -0.0. Do NOT compile this file with
// `--use_fast_math`: it implies `-ftz=true`, which flushes a subnormal
// projection to zero and flips its sign relative to `proj > 0`.
__device__ __forceinline__ float rk_f(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float rk_f(__half v)        { return __half2float(v); }
__device__ __forceinline__ float rk_f(float v)         { return v; }

__device__ __forceinline__ bool rk_pos(__nv_bfloat16 v) { return __bfloat162float(v) > 0.0f; }
__device__ __forceinline__ bool rk_pos(__half v)        { return __half2float(v) > 0.0f; }
__device__ __forceinline__ bool rk_pos(float v)         { return v > 0.0f; }

// ---------------------------------------------------------------- the bitmap
// A tile's signs live in shared memory as a per-row BITSTREAM: bit j of row r is
// the sign of projection column j of that row's column range, 8 bits per byte,
// LSB first -- which is exactly `sum_p 2^p * sign_p` when the P bits of one
// table are consecutive. The row stride is `nbs` bytes.
//
// `nbs` is chosen == 4 (mod 32) so that phase 2's read of `sbits[r*nbs + by]` is
// bank-conflict free across 32 consecutive r: the word address is
// r*(nbs/4) + by/4 and nbs/4 is odd, hence a bijection on r mod 32.
//
// `+3` because the extractor always reads a 3-byte window (P <= 15, shift <= 7
// => the window [shift, shift+P) never spans more than 3 bytes).
__host__ __device__ constexpr int rk_nbs(int nbits) {
    return ((((nbits + 7) / 8 + 3) - 4 + 31) / 32) * 32 + 4;
}

// ------------------------------------------------------------- the extractor
// SHARED BY BOTH TIERS. Reads `ng` tables' bucket ids out of the per-row
// bitstream and stores them L-major into out[BH, L, T].
//
//   bit0 = g*P   (g is LOCAL to this bitmap)  ->  byte by = bit0>>3, shift = bit0&7
//   P <= 15 and shift <= 7  =>  the window [shift, shift+P) always fits 3 bytes.
//
// Bytes at or past ceil(nbits/8) are never inside any window; callers zero the
// <= 3 that a window could still touch so `compute-sanitizer --tool initcheck`
// stays clean.
//
// `row0` is the flat proj-row index of tile row 0, so a chunked call may straddle
// (b, h) boundaries: bh and t are derived from the flat row.
//
// CONTRACT: `BM` is a compile-time power of two AND `nthreads % BM == 0` (both
// call sites use nthreads = 256 with BM in {128, 256}). That makes a thread's row
// index `i % BM` INVARIANT across its iterations, which is what lets the
// (bh, t) decomposition be hoisted out of the loop. It must be hoisted: `gr / T`
// is a 64-bit division, which nvcc emits as a MUFU.RCP + IMAD.WIDE + software
// slow-path CALL sequence -- ~30 instructions -- and inside the loop that is one
// such sequence PER STORED ID (measured in SASS: 6 MUFU.RCP for 5 STG.E.U8,
// versus 2 for 29 stores once hoisted). At T=131072, L=100, BH=8 the loop body
// stores 1.05e8 ids per layer, so the division alone would cost 0.15-0.30 ms --
// 30-60% of the whole pack stage, and invisible to a byte-traffic model.
//
// Consecutive `tid` still maps to consecutive r within a warp, so the store is
// exactly as coalesced as before, and every index is unchanged: this is a
// bit-identical restructuring of when the arithmetic happens, not what it is.
template<typename BT, int BM>
__device__ __forceinline__ void rk_emit(const unsigned char* __restrict__ sbits, int nbs,
                                        BT* __restrict__ out, long row0, int T, int L,
                                        int g0, int ng, int P, int nrow,
                                        int tid, int nthreads)
{
    const int r = tid & (BM - 1);              // invariant: nthreads % BM == 0
    if (r >= nrow) return;                     // no __syncthreads() inside: safe
    const unsigned mask = (P >= 32) ? 0xffffffffu : ((1u << P) - 1u);
    const long gr = row0 + r;
    const long bh = gr / T, t = gr - bh * T;   // hoisted: ONE 64-bit divide per thread
    BT* dst = out + ((bh * (long)L) + g0) * (long)T + t;
    const unsigned char* p0 = sbits + (long)r * nbs;
    for (int g = tid / BM; g < ng; g += nthreads / BM) {
        const int bit0 = g * P, by = bit0 >> 3;
        const unsigned char* p = p0 + by;
        const unsigned v = (unsigned)p[0] | ((unsigned)p[1] << 8) | ((unsigned)p[2] << 16);
        dst[(long)g * (long)T] = (BT)((v >> (bit0 & 7)) & mask);
    }
}

// ------------------------------------------------------------- host helpers
static inline int rk_cdiv(int a, int b) { return (a + b - 1) / b; }
