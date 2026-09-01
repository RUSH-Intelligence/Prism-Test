// RareKV Tier 1: fuse `sign -> int32 -> *powers -> sum -> permute` into ONE
// streaming kernel that reads `proj` once and writes uint8/int16 bucket ids
// straight into the [BH, L, T] layout the collision kernel wants.
//
// Exactness
// ---------
// The projection GEMM is UNCHANGED (still `keys.reshape(-1, D) @ planes`), so
// the float values this kernel reads are bit-for-bit the ones the torch path
// reads. Everything after the sign test is integer. Therefore Tier 1 is
// UNCONDITIONALLY bit-identical to the torch reference -- `torch.equal`, not
// `allclose`. See `rarekv_lsh.buckets_from_proj_torch`, which is the reference.
//
// Traffic
// -------
// The torch sequence materialises, per element of [B*H*T, L*P]: a bool `sign`
// (1 B), an int32 `packed` (4 B, written twice -- the cast and the in-place
// multiply), plus an int32 [B*H*T, L] `bucket` and its transposing clone. This
// kernel reads `proj` (2 B/elem) and writes `b` bytes per (row, table), where
// b = 1 for P <= 8 and 2 for P <= 15. At T=128K, L=70, P=6 that is
// 11.8 GB -> 2.8 GB per layer.
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include "rarekv_bits.cuh"

// One CTA owns BM consecutive flat proj rows. Phase 1 streams `proj` into a
// per-row shared-memory sign bitmap with warp-coalesced loads; phase 2 hands
// that bitmap to the shared extractor.
template<typename S, typename BT, int BM>
__global__ __launch_bounds__(256, 4)
void rk_pack_kernel(const S* __restrict__ proj,      // [Rc, LP], flat rows
                    BT* __restrict__ out,            // [BH, L, T]
                    long row0, int Rc, int T, int L, int P, int LP,
                    int nbs, int NB)                 // NB = ceil(LP/8)
{
    extern __shared__ unsigned char sbits[];
    const int tid = threadIdx.x, nthreads = blockDim.x;
    const int t0 = blockIdx.x * BM;
    const int nrow = min(BM, Rc - t0);
    const long base = (long)t0 * (long)LP;           // 64-bit: Rc*LP overflows int32

    // ---- phase 1: warp-coalesced stream, 32 projections -> one 4-byte word ----
    // Lane l reads projection column c0+l, so a warp's load is 32 CONSECUTIVE
    // elements -- 64 contiguous bytes at bf16/fp16, 128 at fp32: ONE L1 wavefront,
    // 100% sector efficiency. (The obvious byte-per-thread form gives each lane 8
    // consecutive elements, i.e. a 16-byte LANE STRIDE: one instruction's 32 lanes
    // then span 512 bytes = 4 cache lines and return 64 useful bytes, 8x the L1
    // wavefronts for identical DRAM traffic. That amplification is invisible to a
    // byte-traffic model and would eat the whole Tier-1 margin, since this stage
    // is budgeted at 3.6 TB/s.)
    //
    // `__ballot_sync` then yields bit l = sign of column c0+l, which is EXACTLY
    // the LSB-first bitstream `rk_nbs`/`rk_emit` specify, and one lane stores the
    // 32 bits as a 4-byte word at byte c0/8. It is bit-identical to the scalar
    // form: same predicate, same bit positions.
    //
    // Alignment: c0 = 32*chunk => c0/8 is a multiple of 4, and nbs == 4 (mod 32)
    // is a multiple of 4, so `sbits + r*nbs + c0/8` is always 4-byte aligned. The
    // last word may run past NB, but never past nbs: with LP = 32a + b (0 < b),
    // the words cover 4(a+1) bytes and nbs >= ceil(LP/8) + 3 = 4a + ceil(b/8) + 3
    // >= 4a + 4. No vector-load alignment fixup is needed anywhere, which is why
    // this replaces the earlier "scalar loads because uint4 needs LP % 8 == 0"
    // trade-off rather than refining it.
    const int nchunks = (LP + 31) >> 5;
    const int warp = tid >> 5, lane = tid & 31, nwarps = nthreads >> 5;
    for (int it = warp; it < nrow * nchunks; it += nwarps) {
        const int r = it / nchunks, c0 = (it - r * nchunks) << 5;
        const int col = c0 + lane;
        bool s = false;
        if (col < LP) s = rk_pos(proj[base + (long)r * (long)LP + (long)col]);
        const unsigned m = __ballot_sync(0xffffffffu, s);   // bit l == column c0+l
        if (lane == 0) *(unsigned*)(sbits + (long)r * nbs + (c0 >> 3)) = m;
    }
    // The ballot words already cover bytes [0, 4*nchunks), which is >= NB. Zero the
    // <= 3 bytes between there and NB+3 -- the highest byte any extraction window
    // can touch -- so `compute-sanitizer --tool initcheck` stays clean. Their bits
    // lie outside every window, so the values are hygiene, not correctness.
    // Disjoint from the ballot stores, hence no ordering hazard.
    const int NBW = nchunks * 4;
    for (int i = tid; i < nrow * 3; i += nthreads) {
        const int r = i / 3, by = NBW + (i - r * 3);
        if (by < NB + 3 && by < nbs) sbits[(long)r * nbs + by] = 0;
    }
    __syncthreads();

    // ---- phase 2: the SHARED extractor ---------------------------------------
    rk_emit<BT, BM>(sbits, nbs, out, row0 + t0, T, L, /*g0=*/0, /*ng=*/L, P,
                    nrow, tid, nthreads);
}

template<typename S, typename BT, int BM>
static void rk_pack_launch_bm(const S* pptr, BT* optr, long row0, int Rc, int T, int L,
                              int P, int LP, int nbs, int NB)
{
    const int threads = 256;                         // rk_emit needs threads % BM == 0
    const size_t smem = (size_t)BM * (size_t)nbs;
    auto kern = rk_pack_kernel<S, BT, BM>;
    // A no-op below 48 KB, and REQUIRED above it: dynamic shared memory is capped
    // at 48 KB per block unless opted in. Without this, block_m=256 at L*P >= 1289
    // (and block_m=128 at L*P >= 2825) failed the launch with a bare
    // `invalid argument`. pack_buckets_into refuses anything past the device's
    // opt-in limit before we get here.
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    kern<<<dim3(rk_cdiv(Rc, BM)), threads, smem>>>(
        pptr, optr, row0, Rc, T, L, P, LP, nbs, NB);
}

template<typename S, typename BT>
static void rk_pack_launch(const S* pptr, BT* optr, long row0, int Rc, int T, int L,
                           int P, int LP, int BM)
{
    const int nbs = rk_nbs(LP);
    const int NB = (LP + 7) / 8;
    if (BM == 256)
        rk_pack_launch_bm<S, BT, 256>(pptr, optr, row0, Rc, T, L, P, LP, nbs, NB);
    else
        rk_pack_launch_bm<S, BT, 128>(pptr, optr, row0, Rc, T, L, P, LP, nbs, NB);
}

// Writes bucket ids for `proj`'s rows into a PRE-ALLOCATED [BH, L, T] tensor.
// In place with a row offset, so the chunked (Tier 1c) caller reuses one kernel.
void pack_buckets_into(torch::Tensor proj, torch::Tensor bucket, int64_t row0,
                       int64_t P, int64_t block_m)
{
    TORCH_CHECK(proj.is_cuda() && proj.is_contiguous() && proj.dim() == 2,
                "proj must be a contiguous 2-D CUDA tensor [Rc, L*P]");
    TORCH_CHECK(bucket.is_cuda() && bucket.is_contiguous() && bucket.dim() == 3,
                "bucket must be a contiguous 3-D CUDA tensor [BH, L, T]");
    TORCH_CHECK(bucket.device() == proj.device(), "proj and bucket must share a device");
    TORCH_CHECK(P >= 1 && P <= 15, "n_planes P must be in [1, 15] for the pack kernel "
                "(int16 bucket ids are signed); got ", P);
    const int64_t Rc = proj.size(0), LP = proj.size(1);
    const int64_t BH = bucket.size(0), L = bucket.size(1), T = bucket.size(2);
    TORCH_CHECK(LP == L * P, "proj.size(1) must equal L*P = ", L * P, "; got ", LP);
    TORCH_CHECK(row0 >= 0 && row0 + Rc <= BH * T,
                "row window [", row0, ", ", row0 + Rc, ") escapes BH*T = ", BH * T);
    TORCH_CHECK(Rc <= (int64_t)2147483647 && LP <= (int64_t)2147483647 && T <= (int64_t)2147483647,
                "pack kernel indexes rows/columns/T in int32");
    const auto bdt = bucket.scalar_type();
    TORCH_CHECK(bdt == (P <= 8 ? torch::kByte : torch::kShort),
                "bucket dtype must be uint8 for P <= 8 and int16 for 9 <= P <= 15; got ", bdt);
    const int BM = (block_m == 0) ? 128 : (int)block_m;
    TORCH_CHECK(BM == 128 || BM == 256, "block_m must be 0 (default 128), 128 or 256; got ", block_m);
    // The sign bitmap is BM * rk_nbs(L*P) bytes of DYNAMIC shared memory, which is
    // capped at 48 KB without the opt-in above and at `sharedMemPerBlockOptin`
    // with it. Refuse here, naming the knob, instead of letting the launch fail
    // with `invalid argument`: `rarekv_lsh.lsh_buckets` checks the same bound and
    // routes past it to the (bit-identical) torch sequence, so this only fires for
    // a direct call.
    const int64_t smem = (int64_t)BM * (int64_t)rk_nbs((int)LP);
    const int64_t cap = (int64_t)at::cuda::getDeviceProperties(
        proj.device().index())->sharedMemPerBlockOptin;
    TORCH_CHECK(smem <= cap, "the pack kernel needs ", smem, " B of shared memory for "
                "block_m=", BM, " and L*P=", LP, ", above this device's ", cap,
                " B opt-in limit. Use block_m=128, lower n_tables*n_planes, or "
                "lsh_mode='torch' (bit-identical, no shared-memory limit).");
    if (Rc == 0) return;

    const auto sdt = proj.scalar_type();
    TORCH_CHECK(sdt == torch::kBFloat16 || sdt == torch::kHalf || sdt == torch::kFloat,
                "proj must be bfloat16, float16 or float32; got ", sdt);

#define RK_PACK_DISPATCH(SCALAR_T, TORCH_T)                                                  \
    if (sdt == TORCH_T) {                                                                    \
        const SCALAR_T* pptr = (const SCALAR_T*)proj.data_ptr();                             \
        if (bdt == torch::kByte)                                                             \
            rk_pack_launch<SCALAR_T, uint8_t>(pptr, (uint8_t*)bucket.data_ptr(),             \
                                              (long)row0, (int)Rc, (int)T, (int)L,           \
                                              (int)P, (int)LP, BM);                          \
        else                                                                                 \
            rk_pack_launch<SCALAR_T, int16_t>(pptr, (int16_t*)bucket.data_ptr(),             \
                                              (long)row0, (int)Rc, (int)T, (int)L,           \
                                              (int)P, (int)LP, BM);                          \
    }
    RK_PACK_DISPATCH(__nv_bfloat16, torch::kBFloat16)
    else RK_PACK_DISPATCH(__half, torch::kHalf)
    else RK_PACK_DISPATCH(float, torch::kFloat)
#undef RK_PACK_DISPATCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Convenience wrapper: allocates the [BH, L, T] bucket tensor and fills it.
torch::Tensor pack_buckets(torch::Tensor proj, int64_t BH, int64_t T,
                           int64_t L, int64_t P, int64_t block_m)
{
    TORCH_CHECK(proj.is_cuda() && proj.dim() == 2, "proj must be a 2-D CUDA tensor");
    TORCH_CHECK(proj.size(0) == BH * T, "proj.size(0) must equal BH*T = ", BH * T);
    auto opt = torch::TensorOptions()
                   .dtype(P <= 8 ? torch::kByte : torch::kShort)
                   .device(proj.device());
    auto bucket = torch::empty({BH, L, T}, opt);
    pack_buckets_into(proj.contiguous(), bucket, 0, P, block_m);
    return bucket;
}
