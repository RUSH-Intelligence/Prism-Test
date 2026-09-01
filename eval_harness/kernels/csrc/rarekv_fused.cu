// RareKV Tier 2: the projection GEMM, the sign test and the bit-pack fused into
// ONE tensor-core kernel, so the [B*H*T, L*P] projection NEVER materialises.
//
// WHAT THIS IS AND IS NOT BIT-EXACT TO
// ------------------------------------
// It is NOT bit-identical to the cuBLAS path (Tier 1 / `lsh_mode="pack"`), and
// nothing here claims otherwise. cuBLAS picks its own accumulation order by
// shape, arch and version, so a hand-written GEMM cannot be pinned to it.
// Instead this kernel is bit-identical to a DECLARED reference:
//
//   REFERENCE R.  d[i][j] = sum_{s=0}^{D/16-1}
//       mma.sync.aligned.m16n8k16.row.col.f32.{bf16|f16}.{bf16|f16}.f32(
//           K[i, 16s:16s+16], Wt[j, 16s:16s+16] )
//   accumulated in fp32 into ONE chained accumulator in strictly increasing s,
//   with no split-k, no atomics and no reassociation, on sm_90a.
//   bucket_l(i) = sum_p 2^p * [ d[i, l*P+p] > 0 ].
//
// R is deterministic and independent of BM, BN, the grid shape, L, T, occupancy,
// stream and run -- a property `torch.mm` has never had. The carve-out: the
// 16-term reduction tree INSIDE one HMMA is not architecturally specified, so R
// is pinned to one arch; any move to another arch (or to wgmma) requires
// re-measuring the divergence table.
//
// Why the divergence from cuBLAS is small and analysable: a bf16 x bf16 product
// is EXACT in fp32 (8+8 = 16 significand bits, inside fp32's 24), so cuBLAS and
// R can differ only by addition order, never by product rounding. Only the SIGN
// of each projection matters, so a difference changes a bucket id only when a
// projection sits within an ulp of zero. The divergence is measured and
// published by scripts/bench_rarekv_kernel.py, not asserted away.
//
// WHY THE LAYOUT IS WHAT IT IS
// ----------------------------
// G = 8 tables per column tile => BN = 8*P columns, so a column tile's bit range
// starts on a byte boundary and n8-tile `n` writes exactly byte `n` of the tile
// bitmap. No table ever straddles a tile, no per-P shift tables, no running
// accumulator across tiles. This deletes an entire bug class.
//
// No `ldmatrix`, no XOR swizzle: for m16n8k16 every A/B fragment register is two
// CONTIGUOUS, 4-byte-aligned elements, so a plain `uint32_t` shared-memory load
// suffices. The row pad LDA = D + 8 makes both the A hoist and the B hot loop
// bank-conflict free (word stride 68 == 4 mod 32); without it they are 8-way
// conflicted. Do not remove it.
//
// WHAT LIMITS THIS KERNEL (state it, do not read "50% efficiency" as slack)
// ------------------------------------------------------------------------
// Each warp owns ONE m16 tile (NW = BM/16 warps), so a (b0, b1) pair feeds
// exactly one HMMA: 2 LDS.32 per mma = 256 shared-memory bytes per 4096 flops =
// 16 flop/SMEM-byte, against an H200 machine balance of
// 989e12 / (132 * 1.98e9 * 128 B) = 29.6. The mma loop is therefore
// SHARED-MEMORY bound at ~54% of tensor-core peak BEFORE the staging, the
// epilogue and rk_emit are counted -- the "989 TFLOP/s at 50% efficiency" in the
// design note is a hard ceiling, not a conservative estimate, and the honest
// expectation for Tier 2 over Tier 1 is nearer 1.3-1.5x than 1.6-2.0x.
// Occupancy compounds it: ~118 registers at P=9/D=128 with
// __launch_bounds__(256, 2) is 16 warps/SM = 25%.
// The fix, if the bench asks for it, is MT = 2 m16 tiles per warp (each (b0, b1)
// then feeds 2 HMMA -> 32 flop/SMEM-byte, above the machine balance) at the cost
// of ~136 registers and __launch_bounds__(..., 1). That is a deliberate
// follow-up, not something to slip in unmeasured.
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <type_traits>

#include "rarekv_bits.cuh"

// ---------------------------------------------------------------- the mma ----
template<typename KT> struct rk_mma_op;

template<> struct rk_mma_op<__nv_bfloat16> {
    __device__ __forceinline__ static void run(float* d, const uint32_t* a,
                                               uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#elif defined(__CUDA_ARCH__)
        // A fatbin built for a pre-Ampere arch has no m16n8k16 mma. Leaving the
        // accumulator at its 0.f initialisation would make `acc > 0.f` false
        // everywhere and return an all-zero bucket tensor -- a SILENT wrong
        // answer. Trap instead. (`fused_buckets` also refuses sm_<80 on the host,
        // so this is the belt to that braces.)
        (void)d; (void)a; (void)b0; (void)b1;
        __trap();
#else
        (void)d; (void)a; (void)b0; (void)b1;      // host pass: body is discarded
#endif
    }
};

template<> struct rk_mma_op<__half> {
    __device__ __forceinline__ static void run(float* d, const uint32_t* a,
                                               uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#elif defined(__CUDA_ARCH__)
        // A fatbin built for a pre-Ampere arch has no m16n8k16 mma. Leaving the
        // accumulator at its 0.f initialisation would make `acc > 0.f` false
        // everywhere and return an all-zero bucket tensor -- a SILENT wrong
        // answer. Trap instead. (`fused_buckets` also refuses sm_<80 on the host,
        // so this is the belt to that braces.)
        (void)d; (void)a; (void)b0; (void)b1;
        __trap();
#else
        (void)d; (void)a; (void)b0; (void)b1;      // host pass: body is discarded
#endif
    }
};

// ------------------------------------------------------------- the geometry --
// One place for every size, so the host smem request and the kernel's own
// pointer arithmetic cannot drift apart.
template<typename KT, int P, int KSTEPS, int BM>
struct RKFusedGeom {
    static constexpr int Dc  = KSTEPS * 16;
    static constexpr int LDA = Dc + 8;                 // row pad: bank conflicts
    static constexpr int BN  = 8 * P;                  // 8 tables per column tile
    static constexpr int GB  = rk_nbs(8 * P);          // tile bitmap row stride
    static constexpr int A_ELEMS = BM * LDA;
    static constexpr int B_ELEMS = BN * LDA;           // single buffered, see below
    static constexpr int AB_BYTES =
        (int)sizeof(KT) * (A_ELEMS > B_ELEMS ? A_ELEMS : B_ELEMS);
    static constexpr int SMEM = AB_BYTES + BM * GB;
};

template<typename KT, typename BT, int P, int KSTEPS, int BM>
__global__ __launch_bounds__(2 * BM, 2)
void rk_fused_kernel(const KT* __restrict__ K,      // [BH, T, D] viewed as [Rc, D]
                     const KT* __restrict__ PT,     // planesT [LP_pad, D], zero padded
                     BT* __restrict__ out,          // [BH, L, T]
                     long row0, int Rc, int T, int L)
{
    using G = RKFusedGeom<KT, P, KSTEPS, BM>;
    constexpr int Dc = G::Dc, LDA = G::LDA, BN = G::BN, GB = G::GB;
    constexpr int NN8 = P;                             // n8 tiles per column tile
    constexpr int TABLES = 8;                          // tables per column tile

    const int tid = threadIdx.x, nthreads = 2 * BM;
    const int warp = tid >> 5, lane = tid & 31, g = lane >> 2, q = lane & 3;
    const int t0 = blockIdx.x * BM, nrow = min(BM, Rc - t0);

    extern __shared__ char rk_smem[];
    KT* sAB = (KT*)rk_smem;                            // sA and sB ALIAS
    unsigned char* sbits = (unsigned char*)(rk_smem + G::AB_BYTES);

    // ---- A: read K once from HBM, hoist the whole k = D reduction to registers.
    // This inversion is what makes the design work: K is never re-read across the
    // ntiles column tiles.
    for (int i = tid; i < BM * (Dc / 8); i += nthreads) {
        const int r = i / (Dc / 8), c = (i % (Dc / 8)) * 8;
        const uint4 v = (r < nrow) ? *(const uint4*)(K + (long)(t0 + r) * Dc + c)
                                   : make_uint4(0u, 0u, 0u, 0u);   // masked rows -> 0
        *(uint4*)&sAB[(long)r * LDA + c] = v;
    }
    __syncthreads();

    uint32_t a[KSTEPS][4];
    {
        const int rb = warp * 16;
        #pragma unroll
        for (int s = 0; s < KSTEPS; ++s) {
            const int kb = s * 16;
            a[s][0] = *(const uint32_t*)&sAB[(long)(rb + g)     * LDA + kb + 2 * q];
            a[s][1] = *(const uint32_t*)&sAB[(long)(rb + g + 8) * LDA + kb + 2 * q];
            a[s][2] = *(const uint32_t*)&sAB[(long)(rb + g)     * LDA + kb + 2 * q + 8];
            a[s][3] = *(const uint32_t*)&sAB[(long)(rb + g + 8) * LDA + kb + 2 * q + 8];
        }
    }
    __syncthreads();                                   // sA dead; sAB becomes sB

    // ---- column sweep: 8 tables at a time -----------------------------------
    const int ntiles = (L + TABLES - 1) / TABLES;
    for (int ct = 0; ct < ntiles; ++ct) {
        const int col0 = ct * BN, g0 = ct * TABLES, ng = min(TABLES, L - g0);
        // ONE B buffer, not two. The loop is stage -> sync -> mma -> epilogue ->
        // sync -> emit -> sync, and nothing issues tile ct+1's global loads during
        // tile ct's mma, so a second buffer bought no overlap -- it was dead shared
        // memory, and at D=256/P=15 it was the difference between 2 CTAs/SM and 1.
        // The trailing __syncthreads() of iteration ct is what makes reusing this
        // one buffer safe. (Real pipelining -- hoisting ct+1's global->register
        // loads above ct's mma loop -- is a documented follow-up, not this change.)
        KT* sB = sAB;
        for (int i = tid; i < BN * (Dc / 8); i += nthreads) {
            const int r = i / (Dc / 8), c = (i % (Dc / 8)) * 8;
            *(uint4*)&sB[(long)r * LDA + c] =
                *(const uint4*)(PT + (long)(col0 + r) * Dc + c);    // PT is zero padded
        }
        __syncthreads();

        float acc[NN8][4];
        #pragma unroll
        for (int n = 0; n < NN8; ++n) {
            acc[n][0] = 0.f; acc[n][1] = 0.f; acc[n][2] = 0.f; acc[n][3] = 0.f;
        }

        // THE PINNED ORDER: k chunks 0..KSTEPS-1, strictly ascending, one chained
        // accumulator per output. This is Reference R.
        #pragma unroll
        for (int s = 0; s < KSTEPS; ++s) {
            #pragma unroll
            for (int n = 0; n < NN8; ++n) {
                const int cb = 8 * n;
                const uint32_t b0 =
                    *(const uint32_t*)&sB[(long)(cb + g) * LDA + s * 16 + 2 * q];
                const uint32_t b1 =
                    *(const uint32_t*)&sB[(long)(cb + g) * LDA + s * 16 + 2 * q + 8];
                rk_mma_op<KT>::run(acc[n], a[s], b0, b1);
            }
        }

        // ---- epilogue: sign -> the SAME bitmap the Tier-1 kernel builds --------
        // C fragment: this lane holds (row g, cols 2q, 2q+1) and (row g+8, same).
        // The 8 columns of ONE row live in 4 lanes, not 32, so __ballot_sync is the
        // WRONG primitive; a 2-step __shfl_xor butterfly OR-reduces the 4-lane
        // group. Both rows ride in one 32-bit word, so one butterfly serves 16 bits.
        //
        // ALL 32 LANES MUST REACH BOTH SHUFFLES: never guard them with a row mask,
        // and never `return`/`continue` between the mma and the second shuffle.
        // Only the STORE is predicated.
        #pragma unroll
        for (int n = 0; n < NN8; ++n) {
            unsigned w = ((unsigned)(acc[n][0] > 0.f) << (2 * q))
                       | ((unsigned)(acc[n][1] > 0.f) << (2 * q + 1))
                       | ((unsigned)(acc[n][2] > 0.f) << (16 + 2 * q))
                       | ((unsigned)(acc[n][3] > 0.f) << (16 + 2 * q + 1));
            w |= __shfl_xor_sync(0xffffffffu, w, 1);
            w |= __shfl_xor_sync(0xffffffffu, w, 2);
            if (q == 0) {
                const int r = warp * 16 + g;           // byte index == n, exactly
                sbits[(long)r       * GB + n] = (unsigned char)(w & 0xff);
                sbits[(long)(r + 8) * GB + n] = (unsigned char)((w >> 16) & 0xff);
            }
        }
        // slack bytes, for compute-sanitizer --tool initcheck
        for (int i = tid; i < BM * 3; i += nthreads) {
            const int r = i / 3, k = i - r * 3;
            if (P + k < GB) sbits[(long)r * GB + P + k] = 0;
        }
        __syncthreads();

        rk_emit<BT, BM>(sbits, GB, out, row0 + t0, T, L, g0, ng, P, nrow, tid, nthreads);
        __syncthreads();                               // before sbits / sB are reused
    }
}

// ------------------------------------------------------- the serial reference
// A deliberately dumb thread-per-row kernel: strictly ascending k, explicit
// __fmul_rn/__fadd_rn (so nvcc cannot contract into an FMA and change the order),
// no shared memory, no fragments. It exists to BISECT a failing Gate A in one
// SLURM job instead of five: if `serial` matches the torch reference but `fused`
// does not, the bug is in the mma fragment mapping or the bitmap, not in the
// dispatch or the layout. It is also an on-GPU oracle for the integer-operand
// gate. It is O(Rc * L * P * D) on CUDA cores -- a debug entry point, never the
// production path.
template<typename KT, typename BT>
__global__ void rk_serial_kernel(const KT* __restrict__ K, const KT* __restrict__ PT,
                                 BT* __restrict__ out, long row0, int Rc, int T,
                                 int L, int P, int D)
{
    const long r = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= (long)Rc) return;
    const KT* krow = K + r * (long)D;
    const long gr = row0 + r, bh = gr / T, t = gr - bh * T;
    for (int l = 0; l < L; ++l) {
        unsigned id = 0;
        for (int p = 0; p < P; ++p) {
            const KT* w = PT + (long)(l * P + p) * (long)D;
            float s = 0.f;
            for (int k = 0; k < D; ++k)
                s = __fadd_rn(s, __fmul_rn(rk_f(krow[k]), rk_f(w[k])));
            if (s > 0.f) id |= (1u << p);
        }
        out[((bh * (long)L) + l) * (long)T + t] = (BT)id;
    }
}

// ------------------------------------------------------------------- launch --
template<typename KT, int P, int KSTEPS>
static void rk_fused_launch_p(const void* kptr, const void* ptptr, torch::Tensor bucket,
                              long row0, int Rc, int T, int L)
{
    constexpr int BM = 128;
    using BT = typename std::conditional<(P <= 8), uint8_t, int16_t>::type;
    using G = RKFusedGeom<KT, P, KSTEPS, BM>;
    auto kern = rk_fused_kernel<KT, BT, P, KSTEPS, BM>;
    // No-op below 48 KB; required above it (D = 256 pushes sA past the static cap).
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM);
    kern<<<dim3(rk_cdiv(Rc, BM)), 2 * BM, G::SMEM>>>(
        (const KT*)kptr, (const KT*)ptptr, (BT*)bucket.data_ptr(), row0, Rc, T, L);
}

#define RK_FUSED_P_CASES(KT, KSTEPS)                                                 \
    switch (P) {                                                                     \
        case  1: rk_fused_launch_p<KT,  1, KSTEPS>(kp, pp, bucket, 0, Rc, T, L); break; \
        case  2: rk_fused_launch_p<KT,  2, KSTEPS>(kp, pp, bucket, 0, Rc, T, L); break; \
        case  3: rk_fused_launch_p<KT,  3, KSTEPS>(kp, pp, bucket, 0, Rc, T, L); break; \
        case  4: rk_fused_launch_p<KT,  4, KSTEPS>(kp, pp, bucket, 0, Rc, T, L); break; \
        case  5: rk_fused_launch_p<KT,  5, KSTEPS>(kp, pp, bucket, 0, Rc, T, L); break; \
        case  6: rk_fused_launch_p<KT,  6, KSTEPS>(kp, pp, bucket, 0, Rc, T, L); break; \
        case  7: rk_fused_launch_p<KT,  7, KSTEPS>(kp, pp, bucket, 0, Rc, T, L); break; \
        case  8: rk_fused_launch_p<KT,  8, KSTEPS>(kp, pp, bucket, 0, Rc, T, L); break; \
        case  9: rk_fused_launch_p<KT,  9, KSTEPS>(kp, pp, bucket, 0, Rc, T, L); break; \
        case 10: rk_fused_launch_p<KT, 10, KSTEPS>(kp, pp, bucket, 0, Rc, T, L); break; \
        case 11: rk_fused_launch_p<KT, 11, KSTEPS>(kp, pp, bucket, 0, Rc, T, L); break; \
        case 12: rk_fused_launch_p<KT, 12, KSTEPS>(kp, pp, bucket, 0, Rc, T, L); break; \
        case 13: rk_fused_launch_p<KT, 13, KSTEPS>(kp, pp, bucket, 0, Rc, T, L); break; \
        case 14: rk_fused_launch_p<KT, 14, KSTEPS>(kp, pp, bucket, 0, Rc, T, L); break; \
        case 15: rk_fused_launch_p<KT, 15, KSTEPS>(kp, pp, bucket, 0, Rc, T, L); break; \
        default: TORCH_CHECK(false, "fused kernel needs P in [1, 15]; got ", P);         \
    }

template<typename KT>
static void rk_fused_launch(const void* kp, const void* pp, torch::Tensor bucket,
                            int Rc, int T, int L, int P, int D)
{
    // D is instantiated for 64 / 128 / 256 only. The dispatcher in rarekv_lsh.py
    // falls back to the (bit-identical) pack path for every other head_dim, so an
    // unusual model degrades in speed, never in answer.
    if (D == 64)       { RK_FUSED_P_CASES(KT, 4) }
    else if (D == 128) { RK_FUSED_P_CASES(KT, 8) }
    else if (D == 256) { RK_FUSED_P_CASES(KT, 16) }
    else TORCH_CHECK(false, "fused kernel is instantiated for head_dim in {64,128,256}; got ", D);
}

torch::Tensor fused_buckets(torch::Tensor keys,      // [B, H, T, D] contiguous
                            torch::Tensor planes_t,  // [LP_pad, D], zero padded
                            int64_t L, int64_t P, int64_t block_m)
{
    TORCH_CHECK(keys.is_cuda() && keys.is_contiguous() && keys.dim() == 4,
                "keys must be a contiguous 4-D CUDA tensor [B, H, T, D]");
    TORCH_CHECK(planes_t.is_cuda() && planes_t.is_contiguous() && planes_t.dim() == 2,
                "planes_t must be a contiguous 2-D CUDA tensor [LP_pad, D]");
    // The C++ entry point enforces the same precondition the python dispatcher
    // does, because a direct `ext.fused_buckets(...)` call -- what a debugging
    // session or the divergence harness would write -- bypasses that dispatcher,
    // and the failure mode below sm_80 is an all-zero answer, not an error.
    const auto* props = at::cuda::getDeviceProperties(keys.device().index());
    TORCH_CHECK(props->major >= 8, "the fused kernel needs sm_80+ (m16n8k16 mma); got sm_",
                props->major, props->minor, ". Use lsh_mode='pack' (bit-identical).");
    TORCH_CHECK(keys.device() == planes_t.device(), "keys and planes_t must share a device");
    TORCH_CHECK(keys.scalar_type() == planes_t.scalar_type(),
                "keys and planes_t must share a dtype");
    const auto dt = keys.scalar_type();
    TORCH_CHECK(dt == torch::kBFloat16 || dt == torch::kHalf,
                "the fused kernel is tensor-core only: keys must be bfloat16 or float16");
    TORCH_CHECK(P >= 1 && P <= 15, "fused kernel needs P in [1, 15] (int16 bucket ids "
                "are signed); got ", P);
    TORCH_CHECK(L >= 1, "L must be >= 1");
    const int64_t B = keys.size(0), H = keys.size(1), T = keys.size(2), D = keys.size(3);
    TORCH_CHECK(D % 16 == 0 && D <= 256, "fused kernel needs head_dim % 16 == 0 and <= 256; got ", D);
    const int64_t ntiles = (L + 7) / 8, LP_pad = ntiles * 8 * P;
    TORCH_CHECK(planes_t.size(0) == LP_pad,
                "planes_t must have ceil(L/8)*8*P = ", LP_pad, " rows; got ", planes_t.size(0));
    TORCH_CHECK(planes_t.size(1) == D, "planes_t.size(1) must equal head_dim = ", D);
    TORCH_CHECK(block_m == 0 || block_m == 128,
                "the fused kernel currently instantiates BM = 128 only; got block_m=", block_m);
    const int64_t BH = B * H, Rc = BH * T;
    TORCH_CHECK(Rc <= (int64_t)2147483647, "fused kernel indexes rows in int32");

    auto opt = torch::TensorOptions()
                   .dtype(P <= 8 ? torch::kByte : torch::kShort)
                   .device(keys.device());
    auto bucket = torch::empty({BH, L, T}, opt);
    if (Rc == 0) return bucket;

    if (dt == torch::kBFloat16)
        rk_fused_launch<__nv_bfloat16>(keys.data_ptr(), planes_t.data_ptr(), bucket,
                                       (int)Rc, (int)T, (int)L, (int)P, (int)D);
    else
        rk_fused_launch<__half>(keys.data_ptr(), planes_t.data_ptr(), bucket,
                                (int)Rc, (int)T, (int)L, (int)P, (int)D);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return bucket;
}

torch::Tensor serial_buckets(torch::Tensor keys, torch::Tensor planes_t,
                             int64_t L, int64_t P)
{
    TORCH_CHECK(keys.is_cuda() && keys.is_contiguous() && keys.dim() == 4,
                "keys must be a contiguous 4-D CUDA tensor [B, H, T, D]");
    TORCH_CHECK(planes_t.is_cuda() && planes_t.is_contiguous() && planes_t.dim() == 2,
                "planes_t must be a contiguous 2-D CUDA tensor [>= L*P, D]");
    TORCH_CHECK(keys.scalar_type() == planes_t.scalar_type(), "dtype mismatch");
    TORCH_CHECK(P >= 1 && P <= 15, "serial reference needs P in [1, 15]; got ", P);
    const auto dt = keys.scalar_type();
    TORCH_CHECK(dt == torch::kBFloat16 || dt == torch::kHalf || dt == torch::kFloat,
                "keys must be bfloat16, float16 or float32");
    const int64_t B = keys.size(0), H = keys.size(1), T = keys.size(2), D = keys.size(3);
    TORCH_CHECK(planes_t.size(0) >= L * P && planes_t.size(1) == D,
                "planes_t must be [>= L*P, head_dim]");
    const int64_t BH = B * H, Rc = BH * T;
    auto opt = torch::TensorOptions()
                   .dtype(P <= 8 ? torch::kByte : torch::kShort)
                   .device(keys.device());
    auto bucket = torch::empty({BH, L, T}, opt);
    if (Rc == 0) return bucket;
    const int threads = 128;
    const dim3 grid((unsigned)((Rc + threads - 1) / threads));

#define RK_SERIAL_DISPATCH(SCALAR_T)                                                    \
    do {                                                                                \
        if (P <= 8)                                                                     \
            rk_serial_kernel<SCALAR_T, uint8_t><<<grid, threads>>>(                     \
                (const SCALAR_T*)keys.data_ptr(), (const SCALAR_T*)planes_t.data_ptr(), \
                (uint8_t*)bucket.data_ptr(), 0, (int)Rc, (int)T, (int)L, (int)P, (int)D);\
        else                                                                            \
            rk_serial_kernel<SCALAR_T, int16_t><<<grid, threads>>>(                     \
                (const SCALAR_T*)keys.data_ptr(), (const SCALAR_T*)planes_t.data_ptr(), \
                (int16_t*)bucket.data_ptr(), 0, (int)Rc, (int)T, (int)L, (int)P, (int)D);\
    } while (0)

    if (dt == torch::kBFloat16)   RK_SERIAL_DISPATCH(__nv_bfloat16);
    else if (dt == torch::kHalf)  RK_SERIAL_DISPATCH(__half);
    else                          RK_SERIAL_DISPATCH(float);
#undef RK_SERIAL_DISPATCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return bucket;
}
