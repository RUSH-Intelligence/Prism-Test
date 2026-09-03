// RareKV collision counting, CUDA.
//
// Layout is [BH, L, T] (L-major, T-minor), copied from SOCKET's
// soft_hash_score.cu: for a fixed table l, consecutive threads read consecutive
// keys, so every bucket load is fully coalesced. The [BH, T, L] layout the torch
// path uses makes a 32-thread warp span L*4*32 bytes instead of 4 cache lines.
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

// ---- 1. Privatised histogram: shared-memory atomics, then one global merge ----
// Global atomic traffic per (bh,l) drops from T to (#blocks * #non-empty bins).
// Shared memory is only R ints (<= 4 KB at R=1024), so this works for ALL R --
// unlike a per-tile dense histogram, whose cost grows with R.
template<typename BT>
__global__ void hist_kernel(const BT* __restrict__ buckets,    // [BH, L, T]
                            int* __restrict__ counts,          // [BH*L, R]
                            int T, int R)
{
    extern __shared__ int smem[];
    const int bhl = blockIdx.y;
    for (int i = threadIdx.x; i < R; i += blockDim.x) smem[i] = 0;
    __syncthreads();

    const long base = (long)bhl * T;
    const int stride = gridDim.x * blockDim.x;
    for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < T; t += stride) {
        atomicAdd(&smem[(int)buckets[base + t]], 1);  // coalesced load
    }
    __syncthreads();
    for (int i = threadIdx.x; i < R; i += blockDim.x)
        if (smem[i]) atomicAdd(&counts[(long)bhl * R + i], smem[i]);
}

// ---- 2. Thread-per-key scoring: each thread owns one key, independently ----
// Directly modelled on soft_hash_score.cu's kernel, with the query-probability
// table swapped for integer collision counts (so the sum is exact).
template<typename BT, int L>
__global__ void gather_kernel_t(const BT* __restrict__ buckets,   // [BH, L, T]
                                const int* __restrict__ counts,   // [BH*L, R]
                                int* __restrict__ csum,           // [BH, T]
                                int T, int R)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int bh = blockIdx.y;
    if (t >= T) return;
    int s = 0;
    #pragma unroll
    for (int l = 0; l < L; ++l) {
        const int r = (int)buckets[((long)bh * L + l) * T + t];   // coalesced
        s += counts[((long)bh * L + l) * R + r];                  // random within a small table
    }
    csum[(long)bh * T + t] = s;
}

template<typename BT>
__global__ void gather_kernel_dyn(const BT* __restrict__ buckets, const int* __restrict__ counts,
                                  int* __restrict__ csum, int T, int R, int L)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int bh = blockIdx.y;
    if (t >= T) return;
    int s = 0;
    #pragma unroll 8
    for (int l = 0; l < L; ++l) {
        const int r = (int)buckets[((long)bh * L + l) * T + t];
        s += counts[((long)bh * L + l) * R + r];
    }
    csum[(long)bh * T + t] = s;
}

static inline int cdiv(int a, int b) { return (a + b - 1) / b; }

template<typename BT>
static void launch(const BT* bptr, int* cptr, int* sptr, int BH, int L, int T, int R,
                   int hist_threads, int hist_blocks, int gather_threads)
{

    // 256 keys per block is too few to privatise: once R >= keys/block, nearly every
    // key lands in its own bin and the `if (smem[i]) atomicAdd` merge degenerates to
    // ONE global atomic per key. Measured: 1.88-1.92 TB/s for R <= 256, collapsing to
    // 1.00 TB/s at R=1024 on identical byte traffic. Require >= 8 keys per bin per
    // block. Integer-only, so this cannot change the (exact) counts.
    int hb = hist_blocks > 0
                 ? hist_blocks
                 : std::max(1, std::min({cdiv(T, hist_threads), T / (8 * R), 512}));
    dim3 hgrid(hb, BH * L);
    // The privatised histogram is R ints of DYNAMIC shared memory, and dynamic
    // shared memory is capped at 48 KB per block unless the kernel opts in. That
    // cap binds at R > 12288, i.e. P >= 14 -- exactly the range the int16 bucket
    // ids advertise (R <= 32768), so without this opt-in a P=14/15 config passed
    // every check above and then died at launch with `invalid argument`.
    // 131072 B at R=32768 is one CTA per SM; that is an edge configuration, and a
    // correct slow launch beats a failed one. `collide_cuda` refuses anything past
    // the device's opt-in limit before we get here.
    const size_t hsmem = (size_t)R * sizeof(int);
    if (hsmem > 48u * 1024u)
        cudaFuncSetAttribute((const void*)hist_kernel<BT>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)hsmem);
    hist_kernel<BT><<<hgrid, hist_threads, hsmem>>>(bptr, cptr, T, R);

    dim3 ggrid(cdiv(T, gather_threads), BH);
    switch (L) {
        case 40: gather_kernel_t<BT,40><<<ggrid, gather_threads>>>(bptr, cptr, sptr, T, R); break;
        case 50: gather_kernel_t<BT,50><<<ggrid, gather_threads>>>(bptr, cptr, sptr, T, R); break;
        case 60: gather_kernel_t<BT,60><<<ggrid, gather_threads>>>(bptr, cptr, sptr, T, R); break;
        case 70: gather_kernel_t<BT,70><<<ggrid, gather_threads>>>(bptr, cptr, sptr, T, R); break;
        case 80: gather_kernel_t<BT,80><<<ggrid, gather_threads>>>(bptr, cptr, sptr, T, R); break;
        case 100: gather_kernel_t<BT,100><<<ggrid, gather_threads>>>(bptr, cptr, sptr, T, R); break;
        default: gather_kernel_dyn<BT><<<ggrid, gather_threads>>>(bptr, cptr, sptr, T, R, L); break;
    }
}

torch::Tensor collide_cuda(torch::Tensor buckets, int64_t R, int64_t hist_threads,
                           int64_t hist_blocks, int64_t gather_threads)
{
    TORCH_CHECK(buckets.is_cuda() && buckets.is_contiguous(),
                "buckets must be a contiguous CUDA tensor [BH, L, T]");
    const auto dt = buckets.scalar_type();
    TORCH_CHECK(dt == torch::kByte || dt == torch::kInt16 || dt == torch::kInt32,
                "buckets must be uint8, int16 or int32; got ", dt);
    // Sub-word buckets halve (uint8: quarter) the traffic on the ONE tensor that
    // dominates it -- read twice and written once, 240 -> 60 MiB at T=128K, L=60,
    // P<=8. rarekv_pack.cu emits these ids natively, so nothing upcasts on the way
    // in. The cast to int in both kernels is free (a widening load).
    // counts and csum stay int32 on purpose: a bucket can hold up to T keys
    // (131072) and csum reaches L*T, both far past int16's 32767.
    TORCH_CHECK(dt != torch::kByte || R <= 256,
                "uint8 buckets require R <= 256 (P <= 8); got R=", R);
    TORCH_CHECK(dt != torch::kInt16 || R <= 32768,
                "int16 buckets require R <= 32768 (P <= 15); got R=", R);
    const int BH = (int)buckets.size(0), L = (int)buckets.size(1), T = (int)buckets.size(2);
    // The histogram privatises R ints per block. Refuse past the device's opt-in
    // shared-memory limit, naming the knob, instead of failing the launch with a
    // bare `invalid argument`: `rarekv_lsh.collision_sums*` checks the same bound
    // and routes past it to the (bit-identical) torch scatter_add_, so this only
    // fires for a direct call to `collide`.
    const int64_t hsmem_cap = (int64_t)at::cuda::getDeviceProperties(
        buckets.device().index())->sharedMemPerBlockOptin;
    TORCH_CHECK((int64_t)R * 4 <= hsmem_cap,
                "the collision histogram needs R*4 = ", R * 4, " B of shared memory, above "
                "this device's ", hsmem_cap, " B opt-in limit. Lower n_planes (R = 2**P) or "
                "use the torch collision path (use_triton=False).");
    auto opt = torch::TensorOptions().dtype(torch::kInt32).device(buckets.device());
    auto counts = torch::zeros({(long)BH * L, (long)R}, opt);
    auto csum = torch::empty({BH, T}, opt);
    int* cptr = counts.data_ptr<int>();
    int* sptr = csum.data_ptr<int>();

    if (dt == torch::kByte)
        launch<uint8_t>((const uint8_t*)buckets.data_ptr<uint8_t>(), cptr, sptr,
                        BH, L, T, (int)R, (int)hist_threads, (int)hist_blocks, (int)gather_threads);
    else if (dt == torch::kInt16)
        launch<int16_t>((const int16_t*)buckets.data_ptr<int16_t>(), cptr, sptr,
                        BH, L, T, (int)R, (int)hist_threads, (int)hist_blocks, (int)gather_threads);
    else
        launch<int>(buckets.data_ptr<int>(), cptr, sptr,
                    BH, L, T, (int)R, (int)hist_threads, (int)hist_blocks, (int)gather_threads);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return csum;
}

// The pybind binding lives in rarekv_module.cpp: this file is now one of several
// sources in the `rarekv_kernels` extension, and only one TU may define the module.

