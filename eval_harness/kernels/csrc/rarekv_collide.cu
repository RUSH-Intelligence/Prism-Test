// RareKV collision counting, CUDA.
//
// Layout is [BH, L, T] (L-major, T-minor), copied from SOCKET's
// soft_hash_score.cu: for a fixed table l, consecutive threads read consecutive
// keys, so every bucket load is fully coalesced. The [BH, T, L] layout the torch
// path uses makes a 32-thread warp span L*4*32 bytes instead of 4 cache lines.
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

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

    int hb = hist_blocks > 0 ? hist_blocks : std::min(cdiv(T, hist_threads), 512);
    dim3 hgrid(hb, BH * L);
    hist_kernel<BT><<<hgrid, hist_threads, R * sizeof(int)>>>(bptr, cptr, T, R);

    dim3 ggrid(cdiv(T, gather_threads), BH);
    switch (L) {
        case 40: gather_kernel_t<BT,40><<<ggrid, gather_threads>>>(bptr, cptr, sptr, T, R); break;
        case 50: gather_kernel_t<BT,50><<<ggrid, gather_threads>>>(bptr, cptr, sptr, T, R); break;
        case 60: gather_kernel_t<BT,60><<<ggrid, gather_threads>>>(bptr, cptr, sptr, T, R); break;
        case 80: gather_kernel_t<BT,80><<<ggrid, gather_threads>>>(bptr, cptr, sptr, T, R); break;
        default: gather_kernel_dyn<BT><<<ggrid, gather_threads>>>(bptr, cptr, sptr, T, R, L); break;
    }
}

torch::Tensor collide_cuda(torch::Tensor buckets, int64_t R, int64_t hist_threads,
                           int64_t hist_blocks, int64_t gather_threads)
{
    TORCH_CHECK(buckets.is_cuda() && buckets.is_contiguous(),
                "buckets must be a contiguous CUDA tensor [BH, L, T]");
    const auto dt = buckets.scalar_type();
    TORCH_CHECK(dt == torch::kInt16 || dt == torch::kInt32,
                "buckets must be int16 or int32; got ", dt);
    // int16 buckets (SOCKET's choice) halve the traffic on the ONE tensor that
    // dominates it -- read twice and written once, 240 -> 120 MiB at T=128K,L=60.
    // counts and csum stay int32 on purpose: a bucket can hold up to T keys
    // (131072) and csum reaches L*T, both far past int16's 32767.
    TORCH_CHECK(dt != torch::kInt16 || R <= 32768,
                "int16 buckets require R <= 32768 (P <= 15); got R=", R);
    const int BH = (int)buckets.size(0), L = (int)buckets.size(1), T = (int)buckets.size(2);
    auto opt = torch::TensorOptions().dtype(torch::kInt32).device(buckets.device());
    auto counts = torch::zeros({(long)BH * L, (long)R}, opt);
    auto csum = torch::empty({BH, T}, opt);
    int* cptr = counts.data_ptr<int>();
    int* sptr = csum.data_ptr<int>();

    if (dt == torch::kInt16)
        launch<int16_t>((const int16_t*)buckets.data_ptr<int16_t>(), cptr, sptr,
                        BH, L, T, (int)R, (int)hist_threads, (int)hist_blocks, (int)gather_threads);
    else
        launch<int>(buckets.data_ptr<int>(), cptr, sptr,
                    BH, L, T, (int)R, (int)hist_threads, (int)hist_blocks, (int)gather_threads);
    return csum;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("collide", &collide_cuda, "RareKV collision sums (CUDA)",
          py::arg("buckets"), py::arg("R"), py::arg("hist_threads") = 256,
          py::arg("hist_blocks") = 0, py::arg("gather_threads") = 256);
}
