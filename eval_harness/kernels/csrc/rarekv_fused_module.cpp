// The pybind module for the OPT-IN RareKV Tier-2 kernels.
//
// Separate from rarekv_module.cpp on purpose: rarekv_fused.cu instantiates
// 15 (P) x 3 (head dim) x 2 (dtype) = 90 kernels, each with a fully unrolled
// P*KSTEPS mma loop, and dominates nvcc time for the whole extension. Splitting
// it out keeps the DEFAULT path's build short -- a normal eval run wants only
// `collide` and `pack_buckets` and never touches this module.
#include <torch/extension.h>

// rarekv_fused.cu
torch::Tensor fused_buckets(torch::Tensor keys, torch::Tensor planes_t,
                            int64_t L, int64_t P, int64_t block_m);
torch::Tensor serial_buckets(torch::Tensor keys, torch::Tensor planes_t,
                             int64_t L, int64_t P);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_buckets", &fused_buckets,
          "RareKV Tier 2: keys x planes^T -> [BH, L, T] bucket ids, proj never materialised",
          py::arg("keys"), py::arg("planes_t"), py::arg("L"), py::arg("P"),
          py::arg("block_m") = 0);
    m.def("serial_buckets", &serial_buckets,
          "RareKV debug oracle: scalar strictly-ascending-k dot, no tensor cores",
          py::arg("keys"), py::arg("planes_t"), py::arg("L"), py::arg("P"));
}
