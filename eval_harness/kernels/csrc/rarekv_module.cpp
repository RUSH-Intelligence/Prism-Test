// The pybind module for the DEFAULT RareKV CUDA path: the collision histogram
// and the Tier-1 bucket packer.
//
// It lives in its own translation unit so each .cu file stays a plain kernel
// file: adding a kernel means adding a source and one m.def(), never touching
// another kernel's PYBIND11_MODULE.
//
// Tier 2 (rarekv_fused.cu) is deliberately NOT here. Its 90 template
// instantiations (15 values of P x 3 head dims x 2 dtypes, each with a fully
// unrolled P*KSTEPS mma loop) dominate the extension's nvcc time, and it is an
// opt-in path. Keeping it in a second, lazily-built extension means a normal
// eval run -- which only ever wants `collide` and `pack_buckets` -- pays a short
// build, and only a run that actually asks for `lsh_mode="fused"` pays the long
// one. See rarekv_fused_module.cpp.
#include <torch/extension.h>

// rarekv_collide.cu
torch::Tensor collide_cuda(torch::Tensor buckets, int64_t R, int64_t hist_threads,
                           int64_t hist_blocks, int64_t gather_threads);
// rarekv_pack.cu
void pack_buckets_into(torch::Tensor proj, torch::Tensor bucket, int64_t row0,
                       int64_t P, int64_t block_m);
torch::Tensor pack_buckets(torch::Tensor proj, int64_t BH, int64_t T,
                           int64_t L, int64_t P, int64_t block_m);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("collide", &collide_cuda, "RareKV collision sums (CUDA)",
          py::arg("buckets"), py::arg("R"), py::arg("hist_threads") = 256,
          py::arg("hist_blocks") = 0, py::arg("gather_threads") = 256);
    m.def("pack_buckets_into", &pack_buckets_into,
          "RareKV Tier 1: proj -> bucket ids, in place with a row offset",
          py::arg("proj"), py::arg("bucket"), py::arg("row0"), py::arg("P"),
          py::arg("block_m") = 0);
    m.def("pack_buckets", &pack_buckets,
          "RareKV Tier 1: proj -> [BH, L, T] bucket ids (allocating wrapper)",
          py::arg("proj"), py::arg("BH"), py::arg("T"), py::arg("L"), py::arg("P"),
          py::arg("block_m") = 0);
}
