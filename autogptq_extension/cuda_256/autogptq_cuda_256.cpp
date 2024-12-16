#include <torch/all.h>
#include <torch/python.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

torch::Tensor vecquant3matmul_faster_cuda_old(
  torch::Tensor vec, torch::Tensor mat,
  torch::Tensor scales, torch::Tensor zeros,
  int64_t groupsize, int64_t vec_height
); 

torch::Tensor vecquant3matmul_faster_old(
  torch::Tensor vec, torch::Tensor mat,
  torch::Tensor scales, torch::Tensor zeros,
  int64_t groupsize, int64_t vec_height
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  return vecquant3matmul_faster_cuda_old(vec, mat, scales, zeros, groupsize, vec_height);
}


// TORCH_LIBRARY(autogptq, m) {
//   m.def("vecquant3matmul_faster_old_256(Tensor vec, Tensor mat, Tensor scales, Tensor zeros, int groupsize, int vec_height) -> Tensor");
// }

TORCH_LIBRARY_IMPL(autogptq, CUDA, m) {
  m.impl("autogptq::vecquant3matmul_faster_old_256", &vecquant3matmul_faster_old);
}