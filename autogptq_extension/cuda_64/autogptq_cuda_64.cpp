#include <torch/all.h>
#include <torch/python.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

void vecquant3matmul_faster_cuda_old(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  int64_t groupsize, int64_t vec_height
); 

void vecquant3matmul_faster_old(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  int64_t groupsize, int64_t vec_height
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant3matmul_faster_cuda_old(vec, mat, mul, scales, zeros, groupsize, vec_height);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vecquant3matmul_faster_old_64", &vecquant3matmul_faster_old, "Vector 3-bit Quantized Matrix Multiplication (CUDA), faster version");
}

TORCH_LIBRARY_IMPL(autogptq, CUDA, m) {
  m.impl("autogptq::vecquant3matmul_faster_old_64", &vecquant3matmul_faster_old);
}