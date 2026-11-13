#include <torch/extension.h>
#include <vector>

std::vector<at::Tensor> h3_qk_norm_rope_cuda(
    const at::Tensor& q, const at::Tensor& k,
    const at::Tensor& wq, const at::Tensor& wk,
    const at::Tensor& cos, const at::Tensor& sin,
    double epsq, double epsk, bool apply_norm, int64_t warps);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("forward", &h3_qk_norm_rope_cuda,
             "Inference-only BF16 H3 QK RMSNorm and partial RoPE (CUDA)");
}
