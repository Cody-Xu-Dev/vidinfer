#include <torch/extension.h>
at::Tensor h3_swiglu_cuda(const at::Tensor& packed);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,module) {
  module.def("forward",&h3_swiglu_cuda,"Inference-only BF16 SwiGLU on packed CUDA input");
}
