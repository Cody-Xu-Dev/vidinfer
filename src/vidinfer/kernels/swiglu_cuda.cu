#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <vector>

__device__ __forceinline__ float rounded_silu(float x) {
  const float y=x/(1.f+expf(-x));
  return __bfloat162float(__float2bfloat16_rn(y));
}

template<bool PAIRS>
__global__ void h3_swiglu_kernel(const __nv_bfloat16* input,__nv_bfloat16* output,
                                int64_t rows,int64_t width) {
  const int64_t row=blockIdx.y+int64_t(blockIdx.z)*65535;
  const int64_t column=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(row>=rows) return;
  if constexpr(PAIRS) {
    if(column>=width/2) return;
    const int64_t base=row*2*width+2*column;
    const auto a=__bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(input+base));
    const auto b=__bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(input+base+width));
    const auto y=__floats2bfloat162_rn(a.x*rounded_silu(b.x),a.y*rounded_silu(b.y));
    *reinterpret_cast<__nv_bfloat162*>(output+row*width+2*column)=y;
  } else {
    if(column>=width) return;
    const int64_t base=row*2*width+column;
    const float a=__bfloat162float(input[base]);
    const float b=__bfloat162float(input[base+width]);
    output[row*width+column]=__float2bfloat16_rn(a*rounded_silu(b));
  }
}

at::Tensor h3_swiglu_cuda(const at::Tensor& packed) {
  TORCH_CHECK(packed.is_cuda() && packed.scalar_type()==at::kBFloat16,
              "Expected CUDA BF16 packed input");
  TORCH_CHECK(packed.dim()>=1 && packed.size(-1)>0 && packed.size(-1)%2==0,
              "Last input dimension must be a positive even number");
  TORCH_CHECK(packed.is_contiguous(),"Packed projection output must be contiguous");
  TORCH_CHECK(!at::GradMode::is_enabled(),"SwiGLU kernel is inference-only");
  c10::cuda::CUDAGuard guard(packed.device());
  auto shape=packed.sizes().vec();
  const int64_t width=shape.back()/2;
  shape.back()=width;
  auto output=at::empty(shape,packed.options());
  const int64_t rows=packed.numel()/(2*width);
  if(rows==0) return output;
  const bool pairs=(width%2==0 && reinterpret_cast<uintptr_t>(packed.data_ptr())%4==0);
  const int64_t columns=pairs ? width/2 : width;
  TORCH_CHECK((columns+255)/256<=2147483647 && (rows+65534)/65535<=65535,
              "Input exceeds CUDA launch dimensions");
  const dim3 grid((columns+255)/256,rows<65535?rows:65535,(rows+65534)/65535);
  const auto stream=c10::cuda::getCurrentCUDAStream(packed.get_device()).stream();
  const auto* x=reinterpret_cast<const __nv_bfloat16*>(packed.data_ptr<at::BFloat16>());
  auto* y=reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>());
  if(pairs) h3_swiglu_kernel<true><<<grid,256,0,stream>>>(x,y,rows,width);
  else h3_swiglu_kernel<false><<<grid,256,0,stream>>>(x,y,rows,width);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
