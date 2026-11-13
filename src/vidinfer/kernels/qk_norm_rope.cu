#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>

// One warp owns one token/head row. A lane owns four adjacent channels.
// Four or eight independent rows share a block; no block-wide barrier is needed.
template <bool APPLY_NORM, int WARPS>
__global__ void h3_qk_norm_rope_kernel(
    const __nv_bfloat16* q, const __nv_bfloat16* k,
    const __nv_bfloat16* wq, const __nv_bfloat16* wk,
    const float* cosine, const float* sine,
    __nv_bfloat16* outq, __nv_bfloat16* outk,
    int64_t rows, int length, int heads, int rotary,
    int64_t qb, int64_t qs, int64_t qh,
    int64_t kb, int64_t ks, int64_t kh, float epsq, float epsk) {
  constexpr unsigned FULL = 0xffffffffu;
  const int lane = threadIdx.x & 31;
  const int64_t row = int64_t(blockIdx.x) * WARPS + (threadIdx.x >> 5);
  if (row >= rows) return;  // Uniform for all lanes in this warp.
  const bool is_key = blockIdx.y != 0;
  const int head = row % heads;
  const int token = (row / heads) % length;
  const int batch = row / (int64_t(heads) * length);
  const auto* input = is_key ? k : q;
  const auto* weight = is_key ? wk : wq;
  auto* output = is_key ? outk : outq;
  const int64_t offset = is_key ? batch*kb + token*ks + head*kh
                               : batch*qb + token*qs + head*qh;
  float values[4];
  float sum = 0.f;
  #pragma unroll
  for (int j=0; j<4; ++j) {
    values[j] = __bfloat162float(input[offset + lane*4 + j]);
    // Explicit local FP32 FMA follows the vec4 square accumulation contract.
    if constexpr (APPLY_NORM) sum = fmaf(values[j], values[j], sum);
  }
  float inv_rms = 1.f;
  if constexpr (APPLY_NORM) {
    #pragma unroll
    for (int delta=16; delta>0; delta>>=1)
      sum += __shfl_down_sync(FULL, sum, delta);
    sum = __shfl_sync(FULL, sum, 0);
    inv_rms = rsqrtf(sum * (1.f/128.f) + (is_key ? epsk : epsq));
  }
  float normalized[4];
  #pragma unroll
  for (int j=0; j<4; ++j) {
    float value = values[j];
    if constexpr (APPLY_NORM) {
      const float gamma = __bfloat162float(weight[lane*4+j]);
      value = gamma * (inv_rms * value);
    }
    // Preserve the original norm's materialized BF16 result in registers.
    normalized[j] = __bfloat162float(__float2bfloat16_rn(value));
  }
  #pragma unroll
  for (int j=0; j<4; ++j) {
    const int d = lane*4+j;
    const bool rotate = d < rotary;
    const bool first_half = d < rotary/2;
    const int partner = rotate ? (first_half ? d+rotary/2 : d-rotary/2) : d;
    // All lanes participate even for the unrotated tail (channels96..127).
    const float peer = __shfl_sync(FULL, normalized[j], partner/4);
    float result = normalized[j];
    if (rotate) {
      const float c = __bfloat162float(__float2bfloat16_rn(cosine[token*rotary+d]));
      const float s = __bfloat162float(__float2bfloat16_rn(sine[token*rotary+d]));
      const float rotated = first_half ? -peer : peer;
      // Preserve BF16 rounding after EACH multiply; no contracted multiply-add.
      const float left = __bfloat162float(__float2bfloat16_rn(normalized[j]*c));
      const float right = __bfloat162float(__float2bfloat16_rn(rotated*s));
      result = left + right;
    }
    output[row*128+d] = __float2bfloat16_rn(result);
  }
}

std::vector<at::Tensor> h3_qk_norm_rope_cuda(
    const at::Tensor& q, const at::Tensor& k,
    const at::Tensor& wq, const at::Tensor& wk,
    const at::Tensor& cos, const at::Tensor& sin,
    double epsq, double epsk, bool apply_norm, int64_t warps) {
  TORCH_CHECK(q.is_cuda() && q.dim()==4 && q.size(3)==128, "Expected CUDA [B,L,H,128]");
  TORCH_CHECK(q.size(0)>0 && q.size(1)>0 && q.size(2)>0, "Empty inputs unsupported");
  TORCH_CHECK(q.scalar_type()==at::kBFloat16 && k.scalar_type()==at::kBFloat16,
              "Q/K must remain BF16");
  TORCH_CHECK(k.sizes()==q.sizes() && q.stride(3)==1 && k.stride(3)==1,
              "Expected matching Q/K with contiguous head channels");
  for (const auto& t : {k,wq,wk,cos,sin})
    TORCH_CHECK(t.device()==q.device(), "All inputs must share a CUDA device");
  TORCH_CHECK(wq.scalar_type()==at::kBFloat16 && wk.scalar_type()==at::kBFloat16 &&
              wq.numel()==128 && wk.numel()==128 && wq.is_contiguous() && wk.is_contiguous(),
              "Expected contiguous BF16 norm weights[128]");
  TORCH_CHECK(cos.scalar_type()==at::kFloat && sin.scalar_type()==at::kFloat &&
              cos.dim()==2 && cos.sizes()==sin.sizes() && cos.size(0)==q.size(1) &&
              cos.size(1)==96 && cos.is_contiguous() && sin.is_contiguous(),
              "This H3 kernel supports contiguous FP32 cos/sin[L,96]");
  TORCH_CHECK(warps==4 || warps==8, "Use four or eight warps per block");
  TORCH_CHECK(std::isfinite(epsq) && std::isfinite(epsk) && epsq>0 && epsk>0,
              "Expected finite positive norm epsilon");
  TORCH_CHECK(!at::GradMode::is_enabled(), "Inference-only extension; disable autograd");
  c10::cuda::CUDAGuard guard(q.device());
  auto oq = at::empty(q.sizes(),q.options());
  auto ok = at::empty(k.sizes(),k.options());
  const auto rows = q.size(0)*q.size(1)*q.size(2);
  const dim3 grid((rows+warps-1)/warps,2);
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(q.get_device()).stream();
  const auto* qp = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>());
  const auto* kp = reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>());
  const auto* wqp = reinterpret_cast<const __nv_bfloat16*>(wq.data_ptr<at::BFloat16>());
  const auto* wkp = reinterpret_cast<const __nv_bfloat16*>(wk.data_ptr<at::BFloat16>());
  auto* opq = reinterpret_cast<__nv_bfloat16*>(oq.data_ptr<at::BFloat16>());
  auto* opk = reinterpret_cast<__nv_bfloat16*>(ok.data_ptr<at::BFloat16>());
  #define LAUNCH(NORM, W) h3_qk_norm_rope_kernel<NORM,W><<<grid,32*W,0,stream>>>( \
      qp,kp,wqp,wkp,cos.data_ptr<float>(),sin.data_ptr<float>(),opq,opk,rows, \
      q.size(1),q.size(2),96,q.stride(0),q.stride(1),q.stride(2), \
      k.stride(0),k.stride(1),k.stride(2),float(epsq),float(epsk))
  if (apply_norm) {
    if (warps==4) { LAUNCH(true,4); } else { LAUNCH(true,8); }
  } else {
    if (warps==4) { LAUNCH(false,4); } else { LAUNCH(false,8); }
  }
  #undef LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {oq,ok};
}
