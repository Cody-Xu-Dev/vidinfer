"""CUDA 后端：算子注册、固定 Stream 和完成事件；不负责 H3 的采样语义。

已有 C++/CUDA 算子从包内 kernels 导入，构建缓存由 RuntimeConfig 设置。
这里仅覆盖 Attention/QK/SwiGLU；其他网络计算仍由 PyTorch 执行。
"""

from contextlib import contextmanager
from threading import RLock
import torch
from ..registry import Implementation, OperatorRegistry


class CUDACompletion:
    """一个 GPU 完成标记；持有 Event 及其所属 Stream。"""

    def __init__(self, event, stream):
        self.event = event
        self.stream = stream

    def wait(self):
        """阻塞调用线程，直到事件之前排入该 Stream 的工作完成。"""
        self.event.synchronize()

    def ready(self):
        """非阻塞查询设备事件；与 RequestHandle.ready() 的 Future 查询层次不同。"""
        return self.event.query()


class CUDABackend:
    """持有有界的设备执行资源，并向注册表提供真实可用的算子实现。"""

    name = "cuda"

    def __init__(self, device=0, enable_custom=True):
        """准备设备与算子。enable_custom=False 可只注册原生参考实现。"""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        self.device = device
        # 复用固定 Stream，避免每请求创建新 Stream 而保留更多 cuBLAS 工作区。
        self._execution_lock = RLock()
        with torch.cuda.device(device):
            self._stream = torch.cuda.Stream(device=device)
        self.registry = OperatorRegistry()
        from torch.nn.attention import SDPBackend
        from ..kernels.attention import NativeStrideAttention
        from diffusers.models.transformers import transformer_minimax_h3 as vendor

        adapter = NativeStrideAttention(SDPBackend.CUDNN_ATTENTION)

        def reference_qk(q, k, attn, rope):
            """参考路径：保留官方 QK 归一化与 RoPE 的运算/舍入顺序。"""
            return vendor._apply_rotary_emb(
                attn.norm_q(q), *rope
            ), vendor._apply_rotary_emb(attn.norm_k(k), *rope)

        def reference_swiglu(z):
            """输入末维打包为 [a, gate]，按 H3 定义计算 a * SiLU(gate)。"""
            a, b = z.chunk(2, dim=-1)
            return a * torch.nn.functional.silu(b)

        # supports 只检查规格；真正计算由 Implementation.run 持有的函数完成。
        bf16 = lambda s: s.backend == "cuda" and s.dtype == "bf16"
        self.registry.register(
            Implementation(
                "cudnn_bf16",
                "attention",
                "cudnn_baseline",
                lambda s: bf16(s) and s.layout == "BLHD" and not s.get("causal", False),
                adapter,
            )
        )
        self.registry.register(
            Implementation(
                "torch_h3_qk", "qk_norm_rope", "h3_eager_bf16", bf16, reference_qk
            )
        )
        self.registry.register(
            Implementation(
                "torch_swiglu", "swiglu", "eager_bf16", bf16, reference_swiglu
            )
        )
        if enable_custom:
            from ..kernels import qk_norm_rope, swiglu_cuda

            # JIT 构建与首次初始化放在请求之前，编译中间产物与 .so 全部留在 NAS。
            qk_norm_rope.extension()
            swiglu_cuda.extension()
            self.registry.register(
                Implementation(
                    "cuda_h3_qk",
                    "qk_norm_rope",
                    "h3_eager_bf16",
                    lambda s: bf16(s)
                    and s.layout == "BLHD"
                    and s.get("head_dim") == 128
                    and s.get("rotary_dim") == 96
                    and s.get("rope_style") == "half_split",
                    lambda q, k, attn, rope: qk_norm_rope.fused_qk(
                        q, k, attn, rope, True, 4
                    ),
                )
            )
            self.registry.register(
                Implementation(
                    "cuda_swiglu",
                    "swiglu",
                    "eager_bf16",
                    lambda s: bf16(s) and s.layout == "packed_ab_contiguous",
                    swiglu_cuda.fused_swiglu,
                )
            )
        # 运行期间只查询注册表，避免请求之间动态改写实现集合。
        self.registry.seal()

    @contextmanager
    def activate(self, ctx):
        """借出后端 Stream，设置线程局部的设备、推理模式与当前 Stream。

        yield 期间执行 Engine 的 with 代码块，退出后恢复原线程环境。
        后端锁覆盖这段执行，多个调用者不会在此处交错提交。
        """
        with (
            self._execution_lock,
            torch.cuda.device(self.device),
            torch.inference_mode(),
        ):
            stream = self._stream
            ctx.device_context = stream
            with torch.cuda.stream(stream):
                start = torch.cuda.Event(enable_timing=True)
                start.record()
                ctx.resources["start_event"] = start
                ctx.metadata["stream_handle"] = stream.cuda_stream
                yield

    def record_completion(self, ctx):
        """在请求所用 Stream 尾部插入事件；插入本身不等于等待完成。"""
        event = torch.cuda.Event(enable_timing=True)
        event.record(ctx.device_context)
        return CUDACompletion(event, ctx.device_context)

    def drain(self, ctx):
        """事件等待未确认完成时，尝试直接同步该 Stream。"""
        ctx.device_context.synchronize()

    def completed_metadata(self, ctx):
        """两个事件的时间跨度，也包含 CPU 提交/文件读取造成的设备空闲。

        不是 GPU 忙碌时间；逐前向时间另由 forward_ms 记录。
        gpu_elapsed_ms 保留为旧接口别名，含义与 stream_elapsed_ms 相同。
        """
        start = ctx.resources.pop("start_event")
        elapsed = start.elapsed_time(ctx.completion.event)
        return dict(stream_elapsed_ms=elapsed, gpu_elapsed_ms=elapsed)
