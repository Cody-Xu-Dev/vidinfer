# 已接入的优化

这些能力来自逐项实验，最终通过执行计划组合。本文描述实现边界，不把单算子加速等同于整模型加速。

| 优化 | 实现位置 | 约束与处理方式 |
|---|---|---|
| cuDNN 原始布局 Attention | `kernels/attention.py` | 通过维度视图传入支持的 stride，返回 BLHD 时包含必要的连续化；强制指定 cuDNN |
| AdaLN 预计算 | `loading/adaln.py` | 按 checkpoint/软件/数值设置与时间表校验；推理时按 timestep 键查表 |
| QKV 打包与共享存储 | `loading/prepare.py` | 用打包投影替代三次独立投影；分离权重改为共享切片，避免持久权重翻倍 |
| QK RMSNorm + RoPE 融合 | `kernels/qk_norm_rope.*` | 一 Warp 处理一个128维 Head，前96维 half-split RoPE，保留原 BF16 舍入 |
| SwiGLU 融合 | `kernels/swiglu_cuda.*` | 保留投影 GEMM，把 SiLU 与逐元素乘法合并为自定义 Kernel |
| 文本条件缓存 | `stages.py` | prompt 与模型/软件身份匹配才复用；缓存 CPU 特征，不缓存不同 seed 的去噪结果 |
| 常驻模型与固定 Stream | `session.py`、`backends/cuda.py` | 同一 Session 复用模型；Stream 有界，避免每请求创建额外工作区 |

前三种计划都固定 cuDNN 和 BF16。QK 与 SwiGLU 可以单独启用，注册表拒绝与要求数值语义不同的 fallback。

QKV 打包单独存在时，收益受 GEMM 形状和库算法选择影响，不能承诺所有输入更快。AdaLN 表仅用于冻结模型、无 LoRA 的已准备时间表；更改模型或数值设置需要重新校验/构建。

框架仍调用 PyTorch 实现 Linear/GEMM、embedding 与其余 norm。适配新芯片需要继续补齐设备 Tensor、执行语义和算子覆盖，不是只换 Attention 就完成跨芯片支持。
