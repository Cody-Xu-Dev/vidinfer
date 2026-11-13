# VidInfer

Python 推理调度 + C++/CUDA 热点算子的视频生成研究框架。当前接入 MiniMax-H3 官方 **t2va（文本生成视频与音频）** 流程。

提供独立权重加载、常驻 Session、文本编码与缓存、Transformer 去噪、视频/音频 VAE 解码及 MP4/WAV 导出。目标是把模型语义、设备执行和请求生命周期分开，逐步验证复用于其他视频模型的边界。

这是研究实现，尚无 HTTP 服务、多卡调度、其他视频模型或 NPU 后端。不要把 CUDA 算子注册接口理解为所有网络算子都已具备跨设备实现。

## 安装

核心调度与测试无需 PyTorch：

```bash
python -m pip install .
python -m unittest discover -s tests -v
```

真实推理需要 Linux、NVIDIA GPU、CUDA 编译工具链、ffmpeg/ffprobe，以及完整的官方本地 checkpoint：

```bash
python -m pip install '.[cuda]'
vidinfer doctor
```

实测环境：H20、Python 3.12、PyTorch 2.10.0+cu129、TorchVision 0.25.0+cu129、CUDA Toolkit 12.9、Diffusers 0.41.0、Transformers 5.20.0、Accelerate 1.13.0、Safetensors 0.8.0、NumPy 1.26.4。先安装与驱动兼容且相互匹配的 PyTorch/TorchVision CUDA 构建；`cuda` extra 的版本约束不替代驱动与工具链安装。其他环境尚未验证。

C++/CUDA 源码随 Python 包一起分发，在首次使用自定义算子时 JIT 编译。已有可用构建会复用缓存；这里不是预编译 CUDA wheel。

## 命令行生成

模型权重由使用者自行取得，本仓库不包含或自动下载权重。`--checkpoint` 指向包含 `transformer/`、`text_encoder/`、`vae/`、`audio_vae/`、tokenizer 和 scheduler 配置的官方目录。

```bash
export CUDA_HOME=/usr/local/cuda-12.9
export PATH="$CUDA_HOME/bin:$PATH"

vidinfer generate \
  --checkpoint /mnt/nas/models/MiniMax-H3 \
  --workspace /mnt/nas/vidinfer \
  --prompt 'A pink pig flies through a bright blue sky above fluffy clouds.' \
  --height 544 --width 960 --frames 124 \
  --steps 29 --seed 42 --plan fully_fused
```

容器环境请把 workspace 放在 NAS 或其他容量充足的持久存储上。临时文件、模型卸载文件、文本缓存、扩展构建、输出都会写到显式 workspace，避免使用容器默认缓存目录。

第一次启动需要读取大型权重并准备时间表专用 AdaLN 缓存。可用 `--adaln-cache DIRECTORY` 复用现有兼容缓存；缓存会核对 checkpoint、软件和数值设置身份。

生成产物位于 `workspace/outputs/REQUEST_ID/`，含 `video.mp4`、`audio.wav` 和 ffmpeg 日志。MP4 使用 24fps、H.264 yuv420p 与 AAC 音频。
每个请求的最终状态、输入参数、阶段耗时和错误堆栈记录到 `workspace/requests/REQUEST_ID.json`，不保存 Tensor 到日志。
`stream_elapsed_ms` 是 CUDA 事件间的时间跨度，会包含 CPU 提交和文件读取造成的设备空闲；不是 GPU 忙碌时间。`gpu_elapsed_ms` 是兼容旧接口的同义字段，逐 DiT 前向另见 `forward_ms`。

## 常驻 Python API

```python
from vidinfer import H3Session

with H3Session.from_pretrained(
    checkpoint='/mnt/nas/models/MiniMax-H3',
    workspace='/mnt/nas/vidinfer',
    readers=2,
    max_pending=4,
) as session:
    first = session.generate('A pig flies over white clouds.', seed=42, steps=29)
    print(first.outputs['video_path'])

    # 同一个 Session 复用 DiT、文本编码器和解码器，不重新加载整套模型。
    handle = session.submit('A turtle floats above a sunny meadow.', seed=43)
    print(handle.status())
    second = handle.wait()
    print(second.outputs['video_path'])
```

仅需要去噪结果时传 `output_type='latent'`，输出 CPU `latents` 和 `audio_latents`。已有同进程常驻模型可通过 `H3Session.from_model(...)` 借用；无需也不能跨进程直接引用 Python 模型对象。

## 执行与内存边界

- `PreparedModel` 持有常驻 DiT；执行计划拥有独立模块树与缓存 Router，底层参数/buffer 存储共享。
- 每个请求新建 scheduler、RNG、PipelineState、计时与完成事件。后端复用固定 Stream，请求串行执行。
- `max_pending` 包括运行中和排队请求。达到上限抛出 `QueueFull`，调用者可稍后重试。
- `wait(timeout=...)` 超时只结束等待。`cancel()` 在安全阶段/DiT 边界响应，不能中断已经运行的 Kernel。
- 正常、取消和模型异常都先确认已提交设备工作完成，再清理临时引用。无法确认完成时隔离资源、禁用 Engine，需要进程级故障处理；不会自动重启。
- 文本编码器 BF16 权重约 62GiB，不能与常驻 DiT 全量共存于单张 H20。默认将其映射为最多 40GiB GPU、4GiB CPU，其余磁盘卸载；这些是放置预算，不是整个进程的硬上限。
- 文本特征缓存在 CPU 文件中，键包含 prompt、配置、权重文件元信息与软件版本。Session 内复用组件；首次冷加载与磁盘卸载的成本单独记录。
- 一张设备同一时间使用一个 Session；当前没有多 Session 并发、跨请求批处理或自动显存调度。

## 输入范围

当前支持单请求 batch=1、BF16、单卡 device=0、t2va，无外部参考图/视频、LoRA 或自定义 Attention mask。

- 高宽必须为 32 的正整数倍，宽高比在 1:4 到 4:1 之间。
- 帧数同时满足官方 5–15 秒时长限制和 VAE 对齐形式 `17*n+5`，可用值为 124、141、158……345；不静默修改帧数。22/39 帧虽然符合 VAE 对齐公式，但不符合完整生成流程的最短时长。
- 当前预计算配置为 5 或 29 点采样；前向次数分别为 4 或 28。
- 默认限制视频 Token 数不超过 20000、视频/音频/文本总预算 25000、单帧画布不超过 1048576 像素；prompt 不超过 4000 字符、编码文本不超过 2048 Token，超限会报错。预算用于拒绝明显超量输入，不是显存峰值的精确预测。
- 支持的输入规则不等于所有组合都已通过质量验证；具体回归范围见验证文档。

## 优化计划

| 计划 | Attention | QKV | QK Norm + RoPE | SwiGLU |
|---|---|---|---|---|
| `reference` | cuDNN | 分离投影 | PyTorch | PyTorch |
| `qk_fused` | cuDNN | 共享存储的打包投影 | C++/CUDA | PyTorch |
| `fully_fused` | cuDNN | 共享存储的打包投影 | C++/CUDA | C++/CUDA |

三条路径均使用已校验的 AdaLN 预计算表。注册表按设备、精度、布局和数值语义选择实现；不匹配时只能回退到满足同一约束的实现。

QK 自定义 Kernel 专用于 D=128、RoPE=96、half-split；保留参考路径的 BF16 舍入。SwiGLU 自定义 Kernel 处理打包 `[a,gate]` 的 BF16 输入，计算 `a*SiLU(gate)`。Linear/GEMM 等其余算子由 PyTorch 执行。

## 阅读与验证

- [代码阅读导图](docs/代码阅读导图.md)：按对象、请求、模型、设备的顺序阅读，主要实现附中文注释。
- [验证范围](docs/validation.md)：区分已完成验证与未覆盖能力。
- [优化边界](docs/optimizations.md)：每项优化的位置、适用条件与组合方式。
- `tests/`：无模型测试及微型 ffmpeg 集成测试；后者缺依赖时明确跳过。
- `.github/workflows/core.yml`：Python 3.11–3.13 核心测试与 wheel 内容检查，不执行 GPU 模型。

本仓库不分发模型 checkpoint、预计算张量或生成的视频。MiniMax-H3 模型的使用遵循其官方许可；第三方代码与依赖来源见 [THIRD_PARTY.md](THIRD_PARTY.md)。
