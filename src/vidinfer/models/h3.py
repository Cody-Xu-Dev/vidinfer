"""H3 模型适配：配置检查、执行计划、官方去噪流程与输出整理。

建议先读 H3Adapter 的 adopt/prepare_plan/execute/finalize，再回看上方工具函数。
这里复用 Diffusers 的 H3 网络和采样流程；只替换已验证的热点算子。
"""

import copy
import hashlib
import json
import time
from pathlib import Path
import torch
from torch import nn
from diffusers import ModularPipeline
from diffusers.modular_pipelines.modular_pipeline import (
    SequentialPipelineBlocks,
    PipelineState,
)
from diffusers import MiniMaxH3Scheduler
from diffusers.models.transformers import transformer_minimax_h3 as vendor
from ..types import PreparedModel, ExecutionPlan
from ..registry import OperatorSpec


def tensor_hash(t):
    """按连续的原始字节计算 SHA256；会回传到 CPU，因此属于诊断开销。"""
    return hashlib.sha256(
        memoryview(t.detach().contiguous().cpu().view(torch.uint8).numpy())
    ).hexdigest()


def storage_map(model):
    """按底层显存地址去重，核对计划是否共享同一份参数与 buffer 存储。"""
    result = {}
    for t in list(model.parameters()) + list(model.buffers()):
        if t.device.type == "cuda":
            storage = t.untyped_storage()
            result[(t.device.index, storage.data_ptr())] = storage.nbytes()
    return result


def source_signature(model):
    """记录处理器身份和 forward 覆写情况，用于检查是否改动了源模型绑定。"""
    return [
        (name, id(m.processor), bool(m.fused_projections))
        for name, m in model.named_modules()
        if isinstance(m, vendor.MiniMaxH3Attention)
    ], [
        (name, "forward" in m.__dict__)
        for name, m in model.named_modules()
        if m.__class__.__name__ == "SwiGLU"
    ]


def clone_module_tree(source, memo=None):
    """复制 H3 模块结构与注册字典，参数和 buffer Tensor 仍共享。

    copy.copy 是浅拷贝；下面再递归复制子模块，让各计划能独立设置 processor。
    memo 保持原图中的重复子模块关系，避免同一模块被复制成多个独立对象。
    此工具只适配当前 H3；实例级 forward 覆写、Accelerate hook 会直接拒绝。
    """
    memo = {} if memo is None else memo
    if id(source) in memo:
        return memo[id(source)]
    if "forward" in source.__dict__ or getattr(source, "_hf_hook", None) is not None:
        raise ValueError(
            "H3 plan preparation requires unpatched, resident inference modules"
        )
    clone = copy.copy(source)
    memo[id(source)] = clone
    # 复制容器，不复制 Tensor；这与复制另一份模型权重有本质区别。
    clone._parameters = source._parameters.copy()
    clone._buffers = source._buffers.copy()
    clone._non_persistent_buffers_set = source._non_persistent_buffers_set.copy()
    clone._modules = {
        k: None if v is None else clone_module_tree(v, memo)
        for k, v in source._modules.items()
    }
    for name, value in source.__dict__.items():
        if "hook" in name and isinstance(value, (dict, set)):
            setattr(clone, name, value.copy())
    return clone


class H3AttentionProcessor:
    """一次 Attention 的算子连接：投影 → 分头 → QK 处理 → Attention → 输出投影。"""

    def __init__(self, attention, qk):
        self.attention = attention
        self.qk = qk

    def __call__(self, attn, x, rotary_emb=None, attention_mask=None):
        """x 是 [B, L, 模型宽度]；Q/K/V 分头后是 [B, L, H, D]。

        B=批大小，L=Token 数，H=头数，D=每头维度；本次主干 H=56、D=128。
        Q/K/V 投影后的宽度 H*D 不要求等于输入模型宽度。
        """
        if attention_mask is not None:
            raise ValueError("H3 v0 supports unmasked t2va only")
        if attn.fused_projections:
            # 一次打包投影后沿末维切出 Q/K/V；不在这里重复打包权重。
            q, k, v = attn.to_qkv(x).chunk(3, dim=-1)
        else:
            q, k, v = attn.to_q(x), attn.to_k(x), attn.to_v(x)
        q, k, v = (t.unflatten(-1, (attn.heads, -1)) for t in (q, k, v))
        if rotary_emb is None:
            # 文本 refiner 没有传入 RoPE，保留其原生 QK 归一化路径。
            q, k = attn.norm_q(q), attn.norm_k(k)
        else:
            # 主干根据计划选择 PyTorch 或融合 CUDA QK Norm + RoPE。
            q, k = self.qk.run(q, k, attn, rotary_emb)
        y = self.attention.run(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
        # [B,L,H,D] 合回 [B,L,H*D]，再执行官方输出投影与 dropout 模块。
        return attn.to_out[1](attn.to_out[0](y.flatten(2, 3).type_as(q)))


class FusedSwiGLU(nn.Module):
    """保留原投影层，仅把投影后的门控激活与逐元素乘法交给融合实现。"""

    def __init__(self, projection, implementation):
        super().__init__()
        self.proj = projection
        self.implementation = implementation

    def forward(self, x):
        return self.implementation.run(self.proj(x))


class H3Adapter:
    """模型层入口：把通用请求转成当前官方 H3 去噪 Pipeline 的调用。"""

    # 三个布尔值依次表示：打包 QKV、融合 QK、融合 SwiGLU。
    presets = {
        "reference": (False, False, False),
        "qk_fused": (True, True, False),
        "fully_fused": (True, True, True),
    }

    def __init__(self, checkpoint, text_cache=None, config=None):
        """读取已缓存的文本条件与 Pipeline 配置；不会在这里加载 DiT 权重。"""
        self.checkpoint = Path(checkpoint)
        self.condition = (
            torch.load(text_cache, map_location="cpu", weights_only=True)
            if text_cache
            else None
        )
        self.prompt = self.condition["prompt"] if self.condition else None
        self.text_hash = (
            tensor_hash(self.condition["prompt_embeds"]) if self.condition else None
        )
        self.config = config
        self.stages = None
        base = ModularPipeline.from_pretrained(
            str(checkpoint), workflow="t2va", local_files_only=True
        )
        self.group = SequentialPipelineBlocks.from_blocks_dict(
            {
                k[len("denoise.") :]: v
                for k, v in base.blocks.sub_blocks.items()
                if k.startswith("denoise.")
            }
        )
        if config is not None:
            from ..stages import H3Stages

            self.stages = H3Stages(checkpoint, config, base.blocks.sub_blocks)

    def adopt(self, model, model_id):
        """接管外部传入模型的引用；权重仍由原常驻进程持有，不重新加载。"""
        if model.training:
            raise ValueError("PreparedModel requires eval mode")
        if not getattr(model, "_resident_qkv_shared", False):
            raise ValueError(
                "Prepare shared QKV storage once before adopting the resident model"
            )
        if any(p.device.type != "cuda" for p in model.parameters()):
            raise ValueError("Weights must remain resident on CUDA")
        # 准备阶段确认已有权重就绪；后续请求在后端的专用 Stream 上执行。
        torch.cuda.synchronize()
        revision = hashlib.sha256(
            json.dumps(dict(model.config), sort_keys=True, default=str).encode()
        ).hexdigest()
        return PreparedModel(model_id, revision, "cuda", model)

    def validate(self, r):
        """只允许当前验证域；新 prompt/形状/时间表需要先补齐缓存和回归。"""
        if not isinstance(r.prompt, str) or not r.prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if self.stages is None and r.prompt != self.prompt:
            raise ValueError("No cached text features for this prompt in H3 v0")
        if self.config is None and (r.height, r.width, r.frames) != (544, 960, 124):
            raise ValueError("H3 v0 validation domain: height544,width960,frames124")
        if self.config is not None:
            from ..validation import validate_generation_request

            validate_generation_request(r, self.config)
        if r.steps not in (5, 29):
            raise ValueError("No matching AdaLN schedule profile")
        if not isinstance(r.seed, int) or not 0 <= r.seed < 2**63:
            raise ValueError("Invalid seed")
        if r.output_type not in ("latent", "video"):
            raise ValueError("output_type must be latent or video")
        if r.output_type == "video" and self.stages is None:
            raise ValueError("Video output requires RuntimeConfig and H3Stages")

    def prepare_plan(self, prepared, name, backend):
        """按配置选算子，创建私有模块视图；不在共享源模型上替换 forward。"""
        if name not in self.presets:
            raise ValueError("Unknown execution plan: " + name)
        packed, custom_qk, custom_swi = self.presets[name]
        # 1. 声明需要的设备、精度、布局、形状约束及数值语义。
        qkspec = OperatorSpec(
            "qk_norm_rope",
            "cuda",
            "bf16",
            "BLHD",
            (("head_dim", 128), ("rotary_dim", 96), ("rope_style", "half_split")),
            "h3_eager_bf16",
        )
        swispec = OperatorSpec(
            "swiglu", "cuda", "bf16", "packed_ab_contiguous", (), "eager_bf16"
        )
        attspec = OperatorSpec(
            "attention", "cuda", "bf16", "BLHD", (), "cudnn_baseline"
        )
        attention, d1 = backend.registry.resolve(attspec, ("cudnn_bf16",))
        # d1/d2/d3 保存候选实现被拒绝的原因，便于确认有没有发生 fallback。
        qk, d2 = backend.registry.resolve(
            qkspec, ("cuda_h3_qk",) if custom_qk else ("torch_h3_qk",), ("torch_h3_qk",)
        )
        swi, d3 = backend.registry.resolve(
            swispec,
            ("cuda_swiglu",) if custom_swi else ("torch_swiglu",),
            ("torch_swiglu",),
        )
        source = prepared.payload
        # 2. 只接受我们了解的缓存 hook，避免把未知的共享状态带入新计划。
        for fn in list(source._forward_pre_hooks.values()) + list(
            source._forward_hooks.values()
        ):
            if getattr(fn, "__self__", None) is not source._adaln_cache_router:
                raise ValueError(
                    "Unexpected source hooks; cannot adopt them as shared plan state"
                )
        view = clone_module_tree(source)
        # 3. Router 记录当前 timestep 对应的预计算表键；每个计划独立拥有。
        # 表的 GPU Tensor 仍通过浅拷贝共享，不会为每个 Router 重建一套表。
        source_router = source._adaln_cache_router
        router = type(source_router)(source_router.keys)
        view._adaln_cache_router = router
        for m in view.modules():
            if getattr(m, "router", None) is source_router:
                m.router = router
        for attr in (
            "_forward_pre_hooks",
            "_forward_hooks",
            "_forward_pre_hooks_with_kwargs",
            "_forward_hooks_with_kwargs",
            "_forward_hooks_always_called",
        ):
            getattr(view, attr).clear()
        view.register_forward_pre_hook(router.pre, with_kwargs=True)
        view.register_forward_hook(router.post, with_kwargs=True, always_call=True)
        main_ids = {id(b.attn) for b in view.transformer_blocks}
        # 4. 在私有视图里装处理器；文本 refiner 不启用主干的打包 QKV。
        for m in view.modules():
            if isinstance(m, vendor.MiniMaxH3Attention):
                m.fused_projections = packed if id(m) in main_ids else False
                m.set_processor(H3AttentionProcessor(attention, qk))
        if custom_swi and swi.name == "cuda_swiglu":
            # 只有注册表确实选中了自定义实现时才替换主干 FFN 的激活模块。
            for b in view.transformer_blocks:
                b.ff.net[0] = FusedSwiGLU(b.ff.net[0].proj, swi)
        view.eval()
        if storage_map(view) != storage_map(source):
            raise RuntimeError("Plan copied model storage")
        return ExecutionPlan(
            name,
            prepared.model_id,
            prepared.revision,
            dict(attention=attention.name, qk_norm_rope=qk.name, swiglu=swi.name),
            view,
            d1 + d2 + d3,
        )

    def execute(self, prepared, plan, ctx):
        """执行一次完整去噪；设备完成等待由外层 Engine 负责。

        prepared 是统一 Adapter 接口的一部分；本方法实际使用计划中的模型视图。
        每次新建采样状态，避免不同 seed/请求复用同一个采样器的可变状态。
        """
        with ctx.stage("text_condition"):
            condition = self.condition
            if condition is None or ctx.request.prompt != self.prompt:
                condition = self.stages.encode(ctx.request.prompt, ctx)
            else:
                ctx.metadata["text_cache_hit"] = True
        # 1. 创建请求专用 Pipeline 外壳，注入已经准备好的模型视图。
        pipeline = self.group.init_pipeline(str(self.checkpoint))
        pipeline.register_components(
            transformer=plan.payload,
            scheduler=MiniMaxH3Scheduler.from_pretrained(
                str(self.checkpoint), subfolder="scheduler", local_files_only=True
            ),
            audio_scheduler=MiniMaxH3Scheduler.from_pretrained(
                str(self.checkpoint), subfolder="audio_scheduler", local_files_only=True
            ),
        )
        if hasattr(pipeline, "set_progress_bar_config"):
            pipeline.set_progress_bar_config(disable=True)
        # 2. 文本特征上 GPU；text_token_tags 等辅助条件按官方流程保持在 CPU。
        state = PipelineState()
        for k, v in condition.items():
            state.set(k, v.cuda() if k == "prompt_embeds" else v)
        # 使用请求专用 CPU Generator；官方 Pipeline 决定噪声的生成和搬运方式。
        generator = torch.Generator().manual_seed(ctx.request.seed)
        ctx.retain(pipeline, state, generator)
        ctx.metadata.update(
            text_features_sha256=tensor_hash(condition["prompt_embeds"])
        )
        ctx.resources.update(forward_events=[], initial_noise={})

        def pre(model, pos, kw):
            """DiT 前向之前：检查取消、记录事件，并在首轮保存初始噪声快照。"""
            ctx.check_cancelled()
            a, b = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            a.record()
            ctx.resources["forward_events"].append((a, b))
            if len(ctx.resources["forward_events"]) == 1:
                # clone 真正复制数据，防止后续采样更新影响诊断快照。
                # 复制发生在起点事件之后，首轮前向计时包含这部分诊断开销。
                ctx.resources["initial_noise"] = {
                    k: kw[k].detach().clone()
                    for k in ("hidden_states", "audio_hidden_states")
                }

        def post(model, pos, out):
            """DiT 的 Python forward 返回后记录终点；GPU 此时不一定已经完成。"""
            ctx.resources["forward_events"][-1][1].record()
            ctx.step_finished()

        # 3. 每次请求注册自己的 hook；prepend 让取消检查先于已有前置 hook。
        handles = [
            plan.payload.register_forward_pre_hook(pre, with_kwargs=True, prepend=True),
            plan.payload.register_forward_hook(post),
        ]
        try:
            # 4. 进入官方采样循环：准备噪声 → 多次 DiT/采样更新 → 输出 latent。
            r = ctx.request
            ctx.metadata["active_stage"] = "denoise"
            with ctx.stage("denoise"):
                raw = pipeline(
                    state=state,
                    height=r.height,
                    width=r.width,
                    num_frames=r.frames,
                    num_inference_steps=r.steps,
                    generator=generator,
                )
                # 仅在完整去噪边界等待；不在每个 DiT 前向同步。
                # 使这个阶段的 Wall 截止于设备完成，供实时界面明确展示。
                torch.cuda.current_stream().synchronize()
            if r.output_type == "video":
                ctx.resources["decoded"] = self.stages.decode(raw, ctx)
            return raw
        finally:
            # 5. 无论成功、取消还是异常，都卸下本次 hook，避免下一次重复触发。
            # 移除 hook 不是 GPU 同步；Engine 仍会等待已提交工作的完成事件。
            for handle in handles:
                handle.remove()

    def finalize(self, raw, ctx):
        """Engine 确认设备完成后调用：回传 CPU latent，核对有限值并整理报告。"""
        outputs = {k: raw.get(k).detach().cpu() for k in ("latents", "audio_latents")}
        if not all(torch.isfinite(t).all().item() for t in outputs.values()):
            raise RuntimeError("Nonfinite latent")
        events = ctx.resources.pop("forward_events")
        noise = ctx.resources.pop("initial_noise")
        ctx.metadata.update(
            # elapsed_time 读取两个 CUDA Event 之间的设备时间，单位为毫秒。
            forward_ms=[a.elapsed_time(b) for a, b in events],
            initial_noise_sha256={k: tensor_hash(v) for k, v in noise.items()},
            latent_sha256={k: tensor_hash(v) for k, v in outputs.items()},
        )
        if "decoded" in ctx.resources:
            from ..media import write_media

            ctx.metadata["active_stage"] = "export"
            started = time.perf_counter()
            with ctx.stage("export"):
                media = write_media(
                    ctx.resources.pop("decoded"),
                    self.config.output_directory(ctx.request.request_id),
                )
            outputs.update(media)
            ctx.metadata.setdefault("stages", {})["export"] = {
                "wall_s": time.perf_counter() - started
            }
        return outputs
