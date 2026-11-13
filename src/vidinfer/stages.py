"""官方文本编码和视频/音频 VAE 阶段；模型在 Session 内复用。

DiT 由 Engine 持有。文本编码器通过 Accelerate 的 GPU/CPU/NAS 磁盘映射加载，
避免 62GiB 文本权重和常驻 DiT 同时全量放进 H20。
"""

from contextlib import ExitStack, contextmanager
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import threading
import time
from unittest.mock import patch

_LOAD_LOCK = threading.RLock()


@contextmanager
def bounded_checkpoint_io():
    """组件加载与磁盘卸载读取都使用 pread，避免 NAS 逐页 mmap 等待。"""
    with _LOAD_LOCK, ExitStack() as stack:
        for name in (
            "safetensors",
            "safetensors.torch",
            "transformers.modeling_utils",
            "accelerate.utils.offload",
        ):
            module = importlib.import_module(name)
            original = module.safe_open

            def pread(*args, _original=original, **kwargs):
                kwargs["backend"] = "pread"
                return _original(*args, **kwargs)

            stack.enter_context(patch.object(module, "safe_open", pread))
        yield


class H3Stages:
    """同一 Session 复用文本编码器/解码器，并缓存 CPU 文本特征。"""

    def __init__(self, checkpoint, config, blocks):
        self.checkpoint = Path(checkpoint)
        self.config = config
        self.blocks = blocks
        self._text = None
        self._decoder = None
        files = (
            "modular_model_index.json",
            "text_encoder/config.json",
            "tokenizer/tokenizer_config.json",
        )
        identity = [
            hashlib.sha256((self.checkpoint / name).read_bytes()).hexdigest()
            for name in files
        ]
        index = json.loads(
            (self.checkpoint / "text_encoder/model.safetensors.index.json").read_text()
        )
        for name in sorted(set(index["weight_map"].values())):
            stat = (self.checkpoint / "text_encoder" / name).stat()
            identity.append((name, stat.st_size, stat.st_mtime_ns))
        import importlib.metadata

        identity += [
            importlib.metadata.version(n)
            for n in ("torch", "diffusers", "transformers")
        ]
        self.revision = hashlib.sha256(json.dumps(identity).encode()).hexdigest()

    def _timed(self, ctx, name, function):
        import torch

        ctx.check_cancelled()
        torch.cuda.current_stream().synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        ctx.metadata["active_stage"] = name
        print("VIDINFER_STAGE_BEGIN", name, flush=True)
        try:
            return function()
        finally:
            # 同步失败时保留 self 的模型拥有者，外层 Engine 会隔离请求。
            torch.cuda.current_stream().synchronize()
            ctx.metadata.setdefault("stages", {})[name] = {
                "wall_s": time.perf_counter() - started,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "allocated_after_bytes": torch.cuda.memory_allocated(),
            }
            print("VIDINFER_STAGE_END", name, ctx.metadata["stages"][name], flush=True)

    def _load_text(self):
        import torch
        from transformers import Qwen3VLForConditionalGeneration

        pipeline = self.blocks["text_encoder"].init_pipeline(str(self.checkpoint))
        offload = self.config.workspace / "offload/text_encoder"
        offload.mkdir(parents=True, exist_ok=True)
        with bounded_checkpoint_io():
            encoder = Qwen3VLForConditionalGeneration.from_pretrained(
                str(self.checkpoint / "text_encoder"),
                local_files_only=True,
                dtype=torch.bfloat16,
                device_map="auto",
                max_memory={
                    self.config.device: self.config.text_gpu_gib * 1024**3,
                    "cpu": self.config.text_cpu_gib * 1024**3,
                },
                offload_folder=str(offload),
                low_cpu_mem_usage=True,
            ).eval()
        # 尽早保存拥有者；后续 tokenizer 加载失败时，也不会无主地遗留 GPU 工作。
        self._text = pipeline
        pipeline.register_components(text_encoder=encoder)
        pipeline.load_components(
            names=["tokenizer", "processor"],
            pretrained_model_name_or_path=str(self.checkpoint),
            local_files_only=True,
        )
        return pipeline

    def encode(self, prompt, ctx):
        import torch

        ctx.retain(self)
        key = hashlib.sha256(
            json.dumps([self.revision, prompt], ensure_ascii=False).encode()
        ).hexdigest()
        folder = self.config.workspace / "cache/text"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / (key + ".pt")
        if path.exists():
            cached = torch.load(path, map_location="cpu", weights_only=True)
            if (
                cached.get("revision") != self.revision
                or cached.get("prompt") != prompt
            ):
                raise RuntimeError("Text cache identity mismatch")
            ctx.metadata["text_cache_hit"] = True
        else:
            if self._text is None:
                self._timed(ctx, "load_text", self._load_text)
            with bounded_checkpoint_io():
                state = self._timed(
                    ctx, "encode_text", lambda: self._text(prompt=prompt)
                )
            cached = dict(
                prompt=prompt,
                revision=self.revision,
                prompt_embeds=state.get("prompt_embeds").detach().cpu(),
                text_token_tags=state.get("text_token_tags").detach().cpu(),
            )
            temporary = path.with_name(path.name + ".tmp-" + ctx.request.request_id)
            torch.save(cached, temporary)
            os.replace(temporary, path)
            ctx.metadata["text_cache_hit"] = False
        embeds, tags = cached["prompt_embeds"], cached["text_token_tags"]
        if embeds.ndim != 3 or embeds.shape[0] != 1 or embeds.shape[-1] != 5120:
            raise RuntimeError("Invalid H3 text feature shape")
        if embeds.shape[1] > 2048 or not torch.isfinite(embeds).all().item():
            raise ValueError("Text feature length/finite-value limit exceeded")
        ctx.metadata["text_tokens"] = embeds.shape[1]
        return dict(prompt=prompt, prompt_embeds=embeds, text_token_tags=tags)

    def _load_decoder(self):
        import torch
        from diffusers.modular_pipelines.modular_pipeline import (
            SequentialPipelineBlocks,
        )

        group = SequentialPipelineBlocks.from_blocks_dict(
            {
                k[len("decode.") :]: v
                for k, v in self.blocks.items()
                if k.startswith("decode.")
            }
        )
        pipeline = group.init_pipeline(str(self.checkpoint))
        self._decoder = pipeline
        with bounded_checkpoint_io():
            pipeline.load_components(
                pretrained_model_name_or_path=str(self.checkpoint),
                local_files_only=True,
                dtype=torch.bfloat16,
                device_map={"vae": "cuda:0", "audio_vae": "cuda:0"},
                low_cpu_mem_usage={"vae": True, "audio_vae": True},
            )
        return pipeline

    def decode(self, state, ctx):
        """在 execute 内提交解码，使 Engine 的最终事件覆盖全部 GPU 阶段。"""
        import numpy as np

        ctx.retain(self)
        if self._decoder is None:
            self._timed(ctx, "load_decoders", self._load_decoder)
        result = self._timed(
            ctx,
            "decode",
            lambda: self._decoder(
                state=state,
                output=["videos", "audio", "sampling_rate"],
                output_type="np",
            ),
        )
        video = np.asarray(result["videos"])
        audio = result["audio"].detach().float().cpu().numpy()
        if not np.isfinite(video).all() or not np.isfinite(audio).all():
            raise RuntimeError("Nonfinite decoded media")
        return dict(video=video, audio=audio, sampling_rate=result["sampling_rate"])

    def close(self):
        """调用者必须先停止 Engine 并确认设备健康，再释放阶段模型。"""
        import torch

        torch.cuda.synchronize()
        for name in ("_text", "_decoder"):
            pipeline = getattr(self, name)
            if pipeline is not None:
                modules = [
                    n
                    for n, component in pipeline.components.items()
                    if isinstance(component, torch.nn.Module)
                ]
                pipeline.unload_components(modules)
                setattr(self, name, None)
        gc.collect()
        torch.cuda.empty_cache()
