"""面向使用者的 H3 Session：加载一次，连续提交不同 prompt。"""

from pathlib import Path
from dataclasses import asdict
import json
import logging
import os
import uuid
from .runtime import RuntimeConfig
from .types import DenoiseRequest


class H3Session:
    """持有 Engine 与阶段模型；支持自有模型和外部常驻模型两种所有权。"""

    def __init__(self, checkpoint, config, model, model_id=None, text_cache=None):
        config.configure()
        from .backends.cuda import CUDABackend
        from .models.h3 import H3Adapter
        from .loading.prepare import pack_qkv_shared
        from .engine import Engine

        self.config = config
        self.checkpoint = Path(checkpoint).resolve()
        pack_qkv_shared(model)
        self.backend = CUDABackend(device=config.device)
        self.adapter = H3Adapter(self.checkpoint, text_cache, config=config)
        self.prepared = self.adapter.adopt(model, model_id or uuid.uuid4().hex)
        self.engine = Engine(
            self.prepared, self.adapter, self.backend, max_pending=config.max_pending
        )
        # 准备阶段建立所有计划；submit 切换计划时不再等待正在运行的请求来构建视图。
        for name in self.adapter.presets:
            self.engine.plan(name)

    @classmethod
    def from_pretrained(cls, checkpoint, workspace, text_cache=None, **runtime_options):
        """独立启动：准备/复用 AdaLN 缓存，以有界主存加载 DiT，再常驻复用。"""
        config = RuntimeConfig(workspace=workspace, **runtime_options).configure()
        from .loading.prepare import load_prepared_model

        model = load_prepared_model(checkpoint, workers=config.readers)
        return cls(checkpoint, config, model, text_cache=text_cache)

    @classmethod
    def from_model(
        cls,
        model,
        checkpoint,
        workspace,
        model_id=None,
        text_cache=None,
        **runtime_options,
    ):
        """借用当前进程已有模型；无法通过此函数引用另一进程的 Python 对象。"""
        return cls(
            checkpoint,
            RuntimeConfig(workspace=workspace, **runtime_options),
            model,
            model_id=model_id,
            text_cache=text_cache,
        )

    def submit(
        self,
        prompt,
        *,
        plan="fully_fused",
        seed=42,
        height=544,
        width=960,
        frames=124,
        steps=29,
        output_type="video",
        on_step=None,
    ):
        if self.engine is None:
            raise RuntimeError("Session is closed")
        request = DenoiseRequest(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            frames=frames,
            steps=steps,
            output_type=output_type,
        )
        handle = self.engine.submit(request, plan=plan, on_step=on_step)
        workspace = self.config.workspace
        # 回调只持有轻量配置和句柄，不额外持有整个 Session。
        handle._future.add_done_callback(
            lambda future: _write_request_record(handle, workspace)
        )
        return handle

    def generate(self, prompt, **options):
        """同步生成；异步使用 submit() 返回的 ready/wait/cancel 句柄。"""
        return self.submit(prompt, **options).wait()

    def close(self):
        if self.engine is None:
            return
        engine = self.engine
        engine.close()
        if not engine._healthy:
            raise RuntimeError("Device unhealthy: retain models and restart process")
        self.adapter.stages.close()
        self.engine = None
        self.prepared = None
        self.adapter = None
        self.backend = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _write_request_record(handle, workspace):
    """记录正常、失败、取消的最终状态；日志写入失败不覆盖推理的原始结果。"""
    context = handle.context
    report = dict(
        request=asdict(context.request),
        status=context.snapshot(),
        started_at=context.started_at,
        finished_at=context.finished_at,
        metadata=dict(context.metadata),
    )
    folder = workspace / "requests"
    try:
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / (context.request.request_id + ".json")
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    except OSError:
        logging.getLogger(__name__).exception("Cannot write request record")
