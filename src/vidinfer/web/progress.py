"""把请求状态转为纯 JSON；CUDA Event 只 query，不在轮询中等待设备。"""

import json
import os
from pathlib import Path
import time
import uuid


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def handle_progress(handle):
    snapshot = handle.status()
    if (
        hasattr(handle, "ready")
        and not handle.ready()
        and snapshot["phase"] in ("completed", "failed", "cancelled")
    ):
        snapshot["phase"] = (
            "cancelling" if snapshot["phase"] == "cancelled" else "finishing"
        )
    ctx = handle.context
    previous = list(ctx.metadata.get("observed_forward_ms", []))
    values = list(ctx.metadata.get("forward_ms", previous))
    # b 在 post hook 记录后才计入 steps_done，避免把未物化 Event 的 query=True 当完成。
    events = tuple(ctx.resources.get("forward_events", ()))[: snapshot["steps_done"]]
    if events:
        values = []
        for start, end in events:
            try:
                if not end.query():
                    break
                values.append(start.elapsed_time(end))
            except RuntimeError as error:
                snapshot["gpu_timing_error"] = str(error)
                break
    if len(values) < len(previous):
        values = previous
    ctx.metadata["observed_forward_ms"] = list(values)
    snapshot.update(
        gpu_forward_ms=values,
        gpu_completed_forwards=len(values),
        total_forwards=ctx.request.steps - 1,
        text_cache_hit=ctx.metadata.get("text_cache_hit"),
        stage_metrics={
            k: dict(v) for k, v in tuple(ctx.metadata.get("stages", {}).items())
        },
        updated_at=time.time(),
    )
    return snapshot


def monitor(handle, folder):
    """独立 worker 的生命周期内轮询句柄，及时写 NAS；返回前确认 Future 完成。"""
    folder = Path(folder)
    while not handle.ready():
        if (folder / "cancel.request").exists():
            handle.cancel()
        atomic_json(folder / "live.json", handle_progress(handle))
        time.sleep(0.25)
    try:
        result = handle.wait()
        outputs = {k: v for k, v in result.outputs.items() if not hasattr(v, "shape")}
        atomic_json(
            folder / "result.json",
            dict(
                request_id=result.request_id,
                outputs=outputs,
                metadata=dict(result.metadata),
            ),
        )
        return result
    finally:
        atomic_json(folder / "live.json", handle_progress(handle))
