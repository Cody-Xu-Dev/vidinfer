"""可选采集入口：只在实际执行同步 callable 的线程内打开 profiler。"""

from contextlib import contextmanager
from pathlib import Path
import json
import time
import sys


@contextmanager
def capture_profile(output, *, label="vidinfer/call", cuda=True):
    """用于算子/Block 调用。不要仅在主线程包住后台 Session.submit/wait。

    CUDA 模式等待本设备完成后才导出。临时文件和输出请由 RuntimeConfig
    事先放入 NAS。采集结果不用于性能加速比结论。
    """
    import torch

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA profiling requested but CUDA is unavailable")
        activities.append(torch.profiler.ProfilerActivity.CUDA)
        torch.cuda.synchronize()
    started = time.perf_counter()
    metadata = dict(
        label=label,
        completed=False,
        diagnostic_only=True,
        cuda_enabled=cuda,
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
    )
    try:
        with torch.profiler.profile(
            activities=activities, record_shapes=True
        ) as profiler:
            with torch.profiler.record_function(label):
                yield
            if cuda:
                torch.cuda.synchronize()
        metadata["completed"] = True
    finally:
        metadata["wall_s"] = time.perf_counter() - started
        original_error = sys.exception()
        try:
            if "profiler" in locals():
                profiler.export_chrome_trace(str(output / "trace.json"))
        except Exception as error:
            metadata.update(
                completed=False, export_error=f"{type(error).__name__}: {error}"
            )
            if original_error is None:
                raise
            original_error.add_note(f"Profiler export failed: {error}")
        finally:
            (output / "capture.json").write_text(
                json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
            )
