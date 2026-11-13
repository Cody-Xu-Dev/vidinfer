"""第三层：统一实验协议。首次调用、误差、稳态计时和显存分别测量。"""

from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time
import traceback
import uuid

from .cases import build_case, file_sha256
from .metrics import compare_outputs, summarize
from .spec import ExperimentSpec, balanced_orders


def run_experiment(config_path, workspace):
    from ..runtime import RuntimeConfig

    # 配置缓存必须早于导入 Torch/扩展；同样适用于另一台机器或容器。
    workspace = RuntimeConfig(workspace=workspace).configure().workspace
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    spec = ExperimentSpec(**config)
    folder = (
        workspace
        / "experiments"
        / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-"
            + uuid.uuid4().hex[:8]
        )
    )
    folder.mkdir(parents=True)
    report = dict(
        schema_version=1,
        scope="operator_only",
        completed=False,
        created_at=datetime.now(timezone.utc).isoformat(),
        spec=spec.to_dict(),
        config_sha256=file_sha256(config_path),
        candidates={},
        orders=[],
        warnings=[
            "单算子通过不代表完整 Block、去噪或视频质量通过。",
            "CUDA Event 是设备时间线跨度，不等于纯 Kernel 忙碌时间。",
            "显存数据为本进程 PyTorch allocated，不含其他进程或全部厂商库分配。",
        ],
    )

    def save():
        from .report import write_reports

        write_reports(report, folder)

    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required to run operator experiments; reports can be rendered without CUDA"
            )
        torch.cuda.set_device(0)
        props = torch.cuda.get_device_properties(0)
        report["environment"] = dict(
            python=platform.python_version(),
            torch=str(torch.__version__),
            cuda=torch.version.cuda,
            cudnn=torch.backends.cudnn.version(),
            gpu=props.name,
            compute_capability=list(torch.cuda.get_device_capability(0)),
            total_memory_bytes=props.total_memory,
            tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
            tf32_cudnn=torch.backends.cudnn.allow_tf32,
        )
        versions = report["environment"]["packages"] = {}
        for package in ("numpy", "diffusers", "safetensors", "ninja"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        source_root = Path(__file__).parents[1]
        source_files = sorted(
            p
            for directory in (Path(__file__).parent, source_root / "kernels")
            for p in directory.rglob("*")
            if p.suffix in (".py", ".cpp", ".cu")
        )
        report["code_sha256"] = {
            p.relative_to(source_root).as_posix(): file_sha256(p) for p in source_files
        }
        report["code_identity"] = hashlib.sha256(
            json.dumps(report["code_sha256"], sort_keys=True).encode()
        ).hexdigest()
        # 只使用本进程的专用 Stream；不能与其他推理请求并发做性能比较。
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.inference_mode(), torch.cuda.stream(stream):
            functions, report["input"], report["contract"] = build_case(
                spec, config_path.parent
            )
            stream.synchronize()
            save()

            def call(name):
                result = functions[name]()
                stream.synchronize()
                return result

            # 基准不可用时停止，而不是悄悄改用另一种基准。
            begin = time.perf_counter()
            reference = call(spec.reference)
            report["reference_first_call_wall_s"] = time.perf_counter() - begin
            valid = []
            for name in spec.candidates:
                item = report["candidates"][name] = {"status": "probing"}
                try:
                    begin = time.perf_counter()
                    output = call(name)
                    item["probe_wall_s"] = time.perf_counter() - begin
                    item["correctness"] = compare_outputs(reference, output, spec)
                    del output
                    item["status"] = "available"
                    valid.append(name)
                except (RuntimeError, ValueError) as exc:
                    # probe_error 不是“硬件不支持”的断言，也不等于数值失败。
                    item.update(
                        status="probe_error", error=f"{type(exc).__name__}: {exc}"
                    )
                    stream.synchronize()  # 设备故障会在此继续抛出并终止整轮，不能带病测下一项。
                    gc.collect()
                    torch.cuda.empty_cache()
                save()
            del reference
            if spec.reference not in valid:
                raise RuntimeError("Reference correctness/probe failed")
            for name in valid:
                for _ in range(spec.warmup):
                    output = functions[name]()
                    del output
            stream.synchronize()
            samples = {
                name: {"cuda_event": [], "synchronized_wall": []} for name in valid
            }
            # Event 本身先物化，避免首次初始化污染样本；每次调用保留输出至完成。
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record(stream)
            end.record(stream)
            end.synchronize()
            report["orders"] = balanced_orders(valid, spec.cycles)
            for order in report["orders"]:
                for name in order:
                    stream.synchronize()
                    begin = time.perf_counter()
                    start.record(stream)
                    output = functions[name]()
                    end.record(stream)
                    end.synchronize()
                    samples[name]["synchronized_wall"].append(
                        (time.perf_counter() - begin) * 1000
                    )
                    samples[name]["cuda_event"].append(start.elapsed_time(end))
                    del output
            for name in valid:
                item = report["candidates"][name]
                item["timing"] = {
                    kind: summarize(values) for kind, values in samples[name].items()
                }
                # 与计时分开的一次稳态调用，基线不包含保留下来的参考输出。
                stream.synchronize()
                before = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                output = call(name)
                peak = torch.cuda.max_memory_allocated()
                after = torch.cuda.memory_allocated()
                item["memory"] = dict(
                    baseline_allocated_bytes=before,
                    peak_allocated_bytes=peak,
                    incremental_peak_bytes=max(0, peak - before),
                    output_and_retained_delta_bytes=after - before,
                )
                del output
                stream.synchronize()
            base = report["candidates"][spec.reference]["timing"]["cuda_event"][
                "median_ms"
            ]
            for name in valid:
                item = report["candidates"][name]
                elapsed = item["timing"]["cuda_event"]["median_ms"]
                item["speedup_vs_reference"] = base / elapsed if elapsed else None
                item["passes_contract"] = item["correctness"]["passes"]
            report["completed"] = True
            report["all_candidates_pass"] = all(
                item.get("passes_contract", False)
                for item in report["candidates"].values()
            )
            # 不自动修改模型计划。微基准最快者还需要 Block/完整去噪验证。
            save()
    except Exception as exc:
        report["fatal_error"] = traceback.format_exc()
        save()
        exc.add_note(f"Experiment report: {folder / 'report.html'}")
        raise
    return folder
