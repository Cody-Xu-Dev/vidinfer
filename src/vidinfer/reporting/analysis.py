"""从 Chrome Trace 生成可复核的数据，不根据耗时猜测算力/带宽瓶颈。"""

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re


def fingerprint(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(2**20), b""):
            digest.update(data)
    return dict(path=str(path), sha256=digest.hexdigest(), bytes=path.stat().st_size)


def union_duration(intervals):
    """多个 Stream 可以重叠：各段 duration 相加不能用来代替 union/span。"""
    merged = 0.0
    end = None
    for start, stop in sorted(intervals):
        if end is None or start > end:
            merged += stop - start
            end = stop
        elif stop > end:
            merged += stop - end
            end = stop
    return merged


def category(name):
    name = name.lower()
    if any(word in name for word in ("flash", "fmha", "attention", "attn")):
        return "Attention"
    if any(
        word in name for word in ("gemm", "matmul", "cublas", "cutlass", "nvjet", "mma")
    ):
        return "GEMM / Linear"
    if any(
        word in name
        for word in ("copy", "catarray", "transpose", "permute", "memcpy", "memset")
    ):
        return "Layout / Copy"
    if any(word in name for word in ("elementwise", "norm", "silu", "swiglu", "rope")):
        return "Elementwise / Norm"
    return "Other"


def _positive(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def analyze_trace(
    path, *, summary_path=None, scope="unspecified", timeline_limit=10000
):
    if type(timeline_limit) is not int or timeline_limit < 1:
        raise ValueError("timeline_limit must be a positive integer")
    path = Path(path).resolve()
    trace = json.loads(path.read_text(encoding="utf-8-sig"))
    events = trace.get("traceEvents") if isinstance(trace, dict) else trace
    if not isinstance(events, list):
        raise ValueError("Expected Chrome Trace traceEvents array")
    gpu, regions, cpu = [], [], defaultdict(list)
    skipped = 0
    for event in events:
        if not isinstance(event, dict) or event.get("ph") != "X":
            continue
        cat = event.get("cat", "")
        if cat not in (
            "kernel",
            "gpu_memcpy",
            "gpu_memset",
            "gpu_user_annotation",
            "cpu_op",
        ):
            continue
        start, duration = event.get("ts"), event.get("dur")
        if (
            not isinstance(start, (int, float))
            or isinstance(start, bool)
            or not math.isfinite(start)
            or not _positive(duration)
        ):
            skipped += 1
            continue
        name = str(event.get("name", "unnamed"))
        if cat == "cpu_op":
            cpu[name].append(duration / 1000)
            continue
        if cat == "gpu_user_annotation":
            regions.append(dict(name=name, start_us=start, duration_us=duration))
            continue
        args = event.get("args") or {}
        device = str(args.get("device", event.get("pid", "unknown")))
        stream = str(args.get("stream", event.get("tid", "unknown")))
        gpu.append(
            dict(
                name=name,
                start_us=start,
                duration_us=duration,
                kind=cat,
                lane=f"GPU {device} / Stream {stream}",
                category=category(name),
            )
        )
    if not gpu:
        raise ValueError(
            "No completed GPU activities found; CPU-only or empty traces need a different scope"
        )
    gpu.sort(key=lambda e: e["start_us"])
    origin = min(e["start_us"] for e in gpu)
    stop = max(e["start_us"] + e["duration_us"] for e in gpu)
    kernels = [e for e in gpu if e["kind"] == "kernel"]
    kernel_sum = sum(e["duration_us"] for e in kernels)
    intervals = [(e["start_us"], e["start_us"] + e["duration_us"]) for e in gpu]
    kernel_union = union_duration(
        [(e["start_us"], e["start_us"] + e["duration_us"]) for e in kernels]
    )
    activity_union = union_duration(intervals)
    kernel_groups, categories = defaultdict(list), defaultdict(float)
    for event in kernels:
        kernel_groups[event["name"]].append(event["duration_us"] / 1000)
        categories[event["category"]] += event["duration_us"] / 1000
    hotspots = sorted(
        (
            dict(
                name=name,
                count=len(values),
                total_ms=sum(values),
                mean_ms=sum(values) / len(values),
                category=category(name),
                kernel_sum_fraction=sum(values) * 1000 / kernel_sum
                if kernel_sum
                else None,
            )
            for name, values in kernel_groups.items()
        ),
        key=lambda row: row["total_ms"],
        reverse=True,
    )
    # 指标使用完整事件集；仅限制 HTML 时间线数据量，并明确披露抽样。
    timeline = (
        gpu
        if len(gpu) <= timeline_limit
        else [gpu[(i * len(gpu)) // timeline_limit] for i in range(timeline_limit)]
    )
    timeline = [dict(event, start_us=event["start_us"] - origin) for event in timeline]
    regions = sorted(
        (dict(event, start_us=event["start_us"] - origin) for event in regions),
        key=lambda e: e["start_us"],
    )
    blocks = [
        e
        for e in regions
        if re.search(r"(?:block[_./-]?\d+|transformer_blocks[./]\d+)$", e["name"], re.I)
    ]
    report = dict(
        schema_version=1,
        report_type="inference",
        completed=True,
        created_at=datetime.now(timezone.utc).isoformat(),
        scope=scope,
        sources={"trace": fingerprint(path)},
        stages=[],
        forwards=[],
        environment={},
        trace=dict(
            activity_count=len(gpu),
            kernel_count=len(kernels),
            copy_count=len(gpu) - len(kernels),
            kernel_sum_ms=kernel_sum / 1000,
            kernel_union_ms=kernel_union / 1000,
            activity_span_ms=(stop - origin) / 1000,
            activity_union_ms=activity_union / 1000,
            observed_gap_fraction=1 - activity_union / (stop - origin)
            if stop > origin
            else None,
            hotspots=hotspots,
            category_ms=dict(categories),
            cpu_hotspots=sorted(
                (
                    dict(name=name, count=len(values), inclusive_ms=sum(values))
                    for name, values in cpu.items()
                ),
                key=lambda e: e["inclusive_ms"],
                reverse=True,
            )[:50],
            timeline=timeline,
            timeline_total_events=len(gpu),
            timeline_sampled=len(gpu) > timeline_limit,
            regions=regions,
            blocks=blocks,
            skipped_invalid_events=skipped,
        ),
        warnings=[
            "Profiler 会影响执行时间；该 trace 用于诊断，不能直接代替无 profiler 的性能实验。",
            "Kernel sum、活动 span 与区间 union 含义不同；重叠 Stream 下 sum 可能超过 span。",
            "活动覆盖率仅描述本 trace 的时间线，不是 SM 利用率、显存带宽利用率或算力占用率。",
            "热点分类依据 Kernel 名称启发式匹配；不能据此断言算力或带宽瓶颈。",
            "只用 GPU annotation 定位 GPU 时间线；CPU annotation 的返回时间不是 GPU 完成时间。",
        ],
    )
    if summary_path is not None:
        report["sources"]["summary"] = fingerprint(summary_path)
        summary = json.loads(Path(summary_path).read_text(encoding="utf-8-sig"))
        report["example_only"] = bool(summary.get("example_only", False))
        metadata = summary.get("metadata", summary)
        phases = summary.get("phases", metadata.get("stages", {}))
        for name, value in phases.items():
            if not isinstance(value, dict) or not _positive(value.get("wall_s")):
                continue
            peak = value.get("peak_allocated_bytes")
            if peak is None and _positive(value.get("peak_allocated_gib")):
                peak = value["peak_allocated_gib"] * 2**30
            report["stages"].append(
                dict(
                    name=name,
                    wall_ms=value["wall_s"] * 1000,
                    kind="load" if name.startswith("load") else "compute",
                    peak_allocated_bytes=peak if _positive(peak) else None,
                )
            )
        if "forwards" in summary:
            report["forwards"] = [
                dict(
                    index=e.get("index", i),
                    elapsed_ms=e["gpu_ms"],
                    profiled=bool(e.get("profiled")),
                )
                for i, e in enumerate(summary["forwards"])
                if _positive(e.get("gpu_ms"))
            ]
        else:
            report["forwards"] = [
                dict(index=i, elapsed_ms=value, profiled=None)
                for i, value in enumerate(metadata.get("forward_ms", []))
                if _positive(value)
            ]
        report["environment"] = {
            key: summary[key] for key in ("torch", "cuda", "gpu") if key in summary
        }
        report["request"] = summary.get("request", summary.get("config", {}))
        report["request_wall_ms"] = (
            metadata["wall_s"] * 1000 if _positive(metadata.get("wall_s")) else None
        )
        report["warnings"].append(
            "阶段数据与 trace 的关联由调用者指定；阶段可能包含加载/CPU 等待，不与 GPU Kernel sum 相加。"
        )
    return report
