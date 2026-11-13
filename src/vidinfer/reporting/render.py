"""共用的干净 HTML 模板；CSS/JS 随 wheel 分发，输出文件完全离线。"""

from datetime import datetime, timezone
from html import escape
from importlib.resources import files
import json
from pathlib import Path
import re


def number(value):
    return "—" if value is None else f"{value:,.5g}"


def template(name, values):
    source = (files("vidinfer.reporting") / "templates" / name).read_text(
        encoding="utf-8"
    )
    # 只替换一次；用户文本即使含 {{SCRIPT}} 也不能再次作为模板执行。
    return re.sub(r"\{\{([A-Z_]+)\}\}", lambda match: values[match.group(1)], source)


def page(title, kind, subtitle, body, nav, payload, *, backlink=None):
    safe_json = (
        json.dumps(payload, ensure_ascii=False, allow_nan=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    assets = files("vidinfer.reporting") / "templates"
    return template(
        "base.html",
        dict(
            TITLE=escape(title),
            KIND=escape(kind),
            SUBTITLE=escape(subtitle),
            BODY=body,
            NAV=(
                f'<a href="{escape(backlink, quote=True)}">← 返回研究总览</a>'
                if backlink
                else ""
            )
            + "".join(
                f'<a href="#{anchor}">{escape(label)}</a>' for anchor, label in nav
            ),
            DATA=safe_json,
            STYLE=(assets / "style.css").read_text(encoding="utf-8"),
            SCRIPT=(assets / "interactive.js").read_text(encoding="utf-8"),
            STAMP=escape(datetime.now(timezone.utc).isoformat(timespec="seconds")),
        ),
    )


def table(headers, rows, *, identity="", numeric=(), attributes=None):
    content = []
    for i, row in enumerate(rows):
        attr = attributes[i] if attributes else ""
        cells = "".join(
            f'<td class="{"num" if j in numeric else "long"}"><span class="cell-text" title="{escape(str(value))}">{escape(str(value))}</span></td>'
            for j, value in enumerate(row)
        )
        content.append(f"<tr {attr}>{cells}</tr>")
    if not content:
        return '<p class="empty">未采集或没有可展示的数据。</p>'
    return (
        f'<div class="table-wrap"><table id="{identity}"><thead><tr>'
        + "".join(f"<th>{escape(h)}</th>" for h in headers)
        + "</tr></thead><tbody>"
        + "".join(content)
        + "</tbody></table></div>"
    )


def metrics(items):
    return (
        '<div class="metrics">'
        + "".join(
            f'<div class="metric"><small>{escape(label)}</small><strong>{escape(str(value))}</strong><em>{escape(note)}</em></div>'
            for label, value, note in items
        )
        + "</div>"
    )


def warnings(report):
    return (
        "<ul>"
        + "".join(f"<li>{escape(w)}</li>" for w in report.get("warnings", []))
        + "</ul>"
    )


def raw(data):
    return escape(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False))


def inference_html(report, *, backlink=None):
    if report.get("report_type") != "inference" or report.get("schema_version") != 1:
        raise ValueError("Expected inference report schema 1")
    trace = report["trace"]
    next_steps = []
    ranked = sorted(trace["category_ms"], key=trace["category_ms"].get, reverse=True)
    for name in ranked[:3]:
        action = {
            "Attention": "比较后端与布局路径，包含 contiguous 和输出转换成本。",
            "GEMM / Linear": "检查真实 MatMul 形状、调用粒度与 QKV 融合；用硬件计数器验证限制。",
            "Layout / Copy": "定位重复布局整理与拷贝，核对上下游的 stride 合约。",
            "Elementwise / Norm": "测试 Norm/RoPE/SwiGLU 的融合，保留精度边界并验证累计误差。",
            "Other": "先核对 Kernel 来源，必要时结合 CPU 操作和厂商工具定位。",
        }[name]
        next_steps.append(f"<li><b>{escape(name)}</b>：{escape(action)}</li>")
    body = template(
        "inference.html",
        dict(
            METRICS=metrics(
                [
                    (
                        "GPU 活动跨度",
                        number(trace["activity_span_ms"]) + " ms",
                        "最早开始 → 最晚结束",
                    ),
                    (
                        "Kernel 时间总和",
                        number(trace["kernel_sum_ms"]) + " ms",
                        "含 Stream 重叠，不能当 Wall",
                    ),
                    (
                        "GPU 活动区间并集",
                        number(trace["activity_union_ms"]) + " ms",
                        "合并重叠活动",
                    ),
                    ("Kernel 数量", str(trace["kernel_count"]), "当前采集范围"),
                ]
            ),
            STAGE_TABLE=table(
                ["阶段", "CPU Wall / s", "峰值 allocated / GiB", "类型"],
                [
                    [
                        s["name"],
                        number(s["wall_ms"] / 1000),
                        number(
                            s["peak_allocated_bytes"] / 2**30
                            if s["peak_allocated_bytes"] is not None
                            else None
                        ),
                        s["kind"],
                    ]
                    for s in report["stages"]
                ],
                numeric=(1, 2),
            ),
            HOTSPOT_TABLE=table(
                [
                    "Kernel",
                    "分类（启发式）",
                    "调用",
                    "总计 ms",
                    "平均 ms",
                    "Kernel sum 占比",
                ],
                [
                    [
                        e["name"],
                        e["category"],
                        e["count"],
                        number(e["total_ms"]),
                        number(e["mean_ms"]),
                        number(
                            e["kernel_sum_fraction"] * 100
                            if e["kernel_sum_fraction"] is not None
                            else None
                        )
                        + "%",
                    ]
                    for e in trace["hotspots"][:50]
                ],
                identity="hotspot-table",
                numeric=(2, 3, 4, 5),
            ),
            NEXT="<ol>" + "".join(next_steps) + "</ol>"
            if next_steps
            else '<p class="empty">未捕获 Kernel，不能推断优化方向。</p>',
            WARNINGS=warnings(report),
            RAW=raw(report),
            CPU_TABLE="<details><summary>CPU 操作 · inclusive 时间，不可相加</summary>"
            + table(
                ["操作", "次数", "inclusive ms"],
                [
                    [e["name"], e["count"], number(e["inclusive_ms"])]
                    for e in trace["cpu_hotspots"]
                ],
                numeric=(1, 2),
            )
            + "</details>",
        ),
    )
    return page(
        "推理性能分析",
        "INFERENCE",
        ("示例数据 · 非真实性能证据 · " if report.get("example_only") else "")
        + f"采集范围：{report['scope']} · Profiler 诊断记录",
        body,
        [
            ("stages", "阶段耗时"),
            ("hotspots", "GPU 热点"),
            ("timeline", "GPU 时间线"),
            ("next", "下一步实验"),
            ("provenance", "来源与边界"),
        ],
        report,
        backlink=backlink,
    )


def optimization_html(report, *, backlink=None):
    if report.get("schema_version") != 1 or "spec" not in report:
        raise ValueError("Expected optimization report schema 1")
    from ..experiments.report import HEADERS, _rows

    spec = report["spec"]
    items = report.get("candidates", {})
    rows = list(_rows(report))
    accuracy = []
    for name, item in items.items():
        for output, value in item.get("correctness", {}).get("outputs", {}).items():
            accuracy.append(
                [
                    name,
                    output,
                    str(value.get("passes")),
                    str(value.get("bitwise_exact")),
                    number(value.get("max_absolute_error")),
                    number(value.get("mean_absolute_error")),
                    number(value.get("relative_l2")),
                    (
                        f"{value['cosine_similarity']:.9f}"
                        if value.get("cosine_similarity") is not None
                        else "—"
                    ),
                    (
                        str(value["failed_elements"])
                        if value.get("failed_elements") is not None
                        else "—"
                    ),
                ]
            )
    passed = [
        name
        for name, item in items.items()
        if item.get("correctness", {}).get("passes") is True
    ]
    failed = [
        name
        for name, item in items.items()
        if item.get("correctness", {}).get("passes") is False
    ]
    errors = [
        f"{name}: {item['error']}" for name, item in items.items() if item.get("error")
    ]
    decision = f"<p>整轮完成：<b>{report['completed']}</b>。数值通过 {len(passed)} 项；数值失败 {len(failed)} 项。</p>"
    if failed:
        decision += (
            '<p class="note warn">未通过预设数值要求：'
            + escape(", ".join(failed))
            + "。即使速度更快，也不能按本轮标准自动启用。</p>"
        )
    if errors:
        decision += "<pre>" + escape("\n".join(errors)) + "</pre>"
    if report.get("fatal_error"):
        decision += (
            '<p class="note warn">整轮未完成，不能据此做性能结论。</p><pre>'
            + escape(report["fatal_error"])
            + "</pre>"
        )
    decision += "<p>本报告不自动选择或启用模型优化。其他层级的历史证据应在研究总览中单独附上。</p>"
    body = template(
        "optimization.html",
        dict(
            METRICS=metrics(
                [
                    ("实验范围", report.get("scope", "operator_only"), "当前为单算子"),
                    ("候选实现", str(len(items)), "同一份准备后的输入"),
                    (
                        "数值通过",
                        f"{len(passed)} / {len(items)}",
                        spec["mode"] + " 验收",
                    ),
                    (
                        "输入来源",
                        report.get("input", {}).get(
                            "kind", spec["source"].get("kind", "synthetic")
                        ),
                        "真实捕获与合成输入分开",
                    ),
                ]
            ),
            CONTRACT=escape(report.get("contract", "算子尚未准备完成。")),
            COMPARISON_TABLE=table(
                HEADERS,
                rows,
                identity="comparison-table",
                numeric=(2, 3, 4, 5, 6, 7),
                attributes=[
                    f'data-pass="{str(items[row[0]].get("correctness", {}).get("passes") is True).lower()}"'
                    for row in rows
                ],
            ),
            TOLERANCE=escape(
                f"参考：{spec['reference']}；mode={spec['mode']}；atol={spec['atol']}；rtol={spec['rtol']}。"
            ),
            ACCURACY_TABLE=table(
                [
                    "实现",
                    "输出",
                    "通过",
                    "位模式一致",
                    "最大绝对误差",
                    "MAE",
                    "相对 L2",
                    "余弦",
                    "失败元素",
                ],
                accuracy,
                numeric=(4, 5, 6, 7, 8),
            ),
            DECISION=decision,
            WARNINGS=warnings(report),
            RAW=raw(report),
        ),
    )
    return page(
        spec["operation"] + " 优化实验",
        "OPTIMIZATION",
        ("示例数据 · 非真实性能证据 · " if report.get("example_only") else "")
        + "同输入对照 · 数值验收与性能测量分开",
        body,
        [
            ("comparison", "实现对比"),
            ("accuracy", "数值验收"),
            ("samples", "原始样本"),
            ("decision", "结论与后续"),
            ("provenance", "来源与边界"),
        ],
        dict(report_type="optimization", report=report),
        backlink=backlink,
    )


def write_inference(report, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    markdown = f"# 推理分析\n\n范围：{report['scope']}\n\nKernel sum: {report['trace']['kernel_sum_ms']:.6f} ms\nGPU activity span: {report['trace']['activity_span_ms']:.6f} ms\nGPU activity union: {report['trace']['activity_union_ms']:.6f} ms\n\n以上三项不可互相替代；详见 report.html。\n"
    for name, text in (
        ("analysis.json", payload),
        ("report.html", inference_html(report)),
        ("report.md", markdown),
    ):
        temporary = folder / (name + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(folder / name)
