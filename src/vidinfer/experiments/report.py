"""第四层：只读实验数据生成报告，不导入 Torch，不启动 GPU。"""

import json
from pathlib import Path


def _number(value):
    return "—" if value is None else f"{value:.6g}"


def _rows(report):
    for name, item in report.get("candidates", {}).items():
        outputs = item.get("correctness", {}).get("outputs", {}).values()
        maximum = max(
            (
                x["max_absolute_error"]
                for x in outputs
                if x.get("max_absolute_error") is not None
            ),
            default=None,
        )
        event = item.get("timing", {}).get("cuda_event", {})
        wall = item.get("timing", {}).get("synchronized_wall", {})
        correctness = item.get("correctness", {}).get("passes")
        status = (
            ("PASS" if correctness else "FAIL")
            if item.get("status") == "available"
            else item.get("status", "pending")
        )
        memory = item.get("memory", {}).get("incremental_peak_bytes")
        yield [
            name,
            status,
            _number(maximum),
            _number(event.get("median_ms")),
            _number(wall.get("median_ms")),
            _number(event.get("p90_ms")),
            _number(memory / 2**20 if memory is not None else None),
            _number(item.get("speedup_vs_reference")),
        ]


HEADERS = [
    "实现",
    "数值验收",
    "最大绝对误差",
    "Event 中位数 ms",
    "同步 Wall 中位数 ms",
    "Event P90 ms",
    "增量峰值 MiB",
    "相对基准倍数",
]


def render_markdown(report):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    environment = report.get("environment", {})
    spec = report["spec"]
    lines = [
        f"# {spec['operation']} 单算子实验",
        "",
        f"完成：{report['completed']}；设备：{environment.get('gpu', '尚未初始化')}；参考：`{spec['reference']}`。",
        "",
        f"输入来源：{report.get('input', {}).get('kind', spec['source'].get('kind', 'synthetic'))}；验收模式：{spec['mode']}；atol={spec['atol']}，rtol={spec['rtol']}。",
        "",
        report.get("contract", "算子尚未准备完成。"),
        "",
        "| " + " | ".join(HEADERS) + " |",
        "| " + " | ".join(["---"] * len(HEADERS)) + " |",
    ]
    lines.extend(
        "| " + " | ".join(cell(value) for value in row) + " |" for row in _rows(report)
    )
    lines.extend(
        [
            "",
            "PASS 仅表示相对于指定参考的数值验收通过，不表示性能提升或完整模型质量通过。",
            "相对基准倍数 = 基准 Event 中位数 / 当前中位数；小于 1 表示更慢。",
            "",
            "## 计量边界",
            "",
            "首次调用/编译、误差计算、输入文件读取和 H2D 不计入稳态计时。候选函数内部的布局转换、输出分配和输出整理计入；SwiGLU/QK 的投影不在本轮算子边界内。",
            "Event 与同步 Wall 均记录原始样本，P90 是本轮调用样本分位数，不是生产请求 P90。显存单独测一次稳态调用，报告本进程 allocated 的增量峰值，不代表整张卡总占用。",
            "",
        ]
    )
    lines.extend(f"- {warning}" for warning in report.get("warnings", []))
    for name, item in report.get("candidates", {}).items():
        if "error" in item:
            lines.extend(["", f"{cell(name)}：{cell(item['error'])}"])
    if "fatal_error" in report:
        lines.extend(
            [
                "",
                "**整轮未完成，不能据此做性能结论。**",
                "",
                "```text",
                report["fatal_error"],
                "```",
            ]
        )
    return "\n".join(lines) + "\n"


def render_html(report):
    from ..reporting.render import optimization_html

    return optimization_html(report)


def write_reports(report, folder):
    if report.get("schema_version") != 1:
        raise ValueError("Unsupported report schema")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    contents = {
        "results.json": json.dumps(
            report, indent=2, ensure_ascii=False, allow_nan=False
        )
        + "\n",
        "report.md": render_markdown(report),
        "report.html": render_html(report),
    }
    for name, content in contents.items():
        target = folder / name
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(target)
