"""第四层：只读实验数据生成报告，不导入 Torch，不启动 GPU。"""

from html import escape
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
    spec = report["spec"]
    headers = "".join(f"<th>{escape(label)}</th>" for label in HEADERS)
    rows = "".join(
        "<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in row) + "</tr>"
        for row in _rows(report)
    )
    notes = "".join(
        f"<li>{escape(warning)}</li>" for warning in report.get("warnings", [])
    )
    details = ""
    for name, item in report.get("candidates", {}).items():
        details += (
            f"<details><summary>{escape(name)} · 误差、计时原始样本与显存</summary>"
            f"<pre>{escape(json.dumps(item, indent=2, ensure_ascii=False, allow_nan=False))}</pre></details>"
        )
    payload = escape(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    title = escape(spec["operation"])
    # 所有外部文本先 HTML 转义；报告无需联网、JS、字体或第三方库。
    return f"""<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>VidInfer · {title}</title>
<style>body{{font:16px/1.65 system-ui,sans-serif;margin:36px auto;padding:0 24px;max-width:1280px;background:#f6f7fb;color:#182435}}h1{{margin-bottom:0}}.muted{{color:#526078}}section,details{{background:white;border:1px solid #d8dfeb;border-radius:12px;padding:20px;margin:20px 0}}.table{{overflow:auto}}table{{border-collapse:collapse;width:100%;white-space:nowrap}}th,td{{text-align:left;padding:12px;border-bottom:1px solid #e3e8f0}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}}summary{{cursor:pointer;font-weight:600}}code{{background:#eef1f7;padding:2px 5px}}a{{color:#075dcc}}</style>
<h1>VidInfer · {title}</h1><p class="muted">统一单算子实验报告 · schema 1 · {escape(report["created_at"])}</p>
<section><b>整轮完成：{report["completed"]}</b> · 参考 <code>{escape(spec["reference"])}</code> · 输入来源 <code>{escape(report.get("input", {}).get("kind", spec["source"].get("kind", "synthetic")))}</code>
<p>{escape(report.get("contract", "准备阶段尚未完成"))}</p><p>验收 {escape(spec["mode"])} · atol={spec["atol"]} · rtol={spec["rtol"]}</p></section>
<section class="table"><table><thead><tr>{headers}</tr></thead><tbody>{rows}</tbody></table>
<p class="muted">PASS 是指定参考下的数值验收；倍数小于 1 表示更慢。错误或未完成项不能参与结论。</p></section>
<section><h2>如何读这张表</h2><p>先检查数值，再比较 Event 与同步 Wall，最后结合显存。输入加载、H2D、首次编译与误差检查不计入稳态时间；调用内部的布局转换、输出分配和整理计入。P90 是当前微基准样本分位数。</p><ul>{notes}</ul><p>完整输入形状/stride、环境、源码与输入文件指纹、均衡顺序及所有原始样本见下方。报告不自动启用模型优化。</p></section>
{details}<details><summary>完整 JSON / provenance / 错误记录</summary><pre>{payload}</pre></details></html>"""


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
