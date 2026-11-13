"""研究清单 → 总览、推理分析、优化实验与证据副本。不会启动推理。"""

from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import shutil

from .analysis import analyze_trace, fingerprint
from .render import (
    page,
    raw,
    table,
    template,
    write_inference,
    inference_html,
    optimization_html,
)


def build_workflow(manifest_path, output):
    manifest_path = Path(manifest_path).resolve()
    output = Path(output).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Expected workflow schema_version=1")
    if not manifest.get("inferences") and not manifest.get("experiments"):
        raise ValueError("Workflow needs inference or experiment inputs")
    output.mkdir(parents=True, exist_ok=False)
    data = output / "data"
    data.mkdir()

    def resolve(value):
        path = Path(value).expanduser()
        path = (
            (manifest_path.parent / path).resolve()
            if not path.is_absolute()
            else path.resolve()
        )
        if path.is_relative_to(output):
            raise ValueError("Source inputs must stay outside the generated bundle")
        if not path.is_file():
            raise ValueError(f"Missing input: {path}")
        return path

    records = dict(
        schema_version=1,
        report_type="workflow",
        completed=False,
        created_at=datetime.now(timezone.utc).isoformat(),
        manifest=manifest,
        manifest_source=fingerprint(manifest_path),
        inferences=[],
        experiments=[],
        evidence=[],
    )
    try:
        for i, entry in enumerate(manifest.get("inferences", []), 1):
            trace = resolve(entry["trace"])
            summary = resolve(entry["summary"]) if entry.get("summary") else None
            report = analyze_trace(
                trace, summary_path=summary, scope=entry.get("scope", "unspecified")
            )
            name = f"inference-{i:02d}"
            write_inference(report, data / name)
            (output / (name + ".html")).write_text(
                inference_html(report, backlink="index.html"), encoding="utf-8"
            )
            records["inferences"].append(
                dict(
                    name=entry.get("name", name),
                    href=name + ".html",
                    note=entry.get("note", ""),
                    scope=report["scope"],
                    sources=report["sources"],
                )
            )
        from ..experiments.report import write_reports

        for i, entry in enumerate(manifest.get("experiments", []), 1):
            source = resolve(entry["results"])
            report = json.loads(source.read_text(encoding="utf-8-sig"))
            if report.get("schema_version") != 1 or "spec" not in report:
                raise ValueError("Expected operator experiment results schema 1")
            name = f"optimization-{i:02d}"
            write_reports(report, data / name)
            (output / (name + ".html")).write_text(
                optimization_html(report, backlink="index.html"), encoding="utf-8"
            )
            passed = sum(
                v.get("correctness", {}).get("passes") is True
                for v in report.get("candidates", {}).values()
            )
            records["experiments"].append(
                dict(
                    name=entry.get("name", name),
                    href=name + ".html",
                    hypothesis=entry.get("hypothesis", ""),
                    scope=report.get("scope", "operator_only"),
                    completed=report["completed"],
                    passed=passed,
                    candidates=len(report.get("candidates", {})),
                    source=fingerprint(source),
                )
            )
        for i, entry in enumerate(manifest.get("evidence", []), 1):
            # 状态是操作者的声明。只附显式指定的证据，不读取报告里未经授权的路径。
            status = entry.get("status", "pending")
            if status not in ("pass", "fail", "pending", "historical"):
                raise ValueError("Evidence status must be pass/fail/pending/historical")
            record = dict(
                level=entry.get("level", "unspecified"),
                status=status,
                note=entry.get("note", ""),
                verification="operator-declared; file existence/hash only",
            )
            if entry.get("file"):
                source = resolve(entry["file"])
                if source.suffix.lower() not in (".json", ".md", ".txt"):
                    raise ValueError(
                        "Bundle only JSON/Markdown/text evidence; do not embed executable HTML or model files"
                    )
                target = f"evidence-{i:02d}" + source.suffix.lower()
                shutil.copyfile(source, data / target)
                record.update(source=fingerprint(source), href="data/" + target)
            records["evidence"].append(record)
        records["completed"] = True
    except Exception as exc:
        records["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (output / "workflow.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    def links(entries, experiment=False):
        if not entries:
            return '<p class="empty">未附此类报告。</p>'
        result = []
        for entry in entries:
            note = entry.get("hypothesis", "") if experiment else entry.get("note", "")
            summary = (
                f"{entry['passed']}/{entry['candidates']} 项数值通过 · {'已完成' if entry['completed'] else '未完成'}"
                if experiment
                else entry["scope"]
            )
            result.append(
                f'<a class="report-link" href="{entry["href"]}"><b>{escape(entry["name"])}</b><span class="tag">{escape(summary)}</span><p>{escape(note)}</p></a>'
            )
        return "".join(result)

    evidence = table(
        ["验证层级", "声明状态", "说明"],
        [[e["level"], e["status"], e["note"]] for e in records["evidence"]],
    )
    evidence += "".join(
        f'<p><a href="{e["href"]}">下载证据 {i + 1}</a> · SHA256 <code>{e["source"]["sha256"]}</code></p>'
        for i, e in enumerate(records["evidence"])
        if "href" in e
    )
    body = template(
        "index.html",
        dict(
            INFERENCES=links(records["inferences"]),
            EXPERIMENTS=links(records["experiments"], True),
            EVIDENCE=evidence,
            RAW=raw(records),
        ),
    )
    (output / "index.html").write_text(
        page(
            manifest.get("title", "推理分析与优化"),
            "WORKFLOW",
            manifest.get("note", "从性能诊断到实验验证的可追溯记录"),
            body,
            [
                ("chain", "工具链路"),
                ("inferences", "推理分析"),
                ("experiments", "优化实验"),
                ("evidence", "验证证据"),
                ("provenance", "研究清单"),
            ],
            records,
        ),
        encoding="utf-8",
    )
    return output
