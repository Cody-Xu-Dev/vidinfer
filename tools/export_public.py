"""按白名单导出可公开源码；不携带模型、运行日志或机器专用集成脚本。

用法：python tools/export_public.py DESTINATION
目标目录应为空；只读源文件，导出目录可直接提交至独立 GitHub 仓库。
"""

import hashlib
import json
from pathlib import Path
import re
import shutil
import sys


def export(destination):
    source = Path(__file__).resolve().parents[1]
    destination = Path(destination).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Export destination must be empty")
    if destination == source or destination.is_relative_to(source):
        raise ValueError("Export outside the source project directory")
    destination.mkdir(parents=True, exist_ok=True)
    files = [
        source / name
        for name in ("README.md", "pyproject.toml", ".gitignore", "THIRD_PARTY.md")
    ]
    for directory in ("src", "tests", "docs", ".github", "tools"):
        for path in sorted((source / directory).rglob("*")):
            if (
                path.is_file()
                and "__pycache__" not in path.parts
                and path.suffix
                in (
                    ".py",
                    ".cpp",
                    ".cu",
                    ".md",
                    ".yml",
                    ".json",
                    ".html",
                    ".css",
                    ".js",
                )
            ):
                if path.name == "codex-dev-notes.md":
                    continue
                files.append(path)
    manifest = {}
    private = re.compile(r"/data-" + r"koolab-nas/|\b10[.]\d+[.]\d+[.]\d+\b")
    for path in files:
        text = path.read_text(encoding="utf-8")
        if private.search(text):
            raise ValueError(f"Machine-specific content in {path.relative_to(source)}")
        name = path.relative_to(source)
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        manifest[name.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return manifest


if __name__ == "__main__":
    manifest = export(sys.argv[1])
    print(json.dumps({"files": len(manifest), "sha256": manifest}, indent=2))
