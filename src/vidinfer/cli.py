"""命令行入口；help 不加载 Torch，generate 才准备 GPU 模型。"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(prog="vidinfer")
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate", help="Load once and generate video/audio")
    generate.add_argument("--checkpoint", type=Path, required=True)
    generate.add_argument("--workspace", type=Path, required=True)
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--steps", type=int, choices=(5, 29), default=29)
    generate.add_argument("--height", type=int, default=544)
    generate.add_argument("--width", type=int, default=960)
    generate.add_argument("--frames", type=int, default=124)
    generate.add_argument(
        "--plan",
        choices=("reference", "qk_fused", "fully_fused"),
        default="fully_fused",
    )
    generate.add_argument("--adaln-cache", type=Path)
    generate.add_argument("--readers", type=int, default=2)
    generate.add_argument("--text-gpu-gib", type=int, default=40)
    generate.add_argument("--text-cpu-gib", type=int, default=4)
    sub.add_parser("doctor", help="Print installed runtime and GPU versions")
    experiment = sub.add_parser(
        "experiment",
        help="Compare operators on identical inputs; write JSON/Markdown/HTML",
    )
    experiment.add_argument("--config", type=Path, required=True)
    experiment.add_argument("--workspace", type=Path, required=True)
    report = sub.add_parser(
        "report", help="Render an existing experiment JSON without loading Torch"
    )
    report.add_argument("results", type=Path)
    report.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "experiment":
        from .experiments.runner import run_experiment

        folder = run_experiment(args.config, args.workspace)
        print(json.dumps({"report_directory": str(folder)}, ensure_ascii=False))
        return
    if args.command == "report":
        from .experiments.report import write_reports

        data = json.loads(args.results.read_text(encoding="utf-8-sig"))
        folder = args.output or args.results.parent
        write_reports(data, folder)
        print(str(folder.resolve() / "report.html"))
        return
    if args.command == "doctor":
        import torch
        import importlib.metadata

        print(
            json.dumps(
                {
                    "versions": {
                        n: importlib.metadata.version(n)
                        for n in (
                            "torch",
                            "diffusers",
                            "transformers",
                            "safetensors",
                            "accelerate",
                        )
                    },
                    "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(0)
                    if torch.cuda.is_available()
                    else None,
                },
                indent=2,
            )
        )
        return
    from .session import H3Session
    from .runtime import RuntimeConfig
    from .types import DenoiseRequest
    from .validation import validate_generation_request

    # 参数错误应在大型权重加载之前报告。
    validate_generation_request(
        DenoiseRequest(
            prompt=args.prompt,
            seed=args.seed,
            steps=args.steps,
            height=args.height,
            width=args.width,
            frames=args.frames,
        ),
        RuntimeConfig(workspace=args.workspace),
    )

    with H3Session.from_pretrained(
        args.checkpoint,
        args.workspace,
        readers=args.readers,
        adaln_cache=args.adaln_cache,
        text_gpu_gib=args.text_gpu_gib,
        text_cpu_gib=args.text_cpu_gib,
    ) as session:
        result = session.generate(
            args.prompt,
            seed=args.seed,
            steps=args.steps,
            height=args.height,
            width=args.width,
            frames=args.frames,
            plan=args.plan,
        )
        report = {
            "request_id": result.request_id,
            "outputs": {
                k: v for k, v in result.outputs.items() if not hasattr(v, "shape")
            },
            "metadata": dict(result.metadata),
        }
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
