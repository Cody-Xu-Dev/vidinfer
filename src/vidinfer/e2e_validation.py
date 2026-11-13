"""集成验证入口；配置来自输出目录的 experiment.json，不内置机器路径。"""

import gc
import json
from pathlib import Path
import subprocess
import time


def main(args, model=None, resident=None):
    import torch
    from .session import H3Session
    from .models.h3 import source_signature, storage_map
    from .types import Cancelled

    spec = json.loads((args.out / "experiment.json").read_text())
    checkpoint, workspace = Path(spec["checkpoint"]), Path(spec["workspace"])
    report = dict(completed=False, resident=resident, runs=[], checks={})

    def save():
        (args.out / "e2e-results.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    before_signature, before_storage = source_signature(model), storage_map(model)
    pig_cache = Path(spec["text_cache"])
    pig = torch.load(pig_cache, map_location="cpu", weights_only=True)["prompt"]
    novel = spec["prompt"]
    sessions = dict(
        workspace=workspace,
        adaln_cache=Path(spec["adaln_cache"]),
        readers=spec.get("readers", 2),
    )
    golden = torch.load(spec["golden"], map_location="cpu", weights_only=True)
    from .models.h3 import tensor_hash

    golden_hash = {k: tensor_hash(v) for k, v in golden.items()}

    def run(session, label, prompt, **options):
        started = time.perf_counter()
        result = session.generate(prompt, **options)
        item = dict(
            label=label,
            wall_s=time.perf_counter() - started,
            metadata=dict(result.metadata),
            outputs={
                k: v for k, v in result.outputs.items() if not hasattr(v, "shape")
            },
        )
        if "video_path" in result.outputs:
            probe = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_streams",
                    "-of",
                    "json",
                    result.outputs["video_path"],
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            streams = json.loads(probe.stdout)["streams"]
            assert any(s["codec_type"] == "video" for s in streams)
            assert any(s["codec_type"] == "audio" for s in streams)
            item["media_probe"] = streams
        report["runs"].append(item)
        save()
        print("E2E_REQUEST_COMPLETE", label, item["wall_s"], flush=True)
        return result

    with H3Session.from_model(
        model,
        checkpoint,
        model_id=resident["model_instance_id"],
        text_cache=pig_cache,
        **sessions,
    ) as session:
        baseline = run(
            session, "packaged-kernels-pig", pig, steps=5, output_type="latent"
        )
        assert baseline.metadata["latent_sha256"] == golden_hash
        report["checks"]["packaged_kernels_exact_historical"] = True
        if spec.get("smoke_only"):
            # 关闭执行队列以等待记录回调落盘；with 退出仍负责关闭阶段模型。
            session.engine.close()
            record = workspace / "requests" / (baseline.request_id + ".json")
            assert json.loads(record.read_text(encoding="utf-8"))["status"]["cleaned"]
            import vidinfer

            if spec.get("package_path"):
                assert (
                    Path(vidinfer.__file__)
                    .resolve()
                    .is_relative_to(Path(spec["package_path"]).resolve())
                )
            assert (
                source_signature(model) == before_signature
                and storage_map(model) == before_storage
            )
            report["checks"].update(
                request_record_written=True,
                installed_package_smoke=True,
                borrowed_model_unchanged=True,
            )
            report["completed"] = True
            save()
            return dict(summary=dict(completed=True, checks=report["checks"]))
        if spec.get("prior_report"):
            prior = json.loads(Path(spec["prior_report"]).read_text())
            assert (
                prior["resident"]["model_instance_id"] == resident["model_instance_id"]
            )
            entries = {item["label"]: item for item in prior["runs"]}
            generated_meta = entries["novel-prompt-29-video"]["metadata"]
            reference_meta = entries["novel-prompt-29-reference"]["metadata"]
            assert Path(
                entries["novel-prompt-29-video"]["outputs"]["video_path"]
            ).is_file()
            assert generated_meta["text_cache_hit"] is False
            report["reused_results_from"] = spec["prior_report"]
            report["reused_runs"] = prior["runs"]
        else:
            generated = run(session, "novel-prompt-29-video", novel, steps=29, seed=43)
            reference = run(
                session,
                "novel-prompt-29-reference",
                novel,
                steps=29,
                seed=43,
                plan="reference",
                output_type="latent",
            )
            generated_meta, reference_meta = generated.metadata, reference.metadata
        assert generated_meta["latent_sha256"] == reference_meta["latent_sha256"]
        assert generated_meta["steps"] == 28 and reference_meta["steps"] == 28
        assert reference_meta["text_cache_hit"]
        report["checks"].update(
            new_prompt_encoded=True, full29_exact_reference=True, text_cache_reused=True
        )
        run(
            session,
            "alternate-geometry-video",
            novel,
            steps=5,
            seed=44,
            height=320,
            width=544,
            frames=124,
        )
        report["checks"]["alternate_geometry_video"] = True

        def cancel(ctx):
            if ctx.steps_done == 1:
                ctx.cancel_event.set()

        handle = session.submit(pig, steps=5, output_type="latent", on_step=cancel)
        try:
            handle.wait()
            raise AssertionError("Cancellation ignored")
        except Cancelled:
            assert handle.context.cleaned and handle.context.completion.ready()
        recovered = run(session, "recovery-pig", pig, steps=5, output_type="latent")
        assert recovered.metadata["latent_sha256"] == golden_hash
        report["checks"]["cancel_recovery"] = True
    del handle, session
    gc.collect()
    torch.cuda.empty_cache()
    assert (
        source_signature(model) == before_signature
        and storage_map(model) == before_storage
    )
    report["checks"]["borrowed_model_unchanged"] = True
    save()
    if spec.get("cold_load", True):
        # 保留原常驻模型；第二份仅用于独立加载/去噪验证，不加载文本编码器或 VAE。
        with H3Session.from_pretrained(
            checkpoint, text_cache=pig_cache, **sessions
        ) as cold:
            result = run(
                cold, "standalone-cold-load-pig", pig, steps=5, output_type="latent"
            )
            assert result.metadata["latent_sha256"] == golden_hash
        del cold
        gc.collect()
        torch.cuda.empty_cache()
        report["checks"]["standalone_cold_load_exact"] = True
    assert (
        source_signature(model) == before_signature
        and storage_map(model) == before_storage
    )
    report["completed"] = True
    save()
    return dict(summary=dict(completed=True, checks=report["checks"]))
