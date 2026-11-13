"""报告链路的统计、安全与来源边界；无需 Torch/GPU。"""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

from vidinfer.reporting.analysis import analyze_trace, union_duration
from vidinfer.reporting.render import inference_html, optimization_html
from vidinfer.reporting.workflow import build_workflow


def event(name, start, duration, cat="kernel", stream=1):
    return dict(
        name=name,
        ts=start,
        dur=duration,
        cat=cat,
        ph="X",
        args={"stream": stream, "device": 0},
    )


class ReportingTests(unittest.TestCase):
    def fixture(self, directory, events):
        path = Path(directory) / "trace.json"
        path.write_text(json.dumps({"traceEvents": events}), encoding="utf-8")
        return path

    def test_overlapping_streams_do_not_turn_sum_into_span(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.fixture(
                directory,
                [
                    event("flash", 100, 10),
                    event("gemm", 105, 10, stream=2),
                    event("copy", 120, 5, "gpu_memcpy"),
                ],
            )
            report = analyze_trace(path, scope="test")
        trace = report["trace"]
        self.assertEqual(trace["kernel_sum_ms"], 0.02)
        self.assertEqual(trace["kernel_union_ms"], 0.015)
        self.assertEqual(trace["activity_span_ms"], 0.025)
        self.assertEqual(trace["activity_union_ms"], 0.02)
        self.assertAlmostEqual(trace["observed_gap_fraction"], 0.2)
        self.assertEqual(union_duration([(0, 5), (2, 3), (5, 9)]), 9)

    def test_scope_uses_gpu_annotation_and_sampling_does_not_change_totals(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.fixture(
                directory,
                [event("op", i * 10, 5) for i in range(8)]
                + [
                    event("h3/block_00", 0, 1, "user_annotation"),
                    event("h3/block_00", 0, 80, "gpu_user_annotation"),
                ],
            )
            report = analyze_trace(path, timeline_limit=3)
        self.assertEqual(report["trace"]["blocks"][0]["duration_us"], 80)
        self.assertEqual(report["trace"]["kernel_sum_ms"], 0.04)
        self.assertEqual(len(report["trace"]["timeline"]), 3)
        self.assertTrue(report["trace"]["timeline_sampled"])

    def test_invalid_events_and_empty_trace_are_not_valid_results(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.fixture(
                directory,
                [
                    event("valid", -10, 1),
                    event("bad", 0, -1),
                    event("nan", 0, float("nan")),
                ],
            )
            report = analyze_trace(path)
            self.assertEqual(report["trace"]["skipped_invalid_events"], 2)
            self.assertEqual(report["trace"]["kernel_count"], 1)
            path = self.fixture(directory, [])
            with self.assertRaises(ValueError):
                analyze_trace(path)

    def test_legacy_and_request_stage_units_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.fixture(directory, [event("flash", 100, 10)])
            summary = Path(directory) / "summary.json"
            summary.write_text(
                json.dumps(
                    dict(
                        phases={
                            "load_text": {"wall_s": 2, "peak_allocated_gib": 1},
                            "denoise": {"wall_s": 3},
                        },
                        forwards=[{"index": 1, "gpu_ms": 7, "profiled": True}],
                    )
                )
            )
            report = analyze_trace(path, summary_path=summary)
            self.assertEqual(report["stages"][0]["wall_ms"], 2000)
            self.assertEqual(report["stages"][0]["peak_allocated_bytes"], 2**30)
            self.assertTrue(report["forwards"][0]["profiled"])
            summary.write_text(
                json.dumps(
                    dict(
                        request={"seed": 1},
                        metadata={
                            "stages": {"decode": {"wall_s": 1}},
                            "forward_ms": [1, 2],
                        },
                    )
                )
            )
            report = analyze_trace(path, summary_path=summary)
            self.assertEqual(report["forwards"][1]["elapsed_ms"], 2)

    def test_embedded_json_and_template_placeholders_cannot_inject_script(self):
        attack = "</script><img src=x onerror=alert(99)>{{SCRIPT}}"
        with tempfile.TemporaryDirectory() as directory:
            path = self.fixture(directory, [event(attack, 0, 1)])
            report = analyze_trace(path, scope=attack)
            html = inference_html(report)
        self.assertNotIn("<img src=x", html)
        self.assertIn("\\u003c/script", html)
        self.assertIn("{{SCRIPT}}", html)

    def test_optimization_failure_is_not_promoted_to_adoption(self):
        report = dict(
            schema_version=1,
            completed=True,
            created_at="test",
            spec=dict(
                operation="attention",
                reference="a",
                mode="allclose",
                atol=0.001,
                rtol=0.01,
                source={"kind": "synthetic"},
            ),
            candidates={
                "a": {
                    "status": "available",
                    "correctness": {"passes": False, "outputs": {}},
                }
            },
        )
        html = optimization_html(report)
        self.assertIn("不能按本轮标准自动启用", html)
        self.assertIn('data-pass="false"', html)

    def test_workflow_copies_explicit_evidence_and_preserves_unverified_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root, [event("flash", 0, 1)])
            (root / "evidence.json").write_text('{"claim":"old run"}')
            manifest = root / "workflow.json"
            manifest.write_text(
                json.dumps(
                    dict(
                        schema_version=1,
                        inferences=[{"trace": "trace.json"}],
                        evidence=[
                            {
                                "file": "evidence.json",
                                "status": "historical",
                                "note": "different config",
                            },
                            {"level": "quality", "status": "pending"},
                        ],
                    )
                )
            )
            output = build_workflow(manifest, root / "out")
            data = json.loads((output / "workflow.json").read_text(encoding="utf-8"))
            self.assertTrue((output / "index.html").exists())
            self.assertTrue((output / "inference-01.html").exists())
            self.assertTrue((output / "data/evidence-01.json").exists())
            self.assertEqual(data["evidence"][1]["status"], "pending")
            self.assertIn(
                "file existence/hash only", data["evidence"][0]["verification"]
            )
            with self.assertRaises(FileExistsError):
                build_workflow(manifest, output)

    def test_analysis_and_render_import_without_torch(self):
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import vidinfer.reporting.analysis, vidinfer.reporting.render, vidinfer.reporting.capture; assert 'torch' not in sys.modules",
            ],
            check=True,
        )

    def test_capture_preserves_original_exception_if_export_also_fails(self):
        from vidinfer.reporting.capture import capture_profile

        class FakeProfile:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def export_chrome_trace(self, path):
                raise RuntimeError("export failed")

        fake = SimpleNamespace(
            __version__="test",
            version=SimpleNamespace(cuda=None),
            profiler=SimpleNamespace(
                ProfilerActivity=SimpleNamespace(CPU="cpu"),
                profile=lambda **kwargs: FakeProfile(),
                record_function=lambda label: nullcontext(),
            ),
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(sys.modules, {"torch": fake}),
        ):
            folder = Path(directory) / "capture"
            with self.assertRaisesRegex(ValueError, "original"):
                with capture_profile(folder, cuda=False):
                    raise ValueError("original")
            record = json.loads((folder / "capture.json").read_text())
            self.assertFalse(record["completed"])
            self.assertIn("export failed", record["export_error"])

    def test_capture_export_failure_is_reported_after_successful_call(self):
        from vidinfer.reporting.capture import capture_profile

        class FakeProfile:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def export_chrome_trace(self, path):
                raise RuntimeError("export failed")

        fake = SimpleNamespace(
            __version__="test",
            version=SimpleNamespace(cuda=None),
            profiler=SimpleNamespace(
                ProfilerActivity=SimpleNamespace(CPU="cpu"),
                profile=lambda **kwargs: FakeProfile(),
                record_function=lambda label: nullcontext(),
            ),
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(sys.modules, {"torch": fake}),
        ):
            folder = Path(directory) / "capture"
            with self.assertRaisesRegex(RuntimeError, "export failed"):
                with capture_profile(folder, cuda=False):
                    pass
            self.assertFalse(
                json.loads((folder / "capture.json").read_text())["completed"]
            )


if __name__ == "__main__":
    unittest.main()
