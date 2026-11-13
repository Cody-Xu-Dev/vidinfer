"""资源上限、路径约束与包依赖边界，不需要 PyTorch 或真实模型。"""

from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from vidinfer import DenoiseRequest, RuntimeConfig
from vidinfer.validation import validate_generation_request


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.config = RuntimeConfig(Path(self.folder.name))

    def test_all_caches_are_inside_explicit_workspace(self):
        """防止依赖库把缓存和编译产物写回容器默认目录。"""
        with (
            patch.dict(os.environ, clear=False),
            patch.object(tempfile, "tempdir", tempfile.tempdir),
        ):
            self.config.configure()
            for key in (
                "TMPDIR",
                "HF_HOME",
                "HF_HUB_CACHE",
                "TORCH_EXTENSIONS_DIR",
                "TRITON_CACHE_DIR",
                "VIDINFER_ADALN_CACHE",
            ):
                self.assertTrue(
                    Path(os.environ[key]).is_relative_to(self.config.workspace)
                )
                self.assertTrue(Path(os.environ[key]).is_dir())

    def test_geometry_and_token_budget(self):
        validate_generation_request(DenoiseRequest("p"), self.config)
        validate_generation_request(
            DenoiseRequest("p", height=320, width=544, frames=124, steps=29),
            self.config,
        )
        for changes in (
            {"height": 543},
            {"frames": 123},
            {"frames": 22},
            {"frames": 362},
            {"steps": 12},
            {"seed": True},
            {"height": 2048, "width": 2048},
            {"height": 32, "width": 960},
            {"height": 32, "width": 32, "frames": 51005},
            {"height": 1536, "width": 1536, "frames": 5},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_generation_request(
                    replace(DenoiseRequest("p"), **changes), self.config
                )

    def test_request_id_cannot_escape_output_root(self):
        for request_id in ("../outside", "/tmp/escape", "a/b", "", "a" * 101):
            with self.subTest(request_id=request_id), self.assertRaises(ValueError):
                self.config.output_directory(request_id)
        self.assertEqual(
            self.config.output_directory("request-1").parent.name, "outputs"
        )

    def test_prompt_limit_and_bad_limits(self):
        with self.assertRaises(ValueError):
            validate_generation_request(DenoiseRequest("x" * 4001), self.config)
        for prompt in (None, 123, "", "   "):
            with self.subTest(prompt=prompt), self.assertRaises(ValueError):
                validate_generation_request(DenoiseRequest(prompt), self.config)
        with self.assertRaises(ValueError):
            validate_generation_request(
                DenoiseRequest("p", output_type="unknown"), self.config
            )
        for changes in (
            {"readers": 8},
            {"max_pending": 0},
            {"text_cpu_gib": 0},
            {"device": 1},
            {"readers": 2.5},
            {"max_pending": True},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.config, **changes)

    def test_import_and_cli_help_do_not_load_torch(self):
        """CPU 机器也能检查 API、帮助和核心测试。"""
        run = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys,vidinfer; assert 'torch' not in sys.modules; print(vidinfer.H3Session.__name__)",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(run.returncode, 0, run.stderr)
        help_run = subprocess.run(
            [sys.executable, "-m", "vidinfer", "generate", "--help"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(help_run.returncode, 0, help_run.stderr)
        self.assertIn("--workspace", help_run.stdout)


if __name__ == "__main__":
    unittest.main()
