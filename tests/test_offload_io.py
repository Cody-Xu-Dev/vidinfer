"""核对 Accelerate 的磁盘权重读取：CPU/CUDA 数据不变，退出后恢复原函数。"""

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


@unittest.skipUnless(
    all(
        importlib.util.find_spec(name)
        for name in ("torch", "safetensors", "accelerate", "transformers")
    ),
    "Requires CUDA runtime dependencies",
)
class OffloadIOTests(unittest.TestCase):
    def test_pread_offloaded_weights_are_exact(self):
        import torch
        from safetensors.torch import save_file
        from accelerate.utils import offload
        from vidinfer.stages import bounded_checkpoint_io

        with tempfile.TemporaryDirectory() as folder, torch.inference_mode():
            expected = torch.arange(1024, dtype=torch.float32).reshape(32, 32)
            path = Path(folder) / "weights.safetensors"
            save_file({"source": expected}, str(path))
            devices = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])
            original = offload.safe_open
            for device in devices:
                with patch.object(offload, "safe_open", wraps=original) as spy:
                    with bounded_checkpoint_io():
                        loader = offload.OffloadedWeightsLoader(
                            index={
                                "alias": {
                                    "safetensors_file": str(path),
                                    "weight_name": "source",
                                    "dtype": "float32",
                                }
                            },
                            device=device,
                        )
                        actual = loader["alias"]
                    self.assertEqual(spy.call_args.kwargs["backend"], "pread")
                    self.assertTrue(torch.equal(actual.cpu(), expected))
                    self.assertEqual(str(actual.device), device)
            self.assertIs(offload.safe_open, original)

    def test_exception_restores_loader(self):
        from accelerate.utils import offload
        from vidinfer.stages import bounded_checkpoint_io

        original = offload.safe_open
        with self.assertRaisesRegex(ValueError, "test failure"):
            with bounded_checkpoint_io():
                raise ValueError("test failure")
        self.assertIs(offload.safe_open, original)
