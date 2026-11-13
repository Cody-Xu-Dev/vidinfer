"""运行目录与资源上限；不导入 PyTorch，可在加载模型前配置所有缓存。"""

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile


@dataclass(frozen=True)
class RuntimeConfig:
    workspace: Path
    device: int = 0
    readers: int = 2
    max_pending: int = 4
    max_video_tokens: int = 20000
    max_total_tokens: int = 25000
    max_canvas_pixels: int = 1048576
    max_prompt_chars: int = 4000
    text_gpu_gib: int = 40
    text_cpu_gib: int = 4
    adaln_cache: Path | None = None

    def __post_init__(self):
        object.__setattr__(
            self, "workspace", Path(self.workspace).expanduser().resolve()
        )
        if any(
            type(value) is not int
            for value in (
                self.device,
                self.readers,
                self.max_pending,
                self.max_video_tokens,
                self.max_total_tokens,
                self.max_canvas_pixels,
                self.max_prompt_chars,
                self.text_gpu_gib,
                self.text_cpu_gib,
            )
        ):
            raise ValueError("Resource limits and device index must be integers")
        if self.device != 0:
            raise ValueError("H3 loading currently supports device=0 only")
        if not 1 <= self.readers <= 4 or self.max_pending < 1:
            raise ValueError("Use 1..4 readers and max_pending >= 1")
        if (
            min(
                self.max_video_tokens,
                self.max_total_tokens,
                self.max_canvas_pixels,
                self.max_prompt_chars,
                self.text_gpu_gib,
                self.text_cpu_gib,
            )
            < 1
        ):
            raise ValueError("Resource limits must be positive")

    def configure(self):
        """显式工作目录是唯一写入根；在 H20 上必须传 NAS 路径。"""
        folders = {
            "TMPDIR": "tmp",
            "PIP_CACHE_DIR": "cache/pip",
            "HF_HOME": "cache/huggingface",
            "HF_HUB_CACHE": "cache/huggingface/hub",
            "TRITON_CACHE_DIR": "cache/triton",
            "CUDA_CACHE_PATH": "cache/cuda",
            "TORCH_EXTENSIONS_DIR": "cache/torch_extensions",
            "XDG_CACHE_HOME": "cache/xdg",
        }
        for variable, relative in folders.items():
            path = self.workspace / relative
            path.mkdir(parents=True, exist_ok=True)
            os.environ[variable] = str(path)
        tempfile.tempdir = os.environ["TMPDIR"]
        cache = self.adaln_cache or self.workspace / "cache/adaln_h3_diffusers"
        cache = Path(cache).expanduser().resolve()
        cache.mkdir(parents=True, exist_ok=True)
        os.environ["VIDINFER_ADALN_CACHE"] = str(cache)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        os.environ.setdefault("MAX_JOBS", "2")
        return self

    def output_directory(self, request_id):
        """只允许简单请求 ID，防止 ID 被解释成其他目录的路径。"""
        import re

        if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", request_id):
            raise ValueError("request_id must contain only letters, digits, '_' or '-'")
        return self.workspace / "outputs" / request_id
