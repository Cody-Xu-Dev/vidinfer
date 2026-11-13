"""Build a small CUDA C++ extension on NAS and expose an inference-only operator."""

import os
from pathlib import Path
import torch

_EXTENSION = None


def extension():
    global _EXTENSION
    if _EXTENSION is None:
        major, minor = torch.cuda.get_device_capability()
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")
        os.environ.setdefault("MAX_JOBS", "2")
        from torch.utils.cpp_extension import load

        root = Path(__file__).parent
        if not os.environ.get("TORCH_EXTENSIONS_DIR"):
            raise RuntimeError(
                "Configure RuntimeConfig before building CUDA extensions"
            )
        _EXTENSION = load(
            name="h3_qk_norm_rope_cuda_v1",
            sources=[str(root / "qk_norm_rope.cpp"), str(root / "qk_norm_rope.cu")],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--fmad=false", "-lineinfo"],
            verbose=True,
        )
    return _EXTENSION


def fused_qk(q, k, attn, rotary_emb, apply_norm=True, warps=4):
    if torch.is_grad_enabled():
        raise RuntimeError("Use torch.inference_mode() for this experiment")
    cos, sin = rotary_emb
    return extension().forward(
        q,
        k,
        attn.norm_q.weight,
        attn.norm_k.weight,
        cos,
        sin,
        attn.norm_q.eps,
        attn.norm_k.eps,
        apply_norm,
        warps,
    )
