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
        if not os.environ.get("TORCH_EXTENSIONS_DIR"):
            raise RuntimeError(
                "Configure RuntimeConfig before building CUDA extensions"
            )
        from torch.utils.cpp_extension import load

        root = Path(__file__).parent
        _EXTENSION = load(
            name="h3_swiglu_cuda_v1",
            sources=[str(root / "swiglu_cuda.cpp"), str(root / "swiglu_cuda.cu")],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--fmad=false", "-lineinfo"],
            verbose=True,
        )
        # Materialize the scalar entry before the paired entry on first load.
        # Initialization stays outside benchmarks; avoids a driver/tool handle
        # probe event observed with paired-first launch on this CUDA12.9 stack.
        with torch.inference_mode():
            _EXTENSION.forward(torch.zeros((1, 2), device="cuda", dtype=torch.bfloat16))
            torch.cuda.current_stream().synchronize()
    return _EXTENSION


def fused_swiglu(packed):
    if torch.is_grad_enabled():
        raise RuntimeError("Use torch.inference_mode()")
    return extension().forward(packed)
