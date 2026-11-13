"""第二层：复用历史实验的算子边界。这里不加载 H3 模型，也不修改常驻实例。"""

import hashlib
import json
from pathlib import Path


def tensor_info(tensor):
    return dict(
        shape=list(tensor.shape),
        stride=list(tensor.stride()),
        dtype=str(tensor.dtype),
        device=str(tensor.device),
        contiguous=tensor.is_contiguous(),
    )


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_case(spec, config_directory):
    import torch

    source = spec.source
    kind = source.get("kind", "synthetic")
    provenance = dict(kind=kind)
    if kind == "tensor_file":
        path = Path(source["path"]).expanduser()
        if not path.is_absolute():
            path = Path(config_directory) / path
        path = path.resolve()
        # 只读本地 Tensor 文件，weights_only 不接受任意 Python 模型对象。
        provenance.update(
            path=str(path), sha256=file_sha256(path), bytes=path.stat().st_size
        )
        data = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(data, dict):
            raise ValueError("Tensor fixture must be a dictionary")
        provenance["capture"] = data.get("metadata", {})
        json.dumps(provenance, allow_nan=False)
    else:
        seed = source.get("seed", 42)
        generator = torch.Generator(device="cuda").manual_seed(seed)
        b, length, heads = (
            source.get("batch", 1),
            source.get("length", 256),
            source.get("heads", 4),
        )
        dim = source.get("head_dim", 128)

        def random(shape, dtype=torch.bfloat16):
            return torch.randn(shape, generator=generator, device="cuda", dtype=dtype)

        if spec.operation == "attention":
            data = {
                name: random((b, length, heads, dim))
                for name in ("query", "key", "value")
            }
        elif spec.operation == "swiglu":
            data = {
                "packed": random(
                    (source.get("rows", 256), 2 * source.get("width", 1024))
                )
            }
        else:
            packed = random((b, length, 3 * heads * dim))
            q, k, _ = (
                part.unflatten(-1, (heads, dim)) for part in packed.chunk(3, dim=-1)
            )
            angle = random((length, 96), torch.float32)
            data = dict(
                q=q,
                k=k,
                weight_q=random((dim,)),
                weight_k=random((dim,)),
                cos=angle.cos(),
                sin=angle.sin(),
                eps_q=1e-5,
                eps_k=1e-5,
            )
        provenance.update(
            seed=seed, description="Synthetic values; not a captured H3 activation"
        )

    # 输入准备在计时之外；显式保留文件中 Tensor 的 stride，包括打包 QKV 切片的空隙。
    def move(t):
        if (
            not isinstance(t, torch.Tensor)
            or t.layout != torch.strided
            or t.numel() == 0
        ):
            raise ValueError("Expected nonempty strided Tensor fixture")
        if not t.is_floating_point() or not torch.isfinite(t).all().item():
            raise ValueError("Expected finite floating point inputs")
        if t.is_cuda:
            return t
        output = torch.empty_strided(t.shape, t.stride(), dtype=t.dtype, device="cuda")
        output.copy_(t)
        return output

    if spec.operation == "attention":
        values = {key: move(data[key]) for key in ("query", "key", "value")}
        q, k, v = values.values()
        if any(t.ndim != 4 or t.dtype != torch.bfloat16 for t in values.values()):
            raise ValueError("Attention expects BF16 BLHD inputs")
        if q.shape != k.shape or k.shape != v.shape:
            raise ValueError(
                "This first Attention experiment requires identical Q/K/V shapes"
            )
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from ..kernels.attention import NativeStrideAttention

        def make(name):
            backend_name = dict(
                flash="FLASH_ATTENTION",
                cudnn="CUDNN_ATTENTION",
                efficient="EFFICIENT_ATTENTION",
            )[name.split("_")[0]]
            backend = getattr(SDPBackend, backend_name)
            if name.endswith("_view"):
                adapter = NativeStrideAttention(backend)
                return lambda: {"output": adapter(q, k, v)}

            def contiguous():
                # 转换不可提前到这里外面；调用者付出的转换成本属于这条路径。
                x, y, z = (t.permute(0, 2, 1, 3).contiguous() for t in (q, k, v))
                with sdpa_kernel(backend):
                    output = torch.nn.functional.scaled_dot_product_attention(
                        x, y, z, dropout_p=0.0, is_causal=False
                    )
                return {"output": output.permute(0, 2, 1, 3).contiguous()}

            return contiguous

        functions = {name: make(name) for name in spec.candidates}
        contract = "BF16 BLHD -> contiguous BLHD; full unmasked attention, dropout=0; input/output layout copies included"
    elif spec.operation == "swiglu":
        values = {"packed": move(data["packed"])}
        packed = values["packed"]
        if packed.dtype != torch.bfloat16 or packed.ndim < 2 or packed.shape[-1] % 2:
            raise ValueError("SwiGLU expects BF16 [..., 2*width] packed [a,gate]")
        from ..kernels.swiglu_cuda import fused_swiglu

        def original():
            a, gate = packed.chunk(2, dim=-1)
            return {"output": (a * torch.nn.functional.silu(gate)).contiguous()}

        available = {
            "torch": original,
            "cuda": lambda: {"output": fused_swiglu(packed)},
        }
        functions = {name: available[name] for name in spec.candidates}
        contract = "BF16 packed [a,gate] -> contiguous a*SiLU(gate); projection excluded for both implementations"
    else:
        from types import SimpleNamespace
        from diffusers.models.transformers import transformer_minimax_h3 as vendor
        from ..kernels.qk_norm_rope import fused_qk

        keys = ("q", "k", "weight_q", "weight_k", "cos", "sin")
        values = {key: move(data[key]) for key in keys}
        q, k, wq, wk, cos, sin = values.values()
        if q.ndim != 4 or q.shape != k.shape or q.shape[-1] != 128:
            raise ValueError("QK expects equal BLH128 shapes")
        if any(t.dtype != torch.bfloat16 for t in (q, k, wq, wk)):
            raise ValueError("QK and norm weights must be BF16")
        if (
            tuple(wq.shape) != (128,)
            or tuple(wk.shape) != (128,)
            or cos.shape != sin.shape
            or tuple(cos.shape) != (q.shape[1], 96)
        ):
            raise ValueError("H3 QK expects norm weights[128], cos/sin[L,96]")
        if cos.dtype != torch.float32 or sin.dtype != torch.float32:
            raise ValueError("RoPE tables must be FP32")
        eq, ek = float(data.get("eps_q", 1e-5)), float(data.get("eps_k", 1e-5))
        import math

        if not all(math.isfinite(e) and e > 0 for e in (eq, ek)):
            raise ValueError("Norm eps must be finite and positive")
        attn = SimpleNamespace(
            norm_q=SimpleNamespace(weight=wq, eps=eq),
            norm_k=SimpleNamespace(weight=wk, eps=ek),
        )

        def original():
            a = torch.nn.functional.rms_norm(q, (128,), wq, eq)
            b = torch.nn.functional.rms_norm(k, (128,), wk, ek)
            return {
                "q": vendor._apply_rotary_emb(a, cos, sin).contiguous(),
                "k": vendor._apply_rotary_emb(b, cos, sin).contiguous(),
            }

        def fused(warps):
            a, b = fused_qk(q, k, attn, (cos, sin), warps=warps)
            return {"q": a, "k": b}

        available = {
            "torch": original,
            "cuda_w4": lambda: fused(4),
            "cuda_w8": lambda: fused(8),
        }
        functions = {name: available[name] for name in spec.candidates}
        contract = "BF16 Q/K + RMSNorm + FP32 partial half-split RoPE96 -> contiguous BF16 Q/K; projections/attention excluded"
        provenance.update(eps_q=eq, eps_k=ek)
    provenance["tensors"] = {key: tensor_info(tensor) for key, tensor in values.items()}
    return functions, provenance, contract
