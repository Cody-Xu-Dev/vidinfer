"""Bounded-memory checkpoint I/O for a pinned Diffusers diagnostic.
Uses the original Diffusers model class and its tensor-placement/precision helper.
No neural computation or weight transformation is changed.
"""

import concurrent.futures, json, struct, threading, time
from pathlib import Path
import torch
from accelerate import init_empty_weights
from diffusers import MiniMaxH3Transformer3DModel
from diffusers.models.model_loading_utils import load_model_dict_into_meta
from safetensors import safe_open


def empty_model(folder):
    config = MiniMaxH3Transformer3DModel.load_config(folder, local_files_only=True)
    old = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with init_empty_weights(include_buffers=False):
            model = MiniMaxH3Transformer3DModel.from_config(config)
    finally:
        torch.set_default_dtype(old)
    return model


def place(model, key, tensor):
    target = (
        torch.float32
        if any(n in key.split(".") for n in model._keep_in_fp32_modules)
        else torch.bfloat16
    )
    typed = (tensor.to(target) if tensor.is_floating_point() else tensor).contiguous()
    load_model_dict_into_meta(
        model,
        {key: typed},
        dtype=torch.bfloat16,
        model_name_or_path="MiniMaxH3Transformer3DModel",
        keep_in_fp32_modules=model._keep_in_fp32_modules,
        device_map={"": "cuda:0"},
        unexpected_keys=[],
    )
    # The helper uses non-blocking transfers. Retain the CPU tensor until this copy finishes.
    torch.cuda.current_stream().synchronize()


def load_transformer_streaming(root, workers=4):
    assert 1 <= workers <= 4
    folder = Path(root) / "transformer"
    index = json.loads(
        (folder / "diffusion_pytorch_model.safetensors.index.json").read_text()
    )["weight_map"]
    model = empty_model(folder)
    expected = set(model.state_dict())
    if expected != set(index):
        raise RuntimeError("Checkpoint/model key mismatch")
    loaded = set()
    lock = threading.Lock()
    started = time.perf_counter()

    def shard(name):
        path = folder / name
        with path.open("rb") as f:
            length = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(length))
        keys = sorted(
            (k for k in header if k != "__metadata__"),
            key=lambda k: header[k]["data_offsets"][0],
        )
        with (
            torch.cuda.device(0),
            torch.no_grad(),
            safe_open(
                str(path), framework="pt", device="cpu", backend="pread"
            ) as reader,
        ):
            for key in keys:
                if index.get(key) != name:
                    raise RuntimeError("Index points to another shard: " + key)
                tensor = reader.get_slice(key)[...]
                if tuple(tensor.shape) != tuple(model.state_dict()[key].shape):
                    raise RuntimeError("Shape mismatch: " + key)
                place(model, key, tensor)
                del tensor
                with lock:
                    loaded.add(key)
                    n = len(loaded)
                    if n % 32 == 0 or n == len(expected):
                        print(
                            f"STREAM_LOADED {n}/{len(expected)} tensors; {time.perf_counter() - started:.1f}s",
                            flush=True,
                        )
        print(f"STREAM_SHARD_DONE {name}", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(shard, sorted(set(index.values()))))
    if loaded != expected:
        raise RuntimeError("Incomplete checkpoint load")
    remaining = [n for n, p in model.named_parameters() if p.device.type == "meta"]
    if remaining:
        raise RuntimeError("Unmaterialized parameters: " + str(remaining))
    model.to("cuda:0")
    model.eval()
    model.hf_device_map = {"": "cuda:0"}
    model.register_to_config(_name_or_path=str(folder))
    torch.cuda.synchronize()
    return model
