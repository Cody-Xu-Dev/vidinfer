"""Exact, schedule-specific AdaLN tables for the pinned official H3 t2va model.

No vendor source modifications. Checkpoints and cache artifacts stay on NAS.
Run directly to build; import load_transformer_cached for experiments.
"""

import argparse
import concurrent.futures
import contextvars
import hashlib
import inspect
import json
import os
import struct
from pathlib import Path
import time
import uuid

import torch
from torch import nn
from safetensors import safe_open
from safetensors.torch import save_file
from diffusers import MiniMaxH3Scheduler
from diffusers.models.transformers import transformer_minimax_h3 as vendor
from diffusers.modular_pipelines.minimax_h3.before_denoise import (
    MiniMaxH3SetTimestepsStep,
)
from .streaming import empty_model, place

SCHEMA = 1


def cache_parent(root):
    return Path(
        os.environ.get(
            "VIDINFER_ADALN_CACHE", Path(root).parent / "cache/adaln_h3_diffusers"
        )
    )


def digest(data):
    return hashlib.sha256(data).hexdigest()


def timestep_key(t):
    if t.dtype != torch.float32 or t.ndim != 1:
        raise ValueError("Cached H3 requires a 1-D FP32 unique timestep vector")
    return digest(t.detach().contiguous().cpu().numpy().tobytes())


def identity(root):
    root = Path(root).resolve()
    index = json.loads(
        (
            root / "transformer/diffusion_pytorch_model.safetensors.index.json"
        ).read_text()
    )["weight_map"]
    files = [
        "transformer/config.json",
        "transformer/diffusion_pytorch_model.safetensors.index.json",
        "scheduler/scheduler_config.json",
        "audio_scheduler/scheduler_config.json",
    ]
    small = {n: digest((root / n).read_bytes()) for n in files}
    shards = {}
    for name in sorted(set(index.values())):
        st = (root / "transformer" / name).stat()
        shards[name] = [st.st_size, st.st_mtime_ns, st.st_ctime_ns]
    source = "\n".join(
        inspect.getsource(c)
        for c in (
            vendor.MiniMaxH3AdaLayerNormModulation,
            vendor.MiniMaxH3AdaLayerNormOut,
            MiniMaxH3Scheduler,
            MiniMaxH3SetTimestepsStep,
        )
    )
    # Time embedding implementation and numerical contract are also versioned.
    source += Path(inspect.getfile(vendor)).read_text()
    prop = torch.cuda.get_device_properties(0)
    return dict(
        schema=SCHEMA,
        checkpoint=str(root),
        files=small,
        shards=shards,
        math_source_sha256=digest(source.encode()),
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),
        gpu=[prop.name, prop.major, prop.minor, prop.multi_processor_count],
        tf32=torch.backends.cuda.matmul.allow_tf32,
        bf16_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        fp32_precision=torch.get_float32_matmul_precision(),
    )


def plan(root, steps):
    schedulers = [
        MiniMaxH3Scheduler.from_pretrained(
            str(root), subfolder=n, local_files_only=True
        )
        for n in ("scheduler", "audio_scheduler")
    ]
    for s in schedulers:
        s.set_timesteps(steps, device="cpu")
    result = []
    for tv, ta in zip(*(s.timesteps for s in schedulers)):
        values, _ = MiniMaxH3SetTimestepsStep.build_row_timesteps(
            video_indices=torch.tensor([1]),
            audio_indices=torch.tensor([2, 3]),
            num_condition_video_rows=0,
            num_condition_audio_rows=0,
            num_text_tokens=1,
            video_timestep=float(tv),
            audio_timestep=float(ta),
            condition_video_timestep=max(float(tv), 0.0),
            condition_audio_timestep=1.0,
        )
        result.append(values)
    return result


def cache_dir(root, profiles):
    tag = digest(
        json.dumps(
            dict(identity=identity(root), profiles=sorted(profiles)), sort_keys=True
        ).encode()
    )[:24]
    return cache_parent(root) / tag


def read_tensors(folder, index, keys):
    result = {}
    for name in sorted({index[k] for k in keys}):
        with safe_open(
            str(folder / name), framework="pt", device="cpu", backend="pread"
        ) as f:
            for k in keys:
                if index[k] == name:
                    result[k] = f.get_slice(k)[...]
    return result


@torch.inference_mode()
def build(root, profiles=(5, 29), workers=4):
    if not 1 <= workers <= 4 or not profiles or any(n < 2 for n in profiles):
        raise ValueError(
            "Use 1..4 readers and scheduler profiles of at least two points"
        )
    root = Path(root)
    target = cache_dir(root, profiles)
    if (target / "manifest.json").exists():
        verify(target, root)
        print("REUSE_CACHE", target, flush=True)
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = target.parent / (target.name + ".building-" + uuid.uuid4().hex)
    stage.mkdir()
    started = time.perf_counter()
    ident = identity(root)
    vectors, profile_keys = {}, {}
    for steps in profiles:
        values = plan(root, steps)
        profile_keys[str(steps)] = [timestep_key(v) for v in values]
        vectors.update({timestep_key(v): v for v in values})
    folder = root / "transformer"
    index = json.loads(
        (folder / "diffusion_pytorch_model.safetensors.index.json").read_text()
    )["weight_map"]
    model = empty_model(folder)
    time_keys = [k for k in index if k.startswith("time_embedder.")]
    for k, t in read_tensors(folder, index, time_keys).items():
        place(model, k, t)
    tembs = {
        k: model.time_embedder(model.time_proj(v.cuda()).float())
        for k, v in vectors.items()
    }
    tables = {}
    jobs = []
    for i, block in enumerate(model.transformer_blocks):
        jobs.append(
            (
                f"block{i}",
                block.adaln_proj,
                [
                    f"transformer_blocks.{i}.adaln_proj.linear.{s}"
                    for s in ("weight", "bias")
                ],
            )
        )
    jobs.append(
        ("out", model.norm_out, ["norm_out.linear.weight", "norm_out.linear.bias"])
    )
    skipped, source_hashes = [], {}
    # Bounded prefetch: at most workers weights in CPU RAM; GPU holds one projection.
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {}
        for j in range(min(workers, len(jobs))):
            pending[j] = pool.submit(read_tensors, folder, index, jobs[j][2])
        for j, (label, module, keys) in enumerate(jobs):
            tensors = pending.pop(j).result()
            for key, t in tensors.items():
                source_hashes[key] = digest(memoryview(t.view(torch.uint8).numpy()))
                setattr(
                    module.linear,
                    key.rsplit(".", 1)[1],
                    nn.Parameter(t.cuda(), requires_grad=False),
                )
            skipped.extend(keys)
            for k, temb in tembs.items():
                if label == "out":
                    value = module.linear(
                        torch.nn.functional.silu(temb).to(module.linear.weight.dtype)
                    )
                else:
                    value = torch.cat(module(temb), dim=-1)
                tables[f"{label}.{k}"] = value.cpu().contiguous()
            torch.cuda.synchronize()
            module.linear.to_empty(device="meta")
            del tensors, value
            if j + workers < len(jobs):
                n = j + workers
                pending[n] = pool.submit(read_tensors, folder, index, jobs[n][2])
            print(
                "CACHE_PROJECTION",
                label,
                f"{time.perf_counter() - started:.1f}s",
                flush=True,
            )
    for k, v in tembs.items():
        tables[f"temb.{k}"] = v.cpu().contiguous()
    save_file(tables, str(stage / "tables.safetensors"))
    with safe_open(
        str(stage / "tables.safetensors"), framework="pt", device="cpu", backend="pread"
    ) as f:
        assert set(f.keys()) == set(tables)
        for k, t in tables.items():
            assert torch.equal(t, f.get_tensor(k)), k
    if identity(root) != ident:
        raise RuntimeError("Checkpoint/environment changed during cache construction")
    manifest = dict(
        identity=ident,
        profiles=profile_keys,
        skipped=skipped,
        source_tensor_sha256=source_hashes,
        timesteps={k: v.tolist() for k, v in vectors.items()},
        tensor_sha256={
            k: digest(memoryview(t.view(torch.uint8).numpy()))
            for k, t in tables.items()
        },
        cache_bytes=sum(t.numel() * t.element_size() for t in tables.values()),
        build_wall_s=time.perf_counter() - started,
        scope="t2va without conditioning; fixed checkpoint; inference only",
    )
    (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    os.rename(stage, target)
    print("CACHE_COMPLETE", target, manifest["cache_bytes"], flush=True)
    return target


def verify(path, root):
    manifest = json.loads((Path(path) / "manifest.json").read_text())
    if manifest["identity"] != identity(root):
        raise RuntimeError(
            "AdaLN cache checkpoint/software/precision identity mismatch; rebuild cache"
        )
    return manifest


class Router:
    def __init__(self, keys):
        self.keys = set(keys)
        self.active = contextvars.ContextVar("h3_adaln_entry", default=None)
        self.tokens = contextvars.ContextVar("h3_adaln_tokens", default=())

    def pre(self, model, args, kwargs):
        # Always-call post hooks run even when this pre hook raises.
        # A sentinel ensures a rejected nested call cannot reset its outer call.
        self.tokens.set(self.tokens.get() + (None,))
        if (
            model.training
            or torch.is_grad_enabled()
            or getattr(model, "_hf_peft_config_loaded", False)
            or getattr(model, "peft_config", None)
        ):
            raise RuntimeError(
                "AdaLN cache supports frozen inference only, without LoRA"
            )
        timestep = kwargs.get("timestep")
        if timestep is None:
            timestep = (
                inspect.signature(model.forward)
                .bind_partial(*args, **kwargs)
                .arguments["timestep"]
            )
        key = timestep_key(timestep)
        if key not in self.keys:
            raise RuntimeError(
                "Unknown H3 timestep vector; build a matching cache profile"
            )
        token = self.active.set(key)
        self.tokens.set(self.tokens.get()[:-1] + (token,))

    def post(self, model, args, kwargs, output):
        tokens = self.tokens.get()
        if tokens:
            if tokens[-1] is not None:
                self.active.reset(tokens[-1])
            self.tokens.set(tokens[:-1])


class CachedProjection(nn.Module):
    def __init__(self, tables, router):
        super().__init__()
        self.router = router
        for key, value in tables.items():
            self.register_buffer("t_" + key, value, persistent=False)

    def table(self, temb):
        key = self.router.active.get()
        if key is None:
            raise RuntimeError(
                "Cached projection must execute inside its model request"
            )
        table = getattr(self, "t_" + key)
        if (
            temb.shape[0] * 3 != table.shape[0]
            or temb.dtype != torch.float32
            or table.dtype != torch.bfloat16
            or table.device != temb.device
        ):
            raise RuntimeError("Cached modulation input contract changed")
        return table

    def forward(self, temb):
        return self.table(temb).chunk(6, dim=-1)


class CachedOutput(CachedProjection):
    def __init__(self, norm, tables, router):
        super().__init__(tables, router)
        self.norm = norm

    def forward(self, hidden_states, temb, timestep_indices):
        key = self.router.active.get()
        if key is None:
            raise RuntimeError("Cached final norm outside model request")
        table = getattr(self, "t_" + key)
        if (
            table.shape[0] != temb.shape[0]
            or temb.dtype != torch.float32
            or table.dtype != torch.bfloat16
            or table.device != temb.device
        ):
            raise RuntimeError("Cached final norm input contract changed")
        shift, scale = table.chunk(2, dim=-1)
        hidden_states = self.norm(hidden_states)
        return hidden_states * (
            1.0 + scale.index_select(0, timestep_indices)
        ) + shift.index_select(0, timestep_indices)


def find_cache(root, steps):
    parent = cache_parent(root)
    for path in sorted(parent.glob("*/manifest.json")):
        try:
            m = verify(path.parent, root)
            if str(steps) in m["profiles"]:
                return path.parent
        except RuntimeError:
            pass
    raise RuntimeError(
        f"No valid AdaLN cache for steps={steps}; run h3_adaln_cache.py --steps {steps}"
    )


@torch.inference_mode()
def load_transformer_cached(root, steps=5, workers=4, cache=None):
    if not 1 <= workers <= 4:
        raise ValueError("Use 1..4 bounded-memory checkpoint readers")
    started = time.perf_counter()
    root = Path(root)
    cache = Path(cache) if cache else find_cache(root, steps)
    manifest = verify(cache, root)
    keys = set(manifest["profiles"][str(steps)])
    actual = [timestep_key(t) for t in plan(root, steps)]
    if actual != manifest["profiles"][str(steps)]:
        raise RuntimeError("Actual scheduler plan differs from cache")
    folder = root / "transformer"
    index = json.loads(
        (folder / "diffusion_pytorch_model.safetensors.index.json").read_text()
    )["weight_map"]
    model = empty_model(folder)
    router = Router(keys)
    cache_bytes = 0
    with safe_open(
        str(cache / "tables.safetensors"), framework="pt", device="cpu", backend="pread"
    ) as f:

        def read(label):
            nonlocal cache_bytes
            result = {}
            for key in keys:
                name = label + "." + key
                value = f.get_tensor(name)
                if (
                    digest(memoryview(value.view(torch.uint8).numpy()))
                    != manifest["tensor_sha256"][name]
                ):
                    raise RuntimeError("Corrupt cache table: " + name)
                result[key] = value.cuda()
                cache_bytes += value.numel() * value.element_size()
            return result

        for i, block in enumerate(model.transformer_blocks):
            block.adaln_proj = CachedProjection(read(f"block{i}"), router)
        model.norm_out = CachedOutput(model.norm_out.norm, read("out"), router)
    expected = set(model.state_dict())
    if expected != set(index) - set(manifest["skipped"]):
        raise RuntimeError("Cached model/checkpoint key mismatch")
    shapes = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    groups = {
        name: [k for k in expected if index[k] == name] for name in set(index.values())
    }

    def shard(item):
        name, names = item
        with (folder / name).open("rb") as stream:
            header = json.loads(stream.read(struct.unpack("<Q", stream.read(8))[0]))
        names = sorted(names, key=lambda k: header[k]["data_offsets"][0])
        with (
            torch.cuda.device(0),
            torch.inference_mode(),
            safe_open(
                str(folder / name), framework="pt", device="cpu", backend="pread"
            ) as f,
        ):
            for k in names:
                t = f.get_slice(k)[...]
                if tuple(t.shape) != shapes[k]:
                    raise RuntimeError("Shape mismatch: " + k)
                place(model, k, t)
                del t
        print(
            "CACHED_LOAD_SHARD",
            name,
            f"{time.perf_counter() - started:.1f}s",
            flush=True,
        )
        return sum(
            v["data_offsets"][1] - v["data_offsets"][0]
            for k, v in header.items()
            if k in manifest["skipped"]
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        skipped_bytes = sum(pool.map(shard, sorted(groups.items())))
    if any(p.is_meta for p in model.parameters()):
        raise RuntimeError("Unmaterialized cached-model parameters")
    model.to("cuda:0").eval()
    model.hf_device_map = {"": "cuda:0"}
    model.register_to_config(_name_or_path=str(folder))
    model.register_forward_pre_hook(router.pre, with_kwargs=True)
    model.register_forward_hook(router.post, with_kwargs=True, always_call=True)
    model._adaln_cache_router = router
    torch.cuda.synchronize()
    model._adaln_cache_stats = dict(
        path=str(cache),
        steps=steps,
        table_gpu_bytes=cache_bytes,
        parameter_gpu_bytes=sum(
            p.numel() * p.element_size() for p in model.parameters()
        ),
        skipped_keys=len(manifest["skipped"]),
        skipped_checkpoint_bytes=skipped_bytes,
        static_allocated_gib=torch.cuda.memory_allocated() / 2**30,
        load_wall_s=time.perf_counter() - started,
    )
    print("CACHED_MODEL_READY", json.dumps(model._adaln_cache_stats), flush=True)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=[5, 29])
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 5))
    a = parser.parse_args()
    build(a.root, a.steps, a.workers)
