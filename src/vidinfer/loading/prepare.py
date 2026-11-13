"""独立启动时的权重准备，复用已验证的加载与共享 QKV 打包算法。"""

import gc


def pack_qkv_shared(model):
    import torch

    if getattr(model, "_resident_qkv_shared", False):
        return
    from ..models.h3 import storage_map

    before = storage_map(model)
    for block in model.transformer_blocks:
        attn = block.attn
        if attn.head_dim != 128:
            raise ValueError("H3 QKV packing requires head_dim=128")
        attn.fuse_projections()
        for name, weight in zip(
            ("to_q", "to_k", "to_v"), attn.to_qkv.weight.chunk(3, dim=0)
        ):
            if not torch.equal(getattr(attn, name).weight, weight):
                raise RuntimeError("Packing changed QKV weight values")
            layer = torch.nn.Linear(
                weight.shape[1],
                weight.shape[0],
                bias=False,
                device="meta",
                dtype=weight.dtype,
            )
            layer.weight = torch.nn.Parameter(weight, requires_grad=False)
            setattr(attn, name, layer.eval())
        attn.fused_projections = False
    del block, attn, weight, layer
    gc.collect()
    torch.cuda.synchronize()
    if before != storage_map(model):
        # 地址必然改变；必须核对物理总字节，而不是要求地址相同。
        if sum(before.values()) != sum(storage_map(model).values()):
            raise RuntimeError("QKV preparation duplicated persistent weight storage")
    model._resident_qkv_shared = True


def load_prepared_model(checkpoint, workers=2):
    from .adaln import build, find_cache, load_transformer_cached

    try:
        cache = find_cache(checkpoint, 29)
    except RuntimeError:
        cache = build(checkpoint, profiles=(5, 29), workers=workers)
    model = load_transformer_cached(checkpoint, steps=29, workers=workers, cache=cache)
    pack_qkv_shared(model)
    return model
