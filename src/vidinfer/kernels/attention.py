"""Process-local replacement for Diffusers BLHD attention dispatch only."""

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel


class NativeStrideAttention:
    def __init__(self, torch_backend):
        self.torch_backend = torch_backend

    def __call__(
        self,
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False,
        backend=None,
        parallel_config=None,
    ):
        # This experiment supports the original single-device H3 t2va path.
        if parallel_config is not None:
            raise ValueError("Context-parallel attention is outside this adapter")
        if attn_mask is not None or dropout_p != 0.0 or is_causal or enable_gqa:
            raise ValueError("Expected full, unmasked, dropout-free H3 attention")
        if not all(
            x.ndim == 4 and x.dtype == torch.bfloat16 and x.is_cuda
            for x in (query, key, value)
        ):
            raise ValueError("Expected CUDA BF16 BLHD tensors")
        # Views only: preserve original data and pass supported strides directly.
        q, k, v = (x.permute(0, 2, 1, 3) for x in (query, key, value))
        # Force one implementation; never silently fall back to another backend.
        with sdpa_kernel(self.torch_backend):
            output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                scale=scale,
                enable_gqa=False,
            )
        # Return the official BLHD contract; include any necessary copy here.
        return output.permute(0, 2, 1, 3).contiguous()
