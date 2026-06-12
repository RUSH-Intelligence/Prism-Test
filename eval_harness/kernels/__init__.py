"""Kernels for efficient long-context attention.

Ported from reference implementations:
- ReAttention (OpenMOSS): fused einsum + top-k Triton kernel.
- ChunkLlama (DCA): flash-attention-with-LSE + LSE merge.
"""

# Triton is optional (CPU-only installs / CI): the fused top-k kernel only
# runs on CUDA anyway.  Importing any kernels submodule executes this
# __init__, so the triton-dependent import must not break triton-less envs.
# Consumers already handle einsum_topk_func being None (dense fallback).
try:
    from .einsum_topk import einsum_topk_func
except Exception:  # pragma: no cover - triton not installed
    einsum_topk_func = None  # type: ignore[assignment]

from .dca_flash import (
    attention_with_lse,
    flash_attn_with_lse,
    get_mscale,
    merge_attn_outputs,
    new_flash_attn_with_kvcache,
)

__all__ = [
    "einsum_topk_func",
    "attention_with_lse",
    "flash_attn_with_lse",
    "get_mscale",
    "merge_attn_outputs",
    "new_flash_attn_with_kvcache",
]
