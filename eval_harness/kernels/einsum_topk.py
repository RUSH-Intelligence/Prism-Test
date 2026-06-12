"""Fused einsum + top-k Triton kernels for ReAttention.

Ported from the ReAttention reference implementation
(re_attention/einsum_topk_fuse20240907.py).

Supports top-1 and top-4 selection with the following constraints:
- head_dim == 128
- seqlen_q % 128 == 0
- seqlen_k % 128 == 0
- fp16 or bf16 input
- CUDA tensors only
"""

import torch
import triton
import triton.language as tl

from .bitonic_merge_wrapper import bitonic_merge_wrapper


@triton.heuristics(
    {
        "BLOCK_M": lambda args: 128,
        "BLOCK_N": lambda args: 128,
        "num_warps": lambda args: 8 if args["actual_seqlen_k"] < 250000 else 16,
        "num_stages": lambda args: 3,
    }
)
@triton.jit
def _einsum_topk_1_kernel(
        Q: tl.const,
        K: tl.const,
        Out,
        stride_qb: tl.constexpr,
        stride_qh: tl.constexpr,
        stride_qm: tl.constexpr,
        stride_kb: tl.constexpr,
        stride_kh: tl.constexpr,
        stride_kn: tl.constexpr,
        stride_ob: tl.constexpr,
        stride_oh: tl.constexpr,
        stride_om: tl.constexpr,
        h_hk_ratio: tl.constexpr,
        actual_seqlen_k,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_HEADDIM: tl.constexpr,
):
    start_m = tl.program_id(1)
    off_b = tl.program_id(2)
    off_h = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    q_ptrs = ((Q + off_b * stride_qb + off_h * stride_qh)
              + stride_qm * (start_m * BLOCK_M + offs_m[:, None])
                + offs_d[None, :])
    out_ptrs = ((Out + off_b * stride_ob + off_h * stride_oh)
               + stride_om * (start_m * BLOCK_M + offs_m))

    off_h_k = off_h // h_hk_ratio
    offs_n = tl.arange(0, BLOCK_N)
    k_ptrs = ((K + off_b * stride_kb + off_h_k * stride_kh) +
               offs_n[None, :] * stride_kn + offs_d[:, None])

    q = tl.load(q_ptrs)
    k_max = tl.full([BLOCK_M], value=float("-inf"), dtype=tl.float32)
    k_argmax = tl.zeros([BLOCK_M], dtype=tl.int32)

    for start_n in range(0, actual_seqlen_k, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(k_ptrs + start_n * stride_kn, cache_modifier=".cg")
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, k, qk, allow_tf32=True, out_dtype=tl.float32)

        k_max_curr, k_argmax_curr = tl.max(qk, axis=1, return_indices=True)
        mask = k_max_curr >= k_max
        k_max = tl.where(mask, k_max_curr, k_max)
        k_argmax = tl.where(mask, (start_n + k_argmax_curr), k_argmax)

    o = k_argmax.to(Out.dtype.element_ty)
    tl.store(out_ptrs, o, cache_modifier=".cs")

    return


@triton.heuristics(
    {
        "BLOCK_M": lambda args: 128,
        "BLOCK_N": lambda args: 128,
        "BLOCK_HEADDIM": lambda args: 128,
        "TOPK": lambda args: 4,
        "num_warps": lambda args: 16,
        "num_stages": lambda args: 3,
    }
)
@triton.jit
def _einsum_topk_4_kernel(
        Q: tl.const,
        K: tl.const,
        Out,
        stride_qb: tl.constexpr,
        stride_qh: tl.constexpr,
        stride_qm: tl.constexpr,
        stride_kb: tl.constexpr,
        stride_kh: tl.constexpr,
        stride_kn: tl.constexpr,
        stride_ob: tl.constexpr,
        stride_oh: tl.constexpr,
        stride_om: tl.constexpr,
        h_hk_ratio: tl.constexpr,
        actual_seqlen_k,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_HEADDIM: tl.constexpr,
        TOPK: tl.constexpr,
):
    actual_seqlen_k = tl.multiple_of(actual_seqlen_k, BLOCK_N)

    start_m = tl.program_id(1)
    off_b = tl.program_id(2)
    off_h = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    q_ptrs = ((Q + off_b * stride_qb + off_h * stride_qh)
              + stride_qm * (start_m * BLOCK_M + offs_m[:, None])
                + offs_d[None, :])
    out_ptrs = ((Out + off_b * stride_ob + off_h * stride_oh)
               + stride_om * (start_m * BLOCK_M + offs_m[:, None])
                 + tl.arange(0, TOPK)[None, :])

    off_h_k = off_h // h_hk_ratio
    offs_n = tl.arange(0, BLOCK_N)
    k_ptrs = ((K + off_b * stride_kb + off_h_k * stride_kh)
              + offs_n[None, :] * stride_kn + offs_d[:, None])

    q = tl.load(q_ptrs)
    # -3.40282e+38 is torch.finfo(torch.float32).min; using exact value
    # instead of float("-inf") to avoid wrong bitwise calculation in xor()
    k_max = tl.full([BLOCK_M, TOPK], value=-3.40282e+38, dtype=tl.float32)
    k_argmax = tl.zeros([BLOCK_M, TOPK], dtype=tl.int32)
    postion_mask: tl.constexpr = (tl.arange(0, TOPK) >= (TOPK - 1))[None, :]

    for start_n in range(0, actual_seqlen_k, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(k_ptrs + start_n * stride_kn, cache_modifier=".cg")
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, k, qk, allow_tf32=True)

        k_max_curr, k_argmax_curr = tl.max(qk, axis=1, return_indices=True)

        mask = postion_mask & (k_max_curr[:, None] > k_max)
        k_max_unsort = tl.where(mask, k_max_curr[:, None], k_max)
        k_argmax_unsort = tl.where(mask, (k_argmax_curr + start_n)[:, None], k_argmax)
        k_max, k_argmax = bitonic_merge_wrapper(
            k_max_unsort, k_argmax_unsort, TOPK
        )

    o = k_argmax.to(Out.dtype.element_ty)
    tl.store(out_ptrs, o, cache_modifier=".cs")

    return


def einsum_topk_func(q: torch.Tensor, k: torch.Tensor, topk: int) -> torch.Tensor:
    """Fused einsum + top-k using Triton kernels.

    Parameters
    ----------
    q : Tensor [batch, num_heads, seqlen_q, head_dim]
    k : Tensor [batch, num_kv_heads, seqlen_k, head_dim]
    topk : int (1 or 4)

    Returns
    -------
    indices : Tensor [batch, num_heads, seqlen_q, topk]  (int32)

    Constraints: head_dim==128, seqlen_q % 128 == 0, seqlen_k % 128 == 0,
    fp16/bf16, CUDA.
    """
    batch, num_heads, seqlen_q, d = q.shape
    batch_k, num_heads_k, seqlen_k, dk = k.shape
    assert q.stride(-1) == 1 and k.stride(-1) == 1
    assert num_heads % num_heads_k == 0, "num_heads must be divisible by num_heads_k"
    assert d == dk and batch_k == batch, (
        f"batch & head dimensions must match, but find q.shape={q.shape} and k.shape={k.shape}"
    )
    assert q.dtype == k.dtype and q.dtype in [torch.float16, torch.bfloat16], (
        "All tensors must have the same type. Only support fp16 and bf16"
    )
    assert q.is_cuda and k.is_cuda, "All tensors must be sent to gpu"
    assert d == 128 and seqlen_q % 128 == 0 and seqlen_k % 128 == 0, (
        f"Only support d == 128 && seqlen_q % 128 == 0 && seqlen_k % 128 == 0, "
        f"but find d={d}, seqlen_q={seqlen_q}, seqlen_k={seqlen_k}"
    )
    assert topk == 1 or topk == 4, "Only support k == 1 or 4"

    curr_device = torch.cuda.current_device()
    if torch.cuda.current_device() != q.get_device():
        torch.cuda.set_device(q.get_device())

    o = torch.empty((batch, num_heads, seqlen_q, topk), dtype=torch.int32, device=q.device)
    grid = lambda META: (num_heads, triton.cdiv(seqlen_q, META["BLOCK_M"]), batch)

    if topk == 1:
        _einsum_topk_1_kernel[grid](
            q, k, o,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            num_heads // num_heads_k,
            seqlen_k,
            BLOCK_HEADDIM=d,
        )
    else:
        _einsum_topk_4_kernel[grid](
            q, k, o,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            num_heads // num_heads_k,
            seqlen_k,
        )

    torch.cuda.set_device(curr_device)
    return o
