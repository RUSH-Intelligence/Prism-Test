"""Triton collision-counting kernel for RareKV, bit-identical to the torch path.

Why this exists
---------------
H200 profiling of the torch scorer showed the bottleneck is NOT the projection
GEMM but the collision histogram's global atomics. The controlled comparison:
``p3l50`` and ``p8l50`` have identical ``L=50`` -- hence identical scatter and
gather work -- yet ``p8l50`` does 2.67x MORE GEMM work and runs 1.58x CHEAPER at
T=127K. The only difference is the bucket count (R=8 vs R=256), i.e. 32x more
atomic contention. At R=8 each of the 8 counters per (b,h,l) absorbs ~16k
increments, and they serialise.

The fix is a block-privatised histogram: each program builds the histogram of
its own token tile with ``tl.histogram`` (registers/SRAM) and issues ONE
``atomic_add`` of the whole R-vector. Global atomics per (b,h,l) drop from ``T``
to ``(T/BLOCK)*R``.

That reduction factor is ``BLOCK/R``, which is why this only pays off for small
R -- hence :func:`should_use_triton`. Measured on an H200 (speedup over torch,
BLOCK=2048): R=4 -> 7.0x, R=8 -> 7.8x, R=64 -> 1.7x, R=256 -> 1.3x, and
R=1024 -> 0.8x (SLOWER). At R=1024 the torch scatter already has little
contention and the per-tile R-vector costs more than it saves, so we fall back.

Bit-identity
------------
Both fused steps are integer: an integer histogram is order-independent, so
block privatisation plus an atomic merge gives *exactly* the torch counts, and
the gather-reduce over L is an integer sum. Verified with ``torch.equal`` over
all five (P,L) configs x {8K, 32K, 128K} x {uniform, skewed} bucket
distributions.

The float tail (density, ``(eps+d)**-alpha``, ``* ||v||``) is deliberately NOT
fused: it is O(B*H*T) against this kernel's O(B*H*T*L), so it would buy nothing
while risking a last-ulp difference between Triton's ``pow`` and torch's.
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except Exception:                                                # noqa: BLE001
    triton = None
    tl = None
    HAVE_TRITON = False


# ---------------------------------------------------------------- CUDA ----
# The CUDA extension is the preferred path: it is faster than Triton on EVERY
# measured config, and it is the only variant with no bad case. Two reasons,
# both measured on an H200:
#   * thread-per-key gather on the [BH, L, T] layout (from SOCKET's
#     soft_hash_score.cu) -- coalesced bucket loads, worth 2-8.6x over the
#     [BH, T, L] layout with everything else held fixed;
#   * a shared-memory privatised histogram -- atomicAdd into R ints of SRAM
#     (<= 4 KB even at R=1024), so it absorbs bucket skew at ALL R. The Triton
#     tl.histogram version regressed BELOW torch at R=1024 and had to fall back.
_CUDA_EXT = None
_CUDA_TRIED = False


def _cuda_ext():
    """JIT-load the extension once. Returns None if it cannot be built.

    NOTE: the first build costs ~100 s of nvcc. torch caches it under
    TORCH_EXTENSIONS_DIR, which MUST be a persistent path -- a per-job scratch
    dir would rebuild on every SLURM job and swamp the speedup on short cells.
    """
    global _CUDA_EXT, _CUDA_TRIED
    if _CUDA_TRIED:
        return _CUDA_EXT
    _CUDA_TRIED = True
    if os.environ.get("PRISM_RAREKV_CUDA", "1") == "0" or not torch.cuda.is_available():
        return None
    try:
        from torch.utils.cpp_extension import load
        src = os.path.join(os.path.dirname(__file__), "csrc", "rarekv_collide.cu")
        _CUDA_EXT = load(name="rarekv_collide", sources=[src],
                         extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)
    except Exception as exc:                                     # noqa: BLE001
        logger.info("rarekv CUDA extension unavailable (%s); falling back", exc)
        _CUDA_EXT = None
    return _CUDA_EXT


DEFAULT_BLOCK = 2048
# Minimum atomic-reduction factor (BLOCK/R) for the privatised histogram to pay
# for itself. Measured crossover: R=256 wins (factor 8), R=1024 loses (factor 2).
MIN_REDUCTION = 8


def should_use_triton(n_buckets: int, device, block: int = DEFAULT_BLOCK) -> bool:
    """True when the privatised histogram is expected to beat torch's scatter_add_."""
    return (
        HAVE_TRITON
        and getattr(device, "type", str(device)) == "cuda"
        and n_buckets * MIN_REDUCTION <= block
    )


if HAVE_TRITON:

    @triton.jit
    def _hist_kernel(BUCKET, COUNTS, T, L, stride_bh, R: tl.constexpr, BLOCK: tl.constexpr):
        """Block-privatised histogram. grid = (cdiv(T, BLOCK), BH*L)."""
        pid_t = tl.program_id(0)
        pid_bhl = tl.program_id(1)
        bh = pid_bhl // L
        l = pid_bhl % L
        offs = pid_t * BLOCK + tl.arange(0, BLOCK)
        mask = offs < T
        b = tl.load(BUCKET + bh * stride_bh + offs * L + l, mask=mask, other=0)
        # masked so padding lanes do not land in bin 0
        h = tl.histogram(b, R, mask=mask)
        tl.atomic_add(COUNTS + pid_bhl * R + tl.arange(0, R), h)

    @triton.jit
    def _gather_kernel(BUCKET, COUNTS, CSUM, T, L, stride_bh, R: tl.constexpr, BLOCK: tl.constexpr):
        """csum[i] = sum_l COUNTS[l, bucket_l(i)] -- integer, hence exact."""
        pid_t = tl.program_id(0)
        bh = tl.program_id(1)
        offs = pid_t * BLOCK + tl.arange(0, BLOCK)
        mask = offs < T
        acc = tl.zeros([BLOCK], dtype=tl.int32)
        for l in range(L):
            b = tl.load(BUCKET + bh * stride_bh + offs * L + l, mask=mask, other=0)
            acc += tl.load(COUNTS + (bh * L + l) * R + b, mask=mask, other=0)
        tl.store(CSUM + bh * T + offs, acc, mask=mask)


def collision_sums_triton(bucket: torch.Tensor, n_buckets: int,
                          block: int = DEFAULT_BLOCK) -> torch.Tensor:
    """``bucket`` [BH, T, L] int32 (values in [0, R)) -> csum [BH, T] int32."""
    if not HAVE_TRITON:
        raise RuntimeError("triton is not available")
    BH, T, L = bucket.shape
    bucket = bucket.contiguous()
    counts = torch.zeros(BH * L, n_buckets, device=bucket.device, dtype=torch.int32)
    stride_bh = T * L
    _hist_kernel[(triton.cdiv(T, block), BH * L)](
        bucket, counts, T, L, stride_bh, R=n_buckets, BLOCK=block)
    csum = torch.empty(BH, T, device=bucket.device, dtype=torch.int32)
    _gather_kernel[(triton.cdiv(T, block), BH)](
        bucket, counts, csum, T, L, stride_bh, R=n_buckets, BLOCK=block)
    return csum


def collision_sums_torch(bucket: torch.Tensor, n_buckets: int) -> torch.Tensor:
    """Reference path: one 2-D scatter_add_ with the table offset folded in."""
    BH, T, L = bucket.shape
    b = bucket + torch.arange(L, device=bucket.device, dtype=bucket.dtype) * n_buckets
    idx = b.view(BH, T * L).to(torch.int64)
    counts = torch.zeros(BH, L * n_buckets, device=bucket.device, dtype=torch.int32)
    ones = torch.ones(1, device=bucket.device, dtype=torch.int32).expand_as(idx)
    counts.scatter_add_(1, idx, ones)
    return counts.gather(1, idx).view(BH, T, L).sum(dim=2, dtype=torch.int32)


def collision_sums_cuda(bucket: torch.Tensor, n_buckets: int) -> torch.Tensor:
    """``bucket`` [BH, T, L] -> csum [BH, T]. Transposes to the L-major layout."""
    ext = _cuda_ext()
    if ext is None:
        raise RuntimeError("rarekv CUDA extension unavailable")
    # The transpose costs ~0.3-0.6 ms/layer at T=128K, already included in every
    # measurement quoted above; the CUDA path still wins on every config. Folding
    # it into the bit-pack would make it ~free -- a worthwhile follow-up. Note we
    # do NOT reformulate the GEMM as planes.T @ keys.T to get L-major directly:
    # that changes the cuBLAS call, which can flip a near-zero projection's sign
    # and break bit-identity. Not worth trading a proven invariant for 0.3 ms.
    return ext.collide(bucket.permute(0, 2, 1).contiguous(), n_buckets)


def collision_sums(bucket: torch.Tensor, n_buckets: int, *, prefer_triton: bool = True,
                   block: int = DEFAULT_BLOCK) -> torch.Tensor:
    """Fastest available path. All three are BIT-IDENTICAL, so this is pure speed.

    Order: CUDA (best everywhere) -> Triton (only where BLOCK/R is large enough
    to pay for itself) -> torch (always correct, no build dependency).
    """
    if bucket.is_cuda and prefer_triton and _cuda_ext() is not None:
        return collision_sums_cuda(bucket, n_buckets)
    if prefer_triton and should_use_triton(n_buckets, bucket.device, block):
        return collision_sums_triton(bucket, n_buckets, block=block)
    return collision_sums_torch(bucket, n_buckets)
