"""RareKV LSH kernels: bucket-id packing and collision counting.

Two independent pipelines live here, both feeding the same ``[BH, L, T]``
bucket-id tensor into the same collision histogram.

Bucket ids -- ``lsh_buckets`` (three modes)
-------------------------------------------
``keys [B,H,T,D] -> bucket [BH, L, T]`` via signed random projections.

``"torch"``
    The REFERENCE: ``keys.reshape(-1, D) @ planes`` then
    :func:`buckets_from_proj_torch` (``sign`` -> int32 -> ``*powers`` -> ``sum``
    -> ``permute``). Always available, no build dependency.
``"pack"`` (TIER 1, the default)
    The identical cuBLAS GEMM, then ONE CUDA kernel that reads ``proj`` once and
    writes uint8/int16 ids straight into the L-major layout. The bool ``sign``,
    the int32 ``packed``, its in-place multiply, the int32 ``bucket`` and the
    transposing clone all stop existing: ~11.8 -> ~2.8 GB of traffic per layer at
    ``T=128K, L=70, P=6``, and the transient falls from ``5`` to ``2 + b/P``
    bytes per ``(row, plane)`` element.

    **Unconditionally bit-identical to** ``"torch"`` -- ``torch.equal``, not
    ``allclose``. The one op that cannot be pinned (cuBLAS picks its own kernel
    by shape, arch and version) is deliberately left alone, and everything after
    the sign test is integer.
``"fused"`` (TIER 2, opt-in)
    The GEMM, the sign test and the bit-pack in one tensor-core kernel, so
    ``proj`` never materialises. NOT bit-identical to ``"pack"``: it is
    bit-identical to a declared reference (see ``csrc/rarekv_fused.cu``), and the
    divergence from the shipped path is MEASURED and published by
    ``scripts/bench_rarekv_kernel.py``, not asserted away. Falls back to
    ``"pack"`` -- never straight to ``"torch"`` -- whenever its preconditions
    (bf16/fp16, ``head_dim in {64,128,256}``, ``P <= 16``, sm_80+) do not hold,
    so a fallback never changes float behaviour more than it has to.

Collision sums -- ``collision_sums`` / ``collision_sums_lmajor``
---------------------------------------------------------------
H200 profiling of the torch scorer showed the collision histogram's global
atomics, not the projection GEMM, dominated the original implementation. The
controlled comparison: ``p3l50`` and ``p8l50`` have identical ``L=50`` -- hence
identical scatter and gather work -- yet ``p8l50`` does 2.67x MORE GEMM work and
ran 1.58x CHEAPER at T=127K. The only difference is the bucket count (R=8 vs
R=256), i.e. 32x more atomic contention.

The fix is a block-privatised histogram: each program builds the histogram of its
own token tile in registers/SRAM and issues ONE atomic add of the whole R-vector.
Global atomics per ``(b,h,l)`` drop from ``T`` to ``(T/BLOCK)*R``. That reduction
factor is ``BLOCK/R``, which is why the Triton variant only pays off for small R
-- hence :func:`should_use_triton`. Measured on an H200 (speedup over torch,
BLOCK=2048): R=4 -> 7.0x, R=8 -> 7.8x, R=64 -> 1.7x, R=256 -> 1.3x, R=1024 -> 0.8x
(SLOWER). The CUDA kernel uses a shared-memory privatised histogram instead and
has no bad case, so it is preferred everywhere it builds.

Bit-identity of the collision path
----------------------------------
Both fused steps are integer: an integer histogram is order-independent, so block
privatisation plus an atomic merge gives *exactly* the torch counts, and the
gather-reduce over L is an integer sum. Verified with ``torch.equal`` over all
five (P,L) configs x {8K, 32K, 128K} x {uniform, skewed} bucket distributions.

The float tail (density, ``(eps+d)**-alpha``, ``* ||v||``) is deliberately NOT
fused: it is O(B*H*T) against this kernel's O(B*H*T*L), so it would buy nothing
while risking a last-ulp difference between a kernel's ``pow`` and torch's.

Kill switches
-------------
``PRISM_RAREKV_CUDA=0`` disables both extensions entirely (``lsh_buckets`` then
resolves to ``"torch"``). ``PRISM_RAREKV_LSH={torch,pack,fused}`` overrides the
requested mode -- in BOTH directions: :func:`resolve_mode` is public precisely so
a caller that must prepare an operand for a mode (``score()`` builds the
transposed planes only for ``fused``) resolves the env var first. Both are read on
every call except the extension builds, which are memoised. The Tier-2 extension
is built lazily, on the first request for ``fused`` or the serial oracle, so an
ordinary run never pays for it.
"""

from __future__ import annotations

import functools
import logging
import os
from typing import Optional, Tuple

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
_KERNEL_EXT = None
_KERNEL_TRIED = False

# NO --use_fast_math. nvcc 12.9 on this cluster: "'--use_fast_math' implies
# '--ftz=true ...'". -ftz=true would flush a subnormal projection to zero and flip
# its sign relative to torch's `proj > 0`. Verified probe: products 7.175e-43 /
# -3.587e-43, exact sum +3.587e-43 -> True; under FTZ -> False. The incidence for
# Gaussian keys with sigma_proj = sqrt(128) is ~8e-40 per element, i.e. ~7e-31
# events per layer -- this is hygiene, not a bug fix, but it is not negotiable
# once float math enters the .cu.
_CUDA_CFLAGS = ["-O3", "-lineinfo", "-ftz=false", "-prec-div=true", "-prec-sqrt=true"]

# TWO extensions, not one. `rarekv_fused.cu` instantiates 15 (P) x 3 (head dim)
# x 2 (dtype) = 90 kernels, each with a fully unrolled P*KSTEPS mma loop, and
# dominates nvcc time. Tier 2 is opt-in, so it gets its own lazily built
# extension: an ordinary eval run -- which only wants `collide` and
# `pack_buckets` -- pays a short build, and only a run that actually asks for
# `lsh_mode="fused"` pays the long one.
_EXT_SOURCES = ("rarekv_module.cpp", "rarekv_collide.cu", "rarekv_pack.cu")
_FUSED_EXT_SOURCES = ("rarekv_fused_module.cpp", "rarekv_fused.cu")

_FUSED_EXT = None
_FUSED_TRIED = False


def _load_ext(name, sources):
    if os.environ.get("PRISM_RAREKV_CUDA", "1") == "0" or not torch.cuda.is_available():
        return None
    try:
        from torch.utils.cpp_extension import load
        here = os.path.join(os.path.dirname(__file__), "csrc")
        return load(name=name, sources=[os.path.join(here, s) for s in sources],
                    extra_cuda_cflags=list(_CUDA_CFLAGS), verbose=False)
    except Exception as exc:                                     # noqa: BLE001
        logger.info("rarekv CUDA extension %r unavailable (%s); falling back", name, exc)
        return None


def _kernel_ext():
    """JIT-load the default extension (collide + Tier-1 pack) once.

    Returns None if it cannot be built. NOTE: the first build costs ~150 s of
    nvcc. torch caches it under TORCH_EXTENSIONS_DIR, which MUST be a persistent
    path -- a per-job scratch dir would rebuild on every SLURM job and swamp the
    speedup on short cells.
    """
    global _KERNEL_EXT, _KERNEL_TRIED
    if _KERNEL_TRIED:
        return _KERNEL_EXT
    _KERNEL_TRIED = True
    _KERNEL_EXT = _load_ext("rarekv_kernels", _EXT_SOURCES)
    return _KERNEL_EXT


def _fused_ext():
    """JIT-load the Tier-2 extension once, ON DEMAND.

    Never called unless something actually asks for `fused` or the serial oracle,
    because its build is several times longer than the default extension's.
    """
    global _FUSED_EXT, _FUSED_TRIED
    if _FUSED_TRIED:
        return _FUSED_EXT
    _FUSED_TRIED = True
    _FUSED_EXT = _load_ext("rarekv_fused_kernels", _FUSED_EXT_SOURCES)
    return _FUSED_EXT


# Back-compat alias: tests and older call sites use the pre-rename name.
_cuda_ext = _kernel_ext


DEFAULT_BLOCK = 2048
# Minimum atomic-reduction factor (BLOCK/R) for the privatised histogram to pay
# for itself. Measured crossover: R=256 wins (factor 8), R=1024 loses (factor 2).
MIN_REDUCTION = 8

# The fused kernel is instantiated for these head dims only (KSTEPS in {4,8,16});
# every other head_dim falls back to the bit-identical pack path.
FUSED_HEAD_DIMS = (64, 128, 256)
# Below this many rows the fused launch is degenerate (fewer than 8 CTAs), so the
# pack path -- which has no minimum size -- wins and is also bit-identical.
FUSED_MIN_ROWS = 8 * 128
# Kernel bucket ids are uint8/int16, so the packing kernels cap at P=15: int16 is
# SIGNED, so P=16 (ids up to 65535) does not fit -- which is exactly the bound
# rarekv_collide.cu has always enforced ("int16 buckets require R <= 32768").
# The torch reference has no such cap (int32 ids) and the validator allows P<=30.
KERNEL_MAX_PLANES = 15


def should_use_triton(n_buckets: int, device, block: int = DEFAULT_BLOCK) -> bool:
    """True when the privatised histogram is expected to beat torch's scatter_add_."""
    return (
        HAVE_TRITON
        and getattr(device, "type", str(device)) == "cuda"
        and n_buckets * MIN_REDUCTION <= block
    )


# ------------------------------------------------------------ bucket ids ----

def bucket_dtype(P: int) -> torch.dtype:
    """Smallest lossless dtype for a bucket id in ``[0, 2**P)``.

    uint8 for ``P <= 8`` and int16 for ``9 <= P <= 15`` -- both are what the CUDA
    kernels emit natively, and both quarter/halve the traffic on the ONE tensor
    the collision kernel reads twice and writes once. int16 stops at 15, not 16,
    because it is SIGNED: an id of 65535 does not fit.

    ``16 <= P <= 30`` returns int32: the packing kernels do not go there (see
    :data:`KERNEL_MAX_PLANES`) but the torch reference does, and
    ``RareKVSketch`` has always accepted ``n_planes`` up to 30. Refusing here
    would turn a supported configuration into a hard error.
    """
    if P < 1:
        raise ValueError(f"n_planes P must be >= 1; got {P}")
    if P <= 8:
        return torch.uint8
    if P <= 15:
        return torch.int16
    if P <= 30:
        return torch.int32
    raise ValueError(f"n_planes P must be <= 30; got {P}")


def buckets_from_proj_torch(proj: torch.Tensor, BH: int, T: int, L: int, P: int) -> torch.Tensor:
    """REFERENCE DEFINITION of the bucket ids -> ``[BH, L, T]``.

    Byte for byte the pre-fusion sequence from ``RareKVSketch.score``, kept as the
    thing every kernel is asserted ``torch.equal`` to. The ``del``s preserve the
    5 B/elem transient ordering the class docstring documents: holding the
    projection alive across a widening cast and a product costs 20 B/elem instead.

    ``proj`` is CONSUMED (deleted from this frame); callers must drop their own
    reference first if they want the saving.
    """
    device = proj.device
    pw = (2 ** torch.arange(P, dtype=torch.int64)).to(device, torch.int32)
    sign = (proj > 0).view(-1, L, P)
    del proj
    packed = sign.to(torch.int32)
    del sign
    packed.mul_(pw)
    bucket = packed.sum(-1, dtype=torch.int32)             # [BH*T, L]
    del packed
    out = bucket.view(BH, T, L).permute(0, 2, 1).contiguous()
    del bucket
    return out.to(bucket_dtype(P))


def pack_buckets(proj: torch.Tensor, BH: int, T: int, L: int, P: int,
                 block_m: int = 0) -> torch.Tensor:
    """Tier 1: ``proj [BH*T, L*P] -> bucket [BH, L, T]`` in one CUDA kernel."""
    ext = _kernel_ext()
    if ext is None:
        raise RuntimeError("rarekv CUDA extension unavailable")
    return ext.pack_buckets(proj.contiguous(), BH, T, L, P, block_m)


def _empty_buckets(BH: int, T: int, L: int, P: int, device) -> torch.Tensor:
    return torch.empty(BH, L, T, device=device, dtype=bucket_dtype(P))


def planes_to_planes_t(planes: torch.Tensor, L: int, P: int) -> torch.Tensor:
    """``planes [D, L*P] -> planesT [ceil(L/8)*8*P, D]``, zero padded.

    The fused kernel's B operand is ``B[k][n] = planesT[col][k]`` with ``k``
    contiguous, and it sweeps 8 tables per column tile, so the row count is
    rounded up to a whole number of tiles. The padding rows are zero, so the
    dot products they produce are ``+0.0`` -- never read back (``ng`` clips the
    extractor to the real tables), but defined.
    """
    D = planes.shape[0]
    ntiles = (L + 7) // 8
    rows = ntiles * 8 * P
    out = planes.new_zeros(rows, D)
    out[: L * P] = planes.t()
    return out.contiguous()


def resolve_mode(mode: str) -> str:
    """The mode that will ACTUALLY be requested, after the env overrides.

    Public because a caller has to know it BEFORE the call: ``score()`` only
    builds the transposed plane operand when the resolved mode is ``fused``, and
    resolving that inside :func:`lsh_buckets` alone made ``PRISM_RAREKV_LSH=fused``
    a no-op on any config whose ``lsh_mode`` was not already ``"fused"`` -- the
    env var could downgrade but never upgrade, and a divergence study driven by it
    would have published a Tier-2 flip rate of exactly zero for a path that never
    ran.
    """
    requested = os.environ.get("PRISM_RAREKV_LSH", mode)
    if os.environ.get("PRISM_RAREKV_CUDA", "1") == "0":
        requested = "torch"
    if requested not in ("torch", "pack", "fused"):
        raise ValueError(f"lsh mode must be one of torch|pack|fused; got {requested!r}")
    return requested


# Back-compat alias: the pre-rename private name.
_resolve_mode = resolve_mode


# Dynamic shared memory is capped at 48 KB per block unless the kernel opts in
# (`cudaFuncSetAttribute`), and at `sharedMemPerBlockOptin` even then. Both
# kernels that use it are checked here as well as in C++, because the python
# check can ROUTE AROUND the limit (to the bit-identical torch sequence) while
# the C++ one can only refuse.
_DEFAULT_SMEM_CAP = 48 * 1024


def pack_smem_bytes(L: int, P: int, block_m: int = 0) -> int:
    """Dynamic shared memory the Tier-1 pack kernel requests: ``BM * rk_nbs(L*P)``."""
    nbits = L * P
    nbs = ((((nbits + 7) // 8 + 3) - 4 + 31) // 32) * 32 + 4      # mirrors rk_nbs
    return (block_m or 128) * nbs


@functools.lru_cache(maxsize=8)
def _smem_optin_for_index(index: Optional[int]) -> int:
    try:
        props = torch.cuda.get_device_properties(index)
    except Exception:                                            # noqa: BLE001
        return _DEFAULT_SMEM_CAP
    v = getattr(props, "shared_memory_per_block_optin", None) or getattr(
        props, "shared_memory_per_block", None)
    return int(v) if v else _DEFAULT_SMEM_CAP


def device_smem_optin(device=None) -> int:
    """The device's max opt-in dynamic shared memory per block, in bytes.

    227 KB on an H200; falls back to the 48 KB no-opt-in default if the property
    cannot be read (CPU-only build, older torch), which is the conservative
    direction -- it can only send a caller to the torch path, never past a limit.
    Memoised per device index: this is consulted once per layer per prefill.
    """
    index = getattr(device, "index", device) if device is not None else None
    return _smem_optin_for_index(index if isinstance(index, int) else None)


def lsh_buckets(keys: torch.Tensor, planes: torch.Tensor,
                planes_t: Optional[torch.Tensor], L: int, P: int, *,
                mode: str = "pack", block_m: int = 0,
                gemm_chunk_rows: int = 0) -> Tuple[torch.Tensor, str]:
    """``keys [B,H,T,D] -> (bucket [BH, L, T], resolved path)``.

    The resolved path is RETURNED, not merely logged: a run that silently
    degraded from ``fused`` to ``pack`` is otherwise indistinguishable from one
    that did not, and its divergence table would be meaningless.
    """
    requested = resolve_mode(mode)
    ext = _kernel_ext()
    B, H, T, D = keys.shape
    BH, N = B * H, B * H * T
    # The pack kernel's sign bitmap is BM * rk_nbs(L*P) bytes of dynamic shared
    # memory. Past the device's opt-in limit the launch would fail outright, so
    # route to the torch sequence -- which is bit-identical, hence a pure speed
    # loss -- instead. Binds only at L*P >= 2825 (block_m=128) or >= 1289
    # (block_m=256) on a 48 KB cap; the whole target grid is far below it.
    smem_ok = (not keys.is_cuda
               or pack_smem_bytes(L, P, block_m) <= device_smem_optin(keys.device))

    # `_fused_ext()` triggers a long JIT build, so it is only consulted when the
    # shape/dtype preconditions already hold AND `fused` was actually requested.
    fused_ok = (
        requested == "fused"
        and ext is not None
        and keys.is_cuda
        and keys.dtype in (torch.bfloat16, torch.float16)
        and planes_t is not None
        and D in FUSED_HEAD_DIMS
        and 1 <= P <= KERNEL_MAX_PLANES
        and torch.cuda.get_device_capability(keys.device) >= (8, 0)
        and N >= FUSED_MIN_ROWS
    )
    if fused_ok:
        fext = _fused_ext()
        if fext is not None:
            return fext.fused_buckets(keys.contiguous(), planes_t, L, P, block_m), "fused"

    pack_ok = (requested in ("fused", "pack") and ext is not None and keys.is_cuda
               and 1 <= P <= KERNEL_MAX_PLANES and smem_ok)
    path = "pack" if pack_ok else "torch"

    k2 = keys.reshape(-1, D)
    if not pack_ok:
        # The reference: one GEMM, then the four ATen ops. Written as ONE
        # expression on purpose -- the projection must have no name in this frame,
        # so `buckets_from_proj_torch`'s own `del proj` is what frees it and the
        # transient really is 5 B/elem rather than 6.
        return buckets_from_proj_torch(k2 @ planes, BH, T, L, P), path

    if gemm_chunk_rows <= 0 or N <= gemm_chunk_rows:
        proj = k2 @ planes
        bucket = _empty_buckets(BH, T, L, P, keys.device)
        ext.pack_buckets_into(proj, bucket, 0, P, block_m)
        del proj
        return bucket, path

    # Tier 1c: chunk the GEMM along M so the projection never exists in full.
    # Chunking along M cannot change any output element's reduction order over D;
    # the only exposure is cuBLAS re-selecting an algorithm at the smaller M, which
    # is an empirical question -- hence this ships OFF and is gated by the bench's
    # `gemm_chunk_exact` column.
    bucket = _empty_buckets(BH, T, L, P, keys.device)
    scratch = torch.empty(gemm_chunk_rows, L * P, device=keys.device, dtype=keys.dtype)
    for i in range(0, N, gemm_chunk_rows):
        rows = min(gemm_chunk_rows, N - i)
        out = scratch[:rows]
        torch.mm(k2[i:i + rows], planes, out=out)
        ext.pack_buckets_into(out, bucket, i, P, block_m)
    del scratch
    return bucket, path


def serial_buckets(keys: torch.Tensor, planes_t: torch.Tensor, L: int, P: int) -> torch.Tensor:
    """Debug oracle: scalar, strictly-ascending-k, no tensor cores, no shared memory.

    Bisects a failing Tier-2 parity check in ONE job: if this matches the torch
    reference but ``fused`` does not, the bug is in the mma fragment mapping or
    the tile bitmap, not in the dispatch or the layout. O(N*L*P*D) on CUDA cores
    -- never a production path.
    """
    ext = _fused_ext()
    if ext is None:
        raise RuntimeError("rarekv Tier-2 CUDA extension unavailable")
    return ext.serial_buckets(keys.contiguous(), planes_t.contiguous(), L, P)


# ------------------------------------------------------- collision sums ----

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
        b = tl.load(BUCKET + bh * stride_bh + offs * L + l, mask=mask, other=0).to(tl.int32)
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
            b = tl.load(BUCKET + bh * stride_bh + offs * L + l, mask=mask, other=0).to(tl.int32)
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
    # Upcast BEFORE folding in the offset: bucket may be uint8/int16, and the
    # offset reaches (L-1)*R = 60416 at L=60, R=1024, which overflows both.
    b = bucket.to(torch.int32) + torch.arange(L, device=bucket.device, dtype=torch.int32) * n_buckets
    idx = b.view(BH, T * L).to(torch.int64)
    counts = torch.zeros(BH, L * n_buckets, device=bucket.device, dtype=torch.int32)
    ones = torch.ones(1, device=bucket.device, dtype=torch.int32).expand_as(idx)
    counts.scatter_add_(1, idx, ones)
    return counts.gather(1, idx).view(BH, T, L).sum(dim=2, dtype=torch.int32)


def collision_sums_cuda(bucket: torch.Tensor, n_buckets: int) -> torch.Tensor:
    """``bucket`` [BH, T, L] -> csum [BH, T]. Transposes to the L-major layout.

    This is the T-major entry point, kept because it is the shape the pre-pack
    scorer produced and the shape the existing tests pin. New code that already
    has the L-major layout (everything going through :func:`lsh_buckets`) should
    call :func:`collision_sums_lmajor` and skip the transposing clone -- which
    cost ~0.3-0.6 ms/layer at T=128K, and is exactly what folding the transpose
    into the bit-pack removed. Note we do NOT reformulate the GEMM as
    ``planes.T @ keys.T`` to get L-major directly: that changes the cuBLAS call,
    which can flip a near-zero projection's sign and break bit-identity.
    """
    ext = _kernel_ext()
    if ext is None:
        raise RuntimeError("rarekv CUDA extension unavailable")
    return ext.collide(bucket.permute(0, 2, 1).contiguous(), n_buckets)


def hist_smem_ok(n_buckets: int, device=None) -> bool:
    """Can the privatised histogram's ``R`` ints fit in one block's shared memory?

    False at ``R > 58112`` on an H200 (``P >= 16``, only reachable on the int32
    torch-reference path) and at ``R > 12288`` on a 48 KB device (``P >= 14``).
    The CUDA kernel opts in above 48 KB; past the device limit there is nothing
    to opt into, so the caller must take the (bit-identical) torch path rather
    than eat an `invalid argument` at launch.
    """
    return n_buckets * 4 <= device_smem_optin(device)


def collision_sums_lmajor(bucket_lmajor: torch.Tensor, n_buckets: int, *,
                          prefer_kernels: bool = True) -> torch.Tensor:
    """``bucket`` [BH, L, T] -> csum [BH, T] int32. No transpose on the kernel path."""
    ext = _kernel_ext()
    if (bucket_lmajor.is_cuda and prefer_kernels and ext is not None
            and hist_smem_ok(n_buckets, bucket_lmajor.device)):
        return ext.collide(bucket_lmajor.contiguous(), n_buckets)
    tmajor = bucket_lmajor.permute(0, 2, 1).contiguous()
    if tmajor.dtype != torch.int32:
        tmajor = tmajor.to(torch.int32)
    return collision_sums(tmajor, n_buckets, prefer_triton=prefer_kernels)


def collision_sums(bucket: torch.Tensor, n_buckets: int, *, prefer_triton: bool = True,
                   block: int = DEFAULT_BLOCK) -> torch.Tensor:
    """Fastest available path for a [BH, T, L] bucket tensor.

    All three are BIT-IDENTICAL, so this is pure speed. Order: CUDA (best
    everywhere) -> Triton (only where BLOCK/R is large enough to pay for itself)
    -> torch (always correct, no build dependency).
    """
    if (bucket.is_cuda and prefer_triton and _kernel_ext() is not None
            and hist_smem_ok(n_buckets, bucket.device)):
        return collision_sums_cuda(bucket, n_buckets)
    if prefer_triton and should_use_triton(n_buckets, bucket.device, block):
        return collision_sums_triton(bucket, n_buckets, block=block)
    return collision_sums_torch(bucket, n_buckets)
