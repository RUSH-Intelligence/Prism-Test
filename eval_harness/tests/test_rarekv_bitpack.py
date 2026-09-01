"""Contract for the RareKV bucket-packing kernels (Tier 1 `pack`, Tier 2 `fused`).

The expensive risk in these kernels is INDEXING, not arithmetic: a tile bitmap, a
3-byte extraction window, an L-major store, a ragged tail, a column-tile boundary,
an mma fragment map. All of that is pure integer bookkeeping, so it can be
emulated in python and asserted `torch.equal` against the reference WITHOUT a GPU
-- which is the point of this file. Every bug caught here is a SLURM round trip
saved.

The exactness contract these tests pin
--------------------------------------
C1  `lsh_mode="pack"` is UNCONDITIONALLY bit-identical to `lsh_mode="torch"`.
    Four links: (i) `proj` is byte-identical, because it is the same `torch.mm`
    with the same operands, shapes and strides -- the one op that cannot be
    pinned is deliberately left alone; (ii) the sign predicate agrees on every
    float class (`+0.0` F, `-0.0` F, NaN F by IEEE unordered, `+inf` T, `-inf` F,
    positive subnormal T) -- the only way to break it is FTZ, hence the
    `--use_fast_math` ban, asserted here; (iii) the packing is integer, hence
    order-free; (iv) the sub-word ids and the [BH, L, T] layout are lossless.

C2  `lsh_mode="fused"` is bit-identical to a DECLARED reference, not to cuBLAS.
    The layout half of that claim -- column tiles, byte indices, the 4-lane OR
    butterfly, the two-rows-per-word packing -- is emulated and pinned here; the
    float half is measured on hardware by scripts/bench_rarekv_kernel.py.

GPU-gated parity tests live at the bottom and skip on CPU.
"""

from __future__ import annotations

import inspect
import os
import unittest
from types import SimpleNamespace

import torch

from eval_harness.kernels import rarekv_lsh
from eval_harness.kernels.rarekv_lsh import (
    KERNEL_MAX_PLANES, bucket_dtype, buckets_from_proj_torch, lsh_buckets,
    planes_to_planes_t,
)

CUDA = torch.cuda.is_available()

# (P, L): the profiled grid, the L=100/P in 5..9 target grid, and the awkward
# shapes -- LP odd (7,9 -> 63), LP not a multiple of 8 (15,5 -> 75), L=1, P=1.
PL_GRID = [(2, 40), (3, 50), (5, 100), (6, 70), (8, 50), (9, 100),
           (15, 5), (1, 3), (9, 7), (15, 9), (7, 1)]


# --------------------------------------------------------------------------
# A pure-python emulation of the Tier-1 pack kernel's arithmetic.
# --------------------------------------------------------------------------

def rk_nbs(nbits: int) -> int:
    """Mirror of `rk_nbs` in csrc/rarekv_bits.cuh."""
    need = (nbits + 7) // 8 + 3
    return ((need - 4 + 31) // 32) * 32 + 4


def emulate_pack(proj: torch.Tensor, BH: int, T: int, L: int, P: int, *,
                 BM: int = 128, row0: int = 0, out: torch.Tensor = None) -> torch.Tensor:
    """Byte-for-byte emulation of rk_pack_kernel + rk_emit, in python.

    Follows the kernel exactly: the CTA tile loop, `rk_nbs`, the per-row bitmap,
    the 32-projections-per-warp `__ballot_sync` fill (lane l -> bit l -> byte
    c0/8 + l/8, LSB first) with the `col < LP` tail mask, the slack-byte zeroing
    between 4*nchunks and NB+3, the 3-byte extraction window,
    `>> (bit0 & 7) & mask`, and the `row0 + t0 + r -> (bh, t)` decomposition.
    """
    Rc, LP = proj.shape
    assert LP == L * P
    nbs, NB = rk_nbs(LP), (LP + 7) // 8
    nchunks = (LP + 31) // 32
    NBW = nchunks * 4
    assert NBW <= nbs, "the ballot words must fit the row stride"
    assert NBW >= NB, "the ballot words must cover every real byte"
    mask = (1 << P) - 1
    if out is None:
        out = torch.zeros(BH, L, T, dtype=bucket_dtype(P))
    sign = (proj > 0).to(torch.uint8).tolist()

    for t0 in range(0, Rc, BM):
        nrow = min(BM, Rc - t0)
        sbits = [[0] * nbs for _ in range(BM)]
        # phase 1: one warp-wide ballot per 32 projection columns
        for it in range(nrow * nchunks):
            r, c = divmod(it, nchunks)
            c0 = c * 32
            row = sign[t0 + r]
            word = 0
            for lane in range(32):                  # lane l reads column c0+l
                col = c0 + lane
                if col < LP and row[col]:
                    word |= 1 << lane
            for k in range(4):                      # one 4-byte store, LSB first
                sbits[r][(c0 >> 3) + k] = (word >> (8 * k)) & 0xFF
        for r in range(nrow):                       # slack bytes
            for k in range(3):
                by = NBW + k
                if by < NB + 3 and by < nbs:
                    sbits[r][by] = 0
        # phase 2 -- rk_emit
        for g in range(L):
            for r in range(nrow):
                bit0 = g * P
                by = bit0 >> 3
                p = sbits[r]
                v = p[by] | (p[by + 1] << 8) | (p[by + 2] << 16)
                gr = row0 + t0 + r
                bh, t = divmod(gr, T)
                out[bh, g, t] = (v >> (bit0 & 7)) & mask
    return out


# --------------------------------------------------------------------------
# A pure-python emulation of the Tier-2 fused kernel's LAYOUT (exact dot).
# --------------------------------------------------------------------------

# (lane, register) -> (row, col) of the m16n8k16 C/D fragment, per the PTX ISA.
def mma_c_map(lane: int, i: int):
    g, q = lane >> 2, lane & 3
    return (g + 8 * (i // 2), 2 * q + (i % 2))


def emulate_fused(keys: torch.Tensor, planes_t: torch.Tensor, BH: int, T: int,
                  L: int, P: int, *, BM: int = 128) -> torch.Tensor:
    """Emulate the fused kernel's column sweep, epilogue butterfly and store.

    The mma is replaced by an EXACT fp64 dot: this test is about the layout
    (which tables land in which column tile, which byte a n8-tile writes, which
    lane holds which C element, the two-rows-per-32-bit-word packing, the ragged
    last tile), not about float accumulation order. The float half of the Tier-2
    contract is measured on hardware, not asserted here.
    """
    Rc, D = keys.shape
    ntiles = (L + 7) // 8
    assert planes_t.shape[0] == ntiles * 8 * P
    GB = rk_nbs(8 * P)
    mask = (1 << P) - 1
    out = torch.zeros(BH, L, T, dtype=bucket_dtype(P))
    kd = keys.double()
    pd = planes_t.double()

    for t0 in range(0, Rc, BM):
        nrow = min(BM, Rc - t0)
        for ct in range(ntiles):
            col0, g0 = ct * 8 * P, ct * 8
            ng = min(8, L - g0)
            sbits = [[0] * GB for _ in range(BM)]
            for warp in range(BM // 16):
                for n in range(P):                       # n8 tile
                    for lane in range(32):
                        g, q = lane >> 2, lane & 3
                        w = 0
                        for i in range(4):
                            rr, cc = mma_c_map(lane, i)
                            row = warp * 16 + rr
                            gk = t0 + row
                            val = (kd[gk] @ pd[col0 + 8 * n + cc]).item() if gk < Rc else 0.0
                            bit = (2 * q + (i % 2)) + (16 if i >= 2 else 0)
                            if val > 0.0:
                                w |= 1 << bit
                        # the 4-lane OR butterfly is a no-op to emulate: lane q=0
                        # ends up with the OR over q in 0..3, which is what the
                        # loop below reconstructs.
                        if q == 0:
                            for qq in range(4):
                                for i in range(4):
                                    rr, cc = mma_c_map(4 * g + qq, i)
                                    row = warp * 16 + rr
                                    gk = t0 + row
                                    val = (kd[gk] @ pd[col0 + 8 * n + cc]).item() if gk < Rc else 0.0
                                    if val > 0.0:
                                        r = warp * 16 + g + (8 if i >= 2 else 0)
                                        sbits[r][n] |= 1 << (2 * qq + (i % 2))
                        del w
            for r in range(BM):                          # slack bytes
                for k in range(3):
                    if P + k < GB:
                        sbits[r][P + k] = 0
            for g in range(ng):                          # rk_emit
                for r in range(nrow):
                    bit0 = g * P
                    by = bit0 >> 3
                    p = sbits[r]
                    v = p[by] | (p[by + 1] << 8) | (p[by + 2] << 16)
                    bh, t = divmod(t0 + r, T)
                    out[bh, g0 + g, t] = (v >> (bit0 & 7)) & mask
    return out


# ==========================================================================
class TestReferenceDefinition(unittest.TestCase):
    """`buckets_from_proj_torch` must be the shipped inline sequence, transposed."""

    def _inline(self, proj, BH, T, L, P):
        powers = (2 ** torch.arange(P, dtype=torch.int64)).to(proj.device, torch.int32)
        sign = (proj > 0).view(-1, L, P)
        packed = sign.to(torch.int32)
        packed.mul_(powers)
        bucket = packed.sum(-1, dtype=torch.int32)
        return bucket.view(BH, T, L).permute(0, 2, 1).contiguous()

    def test_reference_matches_the_shipped_inline_sequence(self):
        g = torch.Generator().manual_seed(7)
        for P, L in PL_GRID:
            with self.subTest(P=P, L=L):
                BH, T = 3, 29
                proj = torch.randn(BH * T, L * P, generator=g, dtype=torch.float32)
                want = self._inline(proj.clone(), BH, T, L, P)
                got = buckets_from_proj_torch(proj, BH, T, L, P)
                self.assertEqual(got.dtype, bucket_dtype(P))
                self.assertTrue(torch.equal(got.to(torch.int32), want))

    def test_ids_are_in_range(self):
        g = torch.Generator().manual_seed(3)
        for P, L in PL_GRID:
            proj = torch.randn(64, L * P, generator=g)
            b = buckets_from_proj_torch(proj, 4, 16, L, P).to(torch.int32)
            self.assertGreaterEqual(int(b.min()), 0)
            self.assertLess(int(b.max()), 1 << P)


class TestRkNbs(unittest.TestCase):
    def test_congruence_and_headroom(self):
        for n in range(1, 4097):
            nbs = rk_nbs(n)
            self.assertEqual(nbs % 32, 4, f"nbs={nbs} for nbits={n} is not 4 mod 32")
            self.assertGreaterEqual(nbs, (n + 7) // 8 + 3)

    def test_matches_the_documented_table(self):
        # from the SMEM budget table in the design: LP -> nbs
        for lp, want in [(350, 68), (420, 68), (500, 68), (560, 100), (600, 100),
                         (630, 100), (700, 100), (800, 132), (900, 132)]:
            self.assertEqual(rk_nbs(lp), want, f"LP={lp}")

    def test_the_extraction_window_always_fits(self):
        """The 3-byte window must stay inside `nbs` for every shipped (L, P)."""
        for P in range(1, KERNEL_MAX_PLANES + 1):
            for L in (1, 3, 7, 8, 9, 50, 70, 100, 101, 120):
                nbs = rk_nbs(L * P)
                self.assertLess(((L - 1) * P >> 3) + 2, nbs, f"P={P} L={L}")


class TestPackEmulation(unittest.TestCase):
    """The Tier-1 kernel's indexing, emulated and pinned to the reference."""

    def test_matches_the_reference_over_the_grid(self):
        g = torch.Generator().manual_seed(11)
        for P, L in PL_GRID:
            for BH, T in ((1, 1), (2, 7), (3, 43)):
                with self.subTest(P=P, L=L, BH=BH, T=T):
                    proj = torch.randn(BH * T, L * P, generator=g, dtype=torch.float32)
                    want = buckets_from_proj_torch(proj.clone(), BH, T, L, P)
                    got = emulate_pack(proj, BH, T, L, P)
                    self.assertTrue(torch.equal(got, want))

    def test_tile_boundaries_and_block_m(self):
        """Rc around every BM multiple, and BM=256, must not change the answer."""
        g = torch.Generator().manual_seed(13)
        P, L, T = 6, 9, 1
        for Rc in (1, 2, 127, 128, 129, 255, 256, 257, 511, 513):
            for BM in (128, 256):
                with self.subTest(Rc=Rc, BM=BM):
                    proj = torch.randn(Rc, L * P, generator=g, dtype=torch.float32)
                    want = buckets_from_proj_torch(proj.clone(), Rc, T, L, P)
                    got = emulate_pack(proj, Rc, T, L, P, BM=BM)
                    self.assertTrue(torch.equal(got, want))

    def test_chunked_rows_with_row0_offset(self):
        """A chunk may straddle a (b, h) boundary; bh/t come from the flat row."""
        g = torch.Generator().manual_seed(17)
        P, L, BH, T = 5, 11, 3, 13
        Rc = BH * T
        proj = torch.randn(Rc, L * P, generator=g, dtype=torch.float32)
        want = buckets_from_proj_torch(proj.clone(), BH, T, L, P)
        for chunk in (1, 5, 7, 13, 20, Rc):
            with self.subTest(chunk=chunk):
                out = torch.zeros(BH, L, T, dtype=bucket_dtype(P))
                for i in range(0, Rc, chunk):
                    rows = min(chunk, Rc - i)
                    emulate_pack(proj[i:i + rows], BH, T, L, P, row0=i, out=out)
                self.assertTrue(torch.equal(out, want))

    def test_dtypes(self):
        g = torch.Generator().manual_seed(19)
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            for P, L in ((6, 70), (9, 100)):
                with self.subTest(dtype=dtype, P=P, L=L):
                    proj = torch.randn(37, L * P, generator=g).to(dtype)
                    want = buckets_from_proj_torch(proj.clone(), 1, 37, L, P)
                    self.assertTrue(torch.equal(emulate_pack(proj, 1, 37, L, P), want))


class TestFusedLayoutEmulation(unittest.TestCase):
    """The Tier-2 kernel's layout, emulated with an exact dot."""

    def test_mma_c_fragment_map(self):
        """Every (row, col) of a 16x8 tile is held by exactly one (lane, reg)."""
        seen = {}
        for lane in range(32):
            for i in range(4):
                rc = mma_c_map(lane, i)
                self.assertNotIn(rc, seen)
                seen[rc] = (lane, i)
        self.assertEqual(len(seen), 16 * 8)
        # the 8 columns of one row live in 4 lanes, not 32 -- which is why the
        # epilogue is a __shfl_xor butterfly and NOT __ballot_sync.
        lanes_of_row0 = {seen[(0, c)][0] for c in range(8)}
        self.assertEqual(len(lanes_of_row0), 4)

    def test_column_tile_byte_index(self):
        """n8 tile `n` writes byte `n`, and table g's bits are [g*P, (g+1)*P)."""
        for P in range(1, KERNEL_MAX_PLANES + 1):
            BN = 8 * P
            self.assertEqual(BN % 8, 0, "a column tile must start on a byte boundary")
            for g in range(8):
                lo, hi = g * P, (g + 1) * P - 1
                # the table's bits never straddle the tile
                self.assertLess(hi, BN)
                # and byte(bit) for the n8 tile that produced them is consistent
                for bit in range(lo, hi + 1):
                    n, off = divmod(bit, 8)
                    self.assertEqual(bit, 8 * n + off)

    def test_emulated_fused_equals_the_reference(self):
        g = torch.Generator().manual_seed(23)
        for P, L in ((1, 3), (5, 8), (6, 9), (3, 7), (9, 17), (2, 15)):
            with self.subTest(P=P, L=L):
                BH, T, D = 2, 5, 32
                keys = torch.randn(BH * T, D, generator=g, dtype=torch.float32)
                planes = torch.randn(D, L * P, generator=g, dtype=torch.float32)
                pt = planes_to_planes_t(planes, L, P)
                want = buckets_from_proj_torch((keys.double() @ planes.double()).float(),
                                               BH, T, L, P)
                got = emulate_fused(keys, pt, BH, T, L, P, BM=16)
                self.assertTrue(torch.equal(got, want))

    def test_planes_t_is_the_transpose_zero_padded(self):
        for P, L in ((6, 70), (9, 100), (5, 8), (3, 1)):
            with self.subTest(P=P, L=L):
                planes = torch.randn(16, L * P)
                pt = planes_to_planes_t(planes, L, P)
                self.assertEqual(pt.shape, (((L + 7) // 8) * 8 * P, 16))
                self.assertTrue(torch.equal(pt[:L * P], planes.t()))
                self.assertTrue(torch.equal(pt[L * P:], torch.zeros_like(pt[L * P:])))
                self.assertTrue(pt.is_contiguous())


class TestAdversarialFloats(unittest.TestCase):
    """The near-zero battery. These are the inputs where a fused GEMM can differ."""

    def test_sign_predicate_on_every_float_class(self):
        """`> 0` semantics: -0.0 and NaN are FALSE; a positive subnormal is TRUE."""
        vals = torch.tensor([0.0, -0.0, float("nan"), float("inf"), float("-inf"),
                             torch.finfo(torch.float32).smallest_normal / 2.0,
                             -torch.finfo(torch.float32).smallest_normal / 2.0,
                             1e-45, -1e-45, 1.0, -1.0], dtype=torch.float32)
        want = [False, False, False, True, False, True, False, True, False, True, False]
        self.assertEqual([bool(x) for x in (vals > 0)], want)

    def test_subnormal_ftz_probe(self):
        """The exact vector the `--use_fast_math` ban exists for.

        products 7.175e-43 and -3.587e-43, exact sum +3.587e-43 -> sign True.
        Under -ftz=true the subnormal flushes to +0.0 and the sign becomes False.
        """
        s = torch.tensor([7.175e-43], dtype=torch.float32) + torch.tensor([-3.587e-43])
        self.assertTrue(bool(s.item() > 0.0))
        self.assertTrue(0.0 < s.item() < torch.finfo(torch.float32).smallest_normal)

    def test_order_disagreement_witness(self):
        """No fp32 implementation can equal the exact oracle -- documentation.

        k = [2^15, 1, -2^15, 0.5], p = [2^15, 1, 2^15, 1]: exact dot 1.5,
        strictly sequential fp32 gives +0.5 (True), a pairwise tree gives 0.0
        (False). This is why `torch.equal(fused, cublas)` can never be a CI gate.
        """
        k = [2.0 ** 15, 1.0, -(2.0 ** 15), 0.5]
        p = [2.0 ** 15, 1.0, 2.0 ** 15, 1.0]
        seq = 0.0
        for a, b in zip(k, p):
            seq = float(torch.tensor(seq + a * b, dtype=torch.float32).item())
        t0 = torch.tensor(k[0] * p[0] + k[1] * p[1], dtype=torch.float32)
        t1 = torch.tensor(k[2] * p[2] + k[3] * p[3], dtype=torch.float32)
        tree = float((t0 + t1).item())
        exact = sum(a * b for a, b in zip(k, p))
        self.assertEqual(exact, 1.5)
        self.assertEqual(seq, 0.5)
        self.assertEqual(tree, 0.0)
        self.assertTrue(seq > 0.0 and not tree > 0.0)

    def test_all_zero_projections_hash_to_bucket_zero(self):
        proj = torch.zeros(64, 7 * 5)
        b = buckets_from_proj_torch(proj, 4, 16, 7, 5)
        self.assertTrue(torch.equal(b, torch.zeros_like(b)))
        self.assertTrue(torch.equal(emulate_pack(torch.zeros(64, 35), 4, 16, 7, 5), b))

    def test_negative_zero_hashes_with_positive_zero(self):
        """Pins `>` and not `>=`: -0.0 and +0.0 must land in the same bucket."""
        proj = torch.full((32, 12), -0.0)
        self.assertTrue(torch.equal(buckets_from_proj_torch(proj, 2, 16, 4, 3),
                                    torch.zeros(2, 4, 16, dtype=torch.uint8)))

    def test_adversarial_values_survive_the_emulator(self):
        g = torch.Generator().manual_seed(29)
        P, L, BH, T = 6, 9, 2, 20
        proj = torch.randn(BH * T, L * P, generator=g, dtype=torch.float32)
        specials = torch.tensor([0.0, -0.0, float("nan"), float("inf"), float("-inf"),
                                 1e-45, -1e-45, 1e38, -1e38], dtype=torch.float32)
        idx = torch.randint(0, proj.numel(), (specials.numel(),), generator=g)
        flat = proj.view(-1)
        flat[idx] = specials
        want = buckets_from_proj_torch(proj.clone(), BH, T, L, P)
        self.assertTrue(torch.equal(emulate_pack(proj, BH, T, L, P), want))

    def test_duplicated_rows_hash_identically_regardless_of_tile_position(self):
        g = torch.Generator().manual_seed(31)
        P, L = 5, 11
        row = torch.randn(1, L * P, generator=g)
        proj = row.repeat(300, 1)
        b = emulate_pack(proj, 1, 300, L, P)
        self.assertTrue(torch.equal(b[0, :, :1].expand(L, 300).contiguous(), b[0]))


class TestBucketDtype(unittest.TestCase):
    def test_widths(self):
        for P in range(1, 9):
            self.assertEqual(bucket_dtype(P), torch.uint8)
        for P in range(9, 16):
            self.assertEqual(bucket_dtype(P), torch.int16)
        for P in range(16, 31):
            self.assertEqual(bucket_dtype(P), torch.int32)
        with self.assertRaises(ValueError):
            bucket_dtype(31)
        with self.assertRaises(ValueError):
            bucket_dtype(0)

    def test_every_id_fits(self):
        """The whole point of the P<=15 cap: int16 is signed, so P=16 would NOT fit."""
        for P in range(1, 31):
            dt = bucket_dtype(P)
            self.assertLessEqual((1 << P) - 1, torch.iinfo(dt).max, f"P={P} overflows {dt}")
        self.assertEqual(KERNEL_MAX_PLANES, 15)
        self.assertGreater((1 << 16) - 1, torch.iinfo(torch.int16).max)


class TestBuildFlags(unittest.TestCase):
    """Permanent, CPU-only guard on the one compiler flag that breaks exactness."""

    def test_no_fast_math(self):
        flags = rarekv_lsh._CUDA_CFLAGS
        self.assertNotIn("--use_fast_math", flags)
        self.assertNotIn("-use_fast_math", flags)
        self.assertIn("-ftz=false", flags)

    def test_the_loader_passes_exactly_those_flags(self):
        src = inspect.getsource(rarekv_lsh._load_ext)
        self.assertIn("_CUDA_CFLAGS", src)
        self.assertNotIn("use_fast_math", src)
        # and BOTH extensions go through that one loader
        for fn in (rarekv_lsh._kernel_ext, rarekv_lsh._fused_ext):
            self.assertIn("_load_ext(", inspect.getsource(fn))

    def test_every_source_exists(self):
        here = os.path.join(os.path.dirname(rarekv_lsh.__file__), "csrc")
        for s in rarekv_lsh._EXT_SOURCES + rarekv_lsh._FUSED_EXT_SOURCES:
            self.assertTrue(os.path.exists(os.path.join(here, s)), s)

    def test_tier2_is_a_separate_lazily_built_extension(self):
        """The default path must not pay for Tier 2's 90 template instantiations.

        A single extension would put the fused kernel's nvcc time (several times
        the rest combined) in front of every ordinary eval run's first prefill.
        """
        self.assertNotIn("rarekv_fused.cu", rarekv_lsh._EXT_SOURCES)
        self.assertIn("rarekv_fused.cu", rarekv_lsh._FUSED_EXT_SOURCES)
        self.assertEqual(set(rarekv_lsh._EXT_SOURCES) & set(rarekv_lsh._FUSED_EXT_SOURCES),
                         set())
        # and each source set carries exactly one PYBIND11_MODULE
        here = os.path.join(os.path.dirname(rarekv_lsh.__file__), "csrc")
        for srcs in (rarekv_lsh._EXT_SOURCES, rarekv_lsh._FUSED_EXT_SOURCES):
            n = sum(sum(1 for line in open(os.path.join(here, s))
                        if line.startswith("PYBIND11_MODULE(")) for s in srcs)
            self.assertEqual(n, 1, srcs)

    def test_the_default_loader_never_touches_the_fused_extension(self):
        src = inspect.getsource(rarekv_lsh._kernel_ext)
        self.assertNotIn("_FUSED", src)
        self.assertIn("_EXT_SOURCES", src)


class TestSharedMemoryLimits(unittest.TestCase):
    """Both kernels ask for DYNAMIC shared memory, which is capped per block.

    48 KB without `cudaFuncSetAttribute`, `sharedMemPerBlockOptin` (227 KB on an
    H200) with it. Past either bound the launch fails with a bare
    `invalid argument`, so the python side must know the same arithmetic the
    kernels do and route around it to the (bit-identical) torch sequence.
    """

    def test_pack_smem_formula(self):
        for L, P, bm, want in [(70, 6, 0, 128 * 68), (70, 6, 128, 128 * 68),
                               (70, 6, 256, 256 * 68), (100, 9, 0, 128 * 132),
                               (100, 5, 0, 128 * 68), (80, 7, 0, 128 * 100)]:
            self.assertEqual(rarekv_lsh.pack_smem_bytes(L, P, bm), want, (L, P, bm))

    def test_the_target_grid_needs_no_opt_in(self):
        """P in 5..9 x L in {70, 100} at either block_m stays under the 48 KB default."""
        for P in range(5, 10):
            for L in (70, 100):
                for bm in (0, 128, 256):
                    self.assertLessEqual(rarekv_lsh.pack_smem_bytes(L, P, bm), 48 * 1024,
                                         f"P={P} L={L} block_m={bm}")

    def test_the_48k_thresholds_are_where_the_reviewers_computed_them(self):
        """block_m=128 breaks 48 KB at L*P=2825; block_m=256 at L*P=1289."""
        self.assertLessEqual(rarekv_lsh.pack_smem_bytes(2824, 1, 128), 48 * 1024)
        self.assertGreater(rarekv_lsh.pack_smem_bytes(2825, 1, 128), 48 * 1024)
        self.assertLessEqual(rarekv_lsh.pack_smem_bytes(1288, 1, 256), 48 * 1024)
        self.assertGreater(rarekv_lsh.pack_smem_bytes(1289, 1, 256), 48 * 1024)

    def test_a_huge_lp_config_is_still_answerable(self):
        """L=512, P=6 (L*P=3072) is inside max_bucket_slots and must not hard-fail.

        On CPU it takes the torch path anyway; the point is that the guard exists
        and is consulted, so the same config on a GPU degrades in speed only.
        """
        self.assertGreater(rarekv_lsh.pack_smem_bytes(512, 6, 128), 48 * 1024)
        g = torch.Generator().manual_seed(151)
        L, P, D = 512, 6, 16
        keys = torch.randn(1, 1, 4, D, generator=g)
        planes = torch.randn(D, L * P, generator=g)
        got, path = lsh_buckets(keys, planes, None, L, P, mode="pack")
        self.assertEqual(path, "torch")
        self.assertTrue(torch.equal(
            got, buckets_from_proj_torch(keys.reshape(-1, D) @ planes, 1, 4, L, P)))

    def test_hist_smem_ok(self):
        """R*4 bytes of privatised histogram: P<=13 fits 48 KB, P=14/15 need opt-in."""
        cap = rarekv_lsh._DEFAULT_SMEM_CAP
        self.assertTrue(rarekv_lsh.hist_smem_ok(1 << 13, None) == ((1 << 13) * 4 <= cap))
        # the helper is a pure comparison against the queried cap; with no CUDA the
        # cap is the conservative 48 KB default
        self.assertEqual(rarekv_lsh.device_smem_optin(None) >= 48 * 1024, True)
        if not CUDA:
            self.assertTrue(rarekv_lsh.hist_smem_ok(12288))
            self.assertFalse(rarekv_lsh.hist_smem_ok(16384))     # P=14
            self.assertFalse(rarekv_lsh.hist_smem_ok(32768))     # P=15

    def test_the_kernels_opt_in_rather_than_failing_the_launch(self):
        """Source guard: both dynamic-SMEM launches must call cudaFuncSetAttribute.

        This is the only cheap check available without a GPU, and the bug it
        pins -- a launch that dies with `invalid argument` at P>=14 (collide) or
        L*P>=1289 (pack, block_m=256) -- is invisible to every other test here.
        """
        here = os.path.join(os.path.dirname(rarekv_lsh.__file__), "csrc")
        for src in ("rarekv_pack.cu", "rarekv_collide.cu", "rarekv_fused.cu"):
            with open(os.path.join(here, src)) as fh:
                body = fh.read()
            self.assertIn("cudaFuncAttributeMaxDynamicSharedMemorySize", body, src)

    def test_the_fused_kernel_refuses_pre_ampere_rather_than_zeroing(self):
        """Below sm_80 the mma has no encoding; a no-op `#else` returns all zeros."""
        here = os.path.join(os.path.dirname(rarekv_lsh.__file__), "csrc")
        with open(os.path.join(here, "rarekv_fused.cu")) as fh:
            body = fh.read()
        self.assertEqual(body.count("__trap();"), 2)        # one per dtype
        self.assertIn("props->major >= 8", body)            # host-side refusal


class TestDispatchAndKillSwitch(unittest.TestCase):
    """Mode resolution is pure logic; it must be testable without a GPU."""

    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in ("PRISM_RAREKV_LSH", "PRISM_RAREKV_CUDA")}

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_resolution_order(self):
        os.environ.pop("PRISM_RAREKV_LSH", None)
        os.environ.pop("PRISM_RAREKV_CUDA", None)
        self.assertEqual(rarekv_lsh._resolve_mode("pack"), "pack")
        os.environ["PRISM_RAREKV_LSH"] = "fused"
        self.assertEqual(rarekv_lsh._resolve_mode("pack"), "fused")
        os.environ["PRISM_RAREKV_CUDA"] = "0"
        self.assertEqual(rarekv_lsh._resolve_mode("fused"), "torch")

    def test_bad_mode_raises(self):
        os.environ.pop("PRISM_RAREKV_LSH", None)
        with self.assertRaises(ValueError):
            rarekv_lsh._resolve_mode("nope")

    def test_env_var_can_upgrade_a_pack_config_to_fused(self):
        """PRISM_RAREKV_LSH=fused must reach the fused PRECONDITIONS, not be dropped.

        `score()` builds the transposed plane operand only for the fused path. If
        it keyed that off `self.lsh_mode` instead of the RESOLVED mode, the env var
        could only ever downgrade: it would resolve to "fused", fail
        `planes_t is not None`, and fall back to `pack` with nothing above INFO to
        say so -- and a divergence study driven by the documented env var would
        publish a Tier-2 flip rate of exactly zero for a path that never ran.
        """
        from eval_harness.kv_compression import get_kv_compressor
        from eval_harness.kv_compression.compressors import rarekv_sketch as rks

        seen = {}

        def spy(keys, planes, planes_t, L, P, **kw):
            seen["planes_t"] = planes_t
            seen["mode"] = kw.get("mode")
            return real(keys, planes, planes_t, L, P, **kw)

        real = rks.lsh_buckets
        os.environ["PRISM_RAREKV_LSH"] = "fused"
        os.environ.pop("PRISM_RAREKV_CUDA", None)
        sk = get_kv_compressor("rarekv", compression_ratio=0.5, n_planes=4, n_tables=5,
                               lsh_mode="pack")
        g = torch.Generator().manual_seed(149)
        k = torch.randn(1, 1, 8, 16, generator=g)
        v = torch.randn(1, 1, 8, 16, generator=g)
        rks.lsh_buckets = spy
        try:
            sk.score(SimpleNamespace(layer_idx=0), None, k, v, None, {})
        finally:
            rks.lsh_buckets = real
        self.assertIsNotNone(seen["planes_t"],
                             "the fused operand was not built, so `fused` was unreachable")
        self.assertEqual(seen["planes_t"].shape, (8 * 4, 16))

    def test_cpu_always_resolves_to_torch(self):
        """No CUDA -> `lsh_buckets` must still produce the reference answer."""
        g = torch.Generator().manual_seed(37)
        P, L, D = 6, 9, 32
        keys = torch.randn(1, 2, 20, D, generator=g, dtype=torch.float32)
        planes = torch.randn(D, L * P, generator=g, dtype=torch.float32)
        want = buckets_from_proj_torch(keys.reshape(-1, D) @ planes, 2, 20, L, P)
        for mode in ("torch", "pack", "fused"):
            with self.subTest(mode=mode):
                got, path = lsh_buckets(keys, planes, planes_to_planes_t(planes, L, P),
                                        L, P, mode=mode)
                self.assertEqual(path, "torch")
                self.assertTrue(torch.equal(got, want))


class TestSketchWiring(unittest.TestCase):
    """The knobs, their validation, and that nothing else moved."""

    def _sketch(self, **kw):
        from eval_harness.kv_compression import get_kv_compressor
        return get_kv_compressor("rarekv", **kw)

    def test_defaults(self):
        sk = self._sketch(compression_ratio=0.5)
        self.assertEqual(sk.lsh_mode, "pack")
        self.assertEqual(sk.block_m, 0)
        self.assertEqual(sk.gemm_chunk_rows, 0)
        self.assertEqual(sk.lsh_paths, ())

    def test_bad_knobs_raise(self):
        with self.assertRaises(ValueError):
            self._sketch(compression_ratio=0.5, lsh_mode="nope")
        with self.assertRaises(ValueError):
            self._sketch(compression_ratio=0.5, block_m=64)
        with self.assertRaises(ValueError):
            self._sketch(compression_ratio=0.5, gemm_chunk_rows=-1)

    def test_fused_refuses_p_above_the_kernel_dtype_cap(self):
        with self.assertRaises(ValueError):
            self._sketch(compression_ratio=0.5, n_planes=16, n_tables=4, lsh_mode="fused")
        # `pack` does NOT: it degrades to the (identical) torch sequence there.
        sk = self._sketch(compression_ratio=0.5, n_planes=16, n_tables=4, lsh_mode="pack")
        self.assertEqual(sk.n_planes, 16)

    def test_score_is_the_reference_on_cpu(self):
        """End to end on CPU: score() must equal a hand-rolled ICD computation."""
        g = torch.Generator().manual_seed(41)
        B, H, T, D, L, P = 1, 2, 24, 32, 5, 4
        keys = torch.randn(B, H, T, D, generator=g)
        values = torch.randn(B, H, T, D, generator=g)
        mod = SimpleNamespace(layer_idx=0)
        sk = self._sketch(compression_ratio=0.5, n_planes=P, n_tables=L)
        got = sk.score(mod, None, keys, values, None, {})
        self.assertEqual(sk.lsh_paths, ("torch",))

        planes = sk._planes(mod, D, keys.device, keys.dtype)
        bucket = buckets_from_proj_torch(keys.reshape(-1, D) @ planes, B * H, T, L, P)
        csum = torch.zeros(B * H, T, dtype=torch.int32)
        for bh in range(B * H):
            for l in range(L):
                ids = bucket[bh, l].to(torch.int64)
                counts = torch.bincount(ids, minlength=1 << P)
                csum[bh] += counts[ids].to(torch.int32)
        density = (csum.float() / L - 1.0) / float(T - 1)
        want = (sk.eps + density).pow(-sk.alpha).view(B, H, T)
        want = want * torch.linalg.vector_norm(values, dim=-1).float()
        self.assertTrue(torch.equal(got, want))

    def test_planes_t_cache_is_keyed_on_identity(self):
        mod = SimpleNamespace(layer_idx=0)
        sk = self._sketch(compression_ratio=0.5, n_planes=4, n_tables=5)
        a = sk._planes_t(mod, 32, torch.device("cpu"), torch.float32)
        self.assertIs(a, sk._planes_t(mod, 32, torch.device("cpu"), torch.float32))
        sk.seed += 1
        self.assertIsNot(a, sk._planes_t(mod, 32, torch.device("cpu"), torch.float32))


# ==========================================================================
# GPU-gated parity. Everything above runs on the login node; these need a card.
# ==========================================================================
@unittest.skipUnless(CUDA, "needs CUDA")
class TestPackKernelParity(unittest.TestCase):
    """G1/G2: Tier 1 is bit-identical, over shapes and over the float battery."""

    def setUp(self):
        if rarekv_lsh._kernel_ext() is None:
            self.skipTest("CUDA extension could not be built (no nvcc?)")

    def test_pack_matches_the_reference(self):
        g = torch.Generator(device="cuda").manual_seed(43)
        for P, L in PL_GRID:
            if P > KERNEL_MAX_PLANES:
                continue
            for BH, T in ((1, 1), (8, 999), (3, 4097)):
                for dtype in (torch.bfloat16, torch.float32):
                    with self.subTest(P=P, L=L, BH=BH, T=T, dtype=dtype):
                        proj = torch.randn(BH * T, L * P, generator=g,
                                           device="cuda", dtype=torch.float32).to(dtype)
                        want = buckets_from_proj_torch(proj.clone(), BH, T, L, P)
                        got = rarekv_lsh.pack_buckets(proj, BH, T, L, P)
                        self.assertTrue(torch.equal(got, want))

    def test_block_m_does_not_change_the_answer(self):
        g = torch.Generator(device="cuda").manual_seed(47)
        proj = torch.randn(4096, 70 * 6, generator=g, device="cuda", dtype=torch.bfloat16)
        want = buckets_from_proj_torch(proj.clone(), 4, 1024, 70, 6)
        for bm in (128, 256):
            self.assertTrue(torch.equal(
                rarekv_lsh.pack_buckets(proj, 4, 1024, 70, 6, block_m=bm), want))

    def test_adversarial_floats_on_device(self):
        g = torch.Generator(device="cuda").manual_seed(53)
        P, L, BH, T = 6, 70, 2, 1000
        proj = torch.randn(BH * T, L * P, generator=g, device="cuda", dtype=torch.float32)
        specials = torch.tensor([0.0, -0.0, float("nan"), float("inf"), float("-inf"),
                                 1e-45, -1e-45, 1e38, -1e38, 1.4e-45],
                                dtype=torch.float32, device="cuda")
        idx = torch.randint(0, proj.numel(), (specials.numel(),), generator=g, device="cuda")
        proj.view(-1)[idx] = specials
        want = buckets_from_proj_torch(proj.clone(), BH, T, L, P)
        self.assertTrue(torch.equal(rarekv_lsh.pack_buckets(proj, BH, T, L, P), want))

    def test_chunked_gemm_is_exact(self):
        """G4: chunking along M must not move a single bucket id."""
        g = torch.Generator(device="cuda").manual_seed(59)
        D, L, P, BH, T = 128, 70, 6, 8, 4096
        keys = torch.randn(1, BH, T, D, generator=g, device="cuda", dtype=torch.bfloat16)
        planes = torch.randn(D, L * P, generator=g, device="cuda", dtype=torch.bfloat16)
        want, _ = lsh_buckets(keys, planes, None, L, P, mode="pack")
        for chunk in (4096, 8192, 1 << 15):
            with self.subTest(chunk=chunk):
                got, _ = lsh_buckets(keys, planes, None, L, P, mode="pack",
                                     gemm_chunk_rows=chunk)
                self.assertTrue(torch.equal(got, want))

    def test_score_pack_equals_score_torch(self):
        from eval_harness.kv_compression import get_kv_compressor
        mod = SimpleNamespace(layer_idx=0)
        g = torch.Generator(device="cuda").manual_seed(61)
        k = torch.randn(1, 8, 4096, 128, device="cuda", dtype=torch.bfloat16, generator=g)
        v = torch.randn(1, 8, 4096, 128, device="cuda", dtype=torch.bfloat16, generator=g)
        for P, L in ((6, 70), (9, 100), (5, 100), (8, 50)):
            for gamma in (0.0, 1.0, 2.0):
                with self.subTest(P=P, L=L, gamma=gamma):
                    kw = dict(compression_ratio=0.9, n_planes=P, n_tables=L,
                              value_norm_power=gamma)
                    a = get_kv_compressor("rarekv", lsh_mode="pack", **kw)
                    b = get_kv_compressor("rarekv", lsh_mode="torch", **kw)
                    self.assertTrue(torch.equal(a.score(mod, None, k, v, None, {}),
                                                b.score(mod, None, k, v, None, {})))
                    self.assertEqual(a.lsh_paths, ("pack",))


@unittest.skipUnless(CUDA, "needs CUDA")
class TestFusedKernelGateA(unittest.TestCase):
    """G5 (GATE A): integer operands make every accumulation order agree bitwise.

    Draw K and W from {-1, 0, +1}: every product is an exact small integer and
    every partial sum is bounded by D <= 256, so cuBLAS's order, the fused
    kernel's order and an exact oracle all yield the identical fp32 bit pattern.
    That converts the whole swizzle / fragment / tail / store indexing surface
    into a `torch.equal` assertion with no tolerance and no census -- while being
    completely immune to the float ordering question.
    """

    def setUp(self):
        if rarekv_lsh._kernel_ext() is None:
            self.skipTest("CUDA extension could not be built (no nvcc?)")
        if torch.cuda.get_device_capability() < (8, 0):
            self.skipTest("fused kernel needs sm_80+")

    def _int_operands(self, shape, seed, dtype, hi=1):
        g = torch.Generator(device="cuda").manual_seed(seed)
        vals = torch.randint(-hi, hi + 1, shape, generator=g, device="cuda")
        return vals.to(dtype)

    def test_gate_a_over_the_grid(self):
        for P, L in ((5, 100), (6, 70), (7, 50), (8, 50), (9, 100), (1, 8), (15, 9)):
            for D in (64, 128, 256):
                for dtype in (torch.bfloat16, torch.float16):
                    with self.subTest(P=P, L=L, D=D, dtype=dtype):
                        BH, T = 8, 1030
                        keys = self._int_operands((1, BH, T, D), 67, dtype)
                        planes = self._int_operands((D, L * P), 71, dtype)
                        pt = planes_to_planes_t(planes, L, P)
                        want = buckets_from_proj_torch(
                            keys.reshape(-1, D).float() @ planes.float(), BH, T, L, P)
                        got, path = lsh_buckets(keys, planes, pt, L, P, mode="fused")
                        self.assertEqual(path, "fused")
                        self.assertTrue(torch.equal(got, want))

    def test_gate_a_ragged_shapes(self):
        P, L, D, dtype = 6, 70, 128, torch.bfloat16
        planes = self._int_operands((D, L * P), 73, dtype)
        pt = planes_to_planes_t(planes, L, P)
        for T in (128, 129, 255, 511, 4095, 4096):
            with self.subTest(T=T):
                keys = self._int_operands((1, 8, T, D), 79, dtype)
                want = buckets_from_proj_torch(
                    keys.reshape(-1, D).float() @ planes.float(), 8, T, L, P)
                got, path = lsh_buckets(keys, planes, pt, L, P, mode="fused")
                self.assertEqual(path, "fused")
                self.assertTrue(torch.equal(got, want))

    def test_serial_oracle_agrees_with_the_reference(self):
        """Bisection aid: if this passes and `fused` fails, the bug is in the mma."""
        P, L, D = 6, 17, 128
        keys = self._int_operands((1, 2, 300, D), 83, torch.bfloat16)
        planes = self._int_operands((D, L * P), 89, torch.bfloat16)
        want = buckets_from_proj_torch(keys.reshape(-1, D).float() @ planes.float(),
                                       2, 300, L, P)
        got = rarekv_lsh.serial_buckets(keys, planes.t().contiguous(), L, P)
        self.assertTrue(torch.equal(got, want))

    def test_fused_is_self_consistent(self):
        """GATE B: same answer twice in one process, and across grid shapes."""
        P, L, D = 6, 70, 128
        g = torch.Generator(device="cuda").manual_seed(97)
        keys = torch.randn(1, 8, 2048, D, generator=g, device="cuda", dtype=torch.bfloat16)
        planes = torch.randn(D, L * P, generator=g, device="cuda", dtype=torch.bfloat16)
        pt = planes_to_planes_t(planes, L, P)
        a, _ = lsh_buckets(keys, planes, pt, L, P, mode="fused")
        b, _ = lsh_buckets(keys, planes, pt, L, P, mode="fused")
        self.assertTrue(torch.equal(a, b))

    def test_fused_falls_back_to_pack_never_to_torch(self):
        """A degenerate shape must degrade to `pack`, which is bit-identical."""
        P, L, D = 6, 70, 128
        g = torch.Generator(device="cuda").manual_seed(101)
        keys = torch.randn(1, 1, 16, D, generator=g, device="cuda", dtype=torch.bfloat16)
        planes = torch.randn(D, L * P, generator=g, device="cuda", dtype=torch.bfloat16)
        got, path = lsh_buckets(keys, planes, planes_to_planes_t(planes, L, P), L, P,
                                mode="fused")
        self.assertEqual(path, "pack")
        want = buckets_from_proj_torch(keys.reshape(-1, D) @ planes, 1, 16, L, P)
        self.assertTrue(torch.equal(got, want))

    def test_unsupported_head_dim_falls_back_to_pack(self):
        P, L, D = 6, 70, 96                        # 96 is not instantiated
        g = torch.Generator(device="cuda").manual_seed(103)
        keys = torch.randn(1, 8, 2048, D, generator=g, device="cuda", dtype=torch.bfloat16)
        planes = torch.randn(D, L * P, generator=g, device="cuda", dtype=torch.bfloat16)
        _, path = lsh_buckets(keys, planes, planes_to_planes_t(planes, L, P), L, P,
                              mode="fused")
        self.assertEqual(path, "pack")


if __name__ == "__main__":
    unittest.main()


# The L ladder the method is actually swept over when tuning accuracy. The
# collision kernel templates its gather on L for a fixed set of values
# (40/50/60/70/80/100) and falls back to `gather_kernel_dyn` otherwise, so this
# ladder deliberately straddles both: 10/20/30/90 take the dynamic path, the
# rest take a specialisation. A regression that only hit the fallback would
# otherwise be invisible -- PL_GRID contains no L in {10, 20, 30, 60, 80, 90}.
L_LADDER = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100)


@unittest.skipUnless(CUDA, "needs CUDA")
class TestScoreParityOverTheLLadder(unittest.TestCase):
    """End-to-end `score()` must be IDENTICAL across modes for every swept L.

    The bucket-level parity tests above pin the packing. This pins the whole
    scorer -- histogram, gather, the (eps + density)**-alpha estimator and the
    value-norm weighting -- because that is what an accuracy run actually calls.
    A kernel that packs correctly but feeds a mis-strided [BH, L, T] view into
    the histogram would pass every test above and silently change every score.
    """

    def setUp(self):
        if rarekv_lsh._kernel_ext() is None:
            self.skipTest("CUDA extension could not be built (no nvcc?)")

    @staticmethod
    def _sketch(**kw):
        from eval_harness.kv_compression import get_kv_compressor
        return get_kv_compressor("rarekv", **kw)

    def _run(self, mode, L, P, keys, values, mod):
        sk = self._sketch(compression_ratio=0.5, n_planes=P, n_tables=L, lsh_mode=mode)
        return sk.score(mod, None, keys, values, None, {}), sk

    def test_score_is_identical_across_modes_for_every_L(self):
        g = torch.Generator(device="cuda").manual_seed(1234)
        B, H, T, D = 1, 4, 2048, 128
        keys = torch.randn(B, H, T, D, generator=g, device="cuda", dtype=torch.bfloat16)
        values = torch.randn(B, H, T, D, generator=g, device="cuda", dtype=torch.bfloat16)
        mod = SimpleNamespace(layer_idx=0)
        for L in L_LADDER:
            for P in (5, 6, 7, 8, 9):
                with self.subTest(L=L, P=P):
                    ref, sk_t = self._run("torch", L, P, keys, values, mod)
                    got, sk_p = self._run("pack", L, P, keys, values, mod)
                    self.assertEqual(sk_t.lsh_paths, ("torch",))
                    self.assertEqual(sk_p.lsh_paths, ("pack",),
                                     f"pack silently degraded at L={L}, P={P}")
                    self.assertTrue(torch.equal(got, ref),
                                    f"pack != torch at L={L}, P={P}")

    def test_fused_score_matches_on_the_L_ladder(self):
        """Tier 2 owes only its declared reference, so a divergence here is
        reported as a bucket-difference RATE, not asserted to zero -- except
        that it has measured 0.0 on all 45 benchmark cells, so a nonzero rate
        is a regression worth failing on until it is re-characterised."""
        g = torch.Generator(device="cuda").manual_seed(4321)
        B, H, T, D = 1, 4, 2048, 128
        keys = torch.randn(B, H, T, D, generator=g, device="cuda", dtype=torch.bfloat16)
        values = torch.randn(B, H, T, D, generator=g, device="cuda", dtype=torch.bfloat16)
        mod = SimpleNamespace(layer_idx=0)
        for L in L_LADDER:
            for P in (5, 9):
                with self.subTest(L=L, P=P):
                    ref, _ = self._run("torch", L, P, keys, values, mod)
                    got, sk = self._run("fused", L, P, keys, values, mod)
                    self.assertEqual(sk.lsh_paths, ("fused",),
                                     f"fused silently degraded at L={L}, P={P}")
                    self.assertTrue(torch.equal(got, ref),
                                    f"fused != torch at L={L}, P={P}")

    def test_estimator_matches_an_independent_implementation(self):
        """The ICD estimator itself, recomputed from scratch on the GPU path.

        `test_score_is_the_reference_on_cpu` pins this at L=5/P=4 on CPU. This
        repeats it across the L ladder on the kernel path, so a wrong `/L`, a
        wrong `(N-1)`, or an off-by-one in the gather cannot hide behind two
        code paths that share the same bug -- the parity tests above compare
        two arms that share the collision kernel, so a bug INSIDE it cancels.

        The value-norm weight is deliberately switched OFF here
        (`value_norm_power=0.0`). `torch.linalg.vector_norm` on a bf16 tensor
        returns **bf16**, so that term carries ~3e-3 of relative rounding --
        real, documented, and nothing to do with the estimator. Folding it in
        would force a tolerance loose enough to hide a genuine estimator bug.
        It is pinned separately in `test_value_norm_weight_is_applied_as_bf16`.
        """
        g = torch.Generator(device="cuda").manual_seed(99)
        B, H, T, D = 1, 2, 1024, 128
        keys = torch.randn(B, H, T, D, generator=g, device="cuda", dtype=torch.bfloat16)
        values = torch.randn(B, H, T, D, generator=g, device="cuda", dtype=torch.bfloat16)
        mod = SimpleNamespace(layer_idx=0)
        for L in (10, 50, 100):
            for P in (5, 9):
                with self.subTest(L=L, P=P):
                    sk = self._sketch(compression_ratio=0.5, n_planes=P, n_tables=L,
                                      lsh_mode="pack", value_norm_power=0.0)
                    got = sk.score(mod, None, keys, values, None, {})
                    self.assertEqual(sk.lsh_paths, ("pack",))

                    planes = sk._planes(mod, D, keys.device, keys.dtype)
                    bucket = buckets_from_proj_torch(
                        keys.reshape(-1, D) @ planes, B * H, T, L, P)   # [BH, L, T]
                    csum = torch.zeros(B * H, T, dtype=torch.float64, device="cuda")
                    for bh in range(B * H):
                        for l in range(L):
                            ids = bucket[bh, l].to(torch.int64)
                            counts = torch.bincount(ids, minlength=1 << P)
                            csum[bh] += counts[ids].to(torch.float64)
                    density = (csum / L - 1.0) / float(T - 1)
                    want = (sk.eps + density).pow(-sk.alpha).view(B, H, T)
                    # production reduces in fp32, so allow fp32 epsilon, not fp64
                    torch.testing.assert_close(got.double(), want, rtol=1e-5, atol=1e-6)

    def test_value_norm_weight_is_applied_as_bf16(self):
        """The gamma weight multiplies scores by ||v|| computed in the CACHE dtype.

        `vector_norm` on bf16 returns bf16 (fp32 accumulator, rounded output), so
        the weight carries ~3e-3 relative error against an fp64 norm. That is a
        property of the shipped baseline, not of the kernels -- both `torch` and
        `pack` reproduce it identically. Pinned so that a future change to the
        reduction dtype is a deliberate, visible decision: it would move every
        published rarekv retained set.
        """
        g = torch.Generator(device="cuda").manual_seed(7)
        B, H, T, D, L, P = 1, 2, 1024, 128, 50, 6
        keys = torch.randn(B, H, T, D, generator=g, device="cuda", dtype=torch.bfloat16)
        values = torch.randn(B, H, T, D, generator=g, device="cuda", dtype=torch.bfloat16)
        mod = SimpleNamespace(layer_idx=0)
        plain = self._sketch(compression_ratio=0.5, n_planes=P, n_tables=L,
                             lsh_mode="pack", value_norm_power=0.0)
        weighted = self._sketch(compression_ratio=0.5, n_planes=P, n_tables=L,
                                lsh_mode="pack", value_norm_power=1.0)
        icd = plain.score(mod, None, keys, values, None, {})
        got = weighted.score(mod, None, keys, values, None, {})
        vn = torch.linalg.vector_norm(values, dim=-1).float()
        self.assertEqual(torch.linalg.vector_norm(values, dim=-1).dtype, torch.bfloat16)
        torch.testing.assert_close(got, icd * vn, rtol=1e-6, atol=1e-6)
