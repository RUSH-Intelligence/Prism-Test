"""Contract for the RareKV collision kernel (eval_harness/kernels/rarekv_lsh.py).

The Triton path exists only to be FASTER; it must never be a different answer.
Both fused steps are integer (an order-independent histogram and an integer
gather-reduce), so equality is exact by construction, not approximate -- these
tests pin that, plus the dispatch rule that keeps large-R configs off the slow
path.

The Triton kernel itself needs CUDA, so those tests skip on CPU. Everything
that can be checked without a GPU is checked without one.
"""

from __future__ import annotations

import unittest

import torch

from eval_harness.kernels import rarekv_lsh
from eval_harness.kernels.rarekv_lsh import (
    DEFAULT_BLOCK, MIN_REDUCTION, bucket_dtype, collision_sums, collision_sums_lmajor,
    collision_sums_torch, should_use_triton,
)

CUDA = torch.cuda.is_available()
# the profiled (P, L) grid, plus the L=70/100 target grid the unrolled gather
# kernel now has explicit `case` arms for
CONFIGS = [(2, 40), (3, 50), (8, 50), (6, 80), (10, 60), (6, 70), (5, 100), (9, 100)]


def _buckets(BH, T, L, R, seed=0, device="cpu", dtype=torch.int32):
    g = torch.Generator(device=device).manual_seed(seed)
    b = torch.randint(0, R, (BH, T, L), generator=g, device=device, dtype=torch.int32)
    return b.to(dtype)


class TestTorchReference(unittest.TestCase):
    """The torch path is the definition; check it against a naive count."""

    def test_matches_a_naive_per_table_count(self):
        BH, T, L, R = 3, 97, 6, 8
        b = _buckets(BH, T, L, R)
        got = collision_sums_torch(b, R)
        want = torch.zeros(BH, T, dtype=torch.int32)
        for i in range(BH):
            for l in range(L):
                counts = torch.bincount(b[i, :, l].long(), minlength=R)
                want[i] += counts[b[i, :, l].long()].to(torch.int32)
        self.assertTrue(torch.equal(got, want))

    def test_uniform_buckets_give_uniform_counts(self):
        """Every key alone in its bucket -> every count is 1 -> csum == L."""
        BH, L, R = 2, 5, 16
        b = torch.arange(R, dtype=torch.int32).view(1, R, 1).expand(BH, R, L).contiguous()
        self.assertTrue(torch.equal(collision_sums_torch(b, R),
                                    torch.full((BH, R), L, dtype=torch.int32)))

    def test_all_in_one_bucket(self):
        """Every key in the same bucket -> every count is T -> csum == L*T."""
        BH, T, L, R = 2, 33, 4, 8
        b = torch.zeros(BH, T, L, dtype=torch.int32)
        self.assertTrue(torch.equal(collision_sums_torch(b, R),
                                    torch.full((BH, T), L * T, dtype=torch.int32)))

    def test_does_not_mutate_its_input(self):
        """The offset fold must not be applied in place: score() reuses `bucket`."""
        b = _buckets(2, 64, 5, 8)
        before = b.clone()
        collision_sums_torch(b, 8)
        self.assertTrue(torch.equal(b, before))


class TestSubWordBuckets(unittest.TestCase):
    """uint8/int16 ids are what the packing kernels emit; the torch path must eat them."""

    def test_torch_path_upcasts_before_folding_in_the_table_offset(self):
        """The offset reaches (L-1)*R, which overflows both sub-word dtypes."""
        BH, T, L, R = 2, 64, 60, 256
        ref = _buckets(BH, T, L, R, seed=5)
        for dt in (torch.uint8, torch.int16, torch.int32):
            with self.subTest(dtype=dt):
                self.assertTrue(torch.equal(collision_sums_torch(ref.to(dt), R),
                                            collision_sums_torch(ref, R)))

    def test_bucket_dtype_matches_what_the_torch_path_accepts(self):
        for P in (1, 8, 9, 15):
            dt = bucket_dtype(P)
            b = _buckets(2, 32, 4, 1 << P, seed=P, dtype=dt)
            self.assertEqual(b.dtype, dt)
            csum = collision_sums_torch(b, 1 << P)
            self.assertEqual(csum.dtype, torch.int32)


class TestLMajorEntryPoint(unittest.TestCase):
    """`collision_sums_lmajor` must agree with the T-major reference on any device."""

    def test_matches_the_t_major_reference(self):
        for P, L in CONFIGS:
            with self.subTest(P=P, L=L):
                R = 1 << P
                b = _buckets(3, 257, L, R, seed=P + L, dtype=bucket_dtype(P))
                lm = b.permute(0, 2, 1).contiguous()
                self.assertTrue(torch.equal(collision_sums_lmajor(lm, R),
                                            collision_sums_torch(b, R)))


class TestDispatch(unittest.TestCase):
    """The kernel only wins when the atomic-reduction factor BLOCK/R is large."""

    def test_never_dispatches_on_cpu(self):
        for P, _ in CONFIGS:
            self.assertFalse(should_use_triton(1 << P, torch.device("cpu")))

    def test_threshold_is_the_reduction_factor(self):
        cpu_ok = rarekv_lsh.HAVE_TRITON
        if not cpu_ok:
            self.skipTest("triton not installed")
        cuda = torch.device("cuda")
        # measured: R=256 wins (factor 8), R=1024 loses (factor 2)
        self.assertTrue(should_use_triton(256, cuda, block=DEFAULT_BLOCK))
        self.assertFalse(should_use_triton(1024, cuda, block=DEFAULT_BLOCK))
        self.assertEqual(DEFAULT_BLOCK // MIN_REDUCTION, 256)

    def test_a_bigger_block_admits_a_bigger_R(self):
        if not rarekv_lsh.HAVE_TRITON:
            self.skipTest("triton not installed")
        self.assertTrue(should_use_triton(1024, torch.device("cuda"), block=8192))


@unittest.skipUnless(CUDA and rarekv_lsh.HAVE_TRITON, "needs CUDA + triton")
class TestTritonIsActuallyExercised(unittest.TestCase):
    """Closes a real hole: `collision_sums` tries the CUDA extension FIRST.

    On any node where the extension builds, every test that went through
    `collision_sums` (or through `RareKVSketch.score`) exercised CUDA, never
    Triton. These call `collision_sums_triton` directly so the Triton kernel is
    genuinely covered wherever it is installed.
    """

    def test_triton_directly_across_the_grid(self):
        for P, L in CONFIGS:
            if not should_use_triton(1 << P, torch.device("cuda")):
                continue
            with self.subTest(P=P, L=L):
                R = 1 << P
                b = _buckets(4, 3000, L, R, seed=P, device="cuda")
                self.assertTrue(torch.equal(rarekv_lsh.collision_sums_triton(b, R),
                                            collision_sums_torch(b, R)))


@unittest.skipUnless(CUDA and rarekv_lsh.HAVE_TRITON, "needs CUDA + triton")
class TestTritonBitIdentical(unittest.TestCase):
    """The whole point: identical output, not merely close."""

    def test_bit_identical_across_the_profiled_grid(self):
        for P, L in CONFIGS:
            for T in (1, 2, 1000, 8192):
                with self.subTest(P=P, L=L, T=T):
                    R = 1 << P
                    b = _buckets(8, T, L, R, device="cuda")
                    self.assertTrue(torch.equal(
                        rarekv_lsh.collision_sums_triton(b, R), collision_sums_torch(b, R)))

    def test_bit_identical_on_skewed_buckets(self):
        """Real LSH buckets are unbalanced; uniform randint would hide contention bugs."""
        for P, L in CONFIGS:
            with self.subTest(P=P, L=L):
                R = 1 << P
                g = torch.Generator(device="cuda").manual_seed(1)
                z = torch.randn(8, 4096, L, device="cuda", generator=g)
                b = ((z.abs() / 4.0).clamp(max=0.999) * R).to(torch.int32).contiguous()
                self.assertTrue(torch.equal(
                    rarekv_lsh.collision_sums_triton(b, R), collision_sums_torch(b, R)))

    def test_partial_tiles(self):
        """T not a multiple of BLOCK: masked lanes must not land in bin 0."""
        for T in (DEFAULT_BLOCK - 1, DEFAULT_BLOCK + 1, 3 * DEFAULT_BLOCK + 7):
            with self.subTest(T=T):
                b = _buckets(2, T, 5, 8, device="cuda")
                self.assertTrue(torch.equal(
                    rarekv_lsh.collision_sums_triton(b, 8), collision_sums_torch(b, 8)))

    def test_block_size_does_not_change_the_answer(self):
        b = _buckets(4, 5000, 8, 16, device="cuda")
        ref = collision_sums_torch(b, 16)
        for block in (1024, 2048, 4096):
            with self.subTest(block=block):
                self.assertTrue(torch.equal(rarekv_lsh.collision_sums_triton(b, 16, block=block), ref))

    def test_full_scorer_is_bit_identical(self):
        from eval_harness.kv_compression import get_kv_compressor
        from types import SimpleNamespace
        mod = SimpleNamespace(layer_idx=0, head_dim=128)
        g = torch.Generator(device="cuda").manual_seed(0)
        k = torch.randn(1, 8, 4096, 128, device="cuda", dtype=torch.bfloat16, generator=g)
        v = torch.randn(1, 8, 4096, 128, device="cuda", dtype=torch.bfloat16, generator=g)
        for P, L in CONFIGS:
            with self.subTest(P=P, L=L):
                kw = dict(compression_ratio=0.9, n_planes=P, n_tables=L)
                a = get_kv_compressor("rarekv", use_triton=True, **kw).score(mod, None, k, v, None, {})
                b = get_kv_compressor("rarekv", use_triton=False, **kw).score(mod, None, k, v, None, {})
                self.assertTrue(torch.equal(a, b))


@unittest.skipUnless(CUDA, "needs CUDA")
class TestCudaPath(unittest.TestCase):
    """The CUDA extension is the preferred path; it must agree exactly."""

    def setUp(self):
        if rarekv_lsh._cuda_ext() is None:
            self.skipTest("CUDA extension could not be built (no nvcc?)")

    def test_bit_identical_across_the_profiled_grid(self):
        for P, L in CONFIGS:
            for T in (1, 999, 4096):
                with self.subTest(P=P, L=L, T=T):
                    R = 1 << P
                    b = _buckets(8, T, L, R, device="cuda")
                    self.assertTrue(torch.equal(
                        rarekv_lsh.collision_sums_cuda(b, R), collision_sums_torch(b, R)))

    def test_sub_word_bucket_dtypes(self):
        """uint8 (P<=8) and int16 (9<=P<=15) are what the packing kernels emit.

        P=15 is the one that exercises the histogram's shared-memory opt-in:
        R*4 = 131072 B is far past the 48 KB default per-block cap, so without
        `cudaFuncSetAttribute` this cell fails the launch with `invalid argument`
        rather than returning a wrong answer.
        """
        for P, L in ((5, 100), (6, 70), (8, 50), (9, 100), (15, 5)):
            with self.subTest(P=P, L=L):
                R = 1 << P
                if not rarekv_lsh.hist_smem_ok(R, torch.device("cuda")):
                    self.skipTest(f"R={R} exceeds this device's opt-in shared memory")
                ref = _buckets(4, 2049, L, R, seed=P, device="cuda")
                want = collision_sums_torch(ref, R)
                sub = ref.to(bucket_dtype(P))
                self.assertTrue(torch.equal(rarekv_lsh.collision_sums_cuda(sub, R), want))
                lm = sub.permute(0, 2, 1).contiguous()
                self.assertTrue(torch.equal(collision_sums_lmajor(lm, R), want))

    def test_uint8_is_refused_above_R_256(self):
        b = _buckets(2, 64, 4, 256, device="cuda").to(torch.uint8)
        with self.assertRaises(RuntimeError):
            rarekv_lsh._kernel_ext().collide(b.permute(0, 2, 1).contiguous(), 512)

    def test_hist_blocks_override_does_not_change_the_answer(self):
        """The new hist_blocks heuristic is integer-only, hence exact at any value."""
        b = _buckets(4, 8192, 60, 1024, device="cuda")
        lm = b.permute(0, 2, 1).contiguous()
        want = collision_sums_torch(b, 1024)
        ext = rarekv_lsh._kernel_ext()
        for hb in (0, 1, 4, 64, 512):
            with self.subTest(hist_blocks=hb):
                self.assertTrue(torch.equal(ext.collide(lm, 1024, 256, hb, 256), want))

    def test_dispatch_prefers_cuda(self):
        """Even at R=1024, where the Triton path would regress, CUDA is chosen."""
        b = _buckets(2, 4096, 60, 1024, device="cuda")
        self.assertTrue(torch.equal(rarekv_lsh.collision_sums(b, 1024),
                                    collision_sums_torch(b, 1024)))
        self.assertFalse(should_use_triton(1024, torch.device("cuda")))

    def test_full_score_at_every_kernel_P(self):
        """End to end, not just `collide`: P=14/15 must not die at the histogram.

        The pack kernel happily emits int16 ids up to P=15, so a config that
        `bucket_dtype` and `KERNEL_MAX_PLANES` both advertise has to survive the
        collision histogram too -- which needs 65536/131072 B of shared memory.
        """
        from types import SimpleNamespace
        from eval_harness.kv_compression import get_kv_compressor
        mod = SimpleNamespace(layer_idx=0, head_dim=64)
        g = torch.Generator(device="cuda").manual_seed(5)
        k = torch.randn(1, 2, 1024, 64, device="cuda", dtype=torch.bfloat16, generator=g)
        v = torch.randn(1, 2, 1024, 64, device="cuda", dtype=torch.bfloat16, generator=g)
        for P in (12, 13, 14, 15):
            with self.subTest(P=P):
                kw = dict(compression_ratio=0.9, n_planes=P, n_tables=4,
                          max_bucket_slots=1 << 26)
                a = get_kv_compressor("rarekv", lsh_mode="pack", **kw)
                b = get_kv_compressor("rarekv", lsh_mode="torch", **kw)
                self.assertTrue(torch.equal(a.score(mod, None, k, v, None, {}),
                                            b.score(mod, None, k, v, None, {})))

    def test_all_three_paths_agree(self):
        for P, L in CONFIGS:
            with self.subTest(P=P, L=L):
                R = 1 << P
                b = _buckets(4, 2048, L, R, device="cuda")
                ref = collision_sums_torch(b, R)
                self.assertTrue(torch.equal(rarekv_lsh.collision_sums_cuda(b, R), ref))
                if should_use_triton(R, b.device):
                    self.assertTrue(torch.equal(rarekv_lsh.collision_sums_triton(b, R), ref))


if __name__ == "__main__":
    unittest.main()
