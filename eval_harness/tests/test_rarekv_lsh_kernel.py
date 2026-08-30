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
    DEFAULT_BLOCK, MIN_REDUCTION, collision_sums, collision_sums_torch, should_use_triton,
)

CUDA = torch.cuda.is_available()
CONFIGS = [(2, 40), (3, 50), (8, 50), (6, 80), (10, 60)]      # the profiled (P, L) grid


def _buckets(BH, T, L, R, seed=0, device="cpu"):
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randint(0, R, (BH, T, L), generator=g, device=device, dtype=torch.int32)


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


if __name__ == "__main__":
    unittest.main()
