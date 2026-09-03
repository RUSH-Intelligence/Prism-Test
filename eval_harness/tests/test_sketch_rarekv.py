"""Contract for RareKV (LSH inverse-collision-density eviction).

Weight-free: fake attention modules + synthetic K/V, per repo convention.
The load-bearing tests are (a) the estimator matches a literal transcription of
the paper equation, (b) it is bit-reproducible, and (c) rare keys actually win.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from eval_harness.kv_compression import get_kv_compressor, get_kv_compressor_class
from eval_harness.kv_compression.compressors.rarekv_sketch import RareKVSketch

B, H, T, D = 2, 3, 64, 16


def _module(layer_idx: int = 0):
    return SimpleNamespace(layer_idx=layer_idx, head_dim=D)


def _kv(seed: int = 0, t: int = T):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(B, H, t, D, generator=g), torch.randn(B, H, t, D, generator=g))


def _brute_force(sketch, module, keys, values):
    """Literal, loop-per-table transcription of the ICD equation."""
    P, L = sketch.n_planes, sketch.n_tables
    planes = sketch._planes(module, keys.shape[-1], keys.device, keys.dtype)
    out = torch.zeros(keys.shape[0], keys.shape[1], keys.shape[2])
    n = keys.shape[2]
    for b in range(keys.shape[0]):
        for h in range(keys.shape[1]):
            proj = keys[b, h].float() @ planes
            dens = torch.zeros(n)
            for l in range(L):
                bits = (proj[:, l * P:(l + 1) * P] > 0).long()
                bucket = sum(int(2 ** p) * bits[:, p] for p in range(P))
                counts = torch.bincount(bucket, minlength=2 ** P)
                dens += (counts[bucket].float() - 1.0) / max(n - 1, 1)
            icd = (sketch.eps + dens / L) ** (-sketch.alpha)
            if sketch.value_norm_power:
                icd = icd * values[b, h].float().norm(dim=-1) ** sketch.value_norm_power
            out[b, h] = icd
    return out


class TestMatchesTheEquation(unittest.TestCase):
    def test_bit_exact_vs_brute_force(self):
        keys, values = _kv()
        for P, L, alpha, gamma in ((4, 7, 1.3, 1.0), (2, 3, 1.0, 0.0),
                                   (6, 5, 0.5, 2.0), (1, 2, 2.0, 1.0)):
            with self.subTest(P=P, L=L, alpha=alpha, gamma=gamma):
                s = get_kv_compressor("rarekv", compression_ratio=0.5, n_planes=P, n_tables=L,
                                      alpha=alpha, value_norm_power=gamma)
                got = s.score(_module(), None, keys, values, None, {})
                self.assertTrue(torch.allclose(got, _brute_force(s, _module(), keys, values),
                                               rtol=1e-5, atol=1e-4))

    def test_density_bounds(self):
        """(C-1)/(N-1) averaged over L lies in [0,1], so scores lie in a known range."""
        keys, values = _kv()
        s = get_kv_compressor("rarekv", compression_ratio=0.5, n_planes=3, n_tables=5,
                              value_norm_power=0.0, eps=1e-6, alpha=1.0)
        sc = s.score(_module(), None, keys, values, None, {})
        self.assertLessEqual(float(sc.max()), (1e-6) ** -1.0 + 1e-3)   # alone in every bucket
        self.assertGreaterEqual(float(sc.min()), (1.0 + 1e-6) ** -1.0 - 1e-3)  # collides with all

    def test_alpha_zero_reduces_to_value_norm(self):
        keys, values = _kv()
        s = get_kv_compressor("rarekv", compression_ratio=0.5, alpha=0.0, value_norm_power=1.0)
        got = s.score(_module(), None, keys, values, None, {})
        self.assertTrue(torch.allclose(got, values.float().norm(dim=-1), rtol=1e-5, atol=1e-5))


class TestRarityBehaviour(unittest.TestCase):
    def test_duplicated_keys_score_below_unique_keys(self):
        """The whole premise: crowded bucket -> low score -> evicted first."""
        torch.manual_seed(0)
        uniq = torch.randn(1, 1, 8, D) * 3.0
        dup = uniq[:, :, :1].repeat(1, 1, 24, 1)          # 24 identical keys
        keys = torch.cat([dup, uniq[:, :, 1:]], dim=2)     # [1,1,31,D]
        values = torch.ones(1, 1, keys.shape[2], D)        # equal value norms
        s = get_kv_compressor("rarekv", compression_ratio=0.5, n_planes=8, n_tables=32,
                              value_norm_power=0.0)
        sc = s.score(_module(), None, keys, values, None, {})[0, 0]
        self.assertLess(float(sc[:24].max()), float(sc[24:].min()),
                        "every duplicated key must score below every unique key")

    def test_survivors_are_the_unique_keys(self):
        torch.manual_seed(0)
        uniq = torch.randn(1, 1, 8, D) * 3.0
        keys = torch.cat([uniq[:, :, :1].repeat(1, 1, 24, 1), uniq[:, :, 1:]], dim=2)
        values = torch.ones(1, 1, 31, D)
        s = get_kv_compressor("rarekv", compression_ratio=0.5, n_planes=8, n_tables=32,
                              value_norm_power=0.0)
        kept = s.score(_module(), None, keys, values, None, {})[0, 0].topk(15).indices
        self.assertTrue(bool((kept >= 24).all()) or int((kept >= 24).sum()) >= 7,
                        "the 7 unique keys should all survive a 15-token budget")

    def test_value_norm_breaks_ties(self):
        keys = torch.randn(1, 1, 16, D)
        values = torch.ones(1, 1, 16, D)
        values[0, 0, 5] *= 10.0
        s = get_kv_compressor("rarekv", compression_ratio=0.5, alpha=0.0, value_norm_power=1.0)
        sc = s.score(_module(), None, keys, values, None, {})[0, 0]
        self.assertEqual(int(sc.argmax()), 5)


class TestReproducibility(unittest.TestCase):
    def test_two_instances_agree_bitwise(self):
        keys, values = _kv()
        a = get_kv_compressor("rarekv", compression_ratio=0.5, seed=7)
        b = get_kv_compressor("rarekv", compression_ratio=0.5, seed=7)
        self.assertTrue(torch.equal(a.score(_module(), None, keys, values, None, {}),
                                    b.score(_module(), None, keys, values, None, {})))

    def test_repeated_calls_agree_bitwise(self):
        """Integer scatter_add_ is exact, so atomics reordering cannot change it."""
        keys, values = _kv()
        s = get_kv_compressor("rarekv", compression_ratio=0.5)
        first = s.score(_module(), None, keys, values, None, {})
        for _ in range(4):
            self.assertTrue(torch.equal(first, s.score(_module(), None, keys, values, None, {})))

    def test_different_seed_changes_planes(self):
        keys, values = _kv()
        a = get_kv_compressor("rarekv", compression_ratio=0.5, seed=1, n_planes=6, n_tables=8)
        b = get_kv_compressor("rarekv", compression_ratio=0.5, seed=2, n_planes=6, n_tables=8)
        self.assertFalse(torch.equal(a.score(_module(), None, keys, values, None, {}),
                                     b.score(_module(), None, keys, values, None, {})))

    def test_global_rng_untouched(self):
        """A seeded compressor must not perturb the run's sampling stream."""
        keys, values = _kv()
        torch.manual_seed(1234)
        before = torch.randn(4)
        torch.manual_seed(1234)
        get_kv_compressor("rarekv", compression_ratio=0.5).score(
            _module(), None, keys, values, None, {})
        self.assertTrue(torch.equal(before, torch.randn(4)))

    def test_per_layer_planes_decorrelate_layers(self):
        keys, values = _kv()
        s = get_kv_compressor("rarekv", compression_ratio=0.5, per_layer_planes=True)
        self.assertFalse(torch.equal(s.score(_module(0), None, keys, values, None, {}),
                                     s.score(_module(3), None, keys, values, None, {})))
        shared = get_kv_compressor("rarekv", compression_ratio=0.5, per_layer_planes=False)
        self.assertTrue(torch.equal(shared.score(_module(0), None, keys, values, None, {}),
                                    shared.score(_module(3), None, keys, values, None, {})))

    def test_planes_cached_not_redrawn(self):
        s = get_kv_compressor("rarekv", compression_ratio=0.5)
        p1 = s._planes(_module(2), D, torch.device("cpu"), torch.float32)
        self.assertIs(p1, s._planes(_module(2), D, torch.device("cpu"), torch.float32))


class TestCompressAndShapes(unittest.TestCase):
    def test_budget_is_exact(self):
        keys, values = _kv()
        for ratio in (0.1, 0.5, 0.9):
            with self.subTest(ratio=ratio):
                s = get_kv_compressor("rarekv", compression_ratio=ratio)
                k, v = s.compress(_module(), None, keys, values, None, {})
                self.assertEqual(k.shape[2], int(T * (1 - ratio)))
                self.assertEqual(v.shape[2], int(T * (1 - ratio)))

    def test_ratio_zero_is_identity(self):
        keys, values = _kv()
        k, v = get_kv_compressor("rarekv", compression_ratio=0.0).compress(
            _module(), None, keys, values, None, {})
        self.assertTrue(torch.equal(k, keys) and torch.equal(v, values))

    def test_score_shape_and_finiteness(self):
        keys, values = _kv()
        sc = get_kv_compressor("rarekv", compression_ratio=0.5).score(
            _module(), None, keys, values, None, {})
        self.assertEqual(tuple(sc.shape), (B, H, T))
        self.assertTrue(bool(torch.isfinite(sc).all()))

    def test_single_token_sequence(self):
        """T=1 makes N-1 zero; the guard must keep it finite."""
        keys, values = _kv(t=1)
        sc = get_kv_compressor("rarekv", compression_ratio=0.5).score(
            _module(), None, keys, values, None, {})
        self.assertTrue(bool(torch.isfinite(sc).all()))

    def test_accepts_every_float_value_dtype(self):
        """vector_norm refuses a narrowing dtype, so fp64 values must not crash."""
        keys, _ = _kv()
        for dt in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            with self.subTest(dtype=dt):
                sc = get_kv_compressor("rarekv", compression_ratio=0.5).score(
                    _module(), None, keys.to(dt), torch.randn(B, H, T, D).to(dt), None, {})
                self.assertEqual(sc.dtype, torch.float32)
                self.assertTrue(bool(torch.isfinite(sc).all()))

    def test_plane_cache_keyed_on_seed(self):
        """Mutating seed on a warm instance must not return the previous planes.

        A sweep driver reusing one instance across seeds would otherwise report
        identical results for every seed.
        """
        s = get_kv_compressor("rarekv", compression_ratio=0.5, seed=42)
        p42 = s._planes(_module(0), D, torch.device("cpu"), torch.float32).clone()
        s.seed = 123
        self.assertFalse(torch.equal(
            p42, s._planes(_module(0), D, torch.device("cpu"), torch.float32)))

    def test_planes_pinned_to_cpu_generation(self):
        """torch.randn must not obey set_default_device, or the CPU-generator
        reproducibility contract breaks."""
        s = get_kv_compressor("rarekv", compression_ratio=0.5)
        self.assertEqual(s._draw(0, D).device.type, "cpu")

    def test_plane_cache_keyed_on_L_and_P(self):
        """A [D, L*P] plane matrix is reshaped as (L, P); a cache hit across a
        different (L, P) would silently regroup columns into the wrong tables."""
        s = get_kv_compressor("rarekv", compression_ratio=0.5, n_planes=8, n_tables=50)
        first = s._planes(_module(0), D, torch.device("cpu"), torch.float32)
        s.n_planes, s.n_tables = 10, 40          # same L*P = 400, different grouping
        second = s._planes(_module(0), D, torch.device("cpu"), torch.float32)
        self.assertIsNot(first, second)
        self.assertEqual(second.shape, (D, 400))

    def test_needs_no_rope_or_attention(self):
        """No position_embeddings, no attentions: composes with NemotronH-style hybrids."""
        keys, values = _kv()
        sc = get_kv_compressor("rarekv", compression_ratio=0.5).score(
            _module(), None, keys, values, None, {})
        self.assertTrue(bool(torch.isfinite(sc).all()))


class TestConfigValidation(unittest.TestCase):
    def test_rejects_bad_parameters(self):
        for kw in ({"n_planes": 0}, {"n_planes": 31}, {"n_tables": 0},
                   {"alpha": -1.0}, {"eps": 0.0}):
            with self.subTest(**kw):
                with self.assertRaises(ValueError):
                    get_kv_compressor("rarekv", compression_ratio=0.5, **kw)

    def test_bucket_table_guard_at_construction(self):
        """L * 2**P alone over budget -> reject before a model is ever loaded."""
        with self.assertRaises(ValueError) as cm:
            get_kv_compressor("rarekv", compression_ratio=0.5, n_planes=20, n_tables=64,
                              max_bucket_slots=1024)
        self.assertIn("max_bucket_slots", str(cm.exception))

    def test_bucket_table_guard_at_score(self):
        """Only the B*H factor pushes it over -> caught in score()."""
        keys, values = _kv()                      # B=2, H=3 -> x6
        s = get_kv_compressor("rarekv", compression_ratio=0.5, n_planes=4, n_tables=8,
                              max_bucket_slots=256)     # 8*16=128 ok, *6 = 768 not
        with self.assertRaises(ValueError) as cm:
            s.score(_module(), None, keys, values, None, {})
        self.assertIn("max_bucket_slots", str(cm.exception))

    def test_rejects_score_overflowing_configs(self):
        """eps**-alpha must stay inside fp32.

        Otherwise a collision-free key scores +inf, and inf * 0 (a zero-norm value
        row) is NaN -- which torch.topk ranks ABOVE +inf, so the least informative
        token would be retained first.
        """
        for kw in ({"alpha": 7.0, "eps": 1e-6}, {"alpha": 1.0, "eps": 1e-40},
                   {"alpha": 20.0, "eps": 1e-3}):
            with self.subTest(**kw):
                with self.assertRaises(ValueError) as cm:
                    get_kv_compressor("rarekv", compression_ratio=0.5, **kw)
                self.assertIn("overflow", str(cm.exception).lower())

    def test_accepts_configs_just_inside_the_envelope(self):
        for kw in ({"alpha": 6.0, "eps": 1e-6}, {"alpha": 1.0, "eps": 1e-38},
                   {"alpha": 0.0, "eps": 1e-300}):     # alpha=0 -> no exponent at all
            with self.subTest(**kw):
                get_kv_compressor("rarekv", compression_ratio=0.5, **kw)

    def test_scores_stay_finite_for_every_allowed_config(self):
        keys, values = _kv()
        values[0, 0, 0] = 0.0                     # a zero-norm value row: the inf*0 trap
        for kw in ({"alpha": 6.0, "eps": 1e-6}, {"alpha": 1.0}, {"alpha": 3.0, "eps": 1e-8}):
            with self.subTest(**kw):
                sc = get_kv_compressor("rarekv", compression_ratio=0.5, **kw).score(
                    _module(), None, keys, values, None, {})
                self.assertTrue(bool(torch.isfinite(sc).all()))
                self.assertFalse(bool(torch.isnan(sc).any()))

    def test_registered_names(self):
        for name in ("rarekv", "rare_kv"):
            self.assertIs(get_kv_compressor_class(name), RareKVSketch)

    def test_n_buckets(self):
        self.assertEqual(get_kv_compressor("rarekv", compression_ratio=0.5, n_planes=10).n_buckets,
                         1024)


class TestVectorised(unittest.TestCase):
    def test_hot_path_has_no_python_loops(self):
        """The scorer must stay a pure tensor program (GPU-friendly, no per-element work).

        Checked on the AST, not the source text: matching strings would trip over
        the word "for" in a comment and, worse, would miss a comprehension.
        """
        import ast
        import inspect
        import textwrap

        tree = ast.parse(textwrap.dedent(inspect.getsource(RareKVSketch.score)))
        loops = [n for n in ast.walk(tree)
                 if isinstance(n, (ast.For, ast.AsyncFor, ast.While, ast.ListComp,
                                   ast.SetComp, ast.DictComp, ast.GeneratorExp))]
        self.assertEqual(loops, [], "score() must contain no Python-level iteration")

    def test_hot_path_forces_no_host_sync(self):
        """A .item()/.tolist()/.cpu() mid-prefill drains the pipeline once per layer.

        This is exactly the cost snapkv pays at snapkv_sketch.py:190, and it shows
        up in the profiler as a prefill wall-vs-device gap.
        """
        import ast
        import inspect
        import textwrap

        tree = ast.parse(textwrap.dedent(inspect.getsource(RareKVSketch.score)))
        called = {n.func.attr for n in ast.walk(tree)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        self.assertEqual(called & {"item", "tolist", "cpu", "numpy", "nonzero", "unique"}, set())

    def test_scales_to_a_realistic_head_count(self):
        keys, values = _kv(t=512)
        sc = get_kv_compressor("rarekv", compression_ratio=0.9, n_planes=10, n_tables=60).score(
            _module(), None, keys, values, None, {})
        self.assertEqual(tuple(sc.shape), (B, H, 512))
        self.assertTrue(bool(torch.isfinite(sc).all()))


if __name__ == "__main__":
    unittest.main()
