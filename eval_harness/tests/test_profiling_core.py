"""KV-byte accounting, exact-length prompts, artifact schema and the audit gate.

Weight-free: fake caches and a word-level fake tokenizer, per repo convention.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from eval_harness.profiling.audit import (
    achieved_bandwidth_gbps, audit_cell, audit_group, expected_budget, roofline_step_ms,
)
from eval_harness.profiling.cell import PerfCell, _jsonable, newest_perf, write_perf
from eval_harness.profiling.kvsize import analytic_kv_bytes, kv_cache_accounting
from eval_harness.profiling.prompts import build_exact_prompt


# ----------------------------------------------------------------- fakes ----
class _AttnLayer:
    def __init__(self, seq, heads=2, dim=8, dtype=torch.bfloat16):
        self.keys = torch.zeros(1, heads, seq, dim, dtype=dtype)
        self.values = torch.zeros(1, heads, seq, dim, dtype=dtype)


class _Cache:
    def __init__(self, layers):
        self.layers = layers

    def __len__(self):
        return len(self.layers)


class _FakeTokenizer:
    """One id per whitespace token; words starting with 'z' cost TWO ids.

    The variable-length branch is what forces a real trim rather than
    "emit N words and hope".
    """

    model_max_length = 10 ** 9

    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        ids = []
        for w in text.split():
            ids.extend([7, 7] if w.startswith("z") else [7])
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join("w" for _ in ids)


# ------------------------------------------------------------- kv bytes ----
class TestKVAccounting(unittest.TestCase):
    def test_uniform_exact_bytes(self):
        # 4 layers x 2 tensors x (1*2*100*8 elems) x 2 B = 25600
        acc = kv_cache_accounting(_Cache([_AttnLayer(100) for _ in range(4)]))
        self.assertEqual(acc.bytes_total, 25600)
        self.assertEqual(acc.layers_with_kv, 4)
        self.assertFalse(acc.ragged)
        self.assertEqual(acc.bytes_per_token, 2 * 2 * 8 * 2)

    def test_element_size_is_read_not_assumed(self):
        f32 = kv_cache_accounting(_Cache([_AttnLayer(100, dtype=torch.float32)]))
        bf16 = kv_cache_accounting(_Cache([_AttnLayer(100, dtype=torch.bfloat16)]))
        self.assertEqual(f32.bytes_total, 2 * bf16.bytes_total)

    def test_hybrid_none_slots_are_skipped(self):
        """NemotronH's mamba/mlp slots expose keys/values as None."""
        cache = _Cache([_AttnLayer(100), SimpleNamespace(keys=None, values=None), _AttnLayer(100)])
        acc = kv_cache_accounting(cache)
        self.assertEqual(acc.layers_total, 3)
        self.assertEqual(acc.layers_with_kv, 2)
        self.assertEqual(acc.bytes_total, 2 * 64 * 100)
        self.assertEqual(acc.per_layer_seq_len, [100, 100])

    def test_ragged_sums_actual_lengths(self):
        """PyramidKV keeps different amounts per layer; seq_max*n_layers overstates."""
        acc = kv_cache_accounting(_Cache([_AttnLayer(100), _AttnLayer(60), _AttnLayer(20)]))
        # per layer: 2 (k+v) * 2 heads * s * 8 dim * 2 B = 64*s bytes
        naive = 3 * (64 * 100)
        self.assertEqual(acc.per_layer_seq_len, [100, 60, 20])
        self.assertTrue(acc.ragged)
        self.assertLess(acc.bytes_total, naive)
        self.assertEqual(acc.bytes_total, sum(64 * s for s in (100, 60, 20)))

    def test_empty_cache_is_safe(self):
        acc = kv_cache_accounting(_Cache([]))
        self.assertEqual(acc.bytes_total, 0)
        self.assertIsNone(acc.bytes_per_token)

    def test_analytic_matches_measured(self):
        acc = kv_cache_accounting(_Cache([_AttnLayer(100) for _ in range(4)]))
        self.assertEqual(acc.bytes_total, analytic_kv_bytes(100, 4, 2, 8, 2))


# --------------------------------------------------------------- prompts ----
class TestExactPrompt(unittest.TestCase):
    def test_exact_token_count(self):
        tok = _FakeTokenizer()
        for n in (1, 2, 63, 1024, 12347):
            with self.subTest(n=n):
                _, ids = build_exact_prompt(tok, n)
                self.assertEqual(ids.shape[1], n)

    def test_overflow_raises_instead_of_truncating(self):
        """A cell head-truncated to fit is a wrong number, not a noisy one."""
        tok = _FakeTokenizer()
        with self.assertRaises(ValueError) as cm:
            build_exact_prompt(tok, 131072, max_model_len=120832, reserve=136)
        self.assertIn("120832", str(cm.exception))

    def test_reserve_is_accounted(self):
        tok = _FakeTokenizer()
        build_exact_prompt(tok, 900, max_model_len=1024, reserve=100)   # 1000 <= 1024, fine
        with self.assertRaises(ValueError):
            build_exact_prompt(tok, 900, max_model_len=1024, reserve=200)

    def test_deterministic(self):
        tok = _FakeTokenizer()
        a, _ = build_exact_prompt(tok, 200, seed=42)
        b, _ = build_exact_prompt(tok, 200, seed=42)
        self.assertEqual(a, b)


# ---------------------------------------------------------------- schema ----
class TestSchema(unittest.TestCase):
    def _cell(self, method="knorm", ratio=0.9):
        return PerfCell(model_key="m", hf_model="org/m", method=method, compression_ratio=ratio,
                        context_tokens=32768, attn_impl="sdpa", dtype="bfloat16")

    def test_anchor_key_pairs_cells(self):
        c, a = self._cell(), self._cell(method="none", ratio=0.0)
        self.assertEqual(c.anchor_key, a.anchor_key)
        self.assertTrue(a.is_anchor)
        self.assertFalse(c.is_anchor)
        self.assertEqual(a.cell_id, "m/ctx32768/full")
        self.assertEqual(c.cell_id, "m/ctx32768/knorm_r0.9")

    def test_attn_and_dtype_are_part_of_the_anchor_key(self):
        """A flash_attention_2 cell must not borrow an sdpa anchor."""
        base = self._cell()
        other = PerfCell(model_key="m", hf_model="org/m", method="knorm", compression_ratio=0.9,
                         context_tokens=32768, attn_impl="flash_attention_2", dtype="bfloat16")
        self.assertNotEqual(base.anchor_key, other.anchor_key)

    def test_jsonable_coerces_torch_types(self):
        """torch.dtype / Size / 0-dim tensors would make the artifact unreadable."""
        out = _jsonable({"d": torch.bfloat16, "s": torch.Size([1, 2]), "n": torch.tensor(3.5)})
        json.dumps(out)
        self.assertEqual(out["s"], [1, 2])
        self.assertAlmostEqual(out["n"], 3.5)

    def test_write_perf_nests_reruns(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "cell"
            p1 = write_perf(root, {"a": 1})
            p2 = write_perf(root, {"a": 2})
            self.assertEqual(p1, root / "perf.json")
            self.assertEqual(p2, root / "1" / "perf.json")
            os.utime(p2, (10 ** 9, 10 ** 9))
            os.utime(p1, (10 ** 9 - 100, 10 ** 9 - 100))
            self.assertEqual(newest_perf(root), p2)

    def test_newest_perf_missing(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(newest_perf(Path(d)))


# ----------------------------------------------------------------- audit ----
class TestAudit(unittest.TestCase):
    def test_expected_budget_floors_with_float_error(self):
        # 1 - 0.9 == 0.09999999999999998, so 8192 -> 819 not 820.
        self.assertEqual(expected_budget(8192, 0.9), 819)
        self.assertEqual(expected_budget(131072, 0.9), 13107)
        self.assertEqual(expected_budget(192, 0.5), 96)
        self.assertEqual(expected_budget(101, 0.5), 50)
        self.assertEqual(expected_budget(512, 0.0), 512)

    def _payload(self, seq, ctx=1024, samples=254, cv=0.02, median=10.0,
                 p90=10.2, p99=14.0):
        return {"prefill": {"tokens": ctx},
                "kv_cache": {"seq_len_max": seq, "seq_len_min": seq, "ragged": False,
                             "per_layer_seq_len": [seq, seq]},
                "decode": {"n_samples": samples,
                           "per_step": {"median": median, "p90": p90, "p99": p99, "cv": cv}},
                "memory": {"num_alloc_retries": 0}}

    def _cell(self, method="knorm", ratio=0.9, ctx=1024):
        return PerfCell(model_key="m", hf_model="x", method=method, compression_ratio=ratio,
                        context_tokens=ctx, attn_impl="sdpa", dtype="bfloat16",
                        decode_steps=128, repeats=2)

    def test_clean_cell_has_no_problems(self):
        r = audit_cell(self._cell(), self._payload(expected_budget(1024, 0.9)))
        self.assertEqual(r["problems"], [])

    def test_budget_mismatch_is_a_problem(self):
        r = audit_cell(self._cell(), self._payload(500))
        self.assertTrue(any("budget" in p for p in r["problems"]))

    def test_masking_press_detector(self):
        """A press that keeps the cache full-length pays FULL KV traffic."""
        r = audit_cell(self._cell(), self._payload(1024))
        self.assertTrue(any("masking-based press" in p for p in r["problems"]))

    def test_anchor_must_not_be_evicted(self):
        r = audit_cell(self._cell(method="none", ratio=0.0), self._payload(900))
        self.assertTrue(any("anchor cache was evicted" in p for p in r["problems"]))

    def test_wrong_step_count_is_a_problem(self):
        r = audit_cell(self._cell(), self._payload(expected_budget(1024, 0.9), samples=100))
        # 128 forwards x 2 repeats -> (128-1)*2 = 254 token-to-token periods
        self.assertTrue(any("expected 254" in p for p in r["problems"]))

    def test_high_cv_fails_and_jitter_warns(self):
        r = audit_cell(self._cell(), self._payload(expected_budget(1024, 0.9), cv=0.2))
        self.assertTrue(any("CV" in p for p in r["problems"]))
        # Jitter is gated on p90, not p99: each repeat's first decode step pays
        # allocator growth, so a heavy p99 is structural on a healthy cell.
        r2 = audit_cell(self._cell(), self._payload(expected_budget(1024, 0.9), p90=20.0))
        self.assertTrue(any("jitter" in w for w in r2["warnings"]))

    def test_heavy_p99_alone_does_not_warn(self):
        """A few slow first-steps per repeat must not flag every healthy cell."""
        r = audit_cell(self._cell(), self._payload(expected_budget(1024, 0.9),
                                                   p90=10.1, p99=34.0))
        self.assertEqual(r["warnings"], [])

    def test_alloc_retry_is_a_problem(self):
        pl = self._payload(expected_budget(1024, 0.9))
        pl["memory"]["num_alloc_retries"] = 3
        self.assertTrue(any("alloc_retries" in p for p in audit_cell(self._cell(), pl)["problems"]))

    def test_group_requires_an_anchor(self):
        c = self._cell()
        probs = audit_group([(c, {"environment": {}})])
        self.assertTrue(any("no full-KV anchor" in p for p in probs))

    def test_group_flags_mixed_environments(self):
        c, a = self._cell(), self._cell(method="none", ratio=0.0)
        probs = audit_group([(a, {"environment": {"gpu_name": "H200"}}),
                             (c, {"environment": {"gpu_name": "A100"}})])
        self.assertTrue(any("not mutually comparable" in p for p in probs))


class TestRoofline(unittest.TestCase):
    W, KVT = 15.01e9, 131072       # Llama-3.1-8B bf16: weight traffic, KV bytes/token

    def test_dyncache_is_slower_than_ideal(self):
        """DynamicCache re-cats every step (cache_utils.py:143-144) -> ~3x KV traffic."""
        ideal = roofline_step_ms(self.W, self.KVT, 131072, dyncache=False)
        dync = roofline_step_ms(self.W, self.KVT, 131072, dyncache=True)
        self.assertAlmostEqual(ideal, 6.71, places=1)
        self.assertAlmostEqual(dync, 13.86, places=1)
        self.assertGreater(dync, ideal)

    def test_weights_dominate_at_short_context(self):
        """At batch=1 the 15 GB weight read caps the achievable speedup at 8K."""
        full = roofline_step_ms(self.W, self.KVT, 8192, dyncache=False)
        comp = roofline_step_ms(self.W, self.KVT, 819, dyncache=False)
        self.assertLess(full / comp, 1.10)

    def test_achieved_bandwidth_roundtrips(self):
        ms = roofline_step_ms(self.W, self.KVT, 65536)
        bw = achieved_bandwidth_gbps(self.W, self.KVT, 65536, ms)
        self.assertAlmostEqual(bw, 4800.0, places=0)


if __name__ == "__main__":
    unittest.main()
