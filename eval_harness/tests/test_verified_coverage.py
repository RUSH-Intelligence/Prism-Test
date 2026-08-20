"""Tests for the verified-compression output-error CHECK (v2, Step 1).

Pure measurement — no model loading. A fake attention module with an identity
``q_proj`` lets us control the probe queries exactly, so we can assert the
*concept*, not just the plumbing:

* keeping everything → ~0 error (mechanics),
* redundant values → ~0 error even when keeping few (blend/redundancy), and
* dropping a token the query is loud about → higher error than keeping it.
"""

from __future__ import annotations

import unittest

import torch
from torch import nn

from eval_harness.kv_compression.compressors.verified_coverage import (
    measure_output_error,
)
from eval_harness.kv_compression.compressors.verified_sketch import VerifiedSketch


class _IdentityQProjModule(nn.Module):
    """One kv head, one query head, q_proj = identity → query == hidden."""

    def __init__(self, head_dim=8, layer_idx=0):
        super().__init__()
        self.num_key_value_heads = 1
        self.head_dim = head_dim
        self.layer_idx = layer_idx
        self.q_proj = nn.Linear(head_dim, head_dim, bias=False)
        with torch.no_grad():
            self.q_proj.weight.copy_(torch.eye(head_dim))


def _all_idx(B, H, T):
    return torch.arange(T).view(1, 1, T).expand(B, H, T).contiguous()


class TestMechanics(unittest.TestCase):
    def test_keep_all_is_zero_error(self):
        torch.manual_seed(0)
        D, T = 8, 32
        module = _IdentityQProjModule(head_dim=D)
        keys, values = torch.randn(1, 1, T, D), torch.randn(1, 1, T, D)
        hidden = torch.randn(1, T, D)
        rep = measure_output_error(
            module, hidden, keys, values, _all_idx(1, 1, T), {}, rotate=False
        )
        self.assertIsNotNone(rep)
        self.assertLess(rep.worst, 1e-4)

    def test_returns_none_without_qproj(self):
        # No q_proj → queries cannot be built → graceful None (caller skips).
        mod = nn.Module()
        mod.num_key_value_heads, mod.head_dim, mod.layer_idx = 1, 8, 0
        keys = values = torch.randn(1, 1, 16, 8)
        rep = measure_output_error(
            mod, torch.randn(1, 16, 8), keys, values, _all_idx(1, 1, 16), {}, rotate=False
        )
        self.assertIsNone(rep)


class TestConcept(unittest.TestCase):
    def test_redundant_values_low_error_even_keeping_few(self):
        # All value vectors identical → the blend is that vector regardless of
        # weights, so ANY nonempty visible keep-set reproduces the output.
        # Output-error sees this redundancy; a per-token score cannot.
        torch.manual_seed(1)
        D, T = 8, 40
        module = _IdentityQProjModule(head_dim=D)
        keys = torch.randn(1, 1, T, D)
        values = torch.ones(1, 1, T, D) * 3.0  # identical values
        hidden = torch.randn(1, T, D)
        # Keep only 4 early tokens (all visible to the trailing probes).
        keep = torch.tensor([0, 1, 2, 3]).view(1, 1, 4)
        rep = measure_output_error(
            module, hidden, keys, values, keep, {}, n_probe=4, rotate=False
        )
        self.assertIsNotNone(rep)
        self.assertLess(rep.worst, 1e-4)

    def test_dropping_dominant_token_hurts_more_than_keeping_it(self):
        # Construct a token the probe query is loud about AND that carries a big
        # value. Keeping it ≈ reproduces the output; dropping it wrecks it.
        D, T = 8, 32
        module = _IdentityQProjModule(head_dim=D)
        e0 = torch.zeros(D)
        e0[0] = 1.0

        keys = torch.randn(1, 1, T, D) * 0.1
        values = torch.randn(1, 1, T, D) * 0.1
        # Dominant token at position 0: key aligned with e0 (huge logit), big value.
        keys[0, 0, 0] = e0 * 8.0
        values[0, 0, 0] = torch.ones(D) * 20.0

        # Probe queries (== hidden) point strongly along e0 → attend to token 0.
        hidden = torch.randn(1, T, D) * 0.1
        hidden[0, -4:] = e0 * 8.0

        M = 8
        keep_with = torch.arange(M).view(1, 1, M)                 # includes token 0
        keep_without = torch.arange(1, M + 1).view(1, 1, M)       # excludes token 0

        err_with = measure_output_error(
            module, hidden, keys, values, keep_with, {}, n_probe=4, rotate=False
        ).worst
        err_without = measure_output_error(
            module, hidden, keys, values, keep_without, {}, n_probe=4, rotate=False
        ).worst
        self.assertLess(err_with, err_without)
        self.assertLess(err_with, 0.05)     # keeping it ≈ reproduces
        self.assertGreater(err_without, 0.5)  # dropping it is very wrong


class TestVerifiedSketchLogging(unittest.TestCase):
    def test_measure_flag_logs_and_preserves_output(self):
        torch.manual_seed(2)
        D, T = 8, 128
        module = _IdentityQProjModule(head_dim=D)
        keys, values = torch.randn(1, 1, T, D), torch.randn(1, 1, T, D)
        hidden = torch.randn(1, T, D)

        base = VerifiedSketch(inner="knorm", compression_ratio=0.5, sample_seed=0)
        meas = VerifiedSketch(
            inner="knorm", compression_ratio=0.5, sample_seed=0, measure_coverage=True
        )
        k0, v0 = base.compress(module, hidden, keys, values, None, {})
        with self.assertLogs(
            "eval_harness.kv_compression.compressors.verified_sketch", level="INFO"
        ) as cm:
            k1, v1 = meas.compress(module, hidden, keys, values, None, {})

        # Measurement changes nothing about what is kept.
        self.assertTrue(torch.equal(k0, k1))
        self.assertTrue(torch.equal(v0, v1))
        self.assertTrue(any("coverage" in line for line in cm.output))

    def test_drain_aggregates_records(self):
        torch.manual_seed(3)
        D, T = 8, 128
        module = _IdentityQProjModule(head_dim=D)
        keys, values = torch.randn(1, 1, T, D), torch.randn(1, 1, T, D)
        hidden = torch.randn(1, T, D)

        off = VerifiedSketch(inner="knorm", compression_ratio=0.5)
        self.assertIsNone(off.drain_coverage())  # nothing measured → None

        meas = VerifiedSketch(
            inner="knorm", compression_ratio=0.5, sample_seed=0, measure_coverage=True
        )
        # Two compress() calls (two "prompts") on the same single layer.
        meas.compress(module, hidden, keys, values, None, {})
        meas.compress(module, hidden, keys, values, None, {})
        summary = meas.drain_coverage()
        self.assertIsNotNone(summary)
        self.assertEqual(summary["n_records"], 2)
        self.assertEqual(summary["n_layers"], 1)
        self.assertIn("0", summary["per_layer"])
        self.assertEqual(summary["per_layer"]["0"]["n"], 2)
        self.assertIn("worst_rel_err_mean", summary)

    def test_per_prompt_tagging_keys_by_df_row(self):
        torch.manual_seed(5)
        D, T = 8, 128
        module = _IdentityQProjModule(head_dim=D)
        keys, values = torch.randn(1, 1, T, D), torch.randn(1, 1, T, D)
        hidden = torch.randn(1, T, D)
        meas = VerifiedSketch(
            inner="knorm", compression_ratio=0.5, sample_seed=0, measure_coverage=True
        )
        # Question at df row 7, then question at df row 8 (each its own context).
        meas.begin_prompt_group([7])
        meas.compress(module, hidden, keys, values, None, {})
        meas.begin_prompt_group([8])
        meas.compress(module, hidden, keys, values, None, {})

        summary = meas.drain_coverage()
        self.assertIn("per_prompt", summary)
        self.assertEqual(set(summary["per_prompt"].keys()), {"7", "8"})
        self.assertIn("worst_rel_err", summary["per_prompt"]["7"])

    def test_shared_context_group_shares_one_number(self):
        # A group of two questions sharing one context → both rows get the same
        # coverage (compression fired once on the shared cache).
        torch.manual_seed(6)
        D, T = 8, 128
        module = _IdentityQProjModule(head_dim=D)
        keys, values = torch.randn(1, 1, T, D), torch.randn(1, 1, T, D)
        hidden = torch.randn(1, T, D)
        meas = VerifiedSketch(
            inner="knorm", compression_ratio=0.5, sample_seed=0, measure_coverage=True
        )
        meas.begin_prompt_group([3, 4])
        meas.compress(module, hidden, keys, values, None, {})
        pp = meas.drain_coverage()["per_prompt"]
        self.assertEqual(set(pp.keys()), {"3", "4"})
        self.assertEqual(pp["3"]["worst_rel_err"], pp["4"]["worst_rel_err"])


if __name__ == "__main__":
    unittest.main()
