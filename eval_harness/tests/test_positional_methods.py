"""Door-1 positional-method tests (Step 3).

YaRN and NTK are frequency transforms with closed-form references:

* **YaRN** is checked term-for-term against transformers' own
  ``ROPE_INIT_FUNCTIONS["yarn"]`` (inv_freq *and* attention_factor/mscale).
* **NTK** is checked against its closed-form base-scaling formula and identity.
* **Linear-PI** is a position remap (identity on inv_freq).

The interceptor itself is guarded for the identity case (installing the base
``PositionalMethod`` must not change a model's logits) on a tiny config-built
Llama (CPU, eager, random weights).
"""

from __future__ import annotations

import math
import unittest

import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from eval_harness.positional_methods import PositionalMethod, get_positional_method
from eval_harness.positional_methods.base import recover_base_and_dim
from eval_harness.positional_methods.linear_pi import LinearPIMethod
from eval_harness.positional_methods.ntk import NTKMethod
from eval_harness.positional_methods.yarn import YaRNMethod


def _native_inv_freq(base: float, head_dim: int) -> torch.Tensor:
    return 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))


def _build_tiny_model(head_dim: int = 16) -> LlamaForCausalLM:
    cfg = LlamaConfig(
        hidden_size=4 * head_dim,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=256,
        max_position_embeddings=8192,
        rope_theta=10000.0,
        attn_implementation="eager",
    )
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg).eval()


class TestRecoverBaseAndDim(unittest.TestCase):
    def test_round_trips_base_and_dim(self):
        for base, head_dim in [(10000.0, 16), (500000.0, 64), (1000.0, 8)]:
            inv = _native_inv_freq(base, head_dim)
            rec_base, rec_dim = recover_base_and_dim(inv)
            self.assertEqual(rec_dim, head_dim)
            self.assertAlmostEqual(rec_base, base, delta=base * 1e-4)


class TestYaRNAgainstReference(unittest.TestCase):
    CASES = [
        # (factor, original_max, head_dim, base)
        (8.0, 4096, 8, 10000.0),
        (4.0, 8192, 16, 10000.0),
        (16.0, 2048, 64, 500000.0),
        (2.0, 32768, 32, 10000.0),
    ]

    def _hf_reference(self, factor, original_max, head_dim, base):
        cfg = LlamaConfig(
            hidden_size=8 * head_dim,
            num_attention_heads=8,
            num_key_value_heads=2,
            max_position_embeddings=int(original_max * factor),
            rope_theta=base,
            rope_scaling={
                "rope_type": "yarn",
                "factor": factor,
                "original_max_position_embeddings": original_max,
            },
        )
        return ROPE_INIT_FUNCTIONS["yarn"](cfg, torch.device("cpu"))

    def test_inv_freq_and_mscale_match_hf(self):
        for factor, original_max, head_dim, base in self.CASES:
            with self.subTest(factor=factor, head_dim=head_dim, base=base):
                inv_ref, att_ref = self._hf_reference(factor, original_max, head_dim, base)
                method = YaRNMethod(
                    factor=factor, original_max_position_embeddings=original_max,
                )
                inv_mine = method.compute_inv_freq(
                    _native_inv_freq(base, head_dim), seq_len=8192,
                )
                self.assertEqual(inv_mine.shape, inv_ref.shape)
                self.assertTrue(
                    torch.allclose(inv_mine.float(), inv_ref.float(), atol=1e-6, rtol=1e-5),
                    f"inv_freq mismatch: {inv_mine.tolist()} vs {inv_ref.tolist()}",
                )
                self.assertAlmostEqual(method.mscale, att_ref, places=6)

    def test_factor_one_is_identity(self):
        native = _native_inv_freq(10000.0, 16)
        out = YaRNMethod(factor=1.0, original_max_position_embeddings=4096).compute_inv_freq(
            native, seq_len=4096,
        )
        self.assertTrue(torch.equal(out, native))
        self.assertEqual(YaRNMethod(factor=1.0).mscale, 1.0)


class TestNTK(unittest.TestCase):
    def test_factor_one_is_identity(self):
        native = _native_inv_freq(10000.0, 32)
        self.assertTrue(torch.equal(NTKMethod(factor=1.0).compute_inv_freq(native, 4096), native))

    def test_matches_closed_form_base_scaling(self):
        base, head_dim, factor = 10000.0, 32, 8.0
        native = _native_inv_freq(base, head_dim)
        out = NTKMethod(factor=factor).compute_inv_freq(native, seq_len=4096)

        new_base = base * (factor ** (head_dim / (head_dim - 2)))
        expected = 1.0 / (new_base ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim))
        self.assertTrue(torch.allclose(out.double(), expected, atol=1e-9, rtol=1e-7))

    def test_rejects_factor_below_one(self):
        with self.assertRaises(ValueError):
            NTKMethod(factor=0.5)


class TestLinearPI(unittest.TestCase):
    def test_remaps_positions_by_factor(self):
        pos = torch.arange(0, 100).unsqueeze(0)
        out = LinearPIMethod(factor=4.0).remap_position_ids(pos, seq_len=100)
        self.assertTrue(torch.allclose(out, pos / 4.0))

    def test_factor_one_is_identity(self):
        pos = torch.arange(0, 10).unsqueeze(0)
        self.assertTrue(torch.equal(LinearPIMethod(factor=1.0).remap_position_ids(pos, 10), pos))

    def test_inv_freq_untouched(self):
        native = _native_inv_freq(10000.0, 16)
        self.assertIs(LinearPIMethod(factor=8.0).compute_inv_freq(native, 4096), native)


class TestInterceptorIdentity(unittest.TestCase):
    def test_base_positional_method_is_a_noop(self):
        """Installing the identity PositionalMethod must not change logits."""
        model = _build_tiny_model()
        torch.manual_seed(1)
        ids = torch.randint(0, 256, (1, 32))
        pos = torch.arange(32).unsqueeze(0)

        with torch.no_grad():
            baseline = model(input_ids=ids, position_ids=pos).logits
            with PositionalMethod()(model):
                wrapped = model(input_ids=ids, position_ids=pos).logits

        self.assertTrue(torch.equal(baseline, wrapped))

    def test_linear_pi_interceptor_matches_manual_remap(self):
        """The interceptor's remap path equals running the model on positions
        divided by the factor directly."""
        model = _build_tiny_model()
        torch.manual_seed(2)
        ids = torch.randint(0, 256, (1, 48))
        pos = torch.arange(48).unsqueeze(0)

        with torch.no_grad():
            manual = model(input_ids=ids, position_ids=pos / 4.0).logits
            with LinearPIMethod(factor=4.0)(model):
                via_door = model(input_ids=ids, position_ids=pos).logits

        self.assertTrue(torch.allclose(manual, via_door, atol=1e-5, rtol=1e-4))

    def test_yarn_interceptor_runs_and_is_finite(self):
        model = _build_tiny_model()
        torch.manual_seed(3)
        ids = torch.randint(0, 256, (1, 40))
        pos = torch.arange(40).unsqueeze(0)
        method = get_positional_method(
            "yarn", factor=8.0, original_max_position_embeddings=2048,
        )
        with torch.no_grad(), method(model):
            out = model(input_ids=ids, position_ids=pos).logits
        self.assertTrue(torch.isfinite(out).all())


class TestInterceptorFrequencyCache(unittest.TestCase):
    """The interceptor caches the compute_inv_freq result so per-token decode
    forwards don't redo the (GPU-syncing) frequency rebuild on every call."""

    def _count_calls(self, method):
        real = method.compute_inv_freq
        seq_lens: list[int] = []

        def counting(original_inv_freq, seq_len):
            seq_lens.append(seq_len)
            return real(original_inv_freq, seq_len)

        method.compute_inv_freq = counting  # instance attr shadows the bound method
        return seq_lens

    def _run_forwards(self, model, method, lengths):
        with torch.no_grad(), method(model):
            for length in lengths:
                ids = torch.randint(0, 256, (1, length))
                pos = torch.arange(length).unsqueeze(0)
                model(input_ids=ids, position_ids=pos)

    def test_seq_len_agnostic_method_computes_inv_freq_once(self):
        # NTK ignores seq_len (inv_freq_depends_on_seq_len=False), so even across
        # forwards with different lengths the frequency is computed exactly once.
        method = get_positional_method("ntk", factor=8.0)
        calls = self._count_calls(method)
        self._run_forwards(_build_tiny_model(), method, (8, 16, 24))
        self.assertEqual(len(calls), 1)

    def test_seq_len_dependent_method_recomputes_per_distinct_length(self):
        # A method that opts into seq_len dependence keys the cache on seq_len:
        # recompute per distinct length, but a repeated length is still cached.
        method = get_positional_method("ntk", factor=8.0)
        method.inv_freq_depends_on_seq_len = True
        calls = self._count_calls(method)
        self._run_forwards(_build_tiny_model(), method, (8, 16, 16, 24))
        self.assertEqual(len(calls), 3)
        self.assertEqual(sorted(set(calls)), [8, 16, 24])


if __name__ == "__main__":
    unittest.main()
