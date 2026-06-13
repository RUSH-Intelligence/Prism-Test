"""Parity: Door-2 DCAMethod == legacy prefill_methods DCAMethod (Step 4).

The attention math was ported verbatim; only the install path changed (the new
one subclasses AttentionMethod, which owns the forward-replacement + phase
gating).  Driven through the same tiny config-built Llama, the two must produce
identical post-prefill caches and identical question-pass logits.
"""

from __future__ import annotations

import unittest

import torch
from transformers import DynamicCache, LlamaConfig, LlamaForCausalLM

from eval_harness.attention_methods.dca import DCAMethod as NewDCA
from eval_harness.attention_methods.base import AttentionPhase
from eval_harness.prefill_methods.dca import DCAMethod as LegacyDCA


def _build_model(num_hidden_layers: int = 2) -> LlamaForCausalLM:
    cfg = LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=256,
        max_position_embeddings=8192,
        rope_theta=10000.0,
        attn_implementation="eager",
    )
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg).eval()


_DCA_KW = dict(chunk_size=96, local_window=16, pretraining_length=128, use_flash_attn="off")


def _run(model, method, context_ids, question_ids):
    cache = DynamicCache()
    with torch.no_grad(), method(model):
        model.model(input_ids=context_ids, past_key_values=cache)
        ctx_len = context_ids.shape[1]
        position_ids = torch.arange(
            ctx_len, ctx_len + question_ids.shape[1],
        ).unsqueeze(0)
        out = model(
            input_ids=question_ids,
            past_key_values=cache,
            position_ids=position_ids,
        )
    keys = [layer.keys.clone() for layer in cache.layers]
    return out.logits, keys


class TestDCAParity(unittest.TestCase):
    def test_new_dca_matches_legacy(self):
        model = _build_model()
        torch.manual_seed(1)
        ctx = torch.randint(0, 256, (1, 200))
        q = torch.randint(0, 256, (1, 8))

        legacy_logits, legacy_keys = _run(model, LegacyDCA(**_DCA_KW), ctx, q)
        new_logits, new_keys = _run(model, NewDCA(**_DCA_KW), ctx, q)

        self.assertTrue(
            torch.equal(legacy_logits, new_logits),
            f"logits differ, max|Δ|={(legacy_logits - new_logits).abs().max().item():.2e}",
        )
        self.assertEqual(len(legacy_keys), len(new_keys))
        for lk, nk in zip(legacy_keys, new_keys):
            self.assertTrue(torch.equal(lk, nk))

    def test_phase_defaults_to_both(self):
        self.assertIs(NewDCA(**_DCA_KW).phase, AttentionPhase.BOTH)

    def test_missing_inv_freq_runs_as_noop(self):
        """setup() returns False when inv_freq is absent → forwards untouched."""
        model = _build_model()
        method = NewDCA(**_DCA_KW)
        # A model whose rotary exposes no inv_freq: patch the helper's source.
        import eval_harness.attention_methods.dca as dca_mod

        original = dca_mod.get_inv_freq
        dca_mod.get_inv_freq = lambda _m: None
        try:
            attn = model.model.layers[0].self_attn
            class_forward = type(attn).forward  # unbound, identity-stable
            with method(model):
                # No-op install leaves the original bound method in place (its
                # __func__ is the class forward), rather than our closure.
                self.assertIs(getattr(attn.forward, "__func__", None), class_forward)
        finally:
            dca_mod.get_inv_freq = original


if __name__ == "__main__":
    unittest.main()
