"""Pins the Qwen3.5 gated-q_proj de-gate in ``RidgeSketch._get_all_queries``.

Qwen3.5's gated attention makes ``q_proj`` emit ``num_attention_heads * head_dim
* 2`` features ([query | gate] fused per head). RidgeSketch reshapes the query
by ``H_q = q_proj_width // head_dim``; without de-gating, the gate channels are
counted as extra heads (``H_q = 2 * num_heads``) and silently pollute the
``omega = ||Q k||`` query gram (no crash, just wrong scores). The de-gate slices
off the gate to recover the query the model uses; it is a no-op on non-gated
models (Llama/Mistral), where ``q_proj`` width == ``num_heads * head_dim``.
"""
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from eval_harness.kv_compression.compressors.ridge_sketch import RidgeSketch


def _module(num_heads, head_dim, hidden, gated):
    mult = 2 if gated else 1
    q_proj = nn.Linear(hidden, num_heads * head_dim * mult, bias=False)
    torch.manual_seed(0)
    with torch.no_grad():
        q_proj.weight.normal_()
    return SimpleNamespace(
        config=SimpleNamespace(num_attention_heads=num_heads),
        head_dim=head_dim,
        q_proj=q_proj,
    )


class TestRidgeDegate(unittest.TestCase):
    def setUp(self):
        self.B, self.H_kv, self.T, self.D, self.H_q = 1, 2, 7, 8, 4  # GQA: 4 q heads -> 2 kv heads
        torch.manual_seed(1)
        self.hs = torch.randn(self.B, self.T, 32)
        self.keys = torch.randn(self.B, self.H_kv, self.T, self.D)
        self.ridge = object.__new__(RidgeSketch)  # _get_all_queries uses no dataclass fields

    def test_gated_degate_matches_model(self):
        """Gated q_proj: de-gated queries == the model's chunk(...)[0], GQA-averaged to kv heads."""
        mod = _module(self.H_q, self.D, 32, gated=True)
        got = RidgeSketch._get_all_queries(self.ridge, mod, self.hs, self.keys)
        # Reference: take the query half (model's torch.chunk(view(...,2D),2)[0]),
        # then the same view/transpose/GQA-mean the method performs.
        qfull = mod.q_proj(self.hs)
        ref = qfull.view(self.B, self.T, self.H_q, 2 * self.D)[..., : self.D]
        ref = ref.reshape(self.B, self.T, self.H_q * self.D).view(self.B, self.T, self.H_q, self.D).transpose(1, 2)
        ref = ref.view(self.B, self.H_kv, self.H_q // self.H_kv, self.T, self.D).mean(dim=2)
        self.assertEqual(tuple(got.shape), (self.B, self.H_kv, self.T, self.D))
        self.assertTrue(torch.allclose(got, ref, atol=1e-6), f"max diff {(got - ref).abs().max()}")

    def test_nongated_unaffected(self):
        """Non-gated q_proj (Llama/Mistral): de-gate branch must not fire; shape unchanged."""
        mod = _module(self.H_q, self.D, 32, gated=False)
        got = RidgeSketch._get_all_queries(self.ridge, mod, self.hs, self.keys)
        self.assertEqual(tuple(got.shape), (self.B, self.H_kv, self.T, self.D))
        self.assertEqual(mod.q_proj(self.hs).shape[-1], self.H_q * self.D)  # width == heads*head_dim, not *2


if __name__ == "__main__":
    unittest.main()
