"""Pins the Qwen3.5 gated-attention bug in the pre-RoPE query reprojection.

Qwen3.5's ``q_proj`` emits ``num_attention_heads * head_dim * 2`` features per
token: the model splits this per head into ``[query | gate]`` via
``torch.chunk(q_proj(x).view(*, -1, head_dim*2), 2, dim=-1)`` (see
``transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5Attention.forward``). The
KV-compression sketches that re-project the query from ``hidden_states``
(``snapkv``/``pyramidkv``, ``expected_attention``, ``compactor``) must slice off
the gate to recover the query the model actually used — otherwise the head
reshape sees twice the expected width and raises.

Non-gated families (Llama/Qwen3/Gemma3, ``q_proj`` width = heads*head_dim) must
be unaffected.
"""
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from eval_harness.kv_compression.compressors.snapkv_sketch import (
    _get_prerope_query_states as snapkv_q,
)
from eval_harness.kv_compression.compressors.expected_attention_sketch import (
    _get_prerope_query_states as ea_q,
)
from eval_harness.kv_compression.compressors.compactor_sketch import (
    _get_prerope_query_states as compactor_q,
)

HELPERS = [("snapkv", snapkv_q), ("expected_attention", ea_q), ("compactor", compactor_q)]


def _make_module(num_heads, head_dim, hidden, gated, q_norm=None):
    mult = 2 if gated else 1
    q_proj = nn.Linear(hidden, num_heads * head_dim * mult, bias=False)
    torch.manual_seed(0)
    with torch.no_grad():
        q_proj.weight.normal_()
    return SimpleNamespace(
        config=SimpleNamespace(num_attention_heads=num_heads),
        head_dim=head_dim,
        q_proj=q_proj,
        q_norm=q_norm,
    )


def _reference_gated_query(module, hidden_states, num_heads, head_dim):
    """The exact extraction Qwen3_5Attention.forward performs."""
    B, S, _ = hidden_states.shape
    q, _gate = torch.chunk(
        module.q_proj(hidden_states).view(B, S, -1, head_dim * 2), 2, dim=-1
    )
    q = q.reshape(B, S, num_heads, head_dim).transpose(1, 2)
    if module.q_norm is not None:
        q = module.q_norm(q)
    return q


def _reference_plain_query(module, hidden_states, num_heads, head_dim):
    B, S, _ = hidden_states.shape
    q = module.q_proj(hidden_states).view(B, S, num_heads, head_dim).transpose(1, 2)
    if module.q_norm is not None:
        q = module.q_norm(q)
    return q


class TestQwen35GatedQProj(unittest.TestCase):
    def setUp(self):
        self.num_heads, self.head_dim, self.hidden = 4, 8, 32
        torch.manual_seed(1)
        self.hs = torch.randn(1, 7, self.hidden)

    def test_gated_extraction_matches_model(self):
        """Gated q_proj: helper slices off the gate and matches the model's query."""
        for name, helper in HELPERS:
            with self.subTest(method=name):
                mod = _make_module(self.num_heads, self.head_dim, self.hidden, gated=True)
                got = helper(mod, self.hs)
                self.assertEqual(tuple(got.shape), (1, self.num_heads, 7, self.head_dim))
                ref = _reference_gated_query(mod, self.hs, self.num_heads, self.head_dim)
                self.assertTrue(torch.allclose(got, ref, atol=1e-6),
                                f"{name}: gated query mismatch (max diff {(got-ref).abs().max()})")

    def test_gated_with_qnorm(self):
        """q_norm (RMSNorm over head_dim) is applied to the de-gated query."""
        qn = nn.RMSNorm(self.head_dim)
        with torch.no_grad():
            qn.weight.normal_()
        for name, helper in HELPERS:
            with self.subTest(method=name):
                mod = _make_module(self.num_heads, self.head_dim, self.hidden, gated=True, q_norm=qn)
                got = helper(mod, self.hs)
                ref = _reference_gated_query(mod, self.hs, self.num_heads, self.head_dim)
                self.assertTrue(torch.allclose(got, ref, atol=1e-5),
                                f"{name}: gated+qnorm mismatch")

    def test_plain_qproj_unaffected(self):
        """Non-gated q_proj (Llama-style) still extracts the query correctly."""
        for name, helper in HELPERS:
            with self.subTest(method=name):
                mod = _make_module(self.num_heads, self.head_dim, self.hidden, gated=False)
                got = helper(mod, self.hs)
                self.assertEqual(tuple(got.shape), (1, self.num_heads, 7, self.head_dim))
                ref = _reference_plain_query(mod, self.hs, self.num_heads, self.head_dim)
                self.assertTrue(torch.allclose(got, ref, atol=1e-6), f"{name}: plain query mismatch")


if __name__ == "__main__":
    unittest.main()
