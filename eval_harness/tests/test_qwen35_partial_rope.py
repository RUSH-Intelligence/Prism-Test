"""Pins the Qwen3.5 partial-rotary handling in the KV-compression sketches.

Qwen3.5 uses PARTIAL rotary: only the first ``rotary_dim = head_dim *
partial_rotary_factor`` (= 256 * 0.25 = 64) channels of each head are rotated;
the remaining channels pass through unchanged (see
``Qwen3_5Attention``/``apply_rotary_pos_emb`` in ``modeling_qwen3_5``). The
sketches that re-apply RoPE to a re-projected query (``snapkv``/``pyramidkv``
via ``compute_window_attention``, ``compactor`` via ``_non_causal_scores``) and
the matrix-form averaged RoPE in ``expected_attention`` must rotate only the
rotary block, or they broadcast head_dim (256) against cos/sin (64) and crash.

These tests validate the rotation math against the model's own reference
``apply_rotary_pos_emb``; the integration is covered by the GPU smoke runs.
"""
import unittest

import torch
from torch import nn

try:
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        apply_rotary_pos_emb as qwen35_apply_rope,
        rotate_half as qwen35_rotate_half,
    )
    HAS_QWEN35 = True
except Exception:  # pragma: no cover - depends on installed transformers
    HAS_QWEN35 = False

from transformers.models.llama.modeling_llama import rotate_half


def _partial_rope_query(q, cos, sin):
    """The transformation inlined into snapkv/compactor (query side only)."""
    rotary_dim = cos.shape[-1]
    cos_u, sin_u = cos.unsqueeze(1), sin.unsqueeze(1)
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    q_rot = (q_rot * cos_u) + (rotate_half(q_rot) * sin_u)
    return torch.cat([q_rot, q_pass], dim=-1)


def _avg_rope_matrix(cos, sin, head_dim):
    """The block-diagonal rotation matrix built in expected_attention.apply_avg_rope."""
    rotary_dim = cos.shape[-1]
    half = rotary_dim // 2
    Id = torch.eye(rotary_dim, dtype=cos.dtype)
    P = torch.zeros((rotary_dim, rotary_dim), dtype=cos.dtype)
    P[half:, :half] = torch.eye(half, dtype=cos.dtype)
    P[:half, half:] = -torch.eye(half, dtype=cos.dtype)
    R_rot = (cos.unsqueeze(1) * Id + sin.unsqueeze(1) * P).mean(dim=0)
    R = torch.eye(head_dim, dtype=cos.dtype)
    R[:rotary_dim, :rotary_dim] = R_rot
    return R


@unittest.skipUnless(HAS_QWEN35, "transformers build lacks qwen3_5 modeling")
class TestPartialRopeAgainstModel(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.B, self.H, self.S = 1, 4, 6
        self.head_dim, self.rotary_dim = 16, 4  # rotary_dim < head_dim (partial)
        self.q = torch.randn(self.B, self.H, self.S, self.head_dim, dtype=torch.float64)
        # cos/sin as the model produces them: (B, S, rotary_dim), duplicated halves
        ang = torch.randn(self.B, self.S, self.rotary_dim // 2, dtype=torch.float64)
        emb = torch.cat([ang, ang], dim=-1)
        self.cos, self.sin = emb.cos(), emb.sin()

    def test_query_partial_rope_matches_model(self):
        """snapkv/compactor query rotation == model.apply_rotary_pos_emb (q side)."""
        ref_q, _ = qwen35_apply_rope(self.q, self.q.clone(), self.cos, self.sin)
        got_q = _partial_rope_query(self.q, self.cos, self.sin)
        self.assertEqual(got_q.shape, self.q.shape)
        self.assertTrue(torch.allclose(got_q, ref_q, atol=1e-10),
                        f"max diff {(got_q-ref_q).abs().max()}")

    def test_passthrough_channels_unchanged(self):
        """Channels beyond rotary_dim are left identical (not rotated)."""
        got_q = _partial_rope_query(self.q, self.cos, self.sin)
        self.assertTrue(torch.equal(got_q[..., self.rotary_dim:], self.q[..., self.rotary_dim:]))

    def test_qwen35_rotate_half_equiv_llama(self):
        """The model's rotate_half matches the llama rotate_half used in the sketches."""
        x = torch.randn(2, 3, self.rotary_dim, dtype=torch.float64)
        self.assertTrue(torch.equal(qwen35_rotate_half(x), rotate_half(x)))

    def test_avg_rope_matrix_equals_elementwise(self):
        """expected_attention's block-diagonal R reproduces elementwise partial RoPE.

        With a single future position the averaged rotation is exactly that
        position's rotation, so R @ q must equal the elementwise partial RoPE.
        """
        cos1, sin1 = self.cos[0, :1], self.sin[0, :1]  # (1, rotary_dim)
        R = _avg_rope_matrix(cos1, sin1, self.head_dim)
        vec = torch.randn(self.head_dim, dtype=torch.float64)
        got = R @ vec
        # elementwise reference on the rotary block + passthrough
        v = vec.view(1, 1, 1, self.head_dim)
        ref = _partial_rope_query(v, cos1.unsqueeze(0), sin1.unsqueeze(0)).view(self.head_dim)
        self.assertTrue(torch.allclose(got, ref, atol=1e-10),
                        f"max diff {(got-ref).abs().max()}")

    def test_full_rotary_reduces_to_plain(self):
        """When rotary_dim == head_dim (Llama), partial path == full RoPE."""
        torch.manual_seed(1)
        ang = torch.randn(self.B, self.S, self.head_dim // 2, dtype=torch.float64)
        emb = torch.cat([ang, ang], dim=-1)
        cos, sin = emb.cos(), emb.sin()
        got = _partial_rope_query(self.q, cos, sin)
        full = (self.q * cos.unsqueeze(1)) + (rotate_half(self.q) * sin.unsqueeze(1))
        self.assertTrue(torch.allclose(got, full, atol=1e-10))


if __name__ == "__main__":
    unittest.main()
