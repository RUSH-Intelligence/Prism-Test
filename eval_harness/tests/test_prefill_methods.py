"""Tests for eval_harness.prefill_methods package.

All tests are GPU-free — they use small synthetic tensors and the
``object.__new__`` pattern to bypass model loading.
"""

from __future__ import annotations

import math
import unittest
from dataclasses import dataclass
from typing import Optional, Tuple
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from eval_harness.prefill_methods.base import (
    PrefillMethod,
    _rotate_half,
    apply_rotary_pos_emb,
    build_cos_sin,
    get_inv_freq,
    get_rotary_emb,
    undo_rotary_pos_emb,
)
from eval_harness.prefill_methods.registry import (
    _PREFILL_METHOD_REGISTRY,
    available_prefill_methods,
    ensure_methods_loaded,
    get_prefill_method,
    register_prefill_method,
)


# ======================================================================
# Registry tests
# ======================================================================


class TestRegistry(unittest.TestCase):
    def test_ensure_methods_loaded_populates_registry(self):
        ensure_methods_loaded()
        names = available_prefill_methods()
        self.assertIn("reattention", names)
        self.assertIn("dca", names)

    def test_get_default_returns_base(self):
        for alias in ("none", "default", "standard"):
            method = get_prefill_method(alias)
            self.assertIsInstance(method, PrefillMethod)
            self.assertEqual(type(method), PrefillMethod)

    def test_get_reattention(self):
        ensure_methods_loaded()
        method = get_prefill_method(
            "reattention", global_size=16, local_size=128, mid_size=2,
        )
        from eval_harness.prefill_methods.reattention import ReAttentionMethod

        self.assertIsInstance(method, ReAttentionMethod)
        self.assertEqual(method.global_size, 16)
        self.assertEqual(method.local_size, 128)
        self.assertEqual(method.mid_size, 2)

    def test_get_reattention_aliases(self):
        ensure_methods_loaded()
        for alias in ("re_attention", "reatt"):
            method = get_prefill_method(alias)
            from eval_harness.prefill_methods.reattention import ReAttentionMethod

            self.assertIsInstance(method, ReAttentionMethod)

    def test_get_dca(self):
        ensure_methods_loaded()
        method = get_prefill_method("dca", chunk_size=4096)
        from eval_harness.prefill_methods.dca import DCAMethod

        self.assertIsInstance(method, DCAMethod)
        self.assertEqual(method.chunk_size, 4096)

    def test_get_dca_alias(self):
        ensure_methods_loaded()
        method = get_prefill_method("dual_chunk_attention")
        from eval_harness.prefill_methods.dca import DCAMethod

        self.assertIsInstance(method, DCAMethod)

    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            get_prefill_method("nonexistent_method_xyz")

    def test_register_prefill_method_decorator(self):
        @register_prefill_method("_test_method_123", aliases=["_tm123"])
        @dataclass
        class _TestMethod(PrefillMethod):
            pass

        self.assertIn("_test_method_123", _PREFILL_METHOD_REGISTRY)
        self.assertIn("_tm123", _PREFILL_METHOD_REGISTRY)
        self.assertIs(_PREFILL_METHOD_REGISTRY["_test_method_123"], _TestMethod)

        # Clean up.
        del _PREFILL_METHOD_REGISTRY["_test_method_123"]
        del _PREFILL_METHOD_REGISTRY["_tm123"]


# ======================================================================
# RoPE utility tests
# ======================================================================


class TestRotateHalf(unittest.TestCase):
    def test_rotate_half_shape(self):
        x = torch.randn(2, 4, 8, 64)
        y = _rotate_half(x)
        self.assertEqual(y.shape, x.shape)

    def test_rotate_half_values(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        y = _rotate_half(x)
        # First half negated and swapped with second half.
        expected = torch.tensor([-3.0, -4.0, 1.0, 2.0])
        torch.testing.assert_close(y, expected)


class TestApplyUndoRoPE(unittest.TestCase):
    def test_apply_then_undo_is_identity(self):
        """Apply RoPE then undo → should recover original tensor."""
        B, H, S, D = 1, 2, 16, 32
        x = torch.randn(B, H, S, D)
        inv_freq = torch.randn(D // 2).abs() * 0.01
        pos = torch.arange(S)
        cos, sin = build_cos_sin(pos, inv_freq, x.device, torch.float32)

        rotated = apply_rotary_pos_emb(x, cos, sin)
        recovered = undo_rotary_pos_emb(rotated, cos, sin)

        torch.testing.assert_close(recovered, x, atol=1e-5, rtol=1e-5)

    def test_undo_then_apply_is_identity(self):
        """Undo RoPE then apply → should recover original tensor."""
        B, H, S, D = 1, 2, 8, 16
        x = torch.randn(B, H, S, D)
        inv_freq = torch.randn(D // 2).abs() * 0.01
        pos = torch.arange(S)
        cos, sin = build_cos_sin(pos, inv_freq, x.device, torch.float32)

        unrotated = undo_rotary_pos_emb(x, cos, sin)
        recovered = apply_rotary_pos_emb(unrotated, cos, sin)

        torch.testing.assert_close(recovered, x, atol=1e-5, rtol=1e-5)

    def test_rope_changes_tensor(self):
        """RoPE should actually modify the tensor (not be a no-op)."""
        B, H, S, D = 1, 1, 4, 8
        x = torch.ones(B, H, S, D)
        inv_freq = torch.ones(D // 2) * 0.1
        pos = torch.arange(S)
        cos, sin = build_cos_sin(pos, inv_freq, x.device, torch.float32)

        rotated = apply_rotary_pos_emb(x, cos, sin)
        self.assertFalse(torch.allclose(rotated, x))


class TestBuildCosSin(unittest.TestCase):
    def test_output_shapes(self):
        S, D = 16, 32
        inv_freq = torch.randn(D // 2)
        pos = torch.arange(S)
        cos, sin = build_cos_sin(pos, inv_freq, torch.device("cpu"), torch.float32)
        # [1, 1, S, D]
        self.assertEqual(cos.shape, (1, 1, S, D))
        self.assertEqual(sin.shape, (1, 1, S, D))

    def test_position_zero_has_zero_sin(self):
        D = 16
        inv_freq = torch.randn(D // 2)
        pos = torch.tensor([0])
        cos, sin = build_cos_sin(pos, inv_freq, torch.device("cpu"), torch.float32)
        # sin(0) = 0 for all frequencies.
        torch.testing.assert_close(sin, torch.zeros_like(sin), atol=1e-6, rtol=0)

    def test_batched_positions(self):
        D = 8
        inv_freq = torch.randn(D // 2)
        pos = torch.tensor([[0, 1, 2], [3, 4, 5]])  # [2, 3]
        cos, sin = build_cos_sin(pos, inv_freq, torch.device("cpu"), torch.float32)
        self.assertEqual(cos.shape, (2, 1, 3, D))


# ======================================================================
# Base PrefillMethod tests
# ======================================================================


class TestBasePrefillMethod(unittest.TestCase):
    def test_base_hook_returns_none(self):
        """Base method is a no-op — should return None from hook."""
        method = PrefillMethod()
        result = method.prefill_forward_hook(
            module=nn.Linear(4, 4),
            hidden_states=torch.randn(1, 4, 8),
            keys=torch.randn(1, 2, 4, 8),
            values=torch.randn(1, 2, 4, 8),
            kwargs={},
        )
        self.assertIsNone(result)

    def test_base_inv_freq_passthrough(self):
        method = PrefillMethod()
        inv = torch.randn(32)
        result = method.compute_inv_freq(inv, seq_len=1024)
        torch.testing.assert_close(result, inv)

    def test_default_question_position_ids(self):
        method = PrefillMethod()
        pos = method.compute_question_position_ids(100, 10, torch.device("cpu"))
        expected = torch.arange(100, 110).unsqueeze(0)
        torch.testing.assert_close(pos, expected)

    def test_supports_chunked_prefill(self):
        method = PrefillMethod()
        self.assertTrue(method.supports_chunked_prefill)

    def test_supported_backends(self):
        method = PrefillMethod()
        self.assertEqual(method.supported_backends(), {"research"})


# ======================================================================
# ReAttention tests
# ======================================================================


class _FakeAttnModule(nn.Module):
    """Minimal attention module with q_proj, k_proj, and required attributes.

    If ``identity_q`` is set, ``q_proj`` is the identity (hidden_dim ==
    num_heads * head_dim), so the raw query equals the hidden states reshaped
    — letting tests control raw Q directly.
    """

    def __init__(self, hidden_dim=32, num_heads=4, head_dim=8, num_kv_heads=2,
                 identity_q=False, seed=0):
        super().__init__()
        self.num_heads = num_heads
        self.num_key_value_heads = num_kv_heads
        self.head_dim = head_dim
        self.layer_idx = 0
        self.q_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, num_kv_heads * head_dim, bias=False)
        if identity_q:
            assert hidden_dim == num_heads * head_dim
            with torch.no_grad():
                self.q_proj.weight.copy_(torch.eye(hidden_dim))
        else:
            torch.manual_seed(seed)
            with torch.no_grad():
                self.q_proj.weight.normal_()


def _identity_pos_emb(B, S, D):
    """RoPE that is a no-op: cos=1, sin=0 — so un-rotation is exact identity."""
    return torch.ones(B, S, D), torch.zeros(B, S, D)


def _rope_pos_emb(positions, D, base=10000.0):
    """Build real (cos, sin) of shape [B, S, D] for the given positions."""
    from eval_harness.prefill_methods.base import build_cos_sin

    half = D // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))
    cos, sin = build_cos_sin(positions, inv_freq, torch.device("cpu"), torch.float32)
    return cos.squeeze(1), sin.squeeze(1), inv_freq  # [B, S, D]


def _reattention_reference(
    raw_q, raw_k, values, *, global_size, local_size, mid_size, span_size,
    recall_type="qk", recall_clip=-1,
):
    """Pure reference reproduction of the ReAttention prefill selection.

    Returns the sorted 1-D tensor of absolute kept indices.  This mirrors the
    OpenMOSS ``RECacheV2.update`` prefill path exactly.
    """
    B, H_kv, S, D = raw_k.shape
    ms, me = global_size, S - local_size
    n_mid = me - ms

    if recall_type == "qk":
        rk = raw_k
    elif recall_type == "qkv":
        rk = raw_k * values.norm(p=1, dim=-1, keepdim=True)
    elif recall_type == "qkv2":
        rk = raw_k * values.norm(p=2, dim=-1, keepdim=True)
    else:
        raise ValueError(recall_type)

    rk_mid = rk[:, :, ms:me, :]
    H_q = raw_q.shape[1]
    n_rep = max(1, H_q // H_kv)
    if n_rep > 1:
        rk_mid = rk_mid[:, :, None].expand(B, H_kv, n_rep, n_mid, D).reshape(B, H_q, n_mid, D)

    scores = torch.einsum("bhqd,bhmd->bhqm", raw_q.float(), rk_mid.float())
    k = min(mid_size, n_mid)
    _, topk = scores.topk(k, dim=-1)

    flat = topk.reshape(-1)
    if recall_clip < 0:
        uniq = torch.unique(flat)
    else:
        uniq, counts = torch.unique(flat, return_counts=True)
        if uniq.numel() > recall_clip:
            _, keep = torch.topk(counts, k=recall_clip)
            uniq = uniq[keep]

    if span_size > 0:
        off = torch.arange(-(span_size // 2), (span_size + 1) // 2)
        exp = (uniq[:, None] + off[None, :]).reshape(-1)
    else:
        exp = uniq
    exp = torch.unique(exp.clamp(0, n_mid - 1))

    g = torch.arange(0, global_size)
    m = exp + global_size
    loc = torch.arange(S - local_size, S)
    return torch.cat([g, m, loc])


class TestReAttentionMethod(unittest.TestCase):
    def _make_method(self, **kwargs):
        from eval_harness.prefill_methods.reattention import ReAttentionMethod

        defaults = dict(global_size=4, local_size=4, mid_size=4, span_size=2)
        defaults.update(kwargs)
        return ReAttentionMethod(**defaults)

    # -- structural behavior -------------------------------------------------

    def test_short_sequence_returns_none(self):
        """If S <= global_size + local_size, nothing is selected."""
        method = self._make_method(global_size=4, local_size=4)
        result = method.prefill_forward_hook(
            module=_FakeAttnModule(),
            hidden_states=torch.randn(1, 8, 32),
            keys=torch.randn(1, 2, 8, 8),  # S=8 == global_size + local_size
            values=torch.randn(1, 2, 8, 8),
            kwargs={},
        )
        self.assertIsNone(result)

    def test_selects_subset_of_keys(self):
        # recall_clip caps the middle set so the cache is genuinely pruned
        # (with unbounded random scores the top-k union can cover all middle
        # tokens, which is correct but uninteresting here).
        method = self._make_method(
            global_size=4, local_size=4, mid_size=2, span_size=0, recall_clip=6,
        )
        B, H_kv, S, D = 1, 2, 64, 8
        module = _FakeAttnModule(hidden_dim=32, num_heads=4, head_dim=D, num_kv_heads=H_kv)

        keys = torch.randn(B, H_kv, S, D)
        values = torch.randn(B, H_kv, S, D)
        hidden = torch.randn(B, S, 32)
        cos, sin = _identity_pos_emb(B, S, D)

        result = method.prefill_forward_hook(
            module, hidden, keys, values, {"position_embeddings": (cos, sin)},
        )
        self.assertIsNotNone(result)
        new_keys, new_values = result
        self.assertLess(new_keys.shape[2], S)             # pruned
        self.assertLessEqual(new_keys.shape[2], 6 + 4 + 4)  # <= clip + global + local
        self.assertGreaterEqual(new_keys.shape[2], 4 + 4)   # >= global + local
        self.assertEqual(new_keys.shape, new_values.shape)

    def test_mid_size_zero_keeps_only_sink_and_local(self):
        """mid_size == 0 → StreamingLLM-style [global | local] retention."""
        method = self._make_method(global_size=4, local_size=4, mid_size=0)
        B, H_kv, S, D = 1, 2, 32, 8
        module = _FakeAttnModule(hidden_dim=32, num_heads=4, head_dim=D, num_kv_heads=H_kv)
        keys = torch.randn(B, H_kv, S, D)
        values = torch.randn(B, H_kv, S, D)

        result = method.prefill_forward_hook(module, torch.randn(B, S, 32), keys, values, {})
        self.assertIsNotNone(result)
        new_keys, _ = result
        self.assertEqual(new_keys.shape[2], 8)  # 4 global + 4 local exactly
        torch.testing.assert_close(new_keys[:, :, :4, :], keys[:, :, :4, :])
        torch.testing.assert_close(new_keys[:, :, -4:, :], keys[:, :, -4:, :])

    def test_global_and_local_always_retained(self):
        method = self._make_method(global_size=4, local_size=4, mid_size=1, span_size=0)
        B, H_kv, S, D = 1, 2, 32, 8
        module = _FakeAttnModule(hidden_dim=32, num_heads=4, head_dim=D, num_kv_heads=H_kv)

        keys = torch.zeros(B, H_kv, S, D)
        keys[:, :, :4, :] = 1.0   # global
        keys[:, :, -4:, :] = 2.0  # local
        values = keys.clone()
        cos, sin = _identity_pos_emb(B, S, D)

        result = method.prefill_forward_hook(
            module, torch.randn(B, S, 32), keys, values,
            {"position_embeddings": (cos, sin)},
        )
        self.assertIsNotNone(result)
        new_keys, _ = result
        torch.testing.assert_close(new_keys[:, :, :4, :], torch.ones(B, H_kv, 4, D))
        torch.testing.assert_close(new_keys[:, :, -4:, :], torch.ones(B, H_kv, 4, D) * 2.0)

    def test_output_is_causally_sorted(self):
        method = self._make_method(global_size=2, local_size=2, mid_size=4, span_size=1)
        B, H_kv, S, D = 1, 2, 20, 8
        module = _FakeAttnModule(hidden_dim=32, num_heads=4, head_dim=D, num_kv_heads=H_kv)

        # Distinct value per position so ordering is observable.
        keys = torch.arange(S, dtype=torch.float32).view(1, 1, S, 1).expand(B, H_kv, S, D).contiguous()
        values = keys.clone()
        cos, sin = _identity_pos_emb(B, S, D)

        result = method.prefill_forward_hook(
            module, torch.randn(B, S, 32), keys, values,
            {"position_embeddings": (cos, sin)},
        )
        self.assertIsNotNone(result)
        markers = result[0][0, 0, :, 0]
        self.assertTrue(torch.all(markers[1:] >= markers[:-1]))

    def test_fallback_without_q_proj(self):
        """Module without q_proj falls back to key-norm scoring."""
        method = self._make_method(global_size=2, local_size=2, mid_size=4, span_size=0)
        B, H_kv, S, D = 1, 2, 16, 8
        module = nn.Module()
        module.layer_idx = 0
        module.head_dim = D
        module.num_heads = 4
        cos, sin = _identity_pos_emb(B, S, D)

        result = method.prefill_forward_hook(
            module, torch.randn(B, S, 32), torch.randn(B, H_kv, S, D),
            torch.randn(B, H_kv, S, D), {"position_embeddings": (cos, sin)},
        )
        self.assertIsNotNone(result)

    # -- exact reproduction of the original algorithm ------------------------

    def test_matches_reference_selection_identity_rope(self):
        """With identity RoPE, the hook output is bit-identical to the
        pure reference reproduction of the original algorithm."""
        torch.manual_seed(7)
        B, H_kv, S, D = 1, 2, 48, 8
        num_heads = 4
        module = _FakeAttnModule(
            hidden_dim=num_heads * D, num_heads=num_heads, head_dim=D,
            num_kv_heads=H_kv, seed=11,
        )
        keys = torch.randn(B, H_kv, S, D)
        values = torch.randn(B, H_kv, S, D)
        hidden = torch.randn(B, S, num_heads * D)

        params = dict(global_size=4, local_size=6, mid_size=3, span_size=2)
        method = self._make_method(**params)
        cos, sin = _identity_pos_emb(B, S, D)
        new_keys, new_values = method.prefill_forward_hook(
            module, hidden, keys, values, {"position_embeddings": (cos, sin)},
        )

        # Reference: identity RoPE → raw_k == keys; raw_q == q_proj(hidden).
        raw_q = module.q_proj(hidden).view(B, S, num_heads, D).transpose(1, 2)
        ref_idx = _reattention_reference(
            raw_q, keys, values, recall_type="qk", recall_clip=-1, **params,
        )
        exp_keys = keys[:, :, ref_idx, :]
        exp_values = values[:, :, ref_idx, :]

        self.assertEqual(new_keys.shape, exp_keys.shape)
        self.assertTrue(torch.equal(new_keys, exp_keys))
        self.assertTrue(torch.equal(new_values, exp_values))

    def test_matches_reference_qkv2_weighting(self):
        """recall_type='qkv2' weights keys by L2 value norms — verified
        against the reference."""
        torch.manual_seed(3)
        B, H_kv, S, D = 1, 1, 40, 8
        num_heads = 2
        module = _FakeAttnModule(
            hidden_dim=num_heads * D, num_heads=num_heads, head_dim=D,
            num_kv_heads=H_kv, seed=5,
        )
        keys = torch.randn(B, H_kv, S, D)
        values = torch.randn(B, H_kv, S, D) * 3.0  # exaggerate norm spread
        hidden = torch.randn(B, S, num_heads * D)

        params = dict(global_size=2, local_size=4, mid_size=4, span_size=0)
        method = self._make_method(recall_type="qkv2", **params)
        cos, sin = _identity_pos_emb(B, S, D)
        new_keys, _ = method.prefill_forward_hook(
            module, hidden, keys, values, {"position_embeddings": (cos, sin)},
        )

        raw_q = module.q_proj(hidden).view(B, S, num_heads, D).transpose(1, 2)
        ref_idx = _reattention_reference(
            raw_q, keys, values, recall_type="qkv2", recall_clip=-1, **params,
        )
        self.assertTrue(torch.equal(new_keys, keys[:, :, ref_idx, :]))

    def test_rejects_pe_recall_variants(self):
        """The '_pe' variants score against position-encoded (rotated) keys,
        which this hook port cannot honor (it un-rotates the cache
        unconditionally) — they must raise instead of silently degrading to
        the base variants."""
        torch.manual_seed(13)
        B, H_kv, S, D = 1, 2, 16, 8
        num_heads = 4
        module = _FakeAttnModule(
            hidden_dim=num_heads * D, num_heads=num_heads, head_dim=D,
            num_kv_heads=H_kv, seed=13,
        )
        keys = torch.randn(B, H_kv, S, D)
        values = torch.randn(B, H_kv, S, D)
        hidden = torch.randn(B, S, num_heads * D)
        cos, sin = _identity_pos_emb(B, S, D)

        for rt in ("qk_pe", "qkv_pe", "qkv2_pe"):
            method = self._make_method(
                global_size=4, local_size=4, mid_size=2, span_size=0,
                recall_type=rt,
            )
            with self.assertRaisesRegex(ValueError, "_pe"):
                method.prefill_forward_hook(
                    module, hidden, keys, values,
                    {"position_embeddings": (cos, sin)},
                )

    def test_recall_clip_caps_selection(self):
        """recall_clip bounds the unique middle set before span expansion."""
        torch.manual_seed(1)
        B, H_kv, S, D = 1, 2, 64, 8
        num_heads = 4
        module = _FakeAttnModule(
            hidden_dim=num_heads * D, num_heads=num_heads, head_dim=D,
            num_kv_heads=H_kv, seed=2,
        )
        keys = torch.randn(B, H_kv, S, D)
        values = torch.randn(B, H_kv, S, D)
        hidden = torch.randn(B, S, num_heads * D)
        cos, sin = _identity_pos_emb(B, S, D)

        unclipped = self._make_method(
            global_size=4, local_size=4, mid_size=8, span_size=0, recall_clip=-1,
        ).prefill_forward_hook(module, hidden, keys, values, {"position_embeddings": (cos, sin)})
        clipped = self._make_method(
            global_size=4, local_size=4, mid_size=8, span_size=0, recall_clip=3,
        ).prefill_forward_hook(module, hidden, keys, values, {"position_embeddings": (cos, sin)})

        # Clipped middle = at most 3 tokens; total <= 3 + global + local.
        self.assertLessEqual(clipped[0].shape[2], 3 + 4 + 4)
        self.assertLessEqual(clipped[0].shape[2], unclipped[0].shape[2])

    def test_gqa_shapes_and_selection(self):
        """GQA (H_q > H_kv): scoring repeats KV heads; output keeps H_kv."""
        torch.manual_seed(9)
        B, H_kv, S, D = 1, 2, 40, 8
        num_heads = 8  # 4 query heads per kv head
        module = _FakeAttnModule(
            hidden_dim=num_heads * D, num_heads=num_heads, head_dim=D,
            num_kv_heads=H_kv, seed=4,
        )
        keys = torch.randn(B, H_kv, S, D)
        values = torch.randn(B, H_kv, S, D)
        hidden = torch.randn(B, S, num_heads * D)

        params = dict(global_size=2, local_size=4, mid_size=2, span_size=0)
        method = self._make_method(**params)
        cos, sin = _identity_pos_emb(B, S, D)
        new_keys, new_values = method.prefill_forward_hook(
            module, hidden, keys, values, {"position_embeddings": (cos, sin)},
        )
        self.assertEqual(new_keys.shape[1], H_kv)

        raw_q = module.q_proj(hidden).view(B, S, num_heads, D).transpose(1, 2)
        ref_idx = _reattention_reference(raw_q, keys, values, **params)
        self.assertTrue(torch.equal(new_keys, keys[:, :, ref_idx, :]))

    # -- position-agnostic property + reproducibility ------------------------

    def test_selection_is_position_agnostic(self):
        """The same raw keys, rotated to *different* absolute positions, must
        yield the *same* selection — because the hook un-rotates before
        scoring.  This is the defining property of ReAttention."""
        from eval_harness.prefill_methods.base import apply_rotary_pos_emb

        B, H_kv, S, D = 1, 1, 32, 8
        num_heads = 1
        global_size, local_size, mid_size = 4, 4, 2
        n_mid = S - global_size - local_size

        # Construct well-separated affinities so fp jitter cannot flip top-k:
        # raw_q is a fixed unit basis vector; middle key j aligns with it with
        # a distinct descending weight.
        module = _FakeAttnModule(
            hidden_dim=num_heads * D, num_heads=num_heads, head_dim=D,
            num_kv_heads=H_kv, identity_q=True,
        )
        raw_q_vec = torch.zeros(D)
        raw_q_vec[0] = 1.0
        hidden = raw_q_vec.view(1, 1, D).expand(B, S, D).contiguous()

        raw_k = torch.randn(B, H_kv, S, D) * 0.01
        weights = torch.linspace(5.0, 1.0, n_mid)  # strictly descending, gap 4/(n_mid-1)
        raw_k[:, :, global_size:S - local_size, 0] = weights
        values = torch.randn(B, H_kv, S, D)

        method = self._make_method(
            global_size=global_size, local_size=local_size, mid_size=mid_size, span_size=0,
        )

        def run(positions):
            cos, sin, _ = _rope_pos_emb(positions.unsqueeze(0), D)
            rotated = apply_rotary_pos_emb(raw_k, cos.unsqueeze(1), sin.unsqueeze(1))
            nk, _ = method.prefill_forward_hook(
                module, hidden, rotated, values, {"position_embeddings": (cos, sin)},
            )
            return nk.shape[2]

        # Two different absolute position assignments for the cached keys.
        out_a = run(torch.arange(S))
        out_b = run(torch.arange(S) + 1000)
        self.assertEqual(out_a, out_b)

        # And the selected middle tokens are the two highest-weighted ones
        # (positions global_size+0 and global_size+1), regardless of rotation.
        cos, sin, _ = _rope_pos_emb(torch.arange(S).unsqueeze(0), D)
        rotated = apply_rotary_pos_emb(raw_k, cos.unsqueeze(1), sin.unsqueeze(1))
        nk, _ = method.prefill_forward_hook(
            module, hidden, rotated, values, {"position_embeddings": (cos, sin)},
        )
        # Recover which middle key has the largest weight after un-rotation.
        ref_idx = _reattention_reference(
            raw_q_vec.view(1, 1, 1, D), raw_k, values,
            global_size=global_size, local_size=local_size, mid_size=mid_size, span_size=0,
        )
        self.assertEqual(nk.shape[2], ref_idx.numel())

    def test_deterministic_across_repeated_runs(self):
        """Identical inputs (real RoPE) produce byte-identical output every
        time — required for reproducible prefill benchmarking."""
        from eval_harness.prefill_methods.base import apply_rotary_pos_emb

        torch.manual_seed(123)
        B, H_kv, S, D = 1, 2, 56, 8
        num_heads = 4
        module = _FakeAttnModule(
            hidden_dim=num_heads * D, num_heads=num_heads, head_dim=D,
            num_kv_heads=H_kv, seed=42,
        )
        raw_k = torch.randn(B, H_kv, S, D)
        values = torch.randn(B, H_kv, S, D)
        hidden = torch.randn(B, S, num_heads * D)
        cos, sin, _ = _rope_pos_emb(torch.arange(S).unsqueeze(0), D)
        rotated = apply_rotary_pos_emb(raw_k, cos.unsqueeze(1), sin.unsqueeze(1))
        kwargs = {"position_embeddings": (cos, sin)}

        method = self._make_method(global_size=4, local_size=8, mid_size=4, span_size=4)

        outs = []
        for _ in range(3):
            nk, nv = method.prefill_forward_hook(
                module, hidden, rotated.clone(), values.clone(), kwargs,
            )
            outs.append((nk, nv))

        for nk, nv in outs[1:]:
            self.assertTrue(torch.equal(nk, outs[0][0]))
            self.assertTrue(torch.equal(nv, outs[0][1]))

    def test_unrotation_recovers_raw_keys(self):
        """The hook's internal un-rotation matches the analytic raw keys to
        high precision (validates the position-agnostic recovery)."""
        from eval_harness.prefill_methods.base import apply_rotary_pos_emb

        B, H_kv, S, D = 1, 2, 24, 8
        raw_k = torch.randn(B, H_kv, S, D)
        cos, sin, _ = _rope_pos_emb(torch.arange(S).unsqueeze(0), D)
        rotated = apply_rotary_pos_emb(raw_k, cos.unsqueeze(1), sin.unsqueeze(1))

        method = self._make_method()
        recovered = method._unrotate_keys(
            rotated, {"position_embeddings": (cos, sin)},
        )
        torch.testing.assert_close(recovered, raw_k, atol=1e-5, rtol=1e-5)

    def test_unrotation_falls_back_to_inv_freq(self):
        """Without position_embeddings, un-rotation uses the cached inv_freq."""
        from eval_harness.prefill_methods.base import apply_rotary_pos_emb

        B, H_kv, S, D = 1, 1, 20, 8
        raw_k = torch.randn(B, H_kv, S, D)
        cos, sin, inv_freq = _rope_pos_emb(torch.arange(S).unsqueeze(0), D)
        rotated = apply_rotary_pos_emb(raw_k, cos.unsqueeze(1), sin.unsqueeze(1))

        method = self._make_method()
        method._inv_freq = inv_freq  # what __call__ would have stashed
        recovered = method._unrotate_keys(rotated, {})  # no position_embeddings
        torch.testing.assert_close(recovered, raw_k, atol=1e-5, rtol=1e-5)


# ======================================================================
# ReAttention compact end-anchored repositioning tests
# ======================================================================


class TestReAttentionReposition(unittest.TestCase):
    """Tests for the ``reposition`` mode: compact end-anchored re-rotation of
    the retained KV cache."""

    def _make_method(self, **kwargs):
        from eval_harness.prefill_methods.reattention import ReAttentionMethod

        defaults = dict(global_size=4, local_size=4, mid_size=4, span_size=2)
        defaults.update(kwargs)
        return ReAttentionMethod(**defaults)

    def test_reposition_off_is_byte_identical(self):
        """reposition=False must be byte-identical to the existing default
        selection-only path (regression guard)."""
        from eval_harness.prefill_methods.base import apply_rotary_pos_emb

        torch.manual_seed(31)
        B, H_kv, S, D = 1, 2, 56, 8
        num_heads = 4
        module = _FakeAttnModule(
            hidden_dim=num_heads * D, num_heads=num_heads, head_dim=D,
            num_kv_heads=H_kv, seed=17,
        )
        raw_k = torch.randn(B, H_kv, S, D)
        values = torch.randn(B, H_kv, S, D)
        hidden = torch.randn(B, S, num_heads * D)
        cos, sin, inv_freq = _rope_pos_emb(torch.arange(S).unsqueeze(0), D)
        rotated = apply_rotary_pos_emb(raw_k, cos.unsqueeze(1), sin.unsqueeze(1))
        kwargs = {"position_embeddings": (cos, sin)}

        params = dict(global_size=4, local_size=8, mid_size=4, span_size=4, recall_clip=6)
        # Default path (reposition not set → False).
        default_method = self._make_method(**params)
        default_method._inv_freq = inv_freq
        ref_keys, ref_values = default_method.prefill_forward_hook(
            module, hidden, rotated.clone(), values.clone(), kwargs,
        )

        # Explicit reposition=False.
        off_method = self._make_method(reposition=False, **params)
        off_method._inv_freq = inv_freq
        off_keys, off_values = off_method.prefill_forward_hook(
            module, hidden, rotated.clone(), values.clone(), kwargs,
        )

        self.assertTrue(torch.equal(off_keys, ref_keys))
        self.assertTrue(torch.equal(off_values, ref_values))

    def test_reposition_preserves_raw_content(self):
        """With reposition=True, un-rotating the returned keys at the NEW
        compacted positions recovers the same raw K as the original retained
        keys un-rotated at their ORIGINAL positions — only the encoded
        position changed, not the underlying content."""
        from eval_harness.prefill_methods.base import (
            apply_rotary_pos_emb,
            build_cos_sin,
            undo_rotary_pos_emb,
        )

        torch.manual_seed(41)
        B, H_kv, S, D = 1, 2, 96, 8
        num_heads = 4
        module = _FakeAttnModule(
            hidden_dim=num_heads * D, num_heads=num_heads, head_dim=D,
            num_kv_heads=H_kv, seed=23,
        )
        raw_k = torch.randn(B, H_kv, S, D)
        values = torch.randn(B, H_kv, S, D)
        hidden = torch.randn(B, S, num_heads * D)
        cos, sin, inv_freq = _rope_pos_emb(torch.arange(S).unsqueeze(0), D)
        rotated = apply_rotary_pos_emb(raw_k, cos.unsqueeze(1), sin.unsqueeze(1))
        kwargs = {"position_embeddings": (cos, sin)}

        params = dict(global_size=4, local_size=8, mid_size=4, span_size=2,
                      recall_clip=6)

        # First get the kept index set from the OFF path (same selection).
        off = self._make_method(reposition=False, **params)
        off._inv_freq = inv_freq
        off_keys, _ = off.prefill_forward_hook(
            module, hidden, rotated.clone(), values.clone(), kwargs,
        )
        R = off_keys.shape[2]

        # Recover the raw retained keys from the OFF path: un-rotate off_keys
        # against the original positions of the kept tokens.  We recompute the
        # kept absolute indices by matching the off_keys back through raw K.
        # Simpler: re-derive kept indices via the reference helper.
        raw_q = module.q_proj(hidden).view(B, S, num_heads, D).transpose(1, 2)
        ref_idx = _reattention_reference(
            raw_q, raw_k, values, recall_type="qk", **params,
        )
        raw_k_ret_expected = raw_k[:, :, ref_idx, :]  # [B, H_kv, R, D]
        self.assertEqual(raw_k_ret_expected.shape[2], R)

        # Now ON path.
        on = self._make_method(reposition=True, **params)
        on._inv_freq = inv_freq
        on_keys, _ = on.prefill_forward_hook(
            module, hidden, rotated.clone(), values.clone(), kwargs,
        )
        self.assertEqual(on_keys.shape, off_keys.shape)

        # Un-rotate on_keys at the NEW compacted positions [A-R, A-1].
        A = on._reposition_anchor()
        new_pos = torch.arange(A - R, A).unsqueeze(0)
        n_cos, n_sin = build_cos_sin(new_pos, inv_freq, torch.device("cpu"), torch.float32)
        recovered_raw = undo_rotary_pos_emb(on_keys, n_cos, n_sin)

        torch.testing.assert_close(
            recovered_raw, raw_k_ret_expected, atol=1e-4, rtol=1e-4,
        )

    def test_reposition_positions_in_window(self):
        """Long context: R <= A, implied positions in [0, A); and
        compute_question_position_ids switches between anchored (long) and
        original (short) positions."""
        from eval_harness.prefill_methods.base import apply_rotary_pos_emb

        torch.manual_seed(51)
        B, H_kv, D = 1, 2, 8
        num_heads = 4
        global_size, local_size, mid_size, span_size, recall_clip = 4, 16, 4, 8, 8
        S = 512
        module = _FakeAttnModule(
            hidden_dim=num_heads * D, num_heads=num_heads, head_dim=D,
            num_kv_heads=H_kv, seed=29,
        )
        raw_k = torch.randn(B, H_kv, S, D)
        values = torch.randn(B, H_kv, S, D)
        hidden = torch.randn(B, S, num_heads * D)
        cos, sin, inv_freq = _rope_pos_emb(torch.arange(S).unsqueeze(0), D)
        rotated = apply_rotary_pos_emb(raw_k, cos.unsqueeze(1), sin.unsqueeze(1))
        kwargs = {"position_embeddings": (cos, sin)}

        method = self._make_method(
            reposition=True, global_size=global_size, local_size=local_size,
            mid_size=mid_size, span_size=span_size, recall_clip=recall_clip,
        )
        method._inv_freq = inv_freq

        A = method._reposition_anchor()
        self.assertEqual(A, global_size + local_size + recall_clip * span_size)  # 4+16+8*8 = 84
        self.assertEqual(A, 84)

        on_keys, _ = method.prefill_forward_hook(
            module, hidden, rotated.clone(), values.clone(), kwargs,
        )
        R = on_keys.shape[2]
        # R <= A; implied positions [A-R, A-1] all within [0, A).
        self.assertLessEqual(R, A)
        self.assertGreaterEqual(A - R, 0)
        self.assertLess(A - 1, A)

        # compute_question_position_ids: long context → anchored.
        q_pos = method.compute_question_position_ids(S, 5, torch.device("cpu"))
        torch.testing.assert_close(q_pos, torch.arange(A, A + 5).unsqueeze(0))

        # Short context (S <= global+local) → original positions.
        short_ctx = global_size + local_size  # 20
        q_pos_short = method.compute_question_position_ids(short_ctx, 5, torch.device("cpu"))
        torch.testing.assert_close(
            q_pos_short, torch.arange(short_ctx, short_ctx + 5).unsqueeze(0),
        )

    def test_reposition_requires_bound(self):
        """reposition=True with no window and recall_clip<=0 → unbounded
        compacted window → ValueError."""
        from eval_harness.prefill_methods.base import apply_rotary_pos_emb

        method = self._make_method(
            reposition=True, recall_clip=-1, reposition_window=None,
            global_size=4, local_size=8, mid_size=4, span_size=2,
        )
        B, H_kv, S, D = 1, 2, 64, 8
        num_heads = 4
        method._inv_freq = 1.0 / (10000.0 ** (torch.arange(0, D // 2).float() / (D // 2)))
        module = _FakeAttnModule(
            hidden_dim=num_heads * D, num_heads=num_heads, head_dim=D,
            num_kv_heads=H_kv, seed=3,
        )
        raw_k = torch.randn(B, H_kv, S, D)
        values = torch.randn(B, H_kv, S, D)
        hidden = torch.randn(B, S, num_heads * D)
        cos, sin, _ = _rope_pos_emb(torch.arange(S).unsqueeze(0), D)
        rotated = apply_rotary_pos_emb(raw_k, cos.unsqueeze(1), sin.unsqueeze(1))

        with self.assertRaises(ValueError):
            method._reposition_anchor()
        with self.assertRaises(ValueError):
            method.prefill_forward_hook(
                module, hidden, rotated, values, {"position_embeddings": (cos, sin)},
            )

    def test_reposition_streaming_path(self):
        """mid_size=0 (StreamingLLM) + reposition=True over a long context:
        retains exactly global+local, repositions, returns without error."""
        from eval_harness.prefill_methods.base import apply_rotary_pos_emb

        torch.manual_seed(61)
        B, H_kv, S, D = 1, 2, 128, 8
        num_heads = 4
        global_size, local_size = 4, 8
        module = _FakeAttnModule(
            hidden_dim=num_heads * D, num_heads=num_heads, head_dim=D,
            num_kv_heads=H_kv, seed=33,
        )
        raw_k = torch.randn(B, H_kv, S, D)
        values = torch.randn(B, H_kv, S, D)
        hidden = torch.randn(B, S, num_heads * D)
        cos, sin, inv_freq = _rope_pos_emb(torch.arange(S).unsqueeze(0), D)
        rotated = apply_rotary_pos_emb(raw_k, cos.unsqueeze(1), sin.unsqueeze(1))
        kwargs = {"position_embeddings": (cos, sin)}

        method = self._make_method(
            reposition=True, mid_size=0, global_size=global_size,
            local_size=local_size, reposition_window=64,
        )
        method._inv_freq = inv_freq

        on_keys, on_values = method.prefill_forward_hook(
            module, hidden, rotated.clone(), values.clone(), kwargs,
        )
        self.assertEqual(on_keys.shape[2], global_size + local_size)
        self.assertEqual(on_values.shape[2], global_size + local_size)

        # Values carry no RoPE: must equal the StreamingLLM gather of values.
        expected_idx = torch.cat([
            torch.arange(0, global_size),
            torch.arange(S - local_size, S),
        ])
        torch.testing.assert_close(on_values, values[:, :, expected_idx, :])

        # Keys re-rotated to [A-R, A-1]; un-rotating recovers the raw retained.
        from eval_harness.prefill_methods.base import build_cos_sin, undo_rotary_pos_emb
        R = on_keys.shape[2]
        A = method._reposition_anchor()
        new_pos = torch.arange(A - R, A).unsqueeze(0)
        n_cos, n_sin = build_cos_sin(new_pos, inv_freq, torch.device("cpu"), torch.float32)
        recovered = undo_rotary_pos_emb(on_keys, n_cos, n_sin)
        torch.testing.assert_close(
            recovered, raw_k[:, :, expected_idx, :], atol=1e-4, rtol=1e-4,
        )

    def test_reposition_scaled_rope_kwargs_path_single_scale_factor(self):
        """Scaled-RoPE models (HF bakes attention_scaling s into cos/sin, so
        the cache holds s·R(pos)·k_raw): repositioning via the kwargs
        un-rotation path must yield s·R(new)·k_raw — exactly ONE factor of s,
        not the s² double-scaling defect."""
        from eval_harness.prefill_methods.base import (
            apply_rotary_pos_emb,
            build_cos_sin,
        )

        torch.manual_seed(71)
        scale = 1.31
        B, H_kv, S, D = 1, 2, 96, 8
        num_heads = 4
        module = _FakeAttnModule(
            hidden_dim=num_heads * D, num_heads=num_heads, head_dim=D,
            num_kv_heads=H_kv, seed=37,
        )
        raw_k = torch.randn(B, H_kv, S, D)
        values = torch.randn(B, H_kv, S, D)
        hidden = torch.randn(B, S, num_heads * D)
        cos, sin, inv_freq = _rope_pos_emb(torch.arange(S).unsqueeze(0), D)
        # Model-style cache: keys rotated with the SCALED trig → s·R(pos)·k_raw.
        scaled_cos, scaled_sin = cos * scale, sin * scale
        rotated = apply_rotary_pos_emb(
            raw_k, scaled_cos.unsqueeze(1), scaled_sin.unsqueeze(1),
        )
        kwargs = {"position_embeddings": (scaled_cos, scaled_sin)}

        params = dict(global_size=4, local_size=8, mid_size=4, span_size=2,
                      recall_clip=6)
        method = self._make_method(reposition=True, **params)
        method._inv_freq = inv_freq        # what __call__ would have stashed
        method._attention_scaling = scale  # ditto

        on_keys, _ = method.prefill_forward_hook(
            module, hidden, rotated.clone(), values.clone(), kwargs,
        )

        # Expected: kept indices re-derived via the reference helper (the
        # uniform s on scores never changes top-k), re-rotated to the NEW
        # compacted positions carrying exactly ONE factor of s.
        raw_q = module.q_proj(hidden).view(B, S, num_heads, D).transpose(1, 2)
        ref_idx = _reattention_reference(
            raw_q, raw_k, values, recall_type="qk", **params,
        )
        R = on_keys.shape[2]
        self.assertEqual(R, ref_idx.numel())
        A = method._reposition_anchor()
        new_pos = torch.arange(A - R, A).unsqueeze(0)
        n_cos, n_sin = build_cos_sin(new_pos, inv_freq, torch.device("cpu"), torch.float32)
        expected = scale * apply_rotary_pos_emb(raw_k[:, :, ref_idx, :], n_cos, n_sin)
        torch.testing.assert_close(on_keys, expected, atol=1e-4, rtol=1e-4)

    def test_reposition_scaled_rope_fallback_path_agrees(self):
        """Same scaled-RoPE setup but NO position_embeddings kwarg, so
        un-rotation falls back to the unscaled inv_freq trig: the result must
        be the identical s·R(new)·k_raw — the two un-rotation paths agree on
        the amplitude convention."""
        from eval_harness.prefill_methods.base import (
            apply_rotary_pos_emb,
            build_cos_sin,
        )

        torch.manual_seed(71)
        scale = 1.31
        B, H_kv, S, D = 1, 2, 96, 8
        num_heads = 4
        module = _FakeAttnModule(
            hidden_dim=num_heads * D, num_heads=num_heads, head_dim=D,
            num_kv_heads=H_kv, seed=37,
        )
        raw_k = torch.randn(B, H_kv, S, D)
        values = torch.randn(B, H_kv, S, D)
        hidden = torch.randn(B, S, num_heads * D)
        cos, sin, inv_freq = _rope_pos_emb(torch.arange(S).unsqueeze(0), D)
        rotated = apply_rotary_pos_emb(
            raw_k, (cos * scale).unsqueeze(1), (sin * scale).unsqueeze(1),
        )

        params = dict(global_size=4, local_size=8, mid_size=4, span_size=2,
                      recall_clip=6)
        method = self._make_method(reposition=True, **params)
        method._inv_freq = inv_freq
        method._attention_scaling = scale

        on_keys, _ = method.prefill_forward_hook(
            module, hidden, rotated.clone(), values.clone(), {},
        )

        raw_q = module.q_proj(hidden).view(B, S, num_heads, D).transpose(1, 2)
        ref_idx = _reattention_reference(
            raw_q, raw_k, values, recall_type="qk", **params,
        )
        R = on_keys.shape[2]
        self.assertEqual(R, ref_idx.numel())
        A = method._reposition_anchor()
        new_pos = torch.arange(A - R, A).unsqueeze(0)
        n_cos, n_sin = build_cos_sin(new_pos, inv_freq, torch.device("cpu"), torch.float32)
        expected = scale * apply_rotary_pos_emb(raw_k[:, :, ref_idx, :], n_cos, n_sin)
        torch.testing.assert_close(on_keys, expected, atol=1e-4, rtol=1e-4)


# ======================================================================
# ReAttention Triton-kernel wiring tests
# ======================================================================


def _cuda_and_triton_available():
    try:
        import triton  # noqa: F401
    except Exception:
        return False
    return torch.cuda.is_available()


class TestReAttentionUniformRetained(unittest.TestCase):
    """Layer-uniform retained length (the ragged-cache decode fix).

    HF's normal decode shares one causal mask / position grid across layers
    (sized from layer 0), so per-layer top-k selection must be equalized to a
    single retained length per prefill.  These tests drive the hook twice on
    the SAME method instance (two "layers" of one prefill) with engineered
    selections of different natural sizes, pinning the exact retained indices
    of the pad and shrink paths.

    Geometry (identity RoPE, identity q_proj so raw Q == hidden reshaped):
    ``global=4, local=8, S=36 → n_middle=24``; ``mid_size=2, span_size=4``
    (span window = ``[i-2, i+1]``).  Every query is ``e1``, so the Q·K score
    of a middle key is exactly its ``raw_k[..., 0]`` weight — the two weighted
    seeds are the top-2 picks of every query:

    * layer "A": seeds at relative middle 4 and 18 → disjoint spans
      ``{2,3,4,5} ∪ {16,17,18,19}`` → 8 middle indices;
    * layer "B": seeds at 10 and 11 → overlapping spans → ``{8..12}`` →
      5 middle indices.
    """

    G, L, S = 4, 8, 36
    N_MID = S - G - L  # 24
    B_, H_KV, H_Q, D = 1, 2, 4, 8

    A_SEEDS = (4, 18)
    A_MIDDLE = sorted({2, 3, 4, 5, 16, 17, 18, 19})
    B_SEEDS = (10, 11)
    B_MIDDLE = sorted({8, 9, 10, 11, 12})

    def _make_method(self, **kwargs):
        from eval_harness.prefill_methods.reattention import ReAttentionMethod

        defaults = dict(
            global_size=self.G, local_size=self.L, mid_size=2, span_size=4,
            use_triton_kernel="off",
        )
        defaults.update(kwargs)
        return ReAttentionMethod(**defaults)

    def _layer_inputs(self, seeds):
        """Module/hidden/keys/values/kwargs whose top-2 seeds are ``seeds``."""
        module = _FakeAttnModule(
            hidden_dim=self.H_Q * self.D, num_heads=self.H_Q, head_dim=self.D,
            num_kv_heads=self.H_KV, identity_q=True,
        )
        # Every query (all heads, all positions) is e1 → score == k[..., 0].
        hidden = torch.zeros(self.B_, self.S, self.H_Q * self.D)
        hidden[:, :, 0::self.D] = 1.0

        torch.manual_seed(7)
        keys = torch.randn(self.B_, self.H_KV, self.S, self.D) * 0.01
        keys[:, :, :, 0] = 0.0
        for w, rel in zip((5.0, 4.0), seeds):
            keys[:, :, self.G + rel, 0] = w
        values = torch.arange(
            self.B_ * self.H_KV * self.S * self.D, dtype=torch.float32,
        ).reshape(self.B_, self.H_KV, self.S, self.D)

        cos, sin = _identity_pos_emb(self.B_, self.S, self.D)
        return module, hidden, keys, values, {"position_embeddings": (cos, sin)}

    def _run(self, method, seeds):
        module, hidden, keys, values, kwargs = self._layer_inputs(seeds)
        nk, nv = method.prefill_forward_hook(module, hidden, keys, values, kwargs)
        return nk, nv, keys, values

    def _expected_abs(self, middle_rel):
        return (
            list(range(self.G))
            + [self.G + r for r in middle_rel]
            + list(range(self.S - self.L, self.S))
        )

    def _assert_retained(self, nk, nv, keys, values, middle_rel):
        idx = torch.tensor(self._expected_abs(middle_rel))
        self.assertEqual(nk.shape[2], idx.numel())
        self.assertTrue(torch.equal(nk, keys[:, :, idx]))
        self.assertTrue(torch.equal(nv, values[:, :, idx]))

    # -- natural (per-layer) selections, uniform off --------------------------

    def test_uniform_off_preserves_ragged_selection(self):
        method = self._make_method(uniform_retained=False)
        nk_a, nv_a, k_a, v_a = self._run(method, self.A_SEEDS)
        self._assert_retained(nk_a, nv_a, k_a, v_a, self.A_MIDDLE)
        nk_b, nv_b, k_b, v_b = self._run(method, self.B_SEEDS)
        self._assert_retained(nk_b, nv_b, k_b, v_b, self.B_MIDDLE)
        self.assertNotEqual(nk_a.shape[2], nk_b.shape[2])  # genuinely ragged

    # -- first layer sets the target; later layers conform ---------------------

    def test_first_layer_is_untouched_by_default_uniform(self):
        """With no explicit budget the first hooked layer defines the target,
        so its output is byte-identical to ``uniform_retained=False``."""
        nk_u, nv_u, _, _ = self._run(self._make_method(), self.A_SEEDS)
        nk_r, nv_r, _, _ = self._run(
            self._make_method(uniform_retained=False), self.A_SEEDS,
        )
        self.assertTrue(torch.equal(nk_u, nk_r))
        self.assertTrue(torch.equal(nv_u, nv_r))

    def test_pad_path_fills_with_most_recent_unselected(self):
        """Layer B (5 middles) after layer A (8) is padded with the 3 most
        recent unselected middle indices {21, 22, 23}."""
        method = self._make_method()
        nk_a, *_ = self._run(method, self.A_SEEDS)
        nk_b, nv_b, k_b, v_b = self._run(method, self.B_SEEDS)
        self.assertEqual(nk_b.shape[2], nk_a.shape[2])
        self._assert_retained(nk_b, nv_b, k_b, v_b, self.B_MIDDLE + [21, 22, 23])

    def test_shrink_path_keeps_top_frequency_seed_spans(self):
        """Layer A (8 middles) after layer B (5) shrinks by the frequency-clip
        rule: largest seed prefix whose expansion fits (seed 4 → {2,3,4,5}),
        recency-padded to the exact target ({23})."""
        method = self._make_method()
        nk_b, *_ = self._run(method, self.B_SEEDS)
        nk_a, nv_a, k_a, v_a = self._run(method, self.A_SEEDS)
        self.assertEqual(nk_a.shape[2], nk_b.shape[2])
        self._assert_retained(nk_a, nv_a, k_a, v_a, [2, 3, 4, 5, 23])

    def test_on_prefill_start_resets_target(self):
        method = self._make_method()
        self._run(method, self.A_SEEDS)           # target ← 8
        method.on_prefill_start(self.S)           # new prefill: target cleared
        nk_b, nv_b, k_b, v_b = self._run(method, self.B_SEEDS)
        self._assert_retained(nk_b, nv_b, k_b, v_b, self.B_MIDDLE)  # natural 5

    # -- explicit budget --------------------------------------------------------

    def test_uniform_budget_pins_both_layers(self):
        """budget=6: A shrinks (seed-4 span + recency pad {22,23}); B pads
        (+{23}).  Both land on exactly global+6+local tokens."""
        method = self._make_method(uniform_budget=6)
        nk_a, nv_a, k_a, v_a = self._run(method, self.A_SEEDS)
        self._assert_retained(nk_a, nv_a, k_a, v_a, [2, 3, 4, 5, 22, 23])
        nk_b, nv_b, k_b, v_b = self._run(method, self.B_SEEDS)
        self._assert_retained(nk_b, nv_b, k_b, v_b, self.B_MIDDLE + [23])

    def test_uniform_budget_clamped_to_middle_length(self):
        """A budget larger than the middle retains the whole middle (uniform
        across layers because n_middle is layer-independent)."""
        method = self._make_method(uniform_budget=10_000)
        nk_a, nv_a, k_a, v_a = self._run(method, self.A_SEEDS)
        self._assert_retained(nk_a, nv_a, k_a, v_a, list(range(self.N_MID)))

    def test_uniform_budget_must_be_positive(self):
        method = self._make_method(uniform_budget=0)
        with self.assertRaises(ValueError):
            self._run(method, self.A_SEEDS)

    def test_shrink_candidate_pool_matches_global_unique_under_ties(self):
        """Under tied frequencies at the recall_clip boundary, the shrink path
        must draw ONLY from the seeds the layer's own clip actually kept.

        torch.topk and a stable argsort break ties differently (e.g. counts
        [1,3,3,3,1,1] with k=4: topk keeps {9,17,25,33}, stable order keeps
        {9,17,25,3}), so _shrink_selection must re-use _global_unique's exact
        topk clip — a re-derived ranking can re-introduce a clipped-away seed
        and silently drop a kept one (the pre-fix behavior at target=8)."""
        method = self._make_method(span_size=2, recall_clip=4)
        # seeds: 3→1, 9→3, 17→3, 25→3, 33→1, 41→1 selections
        flat = torch.tensor([3, 9, 9, 9, 17, 17, 17, 25, 25, 25, 33, 41])
        topk_idx = flat.view(1, 1, 1, -1)
        n_middle = 64

        kept = method._global_unique(topk_idx, 4)
        kept_exp = set(method._expand_spans(kept, n_middle).tolist())
        shrunk = method._shrink_selection(topk_idx, target=8, n_middle=n_middle)

        self.assertEqual(shrunk.numel(), 8)
        recency_floor = max(kept_exp)
        for i in shrunk.tolist():
            self.assertTrue(
                i in kept_exp or i > recency_floor,
                f"index {i} comes from a seed the recall_clip discarded",
            )

    def test_shrink_respects_recall_clip(self):
        """recall_clip=1 keeps only the most-frequent seed; the uniform target
        from a previous layer must not re-introduce clipped-away seeds.  Seed
        4 carries weight 5.0 > 4.0, and both seeds are picked by every query
        (equal counts) — the stable tie-break keeps the lower index, 4."""
        method = self._make_method(recall_clip=1, uniform_budget=5)
        nk_a, nv_a, k_a, v_a = self._run(method, self.A_SEEDS)
        # clip → seed 4 only → {2,3,4,5}; pad 1 by recency → {23}.
        self._assert_retained(nk_a, nv_a, k_a, v_a, [2, 3, 4, 5, 23])


class TestReAttentionKernelDispatch(unittest.TestCase):
    """The fused einsum-topk kernel is wired in with a dense fallback.

    These tests cover the *dispatch logic* and *plumbing* on CPU (no GPU),
    plus a GPU-gated end-to-end equivalence check.
    """

    def _make_method(self, **kwargs):
        from eval_harness.prefill_methods.reattention import ReAttentionMethod

        defaults = dict(global_size=4, local_size=4, mid_size=4, span_size=0)
        defaults.update(kwargs)
        return ReAttentionMethod(**defaults)

    # -- _should_use_kernel logic -------------------------------------------

    def test_kernel_disabled_when_off(self):
        m = self._make_method(use_triton_kernel="off")
        q = torch.randn(1, 4, 256, 128)
        k = torch.randn(1, 2, 256, 128)
        self.assertFalse(m._should_use_kernel(q, k, n_middle=256))

    def test_kernel_skipped_on_cpu(self):
        """auto mode → CPU tensors never trigger the kernel."""
        m = self._make_method(use_triton_kernel="auto")
        q = torch.randn(1, 4, 256, 128)  # head_dim 128, n_middle %128, mid_size 4
        k = torch.randn(1, 2, 256, 128)
        # CPU tensors → is_cuda False → no kernel.
        self.assertFalse(m._should_use_kernel(q, k, n_middle=256))

    def test_kernel_gate_rejects_bad_constraints(self):
        from eval_harness.prefill_methods import reattention as reatt_mod

        # Pretend the kernel is importable and tensors are on CUDA so we can
        # exercise the constraint checks without a GPU.
        m = self._make_method(use_triton_kernel="auto")

        class _FakeCudaTensor:
            def __init__(self, shape):
                self.shape = shape
                self.is_cuda = True

        def gate(mid_size, D, n_middle):
            m.mid_size = mid_size
            q = _FakeCudaTensor((1, 4, 256, D))
            k = _FakeCudaTensor((1, 2, n_middle, D))
            return m._should_use_kernel(q, k, n_middle)

        with patch.object(reatt_mod, "_einsum_topk_func", lambda *a, **k: None):
            self.assertTrue(gate(mid_size=4, D=128, n_middle=256))   # all good
            self.assertTrue(gate(mid_size=1, D=128, n_middle=128))   # topk=1 ok
            self.assertFalse(gate(mid_size=2, D=128, n_middle=256))  # bad topk
            self.assertFalse(gate(mid_size=4, D=64, n_middle=256))   # bad head_dim
            self.assertFalse(gate(mid_size=4, D=128, n_middle=200))  # n_middle %128

    def test_force_raises_when_kernel_unavailable(self):
        from eval_harness.prefill_methods import reattention as reatt_mod

        m = self._make_method(use_triton_kernel="force")
        q = torch.randn(1, 4, 256, 128)
        k = torch.randn(1, 2, 256, 128)
        with patch.object(reatt_mod, "_einsum_topk_func", None):
            with self.assertRaises(RuntimeError):
                m._should_use_kernel(q, k, n_middle=256)

    def test_force_raises_on_unmet_constraints(self):
        from eval_harness.prefill_methods import reattention as reatt_mod

        m = self._make_method(use_triton_kernel="force")
        q = torch.randn(1, 4, 256, 64)  # head_dim 64 → unmet, CPU → unmet
        k = torch.randn(1, 2, 256, 64)
        with patch.object(reatt_mod, "_einsum_topk_func", lambda *a, **k: None):
            with self.assertRaises(RuntimeError):
                m._should_use_kernel(q, k, n_middle=256)

    # -- 128-alignment of the middle region ---------------------------------

    def test_effective_local_size_alignment(self):
        m = self._make_method(global_size=32, local_size=4096, align_local_to_128=True)
        # S chosen so raw middle = S - 32 - 4096 is not a multiple of 128.
        S = 32 + 4096 + 1000  # raw_mid = 1000, 1000 % 128 = 104
        eff_local = m._effective_local_size(S)
        n_middle = S - 32 - eff_local
        self.assertEqual(n_middle % 128, 0)
        self.assertLessEqual(eff_local, 4096)  # local shrank
        self.assertEqual(eff_local, 4096 - ((-1000) % 128))

    def test_effective_local_size_noop_when_disabled(self):
        m = self._make_method(local_size=4096, align_local_to_128=False)
        self.assertEqual(m._effective_local_size(99999), 4096)

    def test_alignment_makes_n_middle_divisible(self):
        """With alignment on, the hook's middle region is 128-divisible.

        (Uses a realistic local_size >= 128 so the shrink is feasible — the
        reference always runs with local_size in the thousands.)"""
        torch.manual_seed(0)
        B, H_kv, D = 1, 1, 8
        global_size, local_size = 4, 256
        S = global_size + local_size + 300  # raw_mid = 300, not a multiple of 128
        module = _FakeAttnModule(hidden_dim=2 * D, num_heads=2, head_dim=D, num_kv_heads=H_kv)
        keys = torch.randn(B, H_kv, S, D)
        values = torch.randn(B, H_kv, S, D)
        hidden = torch.randn(B, S, 2 * D)
        cos, sin = _identity_pos_emb(B, S, D)

        m = self._make_method(global_size=global_size, local_size=local_size,
                              mid_size=4, span_size=0, align_local_to_128=True)
        eff_local = m._effective_local_size(S)
        n_middle = S - global_size - eff_local
        self.assertEqual(n_middle % 128, 0)
        self.assertLess(eff_local, local_size)  # local window shrank
        # Still produces a valid pruned cache.
        result = m.prefill_forward_hook(
            module, hidden, keys, values, {"position_embeddings": (cos, sin)},
        )
        self.assertIsNotNone(result)

    # -- plumbing: kernel output → selection (mocked kernel, no GPU) ---------

    def test_kernel_path_plumbing_matches_dense(self):
        """When the (mocked) kernel returns the same per-query top-k as the
        dense path, the full hook output is identical — validating that the
        kernel's int32 [B,H_q,q_len,topk] output, query-padding drop, and
        downstream unique/span wiring are all correct."""
        from eval_harness.prefill_methods import reattention as reatt_mod

        torch.manual_seed(5)
        B, H_kv, S, D = 1, 2, 80, 8
        num_heads = 4
        module = _FakeAttnModule(
            hidden_dim=num_heads * D, num_heads=num_heads, head_dim=D,
            num_kv_heads=H_kv, seed=6,
        )
        keys = torch.randn(B, H_kv, S, D)
        values = torch.randn(B, H_kv, S, D)
        hidden = torch.randn(B, S, num_heads * D)
        cos, sin = _identity_pos_emb(B, S, D)
        kwargs = {"position_embeddings": (cos, sin)}
        params = dict(global_size=4, local_size=8, mid_size=4, span_size=2)

        # Dense-only reference output.
        dense_method = self._make_method(use_triton_kernel="off", **params)
        ref_keys, ref_values = dense_method.prefill_forward_hook(
            module, hidden, keys.clone(), values.clone(), kwargs,
        )

        # A fake "kernel" that reproduces the dense top-k in the kernel's
        # output contract: int32, shape [B, H_q, q_len_pad, topk], indices into
        # the middle slice.  It receives padded q (q_len padded to %128).
        def fake_kernel(q, k, topk):
            # q: [B, H_q, q_len_pad, D]; k: [B, H_kv, n_middle, D]
            Bk, Hq, qpad, Dd = q.shape
            Hkv = k.shape[1]
            n_rep = max(1, Hq // Hkv)
            kk = k[:, :, None].expand(Bk, Hkv, n_rep, k.shape[2], Dd).reshape(Bk, Hq, k.shape[2], Dd)
            scores = torch.einsum("bhqd,bhmd->bhqm", q.float(), kk.float())
            _, idx = scores.topk(topk, dim=-1)
            return idx.to(torch.int32)

        # Force the kernel path: patch the func and the gate.
        kernel_method = self._make_method(use_triton_kernel="auto", **params)
        with patch.object(reatt_mod, "_einsum_topk_func", fake_kernel), \
                patch.object(type(kernel_method), "_should_use_kernel", lambda self, q, k, n: True):
            out_keys, out_values = kernel_method.prefill_forward_hook(
                module, hidden, keys.clone(), values.clone(), kwargs,
            )

        self.assertEqual(out_keys.shape, ref_keys.shape)
        self.assertTrue(torch.equal(out_keys, ref_keys))
        self.assertTrue(torch.equal(out_values, ref_values))

    def test_kernel_topk_pads_and_drops_query_rows(self):
        """_kernel_topk pads q_len to %128, calls the kernel, and slices the
        result back to the true q_len."""
        from eval_harness.prefill_methods import reattention as reatt_mod

        m = self._make_method(mid_size=4)
        B, H_q, q_len, D = 1, 2, 50, 128  # q_len 50 → padded to 128
        n_middle = 256
        raw_q = torch.randn(B, H_q, q_len, D)
        recall_k_mid = torch.randn(B, 1, n_middle, D)

        seen = {}

        def fake_kernel(q, k, topk):
            seen["q_shape"] = tuple(q.shape)
            return torch.zeros(q.shape[0], q.shape[1], q.shape[2], topk, dtype=torch.int32)

        with patch.object(reatt_mod, "_einsum_topk_func", fake_kernel):
            out = m._kernel_topk(raw_q, recall_k_mid)

        self.assertEqual(seen["q_shape"][2], 128)        # padded to 128
        self.assertEqual(out.shape, (B, H_q, q_len, 4))  # sliced back to 50
        self.assertEqual(out.dtype, torch.long)

    # -- GPU end-to-end equivalence (skipped without CUDA + triton) ----------

    @unittest.skipUnless(_cuda_and_triton_available(), "requires CUDA + triton")
    def test_kernel_matches_dense_on_gpu(self):
        """Real kernel vs dense selection produce near-identical kept sets."""
        torch.manual_seed(0)
        device = "cuda"
        B, H_kv, D = 1, 4, 128
        global_size, local_size, mid_size = 128, 128, 4
        n_middle = 256
        S = global_size + n_middle + local_size
        num_heads = 8

        module = _FakeAttnModule(
            hidden_dim=num_heads * D, num_heads=num_heads, head_dim=D,
            num_kv_heads=H_kv, seed=1,
        ).to(device).to(torch.bfloat16)
        keys = torch.randn(B, H_kv, S, D, device=device, dtype=torch.bfloat16)
        values = torch.randn(B, H_kv, S, D, device=device, dtype=torch.bfloat16)
        hidden = torch.randn(B, S, num_heads * D, device=device, dtype=torch.bfloat16)
        cos, sin = _identity_pos_emb(B, S, D)
        cos, sin = cos.to(device, torch.bfloat16), sin.to(device, torch.bfloat16)
        kwargs = {"position_embeddings": (cos, sin)}
        params = dict(global_size=global_size, local_size=local_size,
                      mid_size=mid_size, span_size=0)

        from eval_harness.prefill_methods.reattention import ReAttentionMethod
        dense = ReAttentionMethod(use_triton_kernel="off", **params)
        kern = ReAttentionMethod(use_triton_kernel="force", **params)

        dk, _ = dense.prefill_forward_hook(module, hidden, keys.clone(), values.clone(), kwargs)
        kk, _ = kern.prefill_forward_hook(module, hidden, keys.clone(), values.clone(), kwargs)

        # Both retain global + local exactly; middle sets overlap heavily.
        self.assertEqual(dk.shape[1], kk.shape[1])
        # Jaccard overlap of kept counts should be high (tie-breaking/precision
        # differences aside).
        self.assertGreater(kk.shape[2], global_size + local_size)


# ======================================================================
# DCA tests
# ======================================================================


import torch.nn.functional as F

from eval_harness.kernels.dca_flash import (
    attention_with_lse,
    flash_attn_with_lse,
    get_mscale,
    merge_attn_outputs,
)


# ----------------------------------------------------------------------
# DCA kernel-layer tests (flash-attn-with-LSE substitute + LSE merge)
# ----------------------------------------------------------------------


class TestDCAKernels(unittest.TestCase):
    def test_get_mscale(self):
        self.assertEqual(get_mscale(1.0), 1.0)
        self.assertEqual(get_mscale(0.5), 1.0)
        self.assertAlmostEqual(get_mscale(4.0, coeff=0.05), 0.05 * math.log(4.0) + 1.0)
        self.assertAlmostEqual(get_mscale(4.0, coeff=0.1), 0.1 * math.log(4.0) + 1.0)

    def test_attention_with_lse_shapes(self):
        B, H, S, D = 2, 4, 6, 16
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)
        out, lse = attention_with_lse(q, k, v, causal=True)
        self.assertEqual(out.shape, (B, H, S, D))
        self.assertEqual(lse.shape, (B, H, S))

    def test_attention_with_lse_matches_sdpa(self):
        """Output matches torch SDPA; LSE matches logsumexp of scaled scores."""
        torch.manual_seed(0)
        B, H, S, D = 1, 2, 8, 16
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)

        out, lse = attention_with_lse(q, k, v, causal=True)
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

        # LSE reference: logsumexp over (masked) scaled scores.
        scale = 1.0 / math.sqrt(D)
        scores = (q @ k.transpose(-1, -2)) * scale
        mask = torch.ones(S, S, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(mask, float("-inf"))
        ref_lse = torch.logsumexp(scores, dim=-1)
        torch.testing.assert_close(lse, ref_lse, atol=1e-5, rtol=1e-5)

    def test_attention_with_lse_gqa(self):
        B, H_q, H_kv, S, D = 1, 8, 2, 6, 16
        q = torch.randn(B, H_q, S, D)
        k = torch.randn(B, H_kv, S, D)
        v = torch.randn(B, H_kv, S, D)
        out, lse = attention_with_lse(q, k, v, causal=True)
        self.assertEqual(out.shape, (B, H_q, S, D))
        # Manual repeat → same result.
        kr = k[:, :, None].expand(B, H_kv, H_q // H_kv, S, D).reshape(B, H_q, S, D)
        vr = v[:, :, None].expand(B, H_kv, H_q // H_kv, S, D).reshape(B, H_q, S, D)
        out2, _ = attention_with_lse(q, kr, vr, causal=True)
        torch.testing.assert_close(out, out2, atol=1e-5, rtol=1e-5)

    def test_attention_with_lse_noncausal_bottom_right(self):
        """A single query (S_q=1) against many keys attends to all of them."""
        B, H, D = 1, 2, 16
        q = torch.randn(B, H, 1, D)
        k = torch.randn(B, H, 10, D)
        v = torch.randn(B, H, 10, D)
        out_c, _ = attention_with_lse(q, k, v, causal=True)
        out_nc, _ = attention_with_lse(q, k, v, causal=False)
        # Bottom-right causal with S_q=1 leaves all keys visible → identical.
        torch.testing.assert_close(out_c, out_nc, atol=1e-6, rtol=0)

    def test_lse_merge_equals_single_softmax(self):
        """THE key invariant: LSE-merging per-group attentions (same query) ==
        a single softmax over the concatenated key groups."""
        torch.manual_seed(99)
        B, H, S_q, D = 1, 2, 5, 16
        q = torch.randn(B, H, S_q, D)
        k = torch.randn(B, H, 32, D)
        v = torch.randn(B, H, 32, D)
        k1, v1 = k[:, :, :10], v[:, :, :10]
        k2, v2 = k[:, :, 10:20], v[:, :, 10:20]
        k3, v3 = k[:, :, 20:], v[:, :, 20:]

        comps = [
            attention_with_lse(q, k1, v1, causal=False),
            attention_with_lse(q, k2, v2, causal=False),
            attention_with_lse(q, k3, v3, causal=False),
        ]
        merged = merge_attn_outputs(comps, decoding=True)
        full, _ = attention_with_lse(q, k, v, causal=False)
        torch.testing.assert_close(merged, full, atol=1e-4, rtol=1e-4)

    def test_merge_prefill_concatenates_chunks(self):
        """Prefill merge: per-chunk merge then concat along the seq dim."""
        B, H, D = 1, 2, 16
        # chunk A: 1 component over 3 queries; chunk B: 2 components over 2 queries.
        a = attention_with_lse(torch.randn(B, H, 3, D), torch.randn(B, H, 3, D), torch.randn(B, H, 3, D), causal=True)
        b1 = attention_with_lse(torch.randn(B, H, 2, D), torch.randn(B, H, 4, D), torch.randn(B, H, 4, D), causal=False)
        b2 = attention_with_lse(torch.randn(B, H, 2, D), torch.randn(B, H, 4, D), torch.randn(B, H, 4, D), causal=False)
        out = merge_attn_outputs([[a], [b1, b2]], decoding=False)
        self.assertEqual(out.shape, (B, H, 5, D))  # 3 + 2

    def test_flash_attn_with_lse_off_uses_torch(self):
        q = torch.randn(1, 2, 4, 16)
        k = torch.randn(1, 2, 4, 16)
        v = torch.randn(1, 2, 4, 16)
        out_off, lse_off = flash_attn_with_lse(q, k, v, causal=True, backend="off")
        out_torch, lse_torch = attention_with_lse(q, k, v, causal=True)
        torch.testing.assert_close(out_off, out_torch)
        torch.testing.assert_close(lse_off, lse_torch)

    def test_flash_attn_with_lse_auto_on_cpu_uses_torch(self):
        """auto backend on CPU (no CUDA) transparently uses the torch path."""
        q = torch.randn(1, 2, 4, 16)
        k = torch.randn(1, 2, 4, 16)
        v = torch.randn(1, 2, 4, 16)
        out, _ = flash_attn_with_lse(q, k, v, causal=True, backend="auto")
        ref, _ = attention_with_lse(q, k, v, causal=True)
        torch.testing.assert_close(out, ref)

    def test_flash_attn_with_lse_force_raises_without_flash(self):
        from eval_harness.kernels import dca_flash

        if dca_flash.flash_attn_available():
            self.skipTest("flash-attn is available; cannot test the unavailable path")
        q = torch.randn(1, 2, 4, 16)
        with self.assertRaises(RuntimeError):
            flash_attn_with_lse(q, q, q, causal=True, backend="force")


# ======================================================================
# DCA method tests — faithful 3-component Dual Chunk Attention
# ======================================================================


class _FakeDCAAttn(nn.Module):
    """Minimal attention module exposing q/k/v/o projections + head config."""

    def __init__(self, hidden_dim, num_heads, head_dim, num_kv_heads, layer_idx=0, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.num_heads = num_heads
        self.num_key_value_heads = num_kv_heads
        self.head_dim = head_dim
        self.scaling = 1.0 / math.sqrt(head_dim)
        self.layer_idx = layer_idx
        self.config = None
        self.q_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_dim, bias=False)


class _FakeDynamicCache:
    """DynamicCache-like: update concatenates per layer on the seq dim."""

    def __init__(self):
        self.k = {}
        self.v = {}

    def update(self, key, value, layer_idx, cache_kwargs=None):
        if layer_idx in self.k:
            self.k[layer_idx] = torch.cat([self.k[layer_idx], key], dim=2)
            self.v[layer_idx] = torch.cat([self.v[layer_idx], value], dim=2)
        else:
            self.k[layer_idx] = key
            self.v[layer_idx] = value
        return self.k[layer_idx], self.v[layer_idx]

    def get_seq_length(self, layer_idx=0):
        return self.k[layer_idx].shape[2] if layer_idx in self.k else 0


class TestDCAMethod(unittest.TestCase):
    def _make_method(self, **kwargs):
        from eval_harness.prefill_methods.dca import DCAMethod

        defaults = dict(chunk_size=12, local_window=4, pretraining_length=10_000, use_flash_attn="off")
        defaults.update(kwargs)
        m = DCAMethod(**defaults)
        m._inv_freq = 1.0 / (10000.0 ** (torch.arange(0, 8, 2).float() / 8))  # D=8
        return m

    # -- config / derived ----------------------------------------------------

    def test_chunk_len_and_backend(self):
        m = self._make_method(chunk_size=12, local_window=4, use_flash_attn="off")
        self.assertEqual(m.chunk_len, 8)
        self.assertEqual(m._backend, "torch")
        self.assertEqual(self._make_method(use_flash_attn="auto")._backend, "auto")
        self.assertEqual(self._make_method(use_flash_attn="force")._backend, "force")

    def test_resolve_heads(self):
        from eval_harness.prefill_methods.dca import DCAMethod

        attn = _FakeDCAAttn(hidden_dim=32, num_heads=4, head_dim=8, num_kv_heads=2)
        self.assertEqual(DCAMethod._resolve_heads(attn), (4, 2, 8))

    def test_abs_positions_from_cache_position(self):
        from eval_harness.prefill_methods.dca import DCAMethod

        cp = torch.arange(5, 10)
        pos = DCAMethod._abs_positions(cp, None, None, 0, 5, torch.device("cpu"))
        torch.testing.assert_close(pos, torch.arange(5, 10))

    def test_abs_positions_from_cache_length(self):
        from eval_harness.prefill_methods.dca import DCAMethod

        cache = _FakeDynamicCache()
        cache.update(torch.randn(1, 1, 7, 8), torch.randn(1, 1, 7, 8), 0)
        pos = DCAMethod._abs_positions(None, None, cache, 0, 3, torch.device("cpu"))
        torch.testing.assert_close(pos, torch.arange(7, 10))

    # -- RoPE position schemes ----------------------------------------------

    def test_rope_cyclic_key_positions(self):
        """Keys at positions differing by a multiple of chunk_len get the same
        RoPE (the cyclic-key property)."""
        m = self._make_method(chunk_size=12, local_window=4)  # chunk_len=8
        D = 8
        x = torch.randn(1, 1, 24, D)
        positions = torch.arange(24)
        rotated = m._rope(x, positions % m.chunk_len, mscale=1.0)
        # position 0 and 8 and 16 share the same cyclic position → but x differs.
        # Instead verify via constant x: same input → same rotation.
        ones = torch.ones(1, 1, 24, D)
        rot = m._rope(ones, positions % m.chunk_len, mscale=1.0)
        torch.testing.assert_close(rot[:, :, 0], rot[:, :, 8], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(rot[:, :, 0], rot[:, :, 16], atol=1e-5, rtol=1e-5)
        self.assertFalse(torch.allclose(rot[:, :, 0], rot[:, :, 1]))

    def test_succ_and_inter_position_formulas(self):
        m = self._make_method(chunk_size=12, local_window=4)  # chunk_len=8, chunk_size=12
        cl = m.chunk_len
        abs_pos = torch.arange(20)
        intra = abs_pos % cl
        succ = (intra + cl).clamp(max=m.chunk_size)
        inter_scalar = min(2 * cl - 1, m.chunk_size)
        # succ in [chunk_len, chunk_size]; clamped where intra+cl > chunk_size.
        self.assertEqual(int(succ.min()), cl)
        self.assertEqual(int(succ.max()), m.chunk_size)
        self.assertEqual(inter_scalar, 12)  # min(15, 12)

    # -- reduce-to-causal invariant -----------------------------------------

    def test_prefill_reduces_to_standard_attention_single_chunk(self):
        """When kv_len <= chunk_len, DCA prefill == standard causal attention
        (cyclic positions equal absolute positions)."""
        m = self._make_method(chunk_size=100, local_window=20)  # chunk_len=80
        B, H, S, D = 1, 2, 16, 8  # S=16 < chunk_len=80
        torch.manual_seed(1)
        raw_q = torch.randn(B, H, S, D)
        raw_k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)
        pos = torch.arange(S)

        # DCA query/key rotations (cyclic = absolute here).
        q_intra = m._rope(raw_q, pos % m.chunk_len, 1.0)
        q_succ = m._rope(raw_q, (pos % m.chunk_len + m.chunk_len).clamp(max=m.chunk_size), 1.0)
        q_inter = m._rope(raw_q, torch.full((S,), min(2 * m.chunk_len - 1, m.chunk_size)), 1.0)
        k_cyc = m._rope(raw_k, pos % m.chunk_len, 1.0)
        scale = 1.0 / math.sqrt(D)
        out_dca = m._dca_prefill_attention(q_intra, q_succ, q_inter, k_cyc, v, scale)

        # Standard causal attention with absolute-position RoPE.
        q_std = m._rope(raw_q, pos, 1.0)
        k_std = m._rope(raw_k, pos, 1.0)
        out_std, _ = attention_with_lse(q_std, k_std, v, causal=True, scale=scale)
        torch.testing.assert_close(out_dca, out_std, atol=1e-4, rtol=1e-4)

    # -- multi-chunk shape + determinism ------------------------------------

    def test_prefill_multichunk_shape_and_determinism(self):
        m = self._make_method(chunk_size=12, local_window=4)  # chunk_len=8
        B, H, S, D = 1, 2, 30, 8  # spans multiple chunks
        torch.manual_seed(2)
        raw_q, raw_k, v = (torch.randn(B, H, S, D) for _ in range(3))
        pos = torch.arange(S)
        cl, cs = m.chunk_len, m.chunk_size
        q_intra = m._rope(raw_q, pos % cl, 1.0)
        q_succ = m._rope(raw_q, (pos % cl + cl).clamp(max=cs), 1.0)
        q_inter = m._rope(raw_q, torch.full((S,), min(2 * cl - 1, cs)), 1.0)
        k_cyc = m._rope(raw_k, pos % cl, 1.0)
        scale = 1.0 / math.sqrt(D)

        out1 = m._dca_prefill_attention(q_intra, q_succ, q_inter, k_cyc, v, scale)
        out2 = m._dca_prefill_attention(q_intra, q_succ, q_inter, k_cyc, v, scale)
        self.assertEqual(out1.shape, (B, H, S, D))
        self.assertTrue(torch.equal(out1, out2))  # deterministic
        self.assertFalse(torch.isnan(out1).any())

    def test_decode_attention_shape(self):
        m = self._make_method(chunk_size=12, local_window=4)  # chunk_len=8
        B, H, D = 1, 2, 8
        kv_len = 20  # chunk_num = 19//8 = 2 → all three components
        torch.manual_seed(3)
        raw_q = torch.randn(B, H, 1, D)
        k_cyc = torch.randn(B, H, kv_len, D)
        v = torch.randn(B, H, kv_len, D)
        pos = torch.tensor([kv_len - 1])
        cl, cs = m.chunk_len, m.chunk_size
        q_intra = m._rope(raw_q, pos % cl, 1.0)
        q_succ = m._rope(raw_q, (pos % cl + cl).clamp(max=cs), 1.0)
        q_inter = m._rope(raw_q, torch.full((1,), min(2 * cl - 1, cs)), 1.0)
        out = m._dca_decode_attention(q_intra, q_succ, q_inter, k_cyc, v, kv_len, 1.0 / math.sqrt(D))
        self.assertEqual(out.shape, (B, H, 1, D))

    # -- end-to-end forward (fake module + cache) ----------------------------

    def _reference_standard_forward(self, m, attn, hidden, positions):
        """Standard Llama attention with absolute-position RoPE (the model DCA
        must reduce to for a single-chunk sequence)."""
        B, S, _ = hidden.shape
        nH, nKV, D = attn.num_heads, attn.num_key_value_heads, attn.head_dim
        q = attn.q_proj(hidden).view(B, S, nH, D).transpose(1, 2)
        k = attn.k_proj(hidden).view(B, S, nKV, D).transpose(1, 2)
        v = attn.v_proj(hidden).view(B, S, nKV, D).transpose(1, 2)
        q = m._rope(q, positions, 1.0)
        k = m._rope(k, positions, 1.0)
        out, _ = attention_with_lse(q, k, v, causal=True, scale=attn.scaling)
        out = out.transpose(1, 2).reshape(B, S, nH * D)
        return attn.o_proj(out)

    def test_forward_short_seq_equals_standard(self):
        """The replacement forward on a single-chunk sequence equals standard
        attention end-to-end (incl. q/k/v/o projections)."""
        m = self._make_method(chunk_size=200, local_window=40)  # chunk_len=160
        attn = _FakeDCAAttn(hidden_dim=32, num_heads=4, head_dim=8, num_kv_heads=2, seed=7)
        B, S = 1, 24  # < chunk_len
        hidden = torch.randn(B, S, 32)
        dca_forward = m._make_dca_forward(attn, layer_idx=0)
        cache = _FakeDynamicCache()
        out, w = dca_forward(
            hidden_states=hidden, position_embeddings=None, attention_mask=None,
            past_key_values=cache, cache_position=torch.arange(S),
        )
        self.assertIsNone(w)
        ref = self._reference_standard_forward(m, attn, hidden, torch.arange(S))
        torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)

    def test_forward_prefill_then_decode_runs(self):
        """Long prefill (multi-chunk) followed by a decode step produces
        correctly-shaped outputs and stores cyclic-rotated keys in the cache."""
        m = self._make_method(chunk_size=12, local_window=4)  # chunk_len=8
        attn = _FakeDCAAttn(hidden_dim=32, num_heads=4, head_dim=8, num_kv_heads=2, seed=9)
        B, S = 1, 30
        hidden = torch.randn(B, S, 32)
        dca_forward = m._make_dca_forward(attn, layer_idx=0)
        cache = _FakeDynamicCache()

        out_prefill, _ = dca_forward(
            hidden_states=hidden, past_key_values=cache, cache_position=torch.arange(S),
        )
        self.assertEqual(out_prefill.shape, (B, S, 32))
        self.assertEqual(cache.get_seq_length(0), S)  # all keys cached

        # One decode token at absolute position S.
        dec_hidden = torch.randn(B, 1, 32)
        out_dec, _ = dca_forward(
            hidden_states=dec_hidden, past_key_values=cache, cache_position=torch.tensor([S]),
        )
        self.assertEqual(out_dec.shape, (B, 1, 32))
        self.assertEqual(cache.get_seq_length(0), S + 1)
        self.assertFalse(torch.isnan(out_dec).any())

    # -- multi-token decode (question pass) ----------------------------------

    def _decode_block_queries(self, m, raw_q, abs_pos):
        """The three query rotations dca_forward computes for a decode block."""
        cl, cs = m.chunk_len, m.chunk_size
        intra_pos = abs_pos % cl
        q_intra = m._rope(raw_q, intra_pos, 1.0)
        q_succ = m._rope(raw_q, (intra_pos + cl).clamp(max=cs), 1.0)
        q_inter = m._rope(
            raw_q, torch.full((abs_pos.numel(),), min(2 * cl - 1, cs)), 1.0,
        )
        return q_intra, q_succ, q_inter

    def test_decode_multitoken_straddle_finite_and_causal(self):
        """A multi-token decode block STRADDLING a chunk boundary must be
        finite and causal.

        Regression: the LSE decomposition assigned every query the components
        of the LAST key's chunk, so with chunk_len=8, kv=17 (boundary at 16)
        the three pre-boundary queries (abs 13..15) got a fully-masked intra
        slice → softmax over nothing → NaN, while succ/inter leaked future
        in-block keys.  The reference decode branch (concatenated scores + one
        causal mask + softmax) has neither problem."""
        m = self._make_method(chunk_size=12, local_window=4)  # chunk_len=8
        B, H, D = 1, 2, 8
        kv_len, q_len = 17, 4  # queries at abs 13..16; boundary 16 in (13,16]
        torch.manual_seed(5)
        raw_q = torch.randn(B, H, q_len, D)
        k_cyc = torch.randn(B, H, kv_len, D)
        v = torch.randn(B, H, kv_len, D)
        abs_pos = torch.arange(kv_len - q_len, kv_len)
        q_intra, q_succ, q_inter = self._decode_block_queries(m, raw_q, abs_pos)
        scale = 1.0 / math.sqrt(D)

        out = m._dca_decode_attention(q_intra, q_succ, q_inter, k_cyc, v, kv_len, scale)
        self.assertEqual(out.shape, (B, H, q_len, D))
        self.assertTrue(torch.isfinite(out).all())  # was NaN before the fix

        # Causality: perturbing the LAST in-block key/value must not change
        # any earlier query's output (it is that query's future).
        k2, v2 = k_cyc.clone(), v.clone()
        k2[:, :, -1] += 10.0
        v2[:, :, -1] += 10.0
        out2 = m._dca_decode_attention(q_intra, q_succ, q_inter, k2, v2, kv_len, scale)
        torch.testing.assert_close(out2[:, :, :-1], out[:, :, :-1])
        self.assertFalse(torch.allclose(out2[:, :, -1], out[:, :, -1]))

    def test_decode_multitoken_matches_lse_path_when_not_straddling(self):
        """When the block lies inside one chunk, the reference concatenated-
        softmax branch must agree with the (faithful, single-chunk-assumption)
        LSE merge — a non-tautological consistency oracle between the two
        decode factorizations."""
        m = self._make_method(chunk_size=12, local_window=4)  # chunk_len=8
        B, H, D = 1, 2, 8
        kv_len, q_len = 20, 4  # queries at abs 16..19, all in chunk 2 ([16,24))
        torch.manual_seed(6)
        raw_q = torch.randn(B, H, q_len, D)
        k_cyc = torch.randn(B, H, kv_len, D)
        v = torch.randn(B, H, kv_len, D)
        abs_pos = torch.arange(kv_len - q_len, kv_len)
        q_intra, q_succ, q_inter = self._decode_block_queries(m, raw_q, abs_pos)
        scale = 1.0 / math.sqrt(D)
        cl = m.chunk_len
        cn = (kv_len - 1) // cl  # = 2

        out = m._dca_decode_attention(q_intra, q_succ, q_inter, k_cyc, v, kv_len, scale)

        lse_ref = merge_attn_outputs(
            [
                attention_with_lse(
                    q_intra, k_cyc[:, :, cl * cn:kv_len], v[:, :, cl * cn:kv_len],
                    causal=True, scale=scale,
                ),
                attention_with_lse(
                    q_succ, k_cyc[:, :, cl * (cn - 1):cl * cn], v[:, :, cl * (cn - 1):cl * cn],
                    causal=False, scale=scale,
                ),
                attention_with_lse(
                    q_inter, k_cyc[:, :, : cl * (cn - 1)], v[:, :, : cl * (cn - 1)],
                    causal=False, scale=scale,
                ),
            ],
            decoding=True,
        )
        torch.testing.assert_close(out, lse_ref, atol=1e-5, rtol=1e-5)

    def test_decode_multitoken_last_row_matches_single_step(self):
        """In the straddle case the LAST query defines the block's chunk_num,
        so its row must equal the (verified-faithful) q_len=1 LSE decode."""
        m = self._make_method(chunk_size=12, local_window=4)  # chunk_len=8
        B, H, D = 1, 2, 8
        kv_len, q_len = 17, 4
        torch.manual_seed(7)
        raw_q = torch.randn(B, H, q_len, D)
        k_cyc = torch.randn(B, H, kv_len, D)
        v = torch.randn(B, H, kv_len, D)
        abs_pos = torch.arange(kv_len - q_len, kv_len)
        q_intra, q_succ, q_inter = self._decode_block_queries(m, raw_q, abs_pos)
        scale = 1.0 / math.sqrt(D)

        out_block = m._dca_decode_attention(q_intra, q_succ, q_inter, k_cyc, v, kv_len, scale)
        out_single = m._dca_decode_attention(
            q_intra[:, :, -1:], q_succ[:, :, -1:], q_inter[:, :, -1:],
            k_cyc, v, kv_len, scale,
        )
        torch.testing.assert_close(out_block[:, :, -1:], out_single, atol=1e-5, rtol=1e-5)

    def test_context_manager_installs_and_restores_forward(self):
        """__call__ swaps self_attn.forward on full-attn layers and restores it."""
        from eval_harness.prefill_methods.dca import DCAMethod

        class FakeRotary(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("inv_freq", 1.0 / (10000.0 ** (torch.arange(0, 8, 2).float() / 8)))

        class FakeLayer(nn.Module):
            def __init__(self, idx):
                super().__init__()
                self.self_attn = _FakeDCAAttn(32, 4, 8, 2, layer_idx=idx)

        class FakeInner(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([FakeLayer(0), FakeLayer(1)])
                self.rotary_emb = FakeRotary()

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = FakeInner()

        model = FakeModel()
        method = DCAMethod(chunk_size=12, local_window=4, use_flash_attn="off")
        originals = [layer.self_attn.forward for layer in model.model.layers]
        with method(model):
            for layer in model.model.layers:
                self.assertNotIn(layer.self_attn.forward, originals)
        # Restored after exit.
        for layer, orig in zip(model.model.layers, originals):
            self.assertEqual(layer.self_attn.forward, orig)


# ======================================================================
# Pipeline integration tests (no model loading)
# ======================================================================


class _FakeCache:
    """Minimal cache mock for pipeline tests."""

    class _Layer:
        def __init__(self, s, d):
            self.keys = torch.randn(1, 2, s, d)
            self.values = torch.randn(1, 2, s, d)

    def __init__(self, n_layers=2, seq_len=16, dim=8):
        self.layers = [self._Layer(seq_len, dim) for _ in range(n_layers)]

    def get_seq_length(self, layer_idx=0):
        return self.layers[layer_idx].keys.shape[2]

    def __len__(self):
        return len(self.layers)


class TestResearchAdapterPrefillMethodWiring(unittest.TestCase):
    """Test that ResearchConfig.attention_method (door 2) is wired correctly.

    The resolver tries the new ``attention_methods`` registry first (DCA),
    then falls back to the legacy ``prefill_methods`` registry (reattention).
    ``none`` resolves to ``None`` (no method installed).
    """

    def test_build_attention_method_none(self):
        from eval_harness.research_adapter import ResearchConfig, ResearchAdapter

        cfg = ResearchConfig(attention_method="none")
        self.assertIsNone(ResearchAdapter._build_attention_method(cfg))

    def test_build_attention_method_reattention(self):
        from eval_harness.prefill_methods.reattention import ReAttentionMethod
        from eval_harness.research_adapter import ResearchConfig, ResearchAdapter

        cfg = ResearchConfig(
            attention_method="reattention",
            attention_method_kwargs={"global_size": 16, "mid_size": 4},
        )
        method = ResearchAdapter._build_attention_method(cfg)
        self.assertIsInstance(method, ReAttentionMethod)
        self.assertEqual(method.global_size, 16)
        self.assertEqual(method.mid_size, 4)

    def test_build_attention_method_dca(self):
        from eval_harness.attention_methods.dca import DCAMethod
        from eval_harness.research_adapter import ResearchConfig, ResearchAdapter

        cfg = ResearchConfig(
            attention_method="dca",
            attention_method_kwargs={"chunk_size": 4096},
            attention_phase="both",
        )
        method = ResearchAdapter._build_attention_method(cfg)
        self.assertIsInstance(method, DCAMethod)
        self.assertEqual(method.chunk_size, 4096)

    def test_unknown_attention_method_raises(self):
        from eval_harness.research_adapter import ResearchConfig, ResearchAdapter

        cfg = ResearchConfig(attention_method="nonexistent_xyz")
        with self.assertRaises(ValueError):
            ResearchAdapter._build_attention_method(cfg)


class TestPrefillMethodContextManager(unittest.TestCase):
    """Test the __call__ context manager hook registration."""

    def test_base_method_context_manager_is_noop(self):
        """Base method registers hooks but they are no-ops."""
        method = PrefillMethod()

        # Create a minimal fake model.
        class FakeAttn(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer_idx = 0
                self.is_sliding = False

        class FakeLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = FakeAttn()

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([FakeLayer()])

        class FakeOuter(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = FakeModel()

        model = FakeOuter()
        with method(model):
            # Should not raise.
            pass


# ======================================================================
# get_inv_freq / get_rotary_emb utility tests
# ======================================================================


class TestGetRotaryEmb(unittest.TestCase):
    def test_returns_none_for_no_rotary(self):
        class FakeModel:
            class model:
                pass

        self.assertIsNone(get_rotary_emb(FakeModel()))

    def test_finds_model_level_rotary(self):
        class FakeRotary(nn.Module):
            pass

        class FakeModel:
            class model:
                rotary_emb = FakeRotary()

        result = get_rotary_emb(FakeModel())
        self.assertIsInstance(result, FakeRotary)

    def test_finds_layer_level_rotary(self):
        class FakeRotary(nn.Module):
            pass

        class FakeAttn:
            rotary_emb = FakeRotary()

        class FakeLayer:
            self_attn = FakeAttn()

        class FakeModel:
            class model:
                layers = [FakeLayer()]

        result = get_rotary_emb(FakeModel())
        self.assertIsInstance(result, FakeRotary)


class TestGetInvFreq(unittest.TestCase):
    def test_returns_none_when_missing(self):
        class FakeModel:
            class model:
                pass

        self.assertIsNone(get_inv_freq(FakeModel()))

    def test_finds_inv_freq_attribute(self):
        class FakeRotary(nn.Module):
            def __init__(self):
                super().__init__()
                self.inv_freq = torch.tensor([1.0, 2.0])

        class FakeModel:
            class model:
                rotary_emb = FakeRotary()

        result = get_inv_freq(FakeModel())
        self.assertIsNotNone(result)
        torch.testing.assert_close(result, torch.tensor([1.0, 2.0]))

    def test_finds_inv_freq_as_buffer(self):
        class FakeRotary(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("inv_freq", torch.tensor([3.0, 4.0]))

        class FakeModel:
            class model:
                rotary_emb = FakeRotary()

        result = get_inv_freq(FakeModel())
        self.assertIsNotNone(result)
        torch.testing.assert_close(result, torch.tensor([3.0, 4.0]))


if __name__ == "__main__":
    unittest.main()
