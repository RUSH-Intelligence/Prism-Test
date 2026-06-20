"""Pins the Qwen3.5 partial-rotary handling in the KV re-rotation paths.

``finch._rerotate_keys`` and ``KeyRerotationSketch.rerotate_keys`` gather kept
keys and re-rotate them to contiguous positions. On Qwen3.5 (partial rotary:
``rotary_dim < head_dim``) the re-rotation trig (``new_cos``/``new_sin``) spans
only ``rotary_dim`` channels, so the old ``keys * new_cos`` one-liner broadcast
``head_dim`` against ``rotary_dim`` and crashed. The fix rotates only the first
``rotary_dim`` channels and passes the rest through, reducing to the original
behavior bit-identically when ``rotary_dim == head_dim``.

The existing finch test exercises only full rotary, so this pins the partial
branch (and key_rerotation, which had no Qwen3.5 coverage at all) directly.
"""
import unittest
from types import SimpleNamespace

import torch
from transformers.models.llama.modeling_llama import rotate_half

from eval_harness.kv_compression.compressors.finch_sketch import (
    _rerotate_keys as finch_rerotate_keys,
    _rerotate_cos_sin as finch_cos_sin,
)
from eval_harness.kv_compression.compressors.key_rerotation_sketch import KeyRerotationSketch

# (name, rerotate_keys fn, matching cos/sin builder)
REROTATORS = [
    ("finch", finch_rerotate_keys, finch_cos_sin),
    ("key_rerotation", KeyRerotationSketch.rerotate_keys, KeyRerotationSketch._rerotate_cos_sin),
]


def _module(head_dim, rotary_dim):
    """Fake module exposing only what the re-rotators read: head_dim + rotary_emb.inv_freq.

    inv_freq has ``rotary_dim // 2`` entries, so the built cos/sin span rotary_dim
    channels (== head_dim for full rotary, < head_dim for Qwen3.5 partial rotary).
    """
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, rotary_dim, 2, dtype=torch.float64) / rotary_dim))
    return SimpleNamespace(head_dim=head_dim, rotary_emb=SimpleNamespace(inv_freq=inv_freq))


def _old_rerotate(cos_sin_fn, module, indices, keys):
    """The pre-fix one-liner: full-width multiply (crashes on partial rotary)."""
    new_cos, new_sin = cos_sin_fn(keys, module.rotary_emb.inv_freq, indices)
    idx = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)
    g = keys.gather(2, idx).contiguous()
    return (g * new_cos) + (rotate_half(g) * new_sin)


class TestQwen35KeyRerotate(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.B, self.Hkv, self.S, self.nkept, self.head_dim = 1, 2, 12, 5, 16
        self.keys = torch.randn(self.B, self.Hkv, self.S, self.head_dim, dtype=torch.float64)
        idx = [torch.randperm(self.S)[: self.nkept].sort().values for _ in range(self.B * self.Hkv)]
        self.indices = torch.stack(idx).view(self.B, self.Hkv, self.nkept)

    def test_full_rotary_bit_identical(self):
        """rotary_dim == head_dim: the split form equals the old one-liner (non-hybrid unchanged)."""
        mod = _module(self.head_dim, self.head_dim)
        for name, fn, cs in REROTATORS:
            with self.subTest(method=name):
                ref = _old_rerotate(cs, mod, self.indices, self.keys)
                got = fn(mod, self.indices, self.keys)
                self.assertTrue(
                    torch.allclose(got, ref, atol=1e-12),
                    f"{name}: full-rotary not bit-identical (max {(got - ref).abs().max()})",
                )

    def test_partial_rotary_passthrough_and_no_crash(self):
        """rotary_dim < head_dim: passthrough channels untouched, shape kept; old form would crash."""
        rotary_dim = 8
        mod = _module(self.head_dim, rotary_dim)
        gathered = self.keys.gather(
            2, self.indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        ).contiguous()
        for name, fn, _cs in REROTATORS:
            with self.subTest(method=name):
                got = fn(mod, self.indices, self.keys)
                self.assertEqual(tuple(got.shape), tuple(gathered.shape))
                self.assertTrue(
                    torch.equal(got[..., rotary_dim:], gathered[..., rotary_dim:]),
                    f"{name}: passthrough channels were modified",
                )
        # The pre-fix one-liner broadcasts head_dim against rotary_dim -> RuntimeError.
        with self.assertRaises(RuntimeError):
            _old_rerotate(finch_cos_sin, mod, self.indices, self.keys)


if __name__ == "__main__":
    unittest.main()
