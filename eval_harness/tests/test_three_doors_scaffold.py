"""Step-1 scaffolding tests for the three-door research interface.

Pure logic only — no model weights.  Covers the phase/schedule/operation model
and the three per-door registries.  Behavior wiring (pipeline, ports) is tested
in later steps.
"""

import unittest

from eval_harness.attention_methods import (
    AttentionMethod,
    AttentionPhase,
    available_attention_methods,
    get_attention_method,
    register_attention_method,
)
from eval_harness.kv_compression import (
    CompressionOperation,
    CompressionSchedule,
    KVCompressor,
    available_kv_compressors,
    get_kv_compressor,
    register_kv_compressor,
)
from eval_harness.positional_methods import (
    PositionalMethod,
    available_positional_methods,
    get_positional_method,
    register_positional_method,
)


class TestAttentionPhase(unittest.TestCase):
    def test_coerce_from_string_and_enum(self):
        self.assertIs(AttentionPhase.coerce("prefill"), AttentionPhase.PREFILL)
        self.assertIs(AttentionPhase.coerce("BOTH"), AttentionPhase.BOTH)
        self.assertIs(AttentionPhase.coerce(AttentionPhase.DECODE), AttentionPhase.DECODE)

    def test_coerce_rejects_unknown(self):
        with self.assertRaises(ValueError):
            AttentionPhase.coerce("sometimes")

    def test_phase_predicates(self):
        self.assertTrue(AttentionPhase.PREFILL.active_in_prefill())
        self.assertFalse(AttentionPhase.PREFILL.active_in_decode())
        self.assertFalse(AttentionPhase.DECODE.active_in_prefill())
        self.assertTrue(AttentionPhase.DECODE.active_in_decode())
        self.assertTrue(AttentionPhase.BOTH.active_in_prefill())
        self.assertTrue(AttentionPhase.BOTH.active_in_decode())

    def test_base_method_coerces_phase_in_post_init(self):
        m = AttentionMethod(phase="prefill")
        self.assertIs(m.phase, AttentionPhase.PREFILL)

    def test_base_attention_forward_is_abstract(self):
        with self.assertRaises(NotImplementedError):
            AttentionMethod().attention_forward(
                module=None, layer_idx=0, hidden_states=_FakeHidden(), is_decode=False,
            )


class TestCompressionSchedule(unittest.TestCase):
    def test_coerce_single_and_list(self):
        self.assertEqual(
            CompressionSchedule.coerce_set("post_prefill"),
            frozenset({CompressionSchedule.POST_PREFILL}),
        )
        self.assertEqual(
            CompressionSchedule.coerce_set(["streaming", "decode"]),
            frozenset({CompressionSchedule.STREAMING, CompressionSchedule.DECODE}),
        )

    def test_coerce_rejects_unknown_and_empty(self):
        with self.assertRaises(ValueError):
            CompressionSchedule.coerce_set("whenever")
        with self.assertRaises(ValueError):
            CompressionSchedule.coerce_set([])

    def test_schedule_predicates(self):
        post = KVCompressor(schedule="post_prefill")
        self.assertTrue(post.fires_on_prefill)
        self.assertFalse(post.fires_on_decode)

        dec = KVCompressor(schedule="decode")
        self.assertFalse(dec.fires_on_prefill)
        self.assertTrue(dec.fires_on_decode)

        both = KVCompressor(schedule=["streaming", "decode"])
        self.assertTrue(both.fires_on_prefill)
        self.assertTrue(both.fires_on_decode)

    def test_operation_default_and_coercion(self):
        self.assertIs(KVCompressor().operation, CompressionOperation.EVICT)
        self.assertIs(
            KVCompressor(operation="quantize").operation,
            CompressionOperation.QUANTIZE,
        )

    def test_base_compress_is_abstract(self):
        with self.assertRaises(NotImplementedError):
            KVCompressor().compress(None, None, None, None, None, {})


class TestPositionalIdentity(unittest.TestCase):
    def test_identity_defaults(self):
        m = PositionalMethod()
        self.assertEqual(m.mscale, 1.0)
        sentinel = object()
        self.assertIs(m.compute_inv_freq(sentinel, seq_len=10), sentinel)
        self.assertIs(m.remap_position_ids(sentinel, seq_len=10), sentinel)


class TestRegistries(unittest.TestCase):
    def test_positional_none_resolves_to_identity(self):
        self.assertIsInstance(get_positional_method("none"), PositionalMethod)
        self.assertIsInstance(get_positional_method("native"), PositionalMethod)

    def test_attention_none_resolves_to_base(self):
        self.assertIs(type(get_attention_method("none")), AttentionMethod)

    def test_unknown_names_raise(self):
        with self.assertRaises(ValueError):
            get_positional_method("does_not_exist")
        with self.assertRaises(ValueError):
            get_attention_method("does_not_exist")
        with self.assertRaises(ValueError):
            get_kv_compressor("does_not_exist")

    def test_register_and_resolve_round_trip(self):
        @register_positional_method("_test_pos_door")
        class _Pos(PositionalMethod):
            pass

        @register_attention_method("_test_attn_door")
        class _Attn(AttentionMethod):
            pass

        @register_kv_compressor("_test_kv_door")
        class _Kv(KVCompressor):
            pass

        self.assertIsInstance(get_positional_method("_test_pos_door"), _Pos)
        self.assertIsInstance(get_attention_method("_test_attn_door"), _Attn)
        self.assertIsInstance(get_kv_compressor("_test_kv_door"), _Kv)

        self.assertIn("_test_pos_door", available_positional_methods())
        self.assertIn("_test_attn_door", available_attention_methods())
        self.assertIn("_test_kv_door", available_kv_compressors())


class _FakeHidden:
    """Minimal stand-in with a ``.shape`` so abstract-method calls don't crash."""

    shape = (1, 1, 8)


if __name__ == "__main__":
    unittest.main()
