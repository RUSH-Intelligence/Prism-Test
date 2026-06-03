"""Tests for HFAdapter prefill/decode split."""
from __future__ import annotations

import types
import unittest

import torch

from eval_harness.hf_adapter import HFAdapter, HFGenerateConfig, _sample, _sdpa

try:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS as _ALL_ATTN_FNS
except ImportError:
    _ALL_ATTN_FNS = None


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        # Return a short fixed sequence so tests don't depend on a real tokenizer.
        return [1, 5, 7]

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(i) for i in ids if i != self.eos_token_id)


class _FakeModelOutputs:
    def __init__(self, logits, past_key_values):
        self.logits = logits
        self.past_key_values = past_key_values


class _FakeModel(torch.nn.Module):
    """Minimal model stub: always predicts token id 42, then EOS on second call."""

    def __init__(self):
        super().__init__()
        self._call_count = 0
        self.device = torch.device("cpu")
        self.config = types.SimpleNamespace()

    def forward(self, input_ids, use_cache=True, return_dict=True, past_key_values=None):
        batch, seq = input_ids.shape
        vocab = 100
        logits = torch.zeros(batch, seq, vocab)
        # Predict token 42 on first decode step, EOS (2) thereafter.
        predict = 42 if self._call_count == 0 else 2
        logits[:, -1, predict] = 100.0
        self._call_count += 1
        fake_kv = object()
        return _FakeModelOutputs(logits=logits, past_key_values=fake_kv)


def _make_adapter() -> HFAdapter:
    adapter = object.__new__(HFAdapter)
    adapter._tokenizer = _FakeTokenizer()
    adapter._model = _FakeModel()
    return adapter


class TestPrefill(unittest.TestCase):
    def test_returns_last_logits_and_kv_cache(self):
        adapter = _make_adapter()
        input_ids = torch.zeros(1, 5, dtype=torch.long)
        last_logits, past_kv = adapter._prefill(input_ids)

        self.assertEqual(tuple(last_logits.shape), (1, 100))
        self.assertIsNotNone(past_kv)

    def test_model_called_once(self):
        adapter = _make_adapter()
        input_ids = torch.zeros(1, 3, dtype=torch.long)
        adapter._prefill(input_ids)
        self.assertEqual(adapter._model._call_count, 1)


class TestDecode(unittest.TestCase):
    def test_generates_token_then_eos(self):
        adapter = _make_adapter()
        # Prefill once to get initial logits and kv
        input_ids = torch.zeros(1, 3, dtype=torch.long)
        last_logits, past_kv = adapter._prefill(input_ids)

        gen_cfg = HFGenerateConfig(max_tokens=10)
        generated = adapter._decode(last_logits, past_kv, gen_cfg)

        # First call predicts 42, second predicts EOS (2)
        self.assertEqual(generated[0], 42)
        self.assertIn(2, generated)
        # Should stop after EOS
        self.assertLessEqual(len(generated), 2)

    def test_respects_max_tokens(self):
        adapter = _make_adapter()
        # Override model to never produce EOS
        adapter._model._call_count = -999  # keeps predicting 42 forever

        class _NoEosModel(_FakeModel):
            def forward(self, input_ids, **kwargs):
                out = super().forward(input_ids, **kwargs)
                # Always predict token 42
                out.logits[:, -1, :] = 0.0
                out.logits[:, -1, 42] = 100.0
                return out

        adapter._model = _NoEosModel()
        input_ids = torch.zeros(1, 3, dtype=torch.long)
        last_logits, past_kv = adapter._prefill(input_ids)

        gen_cfg = HFGenerateConfig(max_tokens=5)
        generated = adapter._decode(last_logits, past_kv, gen_cfg)
        self.assertLessEqual(len(generated), 5)


class TestGenerate(unittest.TestCase):
    def test_returns_string_per_prompt(self):
        adapter = _make_adapter()
        gen_cfg = HFGenerateConfig(max_tokens=10)
        results = adapter.generate(["hello world", "foo"], gen_cfg)
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertIsInstance(r, str)


class TestSampleHelper(unittest.TestCase):
    def test_greedy_picks_argmax(self):
        logits = torch.tensor([[0.0, 0.0, 5.0, 0.0]])
        gen_cfg = HFGenerateConfig(max_tokens=1, temperature=0.0)
        self.assertEqual(_sample(logits, gen_cfg), 2)

    def test_temperature_sampling_returns_valid_index(self):
        logits = torch.zeros(1, 10)
        gen_cfg = HFGenerateConfig(max_tokens=1, temperature=1.0)
        token = _sample(logits, gen_cfg)
        self.assertGreaterEqual(token, 0)
        self.assertLess(token, 10)

    def test_top_p_zero_picks_top_token(self):
        logits = torch.tensor([[0.0, 0.0, 10.0, 0.0]])
        gen_cfg = HFGenerateConfig(max_tokens=1, temperature=1.0, top_p=0.01)
        self.assertEqual(_sample(logits, gen_cfg), 2)


# ---------------------------------------------------------------------------
# Helpers for attention-hook mechanism tests
# ---------------------------------------------------------------------------

class _FakeModelWithAttnImpl(torch.nn.Module):
    """Model stub with config._attn_implementation so _with_attention can locate it."""

    def __init__(self, vocab: int = 50):
        super().__init__()
        self._vocab = vocab
        self.device = torch.device("cpu")
        self.config = types.SimpleNamespace(_attn_implementation="sdpa")

    def forward(self, input_ids, use_cache=True, return_dict=True,
                past_key_values=None, **kwargs):
        B, T = input_ids.shape
        logits = torch.zeros(B, T, self._vocab)
        logits[:, -1, 1] = 10.0
        return _FakeModelOutputs(logits=logits, past_key_values=past_key_values)


class _AttentionDispatchModel(torch.nn.Module):
    """
    Model that forwards calls through ALL_ATTENTION_FUNCTIONS when a hook
    has been registered, allowing end-to-end verification of hook injection.
    """

    def __init__(self, dim: int = 8, vocab: int = 50):
        super().__init__()
        self._dim = dim
        self._vocab = vocab
        self.device = torch.device("cpu")
        self.config = types.SimpleNamespace(_attn_implementation="sdpa")

    def forward(self, input_ids, use_cache=True, return_dict=True,
                past_key_values=None, **kwargs):
        B, T = input_ids.shape
        D = self._dim
        if _ALL_ATTN_FNS is not None:
            impl = self.config._attn_implementation
            mapping = getattr(_ALL_ATTN_FNS, '_global_mapping', {})
            if impl in mapping:
                q = torch.zeros(B, 1, T, D)
                k = torch.zeros(B, 1, T, D)
                v = torch.zeros(B, 1, T, D)
                mapping[impl](self, q, k, v, None, scaling=1.0)
        logits = torch.zeros(B, T, self._vocab)
        logits[:, -1, 1] = 10.0
        return _FakeModelOutputs(logits=logits, past_key_values=past_key_values)


class _HookTrackingAdapter(HFAdapter):
    """HFAdapter subclass that wires _with_attention into prefill and decode phases."""

    def __init__(self, model: torch.nn.Module) -> None:
        self._tokenizer = _FakeTokenizer()
        self._model = model
        self.prefill_calls = 0
        self.decode_calls = 0

    def prefill_attention(self, module, queries, keys, values, attention_mask,
                          scaling=1.0, dropout=0.0, **kwargs):
        self.prefill_calls += 1
        return _sdpa(queries, keys, values, attention_mask, scaling, dropout)

    def decode_attention(self, module, queries, keys, values, attention_mask,
                         scaling=1.0, dropout=0.0, **kwargs):
        self.decode_calls += 1
        return _sdpa(queries, keys, values, attention_mask, scaling, dropout)

    def _prefill(self, input_ids):
        fn = lambda mod, q, k, v, mask, scaling=1.0, dropout=0.0, **kw: \
            self.prefill_attention(mod, q, k, v, mask, scaling, dropout, **kw)
        with self._with_attention(fn):
            return super()._prefill(input_ids)

    def _decode(self, last_logits, past_key_values, gen_cfg):
        fn = lambda mod, q, k, v, mask, scaling=1.0, dropout=0.0, **kw: \
            self.decode_attention(mod, q, k, v, mask, scaling, dropout, **kw)
        with self._with_attention(fn):
            return super()._decode(last_logits, past_key_values, gen_cfg)


@unittest.skipIf(_ALL_ATTN_FNS is None, "ALL_ATTENTION_FUNCTIONS unavailable in this environment")
class TestWithAttentionMechanism(unittest.TestCase):

    def test_with_attention_swaps_and_restores(self):
        """_with_attention must change _attn_implementation during the context and restore it after."""
        model = _FakeModelWithAttnImpl()
        adapter = object.__new__(HFAdapter)
        adapter._tokenizer = _FakeTokenizer()
        adapter._model = model

        original = model.config._attn_implementation

        def _noop(mod, q, k, v, mask, scaling=1.0, dropout=0.0, **kw):
            return _sdpa(q, k, v, mask, scaling, dropout)

        with adapter._with_attention(_noop):
            during = model.config._attn_implementation

        self.assertNotEqual(during, original)
        self.assertEqual(model.config._attn_implementation, original)

    def test_prefill_attention_called_during_prefill(self):
        """When the model dispatches through ALL_ATTENTION_FUNCTIONS, prefill_attention fires."""
        model = _AttentionDispatchModel()
        adapter = _HookTrackingAdapter(model)

        input_ids = torch.zeros(1, 4, dtype=torch.long)
        adapter._prefill(input_ids)

        self.assertGreater(adapter.prefill_calls, 0)

    def test_decode_attention_called_during_decode(self):
        """When the model dispatches through ALL_ATTENTION_FUNCTIONS, decode_attention fires."""
        model = _AttentionDispatchModel()
        adapter = _HookTrackingAdapter(model)

        input_ids = torch.zeros(1, 4, dtype=torch.long)
        last_logits, past_kv = adapter._prefill(input_ids)

        gen_cfg = HFGenerateConfig(max_tokens=2, temperature=0.0)
        adapter._decode(last_logits, past_kv, gen_cfg)

        self.assertGreater(adapter.decode_calls, 0)


if __name__ == "__main__":
    unittest.main()
