"""Delta checkpoints: round trip, verification, identity delta, HFAdapter weight_delta hook."""
from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from eval_harness.kv_recovery.checkpoint import (
    apply_delta,
    checkpoint_digest,
    frozen_sample_names,
    hashes_of,
    load_metadata,
    write_delta,
    write_identity_delta,
)
from eval_harness.kv_recovery.model_spec import inspect_model


def _tiny_llama():
    cfg = LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, vocab_size=256, max_position_embeddings=8192, attn_implementation="eager")
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg).eval().to(torch.bfloat16)


class TestDeltaRoundTrip(unittest.TestCase):
    def setUp(self):
        self.model = _tiny_llama()
        self.spec = inspect_model(self.model)
        self.names = ["model.layers.1.self_attn.o_proj.weight", "model.layers.1.self_attn.k_proj.weight"]
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name) / "ckpt"

    def tearDown(self):
        self.tmp.cleanup()

    def _train_step(self):
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if n in self.names:
                    p.add_(0.05 * torch.randn_like(p))

    def test_write_apply_verify(self):
        orig_sha = hashes_of(self.model, self.names)
        frozen = frozen_sample_names(self.model, self.names, self.spec)
        self.assertTrue(frozen and not set(frozen) & set(self.names))
        frozen_sha = hashes_of(self.model, frozen)
        self._train_step()
        masters = {n: dict(self.model.named_parameters())[n].detach().float() for n in self.names}
        write_delta(self.dir, self.model, self.names, orig_sha, {"base_model": "tiny/llama", "note": "t"},
                    masters=masters, frozen_sample_sha256=frozen_sha, config_yaml="run_name: x\n")
        meta = load_metadata(self.dir)
        for key in ("format_version", "trainable_parameters", "original_sha256", "adapted_sha256", "frozen_sample_sha256",
                    "weights_sha256", "dtypes", "shapes", "has_fp32_masters", "created_at", "base_model"):
            self.assertIn(key, meta)
        self.assertEqual(meta["trainable_parameters"], sorted(self.names))
        self.assertEqual(meta["weights_sha256"], checkpoint_digest(self.dir))
        self.assertTrue((self.dir / "config.yaml").exists())
        adapted = {n: dict(self.model.named_parameters())[n].detach().clone() for n in self.names}

        fresh = _tiny_llama()                    # same seed -> the original weights
        info = apply_delta(fresh, self.dir, strict=True, expected_sha256=meta["weights_sha256"])
        self.assertEqual(info["applied"], sorted(self.names))
        for n in self.names:
            self.assertTrue(torch.equal(dict(fresh.named_parameters())[n], adapted[n]))
        # untouched tensors stay bitwise identical to the base
        base = _tiny_llama()
        for n, p in fresh.named_parameters():
            if n not in self.names:
                self.assertTrue(torch.equal(p, dict(base.named_parameters())[n]), n)
        # a second application must be refused (base hashes no longer match)
        with self.assertRaises(ValueError):
            apply_delta(fresh, self.dir, strict=True)
        # and wrong expected digest is refused
        with self.assertRaises(ValueError):
            apply_delta(_tiny_llama(), self.dir, strict=True, expected_sha256="0" * 64)
        # non-strict mode warns instead
        info2 = apply_delta(fresh, self.dir, strict=False)
        self.assertTrue(info2["problems"])

    def test_identity_delta_is_a_noop(self):
        before = {n: p.detach().clone() for n, p in self.model.named_parameters()}
        write_identity_delta(self.dir, self.model, self.names, {"base_model": "tiny/llama"})
        info = apply_delta(self.model, self.dir, strict=True)
        self.assertTrue(info["identity_delta"])
        for n, p in self.model.named_parameters():
            self.assertTrue(torch.equal(p, before[n]), n)

    def test_frozen_sample_mismatch_detected(self):
        orig_sha = hashes_of(self.model, self.names)
        frozen = frozen_sample_names(self.model, self.names, self.spec)
        write_delta(self.dir, self.model, self.names, orig_sha, {"base_model": "tiny/llama"},
                    frozen_sample_sha256=hashes_of(self.model, frozen))
        other = _tiny_llama()
        with torch.no_grad():
            dict(other.named_parameters())[frozen[0]].add_(1.0)
        with self.assertRaises(ValueError):
            apply_delta(other, self.dir, strict=True)


class TestHFAdapterWeightDelta(unittest.TestCase):
    def test_weight_delta_is_popped_and_applied_before_cuda(self):
        import eval_harness.hf_adapter as hf_adapter
        from eval_harness.hf_adapter import HFAdapter

        model = _tiny_llama()
        calls = {}

        def fake_load(name, load_kwargs):
            calls["load_kwargs"] = dict(load_kwargs)
            return model

        def fake_apply(m, path, strict=True, expected_sha256=None, verify_frozen=True):
            calls["apply"] = (m is model, path, strict, expected_sha256)
            return {"applied": ["x"], "weights_sha256": "abc"}

        tok = types.SimpleNamespace(pad_token_id=0, eos_token_id=2)
        with patch.object(hf_adapter, "_load_model", side_effect=fake_load), \
             patch.object(hf_adapter.AutoTokenizer, "from_pretrained", return_value=tok), \
             patch("eval_harness.kv_recovery.checkpoint.apply_delta", side_effect=fake_apply):
            adapter = HFAdapter(model="tiny/llama", dtype="bfloat16",
                                weight_delta={"path": "/some/ckpt", "sha256": "abc", "strict": True})
        self.assertNotIn("weight_delta", calls["load_kwargs"])
        self.assertEqual(calls["apply"], (True, "/some/ckpt", True, "abc"))
        self.assertEqual(adapter._weight_delta_meta["weights_sha256"], "abc")
        with patch.object(hf_adapter, "_load_model", side_effect=fake_load), \
             patch.object(hf_adapter.AutoTokenizer, "from_pretrained", return_value=tok):
            adapter = HFAdapter(model="tiny/llama", dtype="bfloat16")
        self.assertIsNone(adapter._weight_delta_meta)


if __name__ == "__main__":
    unittest.main()
