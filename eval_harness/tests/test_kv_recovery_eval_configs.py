"""Three-way EvalConfig builder: validity, drift guards, barcodes, delta/config matching."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from eval_harness.config import EvalConfig
from eval_harness.kv_recovery.checkpoint import hashes_of, write_delta
from eval_harness.kv_recovery.config import RecoveryConfig, training_identity
from eval_harness.kv_recovery.eval_configs import (
    assert_delta_matches_config,
    assert_same_compression,
    barcode_for_eval_config,
    build_cells,
    build_eval_config,
    delta_config_mismatch,
    find_reusable_dense,
    group_by_context_for,
    results_root,
)

BASE = {"run_name": "t", "model": {"name": "tiny/llama", "dequantize_fp8": True},
        "kv_compression": {"kv_compressor": "cur", "compression_ratio": 0.75, "kv_compressor_kwargs": {"num_sinks": 4}},
        "eval": {"benchmarks": [{"benchmark": "ruler16k", "subsets": ["niah_single_1", "qa_1"], "max_requests": 5,
                                 "request_offset": 0, "walltime": "0:30:00"}], "group_by_context": True}}


class TestBuild(unittest.TestCase):
    def setUp(self):
        self.cfg = RecoveryConfig.from_dict(BASE)
        self.bench = self.cfg.eval.benchmarks[0]

    def test_every_condition_is_a_valid_evalconfig(self):
        for cond in ("dense", "compressed", "compressed_recovered", "dense_recovered"):
            d = build_eval_config(self.cfg, cond, self.bench, checkpoint_dir=Path("/ckpt"), checkpoint_sha256="ab")
            ev = EvalConfig(**d)
            self.assertEqual(ev.backend, "research")
            self.assertFalse(ev.query_aware)
            self.assertTrue(ev.deterministic)
            self.assertIsNone(ev.max_new_tokens)
            self.assertEqual(ev.subsets, "niah_single_1,qa_1")
            self.assertEqual(ev.max_requests, 5)
            self.assertEqual(ev.llm_kwargs["attn_implementation"], "sdpa")
            self.assertTrue(ev.llm_kwargs["dequantize_fp8"])
            rc = ev.llm_kwargs["research_config"]
            self.assertTrue(rc["strip_auto_system_block"])
            if cond.startswith("dense"):
                self.assertEqual(rc["kv_compressor"], "none")
                self.assertEqual(rc["compression_ratio"], 0.0)
            else:
                self.assertEqual(rc["kv_compressor"], "cur")
                self.assertEqual(rc["kv_compressor_kwargs"], {"num_sinks": 4})
            if cond.endswith("recovered"):
                self.assertEqual(ev.llm_kwargs["weight_delta"]["sha256"], "ab")
            else:
                self.assertNotIn("weight_delta", ev.llm_kwargs)
        with self.assertRaises(ValueError):
            build_eval_config(self.cfg, "compressed_recovered", self.bench)   # no checkpoint
        with self.assertRaises(ValueError):
            build_eval_config(self.cfg, "weird", self.bench)

    def test_compressed_and_recovered_differ_only_in_weight_delta(self):
        comp = build_eval_config(self.cfg, "compressed", self.bench)
        rec = build_eval_config(self.cfg, "compressed_recovered", self.bench, checkpoint_dir=Path("/c"), checkpoint_sha256="x")
        assert_same_compression(comp, rec)
        tampered = json.loads(json.dumps(rec))
        tampered["llm_kwargs"]["research_config"]["compression_ratio"] = 0.5
        with self.assertRaises(AssertionError):
            assert_same_compression(comp, tampered)
        with self.assertRaises(AssertionError):
            assert_same_compression(comp, comp)      # recovered arm must carry a delta

    def test_barcodes(self):
        comp = build_eval_config(self.cfg, "compressed", self.bench)
        dense = build_eval_config(self.cfg, "dense", self.bench)
        rec_a = build_eval_config(self.cfg, "compressed_recovered", self.bench, checkpoint_dir=Path("/c"), checkpoint_sha256="a")
        rec_b = build_eval_config(self.cfg, "compressed_recovered", self.bench, checkpoint_dir=Path("/c"), checkpoint_sha256="b")
        codes = {barcode_for_eval_config(d) for d in (comp, dense, rec_a, rec_b)}
        self.assertEqual(len(codes), 4)
        moved = dict(rec_a, output_dir="/elsewhere")
        self.assertEqual(barcode_for_eval_config(moved), barcode_for_eval_config(rec_a))

    def test_build_cells_layout(self):
        cells = build_cells(self.cfg, checkpoint_dir=Path("/c"), checkpoint_sha256="z")
        self.assertEqual([c.condition for c in cells], ["dense", "compressed", "compressed_recovered"])
        root = results_root(self.cfg)
        for c in cells:
            self.assertTrue(str(c.run_dir).startswith(str(root / "ruler16k" / f"{c.condition}__")))
            self.assertEqual(c.config["output_dir"], str(c.run_dir))
            self.assertTrue(c.config["output_dir_exact"])
        only = build_cells(self.cfg, conditions=("dense",), benchmarks=["ruler16k"])
        self.assertEqual(len(only), 1)

    def test_group_by_context(self):
        self.assertTrue(group_by_context_for(self.cfg))
        cfg2 = RecoveryConfig.from_dict({**BASE, "eval": {**BASE["eval"], "group_by_context": None}})
        with patch("transformers.AutoConfig.from_pretrained", side_effect=OSError("offline")):
            self.assertTrue(group_by_context_for(cfg2))
            cfg3 = RecoveryConfig.from_dict({**BASE, "model": {"name": "Qwen/Qwen3.5-4B"}, "eval": {**BASE["eval"], "group_by_context": None}})
            self.assertFalse(group_by_context_for(cfg3))


class TestDeltaMatching(unittest.TestCase):
    def test_delta_identity_vs_config(self):
        cfg = RecoveryConfig.from_dict(BASE)
        model = LlamaForCausalLM(LlamaConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
                                             num_key_value_heads=1, vocab_size=64, attn_implementation="eager")).eval()
        names = ["model.layers.0.self_attn.o_proj.weight"]
        with tempfile.TemporaryDirectory() as tmp:
            ck = Path(tmp) / "ckpt"
            write_delta(ck, model, names, hashes_of(model, names), {"base_model": "tiny/llama", **training_identity(cfg)})
            self.assertEqual(delta_config_mismatch(ck, cfg), {})
            assert_delta_matches_config(ck, cfg)
            other = RecoveryConfig.from_dict({**BASE, "kv_compression": {**BASE["kv_compression"], "compression_ratio": 0.5}})
            diff = delta_config_mismatch(ck, other)
            self.assertIn("research_config", diff)
            self.assertIn("compression_ratio", diff["research_config"])
            with self.assertRaises(ValueError):
                assert_delta_matches_config(ck, other)
            self.assertTrue(assert_delta_matches_config(ck, other, allow_mismatch=True))
            # a dense cell is reusable iff its recorded fingerprint equals this dense arm's barcode
            dense = build_eval_config(cfg, "dense", cfg.eval.benchmarks[0])
            cell = Path(tmp) / "dense_cell"
            cell.mkdir()
            self.assertIsNone(find_reusable_dense(cell, dense))
            (cell / "run_spec.json").write_text(json.dumps({"fingerprint": barcode_for_eval_config(dense)}))
            (cell / "DONE.json").write_text("{}")
            self.assertEqual(find_reusable_dense(cell, dense), cell)
            (cell / "run_spec.json").write_text(json.dumps({"fingerprint": "nope"}))
            self.assertIsNone(find_reusable_dense(cell, dense))


if __name__ == "__main__":
    unittest.main()
