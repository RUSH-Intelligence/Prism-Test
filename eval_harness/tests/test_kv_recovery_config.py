"""RecoveryConfig: strict loading, overrides, validation, the shared compression block."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from eval_harness.kv_recovery.config import (
    LONGBENCH_16,
    RecoveryConfig,
    apply_overrides,
    kv_budget_to_ratio,
    load_config,
    model_llm_kwargs,
    parse_override,
    research_config_dict,
    training_identity,
)

REPO = Path(__file__).resolve().parents[2]
CARDS = REPO / "configs" / "kv_recovery"


class TestDefaultsAndCards(unittest.TestCase):
    def test_defaults_validate(self):
        cfg = RecoveryConfig.from_dict({})
        self.assertEqual(cfg.kv_compression.kv_compressor, "knorm")
        self.assertEqual(cfg.data.max_length, 16384)
        self.assertEqual([b.benchmark for b in cfg.eval.benchmarks], ["ruler16k", "ruler32k", "longbench"])
        self.assertEqual(cfg.eval.benchmarks[2].subsets, LONGBENCH_16)

    def test_shipped_cards_load_and_round_trip(self):
        for name in ("ministral_3b.yaml", "qwen35_4b.yaml", "smoke_ministral_3b.yaml", "smoke_qwen35_4b.yaml"):
            with self.subTest(card=name):
                cfg = load_config(CARDS / name)
                again = RecoveryConfig.from_dict(cfg.to_dict())
                self.assertEqual(again.to_dict(), cfg.to_dict())
                self.assertEqual(cfg.model.attn_implementation, "sdpa")
                self.assertTrue(cfg.eval.strip_auto_system_block)
        q = load_config(CARDS / "qwen35_4b.yaml")
        self.assertEqual(q.model.name, "Qwen/Qwen3.5-4B")
        self.assertFalse(q.model.dequantize_fp8)
        self.assertIs(q.eval.group_by_context, False)
        m = load_config(CARDS / "ministral_3b.yaml")
        self.assertTrue(m.model.dequantize_fp8)

    def test_matrix_card_is_valid_yaml_with_expected_axes(self):
        m = yaml.safe_load((CARDS / "matrix.yaml").read_text())
        self.assertEqual(set(m["base_configs"]), {"ministral_3b", "qwen35_4b", "ministral_3b_mix", "qwen35_4b_mix"})
        self.assertEqual(m["compressors"], ["knorm", "cur"])
        self.assertEqual(m["ratios"], [0.75, 0.5])
        self.assertEqual(set(m["trainable"]), {"last1", "last2", "qo_last4", "kv_attn", "qo_sens4", "kv_sens16",
                                               "qo_sens8", "qo_sens16", "kv_sens4", "kv_sens8"})
        self.assertEqual(m["primary"]["trainable"], ["last1", "last2", "qo_last4", "kv_attn", "qo_sens4", "kv_sens16"])
        self.assertEqual(m["ablation_topk"]["trainable"], ["qo_sens4", "qo_sens8", "qo_sens16", "kv_sens4", "kv_sens8", "kv_sens16"])
        for name, overrides in m["trainable"].items():
            cfg = RecoveryConfig.from_dict(apply_overrides({}, shortcuts=overrides))
            self.assertIn(cfg.trainable.strategy, ("last_n_blocks", "attention_projections"), name)
            if "sens" in name:
                self.assertEqual(cfg.trainable.layers, "sensitivity", name)
                self.assertEqual(cfg.trainable.sensitivity.top_k, int(name.rsplit("sens", 1)[1]))
        # budget matching: the sensitivity-selected subsets train the same projections and layer counts
        self.assertEqual(m["trainable"]["qo_sens4"]["trainable.modules"], m["trainable"]["qo_last4"]["trainable.modules"])
        self.assertEqual(m["trainable"]["kv_sens16"]["trainable.modules"], m["trainable"]["kv_attn"]["trainable.modules"])
        self.assertEqual(m["model_overrides"]["qwen35_4b"]["kv_sens16"], {"trainable.sensitivity.top_k": 4})
        # the shipped run cards default to the sensitivity-selected q/o projections
        for name in ("ministral_3b.yaml", "qwen35_4b.yaml", "smoke_ministral_3b.yaml", "smoke_qwen35_4b.yaml"):
            cfg = load_config(CARDS / name)
            self.assertEqual((cfg.trainable.strategy, cfg.trainable.modules, cfg.trainable.layers),
                             ("attention_projections", ["q_proj", "o_proj"], "sensitivity"), name)


class TestStrictness(unittest.TestCase):
    def test_unknown_top_level_key_raises(self):
        with self.assertRaises(KeyError):
            RecoveryConfig.from_dict({"learning_rate": 1e-5})

    def test_unknown_nested_key_raises_with_path(self):
        with self.assertRaises(KeyError) as ctx:
            RecoveryConfig.from_dict({"alignment": {"layers": {"strategie": "last_n"}}})
        self.assertIn("alignment.layers", str(ctx.exception))

    def test_unknown_key_in_benchmark_list_raises(self):
        with self.assertRaises(KeyError):
            RecoveryConfig.from_dict({"eval": {"benchmarks": [{"benchmark": "ruler16k", "max_request": 5}]}})

    def test_validation_rules(self):
        bad = [
            {"kv_compression": {"compression_ratio": 1.0}},
            {"data": {"suffix_length": 16384}},
            {"data": {"format": "markdown"}},
            {"data": {"val_path": "data/kv_recovery/pg19_train.jsonl"}},
            {"alignment": {"loss": "l1"}},
            {"alignment": {"layers": {"strategy": "explicit", "indices": None}}},
            {"alignment": {"positions": {"strategy": "recent", "n": 0}}},
            {"trainable": {"strategy": "lora"}},
            {"trainable": {"layers": "last:4"}},
            {"loss": {"hidden_weight": 0.0, "kl_weight": 0.0}},
            {"teacher": {"mode": "offline"}, "loss": {"kl_weight": 0.5}},
            {"student": {"segment_mode": "chunked"}},
            {"eval": {"benchmarks": []}},
        ]
        for d in bad:
            with self.subTest(d=d):
                with self.assertRaises((ValueError, KeyError)):
                    RecoveryConfig.from_dict(d)


class TestOverrides(unittest.TestCase):
    def test_parse_override_yaml_values(self):
        self.assertEqual(parse_override("optim.learning_rate=1e-5"), ("optim.learning_rate", 1e-5))
        self.assertEqual(parse_override("trainable.modules=[q_proj, o_proj]"), ("trainable.modules", ["q_proj", "o_proj"]))
        self.assertEqual(parse_override("data.val_path=null"), ("data.val_path", None))
        with self.assertRaises(ValueError):
            parse_override("no_equals_sign")

    def test_set_then_shortcuts_precedence(self):
        d = apply_overrides({"optim": {"learning_rate": 1e-5}},
                            overrides=["optim.learning_rate=3e-5", "trainable.n=2"],
                            shortcuts={"optim.learning_rate": 7e-5})
        self.assertEqual(d["optim"]["learning_rate"], 7e-5)
        self.assertEqual(d["trainable"]["n"], 2)

    def test_load_config_with_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "c.yaml"
            p.write_text(yaml.safe_dump({"run_name": "x", "kv_compression": {"kv_compressor": "cur"}}))
            cfg = load_config(p, overrides=["kv_compression.compression_ratio=0.5", "data.max_length=2048",
                                            "data.suffix_length=128"])
            self.assertEqual(cfg.kv_compression.kv_compressor, "cur")
            self.assertAlmostEqual(cfg.kv_compression.compression_ratio, 0.5)
            self.assertEqual(cfg.data.max_length, 2048)

    def test_kv_budget_ratio_maps_to_fraction_pruned(self):
        self.assertAlmostEqual(kv_budget_to_ratio(0.25), 0.75)
        self.assertAlmostEqual(kv_budget_to_ratio(1.0), 0.0)
        with self.assertRaises(ValueError):
            kv_budget_to_ratio(0.0)


class TestSharedCompressionBlock(unittest.TestCase):
    def test_compressed_vs_dense_dicts(self):
        cfg = RecoveryConfig.from_dict({"kv_compression": {"kv_compressor": "cur", "compression_ratio": 0.75,
                                                           "kv_compressor_kwargs": {"num_sinks": 4}}})
        comp = research_config_dict(cfg, compressed=True)
        dense = research_config_dict(cfg, compressed=False)
        self.assertEqual(comp["kv_compressor"], "cur")
        self.assertEqual(comp["kv_compressor_kwargs"], {"num_sinks": 4})
        self.assertAlmostEqual(comp["compression_ratio"], 0.75)
        self.assertEqual(dense["kv_compressor"], "none")
        self.assertEqual(dense["compression_ratio"], 0.0)
        self.assertIsNone(dense["kv_compressor_kwargs"])
        for d in (comp, dense):
            self.assertEqual(d["attention_method"], "none")
            self.assertEqual(d["positional_method"], "none")
            self.assertTrue(d["strip_auto_system_block"])
            self.assertTrue(d["use_chat_template"])
        # The dict must be accepted verbatim by ResearchConfig (field names match).
        from eval_harness.research_adapter import ResearchConfig
        rc = ResearchConfig(**comp)
        self.assertEqual(rc.kv_compressor, "cur")
        # Mutating the returned dict never touches the config (defensive copies).
        comp["kv_compressor_kwargs"]["num_sinks"] = 99
        self.assertEqual(cfg.kv_compression.kv_compressor_kwargs["num_sinks"], 4)

    def test_model_llm_kwargs_and_identity(self):
        cfg = RecoveryConfig.from_dict({"model": {"dequantize_fp8": True, "attn_implementation": "sdpa"}})
        self.assertEqual(model_llm_kwargs(cfg), {"attn_implementation": "sdpa", "dequantize_fp8": True})
        cfg2 = RecoveryConfig.from_dict({})
        self.assertEqual(model_llm_kwargs(cfg2), {"attn_implementation": "sdpa"})
        ident = training_identity(cfg)
        self.assertEqual(set(ident), {"model", "kv_compression", "research_config", "prompt_shaping"})
        self.assertEqual(ident["research_config"], research_config_dict(cfg, compressed=True))

    def test_digest_changes_with_compression_ratio(self):
        a = RecoveryConfig.from_dict({})
        b = RecoveryConfig.from_dict({"kv_compression": {"compression_ratio": 0.5}})
        self.assertNotEqual(a.digest(), b.digest())
        self.assertEqual(a.digest(), RecoveryConfig.from_dict({}).digest())


if __name__ == "__main__":
    unittest.main()
