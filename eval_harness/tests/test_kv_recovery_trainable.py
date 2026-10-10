"""Trainable-subset selection, freezing and accounting on tiny config-built models (CPU)."""
from __future__ import annotations

import unittest

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from eval_harness.kv_recovery.config import TrainableCfg
from eval_harness.kv_recovery.model_spec import inspect_model
from eval_harness.kv_recovery.trainable import (
    assert_no_stray_grads,
    assert_trainable,
    changed_parameters,
    first_trainable_layer,
    freeze_all_but,
    parameter_summary,
    select_layers,
    select_trainable,
    snapshot_parameters,
)

try:
    from transformers import Mistral3Config, Mistral3ForConditionalGeneration
    HAS_MISTRAL3 = True
except Exception:  # pragma: no cover
    HAS_MISTRAL3 = False

try:
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    HAS_QWEN35 = True
except Exception:  # pragma: no cover
    HAS_QWEN35 = False


def _tiny_llama(num_hidden_layers: int = 3) -> LlamaForCausalLM:
    cfg = LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=num_hidden_layers,
                      num_attention_heads=4, num_key_value_heads=2, vocab_size=256,
                      max_position_embeddings=8192, rope_theta=10000.0, attn_implementation="eager",
                      tie_word_embeddings=True)
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg).eval()


def tiny_mistral3():
    text = dict(model_type="ministral3", hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                num_attention_heads=4, num_key_value_heads=2, head_dim=16, vocab_size=512,
                max_position_embeddings=256, rms_norm_eps=1e-5, tie_word_embeddings=True, sliding_window=None,
                rope_parameters={"rope_type": "yarn", "type": "yarn", "rope_theta": 1000000.0, "factor": 16.0,
                                 "original_max_position_embeddings": 16, "beta_fast": 32.0, "beta_slow": 1.0,
                                 "mscale": 1.0, "mscale_all_dim": 1.0, "llama_4_scaling_beta": 0.1})
    vision = dict(model_type="pixtral", hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                  num_attention_heads=2, head_dim=16, image_size=28, patch_size=14, num_channels=3)
    cfg = Mistral3Config(text_config=text, vision_config=vision, image_token_index=10, tie_word_embeddings=True)
    cfg._attn_implementation = "eager"
    cfg.text_config._attn_implementation = "eager"
    torch.manual_seed(0)
    return Mistral3ForConditionalGeneration(cfg).eval()


def tiny_qwen35():
    cfg = Qwen3_5TextConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
                            num_key_value_heads=2, head_dim=16, vocab_size=512, max_position_embeddings=512,
                            layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
                            linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=16,
                            linear_value_head_dim=16, linear_conv_kernel_dim=4, tie_word_embeddings=True)
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    return Qwen3_5ForCausalLM(cfg).eval()


class TestSelectors(unittest.TestCase):
    def test_select_layers(self):
        self.assertEqual(select_layers("all", [3, 7, 11]), [3, 7, 11])
        self.assertEqual(select_layers("last_n:2", [3, 7, 11]), [7, 11])
        self.assertEqual(select_layers("last_n:9", [3, 7, 11]), [3, 7, 11])
        self.assertEqual(select_layers([11, 3], [3, 7, 11]), [3, 11])
        with self.assertRaises(ValueError):
            select_layers([4], [3, 7, 11])


class TestTinyLlama(unittest.TestCase):
    def setUp(self):
        self.model = _tiny_llama()
        self.spec = inspect_model(self.model)

    def test_spec(self):
        self.assertEqual(self.spec.lm_prefix, "model.")
        self.assertEqual(self.spec.n_layers, 3)
        self.assertEqual(self.spec.full_attention_layers, (0, 1, 2))
        self.assertFalse(self.spec.is_hybrid)
        self.assertTrue(self.spec.has_final_norm)
        self.assertEqual(self.spec.hidden_size, 64)

    def test_last_n_blocks(self):
        names = select_trainable(self.model, self.spec, TrainableCfg(strategy="last_n_blocks", n=1))
        self.assertTrue(all(n.startswith("model.layers.2.") for n in names))
        self.assertIn("model.layers.2.mlp.down_proj.weight", names)
        self.assertIn("model.layers.2.input_layernorm.weight", names)
        self.assertEqual(first_trainable_layer(names, self.spec), 2)
        names2 = select_trainable(self.model, self.spec, TrainableCfg(strategy="last_n_blocks", n=2))
        self.assertEqual(first_trainable_layer(names2, self.spec), 1)

    def test_attention_projections(self):
        names = select_trainable(self.model, self.spec, TrainableCfg(strategy="attention_projections",
                                                                      modules=["k_proj", "v_proj"], layers="all"))
        self.assertEqual(names, sorted(f"model.layers.{i}.self_attn.{m}.weight" for i in range(3) for m in ("k_proj", "v_proj")))
        names = select_trainable(self.model, self.spec, TrainableCfg(strategy="attention_projections",
                                                                      modules=["q_proj", "o_proj"], layers="last_n:1"))
        self.assertEqual(names, ["model.layers.2.self_attn.o_proj.weight", "model.layers.2.self_attn.q_proj.weight"])
        with self.assertRaises(ValueError):
            select_trainable(self.model, self.spec, TrainableCfg(strategy="attention_projections", modules=["z_proj"]))

    def test_mlp_norms_full(self):
        mlp = select_trainable(self.model, self.spec, TrainableCfg(strategy="mlp", layers=[0]))
        self.assertEqual(mlp, sorted(f"model.layers.0.mlp.{m}.weight" for m in ("gate_proj", "up_proj", "down_proj")))
        norms = select_trainable(self.model, self.spec, TrainableCfg(strategy="norms", layers="last_n:1"))
        self.assertEqual(norms, ["model.layers.2.input_layernorm.weight", "model.layers.2.post_attention_layernorm.weight"])
        full = select_trainable(self.model, self.spec, TrainableCfg(strategy="full"))
        self.assertNotIn("model.embed_tokens.weight", full)
        self.assertNotIn("lm_head.weight", full)
        self.assertIn("model.norm.weight", full)
        full_e = select_trainable(self.model, self.spec, TrainableCfg(strategy="full", include_embeddings=True))
        self.assertIn("model.embed_tokens.weight", full_e)
        self.assertNotIn("lm_head.weight", full_e)

    def test_freeze_assert_summary_and_grads(self):
        names = select_trainable(self.model, self.spec, TrainableCfg(strategy="attention_projections",
                                                                      modules=["o_proj"], layers="last_n:1"))
        expected = freeze_all_but(self.model, names)
        assert_trainable(self.model, expected, self.spec)
        summary = parameter_summary(self.model, self.spec)
        self.assertEqual(summary["trainable_parameters"], 64 * 64)
        self.assertEqual(summary["trainable_names"], names)
        self.assertEqual(summary["first_trainable_layer"], 2)
        self.assertEqual(summary["text_lm_parameters"], sum(p.numel() for p in self.model.model.parameters()))
        self.assertGreater(summary["percent_trainable_of_text_lm"], 0)
        # gradients only on the selected tensor
        ids = torch.randint(0, 256, (1, 8))
        self.model(input_ids=ids).logits.float().pow(2).mean().backward()
        assert_no_stray_grads(self.model, names)
        with self.assertRaises(KeyError):
            freeze_all_but(self.model, ["model.layers.9.self_attn.o_proj.weight"])

    def test_snapshot_and_changed(self):
        snap = snapshot_parameters(self.model)
        self.assertEqual(changed_parameters(self.model, snap), [])
        with torch.no_grad():
            self.model.model.layers[0].mlp.up_proj.weight.add_(1.0)
        self.assertEqual(changed_parameters(self.model, snap), ["model.layers.0.mlp.up_proj.weight"])


@unittest.skipUnless(HAS_MISTRAL3, "transformers build lacks Mistral3")
class TestTinyMistral3(unittest.TestCase):
    def test_prefix_and_vision_never_selected(self):
        model = tiny_mistral3()
        spec = inspect_model(model)
        self.assertEqual(spec.lm_prefix, "model.language_model.")
        self.assertEqual(spec.full_attention_layers, (0, 1, 2))
        self.assertTrue(spec.tied_embeddings)
        names = select_trainable(model, spec, TrainableCfg(strategy="attention_projections", modules=["q_proj"], layers="all"))
        self.assertEqual(names, sorted(f"model.language_model.layers.{i}.self_attn.q_proj.weight" for i in range(3)))
        full = select_trainable(model, spec, TrainableCfg(strategy="full", include_embeddings=True))
        self.assertFalse(any("vision" in n or "multi_modal" in n or n.startswith("lm_head") for n in full))
        expected = freeze_all_but(model, names)
        assert_trainable(model, expected, spec)


@unittest.skipUnless(HAS_QWEN35, "transformers build lacks Qwen3.5")
class TestTinyQwen35(unittest.TestCase):
    def test_only_full_attention_layers_have_projections(self):
        model = tiny_qwen35()
        spec = inspect_model(model)
        self.assertEqual(spec.lm_prefix, "model.")
        self.assertEqual(spec.full_attention_layers, (3,))
        self.assertTrue(spec.is_hybrid)
        names = select_trainable(model, spec, TrainableCfg(strategy="attention_projections",
                                                            modules=["k_proj", "v_proj"], layers="all"))
        self.assertEqual(names, ["model.layers.3.self_attn.k_proj.weight", "model.layers.3.self_attn.v_proj.weight"])
        last1 = select_trainable(model, spec, TrainableCfg(strategy="last_n_blocks", n=1))
        self.assertTrue(all(n.startswith("model.layers.3.") for n in last1))
        last2 = select_trainable(model, spec, TrainableCfg(strategy="last_n_blocks", n=2))
        self.assertTrue(any(".linear_attn." in n for n in last2))
        self.assertEqual(first_trainable_layer(last2, spec), 2)


if __name__ == "__main__":
    unittest.main()
