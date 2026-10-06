"""Hidden-state capture hooks on a tiny config-built Llama (CPU, random weights)."""
from __future__ import annotations

import unittest

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from eval_harness.kv_recovery.hidden_states import (
    FINAL_NORM_KEY,
    capture_layer_outputs,
    gather_positions,
    state_keys,
)
from eval_harness.kv_recovery.model_spec import decoder_layers, final_norm


def _tiny_llama(num_hidden_layers: int = 2) -> LlamaForCausalLM:
    cfg = LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=num_hidden_layers,
                      num_attention_heads=4, num_key_value_heads=2, vocab_size=256,
                      max_position_embeddings=8192, rope_theta=10000.0, attn_implementation="eager")
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg).eval()


class TestCapture(unittest.TestCase):
    def setUp(self):
        self.model = _tiny_llama()
        self.ids = torch.randint(0, 256, (1, 12))

    def test_shapes_keys_and_hook_removal(self):
        with capture_layer_outputs(self.model, [0, 1], detach=True, include_final_norm=True) as cap:
            with torch.no_grad():
                self.model(input_ids=self.ids, logits_to_keep=1)
        states = cap.states()
        self.assertEqual(set(states), {0, 1, FINAL_NORM_KEY})
        for v in states.values():
            self.assertEqual(tuple(v.shape), (1, 12, 64))
            self.assertFalse(v.requires_grad)
        for layer in decoder_layers(self.model):
            self.assertEqual(len(layer._forward_hooks), 0)
        self.assertEqual(len(final_norm(self.model)._forward_hooks), 0)
        self.assertEqual(state_keys([0, 1], True), [0, 1, FINAL_NORM_KEY])

    def test_final_norm_output_is_lm_head_input(self):
        with capture_layer_outputs(self.model, [], detach=True, include_final_norm=True) as cap:
            with torch.no_grad():
                out = self.model(input_ids=self.ids, logits_to_keep=0)
        normed = cap.states()[FINAL_NORM_KEY]
        self.assertTrue(torch.allclose(self.model.lm_head(normed), out.logits, atol=1e-5))

    def test_grad_flows_when_not_detached(self):
        self.model.requires_grad_(False)
        p = self.model.model.layers[1].self_attn.o_proj.weight
        p.requires_grad_(True)
        with capture_layer_outputs(self.model, [0, 1], detach=False) as cap:
            self.model(input_ids=self.ids, logits_to_keep=1)
        states = cap.states()
        self.assertFalse(states[0].requires_grad)   # upstream of the only trainable parameter
        self.assertTrue(states[1].requires_grad)
        states[1].float().pow(2).mean().backward()
        self.assertIsNotNone(p.grad)

    def test_token_by_token_accumulation_concatenates(self):
        from transformers import DynamicCache
        cache = DynamicCache()
        with capture_layer_outputs(self.model, [1], detach=True) as cap:
            with torch.no_grad():
                self.model(input_ids=self.ids[:, :5], past_key_values=cache)
                self.model(input_ids=self.ids[:, 5:], past_key_values=cache)
        self.assertEqual(tuple(cap.states()[1].shape), (1, 12, 64))

    def test_gather_positions(self):
        with capture_layer_outputs(self.model, [1], detach=True) as cap:
            with torch.no_grad():
                self.model(input_ids=self.ids, logits_to_keep=1)
        g = gather_positions(cap.states(), torch.tensor([0, 11]))
        self.assertEqual(tuple(g[1].shape), (2, 64))
        self.assertTrue(torch.equal(g[1][1], cap.states()[1][0, 11]))
        with self.assertRaises(IndexError):
            with capture_layer_outputs(self.model, [5], detach=True):
                pass


if __name__ == "__main__":
    unittest.main()
