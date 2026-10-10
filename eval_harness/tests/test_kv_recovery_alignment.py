"""Alignment objective: layer / position resolution and the losses (CPU, no models)."""
from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from eval_harness.kv_recovery.alignment import (
    LOSSES,
    alignment_keys_for,
    check_alignment_has_gradient,
    combine,
    dead_alignment_keys,
    hidden_loss,
    kl_loss,
    position_index,
    resolve_layers,
)
from eval_harness.kv_recovery.config import AlignmentCfg, LayersCfg, LossCfg, PositionsCfg


class TestLayersAndPositions(unittest.TestCase):
    def test_resolve_layers(self):
        self.assertEqual(resolve_layers(LayersCfg("last_n", 2), 6), [4, 5])
        self.assertEqual(resolve_layers(LayersCfg("explicit", indices=[3, 1, 3]), 6), [1, 3])
        self.assertEqual(resolve_layers(LayersCfg("all"), 3), [0, 1, 2])
        self.assertEqual(resolve_layers(LayersCfg("from_first_trainable"), 6, first_trainable_layer=4), [4, 5])
        with self.assertRaises(ValueError):
            resolve_layers(LayersCfg("last_n", 7), 6)
        with self.assertRaises(ValueError):
            resolve_layers(LayersCfg("explicit", indices=[6]), 6)
        with self.assertRaises(ValueError):
            resolve_layers(LayersCfg("from_first_trainable"), 6, first_trainable_layer=None)

    def test_dead_terms(self):
        keys = [2, 3, 4, "norm"]
        self.assertEqual(dead_alignment_keys(keys, 4), [2, 3])
        self.assertEqual(dead_alignment_keys(keys, 0), [])
        self.assertEqual(dead_alignment_keys(keys, None), keys)
        with self.assertRaises(ValueError):
            check_alignment_has_gradient(keys, 4, allow=False)
        self.assertEqual(check_alignment_has_gradient(keys, 4, allow=True), [2, 3])
        self.assertEqual(check_alignment_has_gradient([4, "norm"], 4, allow=False), [])

    def test_alignment_keys_for(self):
        acfg = AlignmentCfg(layers=LayersCfg("last_n", 2), include_final_norm=True)
        self.assertEqual(alignment_keys_for(acfg, 5, first_trainable_layer=None), [3, 4, "norm"])
        acfg = AlignmentCfg(layers=LayersCfg("last_n", 1), include_final_norm=False)
        self.assertEqual(alignment_keys_for(acfg, 5, first_trainable_layer=None), [4])

    def test_position_index(self):
        self.assertEqual(position_index(PositionsCfg("all"), 5).tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(position_index(PositionsCfg("recent", 2), 5).tolist(), [3, 4])
        self.assertEqual(position_index(PositionsCfg("recent", 9), 5).tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(position_index(PositionsCfg("first_k", 2), 5).tolist(), [0, 1])
        self.assertEqual(position_index(PositionsCfg("post_eviction"), 5).tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(position_index(PositionsCfg("post_eviction"), 5, first_affected=3).tolist(), [3, 4])
        with self.assertRaises(ValueError):
            position_index(PositionsCfg("all"), 0)


class TestLosses(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.t = torch.randn(7, 16)
        self.s = self.t + 0.1 * torch.randn(7, 16)

    def test_identical_inputs_give_zero(self):
        for name, fn in LOSSES.items():
            with self.subTest(loss=name):
                self.assertEqual(float(fn(self.t, self.t).sum()), 0.0)

    def test_normalized_mse_is_twice_one_minus_cos(self):
        nm = LOSSES["normalized_mse"](self.s, self.t)
        cos = LOSSES["cosine"](self.s, self.t)
        self.assertTrue(torch.allclose(nm, 2.0 * cos, atol=1e-6))
        self.assertTrue(bool((cos >= 0).all()) and bool((cos <= 2).all()))

    def test_elementwise_matches_spec_snippet(self):
        spec = F.mse_loss(F.normalize(self.s.float(), dim=-1), F.normalize(self.t.float(), dim=-1))
        ours = LOSSES["normalized_mse_elementwise"](self.s, self.t).mean()
        self.assertAlmostEqual(float(spec), float(ours), places=7)
        # and it is normalized_mse / H
        self.assertAlmostEqual(float(ours), float(LOSSES["normalized_mse"](self.s, self.t).mean()) / 16, places=7)

    def test_relative_mse_scale_aware(self):
        scaled = 2.0 * self.t
        self.assertEqual(float(LOSSES["cosine"](scaled, self.t).sum()), 0.0)
        self.assertGreater(float(LOSSES["relative_mse"](scaled, self.t).mean()), 0.5)

    def test_hidden_loss_mean_over_layers_and_buckets(self):
        student = {0: self.s, 1: self.s * 0.5 + 1.0, "norm": self.t.clone()}
        teacher = {0: self.t, 1: self.t, "norm": self.t}
        positions = torch.tensor([0, 1, 2, 20, 30, 70, 90])
        loss, per_layer, per_bucket = hidden_loss(student, teacher, "normalized_mse", [0, 1, "norm"], positions=positions)
        self.assertEqual(set(per_layer), {"0", "1", "norm"})
        self.assertAlmostEqual(float(loss), sum(per_layer.values()) / 3, places=6)
        self.assertEqual(per_layer["norm"], 0.0)
        self.assertEqual(set(per_bucket), {"0-16", "16-64", "64-end"})
        # weighted: all weight on the identical layer -> 0
        loss_w, _, _ = hidden_loss(student, teacher, "normalized_mse", [0, "norm"], layer_weights=[0.0, 1.0])
        self.assertEqual(float(loss_w), 0.0)
        with self.assertRaises(ValueError):
            hidden_loss(student, teacher, "normalized_mse", [0], layer_weights=[1.0, 1.0])
        with self.assertRaises(ValueError):
            hidden_loss(student, teacher, "l1", [0])

    def test_masked_positions_are_excluded(self):
        # Gathering happens before the loss: positions outside the gathered set never contribute.
        full_s = torch.randn(1, 10, 8)
        full_t = full_s.clone()
        full_s[0, 5:] += 10.0   # corrupt the tail
        from eval_harness.kv_recovery.hidden_states import gather_positions
        pos = torch.arange(0, 5)
        loss, _, _ = hidden_loss(gather_positions({0: full_s}, pos), gather_positions({0: full_t}, pos), "cosine", [0])
        self.assertEqual(float(loss), 0.0)

    def test_kl_matches_manual(self):
        torch.manual_seed(1)
        zs, zt, T = torch.randn(1, 4, 11), torch.randn(1, 4, 11), 2.0
        pt = F.softmax(zt / T, -1)
        manual = (pt * (F.log_softmax(zt / T, -1) - F.log_softmax(zs / T, -1))).sum(-1).mean() * T * T
        self.assertAlmostEqual(float(kl_loss(zs, zt, T)), float(manual), places=6)
        self.assertAlmostEqual(float(kl_loss(zt, zt, T)), 0.0, places=6)

    def test_combine(self):
        h, k = torch.tensor(2.0), torch.tensor(3.0)
        self.assertEqual(float(combine(h, k, LossCfg(1.0, 0.0, 1.0))), 2.0)
        self.assertEqual(float(combine(h, k, LossCfg(1.0, 0.5, 1.0))), 3.5)
        self.assertEqual(float(combine(None, k, LossCfg(1.0, 0.5, 1.0))), 1.5)
        with self.assertRaises(ValueError):
            combine(None, None, LossCfg(1.0, 0.0, 1.0))


if __name__ == "__main__":
    unittest.main()
