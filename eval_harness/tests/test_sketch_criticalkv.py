"""Tests for CriticalKVSketch / CriticalAdaKVSketch (ports of kvpress
CriticalKVPress / CriticalAdaKVPress, kvpress/presses/criticalkv_press.py).

Expectations are hand-computed or pinned against in-test verbatim
transcriptions of the kvpress reference math. No model loading.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from eval_harness.sketch.attention_patch import attention_patch
from eval_harness.sketch.sketches.criticalkv_sketch import (
    CriticalAdaKVSketch,
    CriticalKVSketch,
)
from eval_harness.sketch.sketches.knorm_sketch import KnormSketch
from eval_harness.sketch.sketches.registry import get_sketch_class
from eval_harness.sketch.sketches.scorer_sketch import ScorerSketch


class _FakeAttnModule(nn.Module):
    def __init__(self, num_heads=2, num_kv_heads=1, head_dim=2, hidden_size=None,
                 o_weight=None, attn_implementation="sdpa", layer_idx=0, seed=0):
        super().__init__()
        hidden_size = num_heads * head_dim if hidden_size is None else hidden_size
        self.num_heads = num_heads
        self.num_key_value_heads = num_kv_heads
        self.num_key_value_groups = num_heads // num_kv_heads
        self.head_dim = head_dim
        self.layer_idx = layer_idx
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        with torch.no_grad():
            if o_weight is not None:
                self.o_proj.weight.copy_(o_weight)
            else:
                torch.manual_seed(seed)
                self.o_proj.weight.normal_()
        self.config = SimpleNamespace(
            num_attention_heads=num_heads,
            head_dim=head_dim,
            hidden_size=hidden_size,
            _attn_implementation=attn_implementation,
        )


class _StubScorer(ScorerSketch):
    """Returns a fixed (cloned) score tensor and counts score() calls."""

    def __init__(self, scores: torch.Tensor, compression_ratio: float = 0.5):
        super().__init__(compression_ratio=compression_ratio)
        self.fixed_scores = scores
        self.score_calls = 0

    def score(self, module, hidden_states, keys, values, attentions, kwargs):
        self.score_calls += 1
        return self.fixed_scores.clone()


def _repeat_kv(x, n_rep):
    b, h_kv, s, d = x.shape
    if n_rep == 1:
        return x
    x = x[:, :, None, :, :].expand(b, h_kv, n_rep, s, d)
    return x.reshape(b, h_kv * n_rep, s, d)


# ----------------------------------------------------------------------
# Verbatim transcriptions of the kvpress reference math (config-based, as
# upstream) used as oracles against the production port.
# ----------------------------------------------------------------------


def _ref_vwl1norm(values, module):
    bsz, num_key_value_heads, k_len, _ = values.shape
    num_key_value_groups = module.config.num_attention_heads // num_key_value_heads
    Wo = module.o_proj.weight.transpose(0, 1)
    Wo = Wo.view(module.config.num_attention_heads, module.config.head_dim, module.config.hidden_size)
    V = _repeat_kv(values, num_key_value_groups)
    head_WoV_norm_list = []
    for head in range(V.size(1)):
        head_WoV = V[:, head, :, ...].matmul(Wo[head, ...].unsqueeze(0))
        head_WoV_norm = torch.norm(head_WoV, p=1, dim=-1)
        head_WoV_norm_list.append(head_WoV_norm)
    WoV_norm = torch.stack(head_WoV_norm_list, dim=1)
    return WoV_norm.view(bsz, num_key_value_heads, module.num_key_value_groups, k_len).mean(dim=2)


def _ref_criticalkv_score(raw_scores, values, module, compression_ratio, epsilon, first_stage_ratio):
    scores = raw_scores.clone()
    k_len = values.shape[2]
    selection_budget = int((1 - compression_ratio) * k_len * first_stage_ratio)
    top_k_index = torch.topk(scores, selection_budget, sorted=True, dim=-1).indices
    projected_norm = _ref_vwl1norm(values, module)
    scores = (scores + epsilon) * projected_norm
    scores.scatter_(-1, top_k_index, torch.finfo(scores.dtype).max)
    return scores


def _ref_scorer_compress(scores, keys, values, module, compression_ratio):
    k_len = keys.shape[2]
    n_kept = int(k_len * (1 - compression_ratio))
    indices = scores.topk(n_kept, dim=-1).indices
    indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)
    return keys.gather(2, indices).contiguous(), values.gather(2, indices).contiguous()


def _ref_critical_adakv_masked(raw_scores, values, module, compression_ratio,
                               alpha_safeguard, epsilon, first_stage_ratio):
    scores = raw_scores.clone()
    bsz, num_key_value_heads, k_len = scores.shape
    n_kept = int(k_len * (1 - compression_ratio))
    n_safe = int(n_kept * alpha_safeguard)
    top_indices = torch.topk(scores, n_safe, dim=-1).indices
    scores.scatter_(-1, top_indices, torch.finfo(scores.dtype).max)

    budget_scores = scores.scatter(-1, top_indices, torch.finfo(scores.dtype).max)
    budget_scores = budget_scores.reshape(bsz, -1)
    top_indices = torch.topk(budget_scores, n_kept * num_key_value_heads, dim=-1).indices
    top_indices_head_idx = top_indices // k_len
    head_budgets = torch.zeros(num_key_value_heads, dtype=torch.int64)
    head_budgets.scatter_add_(0, top_indices_head_idx.flatten(), torch.ones_like(top_indices_head_idx.flatten()))

    head_selection_budget_1st = (head_budgets * first_stage_ratio).to(torch.int64).tolist()
    top_k_index = torch.topk(scores, max(head_selection_budget_1st), sorted=True, dim=-1).indices
    for head_idx in range(num_key_value_heads):
        phase1_budget = head_selection_budget_1st[head_idx]
        scores[:, head_idx, :].scatter_(-1, top_k_index[:, head_idx, :phase1_budget], torch.finfo(scores.dtype).max)

    projected_norm = _ref_vwl1norm(values, module)
    scores = (scores + epsilon) * projected_norm
    top_k_index = torch.topk(scores, max(head_budgets), sorted=True, dim=-1).indices
    for head_idx in range(num_key_value_heads):
        budget = head_budgets[head_idx]
        scores[:, head_idx, :].scatter_(-1, top_k_index[:, head_idx, :budget], torch.finfo(scores.dtype).max)

    n_pruned = num_key_value_heads * (k_len - n_kept)
    indices = torch.topk(-scores.reshape(bsz, -1), n_pruned, dim=1).indices.flatten()
    batch_indices = torch.arange(bsz).repeat_interleave(n_pruned)
    head_indices = indices // k_len
    seq_indices = indices % k_len
    return batch_indices, head_indices, seq_indices


def _masked_set(indices_triplet):
    b, h, s = indices_triplet
    return set(zip(b.tolist(), h.tolist(), s.tolist()))


def _kept_positions(out_keys):
    return set(int(x) for x in out_keys[0, 0, :, 0].tolist())


def _unit_value_module(num_kv_heads=2):
    """H_q == H_kv, head_dim=2, hidden=4, identity o_proj: unit values
    [0.5, 0.5] give projected_norm == 1 everywhere."""
    return _FakeAttnModule(num_heads=num_kv_heads, num_kv_heads=num_kv_heads,
                           head_dim=2, hidden_size=2 * num_kv_heads,
                           o_weight=torch.eye(2 * num_kv_heads))


class TestRegistryAndConstruction(unittest.TestCase):
    def test_registered_names(self):
        self.assertIs(get_sketch_class("criticalkv"), CriticalKVSketch)
        self.assertIs(get_sketch_class("critical_adakv"), CriticalAdaKVSketch)

    def test_non_scorer_press_raises(self):
        with self.assertRaises(AssertionError):
            CriticalKVSketch(press="not a sketch")
        with self.assertRaises(AssertionError):
            CriticalAdaKVSketch(press=None)

    def test_alpha_out_of_bounds_raises(self):
        with self.assertRaises(AssertionError):
            CriticalAdaKVSketch(press=KnormSketch(compression_ratio=0.5), alpha_safeguard=1.1)

    def test_compression_ratio_delegation(self):
        for wrapper in (
            CriticalKVSketch(press=KnormSketch(compression_ratio=0.5)),
            CriticalAdaKVSketch(press=KnormSketch(compression_ratio=0.5)),
        ):
            self.assertAlmostEqual(wrapper.compression_ratio, 0.5)
            wrapper.compression_ratio = 0.3
            self.assertAlmostEqual(wrapper.press.compression_ratio, 0.3)
            wrapper.press.compression_ratio = 0.7
            self.assertAlmostEqual(wrapper.compression_ratio, 0.7)


class TestVWL1Norm(unittest.TestCase):
    def test_hand_computed_exact(self):
        # H_q=2, H_kv=1 (groups=2), S=3, head_dim=2, hidden=4.
        # Wo = weight.T.view(2, 2, 4); per-head L1 norms group-meaned to KV heads.
        Wt = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0],
            [0.0, 0.0, 0.0, 4.0],
        ])
        module = _FakeAttnModule(num_heads=2, num_kv_heads=1, head_dim=2,
                                 hidden_size=4, o_weight=Wt.t())
        values = torch.tensor([[[[1.0, 1.0], [2.0, 0.0], [0.0, 3.0]]]])
        # head0 norms: |v0| + 2|v1| -> [3, 2, 6]; head1: 3|v0| + 4|v1| -> [7, 6, 12]
        expected = torch.tensor([[[5.0, 4.0, 9.0]]])
        out = CriticalKVSketch.vwl1norm(values, module)
        self.assertEqual(out.shape, (1, 1, 3))
        self.assertTrue(torch.equal(out, expected))

    def test_einsum_cross_check(self):
        torch.manual_seed(0)
        module = _FakeAttnModule(num_heads=4, num_kv_heads=2, head_dim=3, hidden_size=12, seed=1)
        values = torch.randn(2, 2, 5, 3)
        Wo = module.o_proj.weight.transpose(0, 1).view(4, 3, 12)
        expected = (
            torch.einsum("bhsd,hdo->bhso", _repeat_kv(values, 2), Wo)
            .abs().sum(-1).view(2, 2, 2, 5).mean(2)
        )
        out = CriticalKVSketch.vwl1norm(values, module)
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)

    def test_gqa_group_mean(self):
        # H_q=4, H_kv=2 (groups=2), head_dim=1: per query head h the norm is
        # |v| * ||Wo[h]||_1; kv-head means must be (2+4)/2 and (6+10)/2.
        Wt = torch.tensor([
            [1.0, 1.0, 0.0, 0.0],
            [2.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 3.0],
            [0.0, 0.0, 5.0, 5.0],
        ])
        module = _FakeAttnModule(num_heads=4, num_kv_heads=2, head_dim=1,
                                 hidden_size=4, o_weight=Wt.t())
        S = 3
        values = torch.arange(1.0, S + 1).view(1, 1, S, 1).expand(1, 2, S, 1).contiguous()
        out = CriticalKVSketch.vwl1norm(values, module)
        self.assertEqual(out.shape, (1, 2, S))
        tokens = torch.arange(1.0, S + 1)
        self.assertTrue(torch.equal(out[0, 0], tokens * 3.0))
        self.assertTrue(torch.equal(out[0, 1], tokens * 8.0))
        # The fake module has no q_proj: queries are never touched.
        self.assertFalse(hasattr(module, "q_proj"))


class TestCriticalKVCompress(unittest.TestCase):
    def _norm_setup(self, norms, scores, ratio=0.5, fsr=0.5, dtype=torch.float32):
        # Identity o_proj (H_q=H_kv=1, head_dim=hidden=4) => projected_norm
        # equals the per-token value L1 norm; keys[0,0,i,:]=i tags positions.
        S = len(norms)
        module = _FakeAttnModule(num_heads=1, num_kv_heads=1, head_dim=4,
                                 hidden_size=4, o_weight=torch.eye(4))
        module.to(dtype)
        values = torch.zeros(1, 1, S, 4, dtype=dtype)
        values[0, 0, :, 0] = torch.tensor(norms, dtype=dtype)
        keys = torch.arange(S, dtype=dtype).view(1, 1, S, 1).expand(1, 1, S, 4).contiguous()
        stub = _StubScorer(torch.tensor([[scores]], dtype=dtype), compression_ratio=ratio)
        sketch = CriticalKVSketch(press=stub, first_stage_ratio=fsr)
        return sketch, module, keys, values

    def test_zero_ratio_noop(self):
        stub = _StubScorer(torch.zeros(1, 1, 8), compression_ratio=0.0)
        sketch = CriticalKVSketch(press=stub)
        module = _FakeAttnModule()
        keys = torch.randn(1, 1, 8, 2)
        values = torch.randn(1, 1, 8, 2)
        out_k, out_v = sketch.compress(module, torch.zeros(1, 8, 4), keys, values, None, {})
        self.assertIs(out_k, keys)
        self.assertIs(out_v, values)
        self.assertEqual(stub.score_calls, 0)

    def test_exact_selection_value_pinned(self):
        # scores [8..1], value L1 norms [1,1,1,10,10,1,1,1], ratio=0.5
        # (n_kept=4), fsr=0.5 -> selection_budget=int(0.5*8*0.5)=2 pins {0,1};
        # rescaled stage-2 ranks 3 (50) and 4 (40) above 2 (6.0006): kept
        # {0,1,3,4}.
        sketch, module, keys, values = self._norm_setup(
            norms=[1, 1, 1, 10, 10, 1, 1, 1],
            scores=[8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        )
        out_k, out_v = sketch.compress(module, torch.zeros(1, 8, 4), keys, values, None, {})
        self.assertEqual(out_k.shape, (1, 1, 4, 4))
        self.assertEqual(_kept_positions(out_k), {0, 1, 3, 4})

    def test_merge_pinning_after_multiply(self):
        # Token 0: top inner score but zero value norm. The post-multiply
        # scatter (criticalkv_press.py:90) restores its finfo.max pin, so it
        # must be kept despite its rescaled score being 0.
        sketch, module, keys, values = self._norm_setup(
            norms=[0, 1, 1, 1],
            scores=[10.0, 1.0, 1.0, 1.0],
        )
        out_k, _ = sketch.compress(module, torch.zeros(1, 4, 4), keys, values, None, {})
        kept = _kept_positions(out_k)
        self.assertEqual(len(kept), 2)
        self.assertIn(0, kept)

    def test_selection_budget_zero(self):
        # fsr=0.01 -> selection_budget=int(4*0.01)=0: topk(k=0) must not raise
        # and selection is purely rescaled-score driven (token 1's tiny norm
        # demotes it; token 2 enters instead).
        sketch, module, keys, values = self._norm_setup(
            norms=[1, 0.01, 1, 10, 10, 1, 1, 1],
            scores=[8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            fsr=0.01,
        )
        out_k, _ = sketch.compress(module, torch.zeros(1, 8, 4), keys, values, None, {})
        self.assertEqual(_kept_positions(out_k), {0, 2, 3, 4})

    def test_bf16_same_kept_set(self):
        sketch, module, keys, values = self._norm_setup(
            norms=[1, 1, 1, 10, 10, 1, 1, 1],
            scores=[8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            dtype=torch.bfloat16,
        )
        out_k, out_v = sketch.compress(
            module, torch.zeros(1, 8, 4, dtype=torch.bfloat16), keys, values, None, {}
        )
        self.assertFalse(torch.isnan(out_k).any())
        self.assertFalse(torch.isnan(out_v).any())
        self.assertEqual(_kept_positions(out_k), {0, 1, 3, 4})

    def test_odd_length(self):
        sketch, module, keys, values = self._norm_setup(
            norms=[1, 1, 1, 1, 1, 1, 1],
            scores=[7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        )
        out_k, out_v = sketch.compress(module, torch.zeros(1, 7, 4), keys, values, None, {})
        self.assertEqual(out_k.shape, (1, 1, 3, 4))
        self.assertEqual(out_v.shape, (1, 1, 3, 4))


class TestCriticalKVReferenceParity(unittest.TestCase):
    def test_matches_kvpress_transcription(self):
        torch.manual_seed(3)
        module = _FakeAttnModule(num_heads=4, num_kv_heads=2, head_dim=8,
                                 hidden_size=32, seed=4)
        keys = torch.randn(1, 2, 16, 8)
        values = torch.randn(1, 2, 16, 8)
        hidden = torch.zeros(1, 16, 32)
        for ratio in (0.25, 0.5):
            for fsr in (0.0, 0.5, 1.0):
                with self.subTest(ratio=ratio, fsr=fsr):
                    sketch = CriticalKVSketch(
                        press=KnormSketch(compression_ratio=ratio), first_stage_ratio=fsr
                    )
                    out_k, out_v = sketch.compress(module, hidden, keys, values, None, {})

                    raw_scores = -keys.norm(dim=-1)
                    ref_scores = _ref_criticalkv_score(raw_scores, values, module, ratio, 1e-4, fsr)
                    ref_k, ref_v = _ref_scorer_compress(ref_scores, keys, values, module, ratio)
                    self.assertTrue(torch.equal(out_k, ref_k))
                    self.assertTrue(torch.equal(out_v, ref_v))


class TestCriticalAdaKVCompress(unittest.TestCase):
    def test_zero_ratio_noop_even_under_eager(self):
        stub = _StubScorer(torch.zeros(1, 2, 4), compression_ratio=0.0)
        sketch = CriticalAdaKVSketch(press=stub)
        module = _FakeAttnModule(attn_implementation="eager")
        keys = torch.randn(1, 2, 4, 2)
        values = torch.randn(1, 2, 4, 2)
        out_k, out_v = sketch.compress(module, torch.zeros(1, 4, 4), keys, values, None, {})
        self.assertIs(out_k, keys)
        self.assertIs(out_v, values)
        self.assertEqual(stub.score_calls, 0)
        self.assertFalse(hasattr(module, "masked_key_indices"))

    def test_eager_raises_for_nonzero_ratio(self):
        sketch = CriticalAdaKVSketch(press=KnormSketch(compression_ratio=0.5))
        module = _FakeAttnModule(attn_implementation="eager")
        keys = torch.randn(1, 1, 4, 2)
        with self.assertRaisesRegex(AssertionError, "eager"):
            sketch.compress(module, torch.zeros(1, 4, 4), keys, keys.clone(), None, {})

    def test_hand_computed_masked_indices(self):
        # H_kv=H_q=2, S=4, ratio=0.5 (n_kept=2), alpha=0.5 (n_safe=1), fsr=0.5,
        # unit projected norms. Trace: safeguard pins (h0,0),(h1,3); global
        # top-4 -> head_budgets [2,2]; stage1 budgets [1,1] re-pin; stage2 pins
        # h0 {0,1}, h1 {3,2}; bottom-4 = (0,2),(0,3),(1,0),(1,1).
        module = _unit_value_module()
        scores = torch.tensor([[[10.0, 9.0, 1.0, 1.0], [2.0, 3.0, 4.0, 5.0]]])
        sketch = CriticalAdaKVSketch(
            press=_StubScorer(scores, compression_ratio=0.5),
            alpha_safeguard=0.5,
            first_stage_ratio=0.5,
        )
        keys = torch.randn(1, 2, 4, 2)
        values = torch.full((1, 2, 4, 2), 0.5)
        out_k, out_v = sketch.compress(module, torch.zeros(1, 4, 4), keys, values, None, {})

        self.assertIs(out_k, keys)
        self.assertIs(out_v, values)
        self.assertEqual(out_k.shape[2], 4)
        expected = {(0, 0, 2), (0, 0, 3), (0, 1, 0), (0, 1, 1)}
        self.assertEqual(_masked_set(module.masked_key_indices), expected)
        self.assertTrue((module.masked_key_indices[0] == 0).all())

    def test_zero_budget_head_reallocation(self):
        # alpha=0 and h0 dominating all global slots -> head_budgets=[4, 0]:
        # the empty per-head stage slices must not raise, and every h1 token is
        # masked while h0 keeps all four (budget above the uniform n_kept=2).
        module = _unit_value_module()
        scores = torch.tensor([[[40.0, 30.0, 20.0, 10.0], [4.0, 3.0, 2.0, 1.0]]])
        sketch = CriticalAdaKVSketch(
            press=_StubScorer(scores, compression_ratio=0.5),
            alpha_safeguard=0.0,
            first_stage_ratio=0.5,
        )
        keys = torch.randn(1, 2, 4, 2)
        values = torch.full((1, 2, 4, 2), 0.5)
        sketch.compress(module, torch.zeros(1, 4, 4), keys, values, None, {})
        expected = {(0, 1, s) for s in range(4)}
        self.assertEqual(_masked_set(module.masked_key_indices), expected)

    def test_alpha_one_reduces_to_per_head_selection(self):
        # alpha=1 -> n_safe=n_kept, uniform head_budgets: masked set is each
        # head's bottom S-n_kept by the inner score.
        module = _unit_value_module()
        scores = torch.tensor([[[10.0, 9.0, 1.0, 2.0], [5.0, 6.0, 7.0, 8.0]]])
        sketch = CriticalAdaKVSketch(
            press=_StubScorer(scores, compression_ratio=0.5),
            alpha_safeguard=1.0,
        )
        keys = torch.randn(1, 2, 4, 2)
        values = torch.full((1, 2, 4, 2), 0.5)
        sketch.compress(module, torch.zeros(1, 4, 4), keys, values, None, {})
        expected = {(0, 0, 2), (0, 0, 3), (0, 1, 0), (0, 1, 1)}
        self.assertEqual(_masked_set(module.masked_key_indices), expected)

    def test_batch_size_guard(self):
        # kvpress accumulates head budgets across the batch (scatter_add_ into
        # one (H_kv,) tensor); the port asserts bsz == 1 instead.
        module = _unit_value_module()
        scores = torch.rand(2, 2, 4)
        sketch = CriticalAdaKVSketch(press=_StubScorer(scores, compression_ratio=0.5))
        keys = torch.randn(2, 2, 4, 2)
        with self.assertRaisesRegex(AssertionError, "batch size 1"):
            sketch.compress(module, torch.zeros(2, 4, 4), keys, keys.clone(), None, {})

    def test_odd_length_masked_count(self):
        torch.manual_seed(5)
        module = _unit_value_module()
        scores = torch.rand(1, 2, 7)
        sketch = CriticalAdaKVSketch(press=_StubScorer(scores, compression_ratio=0.5))
        keys = torch.randn(1, 2, 7, 2)
        values = torch.full((1, 2, 7, 2), 0.5)
        out_k, _ = sketch.compress(module, torch.zeros(1, 7, 4), keys, values, None, {})
        self.assertEqual(out_k.shape[2], 7)
        # n_kept = int(7*0.5) = 3 -> n_pruned = 2*4 = 8.
        b, h, s = module.masked_key_indices
        self.assertEqual(b.numel(), 8)
        self.assertTrue((s >= 0).all() and (s < 7).all())


class TestCriticalAdaKVReferenceParity(unittest.TestCase):
    def test_matches_kvpress_transcription(self):
        torch.manual_seed(6)
        module = _FakeAttnModule(num_heads=4, num_kv_heads=2, head_dim=8,
                                 hidden_size=32, seed=7)
        keys = torch.randn(1, 2, 16, 8)
        values = torch.randn(1, 2, 16, 8)
        hidden = torch.zeros(1, 16, 32)
        for ratio in (0.25, 0.5):
            for fsr in (0.0, 0.5, 1.0):
                for alpha in (0.0, 0.2, 1.0):
                    with self.subTest(ratio=ratio, fsr=fsr, alpha=alpha):
                        module.masked_key_indices = None
                        sketch = CriticalAdaKVSketch(
                            press=KnormSketch(compression_ratio=ratio),
                            alpha_safeguard=alpha,
                            first_stage_ratio=fsr,
                        )
                        out_k, out_v = sketch.compress(module, hidden, keys, values, None, {})
                        self.assertIs(out_k, keys)
                        self.assertIs(out_v, values)

                        raw_scores = -keys.norm(dim=-1)
                        ref = _ref_critical_adakv_masked(
                            raw_scores, values, module, ratio, alpha, 1e-4, fsr
                        )
                        self.assertEqual(
                            _masked_set(module.masked_key_indices), _masked_set(ref)
                        )
                        self.assertEqual(
                            module.masked_key_indices[0].numel(),
                            2 * (16 - int(16 * (1 - ratio))),
                        )


class TestDecodeMaskIntegration(unittest.TestCase):
    @staticmethod
    def _plain_attention(module, query, key, value, attention_mask, dropout, **kwargs):
        num_groups = query.shape[1] // key.shape[1]
        k = key.repeat_interleave(num_groups, dim=1)
        v = value.repeat_interleave(num_groups, dim=1)
        logits = query @ k.transpose(-1, -2) / (query.shape[-1] ** 0.5)
        weights = torch.softmax(logits, dim=-1)
        return weights @ v, weights

    def test_masked_attention_equals_physical_pruning(self):
        torch.manual_seed(8)
        module = SimpleNamespace(
            masked_key_indices=(torch.tensor([0]), torch.tensor([0]), torch.tensor([3]))
        )
        H_q, S, D = 2, 8, 4
        q = torch.randn(1, H_q, 1, D)
        k = torch.randn(1, 1, S, D)
        v = torch.randn(1, 1, S, D)
        out, _ = attention_patch(self._plain_attention)(module, q, k.clone(), v, None, 0.0)

        kept = [s for s in range(S) if s != 3]
        for h in range(H_q):
            logits = q[0, h, 0] @ k[0, 0, kept].T / (D ** 0.5)
            ref = torch.softmax(logits, dim=-1) @ v[0, 0, kept]
            torch.testing.assert_close(out[0, h, 0], ref, rtol=1e-5, atol=1e-6)

    def test_prefill_resets_masked_indices(self):
        module = SimpleNamespace(
            masked_key_indices=(torch.tensor([0]), torch.tensor([0]), torch.tensor([3]))
        )
        q = torch.randn(1, 1, 8, 4)
        k = torch.randn(1, 1, 8, 4)
        k_orig = k.clone()
        attention_patch(self._plain_attention)(module, q, k, k.clone(), None, 0.0)
        self.assertIsNone(module.masked_key_indices)
        self.assertTrue(torch.equal(k, k_orig))


if __name__ == "__main__":
    unittest.main()
