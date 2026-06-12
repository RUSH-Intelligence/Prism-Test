"""Tests for FastKVzipSketch — port of kvpress FastKVzipPress (fastkvzip_press.py).

No model loading, no network: gates are injected via the constructor and the
HF-hub loader is stubbed with ``unittest.mock``. The kvpress ``compress_post``
math is transcribed in-test as a reference oracle, and the gate forward is
checked against an explicit-loop reimplementation.
"""

import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from eval_harness.sketch.sketches.fastkvzip_sketch import (
    FastKVzipGate,
    FastKVzipSketch,
    load_fastkvzip,
)
from eval_harness.sketch.sketches.registry import available_sketches, get_sketch, get_sketch_class


class _FakeAttnModule(nn.Module):
    def __init__(self, layer_idx=0, head_dim=4, attn_implementation="sdpa"):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = head_dim
        self.config = SimpleNamespace(_attn_implementation=attn_implementation)

    def forward(self, hidden_states=None, cache_position=None, past_key_values=None, **kwargs):
        return (hidden_states, None)


class _StubGate(nn.Module):
    """Returns a fixed (1, H, S) score tensor (cloned: the sketch mutates scores in place)."""

    def __init__(self, scores):
        super().__init__()
        self.scores = scores
        self.calls = 0

    def forward(self, hidden_states):
        self.calls += 1
        return self.scores.clone()


class _RaisingGate(nn.Module):
    def forward(self, hidden_states):
        raise AssertionError("gate must not be called")


def _seeded_gate(seed=0, input_dim=8, nhead=2, ngroup=3, output_dim=4, sink=2):
    gate = FastKVzipGate(
        index=0, input_dim=input_dim, nhead=nhead, ngroup=ngroup,
        dtype=torch.float32, output_dim=output_dim, sink=sink,
    )
    torch.manual_seed(seed)
    with torch.no_grad():
        for p in gate.parameters():
            p.copy_(torch.randn_like(p))
    return gate


def _rmsnorm_ref(x, weight, eps=1e-6):
    variance = x.pow(2).mean(-1, keepdim=True)
    return weight * (x * torch.rsqrt(variance + eps))


def _reference_gate_scores(gate, hidden):
    """Explicit-loop reimplementation of the FastKVzipGate math.

    Returns (final (1,H,S) scores, per-group (H,S,G) scores, logits, base logits).
    """
    hs = hidden[0]
    n_seq = hs.shape[0]
    H, G, d_out, sink = gate.nhead, gate.ngroup, gate.output_dim, gate.sink
    scale = math.sqrt(d_out)

    out = torch.zeros(1, H, n_seq)
    per_group = torch.zeros(H, n_seq, G)
    logits = torch.zeros(H, n_seq, G)
    base_logits = torch.zeros(H, n_seq, sink, G)
    for s in range(n_seq):
        q_lin = gate.q_proj.weight @ hs[s] + gate.q_proj.bias
        k_lin = gate.k_proj.weight @ hs[s]
        q = _rmsnorm_ref(q_lin.view(H, G, d_out), gate.q_norm.weight)
        k = _rmsnorm_ref(k_lin.view(H, 1, d_out), gate.k_norm.weight)
        for h in range(H):
            for g in range(G):
                logit = torch.dot(k[h, 0], q[h, g]) / scale + gate.b[h, 0, g]
                acc = torch.zeros(())
                for j in range(sink):
                    base = torch.dot(gate.k_base[h, 0, j], q[h, g]) / scale
                    base_logits[h, s, j, g] = base
                    acc = acc + torch.exp(base - logit)
                logits[h, s, g] = logit
                per_group[h, s, g] = 1.0 / (1.0 + acc)
            out[0, h, s] = per_group[h, s].mean()
    return out, per_group, logits, base_logits


def _kvpress_compress_post_reference(score_stack, compression_ratio, layerwise):
    """Verbatim transcription of kvpress FastKVzipPress.compress_post (lines 256-287)."""
    n_layer, bsz, num_key_value_heads, ctx_len = score_stack.shape

    if layerwise:
        nl = int(bsz * num_key_value_heads * ctx_len * compression_ratio)
        n_pruned_layers = nl * torch.ones(n_layer, device=score_stack.device, dtype=torch.int)
    else:
        n_pruned_indices = int(score_stack.numel() * compression_ratio)
        pruned_indices = torch.topk(-score_stack.reshape(-1), n_pruned_indices).indices
        n_tokens_per_layer = bsz * num_key_value_heads * ctx_len
        n_pruned_layers = torch.bincount(pruned_indices // n_tokens_per_layer, minlength=n_layer).int()

    per_layer = []
    for layer_idx in range(n_layer):
        scores = score_stack[layer_idx]
        n_pruned = n_pruned_layers[layer_idx].cpu()
        indices = torch.topk(-scores.reshape(bsz, -1), n_pruned, dim=1).indices.flatten().cpu()
        batch_indices = torch.arange(bsz, device=n_pruned.device).repeat_interleave(n_pruned)
        head_indices = indices // ctx_len
        seq_indices = indices % ctx_len
        per_layer.append((batch_indices, head_indices, seq_indices))
    return per_layer


def _prefill_kwargs(hidden):
    seq = hidden.shape[1]
    return {"hidden_states": hidden, "cache_position": torch.arange(seq)}


def _run_prefill(sketch, module, hidden):
    output = (hidden, None)
    result = sketch.forward_hook(module, [], _prefill_kwargs(hidden), output)
    return result, output


class TestFastKVzipRegistry(unittest.TestCase):
    def test_registered_name_resolves(self):
        self.assertIn("fastkvzip", available_sketches())
        self.assertIs(get_sketch_class("fastkvzip"), FastKVzipSketch)

    def test_get_sketch_instantiates_with_kwargs(self):
        sketch = get_sketch("fastkvzip", compression_ratio=0.3, layerwise=True, n_sink=8)
        self.assertIsInstance(sketch, FastKVzipSketch)
        self.assertAlmostEqual(sketch.compression_ratio, 0.3)
        self.assertTrue(sketch.layerwise)
        self.assertEqual(sketch.n_sink, 8)
        self.assertIsNone(sketch.gates)

    def test_ratio_validation(self):
        with self.assertRaises(AssertionError):
            FastKVzipSketch(compression_ratio=1.0)
        with self.assertRaises(AssertionError):
            FastKVzipSketch(compression_ratio=-0.1)


class TestFastKVzipGateMath(unittest.TestCase):
    def test_gate_forward_matches_loop_reference(self):
        gate = _seeded_gate()
        torch.manual_seed(123)
        hidden = torch.randn(1, 5, 8)

        actual = gate(hidden)
        expected, _, _, _ = _reference_gate_scores(gate, hidden)

        self.assertEqual(tuple(actual.shape), (1, 2, 5))
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_score_is_token_vs_sinks_softmax_probability(self):
        gate = _seeded_gate(seed=7)
        torch.manual_seed(11)
        hidden = torch.randn(1, 4, 8)

        actual = gate(hidden)
        _, per_group, logits, base_logits = _reference_gate_scores(gate, hidden)

        for h in range(gate.nhead):
            for s in range(hidden.shape[1]):
                for g in range(gate.ngroup):
                    stacked = torch.cat([logits[h, s, g].view(1), base_logits[h, s, :, g]])
                    softmax_first = torch.softmax(stacked, dim=0)[0]
                    torch.testing.assert_close(per_group[h, s, g], softmax_first, atol=1e-6, rtol=1e-6)
                torch.testing.assert_close(
                    actual[0, h, s], per_group[h, s].mean(), atol=1e-6, rtol=1e-6
                )

    def test_gate_scores_strictly_in_unit_interval(self):
        gate = _seeded_gate(seed=3)
        torch.manual_seed(5)
        scores = gate(torch.randn(1, 16, 8))
        self.assertTrue((scores > 0).all())
        self.assertTrue((scores < 1).all())

    def test_gqa_gate_score_shape_is_kv_heads(self):
        gate = _seeded_gate(seed=1, nhead=2, ngroup=4)  # H_q = 8, H_kv = 2
        torch.manual_seed(2)
        scores = gate(torch.randn(1, 5, 8))
        self.assertEqual(tuple(scores.shape), (1, 2, 5))


class TestFastKVzipScoreOverrides(unittest.TestCase):
    def _scored(self, sketch, stub_scores, seq):
        sketch.gates = [_StubGate(stub_scores)]
        module = _FakeAttnModule(layer_idx=0)
        _run_prefill(sketch, module, torch.randn(1, seq, 8))
        return sketch.score_val[0], module

    def test_sink_and_window_overrides(self):
        sketch = FastKVzipSketch(compression_ratio=0.5, n_sink=4, window_ratio=0.05)
        scores, _ = self._scored(sketch, torch.full((1, 1, 200), 0.5), 200)
        self.assertTrue((scores[:, :, :4] == 1.0).all())
        self.assertTrue((scores[:, :, 190:] == 1.0).all())
        self.assertTrue((scores[:, :, 4:190] == 0.5).all())

    def test_window_zero_guard_keeps_gate_scores(self):
        # S=40, window_ratio=0.02 -> int(0.8) == 0. kvpress would execute
        # scores[:, :, -0:] = 1.0 (whole axis); this port guards window_size > 0
        # (documented deviation), so only the n_sink prefix is protected.
        sketch = FastKVzipSketch(compression_ratio=0.5, n_sink=4, window_ratio=0.02)
        scores, _ = self._scored(sketch, torch.full((1, 1, 40), 0.5), 40)
        self.assertTrue((scores[:, :, :4] == 1.0).all())
        self.assertTrue((scores[:, :, 4:] == 0.5).all())

    def test_long_context_uses_fixed_window_size(self):
        sketch = FastKVzipSketch(compression_ratio=0.5, n_sink=0, window_size=4096, window_ratio=0.02)
        scores, _ = self._scored(sketch, torch.zeros(1, 1, 32000), 32000)
        self.assertTrue((scores[:, :, -4096:] == 1.0).all())
        self.assertEqual(scores[0, 0, 32000 - 4097].item(), 0.0)

    def test_threshold_boundary_below_32000_uses_ratio(self):
        sketch = FastKVzipSketch(compression_ratio=0.5, n_sink=0, window_size=4096, window_ratio=0.02)
        scores, _ = self._scored(sketch, torch.zeros(1, 1, 31999), 31999)
        window = int(31999 * 0.02)  # 639
        self.assertTrue((scores[:, :, -window:] == 1.0).all())
        self.assertEqual(scores[0, 0, 31999 - window - 1].item(), 0.0)

    def test_n_sink_larger_than_context_truncates(self):
        sketch = FastKVzipSketch(compression_ratio=0.5, n_sink=4, window_ratio=0.02)
        scores, _ = self._scored(sketch, torch.full((1, 1, 3), 0.5), 3)
        self.assertTrue((scores == 1.0).all())

    def test_batch_size_two_raises_loudly(self):
        sketch = FastKVzipSketch(compression_ratio=0.5, gates=[_RaisingGate()])
        module = _FakeAttnModule(layer_idx=0)
        hidden = torch.randn(2, 5, 8)
        with self.assertRaisesRegex(AssertionError, "batch size 1"):
            sketch.forward_hook(module, [], _prefill_kwargs(hidden), (hidden, None))

    def test_decode_step_is_noop_and_gate_not_called(self):
        sketch = FastKVzipSketch(compression_ratio=0.5, gates=[_RaisingGate()])
        module = _FakeAttnModule(layer_idx=0)
        hidden = torch.randn(1, 1, 8)
        kwargs = {"hidden_states": hidden, "cache_position": torch.tensor([11])}
        output = (hidden, None)
        result = sketch.forward_hook(module, [], kwargs, output)
        self.assertIs(result, output)
        self.assertEqual(sketch.score_val, {})


class TestFastKVzipCompressPost(unittest.TestCase):
    def test_exact_selection_masks_bottom_scores_and_leaves_cache_untouched(self):
        from transformers import DynamicCache

        stub_scores = torch.tensor([[[0.9, 0.1, 0.8, 0.2, 0.7, 0.3]]])
        sketch = FastKVzipSketch(
            compression_ratio=0.5, layerwise=True, n_sink=0, window_ratio=0.0,
            gates=[_StubGate(stub_scores)],
        )
        module = _FakeAttnModule(layer_idx=0, head_dim=4)
        keys = torch.randn(1, 1, 6, 4)
        values = torch.randn(1, 1, 6, 4)
        cache = DynamicCache()
        cache.update(keys.clone(), values.clone(), 0)

        hidden = torch.randn(1, 6, 8)
        kwargs = dict(_prefill_kwargs(hidden), past_key_values=cache)
        sketch.forward_hook(module, [], kwargs, (hidden, None))
        sketch.compress_post()

        batch_indices, head_indices, seq_indices = module.masked_key_indices
        self.assertTrue(torch.equal(batch_indices, torch.tensor([0, 0, 0])))
        self.assertTrue(torch.equal(head_indices, torch.tensor([0, 0, 0])))
        self.assertTrue(torch.equal(seq_indices, torch.tensor([1, 3, 5])))

        # Fake compression: the cache keeps its full physical length and contents.
        self.assertEqual(cache.layers[0].keys.shape[2], 6)
        self.assertTrue(torch.equal(cache.layers[0].keys, keys))
        self.assertTrue(torch.equal(cache.layers[0].values, values))

    def test_zero_ratio_scores_but_never_masks(self):
        # kvpress parity: the gate runs during prefill even at ratio 0 (the hook
        # always scores); compress_post then writes no masked_key_indices.
        gate = _StubGate(torch.full((1, 1, 6), 0.5))
        sketch = FastKVzipSketch(compression_ratio=0.0, n_sink=0, window_ratio=0.0, gates=[gate])
        module = _FakeAttnModule(layer_idx=0)
        _run_prefill(sketch, module, torch.randn(1, 6, 8))
        sketch.compress_post()
        self.assertEqual(gate.calls, 1)
        self.assertIn(0, sketch.score_val)
        self.assertIsNone(getattr(module, "masked_key_indices", None))

    def test_per_head_ragged_budget_matches_kvpress(self):
        # head0 scores all 0.9, head1 all 0.1, S=8, ratio=0.5, layerwise=True:
        # n_pruned = int(1*2*8*0.5) = 8 and the pooled bottom-k puts ALL 8 pruned
        # entries in head1 (kvpress per-head ragged masking, kept faithfully).
        stub_scores = torch.cat([torch.full((1, 1, 8), 0.9), torch.full((1, 1, 8), 0.1)], dim=1)
        sketch = FastKVzipSketch(
            compression_ratio=0.5, layerwise=True, n_sink=0, window_ratio=0.0,
            gates=[_StubGate(stub_scores)],
        )
        module = _FakeAttnModule(layer_idx=0)
        _run_prefill(sketch, module, torch.randn(1, 8, 8))
        sketch.compress_post()

        batch_indices, head_indices, seq_indices = module.masked_key_indices
        self.assertEqual(batch_indices.numel(), 8)
        self.assertTrue((head_indices == 1).all())
        self.assertEqual(sorted(seq_indices.tolist()), list(range(8)))

        ref = _kvpress_compress_post_reference(stub_scores.unsqueeze(0), 0.5, layerwise=True)
        self.assertTrue(torch.equal(head_indices, ref[0][1]))
        self.assertTrue(torch.equal(seq_indices, ref[0][2]))

    def test_global_allocation_concentrates_pruning_in_low_score_layer(self):
        # layerwise=False (kvpress default): bottom-k over BOTH layers' scores,
        # bincount allocation -> layer0 (all 0.9) prunes nothing, layer1 (all 0.1)
        # prunes everything.
        g0 = _StubGate(torch.full((1, 1, 4), 0.9))
        g1 = _StubGate(torch.full((1, 1, 4), 0.1))
        sketch = FastKVzipSketch(
            compression_ratio=0.5, layerwise=False, n_sink=0, window_ratio=0.0, gates=[g0, g1],
        )
        m0 = _FakeAttnModule(layer_idx=0)
        m1 = _FakeAttnModule(layer_idx=1)
        hidden = torch.randn(1, 4, 8)
        sketch.forward_hook(m0, [], _prefill_kwargs(hidden), (hidden, None))
        sketch.forward_hook(m1, [], _prefill_kwargs(hidden), (hidden, None))
        sketch.compress_post()

        b0, h0, s0 = m0.masked_key_indices
        self.assertEqual(b0.numel(), 0)
        self.assertEqual(h0.numel(), 0)
        self.assertEqual(s0.numel(), 0)

        b1, h1, s1 = m1.masked_key_indices
        self.assertTrue((h1 == 0).all())
        self.assertEqual(sorted(s1.tolist()), [0, 1, 2, 3])

    def test_randomized_oracle_against_kvpress_transcription(self):
        torch.manual_seed(42)
        n_layer, h_kv, seq = 3, 2, 7
        stacks = torch.rand(n_layer, 1, h_kv, seq)
        for layerwise in (True, False):
            sketch = FastKVzipSketch(
                compression_ratio=0.43, layerwise=layerwise, n_sink=0, window_ratio=0.0,
                gates=[_StubGate(stacks[i]) for i in range(n_layer)],
            )
            modules = [_FakeAttnModule(layer_idx=i) for i in range(n_layer)]
            hidden = torch.randn(1, seq, 8)
            for module in modules:
                sketch.forward_hook(module, [], _prefill_kwargs(hidden), (hidden, None))
            sketch.compress_post()

            expected = _kvpress_compress_post_reference(stacks, 0.43, layerwise)
            for module, (ref_b, ref_h, ref_s) in zip(modules, expected):
                act_b, act_h, act_s = module.masked_key_indices
                self.assertTrue(torch.equal(act_b, ref_b), msg=f"layerwise={layerwise}")
                self.assertTrue(torch.equal(act_h, ref_h), msg=f"layerwise={layerwise}")
                self.assertTrue(torch.equal(act_s, ref_s), msg=f"layerwise={layerwise}")

    def test_protected_positions_never_pruned_with_real_gate(self):
        # Gate scores are strictly < 1, so the 1.0 sink/window overrides dominate
        # the bottom-k whenever the pruned count fits in the unprotected region.
        gate = _seeded_gate(seed=9, nhead=1, ngroup=2)
        sketch = FastKVzipSketch(
            compression_ratio=0.5, layerwise=True, n_sink=2, window_ratio=0.15, gates=[gate],
        )
        module = _FakeAttnModule(layer_idx=0)
        torch.manual_seed(10)
        _run_prefill(sketch, module, torch.randn(1, 20, 8))
        sketch.compress_post()

        _, _, seq_indices = module.masked_key_indices
        self.assertEqual(seq_indices.numel(), 10)  # int(1*1*20*0.5)
        protected = {0, 1, 17, 18, 19}  # n_sink=2 + window=int(20*0.15)=3
        self.assertEqual(set(seq_indices.tolist()) & protected, set())

    def test_eager_attention_rejected(self):
        sketch = FastKVzipSketch(
            compression_ratio=0.5, layerwise=True, n_sink=0, window_ratio=0.0,
            gates=[_StubGate(torch.rand(1, 1, 6))],
        )
        module = _FakeAttnModule(layer_idx=0, attn_implementation="eager")
        _run_prefill(sketch, module, torch.randn(1, 6, 8))
        with self.assertRaisesRegex(AssertionError, "eager mode not supported"):
            sketch.compress_post()


class TestFastKVzipGateLoading(unittest.TestCase):
    def test_injected_gates_skip_hub_download(self):
        sketch = FastKVzipSketch(compression_ratio=0.5, gates=[_RaisingGate()])
        model = SimpleNamespace(config=SimpleNamespace(name_or_path="Qwen/Qwen3-8B"), device="cpu")
        with patch("huggingface_hub.hf_hub_download", side_effect=AssertionError("no network")) as dl:
            sketch.post_init_from_model(model)
        dl.assert_not_called()
        self.assertEqual(len(sketch.gates), 1)
        self.assertIsInstance(sketch.gates[0], _RaisingGate)

    def test_missing_gates_raise_runtime_error(self):
        sketch = FastKVzipSketch(compression_ratio=0.5)
        model = SimpleNamespace(config=SimpleNamespace(name_or_path="no/such-model"), device="cpu")
        with patch(
            "eval_harness.sketch.sketches.fastkvzip_sketch.get_gate_weight",
            side_effect=OSError("404"),
        ):
            with self.assertRaisesRegex(RuntimeError, "not released"):
                sketch.post_init_from_model(model)

    def test_load_fastkvzip_infers_dims_and_loads_weights(self):
        reference_gates = [_seeded_gate(seed=i, nhead=2, ngroup=3, output_dim=4, sink=2) for i in range(2)]
        weights = [g.state_dict() for g in reference_gates]
        with patch(
            "eval_harness.sketch.sketches.fastkvzip_sketch.get_gate_weight",
            return_value=(weights, "qwen3-8b/q3_dim4_sink2.pt"),
        ):
            gates = load_fastkvzip(model_name="Qwen/Qwen3-8B", device="cpu")

        self.assertEqual(len(gates), 2)
        for idx, (gate, ref) in enumerate(zip(gates, reference_gates)):
            self.assertEqual(gate.index, idx)
            self.assertEqual(gate.nhead, 2)
            self.assertEqual(gate.ngroup, 3)
            self.assertEqual(gate.output_dim, 4)
            self.assertEqual(gate.sink, 2)  # parsed from the sink(\d+) pattern
            self.assertEqual(gate.q_proj.weight.dtype, torch.float32)
            torch.manual_seed(100 + idx)
            hidden = torch.randn(1, 5, 8)
            torch.testing.assert_close(gate(hidden), ref(hidden))

    def test_empty_model_name_raises(self):
        with self.assertRaises(AssertionError):
            load_fastkvzip(model_name="", device="cpu")


class TestFastKVzipCallIntegration(unittest.TestCase):
    def _fake_model(self, n_layers):
        layers = [SimpleNamespace(self_attn=_FakeAttnModule(layer_idx=i)) for i in range(n_layers)]
        # 4-level nesting: outer model -> model.model -> layers -> self_attn
        inner = SimpleNamespace(layers=layers, rotary_emb=nn.Module())
        return SimpleNamespace(model=inner)

    def test_context_manager_scores_then_masks_on_exit(self):
        g0 = _StubGate(torch.full((1, 1, 4), 0.9))
        g1 = _StubGate(torch.full((1, 1, 4), 0.1))
        sketch = FastKVzipSketch(
            compression_ratio=0.5, layerwise=False, n_sink=0, window_ratio=0.0, gates=[g0, g1],
        )
        model = self._fake_model(2)
        hidden = torch.randn(1, 4, 8)

        with sketch(model):
            for layer in model.model.layers:
                self.assertEqual(len(layer.self_attn._forward_hooks), 1)
                self.assertIs(layer.self_attn.rotary_emb, model.model.rotary_emb)
                self.assertIsNone(getattr(layer.self_attn, "masked_key_indices", None))
                layer.self_attn(hidden_states=hidden, cache_position=torch.arange(4))

        for layer in model.model.layers:
            self.assertEqual(len(layer.self_attn._forward_hooks), 0)

        m0, m1 = (layer.self_attn for layer in model.model.layers)
        self.assertEqual(m0.masked_key_indices[2].numel(), 0)
        self.assertEqual(sorted(m1.masked_key_indices[2].tolist()), [0, 1, 2, 3])

    def test_state_resets_between_contexts(self):
        scores = torch.tensor([[[0.9, 0.1, 0.8, 0.2]]])
        sketch = FastKVzipSketch(
            compression_ratio=0.5, layerwise=True, n_sink=0, window_ratio=0.0,
            gates=[_StubGate(scores)],
        )
        model = self._fake_model(1)
        hidden = torch.randn(1, 4, 8)
        for _ in range(2):
            with sketch(model):
                model.model.layers[0].self_attn(hidden_states=hidden, cache_position=torch.arange(4))
            module = model.model.layers[0].self_attn
            self.assertEqual(len(sketch._scored_modules), 1)
            self.assertEqual(sorted(module.masked_key_indices[2].tolist()), [1, 3])

    def test_exit_without_prefill_is_noop(self):
        sketch = FastKVzipSketch(compression_ratio=0.5, gates=[_RaisingGate()])
        model = self._fake_model(1)
        with sketch(model):
            pass
        self.assertIsNone(getattr(model.model.layers[0].self_attn, "masked_key_indices", None))


if __name__ == "__main__":
    unittest.main()
