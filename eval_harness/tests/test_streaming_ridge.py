"""Tests for StreamingRidgeSketch (decode-time streaming ridge compression).

Pins the contract: prefill event bitwise-identical to RidgeSketch; decode
events every `decode_interval` tokens; append-only full-history key Gram
(streaming tau == batch tau, exactly); EMA query Gram; budget computed
against tokens-ever-seen (no geometric over-eviction); sink/local windows and
cross-layer rectangularity preserved; state re-initialized per prefill.

No model loading — fake attention modules + real transformers DynamicCache,
mirroring test_sketch_ridge / test_compression_schedule patterns.
"""

from __future__ import annotations

import unittest

import torch
from torch import nn
from transformers import DynamicCache

from eval_harness.kv_compression.registry import get_kv_compressor_class
from eval_harness.kv_compression.compressors.ridge_sketch import RidgeSketch
from eval_harness.kv_compression.compressors.streaming_ridge_sketch import StreamingRidgeSketch


class _FakeAttnModule(nn.Module):
    def __init__(self, hidden_dim=32, num_heads=4, head_dim=8, num_kv_heads=2,
                 layer_idx=0, identity_q=False, seed=0):
        super().__init__()
        self.num_heads = num_heads
        self.num_key_value_heads = num_kv_heads
        self.head_dim = head_dim
        self.layer_idx = layer_idx
        torch.manual_seed(seed)
        self.q_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        if identity_q:
            assert hidden_dim == num_heads * head_dim
            with torch.no_grad():
                self.q_proj.weight.copy_(torch.eye(hidden_dim))


def _norm_gram(keys: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    k = keys.float()
    k = k / k.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
    return k.transpose(-2, -1) @ k


def _mk_sketch(**kw) -> StreamingRidgeSketch:
    defaults = dict(compression_ratio=0.5, sink_size=4, local_size=8,
                    min_tokens_to_compress=0, decode_interval=4)
    defaults.update(kw)
    return StreamingRidgeSketch(**defaults)


def _prefill(sketch, module, T=100, hidden_dim=32, H_kv=2, D=8, seed=0):
    torch.manual_seed(seed)
    keys = torch.randn(1, H_kv, T, D)
    values = torch.randn(1, H_kv, T, D)
    hidden = torch.randn(1, T, hidden_dim)
    sketch.set_phase("prefill")
    out_k, out_v = sketch.compress(module, hidden, keys, values, None, {})
    sketch.set_phase("decode")
    return keys, values, hidden, out_k, out_v


def _decode_step(sketch, module, cur_k, cur_v, hidden_dim=32, n_new=1, seed=0):
    """Append n_new tokens to the current cache tensors and run compress."""
    torch.manual_seed(seed)
    H_kv, D = cur_k.shape[1], cur_k.shape[3]
    k_new = torch.randn(1, H_kv, n_new, D)
    v_new = torch.randn(1, H_kv, n_new, D)
    h_new = torch.randn(1, n_new, hidden_dim)
    keys = torch.cat([cur_k, k_new], dim=2)
    values = torch.cat([cur_v, v_new], dim=2)
    out_k, out_v = sketch.compress(module, h_new, keys, values, None, {})
    return out_k, out_v, k_new


class TestRegistryAndConstruction(unittest.TestCase):
    def test_registered_and_kwargs_constructible(self):
        cls = get_kv_compressor_class("streaming_ridge")
        self.assertIs(cls, StreamingRidgeSketch)
        sk = cls(compression_ratio=0.4, decode_interval=32, decode_ratio=0.6,
                 query_ema_beta=0.9, envelope_gamma=2.0)
        self.assertEqual(sk.decode_interval, 32)
        self.assertEqual(sk.decode_ratio, 0.6)
        self.assertTrue(sk.fires_on_prefill)
        self.assertTrue(sk.fires_on_decode)

    def test_invalid_params_assert(self):
        with self.assertRaises(AssertionError):
            _mk_sketch(decode_interval=0)
        with self.assertRaises(AssertionError):
            _mk_sketch(query_ema_beta=0.0)
        with self.assertRaises(AssertionError):
            _mk_sketch(decode_ratio=1.0)

    def test_unsafe_schedule_overrides_rejected(self):
        # decode-only: prefill event never initializes/resets the streaming
        # state -> cross-prompt leak; must be rejected at construction.
        with self.assertRaises(AssertionError):
            _mk_sketch(schedule=["decode"])
        # streaming (chunked prefill) resets the full-history Gram per chunk.
        with self.assertRaises(AssertionError):
            _mk_sketch(schedule=["streaming", "post_prefill", "decode"])
        # post_prefill-only (no decode) is a valid degenerate config.
        _mk_sketch(schedule=["post_prefill"])

    def test_decode_capable_flag(self):
        self.assertTrue(StreamingRidgeSketch.decode_capable)
        self.assertFalse(getattr(RidgeSketch, "decode_capable", False))


class TestPrefillEquivalence(unittest.TestCase):
    def test_prefill_event_bitwise_equals_ridge(self):
        module = _FakeAttnModule(seed=1)
        torch.manual_seed(7)
        keys = torch.randn(1, 2, 100, 8)
        values = torch.randn(1, 2, 100, 8)
        hidden = torch.randn(1, 100, 32)

        streaming = _mk_sketch()
        streaming.set_phase("prefill")
        s_k, s_v = streaming.compress(module, hidden, keys, values, None, {})

        ridge = RidgeSketch(compression_ratio=0.5, sink_size=4, local_size=8,
                            min_tokens_to_compress=0)
        r_k, r_v = ridge.compress(module, hidden, keys, values, None, {})

        self.assertTrue(torch.equal(s_k, r_k))
        self.assertTrue(torch.equal(s_v, r_v))

    def test_prefill_initializes_full_history_gram(self):
        module = _FakeAttnModule(seed=1)
        sketch = _mk_sketch()
        keys, _, hidden, out_k, _ = _prefill(sketch, module, seed=2)
        # Gram covers ALL 100 context keys, not just the ~50 retained ones.
        expected = _norm_gram(keys)
        torch.testing.assert_close(sketch._gram_k[0], expected, atol=1e-5, rtol=1e-5)
        self.assertEqual(sketch._tokens_seen[0], 100)
        self.assertEqual(sketch._since_event[0], 0)
        self.assertLess(out_k.shape[2], 100)  # prefill actually compressed


class TestStreamingExactness(unittest.TestCase):
    def test_key_gram_folds_all_keys_at_events_only(self):
        module = _FakeAttnModule(seed=1)
        sketch = _mk_sketch(decode_interval=3)
        keys, _, _, cur_k, cur_v = _prefill(sketch, module, seed=3)
        all_new = []
        # 6 steps at interval 3 -> events at steps 3 and 6; buffers drained.
        for step in range(6):
            cur_k, cur_v, k_new = _decode_step(sketch, module, cur_k, cur_v, seed=10 + step)
            all_new.append(k_new)
        full_history = torch.cat([keys] + all_new, dim=2)
        # Gram == prefill keys + every decode key, INCLUDING evicted ones.
        torch.testing.assert_close(
            sketch._gram_k[0], _norm_gram(full_history), atol=1e-4, rtol=1e-4,
        )
        self.assertEqual(sketch._pending_tokens[0], 0)

        # Step 7: buffered only — Gram untouched until the next event.
        gram_before = sketch._gram_k[0].clone()
        cur_k, cur_v, _ = _decode_step(sketch, module, cur_k, cur_v, seed=17)
        self.assertTrue(torch.equal(sketch._gram_k[0], gram_before))
        self.assertEqual(sketch._pending_tokens[0], 1)

    def test_event_tau_uses_full_history_gram(self):
        module = _FakeAttnModule(seed=1)
        sketch = _mk_sketch(decode_interval=5)
        keys, _, _, cur_k, cur_v = _prefill(sketch, module, seed=4)
        news = []
        for step in range(5):  # exactly one event at step 5
            cur_k, cur_v, k_new = _decode_step(sketch, module, cur_k, cur_v, seed=20 + step)
            news.append(k_new)
        history = torch.cat([keys] + news, dim=2)
        probe = cur_k[:, :, 4:-8, :]  # current mid region

        tau_stream = sketch._tau_from_gram(probe, sketch._gram_k[0])
        # reference: leverage against the directly-computed full-history Gram
        tau_ref = sketch._tau_from_gram(probe, _norm_gram(history))
        torch.testing.assert_close(tau_stream, tau_ref, atol=1e-5, rtol=1e-5)

    def test_cholesky_tau_matches_explicit_inverse(self):
        sketch = _mk_sketch()
        torch.manual_seed(21)
        hist = torch.randn(1, 2, 60, 8)
        probe = torch.randn(1, 2, 10, 8)
        gram = _norm_gram(hist)
        tau = sketch._tau_from_gram(probe, gram)

        k = probe.float()
        k = k / k.norm(p=2, dim=-1, keepdim=True).clamp_min(sketch.eps)
        eye = torch.eye(8).view(1, 1, 8, 8)
        inv = torch.linalg.inv(gram + sketch.ridge_lambda * eye)
        tau_ref = ((k @ inv) * k).sum(-1).clamp_min(0.0)
        torch.testing.assert_close(tau, tau_ref, atol=1e-5, rtol=1e-5)

    def test_tau_pinv_fallback_on_nonpd_gram(self):
        sketch = _mk_sketch()
        torch.manual_seed(22)
        probe = torch.randn(1, 1, 5, 8)
        bad_gram = -0.5 * torch.eye(8).view(1, 1, 8, 8)  # not PD -> cholesky fails
        tau = sketch._tau_from_gram(probe, bad_gram)
        self.assertTrue(torch.isfinite(tau).all())
        self.assertTrue((tau >= 0).all())


class TestDecodeCadenceAndBudget(unittest.TestCase):
    def test_interval_cadence_and_no_geometric_decay(self):
        module = _FakeAttnModule(seed=1)
        sketch = _mk_sketch(decode_interval=4, compression_ratio=0.5)
        _, _, _, cur_k, cur_v = _prefill(sketch, module, seed=5)
        self.assertEqual(cur_k.shape[2], 50)  # int(100 * 0.5)

        lengths = []
        for step in range(16):
            cur_k, cur_v, _ = _decode_step(sketch, module, cur_k, cur_v, seed=30 + step)
            lengths.append(cur_k.shape[2])

        # Steps 1-3 grow; step 4 prunes to int(104*0.5)=52; and so on.
        self.assertEqual(lengths[:4], [51, 52, 53, 52])
        self.assertEqual(lengths[4:8], [53, 54, 55, 54])   # int(108*.5)=54
        self.assertEqual(lengths[8:12], [55, 56, 57, 56])  # int(112*.5)=56
        self.assertEqual(lengths[12:16], [57, 58, 59, 58])
        # Net growth (1-r) per token — never collapses geometrically.

    def test_decode_ratio_overrides_compression_ratio(self):
        module = _FakeAttnModule(seed=1)
        sketch = _mk_sketch(decode_interval=4, compression_ratio=0.5, decode_ratio=0.8)
        _, _, _, cur_k, cur_v = _prefill(sketch, module, seed=6)
        for step in range(4):
            cur_k, cur_v, _ = _decode_step(sketch, module, cur_k, cur_v, seed=40 + step)
        # T_seen=104, decode budget int(104*0.2)=20 < sink+local=12+... -> 20-12=8 mid
        self.assertEqual(cur_k.shape[2], 20)

    def test_multi_token_decode_chunk_counts_all_tokens(self):
        # The question forward arrives as ONE multi-token decode chunk.
        module = _FakeAttnModule(seed=1)
        sketch = _mk_sketch(decode_interval=4)
        _, _, _, cur_k, cur_v = _prefill(sketch, module, seed=7)
        cur_k, cur_v, _ = _decode_step(sketch, module, cur_k, cur_v, n_new=6, seed=50)
        # 6 >= interval -> event fired immediately: int(106*0.5)=53,
        # and the interval REMAINDER survives: 6 % 4 = 2.
        self.assertEqual(cur_k.shape[2], 53)
        self.assertEqual(sketch._since_event[0], 2)
        self.assertEqual(sketch._tokens_seen[0], 106)

    def test_interval_remainder_keeps_event_phase(self):
        module = _FakeAttnModule(seed=1)
        sketch = _mk_sketch(decode_interval=4)
        _, _, _, cur_k, cur_v = _prefill(sketch, module, seed=17)
        self.assertEqual(cur_k.shape[2], 50)

        # chunk of 3: no event (since=3), cache grows
        cur_k, cur_v, _ = _decode_step(sketch, module, cur_k, cur_v, n_new=3, seed=51)
        self.assertEqual(cur_k.shape[2], 53)
        self.assertEqual(sketch._since_event[0], 3)
        # chunk of 3: since=6 >= 4 -> event; remainder 2; keep int(106*0.5)=53
        cur_k, cur_v, _ = _decode_step(sketch, module, cur_k, cur_v, n_new=3, seed=52)
        self.assertEqual(cur_k.shape[2], 53)
        self.assertEqual(sketch._since_event[0], 2)
        # single token: since=3, no event
        cur_k, cur_v, _ = _decode_step(sketch, module, cur_k, cur_v, seed=53)
        self.assertEqual(cur_k.shape[2], 54)
        self.assertEqual(sketch._since_event[0], 3)
        # single token: since=4 -> event exactly 4 tokens after the previous
        # boundary (phase preserved); keep int(108*0.5)=54
        cur_k, cur_v, _ = _decode_step(sketch, module, cur_k, cur_v, seed=54)
        self.assertEqual(cur_k.shape[2], 54)
        self.assertEqual(sketch._since_event[0], 0)

    def test_sink_and_local_always_survive_decode_events(self):
        module = _FakeAttnModule(seed=1)
        sketch = _mk_sketch(decode_interval=4)
        keys, values, _, cur_k, cur_v = _prefill(sketch, module, seed=8)
        pre_k = cur_k.clone()
        for step in range(4):
            prev_k = cur_k.clone()
            cur_k, cur_v, _ = _decode_step(sketch, module, cur_k, cur_v, seed=60 + step)
        # sink rows unchanged since prefill output
        self.assertTrue(torch.equal(cur_k[:, :, :4, :], pre_k[:, :, :4, :]))
        # local window = last 8 of the pre-event cache (event fired on step 4)
        self.assertTrue(torch.equal(cur_k[:, :, -8:, :],
                                    torch.cat([prev_k, cur_k[:, :, -1:, :]], dim=2)[:, :, -8:, :]))


class TestQueryGramEMA(unittest.TestCase):
    def _identity_module(self, layer_idx=0):
        # hidden_dim == H_q * D and H_q == H_kv: queries == reshaped hidden.
        return _FakeAttnModule(hidden_dim=16, num_heads=2, head_dim=8,
                               num_kv_heads=2, layer_idx=layer_idx, identity_q=True)

    def test_ema_folding_matches_hand_computation(self):
        module = self._identity_module()
        sketch = _mk_sketch(decode_interval=2, query_ema_beta=0.5)
        T, hd = 40, 16
        torch.manual_seed(9)
        keys = torch.randn(1, 2, T, 8)
        values = torch.randn(1, 2, T, 8)
        hidden = torch.randn(1, T, hd)
        sketch.set_phase("prefill")
        cur_k, cur_v = sketch.compress(module, hidden, keys, values, None, {})
        sketch.set_phase("decode")

        g0 = sketch._gram_q[0].clone()
        w0 = sketch._q_weight[0]
        self.assertEqual(w0, float(T))

        # two decode steps -> one event at interval=2
        hs = []
        for step in range(2):
            torch.manual_seed(70 + step)
            k_new = torch.randn(1, 2, 1, 8)
            v_new = torch.randn(1, 2, 1, 8)
            h_new = torch.randn(1, 1, hd)
            hs.append(h_new)
            keys2 = torch.cat([cur_k, k_new], dim=2)
            values2 = torch.cat([cur_v, v_new], dim=2)
            cur_k, cur_v = sketch.compress(module, h_new, keys2, values2, None, {})

        # identity q_proj + no pooling: q == hidden reshaped [1,2,1,8]
        pend = torch.zeros(1, 2, 8, 8)
        for h in hs:
            q = h.view(1, 1, 2, 8).transpose(1, 2).float()
            pend = pend + q.transpose(-2, -1) @ q
        expected = 0.5 * g0 + pend
        torch.testing.assert_close(sketch._gram_q[0], expected, atol=1e-5, rtol=1e-5)
        self.assertAlmostEqual(sketch._q_weight[0], 0.5 * T + 2)


class TestDecodeRotationSafety(unittest.TestCase):
    """rotate_queries=True at decode must never fabricate positions."""

    def _rot_module(self):
        m = _FakeAttnModule(hidden_dim=16, num_heads=2, head_dim=8,
                            num_kv_heads=2, identity_q=True)
        return m

    def _cos_sin(self, n=1, seed=0):
        torch.manual_seed(seed)
        return torch.rand(1, n, 8), torch.rand(1, n, 8)

    def test_position_embeddings_kwarg_is_used(self):
        module = self._rot_module()
        sketch = _mk_sketch(decode_interval=1, query_ema_beta=1.0)
        sketch.rotate_queries = True
        T, hd = 40, 16
        torch.manual_seed(23)
        keys = torch.randn(1, 2, T, 8)
        values = torch.randn(1, 2, T, 8)
        hidden = torch.randn(1, T, hd)
        sketch.set_phase("prefill")
        cur_k, cur_v = sketch.compress(module, hidden, keys, values, None, {})
        sketch.set_phase("decode")
        gq_before = sketch._gram_q[0].clone()

        cos, sin = self._cos_sin(seed=24)
        torch.manual_seed(25)
        h_new = torch.randn(1, 1, hd)
        k_new = torch.randn(1, 2, 1, 8)
        keys2 = torch.cat([cur_k, k_new], dim=2)
        values2 = torch.cat([cur_v, k_new], dim=2)
        sketch.compress(module, h_new, keys2, values2, None,
                        {"position_embeddings": (cos, sin)})

        # identity q_proj, no pooling: q == reshaped hidden, then rotated
        q = h_new.view(1, 1, 2, 8).transpose(1, 2)
        q_rot = sketch._apply_rope_to_queries(q, cos, sin).float()
        expected = gq_before + q_rot.transpose(-2, -1) @ q_rot
        torch.testing.assert_close(sketch._gram_q[0], expected, atol=1e-5, rtol=1e-5)

    def test_cache_position_rejected_after_prune(self):
        module = self._rot_module()
        sketch = _mk_sketch(decode_interval=1)
        sketch.rotate_queries = True
        sketch._pruned_in_decode[0] = True
        module.rotary_emb = lambda x, pos: self._cos_sin(seed=26)
        h = torch.randn(1, 1, 16)
        out = sketch._decode_cos_sin(module, h, {"cache_position": torch.tensor([37])}, 0)
        self.assertIsNone(out)

    def test_cache_position_accepted_before_prune(self):
        module = self._rot_module()
        sketch = _mk_sketch(decode_interval=1)
        sketch.rotate_queries = True
        sketch._pruned_in_decode[0] = False
        seen = {}

        def rotary(x, pos):
            seen["pos"] = pos
            return self._cos_sin(seed=27)

        module.rotary_emb = rotary
        h = torch.randn(1, 1, 16)
        out = sketch._decode_cos_sin(module, h, {"cache_position": torch.tensor([37])}, 0)
        self.assertIsNotNone(out)
        self.assertEqual(seen["pos"].tolist(), [[37]])

    def test_unresolved_positions_accumulate_unrotated_with_single_warning(self):
        module = self._rot_module()  # no rotary_emb attribute
        sketch = _mk_sketch(decode_interval=2)
        sketch.rotate_queries = True
        T, hd = 40, 16
        torch.manual_seed(28)
        keys = torch.randn(1, 2, T, 8)
        values = torch.randn(1, 2, T, 8)
        hidden = torch.randn(1, T, hd)
        sketch.set_phase("prefill")
        cur_k, cur_v = sketch.compress(module, hidden, keys, values, None, {})
        sketch.set_phase("decode")
        gq_before = sketch._gram_q[0].clone()

        hs = []
        with self.assertLogs(
            "eval_harness.kv_compression.compressors.streaming_ridge_sketch",
            level="WARNING",
        ) as cm:
            for step in range(2):  # one event; no position source at all
                torch.manual_seed(60 + step)
                h_new = torch.randn(1, 1, hd)
                k_new = torch.randn(1, 2, 1, 8)
                hs.append(h_new)
                keys2 = torch.cat([cur_k, k_new], dim=2)
                values2 = torch.cat([cur_v, k_new], dim=2)
                cur_k, cur_v = sketch.compress(module, h_new, keys2, values2, None, {})
        self.assertEqual(len([m for m in cm.output if "un-rotated" in m]), 1)

        # accumulated Gram equals the UN-rotated hand computation — positions
        # were never fabricated from zero.
        pend = torch.zeros(1, 2, 8, 8)
        for h in hs:
            q = h.view(1, 1, 2, 8).transpose(1, 2).float()
            pend = pend + q.transpose(-2, -1) @ q
        torch.testing.assert_close(sketch._gram_q[0], gq_before + pend,
                                   atol=1e-5, rtol=1e-5)


class TestLayersAndLifecycle(unittest.TestCase):
    def test_rectangular_across_layers(self):
        sketch = _mk_sketch(decode_interval=4)
        mods = [_FakeAttnModule(seed=1, layer_idx=0), _FakeAttnModule(seed=2, layer_idx=1)]
        states = []
        for m in mods:
            _, _, _, k, v = _prefill(sketch, m, seed=11)
            states.append([k, v])
        for step in range(8):
            for si, m in enumerate(mods):
                k, v, _ = _decode_step(sketch, m, states[si][0], states[si][1],
                                       seed=80 + step)
                states[si] = [k, v]
            self.assertEqual(states[0][0].shape[2], states[1][0].shape[2])

    def test_second_prefill_reinitializes_state(self):
        module = _FakeAttnModule(seed=1)
        sketch = _mk_sketch(decode_interval=4)
        _, _, _, cur_k, cur_v = _prefill(sketch, module, seed=12)
        for step in range(3):
            cur_k, cur_v, _ = _decode_step(sketch, module, cur_k, cur_v, seed=90 + step)
        self.assertEqual(sketch._since_event[0], 3)
        self.assertEqual(sketch._tokens_seen[0], 103)

        keys2, _, _, _, _ = _prefill(sketch, module, T=60, seed=13)
        self.assertEqual(sketch._tokens_seen[0], 60)
        self.assertEqual(sketch._since_event[0], 0)
        torch.testing.assert_close(sketch._gram_k[0], _norm_gram(keys2),
                                   atol=1e-5, rtol=1e-5)

    def test_decode_before_prefill_is_graceful(self):
        module = _FakeAttnModule(seed=1)
        sketch = _mk_sketch(decode_interval=2)
        torch.manual_seed(14)
        keys = torch.randn(1, 2, 50, 8)
        values = torch.randn(1, 2, 50, 8)
        hidden = torch.randn(1, 1, 32)
        sketch.set_phase("decode")
        out_k, out_v = sketch.compress(module, hidden, keys, values, None, {})
        self.assertTrue(torch.equal(out_k, keys))
        self.assertEqual(sketch._tokens_seen[0], 50)


class TestForwardHookIntegration(unittest.TestCase):
    def test_hook_prunes_cache_at_interval_via_dynamic_cache(self):
        T, hd, H_kv, D = 100, 32, 2, 8
        module = _FakeAttnModule(hidden_dim=hd, seed=1)
        sketch = _mk_sketch(decode_interval=4)
        cache = DynamicCache()
        torch.manual_seed(15)
        keys = torch.randn(1, H_kv, T, D)
        values = torch.randn(1, H_kv, T, D)
        hidden = torch.randn(1, T, hd)
        cache.update(keys.clone(), values.clone(), 0)

        sketch.set_phase("prefill")
        out = (torch.randn(1, T, hd), None)
        kwargs = {"hidden_states": hidden, "past_key_values": cache,
                  "cache_position": torch.arange(T)}
        result = sketch.forward_hook(module, [], kwargs, out)
        self.assertIs(result, out)
        self.assertEqual(cache.layers[0].keys.shape[2], 50)

        sketch.set_phase("decode")
        lengths = []
        pos = T
        for step in range(8):
            torch.manual_seed(100 + step)
            cache.update(torch.randn(1, H_kv, 1, D), torch.randn(1, H_kv, 1, D), 0)
            kwargs = {
                "hidden_states": torch.randn(1, 1, hd),
                "past_key_values": cache,
                "cache_position": torch.tensor([pos]),
            }
            sketch.forward_hook(module, [], kwargs, (torch.randn(1, 1, hd), None))
            lengths.append(cache.layers[0].keys.shape[2])
            pos += 1
        self.assertEqual(lengths[:4], [51, 52, 53, 52])
        self.assertEqual(lengths[4:8], [53, 54, 55, 54])


if __name__ == "__main__":
    unittest.main()
