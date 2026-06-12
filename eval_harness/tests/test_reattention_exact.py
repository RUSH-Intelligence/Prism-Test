"""Tests for the exact ReAttention baseline (``reattention_exact.py``).

The exact method replaces ``self_attn.forward`` (same construction as DCA and
the reference monkeypatch), keeps the FULL raw-KV cache, and runs the
pre-attention recall per prefill chunk / decode step.  These tests prove:

* construction parity — forward replacement installs/restores like DCA;
* reduce-to-dense oracles — ``full_attn`` (and short contexts) equal standard
  attention exactly, for any chunk size (validates the chunk loop, raw cache,
  per-chunk RoPE, and the absolute-position causal mask);
* recall semantics — selection equals a verbatim transcription of
  ``cache_utils_v0921.py`` (unconditional 128-alignment quirk included),
  attention output equals the reference's pe-after-cache eager computation;
* reference quirks — odd-span ``-span//2`` floor expansion, ``mid_size=0``
  streaming view, decode steps never dispatch the Triton kernel;
* decode-time recall — ``recall_option`` semantics (whole / prefill_only /
  generate_only / full_attn);
* pipeline integration — multi-layer end-to-end, bitwise ``full_attn``
  equivalence to the no-method baseline, multi-question restore.

GPU-free: tiny synthetic tensors, ``object.__new__`` pipeline stubs, fake
attention modules (the project convention).
"""

from __future__ import annotations

import math
import unittest

import torch
from torch import nn

from eval_harness.prefill_methods.base import (
    PrefillMethod,
    apply_rotary_pos_emb,
    build_cos_sin,
)
from eval_harness.prefill_methods.reattention_exact import ReAttentionExactMethod
from eval_harness.prefill_methods.registry import (
    ensure_methods_loaded,
    get_prefill_method,
)

D_HEAD = 8


def _inv_freq(D=D_HEAD, base=10000.0):
    half = D // 2
    return 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))


class _FakeAttn(nn.Module):
    """Attention module stub with q/k/v/o projections (DCA-test pattern)."""

    def __init__(self, hidden_dim, num_heads, head_dim, num_kv_heads, layer_idx=0, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.num_heads = num_heads
        self.num_key_value_heads = num_kv_heads
        self.head_dim = head_dim
        self.scaling = 1.0 / math.sqrt(head_dim)
        self.layer_idx = layer_idx
        self.config = None
        self.q_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_dim, bias=False)


class _FakeCache:
    """DynamicCache-like: update concatenates per layer on the seq dim."""

    def __init__(self):
        self.k = {}
        self.v = {}

    def update(self, key, value, layer_idx, cache_kwargs=None):
        if layer_idx in self.k:
            self.k[layer_idx] = torch.cat([self.k[layer_idx], key], dim=2)
            self.v[layer_idx] = torch.cat([self.v[layer_idx], value], dim=2)
        else:
            self.k[layer_idx] = key
            self.v[layer_idx] = value
        return self.k[layer_idx], self.v[layer_idx]

    def get_seq_length(self, layer_idx=0):
        return self.k[layer_idx].shape[2] if layer_idx in self.k else 0


def _make_method(**kwargs):
    # chunk_schedule='uniform' so unit tests control chunk boundaries
    # directly; the reference wrapper schedule has dedicated tests.
    defaults = dict(
        global_size=4, local_size=160, mid_size=4, span_size=8,
        prefill_chunk_size=64, use_triton_kernel="off",
        use_flash_attn="off", chunk_schedule="uniform",
    )
    defaults.update(kwargs)
    m = ReAttentionExactMethod(**defaults)
    m._inv_freq = _inv_freq()
    return m


def _repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    B, H, S, D = x.shape
    return x[:, :, None].expand(B, H, n_rep, S, D).reshape(B, H * n_rep, S, D)


def _standard_attention_forward(attn, hidden, positions, inv_freq):
    """Vanilla causal attention with absolute-position RoPE (the function the
    exact method must reduce to when no recall happens)."""
    B, S, _ = hidden.shape
    nH, nKV, D = attn.num_heads, attn.num_key_value_heads, attn.head_dim
    q = attn.q_proj(hidden).view(B, S, nH, D).transpose(1, 2)
    k = attn.k_proj(hidden).view(B, S, nKV, D).transpose(1, 2)
    v = attn.v_proj(hidden).view(B, S, nKV, D).transpose(1, 2)
    cos, sin = build_cos_sin(positions, inv_freq, hidden.device, torch.float32)
    q = apply_rotary_pos_emb(q, cos, sin)
    k = apply_rotary_pos_emb(k, cos, sin)
    k = _repeat_kv(k, nH // nKV)
    v = _repeat_kv(v, nH // nKV)
    attn_w = torch.matmul(q.float(), k.float().transpose(2, 3)) * attn.scaling
    mask = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1)
    attn_w = attn_w.masked_fill(mask, torch.finfo(torch.float32).min)
    attn_w = torch.softmax(attn_w, dim=-1, dtype=torch.float32)
    out = torch.matmul(attn_w, v.float()).to(q.dtype)
    out = out.transpose(1, 2).reshape(B, S, nH * D)
    return attn.o_proj(out)


def _run_forward(method, attn, hidden, cache=None, positions=None):
    """Drive the replaced forward exactly as the model would."""
    fwd = method._make_exact_forward(attn, attn.layer_idx)
    S = hidden.shape[1]
    if positions is None:
        base = cache.get_seq_length(attn.layer_idx) if cache is not None else 0
        positions = torch.arange(base, base + S)
    out, w = fwd(
        hidden_states=hidden,
        past_key_values=cache if cache is not None else _FakeCache(),
        cache_position=positions,
    )
    return out, w


# ======================================================================
# Registry + construction parity
# ======================================================================


class TestExactConstruction(unittest.TestCase):
    def test_registry_resolves_name_and_aliases(self):
        ensure_methods_loaded()
        for name in ("reattention_exact", "re_attention_exact", "reatt_exact"):
            self.assertIsInstance(get_prefill_method(name), ReAttentionExactMethod)

    def test_context_manager_installs_and_restores_forward(self):
        """Same construction as DCA / the reference monkeypatch: swap
        ``self_attn.forward`` on full-attention layers, restore on exit."""

        class FakeRotary(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("inv_freq", _inv_freq())

        class FakeLayer(nn.Module):
            def __init__(self, idx):
                super().__init__()
                self.self_attn = _FakeAttn(32, 4, D_HEAD, 2, layer_idx=idx)

        class FakeInner(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([FakeLayer(0), FakeLayer(1)])
                self.rotary_emb = FakeRotary()

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = FakeInner()

        model = FakeModel()
        method = ReAttentionExactMethod(use_triton_kernel="off", use_flash_attn="off")
        originals = [layer.self_attn.forward for layer in model.model.layers]
        with method(model):
            for layer in model.model.layers:
                self.assertNotIn(layer.self_attn.forward, originals)
        for layer, orig in zip(model.model.layers, originals):
            self.assertEqual(layer.self_attn.forward, orig)

    def test_reposition_rejected(self):
        with self.assertRaises(ValueError):
            ReAttentionExactMethod(reposition=True)

    def test_invalid_recall_option_rejected(self):
        with self.assertRaises(ValueError):
            ReAttentionExactMethod(recall_option="sometimes")

    def test_invalid_chunk_size_rejected(self):
        with self.assertRaises(ValueError):
            ReAttentionExactMethod(prefill_chunk_size=0)


# ======================================================================
# Reduce-to-dense oracles
# ======================================================================


class TestExactReduceToDense(unittest.TestCase):
    def _setup(self, S, seed=3):
        torch.manual_seed(seed)
        attn = _FakeAttn(32, 4, D_HEAD, 2, seed=seed)
        hidden = torch.randn(1, S, 32)
        return attn, hidden

    def test_short_sequence_equals_standard_attention(self):
        """S <= global + local => no recall => vanilla causal attention."""
        attn, hidden = self._setup(S=40)
        method = _make_method(global_size=4, local_size=160, prefill_chunk_size=16)
        out, _ = _run_forward(method, attn, hidden)
        ref = _standard_attention_forward(attn, hidden, torch.arange(40), _inv_freq())
        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

    def test_full_attn_option_equals_standard_any_length(self):
        """recall_option='full_attn' => dense attention even past g+l."""
        attn, hidden = self._setup(S=220)
        method = _make_method(local_size=16, recall_option="full_attn",
                              prefill_chunk_size=32)
        out, _ = _run_forward(method, attn, hidden)
        ref = _standard_attention_forward(attn, hidden, torch.arange(220), _inv_freq())
        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)

    def test_chunk_size_invariance_on_dense_path(self):
        """The chunk loop (cache update, per-chunk RoPE, absolute-position
        mask) is invisible: any chunking of the dense path gives identical
        output.  Uneven chunking (7) exercises partial final chunks."""
        attn, hidden = self._setup(S=50)
        outs = []
        for cs in (7, 16, 50, 10_000):
            method = _make_method(local_size=160, prefill_chunk_size=cs)
            out, _ = _run_forward(method, attn, hidden)
            outs.append(out)
        for out in outs[1:]:
            torch.testing.assert_close(out, outs[0], atol=1e-5, rtol=1e-5)

    def test_cache_stores_raw_keys_full_length(self):
        """The DynamicCache must hold RAW (pre-RoPE) K and full-length V —
        the reference retains everything; selection is a per-forward view."""
        attn, hidden = self._setup(S=200)
        method = _make_method(local_size=16)  # recall active
        cache = _FakeCache()
        _run_forward(method, attn, hidden, cache=cache)
        self.assertEqual(cache.get_seq_length(0), 200)  # never pruned
        raw_k = attn.k_proj(hidden).view(1, 200, 2, D_HEAD).transpose(1, 2)
        torch.testing.assert_close(cache.k[0], raw_k)  # pre-RoPE


# ======================================================================
# Recall semantics vs a verbatim reference transcription
# ======================================================================


def _reference_recall_positions(
    recall_q, raw_k_cache, raw_v_cache, *, global_size, local_size,
    mid_size, span_size, recall_clip, recall_type="qk",
):
    """Verbatim transcription of ``RECacheV2.update``'s recall branch +
    ``recall_operation_for_full`` (cache_utils_v0921.py), MHA dense path.

    Returns the absolute selected positions (``position_ids_real`` without
    the batch dim).  Includes the unconditional 128-alignment quirk.
    """
    seen = raw_k_cache.shape[2]
    g = global_size
    mod_len = (seen - g - local_size) % 128
    local_eff = local_size - (128 - mod_len)
    if local_eff < 1:  # undefined in the reference; mirror the port's guard
        local_eff = local_size
    n_mid = max(seen - g - local_eff, 0)

    if recall_type == "qk":
        rk = raw_k_cache
    elif recall_type == "qkv":
        rk = raw_k_cache * torch.norm(raw_v_cache, p=1, dim=-1, keepdim=True)
    elif recall_type == "qkv2":
        rk = raw_k_cache * torch.norm(raw_v_cache, p=2, dim=-1, keepdim=True)
    elif recall_type == "k":
        # Reference: recall_q = key_states (the chunk's raw keys).
        recall_q = raw_k_cache[:, :, -recall_q.shape[2]:]
        rk = raw_k_cache
    elif recall_type == "v":
        recall_q = raw_v_cache[:, :, -recall_q.shape[2]:]
        rk = raw_v_cache
    else:
        raise ValueError(recall_type)

    # Reference dense scoring: einsum "bnie,bnje->bnji", topk over dim=-2
    # (middle).  MHA only — the reference's GQA dense reshape is scrambled.
    scores = torch.einsum(
        "bnie,bnje->bnji", recall_q.float(), rk[:, :, g:seen - local_eff].float(),
    )
    _, indices = torch.topk(scores, min(mid_size, n_mid), dim=-2)

    if recall_clip < 0:
        indices = torch.unique(indices)
    else:
        indices, counts = torch.unique(indices, return_counts=True)
        if indices.shape[-1] > recall_clip:
            _, keep = torch.topk(counts, k=recall_clip)
            indices = indices[keep]

    offsets = torch.arange(-span_size // 2, (span_size + 1) // 2)
    ids = (indices[:, None] + offsets[None, :]).reshape(-1)
    ids = ids.clamp(0, seen - 1 - g - local_eff)
    fetch = torch.unique(ids) + g

    return torch.cat([
        torch.arange(g),
        fetch,
        torch.arange(seen - local_eff, seen),
    ])


class TestExactRecallSemantics(unittest.TestCase):
    """MHA (num_heads == num_kv_heads) so the reference dense path is
    unambiguous (its GQA dense reshape scrambles head pairing — the port
    follows the reference *kernel* semantics there, per the audit)."""

    G, L = 4, 160

    def _prefill_then_chunk(self, method, S_ctx=300, q_len=16, seed=11):
        """Prefill S_ctx tokens dense-cache-style, then run one chunk call."""
        torch.manual_seed(seed)
        attn = _FakeAttn(32, 4, D_HEAD, 4, seed=seed)  # MHA
        cache = _FakeCache()
        hidden_ctx = torch.randn(1, S_ctx, 32)
        # Seed the raw cache directly (raw K/V == projections, pre-RoPE).
        k_ctx = attn.k_proj(hidden_ctx).view(1, S_ctx, 4, D_HEAD).transpose(1, 2)
        v_ctx = attn.v_proj(hidden_ctx).view(1, S_ctx, 4, D_HEAD).transpose(1, 2)
        cache.update(k_ctx, v_ctx, 0)

        hidden_q = torch.randn(1, q_len, 32)
        out, _ = _run_forward(method, attn, hidden_q, cache=cache,
                              positions=torch.arange(S_ctx, S_ctx + q_len))
        return attn, cache, hidden_q, out

    def test_selection_matches_reference_transcription(self):
        method = _make_method(
            global_size=self.G, local_size=self.L, mid_size=4, span_size=8,
            recall_clip=8, prefill_chunk_size=10_000,
            debug_record_selection=True,
        )
        attn, cache, hidden_q, _ = self._prefill_then_chunk(method)

        q_len = hidden_q.shape[1]
        raw_q = attn.q_proj(hidden_q).view(1, q_len, 4, D_HEAD).transpose(1, 2)
        expected = _reference_recall_positions(
            raw_q, cache.k[0], cache.v[0],
            global_size=self.G, local_size=self.L, mid_size=4, span_size=8,
            recall_clip=8,
        )
        got = method._last_selection[0]
        self.assertTrue(torch.equal(got, expected),
                        f"selection mismatch:\n got {got}\n exp {expected}")

    def test_attention_output_matches_reference_pe_after_cache(self):
        """Full numerical parity for one chunk: reference transcription of
        selection -> RoPE at original positions (keys) / own positions
        (queries, == the tail of the selected cos/sin since the chunk is the
        view tail) -> bottom-right-causal eager attention."""
        method = _make_method(
            global_size=self.G, local_size=self.L, mid_size=4, span_size=8,
            recall_clip=8, prefill_chunk_size=10_000,
        )
        attn, cache_probe, hidden_q, out = self._prefill_then_chunk(method)

        # Rebuild the same pre-chunk cache for the transcription (the method
        # call above appended the chunk's keys to cache_probe).
        S_total = cache_probe.get_seq_length(0)
        q_len = hidden_q.shape[1]
        raw_k_all, raw_v_all = cache_probe.k[0], cache_probe.v[0]

        raw_q = attn.q_proj(hidden_q).view(1, q_len, 4, D_HEAD).transpose(1, 2)
        key_pos = _reference_recall_positions(
            raw_q, raw_k_all, raw_v_all,
            global_size=self.G, local_size=self.L, mid_size=4, span_size=8,
            recall_clip=8,
        )
        gather = key_pos.view(1, 1, -1, 1).expand(1, 4, -1, D_HEAD)
        sel_k = torch.gather(raw_k_all, 2, gather)
        sel_v = torch.gather(raw_v_all, 2, gather)

        inv = _inv_freq()
        cos, sin = build_cos_sin(key_pos, inv, sel_k.device, torch.float32)
        k_rot = apply_rotary_pos_emb(sel_k, cos, sin)
        # Reference: q gets cos[..., -q_len:, :] — the chunk's own positions.
        q_rot = (raw_q * cos[:, :, -q_len:, :]) + (
            _rotate_half(raw_q) * sin[:, :, -q_len:, :]
        )

        attn_w = torch.matmul(q_rot.float(), k_rot.float().transpose(2, 3)) * attn.scaling
        S_sel = key_pos.numel()
        # Bottom-right causal (flash semantics): query i sees keys
        # j <= S_sel - q_len + i.
        j = torch.arange(S_sel)[None, :]
        i = torch.arange(q_len)[:, None]
        attn_w = attn_w.masked_fill(j > S_sel - q_len + i, torch.finfo(torch.float32).min)
        attn_w = torch.softmax(attn_w, dim=-1, dtype=torch.float32)
        ref_ctx = torch.matmul(attn_w, sel_v.float()).to(raw_q.dtype)
        ref_out = attn.o_proj(ref_ctx.transpose(1, 2).reshape(1, q_len, -1))

        torch.testing.assert_close(out, ref_out, atol=1e-5, rtol=1e-5)

    def test_alignment_quirk_local_shrinks_by_128_when_aligned(self):
        """seen chosen so (seen-g-l) % 128 == 0 => local_eff = l - 128
        (the reference's unconditional-alignment quirk, replicated).

        Pinned by comparing the FULL selection against the verbatim
        transcription (which computes local_eff = 32 at seen = 292) — a
        mutant that skips the shrink when already aligned produces a very
        different selection (a 160-wide local block) and fails here."""
        method = _make_method(global_size=4, local_size=160, mid_size=4,
                              span_size=8, recall_clip=4,
                              prefill_chunk_size=10_000,
                              debug_record_selection=True)
        # S_ctx + q_len = seen: pick seen = 4 + 160 + 128 = 292.
        attn, cache, hidden_q, _ = self._prefill_then_chunk(
            method, S_ctx=276, q_len=16,
        )
        self.assertEqual(cache.get_seq_length(0), 292)
        q_len = hidden_q.shape[1]
        raw_q = attn.q_proj(hidden_q).view(1, q_len, 4, D_HEAD).transpose(1, 2)
        expected = _reference_recall_positions(
            raw_q, cache.k[0], cache.v[0],
            global_size=4, local_size=160, mid_size=4, span_size=8,
            recall_clip=4,
        )
        got = method._last_selection[0]
        self.assertTrue(torch.equal(got, expected),
                        f"selection mismatch:\n got {got}\n exp {expected}")
        # Structural pin of the shrink itself: exactly 32 trailing local
        # positions, and the total selection is far below the un-shrunk
        # 4 + mid + 160 layout (mid is bounded by recall_clip * span = 32).
        local_eff = 160 - 128
        self.assertTrue(torch.equal(
            got[-local_eff:], torch.arange(292 - local_eff, 292),
        ))
        self.assertLessEqual(got.numel(), 4 + 4 * 8 + local_eff)

    def test_odd_span_uses_reference_floor_semantics(self):
        """span_size=5 expands to 6 offsets (-3..2) — the reference's
        ``-span//2`` floor, NOT the hook port's ``-(span//2)``.  Forced
        single seed => contiguous block of width span+1."""
        method = _make_method(global_size=4, local_size=160, mid_size=1,
                              span_size=5, recall_clip=1,
                              prefill_chunk_size=10_000,
                              debug_record_selection=True)
        torch.manual_seed(0)
        attn = _FakeAttn(32, 4, D_HEAD, 4, seed=0)
        cache = _FakeCache()
        S_ctx = 300
        k_ctx = torch.randn(1, 4, S_ctx, D_HEAD) * 0.01
        v_ctx = torch.randn(1, 4, S_ctx, D_HEAD)
        # Plant a dominant key far from clamp edges; e1-aligned.
        k_ctx[:, :, 100, :] = 0.0
        k_ctx[:, :, 100, 0] = 50.0
        cache.update(k_ctx, v_ctx, 0)

        hidden_q = torch.zeros(1, 8, 32)  # q_proj(0)=0 → scores 0 except seed col?
        # Zero queries give zero scores everywhere — instead craft hidden so
        # raw_q = e1 for every head: q_proj weight is random, so drive the
        # selection through the planted dominant key: any nonzero query picks
        # column 100 as top-1 because its key dwarfs all others when the
        # query has a positive e1 component; use a positive hidden seed.
        torch.manual_seed(1)
        hidden_q = torch.randn(1, 8, 32)
        out, _ = _run_forward(method, attn, hidden_q, cache=cache,
                              positions=torch.arange(S_ctx, S_ctx + 8))
        sel = method._last_selection[0]
        g, seen = 4, S_ctx + 8
        local_eff = method._aligned_local_size(seen)
        mid = sel[(sel >= g) & (sel < seen - local_eff)]
        # recall_clip=1 → exactly one seed → one contiguous clamped span of
        # width span_size + 1 = 6 (floor semantics).
        self.assertEqual(mid.numel(), 6)
        self.assertTrue(torch.equal(mid, torch.arange(mid[0], mid[0] + 6)))

    def test_mid_size_zero_streaming_view(self):
        method = _make_method(global_size=4, local_size=16, mid_size=0,
                              prefill_chunk_size=10_000,
                              debug_record_selection=True)
        attn, cache, _, _ = self._prefill_then_chunk(method, S_ctx=100, q_len=8)
        sel = method._last_selection[0]
        expected = torch.cat([torch.arange(4), torch.arange(108 - 16, 108)])
        self.assertTrue(torch.equal(sel, expected))

    def test_mid_size_zero_uses_contiguous_pe_positions(self):
        """The reference's mid_size==0 branch returns
        ``position_ids_for_pe=None``, so the rotary embedding rotates the
        [global | local] view at CONTIGUOUS positions ``arange(g + l)`` with
        the queries at the tail (StreamingLLM-style compression) — NOT at the
        original absolute positions.  Numeric pin against a transcription of
        that semantics (regression: the first cut rotated at absolutes)."""
        g, l, q_len = 4, 16, 8
        method = _make_method(global_size=g, local_size=l, mid_size=0,
                              prefill_chunk_size=10_000)
        attn, cache, hidden_q, out = self._prefill_then_chunk(
            method, S_ctx=100, q_len=q_len,
        )
        seen = cache.get_seq_length(0)  # 108
        raw_k, raw_v = cache.k[0], cache.v[0]
        sel_k = torch.cat([raw_k[:, :, :g], raw_k[:, :, -l:]], dim=2)
        sel_v = torch.cat([raw_v[:, :, :g], raw_v[:, :, -l:]], dim=2)

        inv = _inv_freq()
        pe_pos = torch.arange(g + l)
        cos, sin = build_cos_sin(pe_pos, inv, sel_k.device, torch.float32)
        k_rot = apply_rotary_pos_emb(sel_k, cos, sin)
        raw_q = attn.q_proj(hidden_q).view(1, q_len, 4, D_HEAD).transpose(1, 2)
        q_rot = (raw_q * cos[:, :, -q_len:, :]) + (
            _rotate_half(raw_q) * sin[:, :, -q_len:, :]
        )

        S_sel = g + l
        attn_w = torch.matmul(q_rot.float(), k_rot.float().transpose(2, 3)) * attn.scaling
        j = torch.arange(S_sel)[None, :]
        i = torch.arange(q_len)[:, None]
        attn_w = attn_w.masked_fill(j > S_sel - q_len + i, float("-inf"))
        attn_w = torch.softmax(attn_w, dim=-1, dtype=torch.float32)
        ref_ctx = torch.matmul(attn_w, sel_v.float()).to(raw_q.dtype)
        ref_out = attn.o_proj(ref_ctx.transpose(1, 2).reshape(1, q_len, -1))

        torch.testing.assert_close(out, ref_out, atol=1e-5, rtol=1e-5)

    def test_pe_original_false_matches_reference_transcription(self):
        """pe_original=False (the reference's PUBLISHED eval setting): the
        selected view is re-rotated at CONTIGUOUS positions arange(view_len)
        with queries at the tail (position_ids_for_pe=None semantics).
        Full numeric parity for one recall chunk."""
        method = _make_method(
            global_size=self.G, local_size=self.L, mid_size=4, span_size=8,
            recall_clip=8, prefill_chunk_size=10_000, pe_original=False,
        )
        attn, cache_probe, hidden_q, out = self._prefill_then_chunk(method)

        q_len = hidden_q.shape[1]
        raw_k_all, raw_v_all = cache_probe.k[0], cache_probe.v[0]
        raw_q = attn.q_proj(hidden_q).view(1, q_len, 4, D_HEAD).transpose(1, 2)
        key_pos = _reference_recall_positions(
            raw_q, raw_k_all, raw_v_all,
            global_size=self.G, local_size=self.L, mid_size=4, span_size=8,
            recall_clip=8,
        )
        gather = key_pos.view(1, 1, -1, 1).expand(1, 4, -1, D_HEAD)
        sel_k = torch.gather(raw_k_all, 2, gather)
        sel_v = torch.gather(raw_v_all, 2, gather)

        S_sel = key_pos.numel()
        inv = _inv_freq()
        # Reference: rotary(view, None) -> cos/sin over arange(view_len).
        cos, sin = build_cos_sin(torch.arange(S_sel), inv, sel_k.device,
                                 torch.float32)
        k_rot = apply_rotary_pos_emb(sel_k, cos, sin)
        q_rot = (raw_q * cos[:, :, -q_len:, :]) + (
            _rotate_half(raw_q) * sin[:, :, -q_len:, :]
        )

        attn_w = torch.matmul(q_rot.float(), k_rot.float().transpose(2, 3)) * attn.scaling
        j = torch.arange(S_sel)[None, :]
        i = torch.arange(q_len)[:, None]
        attn_w = attn_w.masked_fill(j > S_sel - q_len + i, float("-inf"))
        attn_w = torch.softmax(attn_w, dim=-1, dtype=torch.float32)
        ref_ctx = torch.matmul(attn_w, sel_v.float()).to(raw_q.dtype)
        ref_out = attn.o_proj(ref_ctx.transpose(1, 2).reshape(1, q_len, -1))

        torch.testing.assert_close(out, ref_out, atol=1e-5, rtol=1e-5)

    def test_pe_original_modes_discriminate(self):
        """pe_original=False output equals the CONTIGUOUS-PE transcription
        and differs from the ABSOLUTE-PE one (and vice versa for True) — the
        two position schemes are genuinely different computations, and each
        flag selects exactly its reference semantics."""
        outs, refs = {}, {}
        for pe in (True, False):
            method = _make_method(
                global_size=self.G, local_size=self.L, mid_size=4,
                span_size=8, recall_clip=8, prefill_chunk_size=10_000,
                pe_original=pe,
            )
            attn, cache_probe, hidden_q, out = self._prefill_then_chunk(method)
            outs[pe] = out

            q_len = hidden_q.shape[1]
            raw_q = attn.q_proj(hidden_q).view(1, q_len, 4, D_HEAD).transpose(1, 2)
            key_pos = _reference_recall_positions(
                raw_q, cache_probe.k[0], cache_probe.v[0],
                global_size=self.G, local_size=self.L, mid_size=4,
                span_size=8, recall_clip=8,
            )
            gather = key_pos.view(1, 1, -1, 1).expand(1, 4, -1, D_HEAD)
            sel_k = torch.gather(cache_probe.k[0], 2, gather)
            sel_v = torch.gather(cache_probe.v[0], 2, gather)
            S_sel = key_pos.numel()
            pe_pos = key_pos if pe else torch.arange(S_sel)
            cos, sin = build_cos_sin(pe_pos, _inv_freq(), sel_k.device,
                                     torch.float32)
            k_rot = apply_rotary_pos_emb(sel_k, cos, sin)
            q_rot = (raw_q * cos[:, :, -q_len:, :]) + (
                _rotate_half(raw_q) * sin[:, :, -q_len:, :]
            )
            attn_w = torch.matmul(
                q_rot.float(), k_rot.float().transpose(2, 3),
            ) * attn.scaling
            j = torch.arange(S_sel)[None, :]
            i = torch.arange(q_len)[:, None]
            attn_w = attn_w.masked_fill(j > S_sel - q_len + i, float("-inf"))
            attn_w = torch.softmax(attn_w, dim=-1, dtype=torch.float32)
            ref_ctx = torch.matmul(attn_w, sel_v.float()).to(raw_q.dtype)
            refs[pe] = attn.o_proj(ref_ctx.transpose(1, 2).reshape(1, q_len, -1))

        torch.testing.assert_close(outs[True], refs[True], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(outs[False], refs[False], atol=1e-5, rtol=1e-5)
        # The schemes genuinely differ (so each parity check is load-bearing).
        self.assertFalse(torch.allclose(outs[True], outs[False]))

    def test_pe_original_false_decode_step_matches_transcription(self):
        method = _make_method(
            global_size=self.G, local_size=self.L, mid_size=4, span_size=8,
            recall_clip=8, pe_original=False,
        )
        torch.manual_seed(23)
        attn = _FakeAttn(32, 4, D_HEAD, 4, seed=23)  # MHA
        cache = _FakeCache()
        k_ctx = torch.randn(1, 4, 300, D_HEAD)
        v_ctx = torch.randn(1, 4, 300, D_HEAD)
        cache.update(k_ctx, v_ctx, 0)
        hidden = torch.randn(1, 1, 32)
        out, _ = _run_forward(method, attn, hidden, cache=cache,
                              positions=torch.tensor([300]))

        raw_q = attn.q_proj(hidden).view(1, 1, 4, D_HEAD).transpose(1, 2)
        key_pos = _reference_recall_positions(
            raw_q, cache.k[0], cache.v[0],
            global_size=self.G, local_size=self.L, mid_size=4, span_size=8,
            recall_clip=8,
        )
        gather = key_pos.view(1, 1, -1, 1).expand(1, 4, -1, D_HEAD)
        sel_k = torch.gather(cache.k[0], 2, gather)
        sel_v = torch.gather(cache.v[0], 2, gather)
        S_sel = key_pos.numel()
        cos, sin = build_cos_sin(torch.arange(S_sel), _inv_freq(),
                                 sel_k.device, torch.float32)
        k_rot = apply_rotary_pos_emb(sel_k, cos, sin)
        q_rot = (raw_q * cos[:, :, -1:, :]) + (
            _rotate_half(raw_q) * sin[:, :, -1:, :]
        )
        attn_w = torch.matmul(q_rot.float(), k_rot.float().transpose(2, 3)) * attn.scaling
        attn_w = torch.softmax(attn_w, dim=-1, dtype=torch.float32)
        ref_ctx = torch.matmul(attn_w, sel_v.float()).to(raw_q.dtype)
        ref_out = attn.o_proj(ref_ctx.transpose(1, 2).reshape(1, 1, -1))
        torch.testing.assert_close(out, ref_out, atol=1e-5, rtol=1e-5)

    def test_contiguous_pe_rejects_chunk_longer_than_view(self):
        """A chunk longer than the re-positioned view has no faithful
        semantics (the reference crashes on a broadcast mismatch); the port
        must fail loudly instead of silently rotating queries at negative
        positions (audit finding)."""
        method = _make_method(global_size=4, local_size=160, mid_size=4,
                              span_size=8, recall_clip=2, pe_original=False,
                              prefill_chunk_size=512)
        torch.manual_seed(41)
        attn = _FakeAttn(32, 4, D_HEAD, 2, seed=41)
        hidden = torch.randn(1, 1024, 32)
        with self.assertRaises(ValueError):
            _run_forward(method, attn, hidden, cache=_FakeCache())
        # Same guard for the mid_size==0 branch (g + l < chunk).
        method0 = _make_method(global_size=4, local_size=16, mid_size=0,
                               prefill_chunk_size=512)
        with self.assertRaises(ValueError):
            _run_forward(method0, attn, torch.randn(1, 600, 32),
                         cache=_FakeCache())

    def test_pe_original_flag_is_noop_without_recall(self):
        """The no-recall path returns the full cache with contiguous ==
        absolute positions, so pe_original must not change anything there."""
        torch.manual_seed(31)
        attn = _FakeAttn(32, 4, D_HEAD, 2, seed=31)
        hidden = torch.randn(1, 40, 32)
        outs = []
        for pe in (True, False):
            method = _make_method(global_size=4, local_size=160,
                                  pe_original=pe, prefill_chunk_size=16)
            out, _ = _run_forward(method, attn, hidden, cache=_FakeCache())
            outs.append(out)
        torch.testing.assert_close(outs[0], outs[1])

    def test_recall_type_variants_match_transcription(self):
        """qkv2 (V-norm weighted) and k (key-as-query) recall variants pin
        their selections against the verbatim transcription."""
        for rt in ("qkv2", "k"):
            method = _make_method(global_size=self.G, local_size=self.L,
                                  mid_size=4, span_size=8, recall_clip=8,
                                  recall_type=rt, prefill_chunk_size=10_000,
                                  debug_record_selection=True)
            attn, cache, hidden_q, _ = self._prefill_then_chunk(
                method, seed=13,
            )
            q_len = hidden_q.shape[1]
            raw_q = attn.q_proj(hidden_q).view(1, q_len, 4, D_HEAD).transpose(1, 2)
            expected = _reference_recall_positions(
                raw_q, cache.k[0], cache.v[0],
                global_size=self.G, local_size=self.L, mid_size=4,
                span_size=8, recall_clip=8, recall_type=rt,
            )
            got = method._last_selection[0]
            self.assertTrue(torch.equal(got, expected), f"{rt}: mismatch")


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


# ======================================================================
# Decode-time recall (recall_option semantics)
# ======================================================================


class TestExactDecodeRecall(unittest.TestCase):
    def _decode_step(self, method, seed=5, S_ctx=300):
        torch.manual_seed(seed)
        attn = _FakeAttn(32, 4, D_HEAD, 2, seed=seed)
        cache = _FakeCache()
        k_ctx = torch.randn(1, 2, S_ctx, D_HEAD)
        v_ctx = torch.randn(1, 2, S_ctx, D_HEAD)
        cache.update(k_ctx, v_ctx, 0)
        hidden = torch.randn(1, 1, 32)  # single decode token
        out, _ = _run_forward(method, attn, hidden, cache=cache,
                              positions=torch.tensor([S_ctx]))
        return out, method._last_selection.get(0)

    def test_whole_recalls_on_decode_prefill_only_does_not(self):
        kw = dict(global_size=4, local_size=160, mid_size=4, span_size=8,
                  recall_clip=8, debug_record_selection=True)
        out_w, sel_w = self._decode_step(_make_method(recall_option="whole", **kw))
        out_p, sel_p = self._decode_step(_make_method(recall_option="prefill_only", **kw))
        # 'whole': bounded view; 'prefill_only': the full cache.
        self.assertLess(sel_w.numel(), 301)
        self.assertEqual(sel_p.numel(), 301)
        self.assertTrue(torch.equal(sel_p, torch.arange(301)))
        # Different key sets => different outputs (same weights/inputs).
        self.assertFalse(torch.allclose(out_w, out_p))

    def test_generate_only_skips_prefill_recall(self):
        kw = dict(global_size=4, local_size=16, mid_size=4, span_size=8,
                  recall_clip=8, debug_record_selection=True,
                  recall_option="generate_only", prefill_chunk_size=10_000)
        method = _make_method(**kw)
        torch.manual_seed(7)
        attn = _FakeAttn(32, 4, D_HEAD, 2, seed=7)
        cache = _FakeCache()
        hidden = torch.randn(1, 120, 32)  # multi-token => is_generate=False
        _run_forward(method, attn, hidden, cache=cache)
        self.assertEqual(method._last_selection[0].numel(), 120)  # full view
        # Single-token step now recalls.
        hidden1 = torch.randn(1, 1, 32)
        _run_forward(method, attn, hidden1, cache=cache,
                     positions=torch.tensor([120]))
        self.assertLess(method._last_selection[0].numel(), 121)

    def test_decode_never_dispatches_kernel(self):
        """Reference gate: qlen == 1 always takes the dense path, even when
        the parent gate would accept the kernel."""
        method = _make_method(mid_size=4)
        raw_q = torch.randn(1, 4, 1, 128)
        k_mid = torch.randn(1, 2, 256, 128)
        # Parent gate may be monkeypatched to True; the override must win.
        orig = ReAttentionExactMethod.__mro__[1]._should_use_kernel
        try:
            ReAttentionExactMethod.__mro__[1]._should_use_kernel = (
                lambda self, q, k, n: True
            )
            self.assertFalse(method._should_use_kernel(raw_q, k_mid, 256))
        finally:
            ReAttentionExactMethod.__mro__[1]._should_use_kernel = orig

    def test_force_kernel_unmet_constraints_raise_on_prefill_chunks(self):
        """use_triton_kernel='force' keeps the hook port's loud-failure
        contract for multi-token chunks (CPU here → constraints unmet)."""
        method = _make_method(use_triton_kernel="force", local_size=16,
                              prefill_chunk_size=10_000)
        torch.manual_seed(9)
        attn = _FakeAttn(32, 4, D_HEAD, 2, seed=9)
        cache = _FakeCache()
        hidden = torch.randn(1, 120, 32)
        with self.assertRaises(RuntimeError):
            _run_forward(method, attn, hidden, cache=cache)


# ======================================================================
# Chunk schedule (reference wrapper parity) + guards
# ======================================================================


def _wrapper_bounds(S, g, l, C):
    """Verbatim simulation of re_attention_wrapper.py:276-286 + the
    generate() step consuming the last prompt token."""
    mod_len = (S - 1 - g + l) % 128
    start, end = 0, g + l - (128 - mod_len)
    bounds = []
    while start < S - 1:
        bounds.append((start, min(end, S - 1)))
        start, end = end, end + C
    bounds.append((S - 1, S))
    return bounds


class TestExactChunkSchedule(unittest.TestCase):
    def test_reference_schedule_matches_wrapper_simulation(self):
        """For every config where the wrapper schedule is well-defined
        (first chunk end in [1, S-1]), _chunk_bounds must reproduce its
        boundaries exactly, with only the final 1-token chunk
        generate-classified."""
        cases = [
            (400, 8, 160, 64),   # the parity audit's example
            (300, 4, 160, 64),
            (1000, 32, 256, 128),
            (500, 4, 300, 100),
            (50, 4, 160, 64),    # first chunk clamped at S-1
        ]
        for S, g, l, C in cases:
            method = _make_method(global_size=g, local_size=l,
                                  prefill_chunk_size=C,
                                  chunk_schedule="reference")
            got = method._chunk_bounds(S)
            expected = _wrapper_bounds(S, g, l, C)
            self.assertEqual([(s, e) for s, e, _ in got], expected,
                             f"bounds mismatch for {(S, g, l, C)}")
            self.assertEqual([gen for _, _, gen in got],
                             [False] * (len(got) - 1) + [True])

    def test_reference_first_chunk_never_recalls(self):
        """The wrapper engineers the first chunk to end before g + l, so
        recall cannot fire on it."""
        for S, g, l, C in [(400, 8, 160, 64), (1000, 32, 256, 128)]:
            method = _make_method(global_size=g, local_size=l,
                                  prefill_chunk_size=C,
                                  chunk_schedule="reference")
            first_end = method._chunk_bounds(S)[0][1]
            self.assertLessEqual(first_end, g + l)

    def test_decode_step_is_single_generate_chunk(self):
        method = _make_method(chunk_schedule="reference")
        self.assertEqual(method._chunk_bounds(1), [(0, 1, True)])

    def test_one_token_mid_chunk_is_generate_classified(self):
        """The reference classifies is_generate purely by qlen==1
        (cache_utils_v0921.py:562), so a 1-token MID chunk — reachable only
        with an ODD prefill_chunk_size (the wrapper's alignment keeps
        remainders even) — must be generate-classified: under
        recall_option='prefill_only' it attends the full cache, like the
        reference, instead of recalling (audit finding)."""
        method = _make_method(global_size=8, local_size=160,
                              prefill_chunk_size=51,
                              recall_option="prefill_only",
                              chunk_schedule="reference")
        bounds = method._chunk_bounds(3561)
        one_token_mids = [
            (s, e, gen) for s, e, gen in bounds[1:-1] if e - s == 1
        ]
        self.assertTrue(one_token_mids, "config no longer produces the corner")
        for s, e, gen in one_token_mids:
            self.assertTrue(gen, f"1-token chunk ({s},{e}) not generate-classified")
        # And under prefill_only that chunk must NOT recall.
        self.assertFalse(method._check_recall(seen=3560, is_generate=True))
        # Bounds themselves still match the wrapper schedule.
        self.assertEqual([(s, e) for s, e, _ in bounds],
                         _wrapper_bounds(3561, 8, 160, 51))

    def test_generate_only_and_full_attn_use_uniform_chunks(self):
        """The wrapper skips its chunk loop for these options; chunked full
        attention is numerically identical, so the uniform schedule applies
        (no generate-classified tail)."""
        for opt in ("generate_only", "full_attn"):
            method = _make_method(recall_option=opt, prefill_chunk_size=64,
                                  chunk_schedule="reference")
            got = method._chunk_bounds(200)
            self.assertEqual([(s, e) for s, e, _ in got],
                             [(0, 64), (64, 128), (128, 192), (192, 200)])
            self.assertTrue(all(not gen for _, _, gen in got))

    def test_last_token_generate_semantics_under_prefill_only(self):
        """Under recall_option='prefill_only' with the reference schedule,
        the final context token is a generate-classified 1-token chunk, so
        it must NOT recall (it attends the full cache) — exactly what the
        wrapper's generate() step does; the port's old uniform scheduling
        recalled on it (audit finding)."""
        method = _make_method(global_size=4, local_size=16, mid_size=4,
                              span_size=8, recall_clip=8,
                              recall_option="prefill_only",
                              prefill_chunk_size=64,
                              chunk_schedule="reference",
                              debug_record_selection=True)
        torch.manual_seed(3)
        attn = _FakeAttn(32, 4, D_HEAD, 2, seed=3)
        cache = _FakeCache()
        hidden = torch.randn(1, 193, 32)
        _run_forward(method, attn, hidden, cache=cache)
        # The LAST recorded selection is the final 1-token generate chunk:
        # prefill_only => no recall => the full cache.
        self.assertTrue(torch.equal(method._last_selection[0],
                                    torch.arange(193)))

    def test_chunk_loop_equals_sequential_forwards(self):
        """One forward with internal chunking == several forwards with the
        same boundaries (recall ACTIVE): pins the in-forward chunk loop
        against the reference's model-level chunking."""
        kw = dict(global_size=4, local_size=16, mid_size=4, span_size=8,
                  recall_clip=8, chunk_schedule="uniform")
        torch.manual_seed(21)
        attn = _FakeAttn(32, 4, D_HEAD, 2, seed=21)
        hidden = torch.randn(1, 200, 32)

        m1 = _make_method(prefill_chunk_size=100, **kw)
        out1, _ = _run_forward(m1, attn, hidden, cache=_FakeCache())

        m2 = _make_method(prefill_chunk_size=10_000, **kw)
        cache2 = _FakeCache()
        out2a, _ = _run_forward(m2, attn, hidden[:, :100], cache=cache2)
        out2b, _ = _run_forward(m2, attn, hidden[:, 100:], cache=cache2,
                                positions=torch.arange(100, 200))
        out2 = torch.cat([out2a, out2b], dim=1)
        torch.testing.assert_close(out1, out2, atol=1e-5, rtol=1e-5)

    def test_aligned_local_size_formula(self):
        """Verbatim reference formula incl. the shrink-by-128 quirk, plus
        the tiny-config guard."""
        method = _make_method(global_size=4, local_size=160)
        # mod_len = (seen - 164) % 128; local_eff = 160 - (128 - mod_len)
        self.assertEqual(method._aligned_local_size(292), 32)    # mod 0 quirk
        self.assertEqual(method._aligned_local_size(300), 40)    # mod 8
        self.assertEqual(method._aligned_local_size(419), 159)   # mod 127
        guard = _make_method(global_size=4, local_size=16)
        # aligned would be 16 - (128 - mod) <= 0 -> falls back to 16.
        self.assertEqual(guard._aligned_local_size(300), 16)

    def test_empty_view_rows_are_zero_guarded(self):
        """global_size=0 can leave early queries with NO visible key in the
        recall view; such rows must output ZERO (guarded), never a uniform
        average over (future) keys — audit finding."""
        method = _make_method(global_size=0, local_size=129, mid_size=1,
                              span_size=8, recall_clip=1,
                              prefill_chunk_size=10_000)
        torch.manual_seed(2)
        attn = _FakeAttn(32, 4, D_HEAD, 2, seed=2)
        cache = _FakeCache()
        # Plant a dominant key late in the middle so early queries select
        # nothing at or before their own position.
        k_ctx = torch.randn(1, 2, 300, D_HEAD) * 0.01
        k_ctx[:, :, 250, 0] = 50.0
        v_ctx = torch.randn(1, 2, 300, D_HEAD)
        cache.update(k_ctx, v_ctx, 0)
        torch.manual_seed(4)
        hidden = torch.randn(1, 1, 32)
        # A later forward triggers recall over the seeded cache; but we want
        # in-chunk early rows -> drive a fresh 300-token prefill instead.
        cache2 = _FakeCache()
        method2 = _make_method(global_size=0, local_size=129, mid_size=1,
                               span_size=8, recall_clip=1,
                               prefill_chunk_size=10_000)
        # Build hidden whose k-projection plants the dominant key: easier to
        # drive the unit seam directly.
        q_c = torch.randn(1, 4, 300, D_HEAD)
        out = method2._chunk_attention(
            attn, 0, q_c, torch.arange(300), k_ctx, v_ctx,
            scale=attn.scaling, is_generate=False,
        )
        self.assertTrue(torch.isfinite(out).all())
        local_start = 300 - method2._aligned_local_size(300)
        # Rows before the planted span and the local window see nothing.
        span_lo = 250 - 4  # span_size=8 -> offsets -4..3
        zero_rows = out[:, :, :span_lo]
        self.assertTrue(torch.equal(zero_rows, torch.zeros_like(zero_rows)))
        # Rows inside the local window genuinely attend (non-zero).
        self.assertGreater(out[:, :, local_start:].abs().sum().item(), 0.0)

    def test_flash_force_on_cpu_raises(self):
        method = _make_method(use_flash_attn="force", prefill_chunk_size=16)
        torch.manual_seed(6)
        attn = _FakeAttn(32, 4, D_HEAD, 2, seed=6)
        hidden = torch.randn(1, 40, 32)
        with self.assertRaises(RuntimeError):
            _run_forward(method, attn, hidden, cache=_FakeCache())

    def test_context_manager_not_reentrant(self):
        class FakeLayer(nn.Module):
            def __init__(self, idx):
                super().__init__()
                self.self_attn = _FakeAttn(32, 4, D_HEAD, 2, layer_idx=idx)

        class FakeInner(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([FakeLayer(0)])
                self.register_buffer("inv", _inv_freq())

        class FakeRotary(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("inv_freq", _inv_freq())

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = FakeInner()
                self.model.rotary_emb = FakeRotary()

        model = FakeModel()
        method = _make_method()
        with method(model):
            with self.assertRaises(RuntimeError):
                with method(model):
                    pass
        # Restored despite the nested failure.
        self.assertEqual(method._saved_forwards, {})


# ======================================================================
# Pipeline integration (tiny real model)
# ======================================================================


class _StubTokenizer:
    """Id-preserving stub: equality of decoded strings == equality of the
    generated token ids (a stub that erases ids would make the e2e
    equivalence oracle blind to token identity — audit finding)."""

    model_max_length = 8192

    def decode(self, ids, skip_special_tokens=True):  # noqa: ARG002
        return ",".join(str(int(i)) for i in ids)


def _build_model(num_hidden_layers):
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        hidden_size=64, intermediate_size=128,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=256,
        max_position_embeddings=8192, rope_theta=10000.0,
        attn_implementation="eager",
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg).eval()
    if model.generation_config.eos_token_id is None:
        model.generation_config.eos_token_id = 2
    return model


def _pipe_forward(model, method, ctx=300, questions=1, qlen=8, mnt=4):
    from eval_harness.sketch.cache_adapter import create_cache_adapter
    from eval_harness.sketch.pipeline import SketchTextGenerationPipeline

    pipe = object.__new__(SketchTextGenerationPipeline)
    pipe.model = model
    pipe.tokenizer = _StubTokenizer()
    ca = create_cache_adapter(model)
    cache = ca.initialize_cache(None)
    torch.manual_seed(0)
    inputs = {
        "context_ids": torch.randint(0, 256, (1, ctx)),
        "questions_ids": [torch.randint(0, 256, (1, qlen)) for _ in range(questions)],
    }
    answers = pipe._forward(inputs, max_new_tokens=mnt, sketch=None,
                            prefill_method=method, cache=cache, cache_adapter=ca)
    lens = [int(layer.keys.shape[2]) for layer in cache.layers]
    return answers, lens


class TestExactPipelineIntegration(unittest.TestCase):
    def test_full_attn_equals_no_method_baseline_end_to_end(self):
        """The strongest plumbing oracle: full_attn through the chunked
        forward replacement must reproduce the no-method baseline answers
        and leave an identical full-length cache."""
        ans_base, lens_base = _pipe_forward(_build_model(2), PrefillMethod())
        method = ReAttentionExactMethod(
            global_size=4, local_size=160, recall_option="full_attn",
            prefill_chunk_size=64, use_triton_kernel="off", use_flash_attn="off",
        )
        ans_full, lens_full = _pipe_forward(_build_model(2), method)
        self.assertEqual(ans_full, ans_base)
        self.assertEqual(lens_full, lens_base)

    def test_recall_runs_multilayer_with_full_retention(self):
        for n_layers in (2, 4):
            method = ReAttentionExactMethod(
                global_size=4, local_size=160, mid_size=4, span_size=8,
                recall_clip=8, prefill_chunk_size=64,
                use_triton_kernel="off", use_flash_attn="off",
            )
            answers, lens = _pipe_forward(_build_model(n_layers), method)
            self.assertEqual(len(answers), 1)
            self.assertEqual(len(answers[0].split(",")), 4)  # full decode budget
            # Raw cache retains the whole context (restore trims question).
            self.assertEqual(lens, [300] * n_layers)

    def test_multi_question_restore(self):
        method = ReAttentionExactMethod(
            global_size=4, local_size=160, mid_size=4, span_size=8,
            recall_clip=8, prefill_chunk_size=64,
            use_triton_kernel="off", use_flash_attn="off",
        )
        answers, lens = _pipe_forward(_build_model(2), method, questions=2)
        self.assertEqual(len(answers), 2)
        self.assertEqual(lens, [300, 300])  # restored after each question

    def test_question_pass_spanning_multiple_chunks(self):
        """prefill_chunk_size smaller than the question splits the question
        pass into several recall chunks — must run cleanly."""
        method = ReAttentionExactMethod(
            global_size=4, local_size=160, mid_size=4, span_size=8,
            recall_clip=8, prefill_chunk_size=3,
            use_triton_kernel="off", use_flash_attn="off",
        )
        answers, _ = _pipe_forward(_build_model(2), method, qlen=8)
        self.assertEqual(len(answers[0].split(",")), 4)


class TestExactOversizeViewWarning(unittest.TestCase):
    """pe_original=False with an unbounded view (recall_clip=-1) silently
    produced positionally-OOD prefill on beyond-native contexts (the
    2026-06-11 0%-RULER incident); the method must warn loudly, once."""

    _LOGGER = "eval_harness.prefill_methods.reattention_exact"

    def _drive(self, max_trained_pos):
        method = _make_method(pe_original=False, recall_clip=-1)
        method._max_trained_pos = max_trained_pos
        method._oversize_view_warned = False
        attn = _FakeAttn(32, 2, D_HEAD, 1)
        torch.manual_seed(0)
        # 300 tokens >> global+local (164): recall fires on later chunks and
        # with clip=-1 the view grows past 160.
        hidden = torch.randn(1, 300, 32)
        _run_forward(method, attn, hidden, cache=_FakeCache())

    def test_warns_exactly_once_when_view_exceeds_trained_window(self):
        with self.assertLogs(self._LOGGER, level="WARNING") as cm:
            self._drive(max_trained_pos=160)
        hits = [m for m in cm.output if "exceeds the model's trained window" in m]
        self.assertEqual(len(hits), 1)

    def test_silent_when_view_bounded_by_trained_window(self):
        with self.assertNoLogs(self._LOGGER, level="WARNING"):
            self._drive(max_trained_pos=10_000)

    def test_silent_when_trained_window_unknown(self):
        with self.assertNoLogs(self._LOGGER, level="WARNING"):
            self._drive(max_trained_pos=None)


if __name__ == "__main__":
    unittest.main()
