"""Non-tautological faithfulness check for the ReAttention prefill port.

The existing ``_reattention_reference`` in ``test_prefill_methods.py`` is a
*copy of the port's own logic* (same einsum axis, same conditional alignment,
same ``-(span//2)`` offsets), so a match between them proves nothing about
fidelity to the upstream algorithm.

This module instead reproduces the **upstream reference** —
``RECacheV2.update``'s prefill path in
``re_attention/cache_utils_v0921.py`` from the ReAttention reference
(https://github.com/OpenMOSS/ReAttention) (lines
503-655) — *exactly where it differs from the port*, then characterizes both
the regions of agreement and the known, verified deviations.

The oracle (``reference_reattention_update``) mirrors the reference, NOT the
port.  Specifically:

* **Unconditional 128-alignment** of the local window
  (``cache_utils:580-582``)::

      mod_len   = (S - global - local) % 128
      local_eff = local - (128 - mod_len)            # ALWAYS, never gated

  The port's :meth:`ReAttentionMethod._effective_local_size` instead computes
  ``pad = (-raw_mid) % 128; local - pad`` and only applies it when
  ``align_local_to_128=True``.  ``(-raw_mid) % 128 == (128 - mod_len) % 128``,
  so the two agree *iff* ``mod_len != 0`` (i.e. ``raw_mid % 128 != 0``) AND the
  port has alignment enabled.  All parity configs below pick
  ``raw_mid % 128 != 0`` and set ``align_local_to_128=True``.

* **recall_k weighting** per ``recall_type`` (``cache_utils:595-603``):
  ``qk`` → raw K, ``qkv`` → K·‖V‖₁, ``qkv2`` → K·‖V‖₂.

* **Dense scores** ``einsum("bnie,bnje->bnji", recall_q, recall_k_mid)`` with
  ``topk`` over ``dim=-2`` (the MIDDLE axis ``j``), ``k = min(mid_size,
  n_mid)`` (``cache_utils:625-626``).  The port scores
  ``einsum("bhqd,bhmd->bhqm")`` + ``topk`` over ``dim=-1`` (also the middle
  axis, just transposed) — so the *selected middle index set* is identical
  even though the topk axis label differs.

* **recall_clip** (``cache_utils:628-634``): ``torch.unique`` of the picks;
  if more than ``recall_clip`` survive, keep the top-``recall_clip`` by
  frequency.

* **Span offsets** ``torch.arange(-span_size // 2, (span_size + 1) // 2)``
  (``cache_utils:532``).  This uses the *float-floor* of ``-span``: for an
  ODD span this yields ``-16..15`` for ``span=31`` (Python ``-31 // 2 ==
  -16``), one token further LEFT than the port's ``-(span // 2) == -15``.  For
  an EVEN span the two coincide.  ``fetch = (idx[:, None] +
  off).view(-1).clamp(0, S - 1 - global - local_eff)``, ``unique``,
  ``+ global``, ``sort`` (``cache_utils:533-536``).

* **Assembly** ``cat([arange(global), fetch, arange(S - local_eff, S)])``
  (``cache_utils:643-645``).

To make the comparison meaningful we drive the *real port hook* and recover
the absolute indices it actually kept (by matching the returned, RoPE-free
``values`` rows — which are unique per position — back to their source
positions).  Inputs are constructed so the port's internal un-rotation
recovers EXACTLY the raw K handed to the oracle: raw K is rotated with
``_rope_pos_emb`` to fill the cache and the matching ``position_embeddings``
are passed through ``kwargs``.  MHA (``num_heads == num_kv_heads``) is used so
the GQA grouping reshape is a no-op and cannot mask a real difference.
"""

from __future__ import annotations

import unittest

import torch

from eval_harness.prefill_methods.base import apply_rotary_pos_emb
from eval_harness.prefill_methods.reattention import ReAttentionMethod
from eval_harness.tests.test_prefill_methods import _FakeAttnModule, _rope_pos_emb


# ======================================================================
# The faithful oracle — mirrors cache_utils_v0921.py:503-655, NOT the port
# ======================================================================


def reference_reattention_update(
    raw_q,
    raw_k,
    values,
    *,
    global_size,
    local_size,
    mid_size,
    span_size,
    recall_type,
    recall_clip,
):
    """Faithful reproduction of ``RECacheV2.update``'s prefill path.

    Returns the sorted 1-D tensor of *absolute* kept indices, exactly as the
    upstream cache would gather them.  Every line is annotated with the
    upstream source line it reproduces.
    """
    B, H_kv, S, D = raw_k.shape

    # --- UNCONDITIONAL 128-alignment of the local window (580-582) ---------
    mod_len = (S - global_size - local_size) % 128
    local = local_size - (128 - mod_len)
    mid_region = max(S - global_size - local, 0)  # mid_size in the reference (582)

    # --- recall_k weighting per recall_type (595-603) ----------------------
    if recall_type in ("qk", "qk_pe"):
        rk = raw_k
    elif recall_type in ("qkv", "qkv_pe"):
        rk = raw_k * torch.norm(values, p=1, dim=-1, keepdim=True)
    elif recall_type in ("qkv2", "qkv2_pe"):
        rk = raw_k * torch.norm(values, p=2, dim=-1, keepdim=True)
    else:
        raise ValueError(f"unsupported recall_type {recall_type!r}")

    recall_q = raw_q
    H_q = raw_q.shape[1]
    n_rep = max(1, H_q // H_kv)

    # recall_k[..., global:-local, :]  (slice the middle region) (619/625)
    rk_mid = rk[..., global_size : S - local, :]  # [B, H_kv, mid_region, D]

    # GQA grouping reshape (623-624): recall_q -> (n_rep*B, H_kv, qlen, D).
    if n_rep > 1:
        qlen = raw_q.shape[2]
        recall_q = recall_q.reshape(B, H_kv, n_rep, qlen, D)
        recall_q = recall_q.reshape(n_rep * B, H_kv, qlen, D)

    # einsum("bnie,bnje->bnji", recall_q, recall_k_mid) -> [b, n, j(mid), i(q)] (625)
    scores = torch.einsum("bnie,bnje->bnji", recall_q.float(), rk_mid.float())
    # topk over dim=-2 (the MIDDLE axis j) with min(mid_size, mid_region) (626)
    _, indices = torch.topk(scores, min(mid_size, mid_region), dim=-2)

    # --- global unique + recall_clip (628-634) -----------------------------
    if recall_clip < 0:
        indices = torch.unique(indices)
    else:
        indices, counts = torch.unique(indices, return_counts=True)
        if indices.shape[-1] > recall_clip:
            _, indices_ids = torch.topk(counts, k=recall_clip)
            indices = indices[indices_ids]

    # --- span expansion: recall_operation_for_full (531-536) ---------------
    # REFERENCE uses float-floor offsets: arange(-span // 2, (span + 1) // 2).
    offsets = torch.arange(-span_size // 2, (span_size + 1) // 2)
    fetch = (indices[:, None] + offsets[None, :]).reshape(-1)
    fetch = fetch.clamp(0, S - 1 - global_size - local)
    fetch = torch.unique(fetch) + global_size
    fetch, _ = torch.sort(fetch, dim=-1)

    # --- assemble [global | fetch | local] absolute indices (643-645) ------
    global_range = torch.arange(0, global_size)
    local_range = torch.arange(S - local, S)
    return torch.cat([global_range, fetch, local_range])


def reference_span_offsets(span_size):
    """The reference span-offset tensor (``cache_utils:532``)."""
    return torch.arange(-span_size // 2, (span_size + 1) // 2)


def port_span_offsets(span_size):
    """The port span-offset tensor (``reattention.py:654-657``)."""
    return torch.arange(-(span_size // 2), (span_size + 1) // 2)


# ======================================================================
# Helpers: drive the real port hook and recover what it actually kept
# ======================================================================


def _port_kept_indices(method, module, hidden, rotated_keys, values, kwargs):
    """Run the *real* port hook and recover the absolute indices it kept.

    Values carry no RoPE, so the returned value rows are byte-identical to
    their source rows in the original ``values`` tensor.  Constructing values
    with distinct rows per position lets us map each kept row back to its
    absolute source index unambiguously.
    """
    new_keys, new_values = method.prefill_forward_hook(
        module, hidden, rotated_keys.clone(), values.clone(), kwargs,
    )
    v_src = values[0, 0]  # [S, D]; unique rows per position
    kept = set()
    for r in range(new_values.shape[2]):
        row = new_values[0, 0, r]
        kept.add(int((v_src - row).abs().sum(dim=-1).argmin()))
    return sorted(kept), new_keys.shape[2]


def _build_inputs(*, S, global_size, local_size, D=16, num_heads=2,
                  H_kv=2, hot_rel=50, seed=11):
    """MHA inputs whose raw K is recovered EXACTLY by the port's un-rotation.

    A constant identity query aligned with feature 0 plus a single "hot"
    middle key concentrates the selection on a sparse middle set (so the kept
    set is a strict subset of the cache, making parity non-trivial).  Raw K is
    rotated with the same ``(cos, sin)`` that is passed into the hook kwargs,
    so the hook un-rotates back to exactly this raw K.
    """
    module = _FakeAttnModule(
        hidden_dim=num_heads * D, num_heads=num_heads, head_dim=D,
        num_kv_heads=H_kv, identity_q=True,
    )
    qvec = torch.zeros(num_heads * D)
    qvec[0] = 1.0
    hidden = qvec.view(1, 1, -1).expand(1, S, num_heads * D).contiguous()

    torch.manual_seed(seed)
    raw_k = torch.randn(1, H_kv, S, D) * 0.02
    raw_k[:, :, global_size + hot_rel, 0] = 5.0  # one dominant middle key
    values = torch.randn(1, H_kv, S, D)  # distinct rows per position

    cos, sin, inv_freq = _rope_pos_emb(torch.arange(S).unsqueeze(0), D)
    rotated = apply_rotary_pos_emb(raw_k, cos.unsqueeze(1), sin.unsqueeze(1))
    kwargs = {"position_embeddings": (cos, sin)}

    raw_q = module.q_proj(hidden).view(1, S, num_heads, D).transpose(1, 2)
    return module, hidden, raw_k, raw_q, values, rotated, kwargs, inv_freq


class TestReAttentionFaithfulness(unittest.TestCase):
    # Shared geometry: raw_mid % 128 != 0 so the port's optional alignment
    # (when enabled) matches the reference's unconditional alignment.
    GLOBAL = 8
    LOCAL = 128
    RAW_MID = 300  # 300 % 128 == 44  (!= 0)

    @property
    def S(self):
        return self.GLOBAL + self.LOCAL + self.RAW_MID

    # -- (A) THE faithfulness test: port == faithful oracle ------------------

    def test_port_matches_reference_when_aligned_and_even_span(self):
        """With unconditional-equivalent alignment and an EVEN span, the real
        port hook's kept-index set equals the faithful upstream oracle's,
        for both ``qk`` and ``qkv2`` scoring.

        This is genuinely non-tautological: the oracle topk's over the MIDDLE
        axis (``dim=-2``) and uses the reference's ``-span // 2`` offsets,
        while the port topk's over ``dim=-1`` with ``-(span // 2)`` offsets.
        They agree only because (a) the selected middle *set* is invariant to
        the topk axis label, (b) an even span makes both offset conventions
        identical, and (c) ``raw_mid % 128 != 0`` makes the two alignment
        formulas coincide.  The kept set is a strict subset of the cache, so
        the equality is not the trivial "keep everything" case.
        """
        span_size = 32  # EVEN
        for recall_type in ("qk", "qkv2"):
            with self.subTest(recall_type=recall_type):
                (module, hidden, raw_k, raw_q, values, rotated, kwargs,
                 _) = _build_inputs(
                    S=self.S, global_size=self.GLOBAL, local_size=self.LOCAL,
                )
                method = ReAttentionMethod(
                    global_size=self.GLOBAL, local_size=self.LOCAL,
                    mid_size=1, span_size=span_size, recall_type=recall_type,
                    recall_clip=-1, align_local_to_128=True,
                    use_triton_kernel="off",
                )
                port_set, port_R = _port_kept_indices(
                    method, module, hidden, rotated, values, kwargs,
                )
                oracle = reference_reattention_update(
                    raw_q, raw_k, values,
                    global_size=self.GLOBAL, local_size=self.LOCAL,
                    mid_size=1, span_size=span_size,
                    recall_type=recall_type, recall_clip=-1,
                )
                oracle_set = sorted(set(oracle.tolist()))

                # Non-degenerate: a strict subset of the cache was kept.
                self.assertLess(port_R, self.S)
                self.assertEqual(len(oracle_set), port_R)
                # THE real faithfulness assertion: identical kept sets.
                self.assertEqual(port_set, oracle_set)

    # -- (B) known deviation: ODD span off-by-one ----------------------------

    def test_known_deviation_odd_span(self):
        """An ODD ``span_size`` exposes the verified off-by-one in the span
        window: the reference uses ``arange(-span // 2, ...)`` (Python floor
        of a negative → one token further LEFT), the port uses
        ``arange(-(span // 2), ...)``.

        First we document it at the source: the offset tensors differ (and
        differ by exactly one left-most element).  Then we show it propagates
        end-to-end: the real port's kept set != the faithful oracle's.
        """
        span_size = 31  # ODD

        # Source-level: the span offset tensors disagree by one left element.
        p_off = port_span_offsets(span_size)
        r_off = reference_span_offsets(span_size)
        self.assertFalse(torch.equal(p_off, r_off))
        self.assertEqual(int(r_off[0]), int(p_off[0]) - 1)  # reference reaches 1 left
        self.assertEqual(int(r_off[-1]), int(p_off[-1]))    # same right edge

        # End-to-end: kept sets differ for the same inputs.
        (module, hidden, raw_k, raw_q, values, rotated, kwargs,
         _) = _build_inputs(
            S=self.S, global_size=self.GLOBAL, local_size=self.LOCAL,
        )
        method = ReAttentionMethod(
            global_size=self.GLOBAL, local_size=self.LOCAL, mid_size=1,
            span_size=span_size, recall_type="qk", recall_clip=-1,
            align_local_to_128=True, use_triton_kernel="off",
        )
        port_set, _ = _port_kept_indices(
            method, module, hidden, rotated, values, kwargs,
        )
        oracle = reference_reattention_update(
            raw_q, raw_k, values, global_size=self.GLOBAL,
            local_size=self.LOCAL, mid_size=1, span_size=span_size,
            recall_type="qk", recall_clip=-1,
        )
        oracle_set = sorted(set(oracle.tolist()))
        self.assertNotEqual(port_set, oracle_set)
        # The oracle keeps exactly the one extra (further-left) span token.
        self.assertTrue(set(port_set).issubset(set(oracle_set)))
        self.assertEqual(len(oracle_set) - len(port_set), 1)

    # -- (C) known deviation: default (disabled) alignment -------------------

    def test_known_deviation_default_alignment(self):
        """With ``align_local_to_128=False`` (the port DEFAULT) and an ``S``
        whose raw middle is not a multiple of 128, the port keeps the literal
        ``local_size`` window, so its middle/local boundary differs from the
        reference's *always*-aligned boundary.

        We assert the boundaries differ, and that enabling alignment recovers
        the reference's boundary — pinning the deviation to exactly the
        unconditional-vs-optional alignment behavior.
        """
        S = self.S
        mod_len = (S - self.GLOBAL - self.LOCAL) % 128
        self.assertNotEqual(mod_len, 0)  # the precondition for a deviation

        # Reference: always-aligned local window and its middle/local boundary.
        ref_local = self.LOCAL - (128 - mod_len)
        ref_boundary = S - ref_local

        # Port DEFAULT (alignment off): literal local window.
        port_default = ReAttentionMethod(
            global_size=self.GLOBAL, local_size=self.LOCAL,
            align_local_to_128=False,
        )
        port_local_default = port_default._effective_local_size(S)
        port_boundary_default = S - port_local_default

        self.assertEqual(port_local_default, self.LOCAL)  # untouched
        self.assertNotEqual(port_boundary_default, ref_boundary)

        # Port with alignment ENABLED reproduces the reference boundary.
        port_aligned = ReAttentionMethod(
            global_size=self.GLOBAL, local_size=self.LOCAL,
            align_local_to_128=True,
        )
        port_local_aligned = port_aligned._effective_local_size(S)
        self.assertEqual(port_local_aligned, ref_local)
        self.assertEqual(S - port_local_aligned, ref_boundary)


if __name__ == "__main__":
    unittest.main()
