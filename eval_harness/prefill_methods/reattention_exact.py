"""ReAttention *exact* baseline — full-workflow port of the OpenMOSS reference.

Unlike :class:`~eval_harness.prefill_methods.reattention.ReAttentionMethod`
(a post-attention cache-prune hook that leaves the prefill computation
untouched), this method reproduces the original ReAttention **computation**:

* **Construction** — replaces each full-attention layer's ``self_attn.forward``
  (the same mechanism as the reference's monkeypatched ``LlamaAttention`` and
  as DCA in this framework), active across prefill AND decode.
* **Raw-KV cache** — the HF ``DynamicCache`` stores **raw (pre-RoPE) K** and V
  for the *entire* context and is never pruned (the reference's ``RECacheV2``
  retains everything; selection is an ephemeral per-forward view).
* **Chunked prefill** — the pipeline's single full-context pass is processed
  in ``prefill_chunk_size``-token query chunks *inside* the replaced forward
  (layer-major chunking is mathematically identical to the reference
  wrapper's model-level chunk loop, because chunk ``t`` at layer ``L`` depends
  only on chunks ``<= t`` at layer ``L-1`` and the K/V of chunks ``< t`` at
  layer ``L`` — the same dependency set in either iteration order).
* **Pre-attention recall** — for every chunk (and, under
  ``recall_option='whole'``, every decode step) whose cache exceeds
  ``global_size + local_size``, the raw query scores the raw middle keys
  position-agnostically, and attention runs **only** over the
  ``[global | selected middle | local]`` view — the "finite attention scope"
  of the paper.  This is the path where prefill attention cost is genuinely
  bounded per chunk.
* **pe-after-cache** — RoPE is applied *after* selection: the selected keys
  at their **original absolute positions** (``pe_original=True``), the chunk
  queries at their own absolute positions.

Faithfulness notes (verified against ``cache_utils_v0921.py`` /
``re_attention_llama.py``)
--------------------------------------------------------------------------
* The recall replicates the reference ``update`` branch verbatim, including
  the **unconditional 128-alignment** of the local window
  (``local_eff = local_size - (128 - (seen - g - l) % 128)`` — note this
  *shrinks* local by a full 128 when already aligned, the reference's quirk)
  and ``recall_operation_for_full``'s span expansion
  (``arange(-span//2, (span+1)//2)`` — floor semantics for odd spans,
  empty for ``span_size=0``), clamp, ``+global_size``, unique, sort.
* Kernel parity: the fused ``einsum_topk`` Triton kernel dispatches exactly
  when the reference uses it (``qlen != 1`` and ``mid_size in {1, 4}``, CUDA,
  ``head_dim == 128``; the 128-aligned middle satisfies its constraint by
  construction).  The dense fallback uses the sane GQA pairing of the
  reference's *kernel* path — the reference's own dense fallback reshape
  scrambles head pairing for ``kv_group != n_kv`` (known reference defect,
  documented in the hook port's audit).
* Causality: the reference relies on flash-attention's bottom-right causal
  alignment (its eager path slices the HF mask, which is unsound for chunked
  prefill).  This port masks **by absolute position** (key ``j`` visible to
  query ``i`` iff ``pos_j <= pos_i``) — identical to the flash semantics
  whenever the chunk's keys are the tail of the selected view
  (``local_eff >= chunk``, true for all reference configs), and exact even
  when they are not.  The flash fast path is used only in the tail-aligned
  regime.
* Guard for tiny configs: the reference's alignment is undefined for
  ``local_size <= 128 - mod_len`` (negative local window); this port falls
  back to the un-aligned ``local_size`` there (reference configs always use
  ``local_size = 4096``, where the verbatim formula applies).
* Chunk schedule: ``chunk_schedule='reference'`` (default) replicates the
  wrapper's prefill loop **per forward call** (engineered first chunk so the
  remaining prefill is kernel-aligned; chunks capped at ``q_len - 1``; the
  last token as its own ``qlen==1`` generate-classified chunk).  The wrapper
  sees one concatenated prompt while this pipeline splits context/question
  into two forwards, so boundary placement matches a wrapper run on each
  forward's tokens — e2e numbers match a literal reference run only when
  the token stream enters as one forward.  ``chunk_schedule='uniform'``
  chunks ``[0:C), [C:2C), ...`` with no generate-classified tail (kernel
  alignment via q-padding instead).  For ``recall_option in
  {'generate_only', 'full_attn'}`` the wrapper skips its chunk loop; the
  numerically-identical, memory-bounded uniform chunking is used there.
* ``mid_size == 0`` (StreamingLLM view): the reference re-positions the
  ``[global | local]`` view **contiguously** (``position_ids_for_pe=None``
  → rotary over ``arange(g + l)``, queries at the tail) — replicated here:
  the causal mask still uses absolute positions, the RoPE uses the
  compressed grid.
* Empty recall views (reachable only with ``global_size=0``, outside every
  reference config) are zero-guarded in the eager path (``-inf`` mask +
  ``nan_to_num``) rather than silently attending a uniform average.
* ``pe_original`` — ``True`` (constructor default, matching the reference
  class default): selected keys keep their ORIGINAL absolute RoPE positions,
  so relative distances are unbounded and beyond-native-context behavior is
  position-OOD.  ``False`` — **the setting every published reference
  evaluation uses** (``eval/eval_reattn_niah.py``, ``eval_reattn_infinite``)
  — re-rotates the selected view contiguously (``arange(view_len)``, queries
  at the tail), bounding every relative distance by the view size
  (≈ ``global + recalled + local``): this is the paper's "finite attention
  scope" and the mode to use for context extension experiments.
* Unsupported reference extras: ``q_cache_len > 1``, ``score_record``, and
  ``'_pe'`` recall variants are not implemented (none are used by the
  reference's published evaluations).  ``unique_option`` is accepted-and-dead
  in the reference itself (``unique_operation`` is never called by the v0921
  update path), so it is not a parameter here.

The hook-port-only parameters ``uniform_retained`` / ``uniform_budget`` /
``align_local_to_128`` are inherited but **ignored** (nothing is pruned, so
there is no ragged-cache problem — every layer's cache holds the full
context), and ``reposition=True`` is rejected.
"""

from __future__ import annotations

import logging
import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator, Optional, Set, Tuple

import torch
from torch import nn

from .base import apply_rotary_pos_emb, build_cos_sin, get_inv_freq
from .dca import DCAMethod, _repeat_kv
from .reattention import ReAttentionMethod
from .registry import register_prefill_method

from eval_harness.kernels.dca_flash import flash_attn_with_lse

logger = logging.getLogger(__name__)

_RECALL_OPTIONS = {"whole", "default", "generate_only", "prefill_only", "full_attn"}


@register_prefill_method(
    "reattention_exact", aliases=["re_attention_exact", "reatt_exact"],
)
@dataclass
class ReAttentionExactMethod(ReAttentionMethod):
    """Exact ReAttention workflow: pre-attention recall over a raw-KV cache.

    Inherits the selection parameters/helpers from the hook port
    (``global_size``, ``local_size``, ``mid_size``, ``span_size``,
    ``recall_type``, ``recall_clip``, ``use_triton_kernel``) and adds:

    Parameters
    ----------
    prefill_chunk_size : int
        Query-chunk length for prefill (the reference wrapper's
        ``chunk_size``; default 512).  Decode steps are single chunks.
    recall_option : str
        When the recall runs (reference semantics, default ``'whole'``):
        ``'whole'``/``'default'`` — every chunk and every decode step once the
        cache exceeds ``global_size + local_size``; ``'prefill_only'`` — no
        recall on decode steps (decode attends the FULL cache);
        ``'generate_only'`` — no recall during prefill; ``'full_attn'`` —
        never recall (dense attention over the full raw cache, the exact
        no-op baseline).
    use_flash_attn : str
        Attention backend for the per-chunk attention (same contract as DCA):
        ``'auto'`` (flash-attn via :func:`flash_attn_with_lse` when on CUDA
        *and* the chunk is tail-aligned with the selected view, else the
        exact eager path), ``'force'``, or ``'off'`` (always eager).
    debug_record_selection : bool
        When ``True``, records each layer's most recent selected absolute key
        positions in ``_last_selection[layer_idx]`` (behavior-tracking aid
        for tests; off by default).
    """

    prefill_chunk_size: int = 512
    recall_option: str = "whole"
    chunk_schedule: str = "reference"
    pe_original: bool = True
    use_flash_attn: str = "auto"
    debug_record_selection: bool = False

    _saved_forwards: dict = field(default_factory=dict, repr=False, init=False)
    _last_selection: dict = field(default_factory=dict, repr=False, init=False)
    _max_trained_pos: Optional[int] = field(default=None, repr=False, init=False)
    _oversize_view_warned: bool = field(default=False, repr=False, init=False)

    def __post_init__(self) -> None:
        if self.reposition:
            raise ValueError(
                "ReAttentionExactMethod does not support reposition=True "
                "(pe_original=True is the reference behavior it reproduces).",
            )
        if self.recall_option not in _RECALL_OPTIONS:
            raise ValueError(
                f"recall_option must be one of {sorted(_RECALL_OPTIONS)}, "
                f"got '{self.recall_option}'.",
            )
        if self.chunk_schedule not in ("reference", "uniform"):
            raise ValueError(
                f"chunk_schedule must be 'reference' or 'uniform', "
                f"got '{self.chunk_schedule}'.",
            )
        if self.prefill_chunk_size < 1:
            raise ValueError("prefill_chunk_size must be >= 1.")
        if self.global_size < 1 and self.recall_option != "full_attn":
            # With no sink, an early-chunk query can end up with NO visible
            # key in the recall view (its own key unselected in the middle).
            # The reference never runs g=0; our _attend zero-guards such rows
            # instead of attending uniformly, but the config remains outside
            # the reference-defined regime — warn loudly.
            logger.warning(
                "ReAttentionExact: global_size=0 leaves some queries with an "
                "empty recall view (zero-guarded); the reference always uses "
                "a non-empty sink.",
            )

    # ------------------------------------------------------------------
    # Context manager: replace attention forwards (DCA-style construction)
    # ------------------------------------------------------------------

    @contextmanager
    def __call__(self, model: Any) -> Generator:  # type: ignore[override]
        from eval_harness.sketch.sketches.base_sketch import _is_non_full_attention_layer

        self._inv_freq = get_inv_freq(model)
        if self._inv_freq is None:
            logger.warning(
                "ReAttentionExact: could not extract inv_freq; running as a no-op.",
            )
            yield
            return

        is_gemma3 = DCAMethod._is_gemma3(model)
        language_model = (
            model.model.language_model
            if hasattr(model.model, "language_model")
            else model.model
        )

        if self._saved_forwards:
            raise RuntimeError(
                "ReAttentionExactMethod context manager is not re-entrant; "
                "it is already installed on a model.",
            )
        saved = {}  # local so an overlapping __call__ cannot clobber restore
        self._saved_forwards = saved
        self._last_selection = {}
        self._max_trained_pos = getattr(
            getattr(model, "config", None), "max_position_embeddings", None
        )
        self._oversize_view_warned = False
        try:
            for layer in language_model.layers:
                attn = layer.self_attn
                if is_gemma3 and getattr(attn, "is_sliding", False):
                    continue
                if _is_non_full_attention_layer(layer):
                    continue
                layer_idx = getattr(attn, "layer_idx", None)
                if layer_idx is None:
                    continue
                saved[layer_idx] = attn.forward
                attn.forward = self._make_exact_forward(attn, layer_idx)
            yield
        finally:
            for layer in language_model.layers:
                attn = layer.self_attn
                idx = getattr(attn, "layer_idx", None)
                if idx in saved:
                    attn.forward = saved[idx]
            self._saved_forwards = {}

    # ------------------------------------------------------------------
    # The replacement attention forward
    # ------------------------------------------------------------------

    def _make_exact_forward(self, attn: nn.Module, layer_idx: int):
        method = self

        def exact_forward(
            hidden_states: torch.Tensor,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[Any] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs,
        ):
            # RoPE is applied pe-after-cache with our own positions; the HF
            # mask is replaced by the absolute-position mask built per chunk.
            del position_embeddings, attention_mask
            B, q_len, _ = hidden_states.shape
            n_heads, n_kv_heads, head_dim = DCAMethod._resolve_heads(attn)

            q = attn.q_proj(hidden_states).view(B, q_len, n_heads, head_dim)
            k = attn.k_proj(hidden_states).view(B, q_len, n_kv_heads, head_dim)
            v = attn.v_proj(hidden_states).view(B, q_len, n_kv_heads, head_dim)
            q = DCAMethod._maybe_norm(getattr(attn, "q_norm", None), q)
            k = DCAMethod._maybe_norm(getattr(attn, "k_norm", None), k)
            q = q.transpose(1, 2)  # [B, H_q, q_len, D]  (raw, pre-RoPE)
            k = k.transpose(1, 2)  # [B, H_kv, q_len, D] (raw, cached as-is)
            v = v.transpose(1, 2)

            device = hidden_states.device
            abs_pos = DCAMethod._abs_positions(
                cache_position, kwargs.get("position_ids"),
                past_key_values, layer_idx, q_len, device,
            )
            scale = float(getattr(attn, "scaling", 1.0 / math.sqrt(head_dim)))

            outs = []
            for start, end, is_generate in method._chunk_bounds(q_len):
                q_c = q[:, :, start:end]
                pos_c = abs_pos[start:end]

                # Reference order: cache.update BEFORE selection/RoPE.
                if past_key_values is not None:
                    k_full, v_full = past_key_values.update(
                        k[:, :, start:end], v[:, :, start:end], layer_idx,
                    )
                    if k_full.shape[2] != int(pos_c[-1]) + 1:
                        raise RuntimeError(
                            "ReAttentionExact: cache length "
                            f"{k_full.shape[2]} != last absolute position "
                            f"{int(pos_c[-1])} + 1.  The method requires an "
                            "unpruned raw-KV cache where cache index == "
                            "absolute position; do not compose it with a "
                            "prefill-compression sketch or anything else "
                            "that rewrites the cache.",
                        )
                else:  # cache-less unit-test path
                    k_full, v_full = k[:, :, : end], v[:, :, : end]

                outs.append(
                    method._chunk_attention(
                        attn, layer_idx, q_c, pos_c, k_full, v_full,
                        scale, is_generate,
                    )
                )

            attn_out = torch.cat(outs, dim=2) if len(outs) > 1 else outs[0]
            attn_out = attn_out.transpose(1, 2).contiguous().reshape(B, q_len, -1)
            attn_out = attn.o_proj(attn_out)
            return attn_out, None

        return exact_forward

    # ------------------------------------------------------------------
    # Per-chunk: recall view -> pe-after-cache -> attention
    # ------------------------------------------------------------------

    def _chunk_bounds(self, q_len: int):
        """``(start, end, is_generate)`` chunks for one forward call.

        ``chunk_schedule='reference'`` replicates the reference wrapper's
        prefill loop (``re_attention_wrapper.py:276-286``) *per forward
        call*: the first chunk ends at ``g + l - (128 - mod_len)`` with
        ``mod_len = (q_len - 1 - g + l) % 128`` (so recall never fires on it
        and subsequent chunk lengths stay kernel-aligned), then
        ``prefill_chunk_size`` chunks capped at ``q_len - 1``, and the LAST
        token runs as its own ``qlen==1`` chunk with ``is_generate=True`` —
        the wrapper consumes the prompt's last token through ``generate()``.

        Caveats (documented deviations):

        * The wrapper sees ONE prompt (context + question); this pipeline
          splits them into two forwards, so the schedule restarts per
          forward — boundary placement matches a wrapper run on each
          forward's tokens, not on the concatenated prompt.
        * The wrapper skips its chunk loop entirely for
          ``recall_option in {'generate_only', 'full_attn'}`` (single
          full-attention pass).  Chunked full attention is numerically
          identical (chunk-invariance is tested) and memory-bounded, so
          those options use the uniform schedule here.
        """
        if q_len == 1:
            return [(0, 1, True)]
        cs = self.prefill_chunk_size
        if (
            self.chunk_schedule == "uniform"
            or self.recall_option in ("generate_only", "full_attn")
        ):
            return [
                (s, min(s + cs, q_len), False) for s in range(0, q_len, cs)
            ]

        g, l = self.global_size, self.local_size
        mod_len = (q_len - 1 - g + l) % 128
        first_end = g + l - (128 - mod_len)
        first_end = max(1, min(first_end, q_len - 1))
        # is_generate follows the reference's sole signal — qlen == 1 of the
        # model-level forward (cache_utils_v0921.py:562) — so any 1-token
        # chunk is generate-classified, not just the schedule's final one
        # (a 1-token MID chunk is reachable only with an odd
        # prefill_chunk_size; the wrapper's alignment keeps remainders even).
        bounds = [(0, first_end, first_end == 1)]
        start = first_end
        while start < q_len - 1:
            end = min(start + cs, q_len - 1)
            bounds.append((start, end, end - start == 1))
            start = end
        bounds.append((q_len - 1, q_len, True))
        return bounds

    def _check_recall(self, seen: int, is_generate: bool) -> bool:
        """Verbatim ``RECacheV2.check_recall``."""
        if (
            seen <= self.global_size + self.local_size
            or (self.recall_option == "generate_only" and not is_generate)
            or (self.recall_option == "prefill_only" and is_generate)
            or self.recall_option == "full_attn"
        ):
            return False
        return True

    def _aligned_local_size(self, seen: int) -> int:
        """Reference's unconditional 128-alignment of the local window.

        ``local_eff = local_size - (128 - (seen - g - l) % 128)`` — including
        the quirk that an already-aligned middle still shrinks local by 128.
        Falls back to the un-aligned ``local_size`` when the formula would
        produce a non-positive window (undefined in the reference; only
        reachable with ``local_size <= 128``, far below reference configs).
        """
        mod_len = (seen - self.global_size - self.local_size) % 128
        aligned = self.local_size - (128 - mod_len)
        return aligned if aligned >= 1 else self.local_size

    def _chunk_attention(
        self,
        attn: nn.Module,
        layer_idx: int,
        q_c: torch.Tensor,           # [B, H_q, q_c, D] raw
        pos_c: torch.Tensor,         # [q_c] absolute positions
        k_full: torch.Tensor,        # [B, H_kv, seen, D] raw cache
        v_full: torch.Tensor,
        scale: float,
        is_generate: bool,
    ) -> torch.Tensor:
        device = q_c.device
        seen = k_full.shape[2]
        g = self.global_size
        q_len = q_c.shape[2]

        # ``key_pos``/``pos_c`` are ABSOLUTE positions (causal mask, debug,
        # flash tail-alignment); ``pe_key_pos``/``pe_q_pos`` are the RoPE
        # positions — equal to the absolute ones except in the reference's
        # mid_size==0 branch, which re-positions contiguously.
        if not self._check_recall(seen, is_generate):
            sel_k, sel_v = k_full, v_full
            key_pos = torch.arange(seen, device=device)
            pe_key_pos, pe_q_pos = key_pos, pos_c
        elif self.mid_size == 0:
            # Reference mid_size==0 branch: [global | local], no middle.
            # It returns position_ids_for_pe=None, so the rotary embedding
            # rotates the VIEW at CONTIGUOUS positions arange(g + l) —
            # StreamingLLM-style positional compression — with the queries at
            # the tail of that grid (cache_utils_v0921.py:566-571 +
            # re_attention_llama.py:111-115, apply_rotary_pos_emb q-tail).
            l = self.local_size
            self._require_chunk_fits_view(q_len, g + l)
            sel_k = torch.cat([k_full[:, :, :g], k_full[:, :, -l:]], dim=2)
            sel_v = torch.cat([v_full[:, :, :g], v_full[:, :, -l:]], dim=2)
            key_pos = torch.cat([
                torch.arange(g, device=device),
                torch.arange(seen - l, seen, device=device),
            ])
            pe_key_pos = torch.arange(g + l, device=device)
            pe_q_pos = torch.arange(g + l - q_len, g + l, device=device)
        else:
            key_pos = self._recall_view_positions(q_c, k_full, v_full, seen)
            gather = key_pos.view(1, 1, -1, 1).expand(
                k_full.shape[0], k_full.shape[1], -1, k_full.shape[3],
            )
            sel_k = torch.gather(k_full, 2, gather)
            sel_v = torch.gather(v_full, 2, gather)
            if self.pe_original:
                # pe_original=True: keys at original absolute positions.
                pe_key_pos, pe_q_pos = key_pos, pos_c
            else:
                # pe_original=False (the reference's PUBLISHED long-context
                # eval setting): position_ids_for_pe=None → the rotary
                # embedding rotates the selected view at CONTIGUOUS positions
                # arange(view_len), queries at the tail — every relative
                # distance is bounded by the view size, which is what
                # delivers the paper's beyond-native-context extrapolation.
                view_len = key_pos.numel()
                self._require_chunk_fits_view(q_len, view_len)
                if (
                    self._max_trained_pos is not None
                    and view_len > self._max_trained_pos
                    and not self._oversize_view_warned
                ):
                    self._oversize_view_warned = True
                    logger.warning(
                        "ReAttentionExact: recall view length %d exceeds the "
                        "model's trained window (max_position_embeddings=%d); "
                        "the pe_original=False contiguous re-rotation is "
                        "positionally out-of-distribution and output quality "
                        "will degrade. Bound the view with recall_clip so that "
                        "global_size + recall_clip*span_size + local_size <= "
                        "the trained window (paper: recall_clip=127 for the "
                        "32/4x32/4096 llama3-8b config).",
                        view_len,
                        self._max_trained_pos,
                    )
                pe_key_pos = torch.arange(view_len, device=device)
                pe_q_pos = torch.arange(view_len - q_len, view_len, device=device)

        if self.debug_record_selection:
            self._last_selection[layer_idx] = key_pos.detach().clone()

        # pe after cache: keys at the reference's pe positions (original
        # absolute for the recall view, pe_original=True), queries at theirs.
        inv_freq = self._inv_freq.to(device)
        cos_k, sin_k = build_cos_sin(pe_key_pos, inv_freq, device, torch.float32)
        sel_k = apply_rotary_pos_emb(
            sel_k, cos_k.to(sel_k.dtype), sin_k.to(sel_k.dtype),
        )
        cos_q, sin_q = build_cos_sin(
            pe_q_pos.to(device), inv_freq, device, torch.float32,
        )
        q_rot = apply_rotary_pos_emb(q_c, cos_q.to(q_c.dtype), sin_q.to(q_c.dtype))

        return self._attend(q_rot, sel_k, sel_v, pos_c, key_pos, scale)

    @staticmethod
    def _require_chunk_fits_view(q_len: int, view_len: int) -> None:
        """Contiguous-PE branches need the chunk inside the re-positioned
        view; the reference crashes on a broadcast mismatch here
        (``q * cos[..., -q_len:, :]`` with ``q_len > view_len``), so no
        faithful semantics exist — fail loudly instead of silently rotating
        queries at negative positions."""
        if q_len > view_len:
            raise ValueError(
                f"ReAttentionExact: chunk length {q_len} exceeds the "
                f"re-positioned view length {view_len}; the contiguous-PE "
                "regime (pe_original=False / mid_size=0) requires the chunk "
                "to fit inside the view — increase local_size (>= "
                "prefill_chunk_size + 128) or reduce prefill_chunk_size.",
            )

    def _recall_view_positions(
        self,
        q_c: torch.Tensor,
        k_full: torch.Tensor,
        v_full: torch.Tensor,
        seen: int,
    ) -> torch.Tensor:
        """Absolute positions of the ``[global | selected | local]`` view.

        Verbatim port of the reference ``update`` recall branch +
        ``recall_operation_for_full``.
        """
        device = q_c.device
        g = self.global_size
        local_eff = self._aligned_local_size(seen)
        n_mid = max(seen - g - local_eff, 0)
        if n_mid <= 0:
            return torch.arange(seen, device=device)

        # recall_q / recall_k per recall_type (raw tensors — the cache holds
        # pre-RoPE K, so no un-rotation is needed here, unlike the hook port).
        rt = self.recall_type
        if rt in ("qk", "qkv", "qkv2"):
            recall_q = q_c
            recall_k = self._apply_recall_weight(k_full, v_full)
        elif rt == "k":
            recall_q = k_full[:, :, -q_c.shape[2]:]
            recall_k = k_full
        elif rt == "v":
            recall_q = v_full[:, :, -q_c.shape[2]:]
            recall_k = v_full
        else:
            raise ValueError(
                f"Unsupported recall_type '{rt}' for the exact method. "
                "Use one of: qk, qkv, qkv2, k, v ('_pe' variants are not "
                "implemented).",
            )

        recall_k_mid = recall_k[:, :, g:seen - local_eff, :]
        # Kernel-or-dense top-k, mirroring the reference gate (the kernel runs
        # for qlen != 1 and mid_size in {1, 4}; _should_use_kernel adds the
        # CUDA/head_dim/alignment constraints the kernel itself imposes —
        # n_mid % 128 == 0 holds by construction of local_eff).
        topk_idx = self._select_topk(recall_q, recall_k_mid, n_mid)

        sel_rel = self._global_unique(topk_idx, self.recall_clip)

        # recall_operation_for_full, verbatim (note -span//2 floor semantics
        # and the clamp BEFORE the +global_size shift).
        span = self.span_size
        offsets = torch.arange(-span // 2, (span + 1) // 2, device=device)
        ids = (sel_rel[:, None] + offsets[None, :]).reshape(-1)
        ids = ids.clamp(0, seen - 1 - g - local_eff)
        mid_abs = torch.unique(ids) + g

        return torch.cat([
            torch.arange(g, device=device),
            mid_abs,
            torch.arange(seen - local_eff, seen, device=device),
        ])

    def _should_use_kernel(
        self, raw_q: torch.Tensor, recall_k_mid: torch.Tensor, n_middle: int,
    ) -> bool:
        """Reference kernel gate: dense path for single-token decode steps.

        ``cache_utils_v0921.py:618`` dispatches the fused kernel only for
        ``qlen != 1`` — decode recall always takes the dense einsum.  (The
        inherited gate adds the kernel's own CUDA/head_dim/alignment
        constraints for the prefill-chunk case.)
        """
        if raw_q.shape[2] == 1:
            return False
        return super()._should_use_kernel(raw_q, recall_k_mid, n_middle)

    # ------------------------------------------------------------------
    # Attention over the selected view
    # ------------------------------------------------------------------

    @property
    def _attn_backend(self) -> str:
        mode = (self.use_flash_attn or "auto").lower()
        return {"off": "torch", "auto": "auto", "force": "force"}.get(mode, "auto")

    def _attend(
        self,
        q: torch.Tensor,             # [B, H_q, q_c, D] rotated
        k: torch.Tensor,             # [B, H_kv, S_sel, D] rotated
        v: torch.Tensor,
        q_pos: torch.Tensor,         # [q_c]
        key_pos: torch.Tensor,       # [S_sel]
        scale: float,
    ) -> torch.Tensor:
        q_len = q.shape[2]
        backend = self._attn_backend

        # Flash fast path (reference semantics): bottom-right causal alignment
        # is exact iff the chunk's own keys are the tail of the view and all
        # other keys are strictly past.
        tail_aligned = (
            key_pos.numel() >= q_len
            and bool(torch.equal(key_pos[-q_len:], q_pos.to(key_pos.device)))
            and (key_pos.numel() == q_len or bool(key_pos[-q_len - 1] < q_pos[0]))
        )
        if backend == "force":
            if not tail_aligned:
                raise RuntimeError(
                    "use_flash_attn='force' but the chunk is not tail-aligned "
                    "with the selected view (local window smaller than the "
                    "chunk); the flash bottom-right causal mask would be wrong.",
                )
            if not q.is_cuda:
                raise RuntimeError(
                    "use_flash_attn='force' requires CUDA tensors; got CPU.",
                )
        if backend in ("auto", "force") and tail_aligned and q.is_cuda:
            out, _ = flash_attn_with_lse(
                q, k, v, causal=True, scale=scale, backend=backend,
            )
            return out

        # Exact eager path: additive mask by absolute position, fp32 softmax
        # (the reference eager path upcasts the softmax to fp32 as well).
        # -inf + nan_to_num zero-guards fully-masked rows (reachable only
        # with global_size=0, where a query's view can hold no visible key)
        # instead of leaking a uniform average over future keys — mirroring
        # kernels/dca_flash.attention_with_lse.
        n_rep = max(1, q.shape[1] // k.shape[1])
        k = _repeat_kv(k, n_rep)
        v = _repeat_kv(v, n_rep)
        attn = torch.matmul(q.float(), k.float().transpose(2, 3)) * scale
        masked = key_pos.to(q_pos.device)[None, :] > q_pos[:, None]  # [q_c, S_sel]
        attn = attn.masked_fill(masked, float("-inf"))
        attn = torch.softmax(attn, dim=-1, dtype=torch.float32)
        attn = torch.nan_to_num(attn, nan=0.0)
        return torch.matmul(attn, v.float()).to(q.dtype)

    # ------------------------------------------------------------------
    # Method plumbing
    # ------------------------------------------------------------------

    def prefill_forward_hook(self, *args, **kwargs):  # type: ignore[override]
        """The exact method replaces the forward; the hook seam is unused."""
        return None

    def supported_backends(self) -> Set[str]:
        return {"research"}
