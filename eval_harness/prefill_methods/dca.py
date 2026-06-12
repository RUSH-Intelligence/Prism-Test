"""Dual Chunk Attention (DCA) — faithful port of ChunkLlama.

Based on "Training-Free Long-Context Scaling of Large Language Models"
(ICML 2024, arXiv:2402.17463) and the reference implementation
(https://github.com/HKUNLP/ChunkLlama: ``chunkllama_attn_replace.py``,
``flash_decoding_chunkllama.py``).

DCA extends a model's context window with no training by decomposing attention
into three components, each with a different RoPE position assignment so that
**no relative distance ever exceeds the pretraining window**:

* **intra-chunk** — query and key both at cyclic positions ``pos % chunk_len``
  (standard attention *within* a chunk), causal.
* **successive** — query at ``(pos % chunk_len) + chunk_len`` (clamped to
  ``chunk_size``) attending the *immediately previous* chunk, non-causal.
* **inter-chunk** — query at the fixed position ``clamp(2·chunk_len-1,
  chunk_size)`` attending *all older* chunks, non-causal.

Keys are stored RoPE-rotated at their **cyclic** position ``pos % chunk_len``;
the three components reuse those same keys and only vary the *query* rotation.
The component outputs are merged by an online-softmax (LSE) rescaling — see
``eval_harness.kernels.dca_flash``.

``chunk_len = chunk_size - local_window``.  Defaults follow
``replace_with_chunkllama`` for an 8K-pretrained Llama-3:
``chunk_size = pretraining_length * 3/4``, ``local_window = pretraining_length / 8``.

Integration
-----------
Unlike the post-attention prefill hook used by ReAttention, DCA must intercept
the attention computation itself (it re-applies RoPE with its own positions and
runs a 3-way decomposition).  Following the reference — which monkeypatches
``LlamaAttention.forward`` — this method **replaces each attention layer's
``forward``** for the duration of the context-manager, adapted to the
modern transformers signature (``position_embeddings`` kwarg; absolute
positions recovered from ``position_ids`` — transformers 5.x no longer passes
``cache_position`` to attention modules; it is honoured first when a
4.x-style caller supplies it).  The replacement handles the single
full-context prefill pass *and* the per-step decode calls, so the
``SketchTextGenerationPipeline`` keeps the method active across both phases.

The attention kernel is :func:`eval_harness.kernels.flash_attn_with_lse`, which
uses the real ``flash_attn`` library on CUDA and a pure-torch fallback on CPU.
"""

from __future__ import annotations

import logging
import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator, List, Optional, Set, Tuple

import torch
from torch import nn

from .base import PrefillMethod, apply_rotary_pos_emb, build_cos_sin, get_inv_freq
from .registry import register_prefill_method

from eval_harness.kernels.dca_flash import (
    flash_attn_with_lse,
    get_mscale,
    merge_attn_outputs,
)

logger = logging.getLogger(__name__)


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads to query heads (GQA), mirroring HF ``repeat_kv``."""
    if n_rep == 1:
        return x
    B, H_kv, S, D = x.shape
    x = x[:, :, None, :, :].expand(B, H_kv, n_rep, S, D)
    return x.reshape(B, H_kv * n_rep, S, D)


@register_prefill_method("dca", aliases=["dual_chunk_attention", "chunkllama"])
@dataclass
class DCAMethod(PrefillMethod):
    """Faithful 3-component Dual Chunk Attention.

    Parameters
    ----------
    chunk_size : int
        DCA chunk size (the reference's ``pretraining_length * 3/4``).
        Default 6144 (Llama-3, 8K pretraining).
    local_window : int
        Tokens reserved for the local window; ``chunk_len = chunk_size -
        local_window``.  Default 1024 (``pretraining_length / 8``).
    pretraining_length : int
        Original max position embeddings, used only for the ``mscale`` logit
        scaler (active only when the sequence exceeds it).  Default 8192.
    scaling_factor : float
        Positional-interpolation factor (positions divided by it).  Default 1.0
        (no PI), matching the paper's reported results.
    mscale_coeff : float
        Coefficient in ``get_mscale`` (0.05 in chunkllama_attn_replace, 0.1 in
        flash_decoding).  Default 0.05.
    use_flash_attn : str
        Attention kernel backend: ``'auto'`` (flash-attn on CUDA, else torch),
        ``'force'`` (require flash-attn), or ``'off'`` (always pure-torch).
    """

    chunk_size: int = 6144
    local_window: int = 1024
    pretraining_length: int = 8192
    scaling_factor: float = 1.0
    mscale_coeff: float = 0.05
    use_flash_attn: str = "auto"

    _inv_freq: Optional[torch.Tensor] = field(default=None, repr=False, init=False)
    _saved_forwards: dict = field(default_factory=dict, repr=False, init=False)

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    @property
    def chunk_len(self) -> int:
        return self.chunk_size - self.local_window

    @property
    def _backend(self) -> str:
        mode = (self.use_flash_attn or "auto").lower()
        return {"off": "torch", "auto": "auto", "force": "force"}.get(mode, "auto")

    def setup(self, model: Any) -> None:
        self._inv_freq = get_inv_freq(model)

    @property
    def supports_chunked_prefill(self) -> bool:
        return True

    def supported_backends(self) -> Set[str]:
        return {"research"}

    # ------------------------------------------------------------------
    # Context manager: replace attention forwards (Strategy A monkeypatch)
    # ------------------------------------------------------------------

    @contextmanager
    def __call__(self, model: Any) -> Generator:
        from eval_harness.sketch.sketches.base_sketch import _is_non_full_attention_layer

        self._inv_freq = get_inv_freq(model)
        if self._inv_freq is None:
            logger.warning("DCA: could not extract inv_freq; running as a no-op.")
            yield
            return

        is_gemma3 = self._is_gemma3(model)
        language_model = (
            model.model.language_model
            if hasattr(model.model, "language_model")
            else model.model
        )

        if self._saved_forwards:
            raise RuntimeError(
                "DCAMethod context manager is not re-entrant; it is already "
                "installed on a model.",
            )
        saved = {}  # local so an overlapping __call__ cannot clobber restore
        self._saved_forwards = saved
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
                attn.forward = self._make_dca_forward(attn, layer_idx)
            yield
        finally:
            for layer in language_model.layers:
                attn = layer.self_attn
                idx = getattr(attn, "layer_idx", None)
                if idx in saved:
                    attn.forward = saved[idx]
            self._saved_forwards = {}

    @staticmethod
    def _is_gemma3(model: Any) -> bool:
        try:
            from transformers import Gemma3ForConditionalGeneration as _C
        except Exception:
            _C = None
        try:
            from transformers import Gemma3ForCausalLM as _G
        except Exception:
            _G = None
        return (_C is not None and isinstance(model, _C)) or (
            _G is not None and isinstance(model, _G)
        )

    # ------------------------------------------------------------------
    # The replacement attention forward (transformers 4.57+/5.x signature)
    # ------------------------------------------------------------------

    def _make_dca_forward(self, attn: nn.Module, layer_idx: int):
        dca = self

        def dca_forward(
            hidden_states: torch.Tensor,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[Any] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs,
        ):
            del position_embeddings, attention_mask  # DCA computes its own RoPE.
            B, q_len, _ = hidden_states.shape
            n_heads, n_kv_heads, head_dim = dca._resolve_heads(attn)

            # Raw (pre-RoPE) projections.
            q = attn.q_proj(hidden_states).view(B, q_len, n_heads, head_dim)
            k = attn.k_proj(hidden_states).view(B, q_len, n_kv_heads, head_dim)
            v = attn.v_proj(hidden_states).view(B, q_len, n_kv_heads, head_dim)

            # Optional QK-norm (Gemma3 / Qwen3) on the head dim.
            q = dca._maybe_norm(getattr(attn, "q_norm", None), q)
            k = dca._maybe_norm(getattr(attn, "k_norm", None), k)

            q = q.transpose(1, 2)  # [B, n_heads, q_len, D]
            k = k.transpose(1, 2)  # [B, n_kv_heads, q_len, D]
            v = v.transpose(1, 2)

            device = hidden_states.device
            abs_pos = dca._abs_positions(
                cache_position, kwargs.get("position_ids"),
                past_key_values, layer_idx, q_len, device,
            )  # [q_len] absolute positions

            cl = dca.chunk_len
            # Cyclic key positions; store cyclic-rotated keys in the cache.
            kv_len_total = abs_pos[-1].item() + 1 if abs_pos.numel() else q_len
            mscale = get_mscale(kv_len_total / dca.pretraining_length, dca.mscale_coeff)

            k_cyc = dca._rope(k, abs_pos % cl, mscale)

            if past_key_values is not None:
                k_all, v_all = past_key_values.update(k_cyc, v, layer_idx)
            else:
                k_all, v_all = k_cyc, v

            kv_len = k_all.shape[2]
            scale = float(getattr(attn, "scaling", 1.0 / math.sqrt(head_dim)))

            # Query rotations (three position schemes).
            intra_pos = abs_pos % cl
            succ_pos = (intra_pos + cl).clamp(max=dca.chunk_size)
            inter_scalar = min(2 * cl - 1, dca.chunk_size)
            inter_pos = torch.full((q_len,), inter_scalar, device=device, dtype=torch.long)

            q_intra = dca._rope(q, intra_pos, mscale)
            q_succ = dca._rope(q, succ_pos, mscale)
            q_inter = dca._rope(q, inter_pos, mscale)

            is_prefill = kv_len == q_len
            if is_prefill:
                attn_out = dca._dca_prefill_attention(q_intra, q_succ, q_inter, k_all, v_all, scale)
            else:
                attn_out = dca._dca_decode_attention(
                    q_intra, q_succ, q_inter, k_all, v_all, kv_len, scale,
                )

            attn_out = attn_out.transpose(1, 2).contiguous().reshape(B, q_len, -1)
            attn_out = attn.o_proj(attn_out)
            return attn_out, None

        return dca_forward

    # ------------------------------------------------------------------
    # 3-component attention (prefill + decode) — unit-testable
    # ------------------------------------------------------------------

    def _dca_prefill_attention(
        self,
        q_intra: torch.Tensor,
        q_succ: torch.Tensor,
        q_inter: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        """Chunked prefill: intra (+ succ + inter) per chunk, LSE-merged."""
        cl = self.chunk_len
        kv_len = k.shape[2]
        backend = self._backend

        flash_results: List = []
        # First chunk: intra only.
        end0 = min(cl, kv_len)
        flash_results.append([
            flash_attn_with_lse(
                q_intra[:, :, :end0], k[:, :, :end0], v[:, :, :end0],
                causal=True, scale=scale, backend=backend,
            )
        ])

        remain = kv_len - cl
        k_prev, v_prev = k[:, :, :cl], v[:, :, :cl]
        while remain > 0:
            begin = kv_len - remain
            curr = min(cl, remain)
            end = begin + curr
            per_chunk = [
                flash_attn_with_lse(
                    q_intra[:, :, begin:end], k[:, :, begin:end], v[:, :, begin:end],
                    causal=True, scale=scale, backend=backend,
                ),
                flash_attn_with_lse(
                    q_succ[:, :, begin:end], k_prev, v_prev,
                    causal=False, scale=scale, backend=backend,
                ),
            ]
            prev_len = k_prev.shape[2]
            if begin - prev_len > 0:
                per_chunk.append(
                    flash_attn_with_lse(
                        q_inter[:, :, begin:end],
                        k[:, :, : begin - prev_len], v[:, :, : begin - prev_len],
                        causal=False, scale=scale, backend=backend,
                    )
                )
            flash_results.append(per_chunk)
            k_prev, v_prev = k[:, :, begin:end], v[:, :, begin:end]
            remain -= cl

        return merge_attn_outputs(flash_results, decoding=False)

    def _dca_decode_attention(
        self,
        q_intra: torch.Tensor,
        q_succ: torch.Tensor,
        q_inter: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_len: int,
        scale: float,
    ) -> torch.Tensor:
        """Decode / continuation: 3 components over cached chunks.

        Single-token steps (``q_len == 1``) use the LSE merge, mirroring
        ``flash_decoding_chunkllama.py``.  Multi-token blocks (the question
        pass) must follow the *attn_replace* decode branch instead: the LSE
        decomposition assigns every query the components of the LAST key's
        chunk, so a query block straddling a chunk boundary gives pre-boundary
        queries a fully-masked intra slice (softmax over nothing → NaN) and
        unmasked succ/inter access to future in-block keys.  The reference
        (``chunkllama_attn_replace.py`` decode branch) concatenates the three
        score blocks over the full key range and applies one causal mask +
        softmax, which is exact for any block — so that is what ``q_len > 1``
        does here.
        """
        if q_intra.shape[2] > 1:
            return self._dca_decode_attention_multitoken(
                q_intra, q_succ, q_inter, k, v, kv_len, scale,
            )

        cl = self.chunk_len
        backend = self._backend
        chunk_num = (kv_len - 1) // cl

        results = [
            # intra: current (last) chunk, causal within the query block.
            flash_attn_with_lse(
                q_intra, k[:, :, cl * chunk_num:kv_len], v[:, :, cl * chunk_num:kv_len],
                causal=True, scale=scale, backend=backend,
            )
        ]
        if chunk_num >= 1:
            results.append(
                flash_attn_with_lse(
                    q_succ,
                    k[:, :, cl * (chunk_num - 1):cl * chunk_num],
                    v[:, :, cl * (chunk_num - 1):cl * chunk_num],
                    causal=False, scale=scale, backend=backend,
                )
            )
        if chunk_num >= 2:
            results.append(
                flash_attn_with_lse(
                    q_inter,
                    k[:, :, : cl * (chunk_num - 1)],
                    v[:, :, : cl * (chunk_num - 1)],
                    causal=False, scale=scale, backend=backend,
                )
            )
        return merge_attn_outputs(results, decoding=True)

    def _dca_decode_attention_multitoken(
        self,
        q_intra: torch.Tensor,
        q_succ: torch.Tensor,
        q_inter: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_len: int,
        scale: float,
    ) -> torch.Tensor:
        """Reference decode branch for ``q_len > 1`` (attn_replace:201-234).

        Score blocks are concatenated key-ascending — ``[inter | succ |
        intra]`` covers keys ``[0, cl·(cn-1)) | [cl·(cn-1), cl·cn) |
        [cl·cn, kv)`` — then a causal mask over absolute positions (query
        ``i`` is key ``kv_len - q_len + i``) and ONE fp32 softmax over the
        full key range.  No row can be empty (each query sees at least its
        own key), so no NaN; in-block future keys are masked per query.
        """
        cl = self.chunk_len
        chunk_num = (kv_len - 1) // cl
        B, H_q, q_len, _ = q_intra.shape
        n_rep = max(1, H_q // k.shape[1])
        k = _repeat_kv(k, n_rep).float()
        v = _repeat_kv(v, n_rep)

        scores = [
            q_intra.float() @ k[:, :, cl * chunk_num:kv_len].transpose(2, 3) * scale
        ]
        if chunk_num >= 1:
            scores.insert(
                0,
                q_succ.float()
                @ k[:, :, cl * (chunk_num - 1):cl * chunk_num].transpose(2, 3)
                * scale,
            )
        if chunk_num >= 2:
            scores.insert(
                0,
                q_inter.float() @ k[:, :, : cl * (chunk_num - 1)].transpose(2, 3) * scale,
            )
        attn_weights = torch.cat(scores, dim=-1)  # [B, H_q, q_len, kv_len]

        device = attn_weights.device
        key_idx = torch.arange(kv_len, device=device)
        query_abs = kv_len - q_len + torch.arange(q_len, device=device)
        causal = key_idx[None, :] > query_abs[:, None]  # [q_len, kv_len]
        attn_weights = attn_weights.masked_fill(causal, torch.finfo(torch.float32).min)

        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32)
        return (attn_weights @ v.float()).to(q_intra.dtype)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _rope(self, x: torch.Tensor, positions: torch.Tensor, mscale: float) -> torch.Tensor:
        """Apply RoPE to ``x`` [B, H, S, D] at the given (1-D) positions.

        Positions are divided by ``scaling_factor`` (PI) and cos/sin scaled by
        ``mscale``, matching ChunkLlama.
        """
        pos = positions.to(torch.float32) / self.scaling_factor
        cos, sin = build_cos_sin(pos, self._inv_freq.to(x.device), x.device, torch.float32)
        cos = (cos * mscale).to(x.dtype)
        sin = (sin * mscale).to(x.dtype)
        return apply_rotary_pos_emb(x, cos, sin)

    @staticmethod
    def _maybe_norm(norm: Optional[nn.Module], x: torch.Tensor) -> torch.Tensor:
        if norm is None:
            return x
        try:
            return norm(x)
        except Exception:
            return x

    @staticmethod
    def _resolve_heads(attn: nn.Module) -> Tuple[int, int, int]:
        cfg = getattr(attn, "config", None)
        head_dim = getattr(attn, "head_dim", None)
        n_heads = getattr(attn, "num_heads", None)
        n_kv = getattr(attn, "num_key_value_heads", None)
        if cfg is not None:
            n_heads = n_heads or getattr(cfg, "num_attention_heads", None)
            n_kv = n_kv or getattr(cfg, "num_key_value_heads", None)
            if head_dim is None:
                head_dim = getattr(cfg, "head_dim", None)
        if n_kv is None:
            n_kv = n_heads
        return int(n_heads), int(n_kv), int(head_dim)

    @staticmethod
    def _abs_positions(
        cache_position: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        past_key_values: Optional[Any],
        layer_idx: int,
        q_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Recover the absolute positions of the current query tokens.

        ``cache_position`` is honoured first when present (transformers <5
        passed it to attention modules); on transformers 5.x it is always
        ``None`` and positions come from ``position_ids``.
        """
        if cache_position is not None:
            pos = cache_position
            return (pos.squeeze(0) if pos.dim() > 1 else pos).to(device=device, dtype=torch.long)
        if position_ids is not None:
            pos = position_ids
            return (pos[0] if pos.dim() > 1 else pos).to(device=device, dtype=torch.long)
        past_len = 0
        if past_key_values is not None:
            try:
                past_len = past_key_values.get_seq_length(layer_idx)
            except Exception:
                past_len = 0
        return torch.arange(past_len, past_len + q_len, device=device, dtype=torch.long)
