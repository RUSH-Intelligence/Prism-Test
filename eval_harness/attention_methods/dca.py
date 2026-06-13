"""Dual Chunk Attention (DCA) as a Door-2 attention method.

Faithful port of ChunkLlama (arXiv:2402.17463); the attention math is unchanged
from the legacy ``eval_harness.attention_methods.dca`` — only the *installation*
differs: this subclasses :class:`AttentionMethod`, so the base handles the
``self_attn.forward`` replacement, full-attention-layer discovery and
restore, and DCA only supplies :meth:`setup` (capture ``inv_freq``) and
:meth:`attention_forward` (the 3-component decomposition).

DCA computes its own RoPE at cyclic positions, so it runs with ``phase=both``
(active across the single prefill pass and every decode step) and overrides
Door 1 for the layers it owns.  See the legacy module's docstring for the full
algorithm description.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import torch
from torch import nn

from eval_harness.kernels.dca_flash import (
    flash_attn_with_lse,
    get_mscale,
    merge_attn_outputs,
)
from eval_harness.attention_methods._method_base import (
    apply_rotary_pos_emb,
    build_cos_sin,
    get_inv_freq,
)

from .base import AttentionMethod, AttentionPhase
from .registry import register_attention_method

logger = logging.getLogger(__name__)


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads to query heads (GQA), mirroring HF ``repeat_kv``."""
    if n_rep == 1:
        return x
    B, H_kv, S, D = x.shape
    x = x[:, :, None, :, :].expand(B, H_kv, n_rep, S, D)
    return x.reshape(B, H_kv * n_rep, S, D)


@register_attention_method("dca", aliases=["dual_chunk_attention", "chunkllama"])
@dataclass
class DCAMethod(AttentionMethod):
    """Faithful 3-component Dual Chunk Attention (Door 2, ``phase=both``).

    Parameters mirror the legacy method: ``chunk_size`` (default 6144),
    ``local_window`` (default 1024; ``chunk_len = chunk_size - local_window``),
    ``pretraining_length`` (mscale reference, default 8192), ``scaling_factor``
    (PI, default 1.0), ``mscale_coeff`` (default 0.05), ``use_flash_attn``
    (``'auto'`` | ``'force'`` | ``'off'``).
    """

    phase: AttentionPhase = AttentionPhase.BOTH

    chunk_size: int = 6144
    local_window: int = 1024
    pretraining_length: int = 8192
    scaling_factor: float = 1.0
    mscale_coeff: float = 0.05
    use_flash_attn: str = "auto"

    _inv_freq: Optional[torch.Tensor] = field(default=None, repr=False, init=False)

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

    # ------------------------------------------------------------------
    # Door-2 hooks
    # ------------------------------------------------------------------

    def setup(self, model: Any) -> bool:
        self._inv_freq = get_inv_freq(model)
        if self._inv_freq is None:
            logger.warning("DCA: could not extract inv_freq; running as a no-op.")
            return False
        return True

    def _make_dca_forward(self, attn: nn.Module, layer_idx: int):
        """Standalone DCA forward (no phase gating / no saved-forward fallback).

        DCA is ``phase=both`` and decides prefill/decode from kv_len itself, so
        this is exactly the installed forward without the base's gating layer.
        Kept so the reference-oracle tests can drive the math on an un-installed
        instance.
        """
        def dca_forward(
            hidden_states,
            position_embeddings=None,
            attention_mask=None,
            past_key_values=None,
            cache_position=None,
            **kwargs,
        ):
            return self.attention_forward(
                module=attn,
                layer_idx=layer_idx,
                hidden_states=hidden_states,
                is_decode=False,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )

        return dca_forward

    def attention_forward(
        self,
        module: nn.Module,
        layer_idx: int,
        hidden_states: torch.Tensor,
        *,
        is_decode: bool,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ):
        # DCA computes its own RoPE and decides prefill/decode from kv_len.
        del position_embeddings, attention_mask, is_decode
        attn = module
        B, q_len, _ = hidden_states.shape
        n_heads, n_kv_heads, head_dim = self._resolve_heads(attn)

        q = attn.q_proj(hidden_states).view(B, q_len, n_heads, head_dim)
        k = attn.k_proj(hidden_states).view(B, q_len, n_kv_heads, head_dim)
        v = attn.v_proj(hidden_states).view(B, q_len, n_kv_heads, head_dim)

        q = self._maybe_norm(getattr(attn, "q_norm", None), q)
        k = self._maybe_norm(getattr(attn, "k_norm", None), k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        device = hidden_states.device
        abs_pos = self._abs_positions(
            cache_position, kwargs.get("position_ids"),
            past_key_values, layer_idx, q_len, device,
        )

        cl = self.chunk_len
        kv_len_total = abs_pos[-1].item() + 1 if abs_pos.numel() else q_len
        mscale = get_mscale(kv_len_total / self.pretraining_length, self.mscale_coeff)

        k_cyc = self._rope(k, abs_pos % cl, mscale)

        if past_key_values is not None:
            k_all, v_all = past_key_values.update(k_cyc, v, layer_idx)
        else:
            k_all, v_all = k_cyc, v

        kv_len = k_all.shape[2]
        scale = float(getattr(attn, "scaling", 1.0 / math.sqrt(head_dim)))

        intra_pos = abs_pos % cl
        succ_pos = (intra_pos + cl).clamp(max=self.chunk_size)
        inter_scalar = min(2 * cl - 1, self.chunk_size)
        inter_pos = torch.full((q_len,), inter_scalar, device=device, dtype=torch.long)

        q_intra = self._rope(q, intra_pos, mscale)
        q_succ = self._rope(q, succ_pos, mscale)
        q_inter = self._rope(q, inter_pos, mscale)

        is_prefill = kv_len == q_len
        if is_prefill:
            attn_out = self._dca_prefill_attention(q_intra, q_succ, q_inter, k_all, v_all, scale)
        else:
            attn_out = self._dca_decode_attention(
                q_intra, q_succ, q_inter, k_all, v_all, kv_len, scale,
            )

        attn_out = attn_out.transpose(1, 2).contiguous().reshape(B, q_len, -1)
        attn_out = attn.o_proj(attn_out)
        return attn_out, None

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

        ``q_len == 1`` uses the LSE merge (flash_decoding_chunkllama); multi-token
        blocks use the concatenate-then-mask reference branch (attn_replace).
        """
        if q_intra.shape[2] > 1:
            return self._dca_decode_attention_multitoken(
                q_intra, q_succ, q_inter, k, v, kv_len, scale,
            )

        cl = self.chunk_len
        backend = self._backend
        chunk_num = (kv_len - 1) // cl

        results = [
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
        """Reference decode branch for ``q_len > 1`` (attn_replace:201-234)."""
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
        attn_weights = torch.cat(scores, dim=-1)

        device = attn_weights.device
        key_idx = torch.arange(kv_len, device=device)
        query_abs = kv_len - q_len + torch.arange(q_len, device=device)
        causal = key_idx[None, :] > query_abs[:, None]
        attn_weights = attn_weights.masked_fill(causal, torch.finfo(torch.float32).min)

        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32)
        return (attn_weights @ v.float()).to(q_intra.dtype)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _rope(self, x: torch.Tensor, positions: torch.Tensor, mscale: float) -> torch.Tensor:
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
