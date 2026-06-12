"""Flash-attention-with-LSE kernels for Dual Chunk Attention (DCA).

Ported from the ChunkLlama reference implementation
(https://github.com/HKUNLP/ChunkLlama:
``chunkllama_attn_replace.py``, ``flash_decoding_chunkllama.py``).

DCA does not ship a bespoke Triton kernel.  Its "kernel layer" is the
FlashAttention library used in a non-standard way: every attention call must
return the **log-sum-exp (LSE)** of its scores so the three DCA components
(intra / successive / inter) can be merged by an online-softmax rescaling
(``merge_attn_outputs`` / ``_merge_single_chunk``).  The reference exposes the
LSE via:

* prefill — ``flash_attn_func(..., return_attn_probs=True)`` (``do_flash_attn``)
* decode  — a thin wrapper over ``flash_attn_2_cuda.fwd_kvcache``
  (``new_flash_attn_with_kvcache`` / ``do_flash_decoding``)

This module reproduces that layer:

* :func:`attention_with_lse` — a pure-torch scaled-dot-product attention that
  returns ``(output, softmax_lse)`` (works on CPU, no flash-attn / CUDA needed).
* :func:`flash_attn_with_lse` — the dispatch kernel: uses the real
  ``flash_attn_func`` when CUDA + flash-attn are available and the inputs are
  fp16/bf16, otherwise falls back to :func:`attention_with_lse`.
* :func:`new_flash_attn_with_kvcache` — faithful port of the reference decode
  kernel (direct ``flash_attn_2_cuda.fwd_kvcache`` call); CUDA-only.
* :func:`merge_attn_outputs` / :func:`_merge_single_chunk` — the LSE merge.
* :func:`get_mscale` — the (optional) extrapolation logit scaler.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch

__all__ = [
    "attention_with_lse",
    "flash_attn_with_lse",
    "new_flash_attn_with_kvcache",
    "merge_attn_outputs",
    "get_mscale",
    "flash_attn_available",
]


def get_mscale(scale: float = 1.0, coeff: float = 0.05) -> float:
    """Logit scaler for length extrapolation (ChunkLlama ``get_mscale``).

    ``coeff`` is 0.05 in ``chunkllama_attn_replace.py`` and 0.1 in
    ``flash_decoding_chunkllama.py``.  Returns 1.0 (no scaling) when
    ``scale <= 1`` (i.e. sequence within the pretraining length).
    """
    if scale <= 1:
        return 1.0
    return coeff * math.log(scale) + 1.0


def flash_attn_available() -> bool:
    """Whether the flash-attn library is importable."""
    try:
        import flash_attn  # noqa: F401
        from flash_attn.flash_attn_interface import flash_attn_func  # noqa: F401
        return True
    except Exception:
        return False


def attention_with_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure-torch attention returning output **and** log-sum-exp.

    Mirrors ``flash_attn_func(..., return_attn_probs=True)`` semantics, so it
    is a drop-in CPU substitute for the DCA flash kernels.

    Parameters
    ----------
    q : Tensor [B, H_q, S_q, D]
    k, v : Tensor [B, H_kv, S_k, D]
        GQA/MQA supported: KV heads are repeated to match Q heads.
    causal : bool
        Bottom-right-aligned causal mask (the flash-attn convention): query
        ``i`` attends keys ``j <= i + (S_k - S_q)``.
    scale : float, optional
        Softmax scale.  Defaults to ``1/sqrt(D)``.

    Returns
    -------
    output : Tensor [B, H_q, S_q, D]  (in ``v`` dtype)
    softmax_lse : Tensor [B, H_q, S_q]  (float32)
        ``log(sum_j exp(scores_ij))`` per query row (max-stabilized).
    """
    B, H_q, S_q, D = q.shape
    H_kv, S_k = k.shape[1], k.shape[2]
    n_rep = max(1, H_q // H_kv)
    if n_rep > 1:
        k = k[:, :, None].expand(B, H_kv, n_rep, S_k, D).reshape(B, H_q, S_k, D)
        v = v[:, :, None].expand(B, H_kv, n_rep, S_k, D).reshape(B, H_q, S_k, D)

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale  # [B,H,S_q,S_k]

    if causal:
        # Bottom-right alignment: keep j <= i + (S_k - S_q).
        offset = S_k - S_q
        drop = torch.ones(S_q, S_k, dtype=torch.bool, device=q.device).triu(offset + 1)
        scores = scores.masked_fill(drop, float("-inf"))

    max_score = scores.amax(dim=-1, keepdim=True)  # [B,H,S_q,1]
    # Guard fully-masked rows (none in normal DCA use) against -inf/NaN.
    max_score = torch.nan_to_num(max_score, neginf=0.0)
    exp_scores = torch.exp(scores - max_score)
    denom = exp_scores.sum(dim=-1, keepdim=True)  # [B,H,S_q,1]

    softmax_lse = (torch.log(denom) + max_score).squeeze(-1)  # [B,H,S_q]
    output = torch.matmul(exp_scores / denom, v.float())  # [B,H,S_q,D]
    return output.to(v.dtype), softmax_lse.to(torch.float32)


def flash_attn_with_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    scale: Optional[float] = None,
    backend: str = "auto",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Attention returning ``(output, softmax_lse)`` — the DCA kernel.

    ``backend``:
      * ``'auto'`` — use ``flash_attn_func`` when CUDA + flash-attn are present
        and inputs are fp16/bf16; otherwise pure-torch.
      * ``'flash'`` / ``'force'`` — require flash-attn (raise if unavailable).
      * ``'torch'`` — always pure-torch.

    Inputs are ``[B, H, S, D]`` (head-second); flash-attn's ``[B, S, H, D]``
    transposition is handled internally.
    """
    backend = (backend or "auto").lower()

    if backend == "torch":
        return attention_with_lse(q, k, v, causal=causal, scale=scale)

    can_flash = (
        flash_attn_available()
        and q.is_cuda
        and q.dtype in (torch.float16, torch.bfloat16)
    )
    if backend in ("flash", "force") and not flash_attn_available():
        raise RuntimeError("flash_attn backend requested but flash-attn is unavailable.")

    if can_flash or backend in ("flash", "force"):
        try:
            from flash_attn.flash_attn_interface import flash_attn_func

            out, softmax_lse, _ = flash_attn_func(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                causal=causal, softmax_scale=scale, return_attn_probs=True,
            )
            return out.transpose(1, 2), softmax_lse
        except Exception:
            if backend in ("flash", "force"):
                raise
    return attention_with_lse(q, k, v, causal=causal, scale=scale)


def new_flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens=None,
    cache_batch_idx=None,
    block_table=None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    rotary_interleaved=True,
    alibi_slopes=None,
    num_splits=0,
):
    """Faithful port of the ChunkLlama decode kernel (CUDA-only).

    Direct call to ``flash_attn_2_cuda.fwd_kvcache`` that returns
    ``(out, softmax_lse)`` — the stock ``flash_attn_with_kvcache`` returns only
    ``out``.  Provided for parity with the reference; the DCA integration path
    uses :func:`flash_attn_with_lse` over HF-cache slices instead, which is
    numerically equivalent and works with ``transformers``' ``DynamicCache``.

    Raises ``RuntimeError`` if flash-attn CUDA kernels are unavailable.
    """
    try:
        import flash_attn_2_cuda as flash_attn_cuda
    except Exception as exc:  # pragma: no cover - CUDA-only
        raise RuntimeError(
            "new_flash_attn_with_kvcache requires the flash-attn CUDA extension "
            "(flash_attn_2_cuda), which is unavailable here.",
        ) from exc

    assert k_cache.stride(-1) == 1, "k_cache must have contiguous last dimension"
    assert v_cache.stride(-1) == 1, "v_cache must have contiguous last dimension"
    maybe_contig = lambda x: x.contiguous() if x is not None and x.stride(-1) != 1 else x
    q, k, v = [maybe_contig(x) for x in (q, k, v)]
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    if cache_seqlens is not None and isinstance(cache_seqlens, int):
        cache_seqlens = torch.full(
            (k_cache.shape[0],), cache_seqlens, dtype=torch.int32, device=k_cache.device,
        )
        cache_seqlens = maybe_contig(cache_seqlens)
    cache_batch_idx = maybe_contig(cache_batch_idx)
    block_table = maybe_contig(block_table)
    out, softmax_lse = flash_attn_cuda.fwd_kvcache(
        q, k_cache, v_cache, k, v, cache_seqlens, rotary_cos, rotary_sin,
        cache_batch_idx, block_table, alibi_slopes, None, softmax_scale,
        causal, window_size[0], window_size[1], rotary_interleaved, num_splits,
    )
    return out, softmax_lse


def _merge_single_chunk(
    softmax_lse: torch.Tensor, attn_outputs: torch.Tensor,
) -> torch.Tensor:
    """Online-softmax merge of stacked attention components.

    Faithful port of ``_merge_single_chunk`` (computed in float32 for accuracy;
    the reference downcast the weights to bf16).

    Parameters
    ----------
    softmax_lse : Tensor [N, B, H, S_q]   stacked LSEs of N components
    attn_outputs : Tensor [N, B, H, S_q, D]   stacked outputs of N components

    Returns
    -------
    merged : Tensor [B, H, S_q, D]
    """
    softmax_lse = softmax_lse.to(torch.float32)
    max_lse = torch.max(softmax_lse, dim=0).values
    stable = softmax_lse - max_lse.unsqueeze(0)
    weights = torch.exp(stable)
    weights = weights / weights.sum(dim=0)  # [N, B, H, S_q]
    merged = attn_outputs.to(torch.float32) * weights.unsqueeze(-1)
    return merged.sum(dim=0).to(attn_outputs.dtype)


def merge_attn_outputs(
    flash_results: List, decoding: bool = False,
) -> torch.Tensor:
    """LSE merge of DCA attention components — faithful port.

    * ``decoding=True``: ``flash_results`` is a flat list of ``(out, lse)``
      tuples (intra, succ, inter) for a single query block → one merged tensor.
    * ``decoding=False`` (prefill): ``flash_results`` is a list over chunks;
      each chunk is a list of ``(out, lse)`` components.  Each chunk is merged,
      then concatenated along the sequence dim.
    """
    if decoding:
        outs = torch.stack([o for o, _ in flash_results])
        lses = torch.stack([l for _, l in flash_results])
        return _merge_single_chunk(lses, outs)

    merged_chunks = []
    for per_chunk in flash_results:
        outs = torch.stack([o for o, _ in per_chunk])
        lses = torch.stack([l for _, l in per_chunk])
        merged_chunks.append(_merge_single_chunk(lses, outs))
    return torch.cat(merged_chunks, dim=2)
