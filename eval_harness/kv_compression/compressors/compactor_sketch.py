import math
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import nn
from transformers.models.llama.modeling_llama import repeat_kv, rotate_half

from eval_harness.kv_compression.registry import register_kv_compressor
from eval_harness.kv_compression.base import ScorerKVCompressor


def _get_prerope_query_states(module: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    """Re-project pre-RoPE queries from hidden_states (mirrors the reference's
    direct use of ``qkv_proj`` output before ``rotary_emb``)."""
    bsz, q_len, _ = hidden_states.shape
    num_heads = module.config.num_attention_heads
    head_dim = module.head_dim

    if hasattr(module, "qkv_proj"):
        qkv = module.qkv_proj(hidden_states)
        query_states = qkv[..., : num_heads * head_dim]
    elif hasattr(module, "q_proj"):
        query_states = module.q_proj(hidden_states)
        # Qwen3.5 gated attention fuses [query | gate] per head into q_proj
        # (output dim = num_heads * head_dim * 2). Slice off the gate to recover
        # the pre-RoPE query, matching Qwen3_5Attention.forward's
        # torch.chunk(q_proj(x).view(*, -1, head_dim * 2), 2, dim=-1).
        if query_states.shape[-1] == num_heads * head_dim * 2:
            query_states = query_states.view(bsz, q_len, num_heads, head_dim * 2)[..., :head_dim]
    else:
        raise NotImplementedError(f"CompactorSketch not yet implemented for {module.__class__}.")

    query_states = query_states.reshape(bsz, q_len, num_heads, head_dim).transpose(1, 2)

    q_norm = getattr(module, "q_norm", None)
    if q_norm is not None:
        query_states = q_norm(query_states)
    return query_states


def _get_prerope_key_states(module: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    """Re-project pre-RoPE keys from hidden_states (the reference scores the raw
    ``k`` split from ``qkv_proj`` before ``rotary_emb``)."""
    bsz, k_len, _ = hidden_states.shape
    head_dim = module.head_dim

    if hasattr(module, "qkv_proj"):
        qkv = module.qkv_proj(hidden_states)
        query_pos = module.config.num_attention_heads * head_dim
        key_states = qkv[..., query_pos : query_pos + module.num_key_value_heads * head_dim]
    elif hasattr(module, "k_proj"):
        key_states = module.k_proj(hidden_states)
    else:
        raise NotImplementedError(f"CompactorSketch not yet implemented for {module.__class__}.")

    key_states = key_states.view(bsz, k_len, -1, head_dim).transpose(1, 2)

    k_norm = getattr(module, "k_norm", None)
    if k_norm is not None:
        key_states = k_norm(key_states)
    return key_states


@register_kv_compressor("compactor")
@dataclass
class CompactorSketch(ScorerKVCompressor):
    """Compactor: Calibrated Query-Agnostic KV Cache Compression with Approximate
    Leverage Scores.

    Faithful port of the **authors' reference engine**
    ``/scratch/sj157/compactor-vllm`` (Chari & Van Durme 2025,
    https://arxiv.org/abs/2507.08143) — ``compression/compactor.py``
    (``approximate_leverage_scores`` + ``non_causal_attn_scores`` +
    ``_zscore_per_batch_epilogue_no_window``) and ``utils/arguments.py`` (shared
    PHI). NOTE: prior revisions of this class ported kvpress 0.5.1
    ``CompactorPress`` instead, which computes DIFFERENT scores (global
    un-chunked leverage, value-norm weighting, avg_pool1d smoothing,
    ``blending=None -> compression_ratio``, 8/4 sinks sliced out of scoring,
    fresh per-call sketch). All of that is gone; scores now follow the
    reference:

    1. **Leverage** (pre-RoPE re-projected keys): ``X = K @ PHI`` with ONE
       shared ``[head_dim, sketch_dimension]`` Gaussian sketch drawn at first
       use with ``seed`` (default 42, matching the engine) and reused across
       layers/calls; the sequence is split into ``leverage_chunk_size`` (512)
       chunks plus a shorter epilogue; each chunk is mean-centered WITHIN the
       chunk; per-chunk Gram in float32 with ``regularizer`` (5e-3) added to
       the diagonal; ``SV = V * S^-1/2`` from ONE batched SVD over ALL chunks
       (full + epilogue together; the 10x-regularizer retry and the
       un-regularized QR fallback both apply to the WHOLE sequence, as in the
       reference — never per chunk-group); ``lev = ||X_c @ SV||^2``
       clamped at 0 (leverage of the regularized Gram — algebraically
       ``x^T (G + reg*I)^{-1} x``); z-scored PER CHUNK over (heads x chunk
       tokens), no epsilon.
    2. **Non-causal chunked attention** (post-RoPE re-projected queries vs the
       cached rotated keys, ``sm_scale = 1.0`` — no 1/sqrt(d)): block-diagonal
       softmax within ``chunk_size`` (128) chunks; column sums accumulated over
       query rows AND query-group members (GQA SUM, not mean) into the KV
       head; the short epilogue chunk softmaxes over its real keys only, and
       the reference kernel's padded query rows (tile granularity
       ``BLOCK_M = 64``) each contribute exactly ``1/chunk_size`` to every
       column — replicated via ``_pad_rows = ceil(M/64)*64 - M``; z-scored per
       sequence over (heads x tokens), no epsilon. Values are NOT used
       (``v`` is dead in the reference kernel).
    3. **Blend + protection**: ``scores = attn_z + blending * lev_z`` with
       ``blending = 0.5`` HARDCODED in the reference call site (kept as a
       field for ablations only); then the first ``sink_size_start`` (16) and
       last ``sink_size_end`` (64) positions are set to ``+inf`` across all
       heads — protected tokens are INCLUDED in the scoring statistics and
       merely overridden afterward, and they consume budget.

    Prefill-only (asserted). When protection covers the whole sequence
    (``T <= sink_size_start + sink_size_end``) compression is skipped entirely,
    mirroring the engine forcing ``compression_ratio = 1.0`` there.

    Structural deviations from the reference (framework constraints, documented)
    -----------------------------------------------------------------------
    - **Uniform per-head top-k instead of "calibrated" ragged allocation**: the
      reference runs ONE top-k over the flattened (token x kv-head) axis
      (``compression/common.py``), giving each head its own retained count. A
      dense HF ``DynamicCache`` must stay rectangular, so this port keeps
      ``int(T * (1 - compression_ratio))`` per head (via
      ``ScorerKVCompressor.compress``). This is the one part of the paper's
      method that cannot be replicated under physical eviction here.
    - **Ratio semantics**: framework convention ``compression_ratio`` =
      fraction REMOVED over the full cached length per head; the reference's
      ``compression_ratio`` = fraction KEPT over the interior
      (``round(ratio * (L - 16 - 64) * H_kv)`` pooled across heads). At long
      context the kept counts differ only by the protected-span accounting
      (~80 tokens).
    - Reference kernels run the softmax in the log2 domain
      (``exp2(x * log2 e)``) and z-score in a Triton kernel; this port uses
      ``torch.softmax``/tensor ops — mathematically identical, not bitwise.
      The reference GPU SVD driver (``gesvda``) is CUDA-only; CPU runs use
      torch's default driver (same factorization).
    - The pre-RoPE re-projections apply ``q_norm``/``k_norm`` when the module
      has them and handle fused ``qkv_proj``, Qwen3.5 gated q_proj and partial
      rotary; on no-RoPE models (NemotronH) the identity rotation is used so
      the non-causal term scores raw q.k — exactly what those models compute.
    - ``phi``: injected sketch matrix used verbatim (test hook), overriding
      the shared seeded PHI.

    Replicated hazards (on purpose): the z-score epilogue divides by
    ``sqrt(var)`` with NO epsilon — a constant-score chunk/sequence yields
    NaN, exactly as the reference kernel would.

    Do not combine with ``attention_method: dca``: DCA caches keys rotated at
    cyclic positions, which breaks the non-causal q.k logits.

    Parameters
    ----------
    compression_ratio : float, default 0.0
        Fraction of key-value pairs to remove per head (sinks consume budget).
    sink_size_start : int, default 16
        Protected leading tokens (reference ``protected_first_tokens``).
    sink_size_end : int, default 64
        Protected trailing tokens (reference ``protected_last_tokens``).
    chunk_size : int, default 128
        Non-causal attention chunk (reference ``CompactorCompression.chunk_size``).
    leverage_chunk_size : int, default 512
        Leverage chunking (reference ``BatchCompressionParams.chunk_size``).
        ``<= 0`` treats the whole sequence as one chunk, as upstream.
    sketch_dimension : int, default 48
        Sketch width (reference ``leverage_sketch_size``).
    regularizer : float, default 5e-3
        Diagonal regularizer on the per-chunk Gram before SVD.
    blending : float, default 0.5
        Weight on leverage z-scores (reference hardcodes 0.5).
    seed : int, default 42
        Seed for the ONE shared PHI draw (reference engine seed).
    phi : Optional[torch.Tensor], default None
        Injected sketch matrix ``[head_dim, sketch_dimension]`` (or
        broadcastable); test hook, used verbatim.
    """

    sink_size_start: int = 16
    sink_size_end: int = 64
    chunk_size: int = 128
    leverage_chunk_size: int = 512
    sketch_dimension: int = 48
    regularizer: float = 5e-3
    blending: float = 0.5
    seed: int = 42
    phi: Optional[torch.Tensor] = None

    # Reference kernel tile height: padded query rows in the epilogue chunk come
    # in BLOCK_M-row tiles, each contributing 1/chunk_size per column.
    _PAD_BLOCK_M = 64

    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.chunk_size > 0, "chunk_size must be positive"
        assert self.sketch_dimension > 0, "sketch_dimension must be positive"
        assert self.regularizer >= 0, "regularizer must be >= 0"
        self._phi_cache: dict = {}

    # ------------------------------------------------------------------
    # Shared sketch matrix
    # ------------------------------------------------------------------

    def _get_phi(self, head_dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """One shared ``[head_dim, k]`` Gaussian sketch per (D, device, dtype).

        Mirrors ``PackedTensorArguments.PHI``: fp32 ``randn`` from a generator
        seeded once, CAST to the model dtype and THEN scaled by ``1/sqrt(k)``
        (that order matters in low precision), reused for every layer and call.
        """
        if self.phi is not None:
            return self.phi.to(device=device, dtype=dtype)
        key = (head_dim, str(device), dtype)
        phi = self._phi_cache.get(key)
        if phi is None:
            gen = torch.Generator(device=device).manual_seed(int(self.seed))
            phi = torch.randn(
                head_dim, self.sketch_dimension, device=device, generator=gen,
            ).to(dtype) * (1.0 / math.sqrt(self.sketch_dimension))
            self._phi_cache[key] = phi
        return phi

    # ------------------------------------------------------------------
    # z-score epilogue (reference _zscore_per_batch_epilogue_no_window)
    # ------------------------------------------------------------------

    @staticmethod
    def _zscore(x: torch.Tensor) -> torch.Tensor:
        """z-score over ALL entries of the trailing (heads, tokens) dims per
        batch element, in fp32, with the reference's biased variance and NO
        epsilon (``invstd = 1/sqrt(var)``)."""
        xf = x.float()
        dims = tuple(range(1, xf.ndim))
        mean = xf.mean(dim=dims, keepdim=True)
        var = (xf * xf).mean(dim=dims, keepdim=True) - mean * mean
        var = var.clamp_min(0.0)
        return ((xf - mean) / var.sqrt()).to(x.dtype)

    # ------------------------------------------------------------------
    # Component 1: chunked approximate leverage scores
    # ------------------------------------------------------------------

    def _sv_from_gram(self, gram: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
        """``V * S^-1/2`` of the regularized fp32 Gram (reference SVD path with
        the 10x-regularizer retry). Raises RuntimeError for the QR fallback."""
        G = gram.clone()
        G.diagonal(dim1=-2, dim2=-1).add_(self.regularizer)
        kwargs = {"full_matrices": False}
        if G.is_cuda:
            kwargs["driver"] = "gesvda"
        try:
            V, S, _ = torch.linalg.svd(G, **kwargs)
        except RuntimeError:
            G.diagonal(dim1=-2, dim2=-1).add_(self.regularizer * 10)
            V, S, _ = torch.linalg.svd(G, **kwargs)
        return (V * S.rsqrt().unsqueeze(-2)).to(out_dtype)

    @staticmethod
    def _leverage_qr_full(x_centered: torch.Tensor) -> torch.Tensor:
        """Reference QR fallback, full-chunk branch: Q of the fp32 reduced QR is
        cast back to model dtype BEFORE squaring (compactor.py:308-311)."""
        Q, _ = torch.linalg.qr(x_centered.to(torch.float32), mode="reduced")
        Q = Q.to(x_centered.dtype)
        return (Q * Q).sum(dim=-1).clamp_min(0.0)

    @staticmethod
    def _leverage_qr_epilogue(x_centered: torch.Tensor) -> torch.Tensor:
        """Reference QR fallback, epilogue branch: squared in fp32, THEN cast
        (compactor.py:324-325; no clamp upstream — Q*Q is nonnegative anyway)."""
        Q, _ = torch.linalg.qr(x_centered.to(torch.float32), mode="reduced")
        return (Q * Q).sum(dim=-1).to(x_centered.dtype)

    def _leverage_scores(self, module: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        """Chunked leverage on pre-RoPE re-projected keys, z-scored PER CHUNK
        over (heads x chunk tokens). Returns ``[B, H_kv, S]`` in model dtype.

        Mirrors the reference's GLOBAL failure scope: ALL chunk Grams (full
        512-chunks AND the epilogue) go through ONE batched SVD, the 10x
        regularizer retry re-regularizes every chunk, and on a second failure
        the ENTIRE sequence falls back to un-regularized QR (compactor.py:
        173-196) — never per chunk-group.
        """
        pre_rope_keys = _get_prerope_key_states(module, hidden_states)  # (B, H, S, D)
        B, H, S, D = pre_rope_keys.shape
        phi = self._get_phi(D, pre_rope_keys.device, pre_rope_keys.dtype)
        X = torch.matmul(pre_rope_keys, phi)  # (B, H, S, k)
        k = X.shape[-1]

        cs = self.leverage_chunk_size
        n_full = S // cs if cs > 0 else 0
        epi = S - n_full * cs if cs > 0 else S

        Xf = Xe = None
        grams = []
        if n_full > 0:
            Xf = X[..., : n_full * cs, :].reshape(B, H, n_full, cs, k)
            Xf = Xf - Xf.mean(dim=-2, keepdim=True)
            grams.append((Xf.transpose(-1, -2) @ Xf).to(torch.float32))  # (B,H,n_full,k,k)
        if epi > 0:
            Xe = X[..., S - epi :, :]
            Xe = Xe - Xe.mean(dim=-2, keepdim=True)
            grams.append((Xe.transpose(-1, -2) @ Xe).to(torch.float32).unsqueeze(2))  # (B,H,1,k,k)

        try:
            sv = self._sv_from_gram(torch.cat(grams, dim=2), X.dtype)  # one SVD over ALL chunks
        except RuntimeError:
            lev_f = self._leverage_qr_full(Xf) if Xf is not None else None
            lev_e = self._leverage_qr_epilogue(Xe) if Xe is not None else None
        else:
            lev_f = None
            if Xf is not None:
                u = Xf @ sv[:, :, :n_full]
                lev_f = (u * u).sum(dim=-1).clamp_min(0.0)  # (B, H, n_full, cs)
            lev_e = None
            if Xe is not None:
                u = Xe @ sv[:, :, -1]
                lev_e = (u * u).sum(dim=-1).clamp_min(0.0)  # (B, H, epi)

        pieces = []
        if lev_f is not None:
            # z-norm per (batch, chunk) over (heads x chunk tokens), no eps
            lf = lev_f.float()
            mean = lf.mean(dim=(1, 3), keepdim=True)
            var = ((lf * lf).mean(dim=(1, 3), keepdim=True) - mean * mean).clamp_min(0.0)
            pieces.append(((lf - mean) / var.sqrt()).to(lev_f.dtype).reshape(B, H, n_full * cs))
        if lev_e is not None:
            pieces.append(self._zscore(lev_e))
        return torch.cat(pieces, dim=-1)

    # ------------------------------------------------------------------
    # Component 2: non-causal chunked attention column sums
    # ------------------------------------------------------------------

    def _non_causal_scores(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Block-diagonal softmax column sums (reference ``_non_causal_attn_kernel``):
        post-RoPE q vs cached rotated keys, ``sm_scale = 1.0``, SUM over query
        rows and GQA group members into the KV head; epilogue pad rows add
        ``1/chunk_size`` per column at BLOCK_M=64 tile granularity; z-scored per
        sequence. Returns fp32 ``[B, H_kv, S]``."""
        q = _get_prerope_query_states(module, hidden_states)  # (B, H_q, S, D)

        q_len = q.shape[-2]
        num_kv_heads = keys.shape[1]
        num_groups = q.shape[1] // num_kv_heads
        # Partial rotary (Qwen3.5: rotary_dim < head_dim) — rotate only the first
        # rotary_dim channels of q so it matches the (partially) RoPE-rotated
        # cached keys. Reduces to full RoPE when rotary_dim == head_dim.
        cos_q, sin_q = cos[:, -q_len:, :].unsqueeze(1), sin[:, -q_len:, :].unsqueeze(1)
        rotary_dim = cos_q.shape[-1]
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        q_rot = (q_rot * cos_q) + (rotate_half(q_rot) * sin_q)
        q = torch.cat([q_rot, q_pass], dim=-1)

        k_full = repeat_kv(keys, num_groups)  # (B, H_q, S, D)
        B, H_q, S, D = q.shape
        cs = self.chunk_size
        out = torch.zeros(B, H_q, S, dtype=torch.float32, device=q.device)

        # fp32 logits like the reference kernel: tl.dot uses an fp32 accumulator
        # and never materializes bf16 logits (sm_scale=1.0 makes them large, so
        # a bf16 round-trip here would visibly perturb the softmax).
        qf = q.to(torch.float32)
        kf = k_full.to(torch.float32)

        for start in range(0, S, cs):
            end = min(start + cs, S)
            qc = qf[..., start:end, :]
            kc = kf[..., start:end, :]
            logits = torch.matmul(qc, kc.transpose(-2, -1))  # sm_scale = 1.0
            p = torch.softmax(logits, dim=-1)
            contrib = p.sum(dim=-2)  # column sums over the chunk's query rows
            m = end - start
            pad_rows = math.ceil(m / self._PAD_BLOCK_M) * self._PAD_BLOCK_M - m
            if pad_rows > 0:
                # reference kernel: invalid query rows contribute exactly
                # 1/CHUNK_SIZE to every (valid) column of the epilogue chunk
                contrib = contrib + pad_rows * (1.0 / cs)
            out[..., start:end] = contrib

        # GQA: SUM over the group's query heads into the KV head (reference
        # tl.sum over BLOCK_M * QUERY_GROUP_SIZE rows).
        out = out.view(B, num_kv_heads, num_groups, S).sum(dim=2)
        return self._zscore(out)

    # ------------------------------------------------------------------
    # Position embeddings for the post-RoPE query rotation
    # ------------------------------------------------------------------

    def _position_embeddings(
        self, module: nn.Module, hidden_states: torch.Tensor, kwargs: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pos_emb = kwargs.get("position_embeddings")
        if isinstance(pos_emb, (tuple, list)) and len(pos_emb) == 2 and pos_emb[0] is not None:
            return pos_emb[0], pos_emb[1]

        rotary = getattr(module, "rotary_emb", None)
        if rotary is None:
            # No-RoPE model (e.g. NemotronH attention): cached keys are
            # un-rotated, so use the identity rotation (cos=1, sin=0) — the
            # non-causal q.k logits then use raw queries against raw keys,
            # exactly what the model computes.
            q_len = hidden_states.shape[-2]
            head_dim = module.head_dim
            cos = torch.ones(
                hidden_states.shape[0], q_len, head_dim,
                device=hidden_states.device, dtype=hidden_states.dtype,
            )
            sin = torch.zeros_like(cos)
            return cos, sin
        cache_position = kwargs.get("cache_position")
        if cache_position is not None:
            position_ids = cache_position.unsqueeze(0)
        else:
            position_ids = torch.arange(hidden_states.shape[-2], device=hidden_states.device).unsqueeze(0)
        cos, sin = rotary(hidden_states, position_ids)
        return cos, sin

    # ------------------------------------------------------------------
    # Score + compress
    # ------------------------------------------------------------------

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        del values, attentions  # the reference kernel never reads v
        n_queries = hidden_states.shape[-2]
        assert keys.shape[-2] == n_queries, "CompactorSketch only supports prefill at the moment"
        T = n_queries
        first = min(self.sink_size_start, T)
        last = min(self.sink_size_end, max(0, T - first))

        cos, sin = self._position_embeddings(module, hidden_states, kwargs)
        attn_z = self._non_causal_scores(module, hidden_states, keys, cos, sin)  # fp32
        lev_z = self._leverage_scores(module, hidden_states)  # model dtype

        scores = attn_z + lev_z.to(torch.float32) * float(self.blending)

        # Protection: +inf rows across ALL heads, AFTER scoring (protected
        # tokens are included in the z-norm statistics, as in the reference).
        if first > 0:
            scores[:, :, :first] = torch.inf
        if last > 0:
            scores[:, :, T - last :] = torch.inf
        return scores

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Reference engine forces compression_ratio = 1.0 (keep everything)
        # when the protected spans cover the whole prompt.
        if keys.shape[-2] <= self.sink_size_start + self.sink_size_end:
            return keys, values
        return super().compress(module, hidden_states, keys, values, attentions, kwargs)
