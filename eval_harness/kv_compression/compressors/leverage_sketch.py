import math
from dataclasses import dataclass

import torch
from torch import nn

from eval_harness.kv_compression.registry import register_kv_compressor
from eval_harness.kv_compression.base import ScorerKVCompressor


def _get_prerope_key_states(module: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    """Recompute pre-RoPE key states from hidden states.

    Port of ``kvpress/utils.py::get_prerope_key_states`` with duck-typed module
    dispatch: the Phi3 fused-projection branch keys on ``hasattr(module, "qkv_proj")``
    and the Qwen3/Gemma3 qk-norm branch on ``getattr(module, "k_norm", None)``,
    replacing kvpress's ``isinstance`` checks on transformers attention classes.

    Parameters
    ----------
    module : nn.Module
        Attention module exposing either a fused ``qkv_proj`` (Phi3-like) or a
        Llama-like ``k_proj``, plus ``head_dim``.
    hidden_states : torch.Tensor
        Input hidden states of shape (batch_size, seq_len, hidden_dim).

    Returns
    -------
    key_states : torch.Tensor
        Pre-RoPE key states of shape (batch_size, num_kv_heads, seq_len, head_dim).
    """
    bsz, k_len, _ = hidden_states.shape
    head_dim = module.head_dim
    if hasattr(module, "qkv_proj"):
        qkv = module.qkv_proj(hidden_states)
        query_pos = module.config.num_attention_heads * module.head_dim
        key_states = qkv[..., query_pos : query_pos + module.num_key_value_heads * module.head_dim]
    elif hasattr(module, "k_proj"):
        key_states = module.k_proj(hidden_states)
    else:
        raise NotImplementedError(f"Sketch not yet implemented for {module.__class__}.")

    key_states = key_states.view(bsz, k_len, -1, head_dim).transpose(1, 2)

    k_norm = getattr(module, "k_norm", None)
    if k_norm is not None:
        key_states = k_norm(key_states)
    return key_states


@register_kv_compressor("leverage")
@dataclass
class LeverageScoreSketch(ScorerKVCompressor):
    """
    Approximate leverage-score scorer on pre-RoPE keys.

    Port of kvpress ``LeverageScorePress`` (kvpress/presses/leverage_press.py).

    Computes geometry-based outlier scores via (approximate) statistical leverage
    on key embeddings using a right Gaussian sketch. Scores are z-score normalized
    (one global affine map over all (B, H, S) elements jointly — it never changes
    standalone top-k selection; it exists for Compactor-style score blending).
    The presented version slightly differs from the paper in that a Cholesky
    decomposition is used to compute the leverage scores.

    Scoring deliberately ignores the cached (RoPE-rotated) keys and rebuilds
    pre-RoPE keys from the hook's ``hidden_states`` via the key projection, so it
    is valid on the research path's rotated cache (including DCA's cyclic
    rotation); selection then gathers the rotated cache K/V unchanged.

    References:
    - Chari & Van Durme (2025): "Compactor: Calibrated Query-Agnostic KV Cache
      Compression with Approximate Leverage Scores" (https://arxiv.org/pdf/2507.08143v1)

    Parameters
    ----------
    compression_ratio : float, default 0.0
        Fraction of KV pairs pruned; ``n_kept = int(S * (1 - compression_ratio))``.
    sketch_dimension : int, default 48
        Size of the Gaussian sketch.

    Notes
    -----
    Only supports prefill (``score`` asserts that ``keys`` and ``hidden_states``
    cover the same tokens). Upstream quirks replicated faithfully: ``torch.randn``
    is drawn without a generator, so scores are nondeterministic run-to-run unless
    the global torch seed is fixed; for a single-token sequence the global z-score
    is NaN (``std`` over one element, and ``clamp_min`` does not cure NaN).

    Deviations from kvpress
    -----------------------
    - ``get_prerope_key_states`` is not available in ``eval_harness.kv_compression.utils``;
      it is ported here as a module-level helper with duck-typed dispatch
      (``qkv_proj``/``k_norm`` attribute checks instead of transformers
      ``isinstance`` gates), per the established Prism pattern.
    """

    sketch_dimension: int = 48

    @staticmethod
    def chol_with_jitter(G: torch.Tensor, jitter: float = 0.0, max_tries: int = 5) -> torch.Tensor:
        """Cholesky factorization with adaptive jitter."""
        identity = torch.eye(G.shape[-1], device=G.device, dtype=G.dtype)
        cur = float(jitter)
        for _ in range(max_tries):
            L, info = torch.linalg.cholesky_ex(G + cur * identity, upper=False)
            if bool((info == 0).all()):
                return L
            cur = max(1e-8, (1e-2 if cur == 0.0 else 10.0 * cur))
        raise RuntimeError(f"Cholesky failed after {max_tries} tries.")

    @staticmethod
    def compute_leverage_scores(key_states: torch.Tensor, sketch_dimension: int) -> torch.Tensor:
        """
        Approximate leverage scores ``diag(X (X^T X)^{-1} X^T)`` on pre-RoPE keys
        via right Gaussian sketching and a Cholesky solve.
        """
        d, k = key_states.shape[-1], sketch_dimension
        Phi = torch.randn(
            key_states.shape[0],
            key_states.shape[1],
            d,
            k,
            device=key_states.device,
            dtype=key_states.dtype,
        ) * (1 / math.sqrt(k))

        X = key_states - key_states.mean(dim=-2, keepdim=True)
        X = torch.matmul(X, Phi).to(torch.float32)  # (B, H, S, k)
        XT = X.transpose(-2, -1)  # (B, H, k, S)
        G = XT @ X  # (B, H, k, k)
        L = LeverageScoreSketch.chol_with_jitter(0.5 * (G + G.transpose(-2, -1)), jitter=1e-2, max_tries=5)
        inv_Xt = torch.cholesky_solve(XT, L, upper=False)  # (X^T X)^{-1} X^T, (B, H, k, S)
        scores = (X * inv_Xt.transpose(-2, -1)).sum(dim=-1).clamp_min(0)  # (B, H, S)
        return scores

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        n_queries = hidden_states.shape[-2]
        assert keys.shape[-2] == n_queries, "LeverageScoreSketch only supports prefill "
        pre_rope_keys = _get_prerope_key_states(module, hidden_states)  # (B, H_kv, S, d)
        scores = self.compute_leverage_scores(pre_rope_keys, self.sketch_dimension)  # (B, H_kv, S)
        z_scores = (scores - scores.mean()) / scores.std().clamp_min(1e-6)
        return z_scores
