import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.llama.modeling_llama import repeat_kv

from eval_harness.kv_compression.registry import register_kv_compressor
from eval_harness.kv_compression.base import ScorerKVCompressor


def _get_prerope_query_states(module: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    """Extract pre-RoPE query states ``[B, H_q, S, D]``.

    Duck-typed port of kvpress ``utils.get_prerope_query_states``: fused
    ``qkv_proj`` slice (Phi3-style), ``q_proj`` otherwise, with an optional
    ``q_norm`` applied after the head reshape (Qwen3/Gemma3 qk-norm families).
    """
    bsz, q_len, _ = hidden_states.shape
    num_heads = module.config.num_attention_heads
    head_dim = module.head_dim

    qkv_proj = getattr(module, "qkv_proj", None)
    if qkv_proj is not None:
        query_states = qkv_proj(hidden_states)[..., : num_heads * head_dim]
    elif hasattr(module, "q_proj"):
        query_states = module.q_proj(hidden_states)
        # Qwen3.5 gated attention fuses [query | gate] per head into q_proj
        # (output dim = num_heads * head_dim * 2). Slice off the gate to recover
        # the pre-RoPE query, matching Qwen3_5Attention.forward's
        # torch.chunk(q_proj(x).view(*, -1, head_dim * 2), 2, dim=-1).
        if query_states.shape[-1] == num_heads * head_dim * 2:
            query_states = query_states.view(bsz, q_len, num_heads, head_dim * 2)[..., :head_dim]
    else:
        raise NotImplementedError(f"Sketch not yet implemented for {module.__class__}.")

    query_states = query_states.reshape(bsz, q_len, num_heads, head_dim).transpose(1, 2)

    q_norm = getattr(module, "q_norm", None)
    if q_norm is not None:
        query_states = q_norm(query_states)

    return query_states


@register_kv_compressor("expected_attention")
@dataclass
class ExpectedAttentionSketch(ScorerKVCompressor):
    """
    Expected attention-based KV cache compression.

    Port of kvpress ``ExpectedAttentionPress``
    (kvpress/presses/expected_attention_press.py).

    Computes importance scores based on expected attention that future queries
    will pay to current key-value pairs. Uses statistical modeling of query
    patterns and RoPE rotation matrices to predict future attention.
    In particular:
        1. Compute the mean and covariance matrix of the queries before RoPE.
        2. Compute the RoPE rotation matrix R on next n_future_positions and average it
        3. Apply R to the mean and covariance matrice of the queries.
        4. As attention A = exp(Q @ K / sqrt(d)), we compute the expected attention
        E(A) = exp(K @ mean.T / sqrt(d) + 1/2 K @ cov @ K.T / d)
        5. Rescale the scores using (scores + epsilon) * ||V||_2

    The cached keys stay RoPE-rotated at their absolute positions (this is the
    frame the method is designed for); only the query statistics are rotated
    into the averaged future frame. Do not un-rotate the cache.

    Parameters
    ----------
    compression_ratio : float, default=0.0
        Fraction of key-value pairs to remove during compression.
    n_future_positions : int, default=512
        Number of future positions to consider when computing expected attention.
    n_sink : int, default=4
        Number of initial tokens to exclude from compression (sink tokens).
        Preserves first few tokens due to "sink attention" phenomenon where models
        assign high attention to early tokens regardless of semantic importance.
    use_covariance : bool, default=True
        Whether to include covariance information in expected attention computation.
        When True, uses both mean and covariance of query distributions for more
        accurate but computationally expensive scoring. When False, uses only mean.
    use_vnorm : bool, default=True
        Whether to rescale scores using value vector norms.
        Rescales expected attention scores by L2 norm of corresponding value vectors:
        (scores + epsilon) * ||V||_2. Accounts for magnitude of attended information.
    epsilon : float, default=0.0
        Small constant added to scores before value norm rescaling for numerical stability.

    Deviations from kvpress
    -----------------------
    - ``get_prerope_query_states`` is inlined here with duck-typing
      (``qkv_proj`` presence for Phi3-style fused projections, ``q_proj``
      otherwise, ``getattr(module, "q_norm", None)`` for qk-norm families)
      instead of isinstance checks against transformers attention classes.

    Notes
    -----
    - Upstream quirks replicated as-is: ``n_future_positions=0`` is unguarded
      (empty rotation mean propagates NaN scores); prompts with
      ``S <= n_sink`` raise an AssertionError; if ``n_kept < n_sink`` topk
      ties on the max-padded sinks drop some sinks arbitrarily; kept K/V are
      stored in descending-score (not positional) order.
    - Requires even ``head_dim`` and full-width rotary (no partial-rotary
      models), as upstream.
    - Do not compose with the DCA prefill method: DCA stores keys rotated at
      cyclic positions ``pos % chunk_len``, which mismatches the
      absolute-future-position frame of the averaged rotation matrix.
    """

    compression_ratio: float = 0.0
    n_future_positions: int = 512
    n_sink: int = 4
    use_covariance: bool = True
    use_vnorm: bool = True
    epsilon: float = 0.0

    def get_query_statistics(self, module: nn.Module, hidden_states: torch.Tensor):
        """
        Compute the mean and covariance matrix of the queries
        """

        q_len = hidden_states.shape[1]

        # Remove first hidden_states that likely contain outliers
        h = hidden_states[:, self.n_sink :]
        query_states = _get_prerope_query_states(module, h)

        # Query mean
        mu = query_states.mean(dim=2, keepdim=True)

        # Query covariance
        cov = None
        if self.use_covariance:
            centered_states = query_states - mu
            cov = torch.einsum("bnsi,bnsj->bnij", centered_states, centered_states) / h.shape[1]
        mu = mu.squeeze(2)

        # Apply RoPE to the mean and covariance matrix of the queries
        mu, cov = self.apply_avg_rope(module, mu, cov, q_len)

        return mu, cov

    def apply_avg_rope(self, module: nn.Module, mu: torch.Tensor, cov: torch.Tensor, q_len: int):
        """
        Apply average RoPE to the mean and covariance matrix of the queries

        Parameters
        ----------
        module : nn.Module
            The module to apply RoPE to.
        mu : torch.Tensor
            The mean of the queries.
        cov : torch.Tensor
            The covariance matrix of the queries.
        q_len : int
            The length of the queries.

        Returns
        -------
        mu : torch.Tensor
            The mean of the queries after RoPE.
        cov : torch.Tensor
            The covariance matrix of the queries after RoPE.
        """
        position_ids = torch.arange(q_len, q_len + self.n_future_positions).unsqueeze(0).to(mu.device)
        head_dim = module.head_dim
        cos, sin = module.rotary_emb(mu, position_ids)
        cos, sin = cos[0], sin[0]
        # Partial rotary (Qwen3.5): cos/sin span only rotary_dim = head_dim *
        # partial_rotary_factor channels. Build the averaged RoPE rotation on the
        # rotary block and embed it into a full head_dim rotation with an identity
        # passthrough block, mirroring the model's apply_rotary_pos_emb. Reduces to
        # a full rotation when rotary_dim == head_dim.
        rotary_dim = cos.shape[-1]
        half = rotary_dim // 2
        Id = torch.eye(rotary_dim, device=cos.device, dtype=cos.dtype)
        P = torch.zeros((rotary_dim, rotary_dim), device=cos.device, dtype=cos.dtype)
        P[half:, :half] = torch.eye(half, device=cos.device, dtype=cos.dtype)
        P[:half, half:] = -torch.eye(half, device=cos.device, dtype=cos.dtype)
        R_rot = (cos.unsqueeze(1) * Id + sin.unsqueeze(1) * P).mean(dim=0)
        R = torch.eye(head_dim, device=cos.device, dtype=cos.dtype)
        R[:rotary_dim, :rotary_dim] = R_rot
        R = R.to(mu.device)
        mu = torch.matmul(mu, R.T)
        if cov is not None:
            cov = torch.matmul(R, torch.matmul(cov, R.T))
        return mu, cov

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:

        # Remove sink tokens
        assert keys.size(2) > self.n_sink, f"Input should contain more tokens than n_sink={self.n_sink}"
        keys = keys[:, :, self.n_sink :]
        values = values[:, :, self.n_sink :]

        # Compute query statistics
        mean_query, cov_query = self.get_query_statistics(module, hidden_states)

        # Compute scores
        bsz, num_key_value_heads, q_len, d = keys.shape
        num_key_value_groups = module.config.num_attention_heads // num_key_value_heads

        keys = repeat_kv(keys, num_key_value_groups).transpose(2, 3)
        scores = torch.matmul(mean_query.unsqueeze(2), keys).squeeze(2) / math.sqrt(d)
        if self.use_covariance:
            scores += torch.einsum("bhin, bhij, bhjn->bhn", keys, cov_query, keys) / d / 2
        scores = F.softmax(scores, dim=-1)

        # Average scores across groups
        scores = scores.view(bsz, num_key_value_heads, num_key_value_groups, q_len)
        scores = scores.mean(dim=2)

        # Rescale scores by the norm of the values
        if self.use_vnorm:
            scores = (scores + self.epsilon) * values.norm(dim=-1)

        # Add back the sink tokens. Use max score to make sure they are not pruned.
        scores = F.pad(scores, (self.n_sink, 0), value=scores.max().item())

        return scores
