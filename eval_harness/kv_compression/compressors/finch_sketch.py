import logging
import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator, Optional

import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.models.llama.modeling_llama import repeat_kv, rotate_half

from eval_harness.kv_compression.registry import register_kv_compressor
from eval_harness.kv_compression.base import ScorerKVCompressor

logger = logging.getLogger(__name__)


def _get_prerope_query_states(module: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    """Pre-RoPE query states, ported from kvpress ``utils.get_prerope_query_states``
    with duck-typing instead of isinstance checks (Phi3 fused ``qkv_proj``,
    Qwen3/Gemma3 ``q_norm``)."""
    bsz, q_len, _ = hidden_states.shape
    num_heads = module.config.num_attention_heads
    head_dim = module.head_dim

    qkv_proj = getattr(module, "qkv_proj", None)
    if qkv_proj is not None:
        qkv = qkv_proj(hidden_states)
        query_states = qkv[..., : num_heads * head_dim]
    elif hasattr(module, "q_proj"):
        # Assume Llama-like attention layer
        query_states = module.q_proj(hidden_states)
    else:
        raise NotImplementedError(f"FinchSketch not yet implemented for {module.__class__}.")

    query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)

    q_norm = getattr(module, "q_norm", None)
    if q_norm is not None:
        query_states = q_norm(query_states)

    return query_states


def _compute_window_attention(
    module: nn.Module,
    hidden_states: torch.Tensor,
    keys: torch.Tensor,
    window_size: int,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Attention weights of the last ``window_size`` queries over the first
    ``k_len - window_size`` keys, transcribed from kvpress
    ``SnapKVPress.compute_window_attention``."""
    bsz, _, k_len, _ = keys.shape
    num_heads = module.config.num_attention_heads
    head_dim = module.head_dim
    num_key_value_groups = num_heads // module.config.num_key_value_heads

    query_states = _get_prerope_query_states(module, hidden_states[:, -window_size:])

    cos, sin = position_embeddings
    cos, sin = cos[:, -window_size:], sin[:, -window_size:]
    query_states = (query_states * cos.unsqueeze(1)) + (rotate_half(query_states) * sin.unsqueeze(1))

    key_states = repeat_kv(keys, num_key_value_groups)
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
    attention_mask = torch.ones_like(attn_weights) * float("-inf")
    attention_mask = torch.triu(attention_mask, diagonal=k_len - window_size + 1)
    attn_weights += attention_mask
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = attn_weights[..., :-window_size]

    return attn_weights


def _rerotate_cos_sin(
    x: torch.Tensor,
    inv_freq: torch.Tensor,
    selected_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(cos, sin) for rotating kept keys from their original positions to the
    contiguous positions ``0..n_kept-1``, transcribed from kvpress
    ``KeyRerotationPress._rerotate_cos_sin``."""
    bsz, num_key_value_heads, n_kept = selected_positions.shape
    device = selected_positions.device
    device_type = x.device.type
    dtype = x.dtype
    idx = torch.arange(0, n_kept, device=device)
    idx = idx.unsqueeze(0)
    inv_freq = inv_freq[None, None, :, None].float().expand(bsz, num_key_value_heads, -1, 1)
    idx = idx[:, None, :].float().expand(bsz, num_key_value_heads, n_kept)
    delta_pos = idx - selected_positions
    delta_pos = delta_pos.unsqueeze(2)

    device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"

    with torch.autocast(device_type=device_type, enabled=False):
        freqs = delta_pos.float() * inv_freq.float()
        freqs = freqs.transpose(2, 3)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().contiguous()
        sin = emb.sin().contiguous()
    return cos.to(dtype=dtype), sin.to(dtype=dtype)


def _rerotate_keys(module: nn.Module, indices: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    """Gather kept keys and re-rotate them to contiguous positions, transcribed
    from kvpress ``KeyRerotationPress.rerotate_keys``."""
    new_cos, new_sin = _rerotate_cos_sin(keys, module.rotary_emb.inv_freq, indices)
    indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)
    keys = keys.gather(2, indices).contiguous()
    return (keys * new_cos) + (rotate_half(keys) * new_sin)


@register_kv_compressor("finch")
@dataclass
class FinchSketch(ScorerKVCompressor):
    """FINCH: Prompt-guided Key-Value Cache Compression.

    SnapKV-style window-attention scoring with per-row normalization, optional
    chunked selection, optional key rerotation, and dynamic window sizing from
    a delimiter token. Requires input format ``context + delimiter_token +
    question``: the delimiter separates context from query, the trailing
    question tokens form the observation window, and the delimiter row is
    removed from the embedding output mid-forward (the model never sees it).
    Call ``update_model_and_tokenizer`` (or set ``delimiter_token_id``
    directly) before entering the sketch context.

    Based on FINCH (https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00716/125280).
    Port of kvpress ``FinchPress`` (kvpress/presses/finch_press.py); window
    attention transcribed from ``SnapKVPress.compute_window_attention``,
    rerotation from ``KeyRerotationPress.rerotate_keys``.

    Parameters
    ----------
    compression_ratio : float, default=0.0
        Fraction of key-value pairs to remove during compression.
    chunk_length : int, optional
        Length of chunks for optional chunked compression (topk applied
        independently per chunk). None processes the entire context at once.
    normalize_scores : bool, default=True
        Whether to normalize attention scores by number of non-zero weights.
    rerotate_keys : bool, default=False
        Whether to rerotate kept keys to contiguous positions ``0..n_kept-1``
        using RoPE. See "Deviations from kvpress" below.
    delimiter_token : str
        Delimiter token string separating context from query (set via
        ``update_model_and_tokenizer``).
    delimiter_token_id : int
        Token ID for the delimiter token (set via ``update_model_and_tokenizer``).
    window_size : int
        Dynamically determined window size based on the delimiter position
        (set automatically by the embedding hook each prefill).

    Deviations from kvpress
    -----------------------
    - ``rerotate_keys`` defaults to ``False`` (kvpress: ``True``). The kvpress
      pipeline special-cases FinchPress to rebase question/decode position ids
      to the compressed cache length; Prism's ``SketchTextGenerationPipeline``
      has no such hook, so rerotated keys (compacted to ``0..n_kept-1``) would
      face inflated relative distances from question/decode positions. With
      ``rerotate_keys=False`` kept keys retain their original rotations and
      positions (kvpress then exhibits the same harmless one-position gap from
      the removed delimiter). A warning is logged when rerotation is enabled.
    - The delimiter-detection hook is registered on the resolved
      ``language_model.embed_tokens`` instead of kvpress's hard-coded
      ``model.model.embed_tokens`` (supports multimodal wrappers).
    - ``compress`` asserts ``k_len > window_size`` so an empty context fails
      with a clear error (kvpress crashes on ``scores.max()`` of an empty
      tensor).
    - ``get_prerope_query_states`` is inlined with duck-typing (``qkv_proj``
      attribute for Phi3-style fused projections, ``q_norm`` for Qwen3/Gemma3)
      instead of isinstance checks; ``compute_window_attention`` is transcribed
      locally rather than imported from a SnapKV port.

    Quirks kept for kvpress parity: the ``attentions is not None`` branch is
    dead in this framework (sdpa never returns attentions); the normalization
    multiplier is the row's absolute position ``p``, not ``p + 1``; without
    rerotation the kept KV pairs stay in unsorted (score-descending) topk
    order; window slots are max-padded so topk tie-breaking may partially drop
    the window when ``n_kept < window_size``. Do not combine with the DCA
    prefill method: its cyclic key positions violate both the window-attention
    and the rerotation position math.
    """

    compression_ratio: float = 0.0
    chunk_length: Optional[int] = None
    normalize_scores: bool = True
    rerotate_keys: bool = False
    delimiter_token: Optional[str] = field(default=None, init=False)
    delimiter_token_id: Optional[int] = field(default=None, init=False)
    window_size: Optional[int] = field(default=None, init=False)

    def __post_init__(self):
        super().__post_init__()
        if self.rerotate_keys:
            logger.warning(
                "FinchSketch(rerotate_keys=True) repositions kept keys to 0..n_kept-1, but "
                "SketchTextGenerationPipeline does not rebase question/decode position ids to "
                "the compressed cache length (kvpress special-cases FinchPress for this); "
                "decode will see inflated relative distances."
            )

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        bsz, num_key_value_heads, k_len, _ = keys.shape
        num_key_value_groups = module.config.num_attention_heads // num_key_value_heads

        if attentions is not None:
            attn_weights = attentions[..., -self.window_size :, : -self.window_size]
        else:
            attn_weights = _compute_window_attention(
                module, hidden_states, keys, self.window_size, kwargs["position_embeddings"]
            )

        if self.normalize_scores:
            non_zero_counts = torch.arange(k_len - self.window_size, k_len)[None, None, :, None]
            non_zero_counts = non_zero_counts.to(attn_weights.device)
            attn_weights = attn_weights * non_zero_counts

        # Average per group
        scores = attn_weights.mean(dim=-2)
        scores = scores.view(bsz, num_key_value_heads, num_key_value_groups, k_len - self.window_size)
        scores = scores.mean(dim=2)

        # Add back the observation window. Use max score to make sure the window is not pruned.
        scores = F.pad(scores, (0, self.window_size), value=scores.max().item())
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
        if self.compression_ratio == 0:
            return keys, values
        assert self.window_size is not None, "window_size must be provided"
        assert keys.shape[2] > self.window_size, (
            f"No context keys to compress: k_len ({keys.shape[2]}) must be greater than "
            f"window_size ({self.window_size})"
        )

        scores = self.score(module, hidden_states, keys, values, attentions, kwargs)

        k_len = keys.shape[2]
        if self.chunk_length is None:
            n_kept = int(k_len * (1 - self.compression_ratio))
            indices = scores.topk(n_kept, dim=-1).indices
        else:
            assert self.chunk_length > self.window_size / (1 - self.compression_ratio)
            indices = []
            for i in range(0, k_len, self.chunk_length):
                chunk_scores = scores[:, :, i : i + self.chunk_length]
                n_kept = max(1, int(chunk_scores.shape[2] * (1 - self.compression_ratio)))
                chunk_indices = i + chunk_scores.topk(n_kept, dim=-1).indices
                indices.append(chunk_indices)
            indices = torch.cat(indices, dim=-1)
        if self.rerotate_keys:
            indices = torch.sort(indices, dim=2).values
            keys = _rerotate_keys(module, indices, keys)
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)
        else:
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)
            keys = keys.gather(2, indices).contiguous()

        values = values.gather(2, indices).contiguous()

        return keys, values

    def embed_token_forward_hook(self, module, input, output):
        """Forward hook on the embedding layer: detect the delimiter token,
        derive the window size, and remove the delimiter row from the output."""
        if input[0].shape[1] > 1 and self.delimiter_token_id in input[0][0]:  # prefilling
            assert len(input[0]) == 1, "Only batch size 1 is supported."
            delim_tokens = input[0][0] == self.delimiter_token_id
            assert delim_tokens.sum() == 1, "Only one delimiter token should be present."
            context_length = int(torch.nonzero(delim_tokens)[0].item())
            self.window_size = len(input[0][0]) - 1 - context_length
            assert self.window_size > 0, "No window detected (window size must be > 0)."
            output = output[:, ~delim_tokens]
        return output

    def update_model_and_tokenizer(self, model, tokenizer, delimiter_token: str = "<|finch_sep|>"):
        """Set the delimiter token and update the tokenizer/model embeddings.
        Must be called before entering the sketch context."""
        self.delimiter_token = delimiter_token
        if delimiter_token not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({"additional_special_tokens": [delimiter_token]})
        self.delimiter_token_id = tokenizer.convert_tokens_to_ids(delimiter_token)
        model.resize_token_embeddings(len(tokenizer))
        return tokenizer

    @contextmanager
    def __call__(self, model: PreTrainedModel) -> Generator:
        if self.delimiter_token_id is None:
            raise ValueError(
                "No delimiter token ID provided. "
                "Use the update_model_and_tokenizer method before calling the sketch."
            )

        with super().__call__(model):
            language_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
            hook = language_model.embed_tokens.register_forward_hook(self.embed_token_forward_hook)
            try:
                yield
            finally:
                hook.remove()
