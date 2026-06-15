import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.models.llama.modeling_llama import repeat_kv, rotate_half

from eval_harness.kv_compression.base import KVCompressor
from eval_harness.kv_compression.registry import register_kv_compressor

logger = logging.getLogger(__name__)


def _get_prerope_query_states(module: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    """Pre-RoPE query states ``[B, H_q, S, D]`` (port of kvpress ``utils.get_prerope_query_states``).

    Duck-typed instead of isinstance checks: a fused ``qkv_proj`` (Phi3-style) is
    sliced for the query block, otherwise a Llama-like ``q_proj`` is used, and a
    ``q_norm`` (Qwen3/Gemma3-style qk-norm) is applied when present.
    """
    bsz, q_len, _ = hidden_states.shape
    num_heads = module.config.num_attention_heads
    head_dim = module.head_dim

    qkv_proj = getattr(module, "qkv_proj", None)
    if qkv_proj is not None:
        qkv = qkv_proj(hidden_states)
        query_states = qkv[..., : num_heads * head_dim]
    elif hasattr(module, "q_proj"):
        query_states = module.q_proj(hidden_states)
    else:
        raise NotImplementedError(f"SimLayerKVSketch not yet implemented for {module.__class__}.")

    query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)

    q_norm = getattr(module, "q_norm", None)
    if q_norm is not None:
        query_states = q_norm(query_states)

    return query_states


@register_kv_compressor("simlayerkv")
@dataclass
class SimLayerKVSketch(KVCompressor):
    """
    SimLayerKV: Similarity-based layer-wise KV cache compression.

    Identifies "lazy" layers that can work effectively with reduced KV cache sizes.
    If a layer is considered "lazy", we only keep the initial and recent KV pairs.
    Otherwise, we keep all KV pairs.

    Recommended lazy_threshold values: Llama3 (0.9), Llama2 (0.65), Mistral (0.8), Qwen (0.85).

    Based on SimLayerKV (https://arxiv.org/abs/2410.13846); port of kvpress
    ``SimLayerKVPress`` (``kvpress/presses/simlayerkv_press.py``), with the window
    attention transcribed from ``SnapKVPress.compute_window_attention``
    (``kvpress/presses/snapkv_press.py``).

    Parameters
    ----------
    lazy_threshold : float, default=1.0
        Threshold for identifying lazy layers based on attention concentration.
        Layer is lazy if sum(attention_weights[last_tokens -> initial+recent]) > threshold.
        Lower values identify more layers as lazy (more aggressive compression).
        The default 1.0 never compresses (the score is a partial softmax mass).
    n_last : int, default=1
        Number of last tokens to analyze for lazy layer identification.
    n_recent : int, default=1024
        Number of recent tokens to preserve in lazy layers.
    n_initial : int, default=4
        Number of initial tokens to preserve in lazy layers (sink tokens).

    Deviations from kvpress
    -----------------------
    - ``compression_ratios`` reset: kvpress resets the telemetry list when
      ``module.layer_idx == 0``; Prism-Test skips non-full-attention layers when
      hooking, so layer 0 may never fire (Gemma3/Qwen3.5). The reset instead
      triggers on the first hooked layer of each prefill (incoming ``layer_idx``
      not greater than the last seen one). Identical behavior on homogeneous
      models (Llama/Mistral/Qwen2/3).
    - ``post_init_from_model`` raises ``ValueError`` when ``lazy_threshold < 1.0``
      and the model is not loaded with ``attn_implementation='flash_attention_2'``:
      lazy layers shrink while non-lazy layers keep full length, and the resulting
      cross-layer ragged cache is only decode-safe under flash attention (same
      implicit constraint as kvpress ``PerLayerCompressionPress``); under
      sdpa/eager the shared causal mask either shape-errors or silently leaks
      causality on the multi-token question forward.
    - Pre-RoPE queries are extracted with duck-typing (``qkv_proj`` slice,
      ``q_norm`` if present) instead of kvpress's isinstance checks — the
      established Prism pattern; numerics are unchanged.
    - ``compression_ratio`` stays a read-only property (not a dataclass field),
      so ``ResearchAdapter._build_kv_compressor`` does not inject the adapter-level
      ratio; configure via ``kv_compressor_kwargs``.
    - Do not combine with the DCA prefill method: DCA stores keys rotated at
      cyclic positions, which breaks the absolute-position window attention.

    Upstream quirks kept verbatim
    -----------------------------
    - The logged compression ratio ``(k_len - n_initial - n_recent + 1) / k_len``
      hardcodes ``n_last=1`` (telemetry only; the selection itself uses
      ``n_last`` correctly).
    - Lazy layers keep ``keys[:, :, -n_recent + n_last:]`` (the last
      ``n_recent - n_last`` tokens) while the laziness score sums the last
      ``n_recent`` entries of the truncated attention vector — an intentional
      reference off-by-window.
    - The short-sequence guard uses ``k_len <= n_initial + n_recent + n_last``
      (boundary inclusive) and only warns.
    """

    lazy_threshold: float = 1.0
    n_last: int = 1  # n_last=1 to match SKLV-decode
    n_recent: int = 1024
    n_initial: int = 4

    def __post_init__(self):
        super().__post_init__()
        assert 0.0 <= self.lazy_threshold <= 1.0, "lazy_threshold should be in [0, 1]"
        self.compression_ratios: list[float] = []
        self._last_layer_idx: Optional[int] = None

    def post_init_from_model(self, model: PreTrainedModel):
        if self.lazy_threshold == 1.0:
            return
        attn_implementation = getattr(getattr(model, "config", None), "_attn_implementation", None)
        if attn_implementation != "flash_attention_2":
            raise ValueError(
                "SimLayerKVSketch with lazy_threshold < 1.0 produces cross-layer ragged caches, "
                "which are only decode-safe under flash_attention_2 (got attn_implementation="
                f"{attn_implementation!r}). Load the model with attn_implementation="
                "'flash_attention_2' or set lazy_threshold=1.0."
            )

    @staticmethod
    def compute_window_attention(
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        window_size: int,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute the last window_size queries and associated attention weights for the first
        q_len - window_size keys (transcribed from kvpress ``SnapKVPress.compute_window_attention``).
        """
        bsz, _, k_len, _ = keys.shape
        head_dim = module.head_dim
        num_key_value_groups = module.config.num_attention_heads // module.config.num_key_value_heads

        query_states = _get_prerope_query_states(module, hidden_states[:, -window_size:])

        cos, sin = position_embeddings
        cos, sin = cos[:, -window_size:], sin[:, -window_size:]
        query_states = (query_states * cos.unsqueeze(1)) + (rotate_half(query_states) * sin.unsqueeze(1))

        key_states = repeat_kv(keys, num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
        attention_mask = torch.ones_like(attn_weights) * float("-inf")
        attention_mask = torch.triu(attention_mask, diagonal=k_len - window_size + 1)
        attn_weights += attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = attn_weights[..., :-window_size]

        return attn_weights

    def is_lazy(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> bool:
        """
        Compute the attention weights of the last tokens over the initial and recent tokens.
        The layer is considered lazy if the sum of these attention weights is above the lazy_threshold.
        """
        attn_weights = self.compute_window_attention(
            module, hidden_states, keys, self.n_last, position_embeddings
        )
        attn_weights = attn_weights.mean((0, 1, 2))  # mean over bsz, heads and window size
        score = attn_weights[: self.n_initial].sum() + attn_weights[-self.n_recent :].sum()
        return score.item() > self.lazy_threshold

    @property
    def compression_ratio(self):
        if len(self.compression_ratios) > 0:
            return sum(self.compression_ratios) / len(self.compression_ratios)
        else:
            raise ValueError("Forward pass must be run to compute the compression ratio")

    @compression_ratio.setter
    def compression_ratio(self, value):
        raise AttributeError(f"compression ratio cannot be set for {type(self).__name__}")

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Reset the compression ratios on the first hooked layer of each prefill
        # (kvpress keys this on layer_idx == 0; see docstring).
        if self._last_layer_idx is None or module.layer_idx <= self._last_layer_idx:
            self.compression_ratios = []
        self._last_layer_idx = module.layer_idx

        k_len = keys.shape[2]
        min_length = self.n_initial + self.n_recent + self.n_last

        if k_len <= min_length:
            logger.warning(f"Sequence length is shorter than {min_length}: no compression applied")

        if (self.lazy_threshold == 1.0) or (k_len <= min_length):
            self.compression_ratios.append(0.0)
            return keys, values

        if self.is_lazy(module, hidden_states, keys, kwargs["position_embeddings"]):
            # If layer is lazy, only keep the initial and recent KV pairs
            keys = torch.cat([keys[:, :, : self.n_initial], keys[:, :, -self.n_recent + self.n_last :]], dim=2)
            values = torch.cat(
                [values[:, :, : self.n_initial], values[:, :, -self.n_recent + self.n_last :]], dim=2
            )
            self.compression_ratios.append((k_len - self.n_initial - self.n_recent + 1) / k_len)
        else:
            self.compression_ratios.append(0.0)

        return keys, values
