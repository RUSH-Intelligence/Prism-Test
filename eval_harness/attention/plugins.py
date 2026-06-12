from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Protocol, Tuple

import torch
import torch.nn.functional as F


AttentionFn = Callable[..., Tuple[torch.Tensor, None]]


class AttentionPlugin(Protocol):
    """Callable attention plugin used by HF ALL_ATTENTION_FUNCTIONS dispatch.

    The model is expected to supply RoPE-applied Q/K tensors.
    """

    def __call__(
        self,
        module: torch.nn.Module,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float = 1.0,
        dropout: float = 0.0,
        **kwargs,
    ) -> Tuple[torch.Tensor, None]:
        ...


@dataclass
class SDPAAttentionPlugin:
    """Reference plugin for Llama-style models using PyTorch SDPA.

    This mirrors the existing HFAdapter fallback behavior and is safe with KV
    pruning because it consumes the pruned K/V tensors as provided.
    """

    def __call__(
        self,
        module: torch.nn.Module,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float = 1.0,
        dropout: float = 0.0,
        **kwargs,
    ) -> Tuple[torch.Tensor, None]:
        del module, kwargs
        keys, values = _expand_kv(queries, keys, values)
        q_len, k_len = queries.shape[-2], keys.shape[-2]
        is_causal = attention_mask is None and q_len == k_len
        try:
            # Keep scaling semantics aligned with HF attention interfaces.
            out = F.scaled_dot_product_attention(
                queries,
                keys,
                values,
                attn_mask=attention_mask,
                dropout_p=dropout,
                is_causal=is_causal,
                scale=scaling,
            )
        except TypeError:
            # Older torch versions do not expose the `scale` kwarg.
            out = F.scaled_dot_product_attention(
                queries * scaling,
                keys,
                values,
                attn_mask=attention_mask,
                dropout_p=dropout,
                is_causal=is_causal,
            )
        return out.transpose(1, 2), None


_PLUGIN_REGISTRY: Dict[str, AttentionPlugin] = {
    "sdpa": SDPAAttentionPlugin(),
}


def list_attention_plugins() -> Iterable[str]:
    return sorted(_PLUGIN_REGISTRY.keys())


def get_attention_plugin(name: Optional[str]) -> Optional[AttentionPlugin]:
    normalized = (name or "").strip().lower()
    if normalized in {"", "none", "default", "hf_default"}:
        return None
    plugin = _PLUGIN_REGISTRY.get(normalized)
    if plugin is None:
        available = ", ".join(list_attention_plugins())
        raise ValueError(f"Unknown attention plugin '{name}'. Available: {available}")
    return plugin


def _expand_kv(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Repeat KV heads to match query heads (GQA/MQA)."""
    h_q, h_kv = queries.shape[1], keys.shape[1]
    if h_q == h_kv:
        return keys, values
    assert h_q % h_kv == 0
    rep = h_q // h_kv
    keys = keys[:, :, None, :, :].expand(-1, h_kv, rep, -1, -1).reshape(
        keys.shape[0], h_q, keys.shape[2], keys.shape[3]
    )
    values = values[:, :, None, :, :].expand(-1, h_kv, rep, -1, -1).reshape(
        values.shape[0], h_q, values.shape[2], values.shape[3]
    )
    return keys, values
