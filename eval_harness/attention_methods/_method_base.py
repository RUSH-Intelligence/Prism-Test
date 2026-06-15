"""Base class for the legacy faithful Door-2 attention methods (ReAttention).

Architecture Note — RoPE State of Q/K/V
----------------------------------------
In the current Prism-Test research backend, Q and K arrive at the
attention hook **already RoPE-rotated** by the model's own layers.  The
KV cache stores **rotated** K/V.  There is no identity-RoPE interceptor
in the current code (``ResearchAdapter`` deletes ``rope_method``).

This means prefill methods that need position-agnostic access to K (like
ReAttention) must either:

1. Use ``hidden_states`` (available in the forward hook kwargs) and
   re-project through the layer's ``q_proj``/``k_proj`` to get raw Q/K.
2. Un-rotate cached K using the model's ``inv_freq`` and the token's
   known absolute position.
3. Use a RoPE-invariant scoring proxy (e.g. key norms).

The base ``PrefillMethod`` class provides utilities for both approaches.

Composition with Sketches
-------------------------
Prefill methods run as a **forward hook on the attention layer**, firing
AFTER the layer's forward pass completes — exactly like sketches.  The
method's hook is registered BEFORE the sketch's hook, so the execution
order is:

    model forward → method hook (select/restructure) → sketch hook (compress)

Both operate on the same ``DynamicCache`` object.
"""

from __future__ import annotations

import logging
import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)


@dataclass
class PrefillMethod:
    """Base class for prefill attention methods.

    **Tier 1** (frequency-only): override ``compute_cos_sin()`` to change
    how RoPE frequencies are generated.  The framework will un-rotate
    cached K/V and re-rotate with your frequencies.

    **Tier 2** (attention restructuring): override ``prefill_forward_hook()``
    to take full control of what happens after each attention layer fires
    during prefill.  This is how ReAttention and DCA are implemented.

    There is no separate post-construction setup step: per-model state
    (``inv_freq``, head dims, saved forwards, …) is initialized inside
    ``__call__`` when the context manager is entered on a model.
    """

    # ------------------------------------------------------------------
    # Tier 1: RoPE frequency override
    # ------------------------------------------------------------------

    def compute_inv_freq(
        self,
        original_inv_freq: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        """Return modified inverse frequencies for RoPE.

        .. warning::
            Tier-1 methods are NOT yet functional: nothing in the pipeline
            calls this hook (it would need a RoPE-level interceptor the
            framework currently lacks).  Declared for forward compatibility.

        Parameters
        ----------
        original_inv_freq : Tensor [head_dim // 2]
            The model's native ``inv_freq`` from ``rotary_emb``.
        seq_len : int
            Total sequence length being processed.

        Returns
        -------
        inv_freq : Tensor [head_dim // 2]
            Modified frequencies.  Default: return ``original_inv_freq``
            unchanged (standard RoPE).
        """
        return original_inv_freq

    # ------------------------------------------------------------------
    # Tier 2: full post-attention hook
    # ------------------------------------------------------------------

    def prefill_forward_hook(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        kwargs: dict,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Called after each attention layer's forward pass during prefill.

        Parameters
        ----------
        module : nn.Module
            The ``self_attn`` module for this layer.
        hidden_states : Tensor [B, S, hidden_dim]
            Input to the attention layer (BEFORE projection/RoPE).
        keys : Tensor [B, H_kv, S_kv, D]
            Keys from the KV cache.  **Already RoPE-rotated.**
        values : Tensor [B, H_kv, S_kv, D]
            Values from the KV cache.
        kwargs : dict
            Full kwargs dict from the forward hook, including
            ``past_key_values`` (the cache object).

        Returns
        -------
        (keys, values) or None
            If not None, the returned K/V replace the cache contents for
            this layer.  Return ``None`` to leave the cache unchanged
            (default behavior — no-op).

        Notes
        -----
        This hook fires BEFORE any sketch hooks.  If you prune K/V here,
        the sketch will see the pruned cache.
        """
        return None

    # ------------------------------------------------------------------
    # Position IDs for question/decode phase
    # ------------------------------------------------------------------

    def compute_question_position_ids(
        self,
        context_length: int,
        question_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Position IDs for question tokens after prefill.

        Returns ``[1, question_length]``.  Default: contiguous starting
        at ``context_length``.
        """
        return torch.arange(
            context_length,
            context_length + question_length,
            device=device,
        ).unsqueeze(0)

    # ------------------------------------------------------------------
    # Chunked prefill lifecycle
    # ------------------------------------------------------------------

    @property
    def supports_chunked_prefill(self) -> bool:
        """Whether this method can process context in chunks."""
        return True

    def on_prefill_start(self, total_context_length: int) -> None:
        """Called once before any chunks are processed."""

    def on_prefill_end(self) -> None:
        """Called once after all chunks are processed."""

    # ------------------------------------------------------------------
    # Backend compatibility
    # ------------------------------------------------------------------

    def supported_backends(self) -> Set[str]:
        """Backends this method works with.  Default: ``{"research"}``."""
        return {"research"}

    # ------------------------------------------------------------------
    # Context manager: install/remove hooks
    # ------------------------------------------------------------------

    @contextmanager
    def __call__(self, model: Any) -> Generator:
        """Context manager that installs this method's hooks on the model.

        Hooks are installed on full-attention ``self_attn`` layers only
        (skipping sliding-window/linear layers on hybrid models).
        """
        from eval_harness.kv_compression.base import _is_non_full_attention_layer

        try:
            _Gemma3Cond = None
            _Gemma3Causal = None
            try:
                from transformers import Gemma3ForConditionalGeneration as _Gemma3Cond
            except ImportError:
                pass
            try:
                from transformers import Gemma3ForCausalLM as _Gemma3Causal
            except ImportError:
                pass

            is_gemma3 = (
                (_Gemma3Cond is not None and isinstance(model, _Gemma3Cond))
                or (_Gemma3Causal is not None and isinstance(model, _Gemma3Causal))
            )
        except Exception:
            is_gemma3 = False

        hooks: List[torch.utils.hooks.RemovableHook] = []
        try:
            language_model = (
                model.model.language_model
                if hasattr(model.model, "language_model")
                else model.model
            )
            for layer in language_model.layers:
                if is_gemma3 and getattr(layer.self_attn, "is_sliding", False):
                    continue
                if _is_non_full_attention_layer(layer):
                    continue
                hooks.append(
                    layer.self_attn.register_forward_hook(
                        self._forward_hook, with_kwargs=True,
                    )
                )
            yield
        finally:
            for h in hooks:
                h.remove()

    def _forward_hook(
        self,
        module: nn.Module,
        input: list[torch.Tensor],
        kwargs: dict,
        output: list,
    ):
        """Internal hook dispatcher — delegates to ``prefill_forward_hook``."""
        from eval_harness.kv_compression.base import KVCompressor
        from eval_harness.kv_compression.utils import extract_keys_and_values
        from transformers import QuantizedCache

        hidden_states = kwargs.get("hidden_states")
        cache = kwargs.get("past_key_values")
        if hidden_states is None or cache is None:
            return output

        q_len = hidden_states.shape[1]

        # Only run during prefill, not decode.
        if KVCompressor._is_decoding_step(module, kwargs, q_len):
            return output

        keys, values = extract_keys_and_values(cache, module.layer_idx)
        result = self.prefill_forward_hook(
            module, hidden_states, keys, values, kwargs,
        )

        if result is not None:
            new_keys, new_values = result
            cache_layer = cache.layers[module.layer_idx]
            if isinstance(cache, QuantizedCache):
                cache_layer._quantized_keys = cache_layer._quantize(
                    new_keys, axis=cache_layer.axis_key,
                )
                cache_layer._quantized_values = cache_layer._quantize(
                    new_values, axis=cache_layer.axis_value,
                )
                cache_layer.keys = torch.zeros(
                    0, dtype=new_keys.dtype, device=new_keys.device,
                )
                cache_layer.values = torch.zeros(
                    0, dtype=new_keys.dtype, device=new_keys.device,
                )
                cache_layer.cumulative_length = new_keys.shape[2]
            else:
                cache_layer.keys = new_keys
                cache_layer.values = new_values

        return output


# ======================================================================
# RoPE utilities — shared by all methods
# ======================================================================


def get_rotary_emb(model: Any) -> Optional[nn.Module]:
    """Extract the shared rotary embedding module from a model."""
    lang = (
        model.model.language_model
        if hasattr(model.model, "language_model")
        else model.model
    )
    rotary = getattr(lang, "rotary_emb", None)
    if rotary is not None:
        return rotary
    # Some architectures store it per-layer.
    if hasattr(lang, "layers") and len(lang.layers) > 0:
        return getattr(lang.layers[0].self_attn, "rotary_emb", None)
    return None


def get_inv_freq(model: Any) -> Optional[torch.Tensor]:
    """Extract ``inv_freq`` from the model's rotary embedding."""
    rotary = get_rotary_emb(model)
    if rotary is None:
        return None
    inv = getattr(rotary, "inv_freq", None)
    if inv is not None:
        return inv
    # Some implementations compute inv_freq lazily and store it as a buffer.
    for name, buf in rotary.named_buffers():
        if "inv_freq" in name:
            return buf
    return None


def apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply RoPE to ``x`` of shape ``[B, H, S, D]``.

    ``cos`` and ``sin`` should be ``[1, 1, S, D]`` or broadcastable.
    """
    return (x * cos) + (_rotate_half(x) * sin)


def undo_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Un-apply RoPE (inverse rotation): R^{-1} = R(-theta).

    Since RoPE is an orthogonal rotation, the inverse is the transpose,
    which is equivalent to negating the sin component.

    This is an exact inverse only for *unit-magnitude* ``(cos, sin)``: trig
    carrying an amplitude scale ``s`` (e.g. HF's baked-in
    ``attention_scaling``) yields ``s²·x`` instead of ``x``.
    """
    return (x * cos) + (_rotate_half(x) * (-sin))


def build_cos_sin(
    position_ids: torch.Tensor,
    inv_freq: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build ``(cos, sin)`` tensors from position IDs and inv_freq.

    Parameters
    ----------
    position_ids : Tensor  [S] or [B, S]
        Absolute positions.
    inv_freq : Tensor  [head_dim // 2]

    Returns
    -------
    cos, sin : Tensor  [1, 1, S, head_dim]
        Ready for broadcast over batch and head dims.
    """
    if position_ids.dim() == 1:
        position_ids = position_ids.unsqueeze(0)
    # [B, S, D/2]
    freqs = torch.einsum(
        "bs, d -> bsd", position_ids.to(dtype=dtype), inv_freq.to(dtype=dtype),
    )
    emb = torch.cat([freqs, freqs], dim=-1)  # [B, S, D]
    cos = emb.cos().unsqueeze(1)  # [B, 1, S, D]
    sin = emb.sin().unsqueeze(1)  # [B, 1, S, D]
    return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Split last dim in half, negate-and-swap."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)
