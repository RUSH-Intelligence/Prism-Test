"""Door 2 — Attention methods.

An *attention method* changes the **attention math** itself: how queries score
keys and how positions are assigned during scoring.  It is installed by
**replacing** each full-attention layer's ``self_attn.forward`` for the duration
of a context manager (the same mechanism the legacy ``PrefillMethod`` DCA path
used), and is gated by an explicit :class:`AttentionPhase` so that **one**
attention implementation can serve prefill, decode, or both with no duplication.

This is the renamed/generalised successor to
``eval_harness.attention_methods`` — the ``prefill_forward_hook`` *prune* sub-style
of the old ``PrefillMethod`` is **not** an attention method (it is KV compression
that happens to fire at prefill time) and moves to Door 3
(:mod:`eval_harness.kv_compression`).  What stays here is the *forward
replacement* sub-style: DCA and reattention_exact.

Pipeline position (outer → inner)::

    positional_method(model)        # door 1
      → attention_method(model)     # door 2  ← THIS
        → kv_compressor(model)      # door 3

Because Door 2 replaces the forward, an attention method that computes its own
RoPE (e.g. DCA) effectively overrides Door 1 for the layers it owns.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Generator, Optional, Set, Tuple

import torch
from torch import nn

logger = logging.getLogger(__name__)


class AttentionPhase(str, Enum):
    """When an attention method's forward replacement is active.

    ``BOTH`` means the *same* code path runs in both phases (no duplication) —
    this is what DCA needs (it keeps its cyclic-RoPE decomposition active across
    the single prefill pass and every decode step).
    """

    PREFILL = "prefill"
    DECODE = "decode"
    BOTH = "both"

    def active_in_prefill(self) -> bool:
        return self in (AttentionPhase.PREFILL, AttentionPhase.BOTH)

    def active_in_decode(self) -> bool:
        return self in (AttentionPhase.DECODE, AttentionPhase.BOTH)

    @classmethod
    def coerce(cls, value: "AttentionPhase | str") -> "AttentionPhase":
        """Parse a phase from an enum member or a config string."""
        if isinstance(value, AttentionPhase):
            return value
        key = str(value).strip().lower()
        try:
            return cls(key)
        except ValueError as exc:
            allowed = ", ".join(p.value for p in cls)
            raise ValueError(
                f"Unknown attention phase {value!r}. Allowed: {allowed}"
            ) from exc


@dataclass
class AttentionMethod:
    """Base class for Door-2 attention methods.

    Subclasses override **one** method, :meth:`attention_forward`, which fully
    replaces the layer's attention computation and must return
    ``(attn_output, attn_weights_or_None)`` in the modern transformers
    convention.  The framework:

    * installs the replacement on full-softmax ``self_attn`` layers only
      (sliding-window / linear layers on hybrid models are skipped);
    * decides per call whether it is a prefill or decode step and, if this
      method is **not** active for that phase (per :attr:`phase`), transparently
      delegates to the layer's original saved forward.

    The default :attr:`phase` is ``both`` — active everywhere.  Set it via
    config (``attention_phase``) or override the default in a subclass.
    """

    phase: AttentionPhase = AttentionPhase.BOTH

    # Per-model state captured on context-manager entry.
    _saved_forwards: Dict[int, Any] = field(default_factory=dict, repr=False, init=False)

    def __post_init__(self) -> None:
        # Allow ``phase="prefill"`` from YAML/config without a manual cast.
        self.phase = AttentionPhase.coerce(self.phase)

    # ------------------------------------------------------------------
    # The single override point
    # ------------------------------------------------------------------

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
        cache_position: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Replacement attention forward.  Override in subclasses.

        Parameters
        ----------
        module : nn.Module
            The ``self_attn`` module being replaced (holds ``q_proj`` … etc.).
        layer_idx : int
            The decoder layer index (cache key).
        hidden_states : Tensor [B, S, hidden_dim]
            Layer input (pre-projection).
        is_decode : bool
            ``True`` on a decode continuation step, ``False`` during prefill —
            supplied so a ``phase=both`` method can still branch internally if
            it needs to (DCA does).
        position_embeddings, attention_mask, past_key_values, cache_position
            The usual transformers attention kwargs.

        Returns
        -------
        (attn_output, attn_weights)
            ``attn_output`` shape ``[B, S, hidden_dim]``; ``attn_weights`` may be
            ``None`` (flash-style).
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement attention_forward()"
        )

    # ------------------------------------------------------------------
    # Capability flags (kept compatible with the legacy PrefillMethod API)
    # ------------------------------------------------------------------

    @property
    def supports_chunked_prefill(self) -> bool:
        return True

    def supported_backends(self) -> Set[str]:
        return {"research"}

    def setup(self, model: Any) -> bool:
        """Per-model setup on context-manager entry (capture ``inv_freq``, …).

        Return ``False`` to skip installation entirely — the method then runs as
        a transparent no-op (the model's original forwards are left in place).
        Default: install.
        """
        return True

    def on_prefill_start(self, total_context_length: int) -> None:
        """Called once before the prefill pass (or its first chunk)."""

    def on_prefill_end(self) -> None:
        """Called once after the prefill pass (or its last chunk)."""

    def compute_question_position_ids(
        self,
        context_length: int,
        question_length: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Optional override of the post-prefill question position IDs.

        Default ``None`` lets the pipeline use contiguous absolute positions.
        """
        return None

    # ------------------------------------------------------------------
    # Context manager: install / restore forward replacements
    # ------------------------------------------------------------------

    @contextmanager
    def __call__(self, model: Any) -> Generator:
        from eval_harness.kv_compression.base import (
            _is_non_full_attention_layer,
        )

        if not self.setup(model):
            yield
            return

        is_gemma3 = _is_gemma3(model)
        language_model = (
            model.model.language_model
            if hasattr(model.model, "language_model")
            else model.model
        )

        if self._saved_forwards:
            raise RuntimeError(
                f"{type(self).__name__} context manager is not re-entrant; "
                "it is already installed on a model."
            )
        saved: Dict[int, Any] = {}
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
                attn.forward = self._make_forward(attn, layer_idx)
            yield
        finally:
            for layer in language_model.layers:
                attn = layer.self_attn
                idx = getattr(attn, "layer_idx", None)
                if idx in saved:
                    attn.forward = saved[idx]
            self._saved_forwards = {}

    def _make_forward(self, attn: nn.Module, layer_idx: int):
        method = self
        saved_forward = self._saved_forwards[layer_idx]

        def forward(
            hidden_states: torch.Tensor,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[Any] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs: Any,
        ):
            is_decode = method._infer_is_decode(
                attn, hidden_states, past_key_values, cache_position, kwargs,
            )
            active = (
                method.phase.active_in_decode()
                if is_decode
                else method.phase.active_in_prefill()
            )
            if not active:
                return saved_forward(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    **kwargs,
                )
            return method.attention_forward(
                module=attn,
                layer_idx=layer_idx,
                hidden_states=hidden_states,
                is_decode=is_decode,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )

        return forward

    @staticmethod
    def _infer_is_decode(
        attn: nn.Module,
        hidden_states: torch.Tensor,
        past_key_values: Optional[Any],
        cache_position: Optional[torch.Tensor],
        kwargs: dict,
    ) -> bool:
        """Prefill-vs-decode detection across transformers versions.

        Mirrors ``KVCompressor._is_decoding_step``: a decode step is one whose
        already-cached length exceeds the current query length.
        """
        q_len = hidden_states.shape[1]
        if cache_position is not None:
            return int(cache_position[-1]) > q_len
        if past_key_values is not None:
            try:
                return past_key_values.get_seq_length(attn.layer_idx) > q_len
            except Exception:
                pass
        return q_len <= 1


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
