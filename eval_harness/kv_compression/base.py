"""Door 3 — KV compression.

A *KV compressor* decides **what stays in the cache**.  It runs as a
post-attention hook that rewrites the layer's cached keys/values, and it is
gated by a combinable :class:`CompressionSchedule` instead of the legacy
class-per-timing hierarchy (``BaseSketch`` / ``DecodingSketch`` /
``PrefillDecodingSketch``):

* ``streaming``     — fire after **each prefill chunk** (memory-bounded; lets
                      context exceed GPU memory).  Needs the chunked-prefill
                      loop; with a single-pass prefill it coincides with
                      ``post_prefill``.
* ``post_prefill``  — fire **once** after the full prefill pass.
* ``decode``        — fire **during decode**, every ``decode_interval`` steps.

A compressor may declare several schedules at once (e.g. ``{streaming,
decode}``).  The :attr:`operation` field keeps the contract operation-agnostic:
eviction, quantization and merging all return replacement ``(keys, values)``.

This is the renamed successor of ``eval_harness.sketch`` /
``BaseSketch.compress`` — the ``compress(...) -> (keys, values)`` contract is
preserved so the ~31 ported kvpress compressors migrate mechanically.

Pipeline position (outer → inner)::

    positional_method(model)
      → attention_method(model)
        → kv_compressor(model)      # door 3  ← THIS
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, FrozenSet, Generator, Iterable, Tuple

import torch
from torch import nn

logger = logging.getLogger(__name__)


class CompressionSchedule(str, Enum):
    """When a KV compressor fires.  Combinable (a compressor may use several)."""

    STREAMING = "streaming"
    POST_PREFILL = "post_prefill"
    DECODE = "decode"

    @classmethod
    def coerce_set(
        cls,
        value: "CompressionSchedule | str | Iterable[Any]",
    ) -> FrozenSet["CompressionSchedule"]:
        """Parse one schedule, or a list of them, into a frozenset.

        Accepts an enum member, a string, or any iterable of those — so YAML
        ``compression_schedule: post_prefill`` and
        ``compression_schedule: [streaming, decode]`` both work.
        """
        if isinstance(value, (CompressionSchedule, str)):
            value = [value]
        out = set()
        for item in value:
            if isinstance(item, CompressionSchedule):
                out.add(item)
                continue
            key = str(item).strip().lower()
            try:
                out.add(cls(key))
            except ValueError as exc:
                allowed = ", ".join(s.value for s in cls)
                raise ValueError(
                    f"Unknown compression schedule {item!r}. Allowed: {allowed}"
                ) from exc
        if not out:
            raise ValueError("compression_schedule must name at least one schedule")
        return frozenset(out)


class CompressionOperation(str, Enum):
    """What a compressor does to the cache — kept open so the contract fits all."""

    EVICT = "evict"
    QUANTIZE = "quantize"
    MERGE = "merge"


@dataclass
class KVCompressor:
    """Base class for Door-3 KV compressors.

    Subclasses override :meth:`compress`, returning the replacement
    ``(keys, values)`` for a layer.  The framework installs the post-attention
    hook on full-softmax layers, writes the result back into the cache
    (handling ``QuantizedCache``), and only fires the hook on the phases named
    in :attr:`schedule`.
    """

    schedule: FrozenSet[CompressionSchedule] = field(
        default_factory=lambda: frozenset({CompressionSchedule.POST_PREFILL})
    )
    operation: CompressionOperation = CompressionOperation.EVICT
    decode_interval: int = 1

    def __post_init__(self) -> None:
        self.schedule = CompressionSchedule.coerce_set(self.schedule)
        if not isinstance(self.operation, CompressionOperation):
            self.operation = CompressionOperation(str(self.operation).strip().lower())

    # ------------------------------------------------------------------
    # Schedule predicates
    # ------------------------------------------------------------------

    @property
    def fires_on_prefill(self) -> bool:
        return bool(
            self.schedule
            & {CompressionSchedule.STREAMING, CompressionSchedule.POST_PREFILL}
        )

    @property
    def fires_on_decode(self) -> bool:
        return CompressionSchedule.DECODE in self.schedule

    # ------------------------------------------------------------------
    # The single override point
    # ------------------------------------------------------------------

    def post_init_from_model(self, model: nn.Module) -> None:
        """Optional per-model setup (head dims, budgets) on install."""

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return replacement ``(keys, values)`` for ``module``'s layer.

        ``keys``/``values`` are the current cache contents ``[B, H_kv, S, D]``
        (RoPE-rotated on the research path).  ``attentions`` is the layer's
        attention-probability output when available (``None`` under
        flash-attention — attention-weight scorers only work on eager/SDPA).
        """
        raise NotImplementedError("compress() must be implemented in a subclass")

    # ------------------------------------------------------------------
    # Hook install / dispatch
    # ------------------------------------------------------------------

    @staticmethod
    def _is_decoding_step(module: nn.Module, kwargs: dict, q_len: int) -> bool:
        from eval_harness.sketch.sketches.base_sketch import BaseSketch

        return BaseSketch._is_decoding_step(module, kwargs, q_len)

    def forward_hook(
        self,
        module: nn.Module,
        inputs: list,
        kwargs: dict,
        output: list,
    ):
        from transformers import QuantizedCache

        from eval_harness.sketch.utils import extract_keys_and_values

        hidden_states = kwargs["hidden_states"]
        cache = kwargs["past_key_values"]
        cache_layer = cache.layers[module.layer_idx]
        q_len = hidden_states.shape[1]

        is_decode = self._is_decoding_step(module, kwargs, q_len)
        if is_decode and not self.fires_on_decode:
            return output
        if not is_decode and not self.fires_on_prefill:
            return output

        keys, values = extract_keys_and_values(cache, module.layer_idx)
        keys, values = self.compress(
            module, hidden_states, keys, values, output[1], kwargs,
        )

        if isinstance(cache, QuantizedCache):
            cache_layer._quantized_keys = cache_layer._quantize(
                keys, axis=cache_layer.axis_key,
            )
            cache_layer._quantized_values = cache_layer._quantize(
                values, axis=cache_layer.axis_value,
            )
            cache_layer.keys = torch.zeros(0, dtype=keys.dtype, device=keys.device)
            cache_layer.values = torch.zeros(0, dtype=keys.dtype, device=keys.device)
            cache_layer.cumulative_length = keys.shape[2]
        else:
            cache_layer.keys = keys
            cache_layer.values = values

        return output

    @contextmanager
    def __call__(self, model: nn.Module) -> Generator:
        from eval_harness.sketch.sketches.base_sketch import (
            _is_non_full_attention_layer,
        )

        from eval_harness.attention_methods.base import _is_gemma3

        is_gemma3 = _is_gemma3(model)
        language_model = (
            model.model.language_model
            if hasattr(model.model, "language_model")
            else model.model
        )

        self.post_init_from_model(model)
        hooks = []
        try:
            for layer in language_model.layers:
                if is_gemma3 and getattr(layer.self_attn, "is_sliding", False):
                    continue
                if _is_non_full_attention_layer(layer):
                    continue
                layer.self_attn.rotary_emb = language_model.rotary_emb
                hooks.append(
                    layer.self_attn.register_forward_hook(
                        self.forward_hook, with_kwargs=True,
                    )
                )
            yield
        finally:
            for hook in hooks:
                hook.remove()
