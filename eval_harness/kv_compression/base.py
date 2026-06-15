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
preserved so the ported kvpress compressors migrate mechanically.  Score-based
compressors extend :class:`ScorerKVCompressor` (renamed ``ScorerSketch``).

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
from typing import FrozenSet, Generator, Iterable, Tuple

import torch
from torch import nn
from transformers import PreTrainedModel, QuantizedCache

from eval_harness.kv_compression.utils import extract_keys_and_values

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Model-family helpers (own copies so Door 3 does not depend on `sketch`)
# ----------------------------------------------------------------------

def _try_import(name: str):
    try:
        import transformers

        return getattr(transformers, name, None)
    except Exception:
        return None


_MODEL_NAMES = (
    "LlamaForCausalLM",
    "MistralForCausalLM",
    "Mistral3ForConditionalGeneration",
    "Phi3ForCausalLM",
    "Qwen2ForCausalLM",
    "Qwen3ForCausalLM",
    "Qwen3_5ForCausalLM",
    "Gemma3ForCausalLM",
    "Gemma3ForConditionalGeneration",
)
SUPPORTED_MODELS = tuple(m for m in (_try_import(n) for n in _MODEL_NAMES) if m is not None)

_Gemma3Causal = _try_import("Gemma3ForCausalLM")
_Gemma3Cond = _try_import("Gemma3ForConditionalGeneration")
_Qwen35 = _try_import("Qwen3_5ForCausalLM")


def _is_gemma3(model) -> bool:
    return (_Gemma3Cond is not None and isinstance(model, _Gemma3Cond)) or (
        _Gemma3Causal is not None and isinstance(model, _Gemma3Causal)
    )


def _is_non_full_attention_layer(layer: nn.Module) -> bool:
    """Best-effort detection of non-full attention layers.

    For mixed-attention families (Gemma3, Qwen3.5) we only hook full-softmax
    layers; this flags sliding/linear (or any non-full typed) layer to skip.
    """
    attn = getattr(layer, "self_attn", None)
    if attn is None:
        return False

    for attr in ("layer_type", "attention_type"):
        val = getattr(layer, attr, None)
        if isinstance(val, str):
            lowered = val.lower()
            if "full" in lowered:
                return False
            if any(token in lowered for token in ("sliding", "linear")):
                return True
            return True

    is_sliding = getattr(attn, "is_sliding", None)
    if is_sliding is not None:
        return bool(is_sliding)
    is_linear = getattr(attn, "is_linear", None)
    if is_linear is not None:
        return bool(is_linear)

    cfg = getattr(attn, "config", None)
    for attr in ("layer_type", "attention_type"):
        val = getattr(cfg, attr, None) if cfg is not None else None
        if isinstance(val, str):
            lowered = val.lower()
            if "full" in lowered:
                return False
            if any(token in lowered for token in ("sliding", "linear")):
                return True
            return True

    sw = getattr(cfg, "sliding_window", None) if cfg is not None else None
    if isinstance(sw, int):
        return sw > 0

    return False


# ----------------------------------------------------------------------
# Schedule / operation model
# ----------------------------------------------------------------------

class CompressionSchedule(str, Enum):
    """When a KV compressor fires.  Combinable (a compressor may use several)."""

    STREAMING = "streaming"
    POST_PREFILL = "post_prefill"
    DECODE = "decode"

    @classmethod
    def coerce_set(
        cls,
        value: "CompressionSchedule | str | Iterable",
    ) -> FrozenSet["CompressionSchedule"]:
        """Parse one schedule, or a list of them, into a frozenset."""
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

    # kw_only so subclasses (the ported compressors) can still declare REQUIRED
    # positional fields like ``press: ScorerKVCompressor`` without hitting the
    # "non-default argument follows default argument" dataclass error — the old
    # ``KVCompressor`` had no fields, so these additions must not reorder theirs.
    schedule: FrozenSet[CompressionSchedule] = field(
        default_factory=lambda: frozenset({CompressionSchedule.POST_PREFILL}),
        kw_only=True,
    )
    # Reserved: declared (and `operation` coerced in __post_init__) for
    # forward-compatibility, but NOT yet consumed by any compressor or the
    # pipeline.  `operation` is intended to let a compressor declare a non-evict
    # op (e.g. quantize / merge); `decode_interval` is intended to throttle
    # decode-time compression to every Nth step.  Setting either today has no
    # effect — wire them up when a method actually needs them.
    operation: CompressionOperation = field(
        default=CompressionOperation.EVICT, kw_only=True,
    )
    decode_interval: int = field(default=1, kw_only=True)

    def __post_init__(self) -> None:
        self.schedule = CompressionSchedule.coerce_set(self.schedule)
        if not isinstance(self.operation, CompressionOperation):
            self.operation = CompressionOperation(str(self.operation).strip().lower())
        # Prefill/decode phase declared explicitly by the pipeline (see
        # set_phase).  ``None`` falls back to the cache_position heuristic.
        self._explicit_phase = None

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
    # Lifecycle / detection
    # ------------------------------------------------------------------

    def post_init_from_model(self, model: PreTrainedModel) -> None:
        """Optional per-model setup (head dims, budgets) on install."""

    def set_phase(self, phase) -> None:
        """Declare the phase of the upcoming forward(s): ``"prefill"`` or
        ``"decode"`` (or ``None`` to restore the heuristic).

        The pipeline calls this because the cache_position heuristic
        (:meth:`_is_decoding_step`) misclassifies every chunked-prefill chunk
        *after the first* as decode — those chunks start at ``cache_position``
        well past ``q_len`` — which would stop a ``streaming`` compressor from
        firing after the first chunk.  An explicit phase removes the guesswork.
        """
        self._explicit_phase = phase

    def _resolve_is_decode(self, module: nn.Module, kwargs: dict, q_len: int) -> bool:
        """Decode-vs-prefill: the pipeline's explicit phase wins; else heuristic."""
        phase = getattr(self, "_explicit_phase", None)
        if phase is not None:
            return phase == "decode"
        return self._is_decoding_step(module, kwargs, q_len)

    @staticmethod
    def _is_decoding_step(module: nn.Module, kwargs: dict, q_len: int) -> bool:
        """Detect decoding vs prefill across transformers versions.

        Heuristic fallback used when the pipeline has not set an explicit phase
        (see :meth:`set_phase`).  Reliable for a single-pass prefill and for
        decode, but NOT for non-first chunked-prefill chunks.
        """
        cache_position = kwargs.get("cache_position")
        if cache_position is not None:
            return cache_position[-1] > q_len

        cache = kwargs.get("past_key_values")
        if cache is not None:
            try:
                return cache.get_seq_length(module.layer_idx) > q_len
            except Exception:
                pass

        return q_len <= 1

    # ------------------------------------------------------------------
    # The single override point
    # ------------------------------------------------------------------

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

    def forward_hook(
        self,
        module: nn.Module,
        inputs: list,
        kwargs: dict,
        output: list,
    ):
        hidden_states = kwargs["hidden_states"]
        cache = kwargs["past_key_values"]
        cache_layer = cache.layers[module.layer_idx]
        q_len = hidden_states.shape[1]

        is_decode = self._resolve_is_decode(module, kwargs, q_len)
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
    def __call__(self, model: PreTrainedModel) -> Generator:
        if not isinstance(model, SUPPORTED_MODELS):
            logger.warning(
                "Model %s not tested, supported models: %s", type(model), SUPPORTED_MODELS,
            )

        is_gemma3_family = _is_gemma3(model)
        if is_gemma3_family or (_Qwen35 is not None and isinstance(model, _Qwen35)):
            logger.warning(
                "Compression is only applied to full-softmax attention layers "
                "for this model family",
            )

        self.post_init_from_model(model)
        hooks = []
        try:
            language_model = (
                model.model.language_model
                if hasattr(model.model, "language_model")
                else model.model
            )
            for layer in language_model.layers:
                if is_gemma3_family and getattr(layer.self_attn, "is_sliding", False):
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


@dataclass
class ScorerKVCompressor(KVCompressor):
    """Score-based eviction compressor (renamed ``ScorerKVCompressor``).

    Subclasses implement :meth:`score`; :meth:`compress` keeps the top
    ``(1 - compression_ratio)`` fraction of tokens by score.
    """

    compression_ratio: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        assert 0 <= self.compression_ratio < 1, "Compression ratio must be between 0 and 1"

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.compression_ratio == 0:
            return keys, values

        scores = self.score(module, hidden_states, keys, values, attentions, kwargs)

        k_len = keys.shape[2]
        n_kept = int(k_len * (1 - self.compression_ratio))
        indices = scores.topk(n_kept, dim=-1).indices
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)

        keys = keys.gather(2, indices).contiguous()
        values = values.gather(2, indices).contiguous()

        return keys, values
