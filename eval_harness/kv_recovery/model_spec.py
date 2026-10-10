"""Per-model-family isolation for the KV-recovery code.

Everything model-specific (where the decoder layers live, which layers carry a
softmax-attention KV cache, how parameters are named) is resolved HERE, reusing
the detection helpers the KV compressors already rely on
(``kv_compression.base._get_language_model`` / ``_is_non_full_attention_layer`` /
``_resolve_attention_module`` and ``cache_adapter._is_hybrid_model``), so the
trainer hooks exactly the layers the compressor hooks.

Verified layouts (transformers 5.10.2):

* ``Mistral3ForConditionalGeneration`` (Ministral-3): text decoder at
  ``model.model.language_model`` -> parameters ``model.language_model.layers.{i}.…``;
  26 full-attention layers (``self_attn.{q,k,v,o}_proj``); tied embeddings; a
  Pixtral vision tower that ALSO owns ``q_proj``-named modules (hence full-path names).
* ``Qwen3_5ForConditionalGeneration`` (Qwen3.5): same decoder path; ``layer_types``
  = (linear, linear, linear, full) x 8 -> only layers 3, 7, ..., 31 carry a K/V cache
  (``self_attn.{q,k,v,o}_proj`` + ``q_norm``/``k_norm``; ``q_proj`` is 2x wide:
  query + gate); linear layers hold ``linear_attn.*`` (no ``self_attn``).
* Plain ``*ForCausalLM`` (tests): decoder at ``model.model`` -> ``model.layers.{i}.…``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from torch import nn

from eval_harness.kv_compression.base import (
    _get_language_model,
    _is_non_full_attention_layer,
    _resolve_attention_module,
)
from eval_harness.kv_compression.cache_adapter import _is_hybrid_model

PROJECTIONS: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
# Submodules that are never trainable unless ``trainable.strategy == full`` with
# ``include_embeddings`` (and even then the vision tower / projector / MTP head stay frozen).
NEVER_TRAINABLE_PREFIXES: Tuple[str, ...] = ("vision_tower", "visual", "multi_modal_projector", "mtp")


@dataclass(frozen=True)
class ModelSpec:
    family: str                               # text_config.model_type (e.g. ministral3, qwen3_5_text, llama)
    lm_prefix: str                            # parameter-name prefix of the text decoder, WITH trailing dot ("" = model itself)
    n_layers: int
    hidden_size: int
    full_attention_layers: Tuple[int, ...]    # layers the KV compressor hooks (carry a softmax K/V cache)
    is_hybrid: bool                           # some layers carry no K/V cache (linear attention / mamba)
    has_final_norm: bool
    tied_embeddings: bool

    def layer_prefix(self, idx: int) -> str:
        return f"{self.lm_prefix}layers.{idx}."

    @property
    def first_full_attention_layer(self) -> int:
        return self.full_attention_layers[0] if self.full_attention_layers else 0


def language_model(model: nn.Module) -> nn.Module:
    """The decoder stack (owns ``.layers``), multimodal wrappers unwrapped."""
    return _get_language_model(model)


def decoder_layers(model: nn.Module) -> nn.ModuleList:
    return language_model(model).layers


def final_norm(model: nn.Module) -> Optional[nn.Module]:
    return getattr(language_model(model), "norm", None)


def lm_prefix_of(model: nn.Module) -> str:
    lm = language_model(model)
    if lm is model:
        return ""
    for name, module in model.named_modules():
        if module is lm:
            return name + "."
    raise RuntimeError("could not locate the text decoder inside the model's module tree")


def text_config(model: nn.Module):
    cfg = getattr(model, "config", None)
    if cfg is None:
        return None
    try:
        return cfg.get_text_config(decoder=True)
    except Exception:
        return cfg


def attention_module_of(layer: nn.Module) -> Optional[nn.Module]:
    """The hookable softmax-attention module of a decoder layer, or ``None``."""
    if _is_non_full_attention_layer(layer):
        return None
    return _resolve_attention_module(layer)


def attention_attr_name(layer: nn.Module) -> Optional[str]:
    """``"self_attn"`` (standard) or ``"mixer"`` (NemotronH-style) — the attribute
    under which the softmax-attention module lives; ``None`` for non-full layers."""
    attn = attention_module_of(layer)
    if attn is None:
        return None
    for name, child in layer.named_children():
        if child is attn:
            return name
    return None


def inspect_model(model: nn.Module) -> ModelSpec:
    layers = decoder_layers(model)
    tcfg = text_config(model)
    hidden = int(getattr(tcfg, "hidden_size", 0) or 0)
    if hidden == 0:
        # Fall back to the embedding width.
        emb = getattr(language_model(model), "embed_tokens", None)
        hidden = int(emb.weight.shape[1]) if emb is not None else 0
    full = tuple(i for i, layer in enumerate(layers) if attention_module_of(layer) is not None)
    family = str(getattr(tcfg, "model_type", None) or getattr(getattr(model, "config", None), "model_type", "unknown"))
    tied = bool(getattr(getattr(model, "config", None), "tie_word_embeddings", False))
    if not tied and tcfg is not None:
        tied = bool(getattr(tcfg, "tie_word_embeddings", False))
    return ModelSpec(
        family=family,
        lm_prefix=lm_prefix_of(model),
        n_layers=len(layers),
        hidden_size=hidden,
        full_attention_layers=full,
        is_hybrid=bool(_is_hybrid_model(model)),
        has_final_norm=final_norm(model) is not None,
        tied_embeddings=tied,
    )


def resolve_base_revision(model: nn.Module) -> Optional[str]:
    """The hub snapshot commit the weights came from, when known."""
    cfg = getattr(model, "config", None)
    rev = getattr(cfg, "_commit_hash", None) if cfg is not None else None
    if rev:
        return str(rev)
    name = getattr(cfg, "_name_or_path", None) if cfg is not None else None
    if not name:
        return None
    try:  # a local snapshot path: .../snapshots/<commit>/
        from pathlib import Path

        parts = Path(str(name)).parts
        if "snapshots" in parts:
            return parts[parts.index("snapshots") + 1]
    except Exception:
        pass
    return None
