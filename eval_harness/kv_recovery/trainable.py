"""Trainable-subset selection, freezing and parameter accounting (spec §9).

Parameters are always addressed by their FULL name (never by suffix): the Pixtral /
Qwen vision towers also own ``q_proj``-named modules, and the tied ``lm_head``
must never be selected. Ported and generalised from
``kv_compression_adaptation/src/training/params.py``.

Strategies (``TrainableCfg.strategy``):

* ``last_n_blocks``          every parameter of the last ``n`` decoder blocks (attention + MLP + norms).
* ``attention_projections``  ``modules`` (q_proj/k_proj/v_proj/o_proj) of the softmax-attention layers
                             selected by ``layers`` (``all`` | ``last_n:<k>`` | explicit indices). On
                             hybrid models only full-attention layers qualify (linear layers have no
                             ``self_attn``).
* ``mlp`` / ``norms``        the MLP / every ``*norm*`` parameter of the selected layers.
* ``full``                   the whole text LM (embeddings only with ``include_embeddings``); vision
                             tower, projector, MTP head and ``lm_head`` stay frozen. Ablation only.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
from torch import nn

from .config import TrainableCfg
from .model_spec import (
    NEVER_TRAINABLE_PREFIXES,
    ModelSpec,
    attention_attr_name,
    decoder_layers,
    language_model,
)


def select_layers(selector: Any, candidates: Sequence[int]) -> List[int]:
    """``'all'`` | ``'last_n:<k>'`` | list of indices -> sorted subset of ``candidates``."""
    cands = sorted(int(c) for c in candidates)
    if selector == "all" or selector is None:
        return cands
    if isinstance(selector, str) and selector.startswith("last_n:"):
        k = int(selector.split(":", 1)[1])
        return cands[-k:] if k < len(cands) else cands
    if isinstance(selector, (list, tuple)):
        wanted = sorted(set(int(i) for i in selector))
        bad = [i for i in wanted if i not in cands]
        if bad:
            raise ValueError(f"layers {bad} are not eligible (eligible: {cands})")
        return wanted
    raise ValueError(f"bad layer selector {selector!r}")


def _params_under(names: Iterable[str], prefix: str) -> List[str]:
    return [n for n in names if n.startswith(prefix)]


def select_trainable(model: nn.Module, spec: ModelSpec, tcfg: TrainableCfg) -> List[str]:
    """Full parameter names selected by ``tcfg``; raises if the selection is empty."""
    all_names = [n for n, _ in model.named_parameters()]
    layers = decoder_layers(model)
    n_layers = len(layers)
    strategy = tcfg.strategy
    out: List[str] = []

    if strategy == "last_n_blocks":
        n = int(tcfg.n)
        if n <= 0 or n > n_layers:
            raise ValueError(f"trainable.n must be in [1, {n_layers}], got {n}")
        for i in range(n_layers - n, n_layers):
            out += _params_under(all_names, spec.layer_prefix(i))

    elif strategy == "attention_projections":
        wanted_modules = [str(m) for m in tcfg.modules]
        if not wanted_modules:
            raise ValueError("trainable.modules must not be empty for attention_projections")
        for i in select_layers(tcfg.layers, spec.full_attention_layers):
            attr = attention_attr_name(layers[i])
            if attr is None:
                continue
            for m in wanted_modules:
                found = _params_under(all_names, f"{spec.layer_prefix(i)}{attr}.{m}.")
                if not found:
                    raise ValueError(f"layer {i}: no parameter under {attr}.{m} "
                                     f"(available: {sorted(set(n.split('.')[-2] for n in _params_under(all_names, spec.layer_prefix(i) + attr + '.')))})")
                out += found

    elif strategy in ("mlp", "norms"):
        for i in select_layers(tcfg.layers, range(n_layers)):
            block = _params_under(all_names, spec.layer_prefix(i))
            if strategy == "mlp":
                out += [n for n in block if n[len(spec.layer_prefix(i)):].startswith("mlp.")]
            else:
                out += [n for n in block if "norm" in n[len(spec.layer_prefix(i)):].lower()]

    elif strategy == "full":
        lm_names = _params_under(all_names, spec.lm_prefix) if spec.lm_prefix else list(all_names)
        for n in lm_names:
            rel = n[len(spec.lm_prefix):]
            if any(rel.startswith(p) for p in NEVER_TRAINABLE_PREFIXES):
                continue
            if rel.startswith("embed_tokens") and not tcfg.include_embeddings:
                continue
            out.append(n)
    else:
        raise ValueError(f"unknown trainable.strategy {strategy!r}")

    out = [n for n in out if not any(seg in n.split(".") for seg in NEVER_TRAINABLE_PREFIXES)]
    out = [n for n in out if not n.startswith("lm_head")]
    out = sorted(dict.fromkeys(out))
    if not out:
        raise ValueError(f"trainable selection {tcfg} matched no parameters")
    return out


def layer_index_of(name: str, spec: ModelSpec) -> Optional[int]:
    marker = f"{spec.lm_prefix}layers."
    if not name.startswith(marker):
        return None
    rest = name[len(marker):]
    head = rest.split(".", 1)[0]
    return int(head) if head.isdigit() else None


def first_trainable_layer(names: Iterable[str], spec: ModelSpec) -> Optional[int]:
    idx = [layer_index_of(n, spec) for n in names]
    idx = [i for i in idx if i is not None]
    return min(idx) if idx else None


def freeze_all_but(model: nn.Module, names: Iterable[str]) -> Dict[str, bool]:
    """Set ``requires_grad`` on EVERY parameter; return the expected flag map."""
    wanted = set(names)
    params = dict(model.named_parameters())
    missing = wanted - set(params)
    if missing:
        raise KeyError(f"not model parameters: {sorted(missing)[:10]}")
    expected: Dict[str, bool] = {}
    for name, p in params.items():
        flag = name in wanted
        p.requires_grad_(flag)
        expected[name] = flag
    return expected


def assert_trainable(model: nn.Module, expected: Dict[str, bool], spec: Optional[ModelSpec] = None) -> None:
    seen = set()
    for name, p in model.named_parameters():
        if p.requires_grad != expected[name]:
            raise AssertionError(f"{name}: requires_grad={p.requires_grad}, expected {expected[name]}")
        seen.add(name)
    if seen != set(expected):
        raise AssertionError("parameter set changed since freeze_all_but")
    # Tied embeddings: lm_head must stay frozen AND share storage with embed_tokens.
    head = getattr(model, "lm_head", None)
    if head is not None and hasattr(head, "weight"):
        if head.weight.requires_grad:
            raise AssertionError("lm_head.weight must stay frozen")
        emb = getattr(language_model(model), "embed_tokens", None)
        tied = bool(getattr(getattr(model, "config", None), "tie_word_embeddings", False))
        if tied and emb is not None and emb.weight.data_ptr() != head.weight.data_ptr():
            raise AssertionError("config says tie_word_embeddings but lm_head and embed_tokens do not share storage")


def trainable_parameters(model: nn.Module, names: Iterable[str]) -> Dict[str, nn.Parameter]:
    params = dict(model.named_parameters())
    return {n: params[n] for n in names}


def parameter_summary(model: nn.Module, spec: ModelSpec) -> Dict[str, Any]:
    """Counts the trainer prints and records (spec §9): total / text-LM / trainable / %."""
    total = sum(p.numel() for p in model.parameters())
    text_lm = sum(p.numel() for p in language_model(model).parameters())
    trainable = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
    n_trainable = sum(c for _, c in trainable)
    return {
        "total_parameters": int(total),
        "text_lm_parameters": int(text_lm),
        "trainable_parameters": int(n_trainable),
        "percent_trainable_of_text_lm": 100.0 * n_trainable / max(text_lm, 1),
        "percent_trainable_of_total": 100.0 * n_trainable / max(total, 1),
        "n_trainable_tensors": len(trainable),
        "trainable_names": [n for n, _ in trainable],
        "first_trainable_layer": first_trainable_layer([n for n, _ in trainable], spec),
        "family": spec.family,
        "n_layers": spec.n_layers,
        "full_attention_layers": list(spec.full_attention_layers),
    }


def format_parameter_summary(summary: Dict[str, Any]) -> str:
    lines = [
        f"Total parameters: {summary['total_parameters']:,} (text LM: {summary['text_lm_parameters']:,})",
        f"Trainable parameters: {summary['trainable_parameters']:,} in {summary['n_trainable_tensors']} tensors",
        f"Percent trainable: {summary['percent_trainable_of_text_lm']:.4f}% of the text LM "
        f"({summary['percent_trainable_of_total']:.4f}% of the full checkpoint)",
        f"First trainable decoder layer: {summary['first_trainable_layer']}",
        "Trainable parameter names:",
    ]
    lines += [f"  {n}" for n in summary["trainable_names"]]
    return "\n".join(lines)


def assert_no_stray_grads(model: nn.Module, trainable: Iterable[str]) -> None:
    """Spec §21.8: only the intended parameters may carry gradients."""
    wanted = set(trainable)
    stray = [n for n, p in model.named_parameters() if p.grad is not None and n not in wanted]
    if stray:
        raise AssertionError(f"gradients on frozen parameters: {stray[:10]}")
    missing = [n for n, p in model.named_parameters() if n in wanted and p.grad is None]
    if missing:
        raise AssertionError(f"trainable parameters received no gradient: {missing[:10]}")


@torch.no_grad()
def snapshot_parameters(model: nn.Module, names: Optional[Iterable[str]] = None, *, device: str = "cpu") -> Dict[str, torch.Tensor]:
    params = dict(model.named_parameters())
    names = list(names) if names is not None else list(params)
    return {n: params[n].detach().to(device).clone() for n in names}


@torch.no_grad()
def changed_parameters(model: nn.Module, snapshot: Dict[str, torch.Tensor]) -> List[str]:
    params = dict(model.named_parameters())
    return [n for n, t in snapshot.items() if not torch.equal(params[n].detach().to(t.device), t)]
