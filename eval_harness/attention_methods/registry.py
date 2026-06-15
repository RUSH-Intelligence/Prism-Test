"""Auto-discovery registry for Door-2 attention methods.

Mirrors ``eval_harness/benchmarks/registry.py``: attention-method modules
decorate their class with ``@register_attention_method("name")`` and are
auto-discovered on first lookup, so adding a method never edits shared files.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any, Dict, List, Optional, Type

from .base import AttentionMethod

_ATTENTION_METHOD_REGISTRY: Dict[str, Type[AttentionMethod]] = {}
_METHODS_LOADED = False


def register_attention_method(
    name: str,
    aliases: Optional[List[str]] = None,
):
    """Decorator registering an attention-method class under ``name``."""

    def decorator(cls: Type[AttentionMethod]) -> Type[AttentionMethod]:
        key = name.strip().lower()
        _ATTENTION_METHOD_REGISTRY[key] = cls
        for alias in aliases or []:
            _ATTENTION_METHOD_REGISTRY[alias.strip().lower()] = cls
        return cls

    return decorator


def ensure_methods_loaded() -> None:
    """Import every method module in this package (auto-discovery)."""
    global _METHODS_LOADED
    if _METHODS_LOADED:
        return

    package_name = __package__ or "eval_harness.attention_methods"
    package = importlib.import_module(package_name)
    for module_info in pkgutil.iter_modules(package.__path__):
        if module_info.ispkg or module_info.name in {"__init__", "base", "registry"}:
            continue
        importlib.import_module(f"{package_name}.{module_info.name}")

    _METHODS_LOADED = True


def get_attention_method(name: str, **kwargs: Any) -> AttentionMethod:
    """Instantiate a registered attention method by name.

    ``none``/``default``/``standard`` resolve to the base no-op
    :class:`AttentionMethod` (which never activates because its
    ``attention_forward`` is never reached — callers should treat the base type
    as "no method installed").
    """
    ensure_methods_loaded()
    key = name.strip().lower()
    if key in {"none", "default", "standard"}:
        return AttentionMethod()
    cls = _ATTENTION_METHOD_REGISTRY.get(key)
    if cls is None:
        available = ", ".join(available_attention_methods())
        raise ValueError(f"Unknown attention method '{name}'. Available: {available}")
    return cls(**kwargs)


def available_attention_methods() -> List[str]:
    ensure_methods_loaded()
    return sorted(_ATTENTION_METHOD_REGISTRY.keys())
