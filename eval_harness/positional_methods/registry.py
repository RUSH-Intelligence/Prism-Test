"""Auto-discovery registry for Door-1 positional methods.

Mirrors ``eval_harness/benchmarks/registry.py``: positional-method modules
decorate their class with ``@register_positional_method("name")`` and are
auto-discovered on first lookup.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any, Dict, List, Optional, Type

from .base import PositionalMethod

_POSITIONAL_METHOD_REGISTRY: Dict[str, Type[PositionalMethod]] = {}
_METHODS_LOADED = False


def register_positional_method(
    name: str,
    aliases: Optional[List[str]] = None,
):
    """Decorator registering a positional-method class under ``name``."""

    def decorator(cls: Type[PositionalMethod]) -> Type[PositionalMethod]:
        key = name.strip().lower()
        _POSITIONAL_METHOD_REGISTRY[key] = cls
        for alias in aliases or []:
            _POSITIONAL_METHOD_REGISTRY[alias.strip().lower()] = cls
        return cls

    return decorator


def ensure_methods_loaded() -> None:
    """Import every method module in this package (auto-discovery)."""
    global _METHODS_LOADED
    if _METHODS_LOADED:
        return

    package_name = __package__ or "eval_harness.positional_methods"
    package = importlib.import_module(package_name)
    for module_info in pkgutil.iter_modules(package.__path__):
        if module_info.ispkg or module_info.name in {"__init__", "base", "registry"}:
            continue
        importlib.import_module(f"{package_name}.{module_info.name}")

    _METHODS_LOADED = True


def get_positional_method(name: str, **kwargs: Any) -> PositionalMethod:
    """Instantiate a registered positional method by name.

    ``none``/``default``/``standard`` resolve to the identity
    :class:`PositionalMethod` (byte-equivalent to installing nothing).
    """
    ensure_methods_loaded()
    key = name.strip().lower()
    if key in {"none", "default", "standard", "native"}:
        return PositionalMethod()
    cls = _POSITIONAL_METHOD_REGISTRY.get(key)
    if cls is None:
        available = ", ".join(available_positional_methods())
        raise ValueError(f"Unknown positional method '{name}'. Available: {available}")
    return cls(**kwargs)


def available_positional_methods() -> List[str]:
    ensure_methods_loaded()
    return sorted(_POSITIONAL_METHOD_REGISTRY.keys())
