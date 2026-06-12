"""Auto-discovery registry for prefill attention methods.

Mirrors the benchmark registry pattern in ``eval_harness/benchmarks/registry.py``.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any, Dict, List, Optional, Type

from .base import PrefillMethod

_PREFILL_METHOD_REGISTRY: Dict[str, Type[PrefillMethod]] = {}
_METHODS_LOADED = False


def register_prefill_method(
    name: str,
    aliases: Optional[List[str]] = None,
):
    """Decorator to register a prefill method class.

    Usage::

        @register_prefill_method("ntk_aware", aliases=["ntk"])
        @dataclass
        class NTKAwareMethod(PrefillMethod):
            ...
    """

    def decorator(cls: Type[PrefillMethod]) -> Type[PrefillMethod]:
        key = name.strip().lower()
        _PREFILL_METHOD_REGISTRY[key] = cls
        for alias in aliases or []:
            _PREFILL_METHOD_REGISTRY[alias.strip().lower()] = cls
        return cls

    return decorator


def ensure_methods_loaded() -> None:
    """Auto-discover all method modules in this package."""
    global _METHODS_LOADED
    if _METHODS_LOADED:
        return

    package_name = __package__ or "eval_harness.prefill_methods"
    package = importlib.import_module(package_name)
    for module_info in pkgutil.iter_modules(package.__path__):
        if module_info.ispkg or module_info.name in {
            "__init__",
            "base",
            "registry",
        }:
            continue
        importlib.import_module(f"{package_name}.{module_info.name}")

    _METHODS_LOADED = True


def get_prefill_method(name: str, **kwargs: Any) -> PrefillMethod:
    """Instantiate a registered prefill method by name."""
    ensure_methods_loaded()
    key = name.strip().lower()
    if key in {"none", "default", "standard"}:
        return PrefillMethod()
    cls = _PREFILL_METHOD_REGISTRY.get(key)
    if cls is None:
        available = ", ".join(available_prefill_methods())
        raise ValueError(
            f"Unknown prefill method '{name}'. Available: {available}"
        )
    return cls(**kwargs)


def available_prefill_methods() -> List[str]:
    """Return sorted list of all registered method names."""
    ensure_methods_loaded()
    return sorted(_PREFILL_METHOD_REGISTRY.keys())
