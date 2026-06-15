"""Auto-discovery registry for Door-3 KV compressors.

Successor to ``eval_harness/sketch/sketches/registry.py``: compressor modules
decorate their class with ``@register_kv_compressor("name")`` and are
auto-discovered on first lookup.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any, Dict, List, Optional, Type

from .base import KVCompressor

_KV_COMPRESSOR_REGISTRY: Dict[str, Type[KVCompressor]] = {}
_COMPRESSORS_LOADED = False


def register_kv_compressor(
    name: str,
    aliases: Optional[List[str]] = None,
):
    """Decorator registering a KV-compressor class under ``name``."""

    def decorator(cls: Type[KVCompressor]) -> Type[KVCompressor]:
        key = name.strip().lower()
        _KV_COMPRESSOR_REGISTRY[key] = cls
        for alias in aliases or []:
            _KV_COMPRESSOR_REGISTRY[alias.strip().lower()] = cls
        return cls

    return decorator


def ensure_compressors_loaded() -> None:
    """Import every compressor module under ``compressors/`` (auto-discovery)."""
    global _COMPRESSORS_LOADED
    if _COMPRESSORS_LOADED:
        return

    package_name = (__package__ or "eval_harness.kv_compression") + ".compressors"
    try:
        package = importlib.import_module(package_name)
    except ModuleNotFoundError:
        # compressors/ is populated in a later migration step; tolerate absence.
        _COMPRESSORS_LOADED = True
        return
    for module_info in pkgutil.iter_modules(package.__path__):
        if module_info.ispkg or module_info.name in {"__init__", "registry"}:
            continue
        importlib.import_module(f"{package_name}.{module_info.name}")

    _COMPRESSORS_LOADED = True


def get_kv_compressor_class(name: str) -> Type[KVCompressor]:
    ensure_compressors_loaded()
    key = name.strip().lower()
    cls = _KV_COMPRESSOR_REGISTRY.get(key)
    if cls is None:
        available = ", ".join(available_kv_compressors())
        raise ValueError(f"Unknown KV compressor '{name}'. Available: {available}")
    return cls


def get_kv_compressor(name: str, **kwargs: Any) -> KVCompressor:
    return get_kv_compressor_class(name)(**kwargs)


def available_kv_compressors() -> List[str]:
    ensure_compressors_loaded()
    return sorted(_KV_COMPRESSOR_REGISTRY.keys())
