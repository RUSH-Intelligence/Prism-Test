"""Auto-discovery registry for KV-compression sketches.

Mirrors ``eval_harness/prefill_methods/registry.py``: sketch modules decorate
their class with ``@register_sketch("name")`` and are auto-discovered on first
lookup, so adding a sketch never requires editing shared files.

Composite sketches that wrap other sketch instances (``DecodingSketch``,
``PrefillDecodingSketch``) are constructed by
``ResearchAdapter._build_sketch``'s named special cases rather than registered
here, because their sub-sketch arguments cannot be expressed as plain config
kwargs.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any, Dict, List, Optional, Type

from .base_sketch import BaseSketch

_SKETCH_REGISTRY: Dict[str, Type[BaseSketch]] = {}
_SKETCHES_LOADED = False


def register_sketch(
    name: str,
    aliases: Optional[List[str]] = None,
):
    """Decorator to register a sketch class.

    ``aliases`` exists for legacy names that predate the registry (e.g.
    ``knorm_sketch``); new sketches register a single canonical name.

    Usage::

        @register_sketch("knorm", aliases=["knorm_sketch"])
        @dataclass
        class KnormSketch(ScorerSketch):
            ...
    """

    def decorator(cls: Type[BaseSketch]) -> Type[BaseSketch]:
        key = name.strip().lower()
        _SKETCH_REGISTRY[key] = cls
        for alias in aliases or []:
            _SKETCH_REGISTRY[alias.strip().lower()] = cls
        return cls

    return decorator


def ensure_sketches_loaded() -> None:
    """Auto-discover all sketch modules in this package."""
    global _SKETCHES_LOADED
    if _SKETCHES_LOADED:
        return

    package_name = __package__ or "eval_harness.sketch.sketches"
    package = importlib.import_module(package_name)
    for module_info in pkgutil.iter_modules(package.__path__):
        if module_info.ispkg or module_info.name in {
            "__init__",
            "registry",
        }:
            continue
        importlib.import_module(f"{package_name}.{module_info.name}")

    _SKETCHES_LOADED = True


def get_sketch_class(name: str) -> Type[BaseSketch]:
    """Resolve a registered sketch class by name (no instantiation)."""
    ensure_sketches_loaded()
    key = name.strip().lower()
    cls = _SKETCH_REGISTRY.get(key)
    if cls is None:
        available = ", ".join(available_sketches())
        raise ValueError(f"Unknown sketch '{name}'. Available: {available}")
    return cls


def get_sketch(name: str, **kwargs: Any) -> BaseSketch:
    """Instantiate a registered sketch by name."""
    return get_sketch_class(name)(**kwargs)


def available_sketches() -> List[str]:
    """Return sorted list of all registered sketch names."""
    ensure_sketches_loaded()
    return sorted(_SKETCH_REGISTRY.keys())
