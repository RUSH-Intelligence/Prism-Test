from __future__ import annotations

import importlib
import pkgutil
from typing import Dict, List, Optional, Type

from .base import Benchmark


_BENCHMARK_REGISTRY: Dict[str, Type[Benchmark]] = {}
_BENCHMARKS_LOADED = False


def register_benchmark(name: Optional[str] = None, aliases: Optional[List[str]] = None):
    def decorator(benchmark_class: Type[Benchmark]) -> Type[Benchmark]:
        key = (name or benchmark_class.__name__).strip().lower()
        _BENCHMARK_REGISTRY[key] = benchmark_class
        if aliases:
            for alias in aliases:
                _BENCHMARK_REGISTRY[alias.strip().lower()] = benchmark_class
        return benchmark_class

    return decorator


def ensure_benchmarks_loaded() -> None:
    global _BENCHMARKS_LOADED
    if _BENCHMARKS_LOADED:
        return

    package_name = __package__ or "eval_harness.benchmarks"
    package = importlib.import_module(package_name)
    for module_info in pkgutil.iter_modules(package.__path__):
        module_name = module_info.name
        if module_info.ispkg or module_name in {"__init__", "base", "common", "registry"}:
            continue
        importlib.import_module(f"{package_name}.{module_name}")

    _BENCHMARKS_LOADED = True


def get_benchmark(name: str) -> Benchmark:
    ensure_benchmarks_loaded()
    key = name.strip().lower()
    if key not in _BENCHMARK_REGISTRY:
        raise ValueError(f"Unknown benchmark '{name}'. Available: {sorted(_BENCHMARK_REGISTRY.keys())}")
    return _BENCHMARK_REGISTRY[key]()


def available_benchmarks() -> List[str]:
    ensure_benchmarks_loaded()
    return sorted(_BENCHMARK_REGISTRY.keys())


def get_registered_benchmarks() -> Dict[str, Type[Benchmark]]:
    ensure_benchmarks_loaded()
    return dict(_BENCHMARK_REGISTRY)
