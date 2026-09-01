"""Timing / memory instrumentation for the eval harness.

Importing this package must never pull CUDA or load a model, so the heavy
modules are resolved lazily.
"""

from __future__ import annotations

__all__ = [
    "PerfCell", "SCHEMA_VERSION", "newest_perf", "write_perf",
    "summarize", "percentile", "decode_throughput_total", "speedup",
    "kv_cache_accounting", "build_exact_prompt", "capture_environment",
    "audit_cell", "audit_group", "expected_budget", "roofline_step_ms",
    "load_runtime", "time_cell",
]

_LAZY = {
    "PerfCell": ("cell", "PerfCell"), "SCHEMA_VERSION": ("cell", "SCHEMA_VERSION"),
    "newest_perf": ("cell", "newest_perf"), "write_perf": ("cell", "write_perf"),
    "summarize": ("stats", "summarize"), "percentile": ("stats", "percentile"),
    "decode_throughput_total": ("stats", "decode_throughput_total"),
    "speedup": ("stats", "speedup"),
    "kv_cache_accounting": ("kvsize", "kv_cache_accounting"),
    "build_exact_prompt": ("prompts", "build_exact_prompt"),
    "capture_environment": ("environment", "capture_environment"),
    "audit_cell": ("audit", "audit_cell"), "audit_group": ("audit", "audit_group"),
    "expected_budget": ("audit", "expected_budget"),
    "roofline_step_ms": ("audit", "roofline_step_ms"),
    "load_runtime": ("runner", "load_runtime"), "time_cell": ("runner", "time_cell"),
}


def __getattr__(name):
    if name in _LAZY:
        import importlib
        mod, attr = _LAZY[name]
        return getattr(importlib.import_module(f"{__name__}.{mod}"), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
