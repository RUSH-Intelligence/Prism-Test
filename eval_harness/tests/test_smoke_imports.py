"""Smoke test: every core module imports without crashing.

Catches broken renames, deleted files, and circular imports that unit tests
miss because they only import the piece they directly exercise.
"""
from __future__ import annotations

import importlib
import importlib.util
import unittest


CORE_MODULES = [
    "eval_harness",
    "eval_harness.cli",
    "eval_harness.config",
    "eval_harness.runner",
    "eval_harness.hf_adapter",
    "eval_harness.research_adapter",
    "eval_harness.benchmarks",
    "eval_harness.benchmarks.base",
    "eval_harness.benchmarks.common",
    "eval_harness.benchmarks.registry",
    "eval_harness.benchmarks.mock_benchmark",
    "eval_harness.kv_compression",
    "eval_harness.kv_compression.cache_adapter",
]

OPTIONAL_MODULES = {
    "eval_harness.vllm_adapter": "vllm",
    "eval_harness.rag_adapter": "lancedb",
}


class TestSmokeImports(unittest.TestCase):
    def test_core_modules_import_cleanly(self):
        for module in CORE_MODULES:
            with self.subTest(module=module):
                importlib.import_module(module)

    def test_optional_modules_import_when_deps_present(self):
        # Optional backends may install partially on some platforms (e.g. vLLM
        # on Windows has no `vllm._C` extension). An ImportError means the
        # backend isn't usable here — treat as skipped, not failed.
        for module, dep in OPTIONAL_MODULES.items():
            if importlib.util.find_spec(dep) is None:
                continue
            try:
                importlib.import_module(module)
            except ImportError:
                continue

    def test_registry_autoload_and_mock_benchmark_present(self):
        from eval_harness.benchmarks.registry import available_benchmarks

        names = available_benchmarks()
        self.assertGreater(len(names), 0, "benchmark registry is empty")
        self.assertIn("mock_benchmark", names)


if __name__ == "__main__":
    unittest.main()
