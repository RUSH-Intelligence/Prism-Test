from __future__ import annotations

from typing import Dict, List

from .aime import AIMEBenchmark
from .base import Benchmark
from .loft import LoftBenchmark
from .longbench import LongBenchBenchmark
from .longbenchv2 import LongBenchV2Benchmark
from .prism1m import Prism1MBenchmark
from .ruler32k import Ruler32KBenchmark


BENCHMARKS: Dict[str, Benchmark] = {
    "aime": AIMEBenchmark(),
    "ruler32k": Ruler32KBenchmark(),
    "longbench": LongBenchBenchmark(),
    "longbenchv2": LongBenchV2Benchmark(),
    "loft": LoftBenchmark(),
    "prism1m": Prism1MBenchmark(),
}


def get_benchmark(name: str) -> Benchmark:
    key = name.strip().lower()
    if key not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark '{name}'. Available: {sorted(BENCHMARKS.keys())}")
    return BENCHMARKS[key]


def available_benchmarks() -> List[str]:
    return sorted(BENCHMARKS.keys())
