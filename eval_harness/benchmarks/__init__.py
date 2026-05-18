from .registry import (
	available_benchmarks,
	ensure_benchmarks_loaded,
	get_benchmark,
	get_registered_benchmarks,
	register_benchmark,
)

__all__ = [
	"register_benchmark",
	"get_registered_benchmarks",
	"ensure_benchmarks_loaded",
	"get_benchmark",
	"available_benchmarks",
]
