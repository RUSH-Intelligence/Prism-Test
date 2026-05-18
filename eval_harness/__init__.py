"""vLLM-based evaluation harness compatible with KVPress benchmark scoring."""

from .config import EvalConfig

__all__ = ["EvalConfig", "EvalRunner"]


def __getattr__(name: str):
	if name == "EvalRunner":
		from .runner import EvalRunner

		return EvalRunner
	raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
