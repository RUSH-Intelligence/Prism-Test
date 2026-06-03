from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class EvalConfig:
    # Benchmark selection.
    benchmark: str = "ruler32k"
    subsets: Optional[str] = None

    # Inference backend: "vllm", "hf", or "rag"
    backend: str = "vllm"

    # Model and runtime.
    model: str = "meta-llama/Llama-3.1-8B-Instruct"
    tensor_parallel_size: int = 1
    dtype: str = "auto"
    max_model_len: Optional[int] = None
    gpu_memory_utilization: float = 0.9
    trust_remote_code: bool = True
    enable_prefix_caching: bool = True

    # Generation.
    max_new_tokens: Optional[int] = None
    temperature: float = 0.0
    top_p: float = 1.0
    system_prompt: Optional[str] = None
    seed: int = 42

    # Evaluation behavior.
    fraction: float = 1.0
    max_requests: Optional[int] = None
    max_requests_per_subset: Optional[Dict[str, int]] = None
    query_aware: bool = False
    output_dir: str = "./results"

    # Extra kwargs passthrough to the backend LLM.
    llm_kwargs: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.backend not in {"vllm", "hf", "rag", "research"}:
            raise ValueError(f"backend must be one of vllm|hf|rag|research, got {self.backend}")
        if not (0.0 < self.fraction <= 1.0):
            raise ValueError(f"fraction must be in (0, 1], got {self.fraction}")
        if not (0.0 <= self.temperature):
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if not (0.0 < self.gpu_memory_utilization <= 1.0):
            raise ValueError(
                f"gpu_memory_utilization must be in (0, 1], got {self.gpu_memory_utilization}"
            )
        if self.llm_kwargs is None:
            self.llm_kwargs = {}
        if self.max_requests_per_subset is None:
            self.max_requests_per_subset = {}
        else:
            cleaned: Dict[str, int] = {}
            for key, value in self.max_requests_per_subset.items():
                name = str(key).strip()
                if not name:
                    continue
                ivalue = int(value)
                if ivalue < 0:
                    raise ValueError(f"max_requests_per_subset[{name}] must be >= 0, got {ivalue}")
                cleaned[name] = ivalue
            self.max_requests_per_subset = cleaned

    def get_results_dir(self) -> Path:
        base = Path(self.output_dir)
        base.mkdir(parents=True, exist_ok=True)

        components = [
            self.benchmark,
            self.model.replace("/", "--"),
            self.backend,
            f"t{self.temperature:g}",
            f"p{self.top_p:g}",
        ]
        if self.subsets:
            subset_tag = self.subsets.replace(",", "-").replace(" ", "")
            components.append(f"subsets_{subset_tag}")
        if self.fraction < 1.0:
            components.append(f"fraction{self.fraction:.3f}")
        if self.query_aware:
            components.append("query_aware")

        run_dir = base / "__".join([c for c in components if c])
        if run_dir.exists():
            idx = 1
            while (run_dir / str(idx)).exists():
                idx += 1
            run_dir = run_dir / str(idx)

        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_yaml(self, path: Path) -> None:
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False)


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    cfg = Path(path)
    if not cfg.exists():
        return {}
    with cfg.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}
