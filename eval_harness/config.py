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

    # Inference backend: "vllm", "hf", "rag", or "research"
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
    # Skip the first N rows of every subset before max_requests* caps apply
    # (deterministic head slicing becomes rows [offset : offset+limit]).
    # Lets disjoint tuning/eval splits share one dataset, e.g. eval on rows
    # 0-99 (offset 0, max_requests 100) and tune on rows 100-104 (offset 100,
    # max_requests 5).
    request_offset: int = 0
    # When False, each row generates as its own group instead of grouping rows
    # that share an identical context. Needed for decode-time KV compression on
    # benchmarks whose rows all carry one trivial context (math500 / aime2025):
    # context-grouping would put every question behind a single prefill, and
    # decode compression is incompatible with multi-question generation.
    group_by_context: bool = True
    query_aware: bool = False
    output_dir: str = "./results"

    # Resume: skip this run if a completed, identical run (matching fingerprint)
    # already exists in its result folder. On by default so re-running the same
    # thing is a no-op; pass CLI --force (sets this False) to redo it.
    resume: bool = True
    # When True, ``output_dir`` IS the exact run folder — write straight into it
    # (no descriptive name, no numbered subdirs). The sweep sets this because it
    # already chose a barcode-named leaf. Off = the runner appends a descriptive,
    # barcode-suffixed subfolder (still flat).
    output_dir_exact: bool = False

    # Opt-in run-to-run determinism. When False (default), only the basic seeds
    # (random/numpy/torch.manual_seed) are pinned — matches main behavior. When
    # True, also pins torch.use_deterministic_algorithms, cudnn.deterministic,
    # and disables the nondeterministic mem-efficient and cuDNN SDPA backends in
    # favor of flash + math (deterministic). Pair with
    # CUBLAS_WORKSPACE_CONFIG=:4096:8 in the environment. Ignored on vLLM (it
    # uses its own kernels).
    deterministic: bool = False

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
        if self.request_offset < 0:
            raise ValueError(f"request_offset must be >= 0, got {self.request_offset}")
        if self.request_offset > 0 and self.fraction < 1.0:
            # fraction sampling reshuffles rows BEFORE the offset slice, so
            # [offset : offset+limit] would index a seed-dependent random
            # subsample and the disjoint tuning/eval guarantee silently breaks.
            raise ValueError(
                "request_offset > 0 requires fraction == 1.0 "
                f"(got request_offset={self.request_offset}, fraction={self.fraction})"
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

    def get_results_dir(self, barcode: Optional[str] = None) -> Path:
        base = Path(self.output_dir)

        # Caller (the sweep) already chose the exact, barcode-named leaf folder.
        if self.output_dir_exact:
            base.mkdir(parents=True, exist_ok=True)
            return base

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

        name = "__".join([c for c in components if c])
        # Barcode suffix makes the folder unique-by-settings: same settings reuse
        # (overwrite/resume) the same folder, different settings get their own.
        # No numbered subdirs — the folder IS the run (flat).
        if barcode:
            name = f"{name}__{barcode}"
        run_dir = base / name
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
        # A silent miss would run the pure-default EvalConfig (vllm/ruler32k),
        # i.e. a completely different eval than the one requested.
        raise FileNotFoundError(f"Config file not found: {cfg.resolve()}")
    with cfg.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}
