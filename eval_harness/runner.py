from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .config import EvalConfig
from .benchmarks.registry import available_benchmarks, get_benchmark

if TYPE_CHECKING:
    from .hf_adapter import HFAdapter
    from .rag_adapter import RAGAdapter
    from .vllm_adapter import VLLMAdapter

logger = logging.getLogger(__name__)


class EvalRunner:
    def __init__(self, config: EvalConfig) -> None:
        self.config = config
        self.benchmark = get_benchmark(config.benchmark)
        self.adapter: "VLLMAdapter | HFAdapter | RAGAdapter | None" = None
        self.df: pd.DataFrame | None = None
        self._setup_logging()
        self._set_seed(config.seed)

    def _setup_logging(self) -> None:
        # Configure the package/root logger so sibling modules (e.g. hf_adapter)
        # propagate to the same console handler.
        root = logging.getLogger()
        root.setLevel(logging.INFO)
        if not root.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            root.addHandler(handler)

        # Keep eval logs at INFO, but suppress noisy per-request HTTP logs.
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Best-effort run-to-run reproducibility. warn_only=True so ops without a
        # deterministic kernel (e.g. some scatter/topk paths in compressors) log
        # a warning instead of raising. Pair with CUBLAS_WORKSPACE_CONFIG=:4096:8
        # in the environment — without it, cuBLAS GEMM choice is not pinned.
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Pin the SDPA backend so runs are reproducible AND comparable to kvpress
        # baselines. The mem-efficient backend is nondeterministic on some shapes
        # and is what use_deterministic_algorithms would silently route AWAY from
        # under warn_only=True, leaving us with an unannounced backend swap vs
        # the published numbers. Disable it; keep flash + math (both
        # deterministic). flash is preferred when available, math is the fallback.
        if torch.cuda.is_available():
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)

    def _build_prompt(self, context: str, question: str, answer_prefix: str) -> str:
        # Match sparse-attention-hub request assembly for RULER benchmarks.
        return f"{context}{question}{answer_prefix}"

    @staticmethod
    def _apply_max_requests(
        df: pd.DataFrame,
        max_requests: int | None,
        max_requests_per_subset: Dict[str, int] | None,
    ) -> pd.DataFrame:
        if max_requests is None and not max_requests_per_subset:
            return df
        if max_requests is not None and max_requests <= 0:
            return df.head(0)

        subset_limits = max_requests_per_subset or {}

        # Apply request cap per task/subset when available.
        if "task" in df.columns:
            parts: List[pd.DataFrame] = []
            for task, task_df in df.groupby("task", sort=False):
                limit = subset_limits.get(str(task), max_requests)
                if limit is None:
                    parts.append(task_df)
                elif limit <= 0:
                    parts.append(task_df.head(0))
                else:
                    parts.append(task_df.head(limit))
            return pd.concat(parts, ignore_index=True) if parts else df.head(0)

        if max_requests is None:
            return df
        return df.head(max_requests)

    def _load_dataset(self) -> None:
        subsets = None
        if self.config.subsets:
            subsets = [s.strip() for s in self.config.subsets.split(",") if s.strip()]

        logger.info(
            "Loading benchmark %s (subsets=%s)",
            self.config.benchmark,
            subsets if subsets else "default",
        )
        df = self.benchmark.load(subsets=subsets)

        if self.config.fraction < 1.0:
            df = df.sample(frac=self.config.fraction, random_state=self.config.seed)

        df = self._apply_max_requests(
            df,
            self.config.max_requests,
            self.config.max_requests_per_subset,
        )

        for col in ["context", "question"]:
            if col not in df.columns:
                raise ValueError(f"Dataset is missing required column: {col}")

        if "answer_prefix" not in df.columns:
            df["answer_prefix"] = ""

        if "max_new_tokens" not in df.columns:
            df["max_new_tokens"] = 64

        if self.config.query_aware:
            df["context"] = df["context"] + df["question"]
            df["question"] = ""

        self.df = df
        logger.info("Loaded %d evaluation rows", len(df))

    def _setup_adapter(self) -> None:
        if self.config.backend == "rag":
            from .rag_adapter import RAGAdapter

            self.adapter = RAGAdapter()
        elif self.config.backend == "research":
            from .research_adapter import ResearchAdapter, ResearchConfig

            llm_kw = dict(self.config.llm_kwargs or {})
            # Research backend depends on consistent ALL_ATTENTION_FUNCTIONS
            # dispatch; SDPA is the most reliable parity path.
            llm_kw.setdefault("attn_implementation", "sdpa")

            # Pull the research_config dict out of llm_kwargs (the three-door
            # configuration) and convert it to a ResearchConfig.
            research_kw = llm_kw.pop("research_config", {}) or {}
            research_cfg = ResearchConfig(**research_kw) if research_kw else ResearchConfig()

            self.adapter = ResearchAdapter(
                model=self.config.model,
                dtype=self.config.dtype,
                max_model_len=self.config.max_model_len,
                trust_remote_code=self.config.trust_remote_code,
                seed=self.config.seed,
                research_config=research_cfg,
                **llm_kw,
            )
        elif self.config.backend == "hf":
            from .hf_adapter import HFAdapter

            self.adapter = HFAdapter(
                model=self.config.model,
                dtype=self.config.dtype,
                max_model_len=self.config.max_model_len,
                trust_remote_code=self.config.trust_remote_code,
                seed=self.config.seed,
                **(self.config.llm_kwargs or {}),
            )
        else:
            from .vllm_adapter import VLLMAdapter

            self.adapter = VLLMAdapter(
                model=self.config.model,
                tensor_parallel_size=self.config.tensor_parallel_size,
                dtype=self.config.dtype,
                max_model_len=self.config.max_model_len,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                trust_remote_code=self.config.trust_remote_code,
                enable_prefix_caching=self.config.enable_prefix_caching,
                seed=self.config.seed,
                **(self.config.llm_kwargs or {}),
            )

    def _run_generation(self) -> None:
        assert self.df is not None
        assert self.adapter is not None

        self.df = self.df.copy()
        self.df["predicted_answer"] = None

        grouped = self.df.groupby("context", sort=False)
        for context, group in tqdm(grouped, total=self.df["context"].nunique(), desc="Generating"):
            if self.config.backend == "rag":
                questions = [str(row["question"]) for _, row in group.iterrows()]
                assert self.adapter is not None
                answers = self.adapter.generate_for_context(context, questions)
            elif self.config.backend == "research":
                from .hf_adapter import HFGenerateConfig
                from .research_adapter import ResearchAdapter

                assert isinstance(self.adapter, ResearchAdapter)

                max_tokens = self.config.max_new_tokens
                if max_tokens is None:
                    max_tokens = int(group["max_new_tokens"].iloc[0])

                answer_prefixes = group["answer_prefix"].astype(str).drop_duplicates().tolist()
                if len(answer_prefixes) != 1:
                    raise ValueError(
                        "Inconsistent answer_prefix values detected within the same context group. "
                        "Research backend expects one shared answer_prefix per context."
                    )

                gen_cfg = HFGenerateConfig(
                    max_tokens=max_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                )
                answers = self.adapter.generate_for_context(
                    context=context,
                    questions=[str(row["question"]) for _, row in group.iterrows()],
                    answer_prefix=answer_prefixes[0],
                    gen_cfg=gen_cfg,
                )
            else:
                prompts: List[str] = []
                for _, row in group.iterrows():
                    prompts.append(
                        self._build_prompt(
                            context=context,
                            question=str(row["question"]),
                            answer_prefix=str(row["answer_prefix"]),
                        )
                    )

                max_tokens = self.config.max_new_tokens
                if max_tokens is None:
                    max_tokens = int(group["max_new_tokens"].iloc[0])

                if self.config.backend == "hf":
                    from .hf_adapter import HFGenerateConfig

                    gen_cfg = HFGenerateConfig(
                        max_tokens=max_tokens,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                    )
                else:
                    from .vllm_adapter import VLLMGenerateConfig

                    gen_cfg = VLLMGenerateConfig(
                        max_tokens=max_tokens,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                    )
                answers = self.adapter.generate(prompts, gen_cfg)

            self.df.loc[group.index, "predicted_answer"] = answers

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _compute_metrics(self) -> Dict[str, float | Dict[str, float]]:
        assert self.df is not None
        return self.benchmark.score(self.df)

    def run(self) -> Path:
        logger.info(
            "Starting evaluation for benchmark=%s model=%s",
            self.config.benchmark,
            self.config.model,
        )
        run_dir = self.config.get_results_dir()
        predictions_path = run_dir / "predictions.csv"
        metrics_path = run_dir / "metrics.json"
        config_path = run_dir / "config.yaml"

        self._setup_adapter()
        self._load_dataset()
        self._run_generation()
        metrics = self._compute_metrics()

        assert self.df is not None
        cols = [c for c in self.df.columns if c != "context"]
        self.df[cols].to_csv(predictions_path, index=False)

        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)

        config_dump = asdict(self.config)
        with config_path.open("w", encoding="utf-8") as handle:
            import yaml

            yaml.safe_dump(config_dump, handle, sort_keys=False)

        logger.info("Saved predictions to %s", predictions_path)
        logger.info("Saved metrics to %s", metrics_path)
        logger.info("Available standalone benchmarks: %s", ", ".join(available_benchmarks()))
        return run_dir
