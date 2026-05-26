from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .config import EvalConfig
from .benchmarks.registry import available_benchmarks, get_benchmark
from .rag_adapter import RAGAdapter

logger = logging.getLogger(__name__)


class EvalRunner:
    def __init__(self, config: EvalConfig) -> None:
        self.config = config
        self.benchmark = get_benchmark(config.benchmark)
        self.adapter: VLLMAdapter | HFAdapter | RAGAdapter | None = None
        self.df: pd.DataFrame | None = None
        self._setup_logging()
        self._set_seed(config.seed)

    def _setup_logging(self) -> None:
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            logger.addHandler(handler)

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

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
            self.adapter = RAGAdapter()
        elif self.config.backend == "hf":
            from .hf_adapter import HFAdapter

            self.adapter = HFAdapter(
                model=self.config.model,
                dtype=self.config.dtype,
                max_model_len=self.config.max_model_len,
                trust_remote_code=self.config.trust_remote_code,
                seed=self.config.seed,
                enable_long_context_compression=self.config.enable_long_context_compression,
                compression_sink_tokens=self.config.compression_sink_tokens,
                compression_local_tokens=self.config.compression_local_tokens,
                compression_top_k_tokens=self.config.compression_top_k_tokens,
                compression_span_tokens=self.config.compression_span_tokens,
                hf_naive_reattn_query_tokens=self.config.hf_naive_reattn_query_tokens,
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
            if isinstance(self.adapter, RAGAdapter):
                questions = [str(row["question"]) for _, row in group.iterrows()]
                answers = self.adapter.generate_for_context(context, questions)
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
