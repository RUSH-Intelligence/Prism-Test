from __future__ import annotations

import json
import logging
import os
import random
import shutil
import uuid
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


def _supersede_prior_run(run_dir: Path) -> "Path | None":
    """Move a prior run's artifacts out of ``run_dir`` into ``superseded/<n>/``.

    Called at the start of a (re)run so the run always begins on a clean top
    level. No-op when the folder is empty (a first run) or holds only the
    ``superseded`` archive. Returns the archive dir it filled, else None.
    """
    from .run_spec import SUPERSEDED_DIRNAME

    if not run_dir.exists():
        return None
    entries = [p for p in run_dir.iterdir() if p.name != SUPERSEDED_DIRNAME]
    if not entries:
        return None

    archive_root = run_dir / SUPERSEDED_DIRNAME
    idx = 1
    while (archive_root / str(idx)).exists():
        idx += 1
    dest = archive_root / str(idx)
    dest.mkdir(parents=True, exist_ok=True)
    for p in entries:
        shutil.move(str(p), str(dest / p.name))
    logger.info("Preserved prior run output (%d item(s)) under %s", len(entries), dest)
    return dest


def _commit_run(work: Path, final: Path) -> None:
    """Atomically install a fully-written temp run dir at its final barcode path.

    Fast path: ``os.rename`` succeeds when ``final`` is missing or an empty dir
    (the common case) — a single atomic step, so a concurrent identical run
    cannot interleave files into the final folder. If ``final`` is non-empty (a
    prior complete run, or a racer that committed first), preserve it under
    ``superseded/<n>/`` and move our files in with DONE.json LAST — old results
    are kept and no torn-yet-stamped state can appear.
    """
    from .run_spec import DONE_FILENAME

    final.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.rename(work, final)          # atomic when final is missing / empty
        return
    except OSError:
        pass                            # final exists and is non-empty -> merge
    _supersede_prior_run(final)
    entries = sorted(work.iterdir(), key=lambda p: p.name == DONE_FILENAME)  # DONE last
    for p in entries:
        shutil.move(str(p), str(final / p.name))
    work.rmdir()


class EvalRunner:
    def __init__(self, config: EvalConfig) -> None:
        self.config = config
        self.benchmark = get_benchmark(config.benchmark)
        self.adapter: "VLLMAdapter | HFAdapter | RAGAdapter | None" = None
        self.df: pd.DataFrame | None = None
        self._setup_logging()
        self._set_seed(config.seed)
        # Determinism flags (cudnn / SDPA backend pinning) shift attention
        # kernels and numerics vs the prior unpinned defaults, so they are
        # opt-in. vLLM uses its own kernels and ignores these.
        if config.deterministic and config.backend != "vllm":
            self._enable_determinism()

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

    @staticmethod
    def _enable_determinism() -> None:
        # Opt-in (EvalConfig.deterministic=True). Best-effort run-to-run
        # reproducibility — warn_only=True so ops without a deterministic kernel
        # (e.g. some scatter/topk paths in compressors) log a warning instead of
        # raising. Pair with CUBLAS_WORKSPACE_CONFIG=:4096:8 in the environment —
        # without it cuBLAS GEMM choice is not pinned.
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
            # The cuDNN SDPA backend (preferred on Hopper for bf16 in recent
            # torch) is NOT run-to-run reproducible: with it enabled, repeated
            # greedy generations of the same prompt differed within one process
            # on H200 (kv_compression_adaptation, 2026-09: 80/520 RULER
            # generations changed between two runs). Disable it so
            # ``deterministic=True`` keeps only flash + math, both of which
            # reproduce bit-for-bit across jobs and nodes.
            if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
                torch.backends.cuda.enable_cudnn_sdp(False)

    def _build_prompt(self, context: str, question: str, answer_prefix: str) -> str:
        # Match sparse-attention-hub request assembly for RULER benchmarks.
        return f"{context}{question}{answer_prefix}"

    @staticmethod
    def _apply_max_requests(
        df: pd.DataFrame,
        max_requests: int | None,
        max_requests_per_subset: Dict[str, int] | None,
        request_offset: int = 0,
    ) -> pd.DataFrame:
        offset = max(0, int(request_offset))
        if max_requests is None and not max_requests_per_subset and offset == 0:
            return df
        if max_requests is not None and max_requests <= 0:
            return df.head(0)

        subset_limits = max_requests_per_subset or {}

        # Apply per-subset offset + request cap: rows [offset : offset+limit].
        if "task" in df.columns:
            parts: List[pd.DataFrame] = []
            for task, task_df in df.groupby("task", sort=False):
                limit = subset_limits.get(str(task), max_requests)
                if limit is None:
                    parts.append(task_df.iloc[offset:])
                elif limit <= 0:
                    parts.append(task_df.head(0))
                else:
                    parts.append(task_df.iloc[offset:offset + limit])
            return pd.concat(parts, ignore_index=True) if parts else df.head(0)

        if max_requests is None:
            return df.iloc[offset:]
        return df.iloc[offset:offset + max_requests]

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

        # Rows available after subset/fraction filtering, before the per-subset
        # request cap — recorded in the completion stamp as ``loaded_before_cap``.
        self._n_loaded_before_cap = int(len(df))

        df = self._apply_max_requests(
            df,
            self.config.max_requests,
            self.config.max_requests_per_subset,
            self.config.request_offset,
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

        if self.config.group_by_context:
            grouped = self.df.groupby("context", sort=False)
            n_groups = self.df["context"].nunique()
        else:
            # Row-per-group: required for decode-time KV compression on
            # benchmarks whose rows share one trivial context (math500/aime2025
            # ship context == " " for every row — grouping them would put all
            # questions behind one prefill, which decode compression forbids).
            grouped = (
                (group["context"].iloc[0], group)
                for _, group in self.df.groupby(self.df.index, sort=False)
            )
            n_groups = len(self.df)
        for context, group in tqdm(grouped, total=n_groups, desc="Generating"):
            # Stamp the compressor with this context group's df rows so any
            # coverage readings can be tied back to the exact questions (no-op
            # unless the compressor tracks it — e.g. VerifiedSketch).
            begin_group = getattr(
                getattr(self.adapter, "_kv_compressor", None), "begin_prompt_group", None
            )
            if callable(begin_group):
                begin_group(list(group.index))

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

                # Per-row chat-template override (e.g. LongBench skips the chat
                # wrapper on trec/triviaqa/samsum/lcc/repobench-p to match the
                # official pred.py). When the column is absent or mixed inside a
                # group, fall back to the adapter's config-level default by
                # passing None.
                chat_override: bool | None = None
                if "use_chat_template" in group.columns:
                    uniq = group["use_chat_template"].drop_duplicates().tolist()
                    if len(uniq) == 1:
                        chat_override = bool(uniq[0])

                # Same pattern for the system-block strip: LongBench opts in
                # per-row; absent column means use the adapter default (off).
                strip_override: bool | None = None
                if "strip_auto_system_block" in group.columns:
                    uniq_s = group["strip_auto_system_block"].drop_duplicates().tolist()
                    if len(uniq_s) == 1:
                        strip_override = bool(uniq_s[0])

                middle_trunc_override: bool | None = None
                if "middle_truncation" in group.columns:
                    uniq_m = group["middle_truncation"].drop_duplicates().tolist()
                    if len(uniq_m) == 1:
                        middle_trunc_override = bool(uniq_m[0])

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
                    use_chat_template=chat_override,
                    strip_auto_system_block=strip_override,
                    middle_truncation=middle_trunc_override,
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
        # Fingerprint the run up front (model-free), so we can name the folder by
        # it and skip if an identical, finished run already lives there.
        spec = None
        barcode = None
        try:
            from .run_spec import build_run_spec

            spec = build_run_spec(self.config)
            barcode = spec["fingerprint"]
        except Exception as exc:
            logger.warning("run_spec fingerprint failed (%s); no barcode/resume", exc)

        run_dir = self.config.get_results_dir(barcode)

        # Universal resume: identical settings already completed here -> skip
        # (no model load, no generation).
        if self.config.resume and barcode is not None:
            from .run_spec import DONE_FILENAME

            done_path = run_dir / DONE_FILENAME
            if done_path.exists():
                try:
                    stored = json.loads(done_path.read_text(encoding="utf-8")).get("fingerprint")
                except (OSError, ValueError):
                    stored = None
                if stored == barcode:
                    logger.info("Already complete (fingerprint %s) — skipping: %s",
                                barcode, run_dir)
                    return run_dir

        # Stage every output in a PRIVATE temp dir, then install it at run_dir in
        # one atomic step at the very end (write-temp-then-rename). Consequences:
        #  * an identical CONCURRENT run cannot interleave files into run_dir;
        #  * run_dir is untouched until the run fully completes, so a run that
        #    dies partway never corrupts a prior complete result;
        #  * the commit preserves any prior run under superseded/<n>/ (so old
        #    results are never overwritten — the #6 fix) and writes DONE last.
        work_dir = run_dir.parent / f".{run_dir.name}.inprogress.{os.getpid()}.{uuid.uuid4().hex[:8]}"
        work_dir.mkdir(parents=True, exist_ok=True)

        predictions_path = work_dir / "predictions.csv"
        metrics_path = work_dir / "metrics.json"
        config_path = work_dir / "config.yaml"

        try:
            self._setup_adapter()
            self._load_dataset()
            self._run_generation()
            metrics = self._compute_metrics()

            assert self.df is not None
            cols = [c for c in self.df.columns if c != "context"]
            self.df[cols].to_csv(predictions_path, index=False)

            with metrics_path.open("w", encoding="utf-8") as handle:
                json.dump(metrics, handle, indent=2)

            # Generic compressor telemetry drain: any compressor exposing
            # ``drain_coverage()`` (currently VerifiedSketch, measure_coverage on)
            # gets its readings persisted next to metrics.json. Guarded so it
            # never affects a normal run.
            compressor = getattr(self.adapter, "_kv_compressor", None)
            drain = getattr(compressor, "drain_coverage", None)
            if callable(drain):
                try:
                    coverage = drain()
                except Exception as exc:  # telemetry must never fail a run
                    logger.warning("Coverage drain failed: %s", exc)
                    coverage = None
                if coverage:
                    coverage_path = work_dir / "coverage.json"
                    with coverage_path.open("w", encoding="utf-8") as handle:
                        json.dump(coverage, handle, indent=2)
                    logger.info("Saved coverage telemetry to %s", coverage_path)

            config_dump = asdict(self.config)
            with config_path.open("w", encoding="utf-8") as handle:
                import yaml

                yaml.safe_dump(config_dump, handle, sort_keys=False)

            # Canonical, fingerprinted settings receipt + completion stamp
            # (foundation for robust sweep resume). Best-effort: these must never
            # fail a real run. The DONE stamp is written LAST, so its presence
            # proves predictions + metrics + receipt all completed.
            try:
                from .run_spec import build_run_spec, build_done_marker, write_done_marker

                if spec is None:                 # fingerprinting failed up top; retry
                    spec = build_run_spec(self.config)
                with (work_dir / "run_spec.json").open("w", encoding="utf-8") as handle:
                    json.dump(spec, handle, indent=2)
                logger.info("Saved run-spec to %s", work_dir / "run_spec.json")

                per_subset = None
                if self.df is not None and "task" in self.df.columns:
                    per_subset = self.df["task"].value_counts().to_dict()
                requested_subsets = None
                if self.config.subsets:
                    requested_subsets = sorted(
                        s.strip() for s in self.config.subsets.split(",") if s.strip())
                marker = build_done_marker(
                    fingerprint=spec["fingerprint"],
                    actual_samples=int(len(self.df)) if self.df is not None else 0,
                    overall_score=metrics.get("overall_score") if isinstance(metrics, dict) else None,
                    max_requests=self.config.max_requests,
                    max_requests_per_subset=self.config.max_requests_per_subset,
                    requested_subsets=requested_subsets,
                    per_subset_actual=per_subset,
                    loaded_before_cap=getattr(self, "_n_loaded_before_cap", None),
                )
                write_done_marker(work_dir, marker)   # LAST write into the temp dir
            except Exception as exc:
                logger.warning("run_spec/DONE generation failed: %s", exc)

            # Atomic install: temp dir -> final barcode folder (preserving any
            # prior run under superseded/). Only after this does run_dir hold the
            # complete result.
            _commit_run(work_dir, run_dir)
        finally:
            if work_dir.exists():                 # nothing to install (crash / merged)
                shutil.rmtree(work_dir, ignore_errors=True)

        logger.info("Saved predictions to %s", run_dir / "predictions.csv")
        logger.info("Saved metrics to %s", run_dir / "metrics.json")
        logger.info("Available standalone benchmarks: %s", ", ".join(available_benchmarks()))
        return run_dir
