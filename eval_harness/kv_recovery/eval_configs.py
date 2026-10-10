"""The mandatory three-way comparison (spec §14): dense / compressed / compressed_recovered
(optionally dense_recovered) ``EvalConfig`` dicts built from ONE ``RecoveryConfig``.

Guards against accidentally evaluating the baseline and the recovered model under different
compression: both arms are produced by the same function from the same block, the recovered
arm differs only in ``llm_kwargs.weight_delta`` (asserted), and the delta's recorded
compression block / prompt shaping / load flags must match the evaluation's unless the caller
explicitly overrides. Cells are barcode-named (``run_spec`` fingerprint) under a per-model
results root, so dense and compressed cells are shared across every training run of the same
model + compression setting and the runner's ``resume`` makes re-runs no-ops.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .config import BenchmarkSpec, RecoveryConfig, model_llm_kwargs, research_config_dict, training_identity

CONDITIONS = ("dense", "compressed", "compressed_recovered", "dense_recovered")
DEFAULT_CONDITIONS = ("dense", "compressed", "compressed_recovered")
COMPRESSED_CONDITIONS = {"compressed", "compressed_recovered"}
RECOVERED_CONDITIONS = {"compressed_recovered", "dense_recovered"}
# Keys that may legitimately differ between the compressed and the recovered arm.
_ARM_NOISE_KEYS = ("output_dir", "output_dir_exact", "resume")


@dataclass
class Cell:
    benchmark: str
    condition: str
    config: Dict[str, Any]
    run_dir: Path
    barcode: str

    @property
    def job_name(self) -> str:
        # Unique per cell (model / compressor / delta differ in the barcode), so the in-flight
        # check never mistakes another model's dense cell for this one.
        return f"kvrec_{self.benchmark}_{self.condition}_{self.barcode[:8]}"


def model_slug(name: str) -> str:
    return name.replace("/", "--")


def results_root(cfg: RecoveryConfig) -> Path:
    if cfg.eval.results_root:
        return Path(cfg.eval.results_root)
    return Path(cfg.output.root) / "eval" / model_slug(cfg.model.name)


def is_hybrid_model_name(name: str) -> bool:
    """Hybrid (linear-attention / mamba) families whose cache restore cannot roll back
    recurrent state → evaluate row-by-row (``group_by_context: false``)."""
    try:
        from transformers import AutoConfig

        try:   # prefer the local snapshot; fall back to the hub only when it is absent
            cfg = AutoConfig.from_pretrained(name, trust_remote_code=True, local_files_only=True)
        except Exception:
            cfg = AutoConfig.from_pretrained(name, trust_remote_code=True)
        try:
            cfg = cfg.get_text_config(decoder=True)
        except Exception:
            pass
        lt = [str(t).lower() for t in (getattr(cfg, "layer_types", None) or getattr(cfg, "layers_block_type", None) or [])]
        if lt:
            return any(("linear" in t or "mamba" in t) for t in lt)
        pattern = getattr(cfg, "hybrid_override_pattern", None)
        if isinstance(pattern, str):
            return "M" in pattern
    except Exception:
        pass
    low = name.lower()
    return any(tag in low for tag in ("qwen3.5", "qwen3_5", "qwen3-next", "nemotron", "mamba"))


def group_by_context_for(cfg: RecoveryConfig) -> bool:
    if cfg.eval.group_by_context is not None:
        return bool(cfg.eval.group_by_context)
    return not is_hybrid_model_name(cfg.model.name)


def barcode_for_eval_config(run_cfg: Dict[str, Any]) -> str:
    """Model-free fingerprint of EXACTLY the config that will run (= what the run writes)."""
    from dataclasses import fields

    from eval_harness.config import EvalConfig
    from eval_harness.run_spec import build_run_spec

    valid = {f.name for f in fields(EvalConfig)}
    ev = EvalConfig(**{k: v for k, v in run_cfg.items() if k in valid})
    return build_run_spec(ev)["fingerprint"]


def build_eval_config(cfg: RecoveryConfig, condition: str, bench: BenchmarkSpec, *,
                      checkpoint_dir: Optional[Path] = None, checkpoint_sha256: Optional[str] = None,
                      group_by_context: Optional[bool] = None) -> Dict[str, Any]:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; choose from {CONDITIONS}")
    compressed = condition in COMPRESSED_CONDITIONS
    llm: Dict[str, Any] = {**model_llm_kwargs(cfg), "research_config": research_config_dict(cfg, compressed=compressed)}
    if condition in RECOVERED_CONDITIONS:
        if checkpoint_dir is None:
            raise ValueError(f"condition {condition} needs a checkpoint_dir")
        llm["weight_delta"] = {"path": str(Path(checkpoint_dir).resolve()), "sha256": checkpoint_sha256, "strict": True}
    gbc = group_by_context_for(cfg) if group_by_context is None else bool(group_by_context)
    return {
        "benchmark": bench.benchmark,
        "subsets": ",".join(bench.subsets) if bench.subsets else None,
        "backend": "research",
        "model": cfg.model.name,
        "tensor_parallel_size": 1,
        "dtype": cfg.model.dtype,
        "max_model_len": cfg.model.max_model_len,
        "gpu_memory_utilization": 0.9,
        "trust_remote_code": bool(cfg.model.trust_remote_code),
        "enable_prefix_caching": True,
        "max_new_tokens": None,            # the benchmark's per-row value wins (never a global cap)
        "temperature": 0.0,
        "top_p": 1.0,
        "system_prompt": None,
        "seed": int(cfg.seed),
        "fraction": 1.0,
        "max_requests": bench.max_requests,
        "max_requests_per_subset": None,
        "request_offset": int(bench.request_offset),
        "group_by_context": gbc,
        "query_aware": False,              # the question is NEVER compressed with the context
        "output_dir": "",                  # filled in by build_cells (not fingerprinted)
        "resume": True,
        "output_dir_exact": True,
        "deterministic": True,
        "llm_kwargs": llm,
    }


def cell_dir(cfg: RecoveryConfig, bench: BenchmarkSpec, condition: str, barcode: str) -> Path:
    return results_root(cfg) / bench.benchmark / f"{condition}__{barcode}"


def build_cells(cfg: RecoveryConfig, *, checkpoint_dir: Optional[Path] = None, checkpoint_sha256: Optional[str] = None,
                conditions: Sequence[str] = DEFAULT_CONDITIONS, benchmarks: Optional[Sequence[str]] = None,
                group_by_context: Optional[bool] = None) -> List[Cell]:
    cells: List[Cell] = []
    gbc = group_by_context_for(cfg) if group_by_context is None else bool(group_by_context)
    for bench in cfg.eval.benchmarks:
        if benchmarks is not None and bench.benchmark not in benchmarks:
            continue
        built = {}
        for condition in conditions:
            d = build_eval_config(cfg, condition, bench, checkpoint_dir=checkpoint_dir,
                                  checkpoint_sha256=checkpoint_sha256, group_by_context=gbc)
            code = barcode_for_eval_config(d)
            d["output_dir"] = str(cell_dir(cfg, bench, condition, code))
            built[condition] = d
            cells.append(Cell(benchmark=bench.benchmark, condition=condition, config=d,
                              run_dir=Path(d["output_dir"]), barcode=code))
        if "compressed" in built and "compressed_recovered" in built:
            assert_same_compression(built["compressed"], built["compressed_recovered"])
        if "dense" in built and "dense_recovered" in built:
            assert_same_compression(built["dense"], built["dense_recovered"])
    return cells


def strip_for_compare(d: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(d)
    for k in _ARM_NOISE_KEYS:
        out.pop(k, None)
    llm = out.get("llm_kwargs") or {}
    llm.pop("weight_delta", None)
    out["llm_kwargs"] = llm
    return out


def assert_same_compression(baseline: Dict[str, Any], recovered: Dict[str, Any]) -> None:
    """The baseline and recovered arms must be identical except for ``weight_delta``."""
    a, b = strip_for_compare(baseline), strip_for_compare(recovered)
    if a != b:
        diff = {k: (a.get(k), b.get(k)) for k in set(a) | set(b) if a.get(k) != b.get(k)}
        raise AssertionError(f"baseline and recovered eval configs differ beyond weight_delta: {diff}")
    if "weight_delta" not in (recovered.get("llm_kwargs") or {}):
        raise AssertionError("recovered arm has no weight_delta")


def delta_config_mismatch(ckpt_dir: Path, cfg: RecoveryConfig) -> Dict[str, Any]:
    """Differences between what the delta was trained with and this evaluation's identity."""
    from .checkpoint import load_metadata

    meta = load_metadata(ckpt_dir)
    want = training_identity(cfg)
    diff: Dict[str, Any] = {}
    for section in ("model", "research_config", "prompt_shaping"):
        have = meta.get(section)
        if have != want[section]:
            if isinstance(have, dict) and isinstance(want[section], dict):
                keys = set(have) | set(want[section])
                diff[section] = {k: {"delta": have.get(k), "eval": want[section].get(k)}
                                 for k in keys if have.get(k) != want[section].get(k)}
            else:
                diff[section] = {"delta": have, "eval": want[section]}
    return diff


def assert_delta_matches_config(ckpt_dir: Path, cfg: RecoveryConfig, *, allow_mismatch: bool = False) -> Dict[str, Any]:
    diff = delta_config_mismatch(ckpt_dir, cfg)
    if diff and not allow_mismatch:
        raise ValueError(
            f"delta {ckpt_dir} was trained under a different model/compression/prompt setup than this evaluation: "
            f"{json.dumps(diff, default=str)}. Pass --allow-compression-mismatch only for deliberate transfer "
            "experiments (the override is recorded in eval_results.json)."
        )
    return diff


def find_reusable_dense(dense_dir: Path, dense_cfg: Dict[str, Any]) -> Optional[Path]:
    """An existing completed dense cell whose run_spec fingerprint equals this dense arm's."""
    spec = Path(dense_dir) / "run_spec.json"
    done = Path(dense_dir) / "DONE.json"
    if not (spec.exists() and done.exists()):
        return None
    try:
        fp = json.loads(spec.read_text()).get("fingerprint")
    except Exception:
        return None
    return Path(dense_dir) if fp == barcode_for_eval_config(dense_cfg) else None
