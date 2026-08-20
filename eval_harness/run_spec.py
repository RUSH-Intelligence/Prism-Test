"""Run-spec receipt: a canonical, fingerprinted record of the result-affecting
settings for a single evaluation run.

Written next to ``metrics.json`` / ``config.yaml`` on **every** run
(``runner.run``). It is the foundation for robust sweep resume: later steps
compare the ``fingerprint`` to decide skip-vs-rerun, and the ``versions`` block
(populated by a later step) folds component version numbers into that
fingerprint so a code change forces a rerun.

Design principle — CAPTURE BY DEFAULT (so the receipt grows itself):
  * Every dataclass field of the three method objects (the doors) is captured,
    INCLUDING defaults the user never set. Add a new knob to a method and it
    appears in the next run's receipt with zero extra work here.
  * Top-level run settings (``EvalConfig`` / ``ResearchConfig``) are grouped for
    readability, but any field NOT explicitly placed and NOT on the small noise
    denylist still lands in ``extra_run_settings`` — nothing is silently dropped.
  * Non-setting values (tensors, ndarrays, callables, modules, arbitrary
    objects) are dropped wherever they appear: they are runtime state, not
    chosen settings, and are not JSON-serializable.

The receipt is method-SCOPED: ridge's knobs appear only when ridge ran; another
method's knobs never leak into a ridge receipt.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

# Bump when the *shape* of this receipt changes. Excluded from the barcode (a
# format change must not force reruns), so this is record-keeping only.
SPEC_SCHEMA_VERSION = 2

# Global behavior version for shared machinery (research pipeline, RoPE handling,
# prompt assembly) that isn't owned by one component. Bump to force a rerun of
# everything after a cross-cutting behavior change. Folds into the barcode.
FRAMEWORK_VERSION = 1

# Keys present in the receipt but NOT part of the identity barcode: the barcode
# itself, the record-keeping schema version, and the code-provenance metadata
# (git SHA/dirty — informational, must not change the barcode per commit).
_NON_FINGERPRINT_KEYS = {"fingerprint", "spec_schema_version", "code"}

# The completion stamp: written DEAD LAST by the runner, so its presence proves
# predictions + metrics + receipt all finished. Resume (a later step) trusts
# this, not "does metrics.json exist".
DONE_FILENAME = "DONE.json"

# ---------------------------------------------------------------------------
# EvalConfig field routing. Denylist = pure runtime/noise (never affects the
# numbers, or changes every run). Everything else is captured; fields not
# grouped below fall through to ``extra_run_settings`` so new fields auto-appear.
# ---------------------------------------------------------------------------
_EVAL_NOISE = {
    "output_dir",            # where results land — pure path noise
    "output_dir_exact",      # ditto — only affects the folder layout
    "resume",                # control flag — whether to skip, not what to compute
    "gpu_memory_utilization",
    "tensor_parallel_size",
    "enable_prefix_caching",
    "trust_remote_code",
    "group_by_context",      # batching strategy, no numeric effect
    "llm_kwargs",            # handled specially (research_config expanded out)
}
_EVAL_MODEL = ("model", "dtype", "max_model_len", "backend")
_EVAL_BENCHMARK = ("benchmark", "subsets", "max_requests", "max_requests_per_subset",
                   "request_offset", "fraction", "system_prompt")
_EVAL_GENERATION = ("max_new_tokens", "temperature", "top_p", "seed",
                    "deterministic", "query_aware")

_SKIP = object()  # sentinel: "this value is not a recordable setting"


# ---------------------------------------------------------------------------
# Value normalization
# ---------------------------------------------------------------------------
def _jsonable(v: Any) -> Any:
    """Convert a setting value to a stable, JSON-serializable form, or return
    ``_SKIP`` for anything that is runtime state rather than a chosen setting."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, Enum):
        return v.value if isinstance(v.value, (str, int, float)) else v.name
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, (set, frozenset)):
        items = [_jsonable(x) for x in v]
        items = [x for x in items if x is not _SKIP]
        try:
            return sorted(items)
        except TypeError:
            return sorted(map(str, items))
    if isinstance(v, (list, tuple)):
        out = []
        for x in v:
            jx = _jsonable(x)
            if jx is _SKIP:      # a list of tensors/objects -> not a setting
                return _SKIP
            out.append(jx)
        return out
    if isinstance(v, dict):
        out = {}
        for k, val in v.items():
            jv = _jsonable(val)
            if jv is not _SKIP:
                out[str(k)] = jv
        return out
    # 0-d numpy/torch scalar -> plain python number.
    if hasattr(v, "item") and getattr(v, "ndim", None) == 0:
        try:
            return v.item()
        except Exception:
            return _SKIP
    # tensors / ndarrays / modules / callables / arbitrary objects: not settings.
    return _SKIP


def _dump_dataclass(obj: Any) -> dict:
    """Every field of a dataclass instance (defaults included), normalized and
    with non-setting values dropped. Non-dataclasses yield ``{}``."""
    if not dataclasses.is_dataclass(obj):
        return {}
    out: dict = {}
    for f in dataclasses.fields(obj):
        jv = _jsonable(getattr(obj, f.name, None))
        if jv is not _SKIP:
            out[f.name] = jv
    return dict(sorted(out.items()))


def _subsets_list(subsets: Any) -> Optional[list]:
    """Comma string -> sorted list (order-independent identity); passthrough
    for a list; ``None`` stays ``None`` (= the benchmark's own default set)."""
    if subsets is None:
        return None
    if isinstance(subsets, str):
        parts = [s.strip() for s in subsets.split(",") if s.strip()]
        return sorted(parts) or None
    if isinstance(subsets, (list, tuple)):
        return sorted(str(s) for s in subsets) or None
    return _jsonable(subsets)


# ---------------------------------------------------------------------------
# The three doors (research backend only)
# ---------------------------------------------------------------------------
def _research_cfg_from(config: Any):
    """Reconstruct the ResearchConfig for a research run (model-free), or None.

    Mirrors ``runner._setup_adapter``: a ``research`` backend always builds a
    ResearchConfig (from ``llm_kwargs['research_config']`` if present, else
    defaults); other backends have no doors.
    """
    if getattr(config, "backend", None) != "research":
        return None
    from .research_adapter import ResearchConfig  # lazy: avoids torch at import
    research_kw = (getattr(config, "llm_kwargs", None) or {}).get("research_config") or {}
    return ResearchConfig(**research_kw)


def _component_version(obj: Any) -> int:
    """Behavior version declared on a component's class (default 1)."""
    return int(getattr(type(obj), "VERSION", 1))


def _dump_methods(config: Any) -> tuple[dict, dict, dict, dict]:
    """Return (doors, pipeline, prompt_shaping, versions), built model-free from
    the config via the SAME door builders the live run uses. ``versions`` is
    scoped to the ACTIVE doors only. Empty for non-research runs."""
    doors = {"positional": None, "attention": None, "kv_compressor": None}
    versions: dict = {}
    cfg = _research_cfg_from(config)
    if cfg is None:
        return doors, {}, {}, versions

    from .research_adapter import build_doors  # lazy: avoids torch at import
    pos, att, kv = build_doors(cfg)

    if pos is not None:
        versions["positional_method"] = {cfg.positional_method: _component_version(pos)}
    if att is not None:
        versions["attention_method"] = {cfg.attention_method: _component_version(att)}
    if kv is not None:
        versions["kv_compressor"] = {cfg.kv_compressor: _component_version(kv)}

    if pos is not None:
        doors["positional"] = {"name": cfg.positional_method, "knobs": _dump_dataclass(pos)}
    if att is not None:
        doors["attention"] = {"name": cfg.attention_method,
                              "phase": cfg.attention_phase,
                              "knobs": _dump_dataclass(att)}
    if kv is not None:
        doors["kv_compressor"] = {"name": cfg.kv_compressor,
                                  # kept at door level too: property-based
                                  # compressors carry no ratio dataclass field.
                                  "compression_ratio": cfg.compression_ratio,
                                  "knobs": _dump_dataclass(kv)}

    pipeline = {
        "compression_interval": cfg.compression_interval,
        "target_size": cfg.target_size,
        "hidden_states_buffer_size": cfg.hidden_states_buffer_size,
        "prefill_chunk_size": cfg.prefill_chunk_size,
        "max_context_length": cfg.max_context_length,
    }
    prompt_shaping = {
        "use_chat_template": cfg.use_chat_template,
        "strip_auto_system_block": cfg.strip_auto_system_block,
        "middle_truncation": cfg.middle_truncation,
    }
    return doors, pipeline, prompt_shaping, versions


def _benchmark_version(name: Any) -> Optional[int]:
    """Behavior version of a benchmark, read from its registered class (no
    instantiation, no dataset load). None if the name isn't registered."""
    if not name:
        return None
    try:
        from .benchmarks.registry import get_registered_benchmarks
        cls = get_registered_benchmarks().get(str(name).strip().lower())
        return int(getattr(cls, "VERSION", 1)) if cls is not None else None
    except Exception:
        return None


# Code provenance is process-stable; compute the git calls at most once.
_GIT_META: Any = "unset"


def _git_metadata() -> Optional[dict]:
    """Record-only: the repo commit + whether the tree had uncommitted changes.
    NOT part of the barcode — an audit trail for the 'forgot to bump' case."""
    global _GIT_META
    if _GIT_META != "unset":
        return _GIT_META
    _GIT_META = None
    try:
        import subprocess
        root = Path(__file__).resolve().parent.parent
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(root),
                                capture_output=True, text=True, timeout=5)
        if commit.returncode != 0:
            return _GIT_META
        status = subprocess.run(["git", "status", "--porcelain"], cwd=str(root),
                                capture_output=True, text=True, timeout=5)
        _GIT_META = {"git_commit": commit.stdout.strip(),
                     "git_dirty": bool(status.stdout.strip())}
    except Exception:
        _GIT_META = None
    return _GIT_META


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------
def fingerprint(spec: dict) -> str:
    """Short stable barcode of the receipt's identity fields (everything except
    the non-identity keys: the fingerprint, the schema version, code metadata)."""
    payload = {k: v for k, v in spec.items() if k not in _NON_FINGERPRINT_KEYS}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_run_spec(config: Any) -> dict:
    """Build the canonical, fingerprinted receipt for one run.

    Pure function of ``config`` (an ``EvalConfig``): the three doors are rebuilt
    model-free via the adapter's own builders, so the barcode computed here from
    a config is byte-identical to the one the live run writes for that config.
    This is what lets the sweep decide skip-vs-rerun WITHOUT loading a model.
    """
    llm = dict(getattr(config, "llm_kwargs", None) or {})
    llm.pop("research_config", None)          # expanded into method/pipeline below
    load_flags = _jsonable(llm)
    if load_flags is _SKIP:
        load_flags = {}

    model = {
        "name": getattr(config, "model", None),
        "dtype": getattr(config, "dtype", None),
        "max_model_len": getattr(config, "max_model_len", None),
        "backend": getattr(config, "backend", None),
        "load_flags": load_flags,
    }
    benchmark = {
        "name": getattr(config, "benchmark", None),
        "subsets": _subsets_list(getattr(config, "subsets", None)),
        "max_requests": getattr(config, "max_requests", None),
        "max_requests_per_subset": _jsonable(getattr(config, "max_requests_per_subset", None)),
        "request_offset": getattr(config, "request_offset", None),
        "fraction": getattr(config, "fraction", None),
        "system_prompt": getattr(config, "system_prompt", None),
    }
    generation = {k: _jsonable(getattr(config, k, None)) for k in _EVAL_GENERATION}

    doors, pipeline, prompt_shaping, method_versions = _dump_methods(config)
    if prompt_shaping:
        benchmark["prompt_shaping"] = prompt_shaping

    # Behavior versions (folded into the barcode): the shared framework, the
    # benchmark, and the active doors — scoped, so only what ran appears.
    versions: dict = {"framework": FRAMEWORK_VERSION}
    bench_v = _benchmark_version(getattr(config, "benchmark", None))
    if bench_v is not None:
        versions["benchmark"] = {config.benchmark: bench_v}
    versions.update(method_versions)

    # Catch-all: any EvalConfig field not grouped above and not pure noise, so a
    # newly-added config field is never silently missed.
    grouped = set(_EVAL_MODEL) | set(_EVAL_BENCHMARK) | set(_EVAL_GENERATION)
    extra: dict = {}
    if dataclasses.is_dataclass(config):
        for f in dataclasses.fields(config):
            if f.name in grouped or f.name in _EVAL_NOISE:
                continue
            jv = _jsonable(getattr(config, f.name, None))
            if jv is not _SKIP:
                extra[f.name] = jv

    spec: dict = {
        "spec_schema_version": SPEC_SCHEMA_VERSION,   # not fingerprinted
        "fingerprint": None,
        "versions": versions,    # fingerprinted: a bump forces a rerun
        "model": model,
        "benchmark": benchmark,
        "generation": generation,
        "method": doors,
    }
    if pipeline:
        spec["pipeline"] = pipeline
    if extra:
        spec["extra_run_settings"] = dict(sorted(extra.items()))

    spec["fingerprint"] = fingerprint(spec)

    # Code provenance: recorded AFTER fingerprinting and excluded from it, so it
    # never affects the barcode — a pure audit trail.
    code = _git_metadata()
    if code:
        spec["code"] = code
    return spec


def write_run_spec(run_dir: Path, config: Any) -> Path:
    """Build and write ``run_spec.json`` into ``run_dir``; return its path."""
    spec = build_run_spec(config)
    path = Path(run_dir) / "run_spec.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(spec, handle, indent=2)
    return path


# ---------------------------------------------------------------------------
# Completion stamp ("finished" marker + sample counts)
# ---------------------------------------------------------------------------
def _intended_count(max_requests: Optional[int],
                    max_requests_per_subset: Optional[dict],
                    subset_names: Optional[list],
                    loaded_before_cap: Optional[int]) -> Optional[int]:
    """How many samples the config *asked for* (max_requests caps PER subset).

    Returns None only when it genuinely can't be inferred. When uncapped, the
    intent is "everything that loaded" -> ``loaded_before_cap``.
    """
    if max_requests_per_subset:
        if subset_names:
            return sum(int(max_requests_per_subset.get(s, max_requests or 0)) for s in subset_names)
        return sum(int(v) for v in max_requests_per_subset.values())
    if max_requests is not None:
        n = len(subset_names) if subset_names else 1
        return int(max_requests) * n
    return loaded_before_cap  # uncapped: intended == everything available


def build_done_marker(*, fingerprint: str, actual_samples: int,
                      overall_score: Any = None,
                      max_requests: Optional[int] = None,
                      max_requests_per_subset: Optional[dict] = None,
                      requested_subsets: Optional[list] = None,
                      per_subset_actual: Optional[dict] = None,
                      loaded_before_cap: Optional[int] = None,
                      finished_at: Optional[str] = None) -> dict:
    """Build the completion stamp. Presence of this file == the run finished;
    the counts let resume (and you) sanity-check *how much* actually ran."""
    subset_names = requested_subsets or (
        sorted(per_subset_actual) if per_subset_actual else None)
    intended = _intended_count(max_requests, max_requests_per_subset,
                               subset_names, loaded_before_cap)
    per_subset = None
    if per_subset_actual:
        per_subset = {str(k): int(v) for k, v in sorted(per_subset_actual.items())}
    return {
        "complete": True,
        "spec_schema_version": SPEC_SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "finished_at": finished_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "samples": {
            "actual": int(actual_samples),
            "intended": intended,
            # True == finished but ran fewer than asked for (e.g. a subset came
            # up short). Informational — never used to force a rerun on its own.
            "short": bool(intended is not None and int(actual_samples) < intended),
            "requested_max_requests": max_requests,
            "max_requests_per_subset": max_requests_per_subset or None,
            "requested_subsets": requested_subsets,
            "loaded_before_cap": loaded_before_cap,
            "per_subset_actual": per_subset,
        },
        "overall_score": _jsonable(overall_score) if overall_score is not None else None,
    }


def write_done_marker(run_dir: Path, marker: dict) -> Path:
    """Write ``DONE.json`` into ``run_dir`` (call LAST); return its path."""
    path = Path(run_dir) / DONE_FILENAME
    with path.open("w", encoding="utf-8") as handle:
        json.dump(marker, handle, indent=2)
    return path
