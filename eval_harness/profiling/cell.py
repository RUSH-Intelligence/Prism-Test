"""Cell identity and the ``perf.json`` artifact schema."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = 1
ARTIFACT = "kv_perf_cell"
PERF_FILENAME = "perf.json"


@dataclass(frozen=True)
class PerfCell:
    """One measured point. ``anchor_key`` is what makes a speedup meaningful."""

    model_key: str
    hf_model: str
    method: str                      # "none" for the full-KV anchor
    compression_ratio: float
    context_tokens: int
    attn_impl: str
    dtype: str
    decode_steps: int = 128
    warmup_repeats: int = 2
    repeats: int = 5
    kv_compressor_kwargs: Dict[str, Any] = field(default_factory=dict)
    compression_schedule: Optional[str] = None
    label: str = ""
    # Distinguishes several parameterisations of the SAME method (e.g. rarekv at
    # different (P, L)). Part of cell_id, so each lands in its own result dir.
    variant: str = ""

    @property
    def is_anchor(self) -> bool:
        return self.method in ("none", "full_kv") or self.compression_ratio == 0.0

    @property
    def anchor_key(self) -> str:
        """Cells only compare against an anchor sharing model/ctx/attn/dtype.

        attn and dtype are in the key because they are first-order timing
        variables: a flash_attention_2 cell beside sdpa siblings is measuring the
        attention kernel, not the compressor.
        """
        return f"{self.model_key}|{self.context_tokens}|{self.attn_impl}|{self.dtype}"

    @property
    def display_name(self) -> str:
        """Row label in report tables."""
        if self.is_anchor:
            return "full KV"
        base = f"{self.method}[{self.variant}]" if self.variant else self.method
        return f"{base} r{self.compression_ratio:g}"

    @property
    def cell_id(self) -> str:
        if self.is_anchor:
            tag = "full"
        else:
            stem = f"{self.method}_{self.variant}" if self.variant else self.method
            tag = f"{stem}_r{self.compression_ratio:g}"
        return f"{self.model_key}/ctx{self.context_tokens}/{tag}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(is_anchor=self.is_anchor, anchor_key=self.anchor_key,
                 cell_id=self.cell_id, display_name=self.display_name)
        return d


def _jsonable(obj):
    """Recursively coerce to JSON-native types.

    torch.dtype / torch.Size / numpy scalars leaking into an artifact make it
    unreadable later on a different box; this is the guard.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "item") and hasattr(obj, "shape") and getattr(obj, "ndim", 1) == 0:
        return obj.item()
    return str(obj)


def write_perf(run_dir: Path, payload: dict) -> Path:
    """Write perf.json, nesting into /1, /2 ... if one already exists (house rule)."""
    run_dir = Path(run_dir)
    target = run_dir
    if (run_dir / PERF_FILENAME).exists():
        i = 1
        while (run_dir / str(i) / PERF_FILENAME).exists():
            i += 1
        target = run_dir / str(i)
    target.mkdir(parents=True, exist_ok=True)
    path = target / PERF_FILENAME
    path.write_text(json.dumps(_jsonable(payload), indent=2) + "\n")
    return path


def newest_perf(cell_dir: Path) -> Optional[Path]:
    """Newest perf.json under a cell dir (re-runs nest /1, /2, ...)."""
    cands = list(Path(cell_dir).rglob(PERF_FILENAME))
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


CSV_COLUMNS: List[str] = [
    "model_key", "label", "hf_model", "context_tokens", "method", "variant", "ratio", "is_anchor",
    "attn_impl", "dtype", "anchor_key",
    "prefill_ms_median", "prefill_ms_p95", "prefill_tok_s", "prefill_host_gap_ms",
    "ttft_ms_median", "question_block_ms_median",
    "step_ms_median", "step_ms_p90", "step_ms_p99", "step_ms_cv",
    "decode_tok_s", "decode_tok_s_median_based", "repeat_spread",
    "kv_seq_post", "kv_seq_expected", "kv_bytes_post", "kv_ragged",
    "peak_alloc_decode_bytes", "peak_reserved_bytes",
    "decode_speedup", "prefill_overhead_pct", "kv_bytes_reduction_x",
    "achieved_bw_GBps", "bw_util_pct",
    "compress_ms_total", "compress_frac_of_prefill_pct",
    "eos_disabled", "repeats", "decode_steps", "gpu_name", "git_sha", "status", "n_problems",
]
