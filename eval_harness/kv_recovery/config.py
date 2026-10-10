"""``RecoveryConfig``: the single configuration object of a hidden-state KV-recovery run.

Rules
-----
* **Strict.** Unknown keys at ANY nesting level raise (``eval_harness.cli`` silently drops
  unknown top-level keys; a research config must fail loudly instead).
* **One compression block.** ``kv_compression`` feeds (a) the student's compressor,
  (b) the delta-checkpoint metadata and (c) the ``compressed`` / ``compressed_recovered``
  evaluation arms, all through :func:`research_config_dict`, so they cannot drift.
* **Overrides.** ``--set a.b.c=value`` (value parsed as YAML) plus named CLI shortcuts
  (see ``scripts/train_kv_recovery.py``); precedence file < ``--set`` < shortcuts.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

import re

import yaml

_INT_RE = re.compile(r"^[+-]?\d+$")
_FLOAT_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")

LONGBENCH_16 = [
    "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique",
    "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum",
    "passage_count", "passage_retrieval_en", "lcc", "repobench-p",
]

LAYER_STRATEGIES = ("last_n", "explicit", "all", "from_first_trainable")
POSITION_STRATEGIES = ("all", "recent", "first_k", "post_eviction")
LOSSES = ("normalized_mse", "normalized_mse_elementwise", "cosine", "relative_mse")
TRAINABLE_STRATEGIES = ("last_n_blocks", "blocks", "attention_projections", "mlp", "norms", "full")
DATA_FORMATS = ("raw", "chat")
SUFFIX_MODES = ("continuation", "recall")
TEACHER_MODES = ("online", "offline")
SEGMENT_MODES = ("auto", "block", "token_by_token")
# ``trainable.layers`` value that selects the top-k compression-sensitive layers (kv_recovery/sensitivity.py).
LAYER_SELECTOR_SENSITIVITY = "sensitivity"
LAYER_SELECTED_STRATEGIES = ("attention_projections", "blocks", "mlp", "norms")   # strategies that honour trainable.layers
SENSITIVITY_SPLITS = ("val", "train")
SENSITIVITY_AGGREGATES = ("mean", "median")


# ---------------------------------------------------------------------------
# Leaf / nested dataclasses
# ---------------------------------------------------------------------------
@dataclass
class ModelCfg:
    name: str = "mistralai/Ministral-3-3B-Instruct-2512"
    # Recorded in metadata and checked by the delta loader; the HF loader itself
    # resolves the cached snapshot (eval_harness has no revision plumbing).
    revision: Optional[str] = None
    dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    trust_remote_code: bool = True
    max_model_len: int = 131072
    dequantize_fp8: bool = False


@dataclass
class KVCompressionCfg:
    kv_compressor: str = "knorm"
    # Fraction of the context KV cache PRUNED per head (repo convention):
    # kept = int(T * (1 - compression_ratio)). The spec's "budget_ratio" is 1 - this.
    compression_ratio: float = 0.75
    kv_compressor_kwargs: Dict[str, Any] = field(default_factory=dict)
    compression_schedule: Optional[Any] = None   # None = the compressor's default (post_prefill)
    prefill_chunk_size: Optional[int] = None     # None = single-pass prefill (the eval default)


@dataclass
class DataCfg:
    path: str = "data/kv_recovery/pg19_train.jsonl"
    val_path: Optional[str] = "data/kv_recovery/pg19_val.jsonl"
    max_length: int = 16384          # context + suffix tokens
    suffix_length: int = 512         # aligned continuation tokens (the post-compression segment)
    num_train_examples: int = 256
    num_val_examples: int = 16
    seed: int = 42                   # example selection / shuffling
    format: str = "raw"              # raw: bos + text | chat: through ResearchGenerationPipeline.preprocess (text rows)
    suffix_mode: str = "continuation"  # continuation | recall (suffix = verbatim span from earlier in the window)
    strip_auto_system_block: bool = True   # chat format and qa rows
    # Rows with ``kind: qa`` (benchmark-shaped: context / question / answer_prefix / answer, e.g. RULER rows outside the
    # evaluated pool) are shaped exactly as the evaluation shapes them and aligned on the question (+ answer prefix)
    # [+ gold answer, teacher-forced] region. ``max_context_tokens`` skips qa rows whose context is longer (None = no cap).
    qa_region: str = "question_answer"     # question_answer | question
    max_context_tokens: Optional[int] = None


@dataclass
class LayersCfg:
    strategy: str = "from_first_trainable"   # last_n | explicit | all | from_first_trainable
    n: int = 4
    indices: Optional[List[int]] = None


@dataclass
class PositionsCfg:
    strategy: str = "all"   # all | recent | first_k | post_eviction
    n: int = 128


@dataclass
class AlignmentCfg:
    layers: LayersCfg = field(default_factory=LayersCfg)
    include_final_norm: bool = True
    positions: PositionsCfg = field(default_factory=PositionsCfg)
    loss: str = "normalized_mse"
    layer_weights: Optional[List[float]] = None
    allow_dead_terms: bool = False
    _NESTED: ClassVar[Dict[str, type]] = {"layers": LayersCfg, "positions": PositionsCfg}


@dataclass
class LossCfg:
    hidden_weight: float = 1.0
    kl_weight: float = 0.0
    temperature: float = 1.0


@dataclass
class SensitivityCfg:
    """``trainable.layers: sensitivity`` — select the top-k layers whose hidden states the compressor
    perturbs most: ``E_l = ||H_l^dense - H_l^comp||_F / (||H_l^dense||_F + eps)`` over the suffix
    tokens, aggregated over held-out calibration windows (``kv_recovery/sensitivity.py``)."""
    top_k: int = 4                   # layers selected among the eligible ones (strategy-dependent pool)
    num_examples: int = 8            # calibration windows, disjoint from the training AND validation windows
    split: str = "val"               # val (rows of data.val_path) | train (rows of data.path)
    positions: PositionsCfg = field(default_factory=PositionsCfg)   # suffix positions forming H_l (default: all)
    eps: float = 1e-6
    aggregate: str = "mean"          # mean | median of the per-window E_l
    _NESTED: ClassVar[Dict[str, type]] = {"positions": PositionsCfg}


@dataclass
class TrainableCfg:
    strategy: str = "last_n_blocks"   # last_n_blocks | blocks | attention_projections | mlp | norms | full
    n: int = 1                        # last_n_blocks only
    modules: List[str] = field(default_factory=lambda: ["k_proj", "v_proj"])
    # For attention_projections / blocks / mlp / norms: "all" | "last_n:<k>" | explicit list of layer
    # indices | "sensitivity" (top-k compression-sensitive layers, configured by ``sensitivity``).
    layers: Any = "all"
    include_embeddings: bool = False
    sensitivity: SensitivityCfg = field(default_factory=SensitivityCfg)
    _NESTED: ClassVar[Dict[str, type]] = {"sensitivity": SensitivityCfg}


@dataclass
class InstabilityCfg:
    grad_norm_factor: float = 10.0
    consecutive_steps: int = 2
    warmup_fraction: float = 0.05
    restart: bool = True


@dataclass
class OptimCfg:
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8
    grad_clip: float = 1.0
    grad_accum: int = 4
    epochs: int = 1
    max_steps: Optional[int] = None
    warmup_fraction: float = 0.0
    val_every_steps: int = 8
    shuffle_seed: int = 0
    master_weights_fp32: bool = True
    deterministic_backward: bool = False
    instability: InstabilityCfg = field(default_factory=InstabilityCfg)
    _NESTED: ClassVar[Dict[str, type]] = {"instability": InstabilityCfg}


@dataclass
class TeacherCfg:
    mode: str = "online"             # online | offline
    states_dir: Optional[str] = None
    store_dtype: str = "bfloat16"


@dataclass
class StudentCfg:
    prefill_grad: bool = False
    segment_mode: str = "auto"       # auto | block | token_by_token


@dataclass
class BenchmarkSpec:
    benchmark: str = "ruler16k"
    subsets: Optional[List[str]] = None
    max_requests: Optional[int] = None
    request_offset: int = 0
    walltime: str = "4:00:00"


@dataclass
class BootstrapCfg:
    n_resamples: int = 10000
    seed: int = 0
    alpha: float = 0.05


@dataclass
class SlurmCfg:
    partition: str = "commons"
    account: str = "as143"
    gres: str = "gpu:h200:1"
    cpus: int = 8
    mem_per_cpu: str = "32GB"
    exclude: str = "bg3u24g1"
    train_time: str = "6:00:00"
    log_dir: str = "/scratch/sj157/kvrec_logs"


@dataclass
class EvalCfg:
    benchmarks: List[BenchmarkSpec] = field(default_factory=lambda: [
        BenchmarkSpec("ruler16k", None, 100, 0, "2:30:00"),
        BenchmarkSpec("ruler32k", None, 100, 0, "3:30:00"),
        BenchmarkSpec("longbench", list(LONGBENCH_16), 200, 0, "10:00:00"),
    ])
    use_chat_template: bool = True
    strip_auto_system_block: bool = True
    group_by_context: Optional[bool] = None   # None = False for hybrid (linear-attention) models, else True
    results_root: Optional[str] = None        # default: <output.root>/eval/<model_slug>
    dense_dir: Dict[str, str] = field(default_factory=dict)  # benchmark -> existing dense cell to reuse
    bootstrap: BootstrapCfg = field(default_factory=BootstrapCfg)
    slurm: SlurmCfg = field(default_factory=SlurmCfg)
    _NESTED: ClassVar[Dict[str, type]] = {"bootstrap": BootstrapCfg, "slurm": SlurmCfg}
    _NESTED_LIST: ClassVar[Dict[str, type]] = {"benchmarks": BenchmarkSpec}


@dataclass
class OutputCfg:
    root: str = "outputs/kv_recovery"
    overwrite: bool = False


@dataclass
class RecoveryConfig:
    run_name: str = "kv_recovery_run"
    seed: int = 42
    deterministic: bool = True
    model: ModelCfg = field(default_factory=ModelCfg)
    kv_compression: KVCompressionCfg = field(default_factory=KVCompressionCfg)
    data: DataCfg = field(default_factory=DataCfg)
    alignment: AlignmentCfg = field(default_factory=AlignmentCfg)
    loss: LossCfg = field(default_factory=LossCfg)
    trainable: TrainableCfg = field(default_factory=TrainableCfg)
    optim: OptimCfg = field(default_factory=OptimCfg)
    teacher: TeacherCfg = field(default_factory=TeacherCfg)
    student: StudentCfg = field(default_factory=StudentCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)
    output: OutputCfg = field(default_factory=OutputCfg)
    _NESTED: ClassVar[Dict[str, type]] = {
        "model": ModelCfg, "kv_compression": KVCompressionCfg, "data": DataCfg,
        "alignment": AlignmentCfg, "loss": LossCfg, "trainable": TrainableCfg,
        "optim": OptimCfg, "teacher": TeacherCfg, "student": StudentCfg,
        "eval": EvalCfg, "output": OutputCfg,
    }

    # -- construction ------------------------------------------------------
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RecoveryConfig":
        cfg = _build(cls, data, "<root>")
        cfg.validate()
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        return _plain(dataclasses.asdict(self))

    @property
    def run_dir(self) -> Path:
        return Path(self.output.root) / self.run_name

    def digest(self) -> str:
        """Short content hash of the whole config (canonical JSON)."""
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    # -- validation --------------------------------------------------------
    def validate(self) -> None:
        kv = self.kv_compression
        if not (0.0 <= float(kv.compression_ratio) < 1.0):
            raise ValueError(f"kv_compression.compression_ratio must be in [0, 1), got {kv.compression_ratio}")
        if kv.prefill_chunk_size is not None and int(kv.prefill_chunk_size) <= 0:
            raise ValueError("kv_compression.prefill_chunk_size must be a positive int or null")
        d = self.data
        if d.suffix_length <= 0 or d.suffix_length >= d.max_length:
            raise ValueError(f"data.suffix_length must satisfy 0 < suffix_length < max_length "
                             f"(got {d.suffix_length} / {d.max_length})")
        if d.format not in DATA_FORMATS:
            raise ValueError(f"data.format must be one of {DATA_FORMATS}, got {d.format!r}")
        if d.suffix_mode not in SUFFIX_MODES:
            raise ValueError(f"data.suffix_mode must be one of {SUFFIX_MODES}, got {d.suffix_mode!r}")
        if d.qa_region not in ("question_answer", "question"):
            raise ValueError(f"data.qa_region must be 'question_answer' or 'question', got {d.qa_region!r}")
        if d.max_context_tokens is not None and int(d.max_context_tokens) <= 0:
            raise ValueError("data.max_context_tokens must be a positive int or null")
        if d.num_train_examples <= 0 or d.num_val_examples < 0:
            raise ValueError("data.num_train_examples must be > 0 and data.num_val_examples >= 0")
        if d.val_path is not None and d.val_path == d.path:
            raise ValueError("data.val_path must differ from data.path (validation windows must be disjoint)")
        a = self.alignment
        if a.layers.strategy not in LAYER_STRATEGIES:
            raise ValueError(f"alignment.layers.strategy must be one of {LAYER_STRATEGIES}")
        if a.layers.strategy == "explicit" and not a.layers.indices:
            raise ValueError("alignment.layers.indices is required for strategy 'explicit'")
        if a.layers.strategy == "last_n" and a.layers.n <= 0:
            raise ValueError("alignment.layers.n must be > 0 for strategy 'last_n'")
        if a.positions.strategy not in POSITION_STRATEGIES:
            raise ValueError(f"alignment.positions.strategy must be one of {POSITION_STRATEGIES}")
        if a.positions.strategy in ("recent", "first_k") and a.positions.n <= 0:
            raise ValueError("alignment.positions.n must be > 0 for 'recent' / 'first_k'")
        if a.loss not in LOSSES:
            raise ValueError(f"alignment.loss must be one of {LOSSES}, got {a.loss!r}")
        t = self.trainable
        if t.strategy not in TRAINABLE_STRATEGIES:
            raise ValueError(f"trainable.strategy must be one of {TRAINABLE_STRATEGIES}")
        if t.strategy == "last_n_blocks" and t.n <= 0:
            raise ValueError("trainable.n must be > 0 for 'last_n_blocks'")
        if t.strategy == "attention_projections" and not t.modules:
            raise ValueError("trainable.modules must list at least one projection")
        _validate_layer_selector(t.layers)
        if t.layers == LAYER_SELECTOR_SENSITIVITY:
            if t.strategy not in LAYER_SELECTED_STRATEGIES:
                raise ValueError(f"trainable.layers='sensitivity' needs trainable.strategy in {LAYER_SELECTED_STRATEGIES} "
                                 f"(got {t.strategy!r}; last_n_blocks / full do not take a layer selector)")
            s = t.sensitivity
            if s.top_k <= 0 or s.num_examples <= 0:
                raise ValueError("trainable.sensitivity.top_k and .num_examples must be > 0")
            if s.split not in SENSITIVITY_SPLITS:
                raise ValueError(f"trainable.sensitivity.split must be one of {SENSITIVITY_SPLITS}, got {s.split!r}")
            if s.aggregate not in SENSITIVITY_AGGREGATES:
                raise ValueError(f"trainable.sensitivity.aggregate must be one of {SENSITIVITY_AGGREGATES}")
            if s.eps < 0:
                raise ValueError("trainable.sensitivity.eps must be >= 0")
            if s.positions.strategy not in POSITION_STRATEGIES:
                raise ValueError(f"trainable.sensitivity.positions.strategy must be one of {POSITION_STRATEGIES}")
            if s.positions.strategy in ("recent", "first_k") and s.positions.n <= 0:
                raise ValueError("trainable.sensitivity.positions.n must be > 0 for 'recent' / 'first_k'")
            if s.split == "val" and d.val_path is None:
                raise ValueError("trainable.sensitivity.split='val' needs data.val_path")
            if self.teacher.mode != "online":
                raise ValueError("trainable.layers='sensitivity' needs teacher.mode == 'online': the dense pass over EVERY "
                                 "layer is measured before the aligned layers are known (offline stores hold only those)")
        lo = self.loss
        if lo.hidden_weight < 0 or lo.kl_weight < 0 or lo.temperature <= 0:
            raise ValueError("loss weights must be >= 0 and loss.temperature > 0")
        if lo.hidden_weight == 0 and lo.kl_weight == 0:
            raise ValueError("at least one of loss.hidden_weight / loss.kl_weight must be > 0")
        o = self.optim
        if o.grad_accum <= 0 or o.epochs <= 0 or o.learning_rate <= 0 or o.val_every_steps <= 0:
            raise ValueError("optim.grad_accum / epochs / learning_rate / val_every_steps must be > 0")
        if len(o.betas) != 2:
            raise ValueError("optim.betas must have two entries")
        if self.teacher.mode not in TEACHER_MODES:
            raise ValueError(f"teacher.mode must be one of {TEACHER_MODES}")
        if self.teacher.mode == "offline" and lo.kl_weight > 0:
            raise ValueError("loss.kl_weight > 0 requires teacher.mode == 'online' (logits are not stored offline)")
        if self.student.segment_mode not in SEGMENT_MODES:
            raise ValueError(f"student.segment_mode must be one of {SEGMENT_MODES}")
        if not self.eval.benchmarks:
            raise ValueError("eval.benchmarks must list at least one benchmark")
        for b in self.eval.benchmarks:
            if b.max_requests is not None and b.max_requests <= 0:
                raise ValueError(f"eval.benchmarks[{b.benchmark}].max_requests must be > 0 or null")
            if b.request_offset < 0:
                raise ValueError("eval.benchmarks[].request_offset must be >= 0")
        if not (0.0 < self.eval.bootstrap.alpha < 1.0):
            raise ValueError("eval.bootstrap.alpha must be in (0, 1)")


# ---------------------------------------------------------------------------
# Strict nested construction
# ---------------------------------------------------------------------------
def _build(cls, data: Any, path: str):
    if data is None:
        data = {}
    if is_dataclass(data) and not isinstance(data, type):
        return data
    if not isinstance(data, dict):
        raise TypeError(f"{path}: expected a mapping for {cls.__name__}, got {type(data).__name__}")
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise KeyError(f"{path}: unknown key(s) {unknown}; allowed keys: {sorted(known)}")
    nested = getattr(cls, "_NESTED", {})
    nested_list = getattr(cls, "_NESTED_LIST", {})
    kwargs: Dict[str, Any] = {}
    for name, value in data.items():
        if name in nested:
            kwargs[name] = _build(nested[name], value, f"{path}.{name}")
        elif name in nested_list:
            if value is None:
                kwargs[name] = []
            elif not isinstance(value, list):
                raise TypeError(f"{path}.{name}: expected a list")
            else:
                kwargs[name] = [_build(nested_list[name], v, f"{path}.{name}[{i}]") for i, v in enumerate(value)]
        else:
            kwargs[name] = copy.deepcopy(value)
    return cls(**kwargs)


def _plain(obj: Any) -> Any:
    """YAML/JSON-safe plain structure (tuples -> lists)."""
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj


def _validate_layer_selector(sel: Any) -> None:
    if isinstance(sel, str):
        if sel in ("all", LAYER_SELECTOR_SENSITIVITY):
            return
        if sel.startswith("last_n:"):
            try:
                k = int(sel.split(":", 1)[1])
            except ValueError as exc:
                raise ValueError(f"trainable.layers: bad selector {sel!r}") from exc
            if k <= 0:
                raise ValueError("trainable.layers 'last_n:<k>' needs k > 0")
            return
        raise ValueError(f"trainable.layers must be 'all', 'last_n:<k>', 'sensitivity' or a list of ints, got {sel!r}")
    if isinstance(sel, (list, tuple)):
        if not all(isinstance(i, int) for i in sel):
            raise ValueError("trainable.layers list must contain ints")
        return
    raise ValueError(f"trainable.layers must be 'all', 'last_n:<k>', 'sensitivity' or a list of ints, got {sel!r}")


# ---------------------------------------------------------------------------
# YAML + overrides
# ---------------------------------------------------------------------------
def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p.resolve()}")
    with p.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def set_dotted(d: Dict[str, Any], key: str, value: Any) -> None:
    """Assign ``value`` at the dotted ``key`` (creating intermediate mappings)."""
    parts = [p for p in key.split(".") if p]
    if not parts:
        raise ValueError("empty override key")
    cur = d
    for part in parts[:-1]:
        nxt = cur.get(part)
        if nxt is None:
            nxt = {}
            cur[part] = nxt
        elif not isinstance(nxt, dict):
            raise ValueError(f"override {key!r}: {part!r} is not a mapping")
        cur = nxt
    cur[parts[-1]] = value


def parse_override(text: str) -> tuple[str, Any]:
    """``'a.b.c=value'`` -> (``'a.b.c'``, YAML-parsed value)."""
    if "=" not in text:
        raise ValueError(f"override must look like key=value, got {text!r}")
    key, raw = text.split("=", 1)
    key = key.strip()
    raw = raw.strip()
    if not key:
        raise ValueError(f"override must look like key=value, got {text!r}")
    value = yaml.safe_load(raw) if raw != "" else None
    if isinstance(value, str):
        # YAML 1.1 reads "1e-5" as a string (it wants "1.0e-5"); accept plain numerics.
        if _INT_RE.match(value):
            return key, int(value)
        if _FLOAT_RE.match(value):
            return key, float(value)
    return key, value


def apply_overrides(data: Dict[str, Any], overrides: Optional[List[str]] = None,
                    shortcuts: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Apply ``--set`` overrides, then named shortcuts (``{dotted_key: value}``)."""
    out = copy.deepcopy(data)
    for item in overrides or []:
        key, value = parse_override(item)
        set_dotted(out, key, value)
    for key, value in (shortcuts or {}).items():
        set_dotted(out, key, value)
    return out


def load_config(path: str | Path, overrides: Optional[List[str]] = None,
                shortcuts: Optional[Dict[str, Any]] = None) -> RecoveryConfig:
    return RecoveryConfig.from_dict(apply_overrides(load_yaml(path), overrides, shortcuts))


def kv_budget_to_ratio(budget_ratio: float) -> float:
    """The spec's ``budget_ratio`` (fraction KEPT) -> the repo's ``compression_ratio`` (fraction PRUNED)."""
    b = float(budget_ratio)
    if not (0.0 < b <= 1.0):
        raise ValueError(f"kv budget ratio must be in (0, 1], got {b}")
    return 1.0 - b


# ---------------------------------------------------------------------------
# The ONE shared compression block
# ---------------------------------------------------------------------------
def research_config_dict(cfg: RecoveryConfig, *, compressed: bool) -> Dict[str, Any]:
    """The ``llm_kwargs.research_config`` dict (= ``ResearchConfig`` kwargs) for the
    student (``compressed=True``), the ``compressed`` / ``compressed_recovered`` arms
    (``compressed=True``) and the ``dense`` arm (``compressed=False``).

    Only door 3 is ever active (``attention_method`` / ``positional_method`` = none);
    prompt-shaping flags come from ``eval`` so every arm shares them.
    """
    kv = cfg.kv_compression
    return {
        "positional_method": "none",
        "positional_method_kwargs": None,
        "attention_method": "none",
        "attention_method_kwargs": None,
        "attention_phase": "both",
        "kv_compressor": kv.kv_compressor if compressed else "none",
        "kv_compressor_kwargs": copy.deepcopy(kv.kv_compressor_kwargs) if compressed else None,
        "compression_ratio": float(kv.compression_ratio) if compressed else 0.0,
        "compression_schedule": copy.deepcopy(kv.compression_schedule) if compressed else None,
        "prefill_chunk_size": kv.prefill_chunk_size,
        "max_context_length": None,
        "log_cache_seq_len": True,
        "use_chat_template": bool(cfg.eval.use_chat_template),
        "strip_auto_system_block": bool(cfg.eval.strip_auto_system_block),
        "middle_truncation": False,
    }


def build_research_config(cfg: RecoveryConfig, *, compressed: bool):
    """``ResearchConfig`` built from :func:`research_config_dict` (lazy import: torch-heavy)."""
    from eval_harness.research_adapter import ResearchConfig

    return ResearchConfig(**research_config_dict(cfg, compressed=compressed))


def model_llm_kwargs(cfg: RecoveryConfig) -> Dict[str, Any]:
    """The non-``research_config`` part of ``llm_kwargs`` (= ``run_spec`` ``load_flags``)."""
    out: Dict[str, Any] = {"attn_implementation": cfg.model.attn_implementation}
    if cfg.model.dequantize_fp8:
        out["dequantize_fp8"] = True
    return out


def training_identity(cfg: RecoveryConfig) -> Dict[str, Any]:
    """What a delta checkpoint must match at evaluation time: model load flags, the
    compression block and the prompt shaping (recorded in the delta metadata)."""
    return {
        "model": {
            "name": cfg.model.name,
            "revision": cfg.model.revision,
            "dtype": cfg.model.dtype,
            "attn_implementation": cfg.model.attn_implementation,
            "dequantize_fp8": bool(cfg.model.dequantize_fp8),
            "trust_remote_code": bool(cfg.model.trust_remote_code),
        },
        "kv_compression": _plain(dataclasses.asdict(cfg.kv_compression)),
        "research_config": research_config_dict(cfg, compressed=True),
        "prompt_shaping": {
            "data_format": cfg.data.format,
            "use_chat_template": bool(cfg.eval.use_chat_template),
            "strip_auto_system_block": bool(cfg.eval.strip_auto_system_block),
        },
    }
