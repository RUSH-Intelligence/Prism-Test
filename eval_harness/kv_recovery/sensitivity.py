"""Compression-sensitivity layer selection ("Identifying Compression-Sensitive Layers").

Instead of choosing the layers to calibrate by hand, measure how strongly the KV
compressor perturbs the hidden representation of every transformer layer and
select the most affected ones.

For one calibration window ``[context | suffix]`` the same model runs twice: once
with the dense KV cache (teacher) and once with the compressed cache (student,
ORIGINAL weights). With ``H_l^dense, H_l^comp ∈ R^{m×d}`` the residual-stream
outputs of decoder layer ``l`` at the ``m`` post-compression (suffix) tokens — the
same hidden states and positions the alignment loss uses — the sensitivity is

    E_l = ||H_l^dense − H_l^comp||_F / (||H_l^dense||_F + eps).

``E_l`` is averaged (or median-aggregated) over a small HELD-OUT calibration set,
layers are ranked by it and the top-``k`` eligible layers are selected. Eligible =
the layers the trainable strategy can touch (softmax-attention layers for
``attention_projections``; on hybrids the linear-attention layers carry no K/V
cache and are never eligible, although their sensitivity is still reported).

Everything is computed in fp32 from the captured (bf16) states, under
``no_grad``, with the production prefill / segment code path (``student.py``), so
the measurement sees exactly the cache the training and evaluation see. No
benchmark data enters the selection.
"""
from __future__ import annotations

import logging
import math
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch

from .alignment import position_index
from .config import PositionsCfg, RecoveryConfig, SensitivityCfg
from .hidden_states import StateKey, gather_positions
from .model_spec import ModelSpec, inspect_model
from .student import Example, run_student, run_teacher

logger = logging.getLogger(__name__)

SELECTOR = "sensitivity"            # the ``trainable.layers`` value that activates this module
BLOCK_STRATEGIES = ("blocks", "mlp", "norms")   # strategies whose candidates are ALL decoder layers


# ---------------------------------------------------------------------------
# the formula
# ---------------------------------------------------------------------------
def layer_sensitivity(teacher: Dict[StateKey, torch.Tensor], student: Dict[StateKey, torch.Tensor], *,
                      eps: float) -> Dict[int, float]:
    """``E_l`` for ONE window from gathered ``[P, H]`` states; only decoder-layer (int) keys count.

    fp32; ``eps`` guards the (never observed) all-zero teacher state. Identical inputs give exactly 0.
    """
    out: Dict[int, float] = {}
    for key, t in teacher.items():
        if not isinstance(key, int):
            continue
        s = student[key]
        if s.shape != t.shape:
            raise ValueError(f"layer {key}: student {tuple(s.shape)} vs teacher {tuple(t.shape)}")
        tf, sf = t.float(), s.float()
        num = float(torch.linalg.norm(tf - sf))
        den = float(torch.linalg.norm(tf)) + float(eps)
        # den == 0 only with eps == 0 AND an all-zero teacher state: identical states are 0, anything else
        # is +inf (rank_layers then refuses the non-finite score instead of silently ranking it).
        out[key] = (num / den) if den > 0.0 else (0.0 if num == 0.0 else math.inf)
    return out


# ---------------------------------------------------------------------------
# aggregation / ranking / selection (pure functions, unit-tested in isolation)
# ---------------------------------------------------------------------------
def aggregate_scores(per_example: Dict[str, Dict[int, float]], aggregate: str = "mean"
                     ) -> tuple[Dict[int, float], Dict[int, float]]:
    """``{example: {layer: E_l}}`` -> (``{layer: aggregate}``, ``{layer: population std}``)."""
    if not per_example:
        raise ValueError("no calibration examples")
    layers = sorted({int(l) for scores in per_example.values() for l in scores})
    agg: Dict[int, float] = {}
    std: Dict[int, float] = {}
    for l in layers:
        vals = [float(scores[l]) for scores in per_example.values() if l in scores]
        if len(vals) != len(per_example):
            raise ValueError(f"layer {l} is missing from some calibration examples")
        if aggregate == "mean":
            agg[l] = sum(vals) / len(vals)
        elif aggregate == "median":
            agg[l] = float(statistics.median(vals))
        else:
            raise ValueError(f"unknown aggregate {aggregate!r}; choose mean | median")
        std[l] = float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0
    return agg, std


def rank_layers(scores: Dict[int, float]) -> List[int]:
    """Layers by descending sensitivity; exact ties resolve toward the DEEPER layer (closer to the output)."""
    for l, v in scores.items():
        if not math.isfinite(v):
            raise ValueError(f"layer {l}: non-finite sensitivity {v}")
    return sorted(scores, key=lambda l: (-scores[l], -int(l)))


def select_top_k(ranking: Sequence[int], candidates: Iterable[int], top_k: int) -> List[int]:
    """The ``top_k`` highest-ranked layers among ``candidates`` (ascending layer order).

    ``top_k`` larger than the candidate set selects every candidate (the selection then equals
    ``layers: all`` and is logged as such by the caller).
    """
    cands = set(int(c) for c in candidates)
    if not cands:
        raise ValueError("no eligible layers to select from")
    if int(top_k) <= 0:
        raise ValueError(f"top_k must be > 0, got {top_k}")
    missing = cands - set(int(l) for l in ranking)
    if missing:
        raise ValueError(f"candidates {sorted(missing)} were not measured")
    picked = [int(l) for l in ranking if int(l) in cands][: int(top_k)]
    return sorted(picked)


def candidate_layers(strategy: str, spec: ModelSpec) -> List[int]:
    """Layers the trainable ``strategy`` may touch (the pool the top-k is drawn from)."""
    if strategy == "attention_projections":
        return list(spec.full_attention_layers)
    if strategy in BLOCK_STRATEGIES:
        return list(range(spec.n_layers))
    raise ValueError(f"trainable.layers={SELECTOR!r} is not meaningful for strategy {strategy!r} "
                     f"(use attention_projections | blocks | mlp | norms)")


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------
@dataclass
class SensitivityReport:
    method: str
    layers: List[int]                          # every measured decoder layer
    per_example: Dict[str, Dict[int, float]]   # calibration window id -> {layer: E_l}
    aggregate: str
    scores: Dict[int, float]                   # aggregated E_l
    std: Dict[int, float]
    ranking: List[int]                         # most -> least sensitive
    candidates: List[int]                      # eligible layers for the trainable strategy
    selected: List[int]                        # top_k among candidates, ascending
    top_k: int
    eps: float
    positions: Dict[str, Any]
    calibration_ids: List[str]
    n_examples: int
    compressor: str
    compression_ratio: float
    hooked_layers: List[int]
    seconds: float
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method, "aggregate": self.aggregate, "top_k": self.top_k, "eps": self.eps,
            "positions": dict(self.positions), "n_examples": self.n_examples, "calibration_ids": list(self.calibration_ids),
            "compressor": self.compressor, "compression_ratio": self.compression_ratio,
            "hooked_layers": list(self.hooked_layers), "layers": list(self.layers),
            "scores": {str(l): self.scores[l] for l in self.layers},
            "std": {str(l): self.std[l] for l in self.layers},
            "ranking": list(self.ranking), "candidates": list(self.candidates), "selected": list(self.selected),
            "per_example": {ex: {str(l): v for l, v in sc.items()} for ex, sc in self.per_example.items()},
            "seconds": self.seconds, "notes": list(self.notes),
        }

    def rows(self) -> List[Dict[str, Any]]:
        rank_of = {l: i + 1 for i, l in enumerate(self.ranking)}
        return [{"layer": l, "sensitivity": self.scores[l], "std": self.std[l], "rank": rank_of[l],
                 "hooked": l in self.hooked_layers, "candidate": l in self.candidates, "selected": l in self.selected}
                for l in self.layers]

    def table(self) -> str:
        lines = [f"layer-wise compression sensitivity E_l = ||H_dense - H_comp||_F / (||H_dense||_F + {self.eps:g})"
                 f"  [{self.aggregate} over {self.n_examples} held-out windows; compressor {self.compressor} @ ratio "
                 f"{self.compression_ratio}]",
                 f"{'layer':>5} {'E_l':>10} {'std':>9} {'rank':>4}  hooked  candidate  selected"]
        for r in self.rows():
            lines.append(f"{r['layer']:>5} {r['sensitivity']:>10.5f} {r['std']:>9.5f} {r['rank']:>4}  "
                         f"{'yes' if r['hooked'] else ' - ':^6}  {'yes' if r['candidate'] else ' - ':^9}  "
                         f"{'<==' if r['selected'] else ''}")
        lines.append(f"selected (top-{self.top_k} of {len(self.candidates)} eligible): {self.selected}")
        for n in self.notes:
            lines.append(f"note: {n}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# measurement through the production teacher / student path
# ---------------------------------------------------------------------------
@torch.no_grad()
def measure_layer_sensitivity(teacher_adapter, student_adapter, compressor, examples: Sequence[Example], *,
                              spec: Optional[ModelSpec] = None, positions_cfg: Optional[PositionsCfg] = None,
                              eps: float = 1e-6, mode: str = "block", compression_ratio: float = 0.0,
                              prefill_chunk_size: Optional[int] = None,
                              layers: Optional[Sequence[int]] = None) -> Dict[str, Dict[int, float]]:
    """Per-window ``{layer: E_l}`` over ``layers`` (default: every decoder layer).

    ``teacher_adapter`` may be the SAME object as ``student_adapter``: the compressor is a context
    manager that leaves no hooks behind, and the weights are untouched, so one model instance
    can serve both passes (``scripts/measure_layer_sensitivity.py`` does this).
    """
    model = student_adapter._model
    spec = spec or inspect_model(model)
    layer_indices = list(layers) if layers is not None else list(range(spec.n_layers))
    pcfg = positions_cfg or PositionsCfg()
    out: Dict[str, Dict[int, float]] = {}
    for ex in examples:
        t = run_teacher(teacher_adapter, ex, layer_indices, include_final_norm=False, want_logits=False, mode=mode, spec=spec)
        s = run_student(student_adapter, ex, compressor, layer_indices, include_final_norm=False, grad=False,
                        want_logits=False, prefill_chunk_size=prefill_chunk_size, mode=mode, spec=spec,
                        compression_ratio=compression_ratio if compressor is not None else 0.0)
        dev = next(iter(t.states.values())).device
        pos = position_index(pcfg, ex.suffix_len, device=dev)
        tg = gather_positions(t.states, pos)
        sg = gather_positions({k: v.to(dev) for k, v in s.states.items()}, pos)
        out[ex.id] = layer_sensitivity(tg, sg, eps=eps)
        del t, s, tg, sg
    return out


def select_layers_by_sensitivity(cfg: RecoveryConfig, teacher_adapter, student_adapter, compressor,
                                 calibration: Sequence[Example], *, spec: Optional[ModelSpec] = None,
                                 mode: str = "block", strategy: Optional[str] = None,
                                 scfg: Optional[SensitivityCfg] = None) -> SensitivityReport:
    """Measure, aggregate, rank and pick the top-k eligible layers for ``cfg.trainable``."""
    scfg = scfg or cfg.trainable.sensitivity
    strategy = strategy or cfg.trainable.strategy
    model = student_adapter._model
    spec = spec or inspect_model(model)
    if compressor is None:
        raise ValueError("sensitivity selection needs a compressor (kv_compression.kv_compressor != none)")
    if not calibration:
        raise ValueError("sensitivity selection needs at least one calibration window")
    t0 = time.time()
    per_example = measure_layer_sensitivity(
        teacher_adapter, student_adapter, compressor, calibration, spec=spec, positions_cfg=scfg.positions,
        eps=float(scfg.eps), mode=mode, compression_ratio=float(cfg.kv_compression.compression_ratio),
        prefill_chunk_size=cfg.kv_compression.prefill_chunk_size)
    scores, std = aggregate_scores(per_example, scfg.aggregate)
    ranking = rank_layers(scores)
    cands = candidate_layers(strategy, spec)
    selected = select_top_k(ranking, cands, int(scfg.top_k))
    notes: List[str] = []
    if all(scores[l] == 0.0 for l in cands):
        raise ValueError("every eligible layer has zero compression sensitivity: the compressor did not change the cache "
                         "(ratio 0? schedule not firing at prefill?) — nothing to select")
    if int(scfg.top_k) >= len(cands):
        notes.append(f"top_k={scfg.top_k} >= {len(cands)} eligible layers: the selection equals 'all'")
    pre = [l for l in range(spec.n_layers) if l < spec.first_full_attention_layer]
    if spec.is_hybrid and pre:
        worst = max(scores[l] for l in pre)
        notes.append(f"hybrid model: layers {pre} precede the first K/V-carrying layer; max E_l there = {worst:.2e} "
                     f"({'as expected, 0' if worst == 0.0 else 'EXPECTED 0 - check the compressor hooks'})")
    report = SensitivityReport(
        method=SELECTOR, layers=sorted(scores), per_example=per_example, aggregate=scfg.aggregate, scores=scores, std=std,
        ranking=ranking, candidates=cands, selected=selected, top_k=int(scfg.top_k), eps=float(scfg.eps),
        positions=dict(scfg.positions.__dict__), calibration_ids=[ex.id for ex in calibration], n_examples=len(calibration),
        compressor=str(cfg.kv_compression.kv_compressor), compression_ratio=float(cfg.kv_compression.compression_ratio),
        hooked_layers=list(spec.full_attention_layers), seconds=round(time.time() - t0, 1), notes=notes)
    logger.info("sensitivity selection: %s", report.selected)
    return report
