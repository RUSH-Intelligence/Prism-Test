"""Alignment objective (spec §7/§8/§10): which layers, which positions, which distance.

Losses operate on ``[P, H]`` tensors (P aligned positions) and are computed in fp32:

* ``normalized_mse``             mean_p ||norm(s_p) - norm(t_p)||_2^2  (sum over hidden) = 2 * mean_p (1 - cos)
* ``normalized_mse_elementwise`` F.mse_loss(norm(s), norm(t)) — the spec's literal snippet; equals
                                 normalized_mse / H, which makes per-element gradients ~1e-7 and lets
                                 Adam's eps dominate; kept for the record, not as the default.
* ``cosine``                     mean_p (1 - cos(s_p, t_p))
* ``relative_mse``               mean_p ||s_p - t_p||^2 / ||t_p||^2   (scale-aware)

The optional output term is ``T^2 * KL(softmax(z_t/T) || softmax(z_s/T))`` on the same positions.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .config import AlignmentCfg, LayersCfg, LossCfg, PositionsCfg
from .hidden_states import FINAL_NORM_KEY, StateKey

EPS = 1e-12
DEFAULT_BUCKETS: Tuple[Tuple[int, Optional[int]], ...] = ((0, 16), (16, 64), (64, None))


# ---------------------------------------------------------------------------
# layers / positions
# ---------------------------------------------------------------------------
def resolve_layers(lcfg: LayersCfg, n_layers: int, *, first_trainable_layer: Optional[int] = None) -> List[int]:
    s = lcfg.strategy
    if s == "last_n":
        n = int(lcfg.n)
        if n <= 0 or n > n_layers:
            raise ValueError(f"alignment.layers.n must be in [1, {n_layers}], got {n}")
        return list(range(n_layers - n, n_layers))
    if s == "explicit":
        idx = sorted(set(int(i) for i in (lcfg.indices or [])))
        bad = [i for i in idx if not (0 <= i < n_layers)]
        if bad or not idx:
            raise ValueError(f"alignment.layers.indices out of range / empty: {bad or idx}")
        return idx
    if s == "all":
        return list(range(n_layers))
    if s == "from_first_trainable":
        if first_trainable_layer is None:
            raise ValueError("alignment.layers.strategy=from_first_trainable needs a trainable decoder layer")
        return list(range(int(first_trainable_layer), n_layers))
    raise ValueError(f"unknown alignment.layers.strategy {s!r}")


def dead_alignment_keys(keys: Sequence[StateKey], first_trainable_layer: Optional[int]) -> List[StateKey]:
    """Aligned layers whose output cannot depend on any trainable parameter (zero gradient)."""
    if first_trainable_layer is None:
        return list(keys)
    return [k for k in keys if isinstance(k, int) and k < first_trainable_layer]


def check_alignment_has_gradient(keys: Sequence[StateKey], first_trainable_layer: Optional[int],
                                 *, allow: bool) -> List[StateKey]:
    dead = dead_alignment_keys(keys, first_trainable_layer)
    if dead and not allow:
        raise ValueError(
            f"aligned layer(s) {dead} lie upstream of every trainable parameter (first trainable "
            f"decoder layer: {first_trainable_layer}); their loss terms carry no gradient. Change "
            f"alignment.layers (e.g. strategy: from_first_trainable) or set alignment.allow_dead_terms: true."
        )
    return dead


def first_affected_suffix_position(compressor) -> int:
    """First suffix position whose computation sees the compressed cache.

    Under the prefill schedules (``post_prefill`` / ``streaming``) the whole context is
    compressed before the first suffix token, so this is 0 for every suffix token.
    TODO(streaming/chunked): return the first position after the first compression event
    when context tokens themselves are included in the aligned segment.
    """
    del compressor
    return 0


def position_index(pcfg: PositionsCfg, suffix_len: int, *, first_affected: int = 0,
                   device: Optional[torch.device] = None) -> torch.Tensor:
    L = int(suffix_len)
    if L <= 0:
        raise ValueError("suffix_len must be > 0")
    s = pcfg.strategy
    if s == "all":
        idx = torch.arange(0, L)
    elif s == "recent":
        idx = torch.arange(max(0, L - int(pcfg.n)), L)
    elif s == "first_k":
        idx = torch.arange(0, min(L, int(pcfg.n)))
    elif s == "post_eviction":
        idx = torch.arange(min(max(0, int(first_affected)), L - 1), L)
    else:
        raise ValueError(f"unknown positions.strategy {s!r}")
    return idx.to(device) if device is not None else idx


# ---------------------------------------------------------------------------
# losses (per-position vectors [P])
# ---------------------------------------------------------------------------
def _f32(x: torch.Tensor) -> torch.Tensor:
    return x.float()


def normalized_mse_per_position(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return (F.normalize(_f32(s), dim=-1) - F.normalize(_f32(t), dim=-1)).pow(2).sum(-1)


def normalized_mse_elementwise_per_position(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return (F.normalize(_f32(s), dim=-1) - F.normalize(_f32(t), dim=-1)).pow(2).mean(-1)


def cosine_per_position(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    # 1 - cos == 0.5 * ||norm(s) - norm(t)||^2: computed this way it is exactly 0 for identical
    # inputs (F.cosine_similarity returns 1 +- 1e-7 in fp32) and stays consistent with normalized_mse.
    return 0.5 * normalized_mse_per_position(s, t)


def relative_mse_per_position(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    s, t = _f32(s), _f32(t)
    return (s - t).pow(2).sum(-1) / t.pow(2).sum(-1).clamp_min(EPS)


LOSSES: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "normalized_mse": normalized_mse_per_position,
    "normalized_mse_elementwise": normalized_mse_elementwise_per_position,
    "cosine": cosine_per_position,
    "relative_mse": relative_mse_per_position,
}


def _bucket_means(per_pos: torch.Tensor, positions: torch.Tensor,
                  buckets: Sequence[Tuple[int, Optional[int]]] = DEFAULT_BUCKETS) -> Dict[str, float]:
    out: Dict[str, float] = {}
    pos = positions.to(per_pos.device)
    for lo, hi in buckets:
        mask = pos >= lo if hi is None else (pos >= lo) & (pos < hi)
        if bool(mask.any()):
            out[f"{lo}-{hi if hi is not None else 'end'}"] = float(per_pos[mask].mean())
    return out


def hidden_loss(student: Dict[StateKey, torch.Tensor], teacher: Dict[StateKey, torch.Tensor],
                loss_name: str, keys: Sequence[StateKey], *, positions: Optional[torch.Tensor] = None,
                layer_weights: Optional[Sequence[float]] = None
                ) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, float]]:
    """Mean over aligned keys (optionally weighted) of the masked mean over positions.

    ``student`` / ``teacher`` hold ``[P, H]`` tensors per key (already gathered at the aligned
    positions). Returns ``(loss, per_layer_detached, per_position_bucket_detached)``.
    """
    if loss_name not in LOSSES:
        raise ValueError(f"unknown loss {loss_name!r}; choose from {sorted(LOSSES)}")
    fn = LOSSES[loss_name]
    keys = list(keys)
    if not keys:
        raise ValueError("no alignment keys")
    if layer_weights is not None:
        if len(layer_weights) != len(keys):
            raise ValueError(f"layer_weights has {len(layer_weights)} entries for {len(keys)} keys")
        w = torch.tensor([float(x) for x in layer_weights])
        w = w / w.sum()
    else:
        w = torch.full((len(keys),), 1.0 / len(keys))
    total: Optional[torch.Tensor] = None
    per_layer: Dict[str, float] = {}
    bucket_acc: Dict[str, List[float]] = {}
    for weight, key in zip(w.tolist(), keys):
        s, t = student[key], teacher[key]
        if s.shape != t.shape:
            raise ValueError(f"{key}: student {tuple(s.shape)} vs teacher {tuple(t.shape)}")
        per_pos = fn(s, t)                       # [P]
        term = per_pos.mean()
        per_layer[str(key)] = float(term.detach())
        if positions is not None:
            for name, val in _bucket_means(per_pos.detach(), positions).items():
                bucket_acc.setdefault(name, []).append(val)
        total = term * weight if total is None else total + term * weight
    per_bucket = {k: sum(v) / len(v) for k, v in bucket_acc.items()}
    assert total is not None
    return total, per_layer, per_bucket


def kl_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """``T^2 * KL(p_teacher || p_student)`` averaged over all positions (fp32); logits ``[..., V]``."""
    T = float(temperature)
    s = _f32(student_logits).reshape(-1, student_logits.shape[-1])
    t = _f32(teacher_logits).reshape(-1, teacher_logits.shape[-1])
    log_p_s = F.log_softmax(s / T, dim=-1)
    log_p_t = F.log_softmax(t / T, dim=-1)
    kl = (log_p_t.exp() * (log_p_t - log_p_s)).sum(-1)
    return (T * T) * kl.mean()


def combine(hidden: Optional[torch.Tensor], kl: Optional[torch.Tensor], lcfg: LossCfg) -> torch.Tensor:
    total: Optional[torch.Tensor] = None
    if hidden is not None and lcfg.hidden_weight > 0:
        total = lcfg.hidden_weight * hidden
    if kl is not None and lcfg.kl_weight > 0:
        total = lcfg.kl_weight * kl if total is None else total + lcfg.kl_weight * kl
    if total is None:
        raise ValueError("no active loss term")
    return total


def alignment_keys_for(acfg: AlignmentCfg, n_layers: int, *, first_trainable_layer: Optional[int]) -> List[StateKey]:
    keys: List[StateKey] = list(resolve_layers(acfg.layers, n_layers, first_trainable_layer=first_trainable_layer))
    if acfg.include_final_norm:
        keys.append(FINAL_NORM_KEY)
    return keys
