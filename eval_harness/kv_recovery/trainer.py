"""Training loop for hidden-state alignment (spec §9-§11, §17, §21-§22).

Precision (prior experiment, decision D10): forward/backward run in BF16 on the model's own
weights; gradients are accumulated into FP32 master copies of the trainable tensors, AdamW
runs on the masters, and the masters are rounded back into the BF16 weights after every
optimizer step (pure-BF16 AdamW at lr 1e-5 silently drops most updates: median |w| ~ 5e-3,
BF16 half-spacing ~ 1e-5). Ported from kv_compression_adaptation/src/training/trainer.py
with the objective replaced by the alignment loss.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch import nn

from .alignment import combine, first_affected_suffix_position, hidden_loss, kl_loss, position_index
from .config import LossCfg, PositionsCfg, RecoveryConfig
from .hidden_states import StateKey, gather_positions
from .model_spec import ModelSpec
from .student import Example, run_student, run_teacher
from .trainable import assert_no_stray_grads

logger = logging.getLogger(__name__)


class TrainingUnstable(RuntimeError):
    pass


def prepare_run_dir(run_dir: Path, *, overwrite: bool = False) -> Optional[Path]:
    """Make ``run_dir`` usable for a new run.

    A COMPLETED run (``checkpoint/metadata.json`` present) is never overwritten unless
    ``overwrite``; an INCOMPLETE leftover (a crashed or killed job) is moved aside into
    ``run_dir/superseded/<n>/`` — the eval runner's convention — so a resubmission can proceed
    without manual cleanup. Returns the archive path when something was moved."""
    run_dir = Path(run_dir)
    if not run_dir.exists() or not any(run_dir.iterdir()):
        return None
    complete = (run_dir / "checkpoint" / "metadata.json").exists()
    if complete and not overwrite:
        raise FileExistsError(f"run dir {run_dir} holds a completed run (use --overwrite or a new --run-name)")
    sup = run_dir / "superseded"
    n = len([d for d in sup.iterdir() if d.is_dir()]) if sup.exists() else 0
    target = sup / str(n)
    target.mkdir(parents=True, exist_ok=True)
    for child in list(run_dir.iterdir()):
        if child.name == "superseded":
            continue
        child.rename(target / child.name)
    return target


# ---------------------------------------------------------------------------
# resolved alignment setup + teacher states (online or offline)
# ---------------------------------------------------------------------------
@dataclass
class AlignmentSetup:
    keys: List[StateKey]               # aligned keys (ints + "norm")
    layer_indices: List[int]           # hooked decoder layers
    include_final_norm: bool
    positions_cfg: PositionsCfg
    loss_name: str
    layer_weights: Optional[List[float]]
    loss_cfg: LossCfg
    want_logits: bool
    mode: str                          # block | token_by_token
    spec: ModelSpec
    compression_ratio: float
    prefill_chunk_size: Optional[int]
    prefill_grad: bool
    deterministic_backward: bool = False


@dataclass
class TeacherStates:
    gathered: Dict[StateKey, torch.Tensor]   # key -> [P, H] at ``positions``
    positions: torch.Tensor                  # suffix positions [P]
    logits: Optional[torch.Tensor] = None    # [1, P, V] at ``positions`` (online + KL only)
    cache_len_after_prefill: Optional[int] = None


def teacher_digest(cfg: RecoveryConfig) -> str:
    """Identity of precomputed teacher states: anything that changes them changes this."""
    d = cfg.data
    payload = {
        "model": cfg.model.name, "revision": cfg.model.revision, "dtype": cfg.model.dtype,
        "attn": cfg.model.attn_implementation, "dequantize_fp8": cfg.model.dequantize_fp8,
        "data": {"path": d.path, "val_path": d.val_path, "max_length": d.max_length,
                 "suffix_length": d.suffix_length, "seed": d.seed, "format": d.format,
                 "suffix_mode": d.suffix_mode, "strip_auto_system_block": d.strip_auto_system_block},
        "alignment": {"layers": cfg.alignment.layers.__dict__, "include_final_norm": cfg.alignment.include_final_norm,
                      "positions": cfg.alignment.positions.__dict__},
        "segment_mode": cfg.student.segment_mode,
        "trainable": cfg.trainable.__dict__,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


class TeacherStore:
    """Offline teacher states written by ``scripts/precompute_teacher_states.py``."""

    MANIFEST = "manifest.json"

    def __init__(self, states_dir: str | Path, device: Optional[torch.device] = None):
        self.dir = Path(states_dir)
        self.device = device
        self.manifest = json.loads((self.dir / self.MANIFEST).read_text())

    def check(self, cfg: RecoveryConfig, keys: Sequence[StateKey]) -> None:
        want = teacher_digest(cfg)
        if self.manifest.get("teacher_digest") != want:
            raise ValueError(f"offline teacher states at {self.dir} were computed for a different "
                             f"model/data/alignment setup ({self.manifest.get('teacher_digest')} != {want})")
        have = [str(k) for k in self.manifest.get("keys", [])]
        missing = [k for k in keys if str(k) not in have]
        if missing:
            raise ValueError(f"offline teacher states lack aligned keys {missing}")

    def path_for(self, ex_id: str) -> Path:
        return self.dir / f"{ex_id}.safetensors"

    def has(self, ex_id: str) -> bool:
        return self.path_for(ex_id).exists()

    def get(self, ex_id: str) -> TeacherStates:
        from safetensors.torch import load_file

        tensors = load_file(str(self.path_for(ex_id)))
        positions = tensors.pop("positions").long()
        gathered: Dict[StateKey, torch.Tensor] = {}
        for name, t in tensors.items():
            key: StateKey = int(name[len("layer_"):]) if name.startswith("layer_") else name
            gathered[key] = t.to(self.device) if self.device is not None else t
        return TeacherStates(gathered=gathered, positions=positions.to(self.device) if self.device is not None else positions)

    @staticmethod
    def save(states_dir: str | Path, ex_id: str, ts: TeacherStates, *, dtype: torch.dtype = torch.bfloat16) -> Path:
        from safetensors.torch import save_file

        out = Path(states_dir) / f"{ex_id}.safetensors"
        tensors = {("layer_%d" % k if isinstance(k, int) else str(k)): v.detach().to("cpu", dtype).contiguous()
                   for k, v in ts.gathered.items()}
        tensors["positions"] = ts.positions.detach().cpu().long().contiguous()
        save_file(tensors, str(out))
        return out


def teacher_states_for(teacher_adapter, ex: Example, setup: AlignmentSetup, *, compressor=None,
                       store: Optional[TeacherStore] = None) -> TeacherStates:
    positions = position_index(setup.positions_cfg, ex.suffix_len,
                               first_affected=first_affected_suffix_position(compressor))
    if store is not None:
        ts = store.get(ex.id)
        if ts.positions.tolist() != positions.tolist():
            raise ValueError(f"{ex.id}: stored teacher positions differ from the configured positions")
        return ts
    out = run_teacher(teacher_adapter, ex, setup.layer_indices, include_final_norm=setup.include_final_norm,
                      want_logits=setup.want_logits, mode=setup.mode, spec=setup.spec)
    dev = next(iter(out.states.values())).device
    pos = positions.to(dev)
    gathered = gather_positions(out.states, pos)
    logits = out.logits[:, pos] if (setup.want_logits and out.logits is not None) else None
    return TeacherStates(gathered=gathered, positions=pos, logits=logits,
                         cache_len_after_prefill=out.cache_len_after_prefill)


def _backward_kernel_ctx(setup: AlignmentSetup):
    if setup.deterministic_backward and torch.cuda.is_available():
        from torch.nn.attention import SDPBackend, sdpa_kernel

        return sdpa_kernel([SDPBackend.MATH])
    return contextlib.nullcontext()


def student_loss(student_adapter, compressor, ex: Example, teacher: TeacherStates, setup: AlignmentSetup,
                 *, grad: bool) -> tuple[torch.Tensor, Dict[str, Any]]:
    with _backward_kernel_ctx(setup) if grad else contextlib.nullcontext():
        out = run_student(student_adapter, ex, compressor, setup.layer_indices, include_final_norm=setup.include_final_norm,
                          grad=grad, want_logits=setup.want_logits, prefill_grad=setup.prefill_grad,
                          prefill_chunk_size=setup.prefill_chunk_size, mode=setup.mode, spec=setup.spec,
                          compression_ratio=setup.compression_ratio)
    pos = teacher.positions.to(next(iter(out.states.values())).device)
    sg = gather_positions(out.states, pos)
    tg = {k: v.to(sg[k].device) for k, v in teacher.gathered.items()}
    hidden, per_layer, per_bucket = hidden_loss(sg, tg, setup.loss_name, setup.keys, positions=pos,
                                                layer_weights=setup.layer_weights)
    kl = None
    if setup.want_logits:
        if teacher.logits is None or out.logits is None:
            raise ValueError("loss.kl_weight > 0 needs teacher and student logits (online teacher)")
        kl = kl_loss(out.logits[:, pos], teacher.logits.to(out.logits.device), setup.loss_cfg.temperature)
    total = combine(hidden if setup.loss_cfg.hidden_weight > 0 else None, kl, setup.loss_cfg)
    info = {"hidden": float(hidden.detach()), "kl": float(kl.detach()) if kl is not None else 0.0,
            "per_layer": per_layer, "per_bucket": per_bucket,
            "cache_len_after_prefill": out.cache_len_after_prefill}
    return total, info


@torch.no_grad()
def validate(student_adapter, compressor, teacher_adapter, examples: Sequence[Example], setup: AlignmentSetup,
             *, store: Optional[TeacherStore] = None, teacher_cache: Optional[Dict[str, TeacherStates]] = None) -> Dict[str, Any]:
    tot, hid, kls, per_layer_acc = [], [], [], {}
    for ex in examples:
        ts = teacher_cache[ex.id] if teacher_cache is not None and ex.id in teacher_cache else \
            teacher_states_for(teacher_adapter, ex, setup, compressor=compressor, store=store)
        loss, info = student_loss(student_adapter, compressor, ex, ts, setup, grad=False)
        tot.append(float(loss)); hid.append(info["hidden"]); kls.append(info["kl"])
        for k, v in info["per_layer"].items():
            per_layer_acc.setdefault(k, []).append(v)
    n = max(len(tot), 1)
    return {"val_loss": sum(tot) / n, "val_hidden": sum(hid) / n, "val_kl": sum(kls) / n,
            "val_per_layer": {k: sum(v) / len(v) for k, v in per_layer_acc.items()}, "n": len(tot)}


@torch.no_grad()
def check_same_model(teacher_adapter, student_adapter, examples: Sequence[Example], setup: AlignmentSetup,
                     *, tol: float = 1e-6) -> Dict[str, Any]:
    """Spec §22 same-model check: without compression and with untouched weights the student's
    states equal the teacher's (bitwise under identical kernels) and the alignment loss is 0."""
    rows = []
    for ex in examples:
        ts = teacher_states_for(teacher_adapter, ex, setup, compressor=None)
        out = run_student(student_adapter, ex, None, setup.layer_indices, include_final_norm=setup.include_final_norm,
                          grad=False, want_logits=False, mode=setup.mode, spec=setup.spec, compression_ratio=0.0)
        sg = gather_positions(out.states, ts.positions.to(next(iter(out.states.values())).device))
        bitwise = all(torch.equal(sg[k].cpu(), ts.gathered[k].cpu()) for k in setup.keys)
        loss, _, _ = hidden_loss(sg, {k: v.to(sg[k].device) for k, v in ts.gathered.items()}, "normalized_mse", setup.keys)
        max_abs = max(float((sg[k].float().cpu() - ts.gathered[k].float().cpu()).abs().max()) for k in setup.keys)
        rows.append({"id": ex.id, "bitwise_equal": bool(bitwise), "loss": float(loss), "max_abs_diff": max_abs})
    return {"passed": all(r["loss"] <= tol for r in rows), "tol": tol, "examples": rows}


# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------
@dataclass
class TrainState:
    step_logs: List[dict] = field(default_factory=list)
    val_logs: List[dict] = field(default_factory=list)
    check3: Dict[str, Any] = field(default_factory=dict)
    masters: Dict[str, torch.Tensor] = field(default_factory=dict)
    optimizer_steps: int = 0
    micro_batches: int = 0
    seconds_train_loop: float = 0.0
    seconds_validation: float = 0.0
    context_tokens_teacher: int = 0
    context_tokens_student: int = 0
    suffix_tokens_aligned: int = 0
    peak_gpu_memory_gib: Optional[float] = None
    final_train_loss: Optional[float] = None
    first_train_loss: Optional[float] = None


def _peak_mem() -> Optional[float]:
    if torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated() / 2**30, 3)
    return None


def train(teacher_adapter, student_adapter, compressor, trainable: Dict[str, nn.Parameter],
          train_examples: Sequence[Example], val_examples: Sequence[Example], cfg: RecoveryConfig,
          setup: AlignmentSetup, *, log_path: Path, store: Optional[TeacherStore] = None,
          warmup_fraction: Optional[float] = None, print_fn=print) -> TrainState:
    o = cfg.optim
    names = list(trainable)
    params = [trainable[n] for n in names]
    student_model = student_adapter._model
    use_masters = bool(o.master_weights_fp32)
    masters = [nn.Parameter(p.detach().float().clone()) for p in params] if use_masters else params
    optimizer = torch.optim.AdamW(masters, lr=o.learning_rate, betas=tuple(o.betas), eps=o.eps, weight_decay=o.weight_decay)
    state = TrainState()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    order: List[int] = []
    for epoch in range(o.epochs):
        rng = random.Random(o.shuffle_seed + epoch)
        order += rng.sample(range(len(train_examples)), len(train_examples))
    total_steps = math.ceil(len(order) / o.grad_accum)
    if o.max_steps is not None:
        total_steps = min(total_steps, int(o.max_steps))
        order = order[: total_steps * o.grad_accum]
    wf = o.warmup_fraction if warmup_fraction is None else warmup_fraction
    warmup_steps = math.ceil(wf * total_steps) if wf > 0 else 0

    # Validation teacher states are computed once (online) and kept on the CPU.
    val_cache: Dict[str, TeacherStates] = {}
    t_val = time.time()
    for ex in val_examples:
        ts = teacher_states_for(teacher_adapter, ex, setup, compressor=compressor, store=store)
        val_cache[ex.id] = TeacherStates(gathered={k: v.cpu() for k, v in ts.gathered.items()}, positions=ts.positions.cpu(),
                                         logits=ts.logits.cpu() if ts.logits is not None else None)
        state.context_tokens_teacher += ex.context_len
    v = validate(student_adapter, compressor, teacher_adapter, val_examples, setup, store=store, teacher_cache=val_cache)
    state.context_tokens_student += sum(ex.context_len for ex in val_examples)
    state.val_logs.append({"step": 0, **v})
    state.seconds_validation += time.time() - t_val
    print_fn(f"step 0/{total_steps} | val_loss {v['val_loss']:.6f} hidden {v['val_hidden']:.6f} kl {v['val_kl']:.6f}")

    step, micro, gn_history, bad_streak = 0, 0, [], 0
    acc = {"loss": 0.0, "hidden": 0.0, "kl": 0.0, "n": 0, "per_layer": {}, "per_bucket": {}}
    t_loop = time.time()
    t_step = time.time()
    for pos, idx in enumerate(order):
        ex = train_examples[idx]
        ts = teacher_states_for(teacher_adapter, ex, setup, compressor=compressor, store=store)
        loss, info = student_loss(student_adapter, compressor, ex, ts, setup, grad=True)
        if not torch.isfinite(loss):
            raise TrainingUnstable(f"non-finite loss at step {step} ({ex.id})")
        (loss / o.grad_accum).backward()
        state.context_tokens_teacher += ex.context_len if store is None else 0
        state.context_tokens_student += ex.context_len
        state.suffix_tokens_aligned += int(ts.positions.numel())
        state.micro_batches += 1
        if state.micro_batches == 1:
            assert_no_stray_grads(student_model, names)          # spec §21.8 (micro-batch part)
        if use_masters:
            for p, m in zip(params, masters):
                g = p.grad.float()
                m.grad = g if m.grad is None else m.grad + g
                p.grad = None
        micro += 1
        acc["loss"] += float(loss.detach()); acc["hidden"] += info["hidden"]; acc["kl"] += info["kl"]; acc["n"] += 1
        for k, val in info["per_layer"].items():
            acc["per_layer"][k] = acc["per_layer"].get(k, 0.0) + val
        for k, val in info["per_bucket"].items():
            acc["per_bucket"][k] = acc["per_bucket"].get(k, 0.0) + val
        del loss
        if micro % o.grad_accum != 0 and pos != len(order) - 1:
            continue

        pre_clip = {n: float(m.grad.norm()) if m.grad is not None else 0.0 for n, m in zip(names, masters)}
        grad_norm = float(torch.nn.utils.clip_grad_norm_(masters, o.grad_clip))
        if step == 0:
            all_named = dict(student_model.named_parameters())
            ranked = sorted(((pre_clip.get(n, 0.0), n) for n in all_named), reverse=True)[:20]
            state.check3 = {"passed": all(pre_clip[n] > 0 for n in names),
                            "top20_pre_clip": [{"name": n, "grad_norm": g, "trainable": n in trainable} for g, n in ranked],
                            "total_norm_pre_clip": grad_norm,
                            "zero_grad_trainable": [n for n in names if pre_clip[n] == 0]}
        if gn_history and grad_norm > o.instability.grad_norm_factor * gn_history[0]:
            bad_streak += 1
            if bad_streak >= o.instability.consecutive_steps:
                raise TrainingUnstable(f"grad norm {grad_norm:.3e} > {o.instability.grad_norm_factor}x the step-1 norm "
                                       f"for {bad_streak} consecutive steps")
        else:
            bad_streak = 0
        gn_history.append(grad_norm)

        lr = o.learning_rate * min(1.0, (step + 1) / warmup_steps) if warmup_steps else o.learning_rate
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        changed = 0
        if use_masters:
            with torch.no_grad():
                for p, m in zip(params, masters):
                    new = m.detach().to(p.dtype)
                    changed += int((new != p).sum())
                    p.copy_(new)
        step += 1
        n = max(acc["n"], 1)
        log = {"step": step, "lr": lr, "loss": acc["loss"] / n, "hidden_loss": acc["hidden"] / n, "kl_loss": acc["kl"] / n,
               "per_layer": {k: v / n for k, v in acc["per_layer"].items()},
               "per_bucket": {k: v / n for k, v in acc["per_bucket"].items()},
               "grad_norm_pre_clip": grad_norm, "bf16_elements_changed": changed,
               "examples_seen": pos + 1, "seconds": round(time.time() - t_step, 3), "peak_gpu_memory_gib": _peak_mem()}
        state.step_logs.append(log)
        state.first_train_loss = state.first_train_loss if state.first_train_loss is not None else log["loss"]
        state.final_train_loss = log["loss"]
        with open(log_path, "a") as f:
            f.write(json.dumps(log) + "\n")
        print_fn(f"step {step}/{total_steps} | loss {log['loss']:.6f} hidden {log['hidden_loss']:.6f} kl {log['kl_loss']:.6f} "
                 f"| gnorm {grad_norm:.3e} | lr {lr:.2e} | changed {changed} | {log['seconds']:.1f}s")
        acc = {"loss": 0.0, "hidden": 0.0, "kl": 0.0, "n": 0, "per_layer": {}, "per_bucket": {}}
        t_step = time.time()

        if step % o.val_every_steps == 0 or step == total_steps:
            state.seconds_train_loop += time.time() - t_loop
            t_val = time.time()
            v = validate(student_adapter, compressor, teacher_adapter, val_examples, setup, store=store, teacher_cache=val_cache)
            state.context_tokens_student += sum(ex.context_len for ex in val_examples)
            state.val_logs.append({"step": step, **v})
            state.seconds_validation += time.time() - t_val
            print_fn(f"step {step} | val_loss {v['val_loss']:.6f} hidden {v['val_hidden']:.6f} kl {v['val_kl']:.6f}")
            t_loop = time.time()
        if step >= total_steps:
            break
    state.seconds_train_loop += time.time() - t_loop
    state.optimizer_steps = step
    state.masters = {n: m.detach() for n, m in zip(names, masters)} if use_masters else {}
    state.peak_gpu_memory_gib = _peak_mem()
    return state


def weight_update_norms(original: Dict[str, torch.Tensor], trainable: Dict[str, nn.Parameter],
                        masters: Dict[str, torch.Tensor], n_steps: int, lr: float, spec: ModelSpec) -> List[dict]:
    """Relative update per tensor (BF16 deployed and FP32 master) + the fraction of changed entries."""
    from .trainable import layer_index_of

    rows = []
    for name, p in trainable.items():
        w0 = original[name].double().cpu()
        w_bf16 = p.detach().double().cpu()
        n = w0.numel()
        norm0 = float(w0.norm()) or 1.0
        row = {"parameter": name, "layer": layer_index_of(name, spec), "module": name.split(".")[-2] if "." in name else name,
               "numel": n, "weight_norm": norm0,
               "rel_update_bf16": float((w_bf16 - w0).norm()) / norm0,
               "frac_bf16_elements_changed": float((p.detach().cpu() != original[name].cpu()).float().mean()),
               "expected_rel_update_random_walk": lr * math.sqrt(max(n_steps, 1)) * math.sqrt(n) / norm0,
               "expected_rel_update_consistent": lr * max(n_steps, 1) * math.sqrt(n) / norm0}
        if name in masters:
            row["rel_update_fp32_master"] = float((masters[name].double().cpu() - w0).norm()) / norm0
        rows.append(row)
    return rows
