"""Lightweight weight-delta checkpoints (spec §19): only the trained tensors + metadata.

Layout (``<run>/checkpoint/``):
  adapted_weights.safetensors   "<param name>" -> tensor in the model dtype; optional
                                "<param name>.fp32_master" -> FP32 master copy
  metadata.json                 format_version, base model / revision, load flags, trainable
                                names, sha256 of original + adapted tensors, a sample of
                                frozen tensors, the compression block, prompt shaping,
                                training config, seeds, determinism, packages, hardware, git
  config.yaml                   the full RecoveryConfig

``apply_delta`` overwrites exactly the listed tensors on a freshly loaded base model, after
verifying (strict mode) that every target tensor still holds the ORIGINAL bytes it was trained
from — so applying a delta twice, or to the wrong base, raises instead of silently corrupting.
Format derived from kv_compression_adaptation/src/common/modeling.py (compatible loader logic).
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
from torch import nn

from .model_spec import ModelSpec, inspect_model, language_model, resolve_base_revision

logger = logging.getLogger(__name__)

FORMAT_VERSION = "prism-kv-recovery-delta-v1"
WEIGHTS_FILE = "adapted_weights.safetensors"
META_FILE = "metadata.json"
CONFIG_FILE = "config.yaml"
MASTER_SUFFIX = ".fp32_master"


def tensor_sha256(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def frozen_sample_names(model: nn.Module, trainable: Iterable[str], spec: Optional[ModelSpec] = None) -> List[str]:
    """A fixed handful of frozen tensors whose hashes pin the base weights the delta applies to."""
    spec = spec or inspect_model(model)
    params = dict(model.named_parameters())
    wanted = [f"{spec.lm_prefix}embed_tokens.weight", f"{spec.lm_prefix}norm.weight"]
    if spec.n_layers:
        first = spec.layer_prefix(0)
        mid = spec.layer_prefix(spec.n_layers // 2)
        for prefix in (first, mid):
            cands = [n for n in params if n.startswith(prefix)]
            if cands:
                wanted.append(sorted(cands)[0])
                mlp = [n for n in cands if ".mlp." in n]
                if mlp:
                    wanted.append(sorted(mlp)[0])
    trainable = set(trainable)
    out = [n for n in dict.fromkeys(wanted) if n in params and n not in trainable]
    return out[:6]


@torch.no_grad()
def hashes_of(model: nn.Module, names: Iterable[str]) -> Dict[str, str]:
    params = dict(model.named_parameters())
    return {n: tensor_sha256(params[n]) for n in names}


@torch.no_grad()
def write_delta(ckpt_dir: str | Path, model: nn.Module, names: Iterable[str], original_sha256: Dict[str, str],
                metadata: Dict[str, Any], *, masters: Optional[Dict[str, torch.Tensor]] = None,
                frozen_sample_sha256: Optional[Dict[str, str]] = None, config_yaml: Optional[str] = None) -> Path:
    from safetensors.torch import save_file

    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    names = sorted(names)
    params = dict(model.named_parameters())
    tensors: Dict[str, torch.Tensor] = {}
    adapted_sha: Dict[str, str] = {}
    dtypes: Dict[str, str] = {}
    shapes: Dict[str, List[int]] = {}
    for n in names:
        p = params[n].detach().cpu().contiguous()
        tensors[n] = p
        adapted_sha[n] = tensor_sha256(p)
        dtypes[n] = str(p.dtype).replace("torch.", "")
        shapes[n] = list(p.shape)
    if masters:
        for n, m in masters.items():
            if n in names:
                tensors[f"{n}{MASTER_SUFFIX}"] = m.detach().cpu().float().contiguous()
    save_file(tensors, str(ckpt_dir / WEIGHTS_FILE), metadata={"format": FORMAT_VERSION})
    meta = {
        "format_version": FORMAT_VERSION,
        **metadata,
        "trainable_parameters": names,
        "n_trainable_parameters": int(sum(params[n].numel() for n in names)),
        "dtypes": dtypes,
        "shapes": shapes,
        "original_sha256": {n: original_sha256[n] for n in names},
        "adapted_sha256": adapted_sha,
        "frozen_sample_sha256": frozen_sample_sha256 or {},
        "has_fp32_masters": bool(masters),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (ckpt_dir / META_FILE).write_text(json.dumps(meta, indent=2, default=str))
    meta["weights_sha256"] = sha256_file(ckpt_dir / WEIGHTS_FILE)
    (ckpt_dir / META_FILE).write_text(json.dumps(meta, indent=2, default=str))
    if config_yaml is not None:
        (ckpt_dir / CONFIG_FILE).write_text(config_yaml)
    return ckpt_dir


def write_identity_delta(ckpt_dir: str | Path, model: nn.Module, names: Iterable[str], metadata: Dict[str, Any]) -> Path:
    """A zero-delta checkpoint (adapted == original) for the identity sanity check."""
    names = sorted(names)
    sha = hashes_of(model, names)
    return write_delta(ckpt_dir, model, names, sha, {**metadata, "identity_delta": True},
                       frozen_sample_sha256=hashes_of(model, frozen_sample_names(model, names)))


def load_metadata(ckpt_dir: str | Path) -> Dict[str, Any]:
    return json.loads((Path(ckpt_dir) / META_FILE).read_text())


def checkpoint_digest(ckpt_dir: str | Path) -> str:
    return sha256_file(Path(ckpt_dir) / WEIGHTS_FILE)


def _base_name_matches(meta_name: Optional[str], model: nn.Module) -> Optional[bool]:
    if not meta_name:
        return None
    cfg = getattr(model, "config", None)
    loaded = str(getattr(cfg, "_name_or_path", "") or "")
    if not loaded:
        return None
    tail = meta_name.split("/")[-1]
    return meta_name == loaded or tail in loaded


@torch.no_grad()
def apply_delta(model: nn.Module, ckpt_dir: str | Path, *, strict: bool = True,
                expected_sha256: Optional[str] = None, verify_frozen: bool = True) -> Dict[str, Any]:
    """Overwrite the trained tensors of ``model`` with the delta in ``ckpt_dir``.

    Checks (raise in strict mode, warn otherwise): checkpoint file digest, base model name /
    revision, tensor set == metadata, shapes/dtypes, every target tensor still carries the
    ORIGINAL bytes (so a second application raises), frozen-sample hashes; after copying,
    the adapted hashes are re-verified.
    """
    from safetensors.torch import load_file

    ckpt_dir = Path(ckpt_dir)
    meta = load_metadata(ckpt_dir)
    problems: List[str] = []

    def problem(msg: str) -> None:
        if strict:
            raise ValueError(f"apply_delta({ckpt_dir}): {msg}")
        problems.append(msg)
        logger.warning("apply_delta(%s): %s", ckpt_dir, msg)

    if meta.get("format_version") != FORMAT_VERSION:
        problem(f"unknown format_version {meta.get('format_version')!r}")
    digest = checkpoint_digest(ckpt_dir)
    if expected_sha256 and digest != expected_sha256:
        problem(f"weights sha256 {digest[:12]} != expected {expected_sha256[:12]}")
    if meta.get("weights_sha256") and meta["weights_sha256"] != digest:
        problem("weights file does not match the digest recorded in metadata.json")
    match = _base_name_matches(meta.get("base_model"), model)
    if match is False:
        problem(f"base_model {meta.get('base_model')!r} != loaded {getattr(model.config, '_name_or_path', None)!r}")
    rev_meta, rev_model = meta.get("base_revision"), resolve_base_revision(model)
    if rev_meta and rev_model and rev_meta != rev_model:
        problem(f"base_revision {rev_meta} != loaded snapshot {rev_model}")

    tensors = load_file(str(ckpt_dir / WEIGHTS_FILE))
    deltas = {k: v for k, v in tensors.items() if not k.endswith(MASTER_SUFFIX)}
    expected = set(meta["trainable_parameters"])
    if set(deltas) != expected:
        raise ValueError(f"checkpoint tensors {sorted(deltas)[:5]}... != metadata {sorted(expected)[:5]}...")
    params = dict(model.named_parameters())
    missing = expected - set(params)
    if missing:
        raise ValueError(f"model has no parameters {sorted(missing)[:5]}")
    for name in sorted(expected):
        p, v = params[name], deltas[name]
        if tuple(p.shape) != tuple(v.shape):
            raise ValueError(f"{name}: shape {tuple(v.shape)} vs model {tuple(p.shape)}")
        if p.dtype != v.dtype:
            problem(f"{name}: dtype {v.dtype} vs model {p.dtype} (casting)")
        if tensor_sha256(p) != meta["original_sha256"][name]:
            problem(f"{name}: the model's current tensor is not the original the delta was trained from "
                    "(wrong base weights, or the delta was already applied)")
    if verify_frozen:
        for name, sha in (meta.get("frozen_sample_sha256") or {}).items():
            if name in params and tensor_sha256(params[name]) != sha:
                problem(f"frozen tensor {name} differs from the training-time base weights")
    for name in sorted(expected):
        p = params[name]
        p.copy_(deltas[name].to(device=p.device, dtype=p.dtype))
    for name in sorted(expected):
        if tensor_sha256(params[name]) != meta["adapted_sha256"][name]:
            problem(f"{name}: post-copy hash mismatch (dtype cast?)")
    logger.info("apply_delta: applied %d tensors from %s (sha256 %s)", len(expected), ckpt_dir, digest[:12])
    return {"applied": sorted(expected), "weights_sha256": digest, "base_model": meta.get("base_model"),
            "base_revision": meta.get("base_revision"), "identity_delta": bool(meta.get("identity_delta", False)),
            "problems": problems}


def delta_summary(ckpt_dir: str | Path) -> Dict[str, Any]:
    meta = load_metadata(ckpt_dir)
    return {
        "path": str(ckpt_dir),
        "sha256": checkpoint_digest(ckpt_dir),
        "base_model": meta.get("base_model"),
        "base_revision": meta.get("base_revision"),
        "n_tensors": len(meta.get("trainable_parameters", [])),
        "n_parameters": meta.get("n_trainable_parameters"),
        "kv_compression": meta.get("kv_compression"),
    }
