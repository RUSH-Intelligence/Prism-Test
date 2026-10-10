"""Reproducibility metadata, seeding and determinism for KV-recovery runs.

Seeding and the determinism flags are the runner's own static helpers
(``EvalRunner._set_seed`` / ``EvalRunner._enable_determinism``) so training and
evaluation can never drift apart. ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` must be
exported BEFORE CUDA initialises (the sbatch files do this); we only check it.

The git commit is resolved without a git binary (compute nodes have none) by
parsing ``.git`` — including the ``gitdir:`` indirection of linked worktrees.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import socket
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

TRACKED_PACKAGES = (
    "torch", "transformers", "safetensors", "accelerate", "numpy", "datasets",
    "huggingface_hub", "flash_attn", "flash-linear-attention", "causal-conv1d", "kernels",
)
CODE_GLOBS = (
    "eval_harness/kv_recovery/*.py",
    "scripts/*kv_recovery*.py",
    "scripts/prepare_kv_recovery_data.py",
    "scripts/precompute_teacher_states.py",
    "scripts/measure_representation_alignment.py",
    "configs/kv_recovery/*.yaml",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# git without a git binary
# ---------------------------------------------------------------------------
def _read(p: Path) -> Optional[str]:
    try:
        return p.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _lookup_ref(ref: str, gitdir: Path, commondir: Path) -> Optional[str]:
    for base in (gitdir, commondir):
        val = _read(base / ref)
        if val:
            return val.split()[0]
    packed = _read(commondir / "packed-refs")
    if packed:
        for line in packed.splitlines():
            if line.startswith("#") or line.startswith("^") or not line.strip():
                continue
            parts = line.split()
            if len(parts) == 2 and parts[1] == ref:
                return parts[0]
    return None


def parse_git_head(root: Path) -> Dict[str, Any]:
    """``{"git_commit", "git_branch", "git_dir"}`` parsed from ``<root>/.git`` (file or dir)."""
    git_path = root / ".git"
    if not git_path.exists():
        return {"git_commit": None, "git_branch": None, "git_dir": None}
    if git_path.is_file():
        content = _read(git_path) or ""
        if not content.startswith("gitdir:"):
            return {"git_commit": None, "git_branch": None, "git_dir": None}
        gitdir = Path(content.split(":", 1)[1].strip())
        if not gitdir.is_absolute():
            gitdir = (root / gitdir).resolve()
    else:
        gitdir = git_path
    commondir = gitdir
    cd = _read(gitdir / "commondir")
    if cd:
        commondir = (gitdir / cd).resolve()
    head = _read(gitdir / "HEAD") or ""
    branch = None
    commit: Optional[str] = None
    if head.startswith("ref:"):
        ref = head.split(":", 1)[1].strip()
        branch = ref[len("refs/heads/"):] if ref.startswith("refs/heads/") else ref
        commit = _lookup_ref(ref, gitdir, commondir)
    elif head:
        commit = head.split()[0]
    return {"git_commit": commit, "git_branch": branch, "git_dir": str(gitdir)}


def git_info(root: Optional[Path] = None) -> Dict[str, Any]:
    """Commit + branch (+ dirty flag when a git binary is available)."""
    root = root or repo_root()
    info = parse_git_head(root)
    info["git_dirty"] = None
    try:
        import subprocess

        r = subprocess.run(["git", "status", "--porcelain"], cwd=str(root),
                           capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            info["git_dirty"] = bool(r.stdout.strip())
            if info["git_commit"] is None:
                c = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(root),
                                   capture_output=True, text=True, timeout=10)
                if c.returncode == 0:
                    info["git_commit"] = c.stdout.strip()
    except Exception:
        pass
    return info


# ---------------------------------------------------------------------------
# packages / device / code
# ---------------------------------------------------------------------------
def package_versions() -> Dict[str, Any]:
    out: Dict[str, Any] = {"python": platform.python_version()}
    for name in TRACKED_PACKAGES:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    try:
        import torch

        out["torch_cuda"] = torch.version.cuda
        out["cudnn"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    except Exception:
        pass
    return out


def device_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    try:
        import torch

        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            info["gpu_name"] = torch.cuda.get_device_name(idx)
            info["gpu_capability"] = ".".join(map(str, torch.cuda.get_device_capability(idx)))
            info["gpu_total_memory_gib"] = round(torch.cuda.get_device_properties(idx).total_memory / 2**30, 2)
            info["n_gpus"] = torch.cuda.device_count()
        else:
            info["gpu_name"] = None
    except Exception:
        info["gpu_name"] = None
    return info


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_sha256(root: Optional[Path] = None, globs: Iterable[str] = CODE_GLOBS) -> Dict[str, str]:
    root = root or repo_root()
    out: Dict[str, str] = {}
    for pattern in globs:
        for p in sorted(root.glob(pattern)):
            if p.is_file():
                out[str(p.relative_to(root))] = sha256_file(p)
    return out


# ---------------------------------------------------------------------------
# seeding / determinism (the runner's own helpers)
# ---------------------------------------------------------------------------
def seed_everything(seed: int) -> None:
    from eval_harness.runner import EvalRunner

    EvalRunner._set_seed(int(seed))


def configure_determinism(enabled: bool) -> Dict[str, Any]:
    """Apply the runner's deterministic-mode pins and report the resulting flags."""
    import torch

    if enabled:
        from eval_harness.runner import EvalRunner

        EvalRunner._enable_determinism()
    flags: Dict[str, Any] = {
        "enabled": bool(enabled),
        "use_deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    if torch.cuda.is_available():
        flags.update({
            "flash_sdp": torch.backends.cuda.flash_sdp_enabled(),
            "mem_efficient_sdp": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "math_sdp": torch.backends.cuda.math_sdp_enabled(),
            "cudnn_sdp": (torch.backends.cuda.cudnn_sdp_enabled()
                          if hasattr(torch.backends.cuda, "cudnn_sdp_enabled") else None),
        })
        if enabled and not os.environ.get("CUBLAS_WORKSPACE_CONFIG"):
            import logging

            logging.getLogger(__name__).warning(
                "deterministic=True but CUBLAS_WORKSPACE_CONFIG is not set; cuBLAS GEMM "
                "selection is not pinned (export CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA init)."
            )
    return flags


def provenance(root: Optional[Path] = None) -> Dict[str, Any]:
    return {
        "git": git_info(root),
        "packages": package_versions(),
        "hardware": device_info(),
        "code_sha256": code_sha256(root),
    }
