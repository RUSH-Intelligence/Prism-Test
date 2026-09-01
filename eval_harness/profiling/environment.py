"""Provenance capture. Every probe is individually guarded: this must NEVER raise.

A perf artifact without an environment block is unusable three weeks later on a
different node, so this runs even when CUDA, git and nvidia-smi are all absent.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def _run(cmd) -> Optional[str]:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return out.stdout.strip() or None
    except Exception:
        return None


def capture_environment(tag: str = "") -> Dict[str, Any]:
    import torch

    env: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hostname": _safe(platform.node),
        "platform": _safe(platform.platform),
        "python_version": sys.version.split()[0],
        "torch_version": _safe(lambda: torch.__version__),
        "cuda_runtime_version": _safe(lambda: torch.version.cuda),
        "cuda_available": _safe(lambda: torch.cuda.is_available(), False),
        "tag": tag,
    }
    for mod in ("transformers", "flash_attn"):
        env[f"{mod}_version"] = _safe(lambda m=mod: __import__(m).__version__)

    if env.get("cuda_available"):
        env["gpu_name"] = _safe(lambda: torch.cuda.get_device_name(0))
        env["gpu_count"] = _safe(torch.cuda.device_count)
        env["gpu_capability"] = _safe(lambda: ".".join(map(str, torch.cuda.get_device_capability(0))))
        env["gpu_total_memory_bytes"] = _safe(
            lambda: int(torch.cuda.get_device_properties(0).total_memory)
        )
        env["cuda_driver_version"] = _run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
        )

    # `module purge` on the compute nodes removes git from PATH, so the launcher
    # stamps the SHA into the job env; fall back to asking git directly.
    env["git_sha"] = os.environ.get("PRISM_GIT_SHA") or _run(
        ["git", "-C", REPO_ROOT, "rev-parse", "HEAD"])
    dirty_env = os.environ.get("PRISM_GIT_DIRTY")
    if dirty_env is not None:
        env["git_dirty"] = dirty_env == "1"
    else:
        env["git_dirty"] = bool(_run(["git", "-C", REPO_ROOT, "status", "--porcelain"]))

    for k in ("SLURM_JOB_ID", "SLURM_JOB_NAME", "PYTORCH_CUDA_ALLOC_CONF",
              "OMP_NUM_THREADS", "CUDA_LAUNCH_BLOCKING"):
        env[k.lower()] = os.environ.get(k)

    # Backend flags that change which kernel we are timing.
    env["backend_flags"] = _safe(
        lambda: {
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
        {},
    )
    return env


def gpu_processes() -> Optional[str]:
    """PIDs currently holding the GPU. Non-empty (besides ours) invalidates timing."""
    return _run(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader"])
