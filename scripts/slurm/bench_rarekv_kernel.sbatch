#!/bin/bash
#SBATCH --output=/scratch/sj157/kvruler_logs/%x_%j.out
#SBATCH --error=/scratch/sj157/kvruler_logs/%x_%j.err
#SBATCH --job-name=rarekv_kernel
#SBATCH --account=as143
#SBATCH --partition=commons
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32GB
#SBATCH --gres=gpu:h200:1
#SBATCH --time=2:00:00
#SBATCH --exclude=bg3u24g1
set -euo pipefail
module purge; module load GCCcore/13.3.0 CUDA/12.9.1
source /scratch/sj157/prism_env/bin/activate
export LD_PRELOAD="/opt/apps/software/OpenSSL/3/lib64/libcrypto.so.3:/opt/apps/software/OpenSSL/3/lib64/libssl.so.3${LD_PRELOAD:+:$LD_PRELOAD}"
export PYTHONUNBUFFERED=1 HF_HOME=/scratch/sj157/hf_home
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# The rarekv kernels are JIT-built by torch. The cache MUST be a persistent path:
# a per-job scratch dir rebuilds them inside the timed job. There are TWO
# extensions: `rarekv_kernels` (collide + Tier-1 pack, a few minutes of nvcc) and
# `rarekv_fused_kernels` (Tier 2 -- 90 template instantiations, several times
# longer). This bench forces BOTH builds up front, before any timed cell.
# NOTE: the default extension was renamed rarekv_collide -> rarekv_kernels and
# dropped --use_fast_math, so the FIRST job after that change pays a full
# rebuild -- warm it with a 1-cell run before any timed sweep. main() forces both
# builds before the first timed cell, so a warm cache only saves wall clock.
export TORCH_EXTENSIONS_DIR=/scratch/sj157/torch_extensions
# TF32 cannot touch a bf16 GEMM (bf16's 8 significand bits sit inside tf32's 10),
# but it does reach the fp32 fallback path, where it raises the projection
# sign-flip rate ~1000x. Pinned off everywhere for comparability.
export NVIDIA_TF32_OVERRIDE=0
# NOTE: do NOT set TORCH_CUDA_ARCH_LIST here. Every other job sharing
# TORCH_EXTENSIONS_DIR (bench_rarekv_dense, bench_blockrare_dense, the eval jobs)
# leaves it unset and gets torch's auto-detected `-gencode arch=compute_90,
# code=sm_90`. A differing -gencode changes the compile command, ninja rebuilds
# the whole extension, and the two flag sets then thrash the shared cache -- which
# is exactly what the persistent path above exists to prevent. Nothing here needs
# the `a` suffix: the only arch-specific instruction is mma.m16n8k16, which is
# sm_80+; there is no wgmma, no TMA and no cluster intrinsic.
JB=/scratch/sj157/kvruler_job_${SLURM_JOB_ID}
export TMPDIR="$JB/tmp" TRITON_CACHE_DIR="$JB/triton" XDG_CACHE_HOME="$JB/xdg"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"
trap 'rm -rf "$JB"' EXIT
cd /scratch/sj157/Prism-Test
export PYTHONPATH=/scratch/sj157/Prism-Test:${PYTHONPATH:-}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python scripts/bench_rarekv_kernel.py "$@"
