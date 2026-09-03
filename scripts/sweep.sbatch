#!/bin/bash
# Generic sweep array task. NOT submitted by hand — scripts/submit_sweep.sh
# (via scripts/sweep.py --submit) submits one array of this per (model x
# benchmark) pair, supplying --array / --partition / --gres / etc. on the
# sbatch command line and passing the three SWEEP_* vars through --export.
#
# Each array task runs ONE cell:
#   python scripts/sweep.py --config $SWEEP_CONFIG \
#       --model $SWEEP_MODEL --benchmark $SWEEP_BENCHMARK \
#       --cell-index $SLURM_ARRAY_TASK_ID --resume

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

: "${SWEEP_CONFIG:?SWEEP_CONFIG not set (submit via scripts/submit_sweep.sh)}"
: "${SWEEP_MODEL:?SWEEP_MODEL not set}"
: "${SWEEP_BENCHMARK:?SWEEP_BENCHMARK not set}"

# --- Environment ------------------------------------------------------------
# Per-cluster values come from the `env:` block of sweep.yaml, handed in by
# scripts/sweep.py as SWEEP_* vars (see that file to change them per-cluster).
# Everything below no-ops gracefully when a var is unset.
#
# Load cluster modules (e.g. cuda/gcc for flash-attn's libstdc++). Space list.
if [ -n "${SWEEP_MODULES:-}" ]; then
    module load ${SWEEP_MODULES}
fi
# Prepend extra library dirs to the search path (colon list).
if [ -n "${SWEEP_LD_LIBRARY_PATH:-}" ]; then
    export LD_LIBRARY_PATH="${SWEEP_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
# Force-load libs FIRST, but KEEP whatever the cluster already preloaded:
# APPEND the existing value instead of clobbering it (else a cluster-injected
# preload like xalt's OpenSSL fix is silently lost).
if [ -n "${SWEEP_LD_PRELOAD:-}" ]; then
    export LD_PRELOAD="${SWEEP_LD_PRELOAD}${LD_PRELOAD:+:$LD_PRELOAD}"
fi
# Pin cuBLAS workspace so GEMM kernel choice is stable under determinism.
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# Skip HF datasets cache-validity flock (fights sibling array tasks on NFS).
# The dataset cache must already be populated (load_dataset once on a login node).
export HF_DATASETS_OFFLINE=1

# Activate the Python env (path from sweep.yaml env.venv; default .venv).
source "${SWEEP_VENV:-.venv}/bin/activate"

echo "Host:       $(hostname)"
echo "Array task: ${SLURM_ARRAY_TASK_ID:-?} of ${SLURM_ARRAY_JOB_ID:-?}"
echo "Config:     $SWEEP_CONFIG"
echo "Pair:       $SWEEP_MODEL / $SWEEP_BENCHMARK"
echo "GPU:        ${CUDA_VISIBLE_DEVICES:-?}"
echo

srun python scripts/sweep.py \
    --config "$SWEEP_CONFIG" \
    --model "$SWEEP_MODEL" \
    --benchmark "$SWEEP_BENCHMARK" \
    --cell-index "${SLURM_ARRAY_TASK_ID:-0}" \
    --resume
