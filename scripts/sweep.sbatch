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

# --- Environment (mirrors the retired per-benchmark sbatch files) ------------
# flash-attn was compiled with gcc/15.1.0; LD_PRELOAD forces its libstdc++ to
# load before conda's older one (else `undefined symbol: __cxa_call_terminate`).
module load cuda/13.0.2 gcc/15.1.0
export LD_LIBRARY_PATH=/opt/local/stow/gcc-15.1.0/lib64:${LD_LIBRARY_PATH:-}
export LD_PRELOAD=/opt/local/stow/gcc-15.1.0/lib64/libstdc++.so.6
# Pin cuBLAS workspace so GEMM kernel choice is stable under determinism.
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# Skip HF datasets cache-validity flock (fights sibling array tasks on NFS).
# The dataset cache must already be populated (load_dataset once on a login node).
export HF_DATASETS_OFFLINE=1

source .venv/bin/activate

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
