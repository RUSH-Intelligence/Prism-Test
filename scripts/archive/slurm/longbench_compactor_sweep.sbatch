#!/bin/bash
# SLURM job-array template for a COMPACTOR-ONLY LongBench KV-compression sweep.
#
# Compactor was rewritten from the kvpress CompactorPress math to the authors'
# reference engine (commit 93206c5, arXiv 2507.08143), so any pre-rewrite
# LongBench compactor numbers are stale and must be regenerated. This launcher
# is the compactor twin of `longbench_sweep.sbatch`: same nexus env, same
# `scripts/longbench_sweep.py` driver, but the method/ratio grid is pinned to
# `--methods compactor --ratios 0.6,0.8,0.9 --skip-full` so it runs exactly the
# 3 compactor cells and nothing else.
#
# Cells (array 0-2):
#   [0] Compactor__r0.6   [1] Compactor__r0.8   [2] Compactor__r0.9
#
# Submit (Llama-3.1-8B-Instruct, default):
#   sbatch scripts/slurm/longbench_compactor_sweep.sbatch
#
# Submit (Ministral-3-3B, FP8->BF16 dequant template; 128k-native, no maxlen override):
#   sbatch --export=ALL,MODEL=mistralai/Ministral-3-3B-Instruct-2512,\
#TEMPLATE=evaluate/evaluate_kv_ministral.yaml \
#       scripts/slurm/longbench_compactor_sweep.sbatch
#
# Smoke (20 rows/subset):
#   sbatch --array=0 --export=ALL,MAX_REQS=20 scripts/slurm/longbench_compactor_sweep.sbatch
#
# Resubmit a single failed cell (e.g. r0.8):
#   sbatch --array=1 scripts/slurm/longbench_compactor_sweep.sbatch
#
# After all array tasks finish, merge per-cell manifest fragments on a login
# node (no GPU) — run once per model:
#   .venv/bin/python scripts/longbench_sweep.py --methods compactor \
#       --ratios 0.6,0.8,0.9 --skip-full --model <MODEL> --resume

#SBATCH --job-name=lb-compactor
#SBATCH --array=0-2
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/longbench_compactor/job_%A/slurm-%a.out
#SBATCH --error=logs/longbench_compactor/job_%A/slurm-%a.err

#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --account=scavenger
# ^ Scavenger required for --cpus-per-task=8 (default QoS caps CPUs at 4).
# Override with `sbatch --partition=tron --qos=medium --account=nexus ...`.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
LOG_DIR="logs/longbench_compactor/job_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
mkdir -p results/longbench_sweep "$LOG_DIR"

# flash-attn was compiled with gcc/15.1.0; LD_PRELOAD forces its libstdc++ to
# load before conda's older one (else `undefined symbol: __cxa_call_terminate`).
# cuda/13.0.2 matches the torch cu130 build.
module load cuda/13.0.2 gcc/15.1.0
export LD_LIBRARY_PATH=/opt/local/stow/gcc-15.1.0/lib64:${LD_LIBRARY_PATH:-}
export LD_PRELOAD=/opt/local/stow/gcc-15.1.0/lib64/libstdc++.so.6

# Pins cuBLAS workspace for torch.use_deterministic_algorithms(True).
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Skip HF datasets cache-validity flock (NFS contention across array tasks).
# The LongBench cache must already be populated (load_dataset once on a login node).
export HF_DATASETS_OFFLINE=1

source .venv/bin/activate

# Overridable via `sbatch --export=ALL,VAR=value`:
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
TEMPLATE="${TEMPLATE:-}"           # empty = sweep default (evaluate/evaluate_kv.yaml);
                                   # set evaluate/evaluate_kv_ministral.yaml for Ministral-3.
MAX_REQS="${MAX_REQS:-}"           # empty = full LongBench set
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}" # empty = template default (both models are 128k-native)

# Pinned compactor grid — kept in sync between the run and the label lookup.
SWEEP_ARGS=(--methods compactor --ratios 0.6,0.8,0.9 --skip-full)

ARGS=("${SWEEP_ARGS[@]}" --model "$MODEL" --cell-index "$SLURM_ARRAY_TASK_ID" --resume)
if [ -n "$TEMPLATE" ]; then
    ARGS+=(--template "$TEMPLATE")
fi
if [ -n "$MAX_REQS" ]; then
    ARGS+=(--max-requests "$MAX_REQS")
fi
if [ -n "$MAX_MODEL_LEN" ]; then
    ARGS+=(--max-model-len "$MAX_MODEL_LEN")
fi

# Resolve the cell_id (e.g. "Compactor__r0.8") for greppable log symlinks.
CELL_LABEL=$(python scripts/longbench_sweep.py "${SWEEP_ARGS[@]}" --model "$MODEL" --dry-run 2>/dev/null \
    | awk -v idx="${SLURM_ARRAY_TASK_ID:-0}" '/^[[:space:]]*\[/ {
            gsub(/\[|\]/, "", $1); if ($1 == idx) { print $2; exit }
        }')
if [ -n "$CELL_LABEL" ]; then
    ln -sf "slurm-${SLURM_ARRAY_TASK_ID}.out" "$LOG_DIR/${CELL_LABEL}.out"
    ln -sf "slurm-${SLURM_ARRAY_TASK_ID}.err" "$LOG_DIR/${CELL_LABEL}.err"
fi

echo "Host:       $(hostname)"
echo "Array task: ${SLURM_ARRAY_TASK_ID:-?} of ${SLURM_ARRAY_JOB_ID:-?}"
echo "Cell:       ${CELL_LABEL:-<unknown>} (task index ${SLURM_ARRAY_TASK_ID:-?})"
echo "Model:      $MODEL"
echo "Template:   ${TEMPLATE:-<sweep default>}"
echo "Log dir:    $LOG_DIR"
echo "GPU:        ${CUDA_VISIBLE_DEVICES:-?}"
echo "Command:    python scripts/longbench_sweep.py ${ARGS[*]}"
echo

srun python scripts/longbench_sweep.py "${ARGS[@]}"
