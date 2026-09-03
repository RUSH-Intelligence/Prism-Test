#!/bin/bash
# SLURM job-array template for the RULER16k Ridge gamma × ridge_lambda ablation.
#
# Each array task runs ONE (gamma, lambda) cell of the Ridge press at a fixed
# compression ratio (default 0.8 → keep 20%) on RULER16k. The default grid is
# 11 gammas × 4 lambdas × 1 ratio = 44 cells, indices 0..43.
#
# Llama-3.1-8B-Instruct (default):
#   sbatch scripts/slurm/ruler_ridge_sweep.sbatch
#
# Ministral-3 (FP8 → BF16 dequantize template):
#   sbatch --export=ALL,MODEL=mistralai/Ministral-3-3B-Instruct-2512,TEMPLATE=evaluate/evaluate_kv_ministral.yaml \
#       scripts/slurm/ruler_ridge_sweep.sbatch
#
# Quick smoke (5 rows per subset):
#   sbatch --array=0-4 --export=ALL,MAX_REQS=5 scripts/slurm/ruler_ridge_sweep.sbatch
#
# Resubmit a single failed cell:
#   sbatch --array=17 scripts/slurm/ruler_ridge_sweep.sbatch
#
# After all array tasks finish, merge the per-cell manifest fragments by re-
# running the sweep once in resume mode on a login node (no GPU needed):
#   .venv/bin/python scripts/longbench_sweep.py \
#       --benchmark ruler16k --methods ridge --skip-full --ratios 0.8 \
#       --ridge-gammas 0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5 \
#       --ridge-lambdas 1e-1,1e-2,1e-4,1e-5 \
#       --resume

#SBATCH --job-name=ruler-ridge
#SBATCH --array=0-43
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/ruler_ridge_sweep/job_%A/slurm-%a.out
#SBATCH --error=logs/ruler_ridge_sweep/job_%A/slurm-%a.err

# --- Cluster-specific knobs ---------------------------------------------------
# Same scavenger setup as the LongBench template; override at submit time if
# your cluster uses different partition/qos names.
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --account=scavenger
# -----------------------------------------------------------------------------

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
LOG_DIR="logs/ruler_ridge_sweep/job_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
mkdir -p results/ruler16k_sweep "$LOG_DIR"

# Same module env as longbench_sweep.sbatch (see that file for the why).
module load cuda/13.0.2 gcc/15.1.0
export LD_LIBRARY_PATH=/opt/local/stow/gcc-15.1.0/lib64:${LD_LIBRARY_PATH:-}
export LD_PRELOAD=/opt/local/stow/gcc-15.1.0/lib64/libstdc++.so.6
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_DATASETS_OFFLINE=1

source .venv/bin/activate

# Overridable via `sbatch --export=ALL,VAR=value`:
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
TEMPLATE="${TEMPLATE:-evaluate/evaluate_kv.yaml}"
RATIO="${RATIO:-0.8}"
RIDGE_GAMMAS="${RIDGE_GAMMAS:-0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5}"
RIDGE_LAMBDAS="${RIDGE_LAMBDAS:-1e-1,1e-2,1e-4,1e-5}"
MAX_REQS="${MAX_REQS:-}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"

ARGS=(
    --benchmark ruler16k
    --model "$MODEL"
    --template "$TEMPLATE"
    --methods ridge          # filters out the 8 standard methods → ridge-only
    --skip-full              # no Full baseline (this is a Ridge ablation)
    --ratios "$RATIO"
    --ridge-gammas "$RIDGE_GAMMAS"
    --ridge-lambdas "$RIDGE_LAMBDAS"
    --cell-index "$SLURM_ARRAY_TASK_ID"
    --resume
)
if [ -n "$MAX_REQS" ]; then
    ARGS+=(--max-requests "$MAX_REQS")
fi
if [ -n "$MAX_MODEL_LEN" ]; then
    ARGS+=(--max-model-len "$MAX_MODEL_LEN")
fi

CELL_LABEL=$(python scripts/longbench_sweep.py \
        --benchmark ruler16k --model "$MODEL" --template "$TEMPLATE" \
        --methods ridge --skip-full --ratios "$RATIO" \
        --ridge-gammas "$RIDGE_GAMMAS" --ridge-lambdas "$RIDGE_LAMBDAS" \
        --dry-run 2>/dev/null \
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
echo "Template:   $TEMPLATE"
echo "Log dir:    $LOG_DIR"
echo "GPU:        ${CUDA_VISIBLE_DEVICES:-?}"
echo "Command:    python scripts/longbench_sweep.py ${ARGS[*]}"
echo

srun python scripts/longbench_sweep.py "${ARGS[@]}"
