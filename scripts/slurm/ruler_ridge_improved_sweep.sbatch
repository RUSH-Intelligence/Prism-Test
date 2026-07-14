#!/bin/bash
# SLURM job-array template for the RULER16k Ridge-improved sweep — mirror of
# `longbench_ridge_improved_sweep.sbatch` so RULER and LongBench numbers are
# apples-to-apples on the same γ × ratio grid with the same Compactor control.
#
# Purpose:
#   Runs the Track 3 winner (RidgeSketch with normalize_keys_for_tau=True and
#   the upstream-reference sink=4 / local=28 windows) on RULER-16K over the
#   same γ × ratio grid that the LongBench-improved sweep uses, so per-benchmark
#   Ridge-vs-Compactor tables line up cell-for-cell. Also runs Compactor at the
#   same 4 ratios as a same-framework control.
#
# Cell grid:
#   7 γ × 4 ratios (Ridge, nkT+sink=4+local=28) = 28 cells
#   + 4 Compactor cells @ 0.6/0.8/0.9/0.95      =  4 cells
#   = 32 cells total, indices 0..31.
# (Identical grid to longbench_ridge_improved_sweep.sbatch.)
#
# Llama-3.1-8B-Instruct (default):
#   sbatch scripts/slurm/ruler_ridge_improved_sweep.sbatch
#
# Ministral-3-3B-Instruct-2512:
#   sbatch --export=ALL,MODEL=mistralai/Ministral-3-3B-Instruct-2512,TEMPLATE=evaluate/evaluate_kv_ministral.yaml \
#       scripts/slurm/ruler_ridge_improved_sweep.sbatch
#
# Quick smoke (5 rows per subset, first 2 cells):
#   sbatch --array=0-1 --export=ALL,MAX_REQS=5 \
#       scripts/slurm/ruler_ridge_improved_sweep.sbatch
#
# Resubmit a single failed cell:
#   sbatch --array=17 scripts/slurm/ruler_ridge_improved_sweep.sbatch
#
# After all array tasks finish, merge the per-cell manifest fragments (login
# node OK, no GPU needed):
#   .venv/bin/python scripts/longbench_sweep.py \
#       --benchmark ruler16k --methods ridge,compactor --skip-full \
#       --ratios 0.6,0.8,0.9,0.95 \
#       --ridge-gammas 0,0.5,1,1.5,2,2.5,3 \
#       --ridge-rotate-queries false --ridge-normalize-keys true \
#       --ridge-sink-size 4 --ridge-local-size 28 \
#       --out-root results/ruler16k_ridge_improved/<MODEL_SLUG> \
#       --resume

#SBATCH --job-name=ruler-ridge-improved
#SBATCH --array=0-31
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/ruler_ridge_improved/job_%A/slurm-%a.out
#SBATCH --error=logs/ruler_ridge_improved/job_%A/slurm-%a.err

#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --account=scavenger
# ^ Scavenger — no per-user CPU cap, so all 32 array tasks can run concurrently
# if scavenger slots are free. Trades priority for parallelism. Override to
# real priority with:
#     sbatch --partition=tron --qos=medium --account=nexus \
#         scripts/slurm/ruler_ridge_improved_sweep.sbatch

set -euo pipefail
set -x
exec 2>&1
echo "[STEP] start preamble on $(hostname) array=${SLURM_ARRAY_TASK_ID:-?}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
LOG_DIR="logs/ruler_ridge_improved/job_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
mkdir -p results/ruler16k_ridge_improved "$LOG_DIR"
echo "[STEP] cwd+mkdir ok"

module load cuda/13.0.2 gcc/15.1.0
echo "[STEP] module load ok"
export LD_LIBRARY_PATH=/opt/local/stow/gcc-15.1.0/lib64:${LD_LIBRARY_PATH:-}
export LD_PRELOAD=/opt/local/stow/gcc-15.1.0/lib64/libstdc++.so.6
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_DATASETS_OFFLINE=1
echo "[STEP] env exports ok"

JITTER=$(( RANDOM % 16 ))
echo "[STEP] sleeping ${JITTER}s jitter"
sleep "$JITTER"

if [ ! -f .venv/bin/activate ]; then
    echo "[FATAL] .venv/bin/activate missing on $(hostname)"
    exit 99
fi
source .venv/bin/activate
echo "[STEP] venv activate ok (python=$(command -v python))"

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
TEMPLATE="${TEMPLATE:-evaluate/evaluate_kv.yaml}"
RATIOS="${RATIOS:-0.6,0.8,0.9,0.95}"
RIDGE_GAMMAS="${RIDGE_GAMMAS:-0,0.5,1,1.5,2,2.5,3}"
RIDGE_SINK_SIZE="${RIDGE_SINK_SIZE:-4}"
RIDGE_LOCAL_SIZE="${RIDGE_LOCAL_SIZE:-28}"
MAX_REQS="${MAX_REQS:-}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"

MODEL_SLUG="${MODEL//\//--}"
OUT_ROOT="results/ruler16k_ridge_improved/${MODEL_SLUG}"

ARGS=(
    --benchmark ruler16k
    --model "$MODEL"
    --template "$TEMPLATE"
    --methods ridge,compactor
    --skip-full
    --ratios "$RATIOS"
    --ridge-gammas "$RIDGE_GAMMAS"
    --ridge-rotate-queries false
    --ridge-normalize-keys true
    --ridge-sink-size "$RIDGE_SINK_SIZE"
    --ridge-local-size "$RIDGE_LOCAL_SIZE"
    --out-root "$OUT_ROOT"
    --cell-index "$SLURM_ARRAY_TASK_ID"
    --resume
)
if [ -n "$MAX_REQS" ]; then
    ARGS+=(--max-requests "$MAX_REQS")
fi
if [ -n "$MAX_MODEL_LEN" ]; then
    ARGS+=(--max-model-len "$MAX_MODEL_LEN")
fi

set +o pipefail
CELL_LABEL=$(python scripts/longbench_sweep.py \
        --benchmark ruler16k --model "$MODEL" --template "$TEMPLATE" \
        --methods ridge,compactor --skip-full --ratios "$RATIOS" \
        --ridge-gammas "$RIDGE_GAMMAS" \
        --ridge-rotate-queries false --ridge-normalize-keys true \
        --ridge-sink-size "$RIDGE_SINK_SIZE" \
        --ridge-local-size "$RIDGE_LOCAL_SIZE" \
        --dry-run 2>/dev/null \
    | awk -v idx="${SLURM_ARRAY_TASK_ID:-0}" '/^[[:space:]]*\[/ {
            gsub(/\[|\]/, "", $1); if ($1 == idx) { print $2; exit }
        }')
set -o pipefail
echo "[STEP] cell label resolved: ${CELL_LABEL:-<unknown>}"
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
