#!/bin/bash
# SLURM job-array template for the LongBench Verified-KV sweep — LongBench twin
# of `ruler_verified_sweep.sbatch` (same det_fraction × γ × ratio grid, same
# Track-3 Ridge inner), so RULER and LongBench verified numbers are
# apples-to-apples on the same grid.
#
# Purpose:
#   VerifiedSketch wraps Ridge and splits the kept budget: `det_fraction` from
#   Ridge's top-k, the rest uniform-random from Ridge's evicted pool. Sweeps
#   det_fraction to measure how much signal Ridge's tail carries on LongBench's
#   16 English tasks, with envelope_gamma swept for the per-task-best-γ recipe.
#   See notes/verified_kv_compression_idea.md.
#
# Cell grid (108 cells, indices 0..107):
#   det=1.0 : 7 γ × 3 ratios                 = 21   (== plain Ridge reference)
#   det∈{0.9,0.8,0.7,0.5} : 7 γ × 4 × 3      = 84
#   det=0.0 : 1 (γ-free anchor) × 3 ratios   =  3   (random + forced sink/local)
#
# Llama-3.1-8B-Instruct (default):
#   sbatch scripts/slurm/longbench_verified_sweep.sbatch
#
# Quick smoke (20 rows per task, first 2 cells):
#   sbatch --array=0-1 --export=ALL,MAX_REQS=20 \
#       scripts/slurm/longbench_verified_sweep.sbatch
#
# Resubmit a single failed cell:
#   sbatch --array=42 scripts/slurm/longbench_verified_sweep.sbatch
#
# After all array tasks finish, merge the per-cell manifest fragments (login
# node OK, no GPU needed):
#   .venv/bin/python scripts/longbench_sweep.py \
#       --benchmark longbench --methods verified --skip-full \
#       --ratios 0.6,0.8,0.9 \
#       --ridge-gammas 0,0.5,1,1.5,2,2.5,3 \
#       --ridge-rotate-queries false --ridge-normalize-keys true \
#       --ridge-sink-size 4 --ridge-local-size 28 \
#       --verified-det-fractions 1.0,0.9,0.8,0.7,0.5,0.0 \
#       --out-root results/longbench_verified/meta-llama--Llama-3.1-8B-Instruct \
#       --resume

#SBATCH --job-name=lb-verified
#SBATCH --array=0-107
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/longbench_verified/job_%A/slurm-%a.out
#SBATCH --error=logs/longbench_verified/job_%A/slurm-%a.err

#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --account=scavenger
# ^ Scavenger — no per-user CPU cap, so all array tasks can run concurrently if
# scavenger slots are free. Trades priority for parallelism. Override to real
# priority with:
#     sbatch --partition=tron --qos=medium --account=nexus \
#         scripts/slurm/longbench_verified_sweep.sbatch

set -euo pipefail
set -x
exec 2>&1
echo "[STEP] start preamble on $(hostname) array=${SLURM_ARRAY_TASK_ID:-?}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
LOG_DIR="logs/longbench_verified/job_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
mkdir -p results/longbench_verified "$LOG_DIR"
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
RATIOS="${RATIOS:-0.6,0.8,0.9}"
RIDGE_GAMMAS="${RIDGE_GAMMAS:-0,0.5,1,1.5,2,2.5,3}"
RIDGE_SINK_SIZE="${RIDGE_SINK_SIZE:-4}"
RIDGE_LOCAL_SIZE="${RIDGE_LOCAL_SIZE:-28}"
DET_FRACTIONS="${DET_FRACTIONS:-1.0,0.9,0.8,0.7,0.5,0.0}"
MAX_REQS="${MAX_REQS:-}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"

MODEL_SLUG="${MODEL//\//--}"
OUT_ROOT="results/longbench_verified/${MODEL_SLUG}"

ARGS=(
    --benchmark longbench
    --model "$MODEL"
    --template "$TEMPLATE"
    --methods verified
    --skip-full
    --ratios "$RATIOS"
    --ridge-gammas "$RIDGE_GAMMAS"
    --ridge-rotate-queries false
    --ridge-normalize-keys true
    --ridge-sink-size "$RIDGE_SINK_SIZE"
    --ridge-local-size "$RIDGE_LOCAL_SIZE"
    --verified-det-fractions "$DET_FRACTIONS"
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
        --benchmark longbench --model "$MODEL" --template "$TEMPLATE" \
        --methods verified --skip-full --ratios "$RATIOS" \
        --ridge-gammas "$RIDGE_GAMMAS" \
        --ridge-rotate-queries false --ridge-normalize-keys true \
        --ridge-sink-size "$RIDGE_SINK_SIZE" \
        --ridge-local-size "$RIDGE_LOCAL_SIZE" \
        --verified-det-fractions "$DET_FRACTIONS" \
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
