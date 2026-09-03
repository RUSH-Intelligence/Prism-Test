#!/bin/bash
# SLURM job-array template for the RULER16k Compactor benchmark-control run.
#
# Purpose:
#   The `Llama-3.1-8B` sheet in `Ridge Press.xlsx` reports Compactor scores
#   at 4 prune ratios on RULER-16K. The improved-Ridge sweeps (`Track 1/2/3`)
#   land ~1.7 pts below that sheet's "Ridge (envelope)" number and no Ridge
#   knob we sweep closes the gap. Before continuing to chase Ridge configs,
#   check whether the benchmark itself drifted: rerun Compactor on the
#   current framework and compare each ratio to the sheet.
#
# Target numbers from `Llama-3.1-8B` sheet (RULER-16K Avg, Llama-3.1-8B-Instruct):
#     20% prune: 92.60
#     40% prune: 87.65
#     60% prune: 83.96
#     80% prune: 75.01
#
# 4 ratios × 1 config = 4 cells, indices 0..3. Compactor uses its default
# `blending` (auto → resolves to compression_ratio; sees CompactorSketch
# post_init) and default sink/chunk/sketch_dim.
#
# Llama-3.1-8B-Instruct (default):
#   sbatch scripts/slurm/ruler_compactor_benchmark_control.sbatch
#
# Quick smoke (5 rows per subset):
#   sbatch --array=0-1 --export=ALL,MAX_REQS=5 \
#       scripts/slurm/ruler_compactor_benchmark_control.sbatch
#
# Resubmit a single failed cell:
#   sbatch --array=2 scripts/slurm/ruler_compactor_benchmark_control.sbatch
#
# After all array tasks finish, merge the per-cell manifest fragments by
# re-running the sweep once in resume mode on a login node (no GPU needed):
#   .venv/bin/python scripts/longbench_sweep.py \
#       --benchmark ruler16k --methods compactor --skip-full \
#       --ratios 0.2,0.4,0.6,0.8 \
#       --out-root results/ruler16k_compactor_control/meta-llama--Llama-3.1-8B-Instruct \
#       --resume

#SBATCH --job-name=ruler-compactor-ctrl
#SBATCH --array=0-3
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/ruler_compactor_control/job_%A/slurm-%a.out
#SBATCH --error=logs/ruler_compactor_control/job_%A/slurm-%a.err

#SBATCH --partition=tron
#SBATCH --qos=medium
#SBATCH --account=nexus
# ^ Real (non-scavenger) priority. To fall back to scavenger, submit with:
#     sbatch --partition=scavenger --qos=scavenger --account=scavenger \
#         scripts/slurm/ruler_compactor_benchmark_control.sbatch

set -euo pipefail
set -x
exec 2>&1
echo "[STEP] start preamble on $(hostname) array=${SLURM_ARRAY_TASK_ID:-?}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
LOG_DIR="logs/ruler_compactor_control/job_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
mkdir -p results/ruler16k_compactor_control "$LOG_DIR"
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
RATIOS="${RATIOS:-0.2,0.4,0.6,0.8}"
MAX_REQS="${MAX_REQS:-}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"

MODEL_SLUG="${MODEL//\//--}"
OUT_ROOT="results/ruler16k_compactor_control/${MODEL_SLUG}"

ARGS=(
    --benchmark ruler16k
    --model "$MODEL"
    --template "$TEMPLATE"
    --methods compactor
    --skip-full
    --ratios "$RATIOS"
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
        --methods compactor --skip-full --ratios "$RATIOS" \
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
