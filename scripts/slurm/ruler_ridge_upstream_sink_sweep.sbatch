#!/bin/bash
# SLURM job-array template for the RULER16k Ridge upstream-sink verification:
# lock the RidgeSketch sink/local windows to the upstream RidgePress reference
# (sink=4, local=28) and sweep envelope_gamma × ratio × normalize_keys_for_tau.
#
# Purpose:
#   The `Llama-3.1-8B` sheet in `Ridge Press.xlsx` reports "Ridge (envelope)"
#   at ~84.3 (r=0.6) / ~73.3 (r=0.8) on RULER-16K, ~1.5-2 pts above the best
#   config found by `ruler_ridge_2x2_sweep.sbatch`. Suspected cause: the old
#   benchmark used upstream sink=4 / local=28 while our sweeps ran the repo
#   defaults sink=8 / local=64 (see `ridge_explained.md` §6 warning).
#
#   This sweep pins sink=4, local=28 and runs:
#     * `nkF` — reproduce the old-sheet "Ridge (envelope)" baseline
#     * `nkT` — the Track 1 winner (normalize_keys_for_tau=True) on top
#
#   rq axis is dropped — Track 1 already showed rotate_queries is flat.
#
#   11 gammas × 2 ratios × 2 nk = 44 cells, indices 0..43.
#
# Llama-3.1-8B-Instruct (default):
#   sbatch scripts/slurm/ruler_ridge_upstream_sink_sweep.sbatch
#
# Quick smoke (5 rows per subset, first 2 cells):
#   sbatch --array=0-1 --export=ALL,MAX_REQS=5 \
#       scripts/slurm/ruler_ridge_upstream_sink_sweep.sbatch
#
# Resubmit a single failed cell:
#   sbatch --array=17 scripts/slurm/ruler_ridge_upstream_sink_sweep.sbatch
#
# After all array tasks finish, merge the per-cell manifest fragments by
# re-running the sweep once in resume mode on a login node (no GPU needed):
#   .venv/bin/python scripts/longbench_sweep.py \
#       --benchmark ruler16k --methods ridge --skip-full --ratios 0.6,0.8 \
#       --ridge-gammas 0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5 \
#       --ridge-rotate-queries false \
#       --ridge-normalize-keys false,true \
#       --ridge-sink-size 4 --ridge-local-size 28 \
#       --out-root results/ruler16k_sweep_upstream_sink/meta-llama--Llama-3.1-8B-Instruct \
#       --resume

#SBATCH --job-name=ruler-ridge-upstream-sink
#SBATCH --array=0-43
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/ruler_ridge_upstream_sink_sweep/job_%A/slurm-%a.out
#SBATCH --error=logs/ruler_ridge_upstream_sink_sweep/job_%A/slurm-%a.err

#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --account=scavenger
# ^ Scavenger — no per-user CPU cap, so all 44 array tasks can run concurrently
# if scavenger slots are free. Trades priority for parallelism. Override to
# real priority with:
#     sbatch --partition=tron --qos=medium --account=nexus \
#         scripts/slurm/ruler_ridge_upstream_sink_sweep.sbatch

set -euo pipefail
set -x
exec 2>&1
echo "[STEP] start preamble on $(hostname) array=${SLURM_ARRAY_TASK_ID:-?}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
LOG_DIR="logs/ruler_ridge_upstream_sink_sweep/job_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
# Dedicated out-root so manifest.cells/ doesn't collide with prior sweeps.
mkdir -p results/ruler16k_sweep_upstream_sink "$LOG_DIR"
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
RATIOS="${RATIOS:-0.6,0.8}"
RIDGE_GAMMAS="${RIDGE_GAMMAS:-0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5}"
RIDGE_ROTATE_QUERIES="${RIDGE_ROTATE_QUERIES:-false}"
RIDGE_NORMALIZE_KEYS="${RIDGE_NORMALIZE_KEYS:-false,true}"
RIDGE_SINK_SIZE="${RIDGE_SINK_SIZE:-4}"
RIDGE_LOCAL_SIZE="${RIDGE_LOCAL_SIZE:-28}"
MAX_REQS="${MAX_REQS:-}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"

MODEL_SLUG="${MODEL//\//--}"
OUT_ROOT="results/ruler16k_sweep_upstream_sink/${MODEL_SLUG}"

ARGS=(
    --benchmark ruler16k
    --model "$MODEL"
    --template "$TEMPLATE"
    --methods ridge
    --skip-full
    --ratios "$RATIOS"
    --ridge-gammas "$RIDGE_GAMMAS"
    --ridge-rotate-queries "$RIDGE_ROTATE_QUERIES"
    --ridge-normalize-keys "$RIDGE_NORMALIZE_KEYS"
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
        --methods ridge --skip-full --ratios "$RATIOS" \
        --ridge-gammas "$RIDGE_GAMMAS" \
        --ridge-rotate-queries "$RIDGE_ROTATE_QUERIES" \
        --ridge-normalize-keys "$RIDGE_NORMALIZE_KEYS" \
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
