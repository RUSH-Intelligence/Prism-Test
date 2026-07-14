#!/bin/bash
# SLURM job-array template for the LongBench Ridge-improved sweep.
#
# Purpose:
#   The existing `LongBench-Llama-3.1-8B-Instruct` sheet in `Ridge Press.xlsx`
#   shows Compactor beating Ridge (envelope) by 2.5-4.8 pts at every ratio,
#   but that Ridge run predates the Track 3 finding that `normalize_keys_for_tau=True`
#   with upstream sink=4/local=28 gives a real ~2-5 pt lift. This sweep repeats
#   the sheet's Ridge γ grid on LongBench with the Track 3 winning config so we
#   can (a) do the per-task-best-γ vs Compactor comparison the RULER results
#   suggested is fair, and (b) confirm/refute that Ridge closes the LongBench
#   gap the same way it does on RULER-16K.
#
#   Also runs Compactor at the sheet's three ratios as a same-framework
#   control (RULER r=0.8 showed ~1.8 pt Gaussian-sketch drift; verifying
#   Compactor on LongBench too).
#
# Cell grid:
#   7 γ × 4 ratios (Ridge, nkT+sink=4+local=28) = 28 cells
#   + 4 Compactor cells @ 0.6/0.8/0.9/0.95      =  4 cells
#   = 32 cells total, indices 0..31.
# Ratio 0.8 is added on top of the sheet's 0.6/0.9/0.95 to match the RULER-16K
# Track 3 grid — makes Ridge vs Compactor plottable at 4 comparable points.
#
# Llama-3.1-8B-Instruct (default):
#   sbatch scripts/slurm/longbench_ridge_improved_sweep.sbatch
#
# Quick smoke (20 rows per task, first 2 cells):
#   sbatch --array=0-1 --export=ALL,MAX_REQS=20 \
#       scripts/slurm/longbench_ridge_improved_sweep.sbatch
#
# Resubmit a single failed cell:
#   sbatch --array=17 scripts/slurm/longbench_ridge_improved_sweep.sbatch
#
# After all array tasks finish, merge the per-cell manifest fragments (login
# node OK, no GPU needed) then fill Ridge Press.xlsx:
#   .venv/bin/python scripts/longbench_sweep.py \
#       --benchmark longbench --methods ridge,compactor --skip-full \
#       --ratios 0.6,0.8,0.9,0.95 \
#       --ridge-gammas 0,0.5,1,1.5,2,2.5,3 \
#       --ridge-rotate-queries false --ridge-normalize-keys true \
#       --ridge-sink-size 4 --ridge-local-size 28 \
#       --out-root results/longbench_ridge_improved/meta-llama--Llama-3.1-8B-Instruct \
#       --resume
#   .venv/bin/python scripts/longbench_to_xlsx.py \
#       --manifest results/longbench_ridge_improved/meta-llama--Llama-3.1-8B-Instruct/manifest.json \
#       --sheet LongBench-Ridge-Improved-Llama-3.1-8B

#SBATCH --job-name=lb-ridge-improved
#SBATCH --array=0-31
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/longbench_ridge_improved/job_%A/slurm-%a.out
#SBATCH --error=logs/longbench_ridge_improved/job_%A/slurm-%a.err

#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --account=scavenger
# ^ Scavenger — no per-user CPU cap, so all 32 array tasks can run concurrently
# if scavenger slots are free. Trades priority for parallelism. Override to
# real priority with:
#     sbatch --partition=tron --qos=medium --account=nexus \
#         scripts/slurm/longbench_ridge_improved_sweep.sbatch

set -euo pipefail
set -x
exec 2>&1
echo "[STEP] start preamble on $(hostname) array=${SLURM_ARRAY_TASK_ID:-?}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
LOG_DIR="logs/longbench_ridge_improved/job_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
mkdir -p results/longbench_ridge_improved "$LOG_DIR"
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
OUT_ROOT="results/longbench_ridge_improved/${MODEL_SLUG}"

ARGS=(
    --benchmark longbench
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
        --benchmark longbench --model "$MODEL" --template "$TEMPLATE" \
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
