#!/bin/bash
# SLURM job-array template for the RULER16k Ridge track-2 query-subset sweep.
#
# Track-2 tests "give omega a smaller, better-chosen set of queries" while
# tau stays as-is. nk=True and rq=False are LOCKED (best Track-1 corner)
# so we never accidentally launch with the wrong tau/omega settings; the
# only swept axes here are gamma, omega_query_selector, omega_query_fraction,
# and ratio.
#
# Selectors:
#   top_norm  (qtop) — keep top-K queries by L2 norm
#   leverage  (qlev) — ridge-leverage coreset over Q itself (diversity)
#   random    (qrnd) — uniform random K queries (sanity baseline)
#
# Cross-product:
#   11 gammas × 2 ratios × 3 selectors × 4 fractions = 264 cells, indices 0..263.
# ridge_lambda fixed at press default 1e-4 (Track 1 showed it's flat).
#
# Llama-3.1-8B-Instruct (default):
#   sbatch scripts/slurm/ruler_ridge_track2_sweep.sbatch
#
# Ministral-3 (FP8 → BF16 dequantize template):
#   sbatch --export=ALL,MODEL=mistralai/Ministral-3-3B-Instruct-2512,TEMPLATE=evaluate/evaluate_kv_ministral.yaml \
#       scripts/slurm/ruler_ridge_track2_sweep.sbatch
#
# Quick smoke (5 rows per subset):
#   sbatch --array=0-4 --export=ALL,MAX_REQS=5 scripts/slurm/ruler_ridge_track2_sweep.sbatch
#
# Resubmit a single failed cell:
#   sbatch --array=37 scripts/slurm/ruler_ridge_track2_sweep.sbatch
#
# After all array tasks finish, merge the per-cell manifest fragments by
# re-running the sweep once in resume mode on a login node (no GPU needed):
#   .venv/bin/python scripts/longbench_sweep.py \
#       --benchmark ruler16k --methods ridge --skip-full --ratios 0.6,0.8 \
#       --ridge-gammas 0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5 \
#       --ridge-rotate-queries false --ridge-normalize-keys true \
#       --ridge-omega-selectors top_norm,leverage,random \
#       --ridge-omega-fractions 0.1,0.25,0.5,0.75 \
#       --out-root results/ruler16k_sweep_track2/meta-llama--Llama-3.1-8B-Instruct \
#       --resume

#SBATCH --job-name=ruler-ridge-track2
#SBATCH --array=0-263
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/ruler_ridge_track2_sweep/job_%A/slurm-%a.out
#SBATCH --error=logs/ruler_ridge_track2_sweep/job_%A/slurm-%a.err

#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --account=scavenger

set -euo pipefail
# Trace every command to stderr so we can pinpoint silent failures in the
# preamble. Step markers below add belt-and-suspenders for diagnosing
# requeue-time races (multiple array tasks landing on the same node at once).
set -x
exec 2>&1
echo "[STEP] start preamble on $(hostname) array=${SLURM_ARRAY_TASK_ID:-?}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
LOG_DIR="logs/ruler_ridge_track2_sweep/job_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
# Dedicated out-root so this sweep doesn't share manifest.cells/ with prior
# sweeps (track-1 lambda / 2x2). Same cell index numbers map to different
# cell_ids between sweeps; sharing manifest.cells would corrupt either.
mkdir -p results/ruler16k_sweep_track2 "$LOG_DIR"
echo "[STEP] cwd+mkdir ok"

module load cuda/13.0.2 gcc/15.1.0
echo "[STEP] module load ok"
export LD_LIBRARY_PATH=/opt/local/stow/gcc-15.1.0/lib64:${LD_LIBRARY_PATH:-}
export LD_PRELOAD=/opt/local/stow/gcc-15.1.0/lib64/libstdc++.so.6
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_DATASETS_OFFLINE=1
echo "[STEP] env exports ok"

# Random jitter (0-15s) to break startup races when SLURM lands many array
# tasks on one node at the same instant — multiple `module load` / venv
# activations colliding has been observed to silently exit ~120.
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
# Track-1 best corner: nk=True, rq=False. Hardcoded as defaults so an
# accidental launch can't drift; override via --export=ALL,RIDGE_*=... if
# you really mean to.
RIDGE_ROTATE_QUERIES="${RIDGE_ROTATE_QUERIES:-false}"
RIDGE_NORMALIZE_KEYS="${RIDGE_NORMALIZE_KEYS:-true}"
RIDGE_OMEGA_SELECTORS="${RIDGE_OMEGA_SELECTORS:-top_norm,leverage,random}"
RIDGE_OMEGA_FRACTIONS="${RIDGE_OMEGA_FRACTIONS:-0.1,0.25,0.5,0.75}"
MAX_REQS="${MAX_REQS:-}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"

MODEL_SLUG="${MODEL//\//--}"
OUT_ROOT="results/ruler16k_sweep_track2/${MODEL_SLUG}"

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
    --ridge-omega-selectors "$RIDGE_OMEGA_SELECTORS"
    --ridge-omega-fractions "$RIDGE_OMEGA_FRACTIONS"
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

# NOTE: `awk … exit` after a match closes the pipe early, and python's
# next print raises BrokenPipeError → exit 120. With set -o pipefail that
# kills the whole script. Disable pipefail for just this substitution so
# the early-exit SIGPIPE doesn't propagate.
set +o pipefail
CELL_LABEL=$(python scripts/longbench_sweep.py \
        --benchmark ruler16k --model "$MODEL" --template "$TEMPLATE" \
        --methods ridge --skip-full --ratios "$RATIOS" \
        --ridge-gammas "$RIDGE_GAMMAS" \
        --ridge-rotate-queries "$RIDGE_ROTATE_QUERIES" \
        --ridge-normalize-keys "$RIDGE_NORMALIZE_KEYS" \
        --ridge-omega-selectors "$RIDGE_OMEGA_SELECTORS" \
        --ridge-omega-fractions "$RIDGE_OMEGA_FRACTIONS" \
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
