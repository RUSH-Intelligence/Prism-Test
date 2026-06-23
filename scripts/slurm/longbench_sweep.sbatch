#!/bin/bash
# SLURM job-array template for the LongBench KV-compression sweep.
#
# Each array task runs ONE cell (one method × one ratio, or the Full baseline)
# of `scripts/longbench_sweep.py` on its own GPU. There are 46 cells with the
# default method/ratio set (Full + 8 methods × 3 ratios + Ridge gamma sweep
# of 7 γ × 3 ratios) — see `python scripts/longbench_sweep.py --dry-run` for
# the index map.
#
# Submit:
#   sbatch scripts/slurm/longbench_sweep.sbatch
#
# Subset / smoke pass:
#   sbatch --array=0-4 --export=ALL,MAX_REQS=20 scripts/slurm/longbench_sweep.sbatch
#
# Resubmit a single failed cell (e.g. cell 33):
#   sbatch --array=33 scripts/slurm/longbench_sweep.sbatch
#
# After all array tasks finish, merge the per-cell manifest fragments into the
# unified manifest.json by running the sweep once in resume mode on a login
# node (no GPU needed — every cell is already done):
#   .venv/bin/python scripts/longbench_sweep.py --resume --dry-run     # confirm
#   .venv/bin/python scripts/longbench_sweep.py --resume                # write merged manifest.json
#   .venv/bin/python scripts/longbench_to_xlsx.py                       # fill Ridge Press.xlsx

#SBATCH --job-name=lb-sweep
#SBATCH --array=0-45
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/longbench_sweep/job_%A/slurm-%a.out
#SBATCH --error=logs/longbench_sweep/job_%A/slurm-%a.err

# --- Cluster-specific knobs ---------------------------------------------------
# RTX A6000 (Ampere, sm_86). flash-attn was originally compiled for L40S
# (sm_89); on A6000 the kernel JIT-compiles via PTX on first call (a few-
# second one-time cost, cached for the run). Override with `sbatch
# --gres=gpu:l40s:1 ...` to go back to the sm_89-native path.
# Scavenger partition: required for --cpus-per-task=8 above (the default QoS
# caps CPUs at 4). Override at submit time with `sbatch --partition=... --qos=...`
# if your cluster uses different names.
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --account=scavenger
# -----------------------------------------------------------------------------

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
# Per-submission log folder: logs/longbench_sweep/job_<JOBID>/slurm-<TASKID>.{out,err}
# (SLURM auto-creates the parent dir when opening the output file on modern
# versions; this mkdir is belt-and-suspenders for resuming/symlinking below.)
LOG_DIR="logs/longbench_sweep/job_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
mkdir -p results/longbench_sweep "$LOG_DIR"

# Module env: flash-attn was compiled with gcc/15.1.0; importing it raises
# ``undefined symbol: __cxa_call_terminate`` against conda's older libstdc++.
# transformers->torch loads conda's libstdc++ *before* flash_attn gets a
# chance, and once libstdc++ is mapped you cannot replace it — so
# LD_LIBRARY_PATH alone is insufficient. LD_PRELOAD forces gcc-15's
# libstdc++ to load first, before any other dependency resolution.
# cuda/13.0.2 matches the torch cu130 build for completeness.
module load cuda/13.0.2 gcc/15.1.0
export LD_LIBRARY_PATH=/opt/local/stow/gcc-15.1.0/lib64:${LD_LIBRARY_PATH:-}
export LD_PRELOAD=/opt/local/stow/gcc-15.1.0/lib64/libstdc++.so.6

# Required by torch.use_deterministic_algorithms(True) — pins cuBLAS workspace
# so GEMM kernel choice is stable across runs. ~0% perf cost. Paired with the
# torch determinism flags set in eval_harness/runner.py::_set_seed.
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Skip the HF datasets cache-validity check (which takes an flock on NFS and
# fights with sibling array tasks → Errno 37/116). Safe because the LongBench
# cache must already be populated before submitting (run load_dataset once on
# a login node). Without this, ~10–20% of array tasks die at startup under
# heavy contention.
export HF_DATASETS_OFFLINE=1

# Activate the uv-managed virtualenv.
source .venv/bin/activate

# Optional: pin HF cache off $HOME (avoid quota issues, share across array tasks).
# export HF_HOME="${HF_HOME:-$PWD/.hf-cache}"

# Overridable via `sbatch --export=ALL,VAR=value`:
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
MAX_REQS="${MAX_REQS:-}"          # empty = full LongBench set; set e.g. 50 for a fast first pass
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}" # empty = template default (128K); set to model's native cap
                                   # for non-Llama-3.1 models (e.g. 32768 for Mistral-7B-v0.2)
TEMPLATE="${TEMPLATE:-}"           # empty = sweep script default (evaluate/evaluate_kv.yaml);
                                   # set to evaluate/evaluate_kv_ministral.yaml for the
                                   # Ministral-3 FP8 → BF16 dequantize variant.

ARGS=(--model "$MODEL" --cell-index "$SLURM_ARRAY_TASK_ID" --resume)
if [ -n "$MAX_REQS" ]; then
    ARGS+=(--max-requests "$MAX_REQS")
fi
if [ -n "$MAX_MODEL_LEN" ]; then
    ARGS+=(--max-model-len "$MAX_MODEL_LEN")
fi
if [ -n "$TEMPLATE" ]; then
    ARGS+=(--template "$TEMPLATE")
fi

# Resolve the cell_id (e.g. "PyramidKv__r0.9") for this task index from
# longbench_sweep.py --dry-run so the log file is greppable by method name.
# Symlink <cell_id>.{out,err} -> slurm-<taskid>.{out,err} in the job folder so
# both filename lookups work.
CELL_LABEL=$(python scripts/longbench_sweep.py --model "$MODEL" --dry-run 2>/dev/null \
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
echo "Log dir:    $LOG_DIR"
echo "GPU:        ${CUDA_VISIBLE_DEVICES:-?}"
echo "Command:    python scripts/longbench_sweep.py ${ARGS[*]}"
echo

srun python scripts/longbench_sweep.py "${ARGS[@]}"
