#!/bin/bash
# SLURM job-array template for the LongBench KV-compression sweep.
#
# Each array task runs ONE cell (one method × one ratio, or the Full baseline)
# of `scripts/longbench_sweep.py` on its own GPU. There are 28 cells with the
# default method/ratio set — see `python scripts/longbench_sweep.py --dry-run`
# for the index map.
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
#SBATCH --array=0-27
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/longbench_sweep/slurm-%A_%a.out
#SBATCH --error=logs/longbench_sweep/slurm-%A_%a.err

# --- Cluster-specific knobs ---------------------------------------------------
# L40S is pinned (Ada Lovelace, sm_89) because flash-attn was compiled on
# this arch; running on a different GPU (A100=sm_80, H100=sm_90) would JIT
# via PTX at first kernel call, which is slower and sometimes fragile.
# Uncomment/edit these for your cluster (partition/QoS/account names vary):
# #SBATCH --partition=scavenger
# #SBATCH --qos=scavenger
# #SBATCH --account=scavenger
# -----------------------------------------------------------------------------

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p results/longbench_sweep logs/longbench_sweep

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

# Activate the uv-managed virtualenv.
source .venv/bin/activate

# Optional: pin HF cache off $HOME (avoid quota issues, share across array tasks).
# export HF_HOME="${HF_HOME:-$PWD/.hf-cache}"

# Overridable via `sbatch --export=ALL,VAR=value`:
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
MAX_REQS="${MAX_REQS:-}"          # empty = full LongBench set; set e.g. 50 for a fast first pass

ARGS=(--model "$MODEL" --cell-index "$SLURM_ARRAY_TASK_ID" --resume)
if [ -n "$MAX_REQS" ]; then
    ARGS+=(--max-requests "$MAX_REQS")
fi

echo "Host:       $(hostname)"
echo "Array task: ${SLURM_ARRAY_TASK_ID:-?} of ${SLURM_ARRAY_JOB_ID:-?}"
echo "GPU:        ${CUDA_VISIBLE_DEVICES:-?}"
echo "Command:    python scripts/longbench_sweep.py ${ARGS[*]}"
echo

srun python scripts/longbench_sweep.py "${ARGS[@]}"
