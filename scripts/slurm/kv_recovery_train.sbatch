#!/bin/bash
#SBATCH --output=/scratch/sj157/kvrec_logs/%x_%j.out
#SBATCH --error=/scratch/sj157/kvrec_logs/%x_%j.err
#SBATCH --job-name=kvrec_train
#SBATCH --account=as143
#SBATCH --partition=commons
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --threads-per-core=1
#SBATCH --mem-per-cpu=32GB
#SBATCH --gres=gpu:h200:1
#SBATCH --time=6:00:00
#SBATCH --export=ALL
#SBATCH --exclude=bg3u24g1
# Train one hidden-state KV-recovery delta.
#   sbatch --job-name=kvrec_train_demo scripts/slurm/kv_recovery_train.sbatch
#   env: CONFIG (required), RUN_NAME, SET (newline-separated "a.b=c" dotted overrides), EXTRA (extra CLI args)
#   Pass variables with --export=ALL and plain environment exports (never --export=ALL,VAR=...).
set -euo pipefail

# ---------------------------------------------------------------- env ------
# Copied from scripts/slurm/kv_perf.sbatch (the validated H200 launcher).
module purge
module load GCCcore/13.3.0
module load CUDA/12.9.1
source /scratch/sj157/prism_env/bin/activate

# Compute nodes set LD_PRELOAD=libxalt_init.so, whose RUNPATH pulls in xalt's
# stale libcrypto.so.3 (3.2.2); python3.13 `import ssl` then dies with
# "OPENSSL_3.3.0 not found". LD_LIBRARY_PATH cannot beat an already-preloaded
# soname, so the correct pair must be preloaded AHEAD of xalt.
export LD_PRELOAD="/opt/apps/software/OpenSSL/3/lib64/libcrypto.so.3:/opt/apps/software/OpenSSL/3/lib64/libssl.so.3${LD_PRELOAD:+:$LD_PRELOAD}"

export PYTHONUNBUFFERED=1
export HF_HOME="/scratch/sj157/hf_home"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TOKENIZERS_PARALLELISM=false
# Deterministic cuBLAS GEMM selection; must be set before CUDA initialises
# (runner._enable_determinism / kv_recovery.provenance.configure_determinism rely on it).
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

JOB_LOCAL_BASE="/scratch/sj157/kvrec_job_${SLURM_JOB_ID}"
export TMPDIR="${JOB_LOCAL_BASE}/tmp"
export XDG_CACHE_HOME="${JOB_LOCAL_BASE}/xdg-cache"
export XDG_CONFIG_HOME="${JOB_LOCAL_BASE}/xdg-config"
export TRITON_CACHE_DIR="${JOB_LOCAL_BASE}/triton"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$TRITON_CACHE_DIR" /scratch/sj157/kvrec_logs
trap 'rm -rf "${JOB_LOCAL_BASE}"' EXIT

REPO="${REPO:-/scratch/sj157/Prism-Test-hsr}"
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
echo "host=$(hostname) job=${SLURM_JOB_ID} name=${SLURM_JOB_NAME:-?} repo=$REPO"
python --version
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

: "${CONFIG:?set CONFIG}"
args=(--config "$CONFIG")
[[ -n "${RUN_NAME:-}" ]] && args+=(--run-name "$RUN_NAME")
while IFS= read -r kv; do [[ -n "$kv" ]] && args+=(--set "$kv"); done <<< "${SET:-}"
# python invoked directly, NOT via srun (srun's spank hooks re-inject xalt's stale libcrypto).
python scripts/train_kv_recovery.py "${args[@]}" ${EXTRA:-}
rc=$?
echo "FINISHED at $(date) rc=$rc"
exit $rc
