#!/bin/bash
#SBATCH --output=/scratch/sj157/kvruler_logs/%x_%j.out
#SBATCH --error=/scratch/sj157/kvruler_logs/%x_%j.err
#SBATCH --job-name=compactor_ruler
#SBATCH --account=as143
#SBATCH --partition=commons
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --threads-per-core=1
#SBATCH --mem-per-cpu=32GB
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --export=ALL
#
# KV-compression runner for the CURRENT research_config format (the older
# kvruler.sbatch writes the retired cache_config shape). Derived from
# /scratch/sj157/kvruler.sbatch (proven H200 launcher: env/module/cache setup
# kept verbatim); only the generated YAML differs.
#
#   sbatch --job-name=compactor16k_r80 \
#     --export=ALL,BENCH=ruler16k,METHOD=compactor,RATIO=0.8,MAXLEN=131072,\
#OUTDIR=/scratch/sj157/results_compactor_ref/ruler16k_r80,SUBSETS=all,MAXREQ=100 \
#     /scratch/sj157/Prism-Test/scripts/slurm/compactor_ruler.sbatch
#
#   BENCH   : ruler16k | ruler32k | ruler64k | ruler128k
#   METHOD  : any registered kv compressor (compactor | ridge | knorm | top_k_sampling | ...)
#   RATIO   : compression_ratio = FRACTION OF KV CACHE PRUNED per head
#   MAXLEN  : max_model_len truncation cap
#   OUTDIR  : unique results output_dir
#   SUBSETS : comma list, or "all" (default) for the benchmark's default 13
#   ATTN    : attn_implementation (default flash_attention_2)
#   MAXREQ  : max_requests per subset (default 100)
#   KWARGS  : optional literal YAML dict for kv_compressor_kwargs (default {})

set -euo pipefail

module purge
module load GCCcore/13.3.0
module load CUDA/12.9.1
source /scratch/sj157/prism_env/bin/activate

# Compute nodes set LD_PRELOAD=libxalt_init.so, whose RUNPATH loads xalt's
# stale libcrypto.so.3 (3.2.2) into the process before anything else — then
# libssl.so.3 (needs OPENSSL_3.3.0) binds to it and python3.13 `import ssl`
# dies (jobs 160939/160946). LD_LIBRARY_PATH cannot beat an already-loaded
# soname; PRELOADING the correct pair ahead of xalt's init can (verified on
# bg4u7g1). xalt stays preloaded after ours, so job tracking still works.
export LD_PRELOAD="/opt/apps/software/OpenSSL/3/lib64/libcrypto.so.3:/opt/apps/software/OpenSSL/3/lib64/libssl.so.3${LD_PRELOAD:+:$LD_PRELOAD}"

export PYTHONUNBUFFERED=1

export HF_HOME="/scratch/sj157/hf_home"

# HF auth for gated repos (e.g. meta-llama): NEVER hardcode a token here.
# Inherit HF_TOKEN from the submitting shell (sbatch --export=ALL), else fall
# back to the standard hub token files.
if [[ -z "${HF_TOKEN:-}" ]]; then
    for tokfile in "${HF_HOME}/token" "$HOME/.cache/huggingface/token"; do
        if [[ -f "$tokfile" ]]; then HF_TOKEN="$(< "$tokfile")"; break; fi
    done
fi
if [[ -n "${HF_TOKEN:-}" ]]; then
    export HF_TOKEN
    export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
    export HF_HUB_TOKEN="${HF_TOKEN}"
fi
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}"
export TRITON_CACHE_DIR="/scratch/sj157/.cache/triton"

JOB_LOCAL_BASE="/scratch/sj157/kvruler_job_${SLURM_JOB_ID}"
export TMPDIR="${JOB_LOCAL_BASE}/tmp"
export XDG_CACHE_HOME="${JOB_LOCAL_BASE}/xdg_cache"
export XDG_CONFIG_HOME="${JOB_LOCAL_BASE}/xdg_config"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TRITON_CACHE_DIR"
trap 'rm -rf "${JOB_LOCAL_BASE}"' EXIT

: "${BENCH:?set BENCH}"
: "${METHOD:?set METHOD}"
: "${RATIO:?set RATIO}"
: "${OUTDIR:?set OUTDIR}"
: "${MAXLEN:?set MAXLEN}"
SUBSETS="${SUBSETS:-all}"
ATTN="${ATTN:-flash_attention_2}"
MAXREQ="${MAXREQ:-100}"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
KWARGS="${KWARGS:-{\}}"

mkdir -p "$OUTDIR"

if [[ "$SUBSETS" == "all" ]]; then
    SUBSETS_LINE="subsets: null"
else
    SUBSETS_LINE="subsets: ${SUBSETS}"
fi

CFG="${OUTDIR}/run_config.yaml"
cat > "$CFG" <<YAML
benchmark: ${BENCH}
${SUBSETS_LINE}
backend: research
model: ${MODEL}
tensor_parallel_size: 1
dtype: bfloat16
max_model_len: ${MAXLEN}
gpu_memory_utilization: 0.9
trust_remote_code: true
enable_prefix_caching: true
max_new_tokens: 128
temperature: 0.0
top_p: 1.0
system_prompt: null
seed: 42
max_requests: ${MAXREQ}
fraction: 1.0
query_aware: false
output_dir: ${OUTDIR}
llm_kwargs:
  attn_implementation: ${ATTN}
  research_config:
    kv_compressor: ${METHOD}
    compression_ratio: ${RATIO}
    kv_compressor_kwargs: ${KWARGS}
    max_context_length: null
    log_cache_seq_len: true
    attention_method: none
    attention_method_kwargs: {}
YAML

if [[ "$BENCH" == "ruler128k" || "$BENCH" == "ruler64k" ]]; then
    JITTER=$(( SLURM_JOB_ID % 90 ))
    echo "${BENCH} stream jitter: sleeping ${JITTER}s before load"
    sleep "${JITTER}"
fi

echo "================ compactor_ruler run ================"
echo "host=$(hostname)  job=${SLURM_JOB_ID}  name=${SLURM_JOB_NAME:-?}"
echo "BENCH=${BENCH} METHOD=${METHOD} RATIO=${RATIO} ATTN=${ATTN} MAXREQ=${MAXREQ} MAXLEN=${MAXLEN}"
echo "SUBSETS=${SUBSETS}"
echo "OUTDIR=${OUTDIR}"
echo "python=$(command -v python)"; python --version
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "----- config -----"; cat "$CFG"; echo "------------------"
echo "starting eval at $(date)"

cd /scratch/sj157/Prism-Test
export PYTHONPATH="/scratch/sj157/Prism-Test:${PYTHONPATH:-}"

# python invoked directly (not via srun): single-task job, and srun's spank
# hooks are what re-inject xalt's stale libcrypto on this cluster.
python -m eval_harness.cli run --config_file "$CFG"
echo "FINISHED eval at $(date)"
