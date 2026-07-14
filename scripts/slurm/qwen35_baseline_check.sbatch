#!/bin/bash
# One-off Qwen3.5-9B correctness check (no KV compression, 5 rows/subset).
# Runs RULER16k (array=0) and LongBench (array=1) as two array tasks.
#
# Submit:
#   sbatch scripts/slurm/qwen35_baseline_check.sbatch
#
# After both tasks finish, inspect metrics.json in each results dir:
#   results/qwen35_baseline/ruler16k__Qwen--Qwen3.5-9B__research__.../metrics.json
#   results/qwen35_baseline/longbench__Qwen--Qwen3.5-9B__research__.../metrics.json

#SBATCH --job-name=qwen35-baseline
#SBATCH --array=0-1
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=logs/qwen35_baseline/job_%A/slurm-%a.out
#SBATCH --error=logs/qwen35_baseline/job_%A/slurm-%a.err

#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --account=scavenger

set -euo pipefail
set -x
exec 2>&1

cd "${SLURM_SUBMIT_DIR:-$PWD}"
LOG_DIR="logs/qwen35_baseline/job_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
mkdir -p results/qwen35_baseline "$LOG_DIR"

module load cuda/13.0.2 gcc/15.1.0
export LD_LIBRARY_PATH=/opt/local/stow/gcc-15.1.0/lib64:${LD_LIBRARY_PATH:-}
export LD_PRELOAD=/opt/local/stow/gcc-15.1.0/lib64/libstdc++.so.6
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_DATASETS_OFFLINE=1

if [ ! -f .venv/bin/activate ]; then
    echo "[FATAL] .venv/bin/activate missing on $(hostname)"
    exit 99
fi
source .venv/bin/activate

CONFIG=evaluate/evaluate_qwen35_baseline.yaml

case "${SLURM_ARRAY_TASK_ID:-0}" in
    0)
        # RULER16k — 3 NIAH subsets, 5 rows each. Config defaults.
        srun python -m eval_harness.cli run --config_file "$CONFIG"
        ;;
    1)
        # LongBench — the 16 English tasks the sweep sheet uses (LONGBENCH_16
        # in scripts/longbench_sweep.py:62). The benchmark's raw default list
        # is 34 tasks including 5 Chinese subsets that aren't in our offline
        # HF cache, so we need the explicit narrowed list here.
        srun python -m eval_harness.cli run --config_file "$CONFIG" \
            --benchmark longbench \
            --subsets narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p
        ;;
    *)
        echo "[FATAL] unknown array index ${SLURM_ARRAY_TASK_ID}"
        exit 1
        ;;
esac
