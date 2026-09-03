#!/bin/bash
# Ridge envelope_gamma tuning sweep on RULER (Llama-3.1-8B-Instruct).
#
# Submits one compactor_ruler.sbatch cell per (bench, gamma, ratio):
#   9 gammas x 4 ratios x 4 lengths = 144 jobs, each 13 subsets x 5 samples
#   on the TUNING split (request_offset=100 -> rows 100-104, disjoint from
#   the 100-sample eval split rows 0-99).
#
# Usage:
#   bash scripts/slurm/launch_ridge_gamma_tune.sh
#
# Env overrides:
#   ROOT          results root       (default /scratch/sj157/results_ridge_gamma)
#   MODEL         HF model id        (default: sbatch default, Llama-3.1-8B-Instruct)
#   MAX_JOBS      submit cap per invocation (default 60) — rerun to submit the
#                 remainder; skip-if-done makes reruns idempotent
#   STAGGER       seconds between job start times (default 10)
#   TUNE_MAXREQ   samples per subset (default 5)
#   TUNE_OFFSET   request_offset     (default 100)
#   FILTER_BENCH / FILTER_GAMMA / FILTER_RATIO
#                 space-separated allowlists, e.g. FILTER_BENCH="ruler16k ruler32k"
#   DRY_RUN=1     print sbatch commands instead of submitting
#
# Manifest: $ROOT/tune_manifest.tsv (appended per submission).

set -euo pipefail

REPO=/scratch/sj157/Prism-Test
SBATCH_SCRIPT="$REPO/scripts/slurm/compactor_ruler.sbatch"

ROOT="${ROOT:-/scratch/sj157/results_ridge_gamma}"
MAX_JOBS="${MAX_JOBS:-60}"
STAGGER="${STAGGER:-10}"
TUNE_MAXREQ="${TUNE_MAXREQ:-5}"
TUNE_OFFSET="${TUNE_OFFSET:-100}"
DRY_RUN="${DRY_RUN:-0}"

GAMMAS=( 0 0.5 1 1.5 2 2.5 3 3.5 4 )
RATIOS=( 0.2 0.4 0.6 0.8 )
BENCHES=( ruler16k ruler32k ruler64k ruler128k )

FILTER_BENCH="${FILTER_BENCH:-}"
FILTER_GAMMA="${FILTER_GAMMA:-}"
FILTER_RATIO="${FILTER_RATIO:-}"

in_filter() {  # $1 = value, $2 = space-separated allowlist ("" = allow all)
    local val="$1" list="$2" item
    [[ -z "$list" ]] && return 0
    for item in $list; do [[ "$item" == "$val" ]] && return 0; done
    return 1
}

walltime_for() {
    case "$1" in
        ruler16k)  echo "0:45:00" ;;
        ruler32k)  echo "1:00:00" ;;
        ruler64k)  echo "1:30:00" ;;
        ruler128k) echo "2:30:00" ;;
        *)         echo "2:00:00" ;;
    esac
}

in_flight() {  # $1 = job name; true if a job with this name is already queued/running
    command -v squeue >/dev/null 2>&1 || return 1
    [[ -n "$(squeue --noheader --name="$1" --format=%i 2>/dev/null)" ]]
}

mkdir -p "$ROOT/tune"
MANIFEST="$ROOT/tune_manifest.tsv"
[[ -f "$MANIFEST" ]] || printf "timestamp\tjobid\tbench\tgamma\tratio\toutdir\n" >> "$MANIFEST"

submitted=0
skipped=0
inflight=0
filtered=0
capped=0

for bench in "${BENCHES[@]}"; do
    in_filter "$bench" "$FILTER_BENCH" || { filtered=$((filtered + 36)); continue; }
    for g in "${GAMMAS[@]}"; do
        in_filter "$g" "$FILTER_GAMMA" || { filtered=$((filtered + 4)); continue; }
        for r in "${RATIOS[@]}"; do
            in_filter "$r" "$FILTER_RATIO" || { filtered=$((filtered + 1)); continue; }

            outdir="$ROOT/tune/${bench}_g${g}_r${r}"
            jobname="rgt_${bench}_g${g}_r${r}"
            if [[ -n "$(find "$outdir" -name metrics.json -print -quit 2>/dev/null)" ]]; then
                skipped=$((skipped + 1))
                continue
            fi
            # metrics.json only appears when a run COMPLETES, so a bare
            # completion check would resubmit cells whose jobs are still
            # queued/running; ask squeue by job name too.
            if in_flight "$jobname"; then
                inflight=$((inflight + 1))
                continue
            fi
            if (( submitted >= MAX_JOBS )); then
                capped=$((capped + 1))
                continue
            fi
            # Vars go into the submitting environment + --export=ALL:
            # `--export=ALL,VAR=val` splits its list on commas and would
            # mangle any comma-containing value (e.g. SUBSETS lists).
            env_args=(BENCH="$bench" METHOD=ridge RATIO="$r" MAXLEN=131072
                      OUTDIR="$outdir" SUBSETS=all MAXREQ="$TUNE_MAXREQ"
                      OFFSET="$TUNE_OFFSET" KWARGS="{envelope_gamma: $g}")
            [[ -n "${MODEL:-}" ]] && env_args+=(MODEL="$MODEL")
            cmd=(sbatch --parsable
                 --job-name="$jobname"
                 --time="$(walltime_for "$bench")"
                 --begin="now+$((submitted * STAGGER))"
                 --export=ALL
                 "$SBATCH_SCRIPT")

            if [[ "$DRY_RUN" == "1" ]]; then
                # %q-quote so the printed line is copy-paste executable
                # (KWARGS contains a space); dry runs never touch the manifest.
                printf "DRY: env"; printf " %q" "${env_args[@]}" "${cmd[@]}"; printf "\n"
            else
                mkdir -p "$outdir"
                jobid="$(env "${env_args[@]}" "${cmd[@]}")"
                echo "submitted $jobid  $jobname"
                printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Is)" "$jobid" "$bench" "$g" "$r" "$outdir" >> "$MANIFEST"
            fi
            submitted=$((submitted + 1))
        done
    done
done

echo "----"
echo "submitted=$submitted skipped(done)=$skipped in-flight=$inflight filtered=$filtered over-cap=$capped"
if (( capped > 0 )); then
    echo "NOTE: $capped cells not submitted (MAX_JOBS=$MAX_JOBS). Re-run this script to submit the remainder."
fi
