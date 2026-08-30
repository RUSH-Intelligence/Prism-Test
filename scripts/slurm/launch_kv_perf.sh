#!/bin/bash
# Fan out the KV-compression performance sweep: one SLURM job per context length.
#
# Each job measures the full-KV anchor plus every method at that context, with
# one SUBPROCESS PER CELL so the caching allocator's pool, cuBLASLt heuristics
# and the never-restored attn_module.rotary_emb graft (kv_compression/base.py:474-475)
# cannot leak between methods.
#
#   DRY_RUN=1 ./scripts/slurm/launch_kv_perf.sh      # print, touch nothing
#   ./scripts/slurm/launch_kv_perf.sh                # submit
#
# Knobs: ROOT MODEL MODEL_KEY METHODS RATIOS CONTEXTS ATTN DTYPE MAXLEN
#        DECODE_STEPS REPEATS WARMUP STAGGER DRY_RUN SMOKE FILTER_CONTEXT
set -euo pipefail

REPO=/scratch/sj157/Prism-Test
SBATCH_SCRIPT="$REPO/scripts/slurm/kv_perf.sbatch"
ROOT="${ROOT:-/scratch/sj157/kv_perf}"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
MODEL_KEY="${MODEL_KEY:-llama8b}"
METHODS="${METHODS:-knorm,cur,keydiff,snapkv,streaming_llm}"
RATIOS="${RATIOS:-0.9}"
ATTN="${ATTN:-flash_attention_2}"
DTYPE="${DTYPE:-bfloat16}"
MAXLEN="${MAXLEN:-131072}"
DECODE_STEPS="${DECODE_STEPS:-128}"
REPEATS="${REPEATS:-5}"
WARMUP="${WARMUP:-2}"
STAGGER="${STAGGER:-10}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"

# 130816 = 131072 - 256: the question block and 128 decode steps must fit inside
# Llama-3.1's 131072 window, or absolute RoPE positions leave the trained range.
# The driver refuses to silently truncate, so the cap is explicit here.
CONTEXTS_DEFAULT=(8192 16384 32768 65536 130816)
# walltimes index-aligned to CONTEXTS_DEFAULT
WALLTIMES=(0:45:00 0:50:00 1:10:00 1:40:00 3:00:00)

if [[ "$SMOKE" == "1" ]]; then
    CONTEXTS_DEFAULT=(8192); WALLTIMES=(0:30:00)
    METHODS="knorm"; DECODE_STEPS=16; REPEATS=2; WARMUP=1
    SUBDIR="perf_smoke"; PREFIX="kvps"
else
    SUBDIR="perf"; PREFIX="kvp"
fi
IFS=' ' read -r -a CONTEXTS <<< "${CONTEXTS:-${CONTEXTS_DEFAULT[*]}}"

# MATRIX mode: sweep every model row in the TSV instead of the single MODEL env var.
# The matrix is the single source of truth for each model's attn/dtype/maxlen/contexts,
# so a per-model flag is never restated at the call site.
MATRIX="${MATRIX:-}"
MODEL_ROWS=()
if [[ -n "$MATRIX" ]]; then
    [[ -f "$MATRIX" ]] || { echo "no such matrix: $MATRIX" >&2; exit 2; }
    while IFS= read -r line; do
        [[ -z "$line" || "$line" == \#* || "$line" == key$'\t'* ]] && continue
        n=$(awk -F'\t' '{print NF}' <<< "$line")
        (( n == 10 )) || { echo "$MATRIX: expected 10 tab-separated fields, got $n: ${line:0:40}" >&2; exit 2; }
        MODEL_ROWS+=("$line")
    done < "$MATRIX"
    (( ${#MODEL_ROWS[@]} )) || { echo "$MATRIX: no model rows" >&2; exit 2; }
else
    MODEL_ROWS+=("$MODEL_KEY"$'\t'"$MODEL"$'\t'"$ATTN"$'\t'"$DTYPE"$'\t'"true"$'\t'"false"$'\t'"$MAXLEN"$'\t'"$(IFS=,; echo "${CONTEXTS[*]}")"$'\t'"$(IFS=,; echo "${WALLTIMES[*]}")"$'\t'"$MODEL_KEY")
fi

OUTROOT="$ROOT/$SUBDIR"
MANIFEST="$ROOT/${SUBDIR}_manifest.tsv"

freeze_provenance() {
    [[ "$DRY_RUN" == "1" ]] && return 0
    mkdir -p "$OUTROOT"
    # Never clobber: overwriting these would destroy the study's provenance.
    [[ -f "$OUTROOT/env_freeze.txt" ]] || \
        /scratch/sj157/prism_env/bin/pip freeze > "$OUTROOT/env_freeze.txt" 2>/dev/null || true
    if [[ ! -f "$OUTROOT/worktree_at_launch.diff" ]]; then
        { echo "# HEAD: $(git -C "$REPO" rev-parse HEAD)"
          git -C "$REPO" status --porcelain | sed 's/^/# /'
          git -C "$REPO" diff; } > "$OUTROOT/worktree_at_launch.diff"
    fi
}

in_flight() { squeue --noheader --name="$1" --format=%i 2>/dev/null | grep -q .; }

freeze_provenance
[[ "$DRY_RUN" == "1" ]] || { mkdir -p "$ROOT"; [[ -f "$MANIFEST" ]] || \
    printf 'timestamp\tjobid\tjobname\tmodel_key\tcontext\tmethods\tratios\toutdir\n' > "$MANIFEST"; }

submitted=0; skipped=0; inflight=0
for row in "${MODEL_ROWS[@]}"; do
    IFS=$'\t' read -r MODEL_KEY MODEL ATTN DTYPE TRC DEQUANT MAXLEN row_ctxs row_walls LABEL <<< "$row"
    [[ -n "${FILTER_MODEL:-}" && " $FILTER_MODEL " != *" $MODEL_KEY "* ]] && continue
    IFS=',' read -r -a CONTEXTS <<< "$row_ctxs"
    IFS=',' read -r -a WALLTIMES <<< "$row_walls"

for i in "${!CONTEXTS[@]}"; do
    ctx="${CONTEXTS[$i]}"
    [[ -n "${FILTER_CONTEXT:-}" && " $FILTER_CONTEXT " != *" $ctx "* ]] && continue
    walltime="${WALLTIME_OVERRIDE:-${WALLTIMES[$i]:-2:00:00}}"
    jobname="${PREFIX}_${MODEL_KEY}_${ctx}"
    outdir="$OUTROOT"

    # Idempotency: a context is done when every expected cell has a perf.json.
    n_methods=$(( $(tr ',' '\n' <<< "$METHODS" | grep -c .) * $(tr ',' '\n' <<< "$RATIOS" | grep -c .) + 1 ))
    # `set -o pipefail` + find on a not-yet-created dir returns 1; guard it.
    n_done=0
    if [[ -d "$outdir/$MODEL_KEY/ctx$ctx" ]]; then
        n_done=$(find "$outdir/$MODEL_KEY/ctx$ctx" -name perf.json | wc -l)
    fi
    if (( n_done >= n_methods )); then
        echo "skip (done $n_done/$n_methods): $jobname"; skipped=$((skipped+1)); continue
    fi
    if in_flight "$jobname"; then
        echo "skip (in flight): $jobname"; inflight=$((inflight+1)); continue
    fi

    # git is not on PATH after `module purge`; stamp provenance in from here.
    git_sha="$(git -C "$REPO" rev-parse HEAD)"
    git_dirty=0; [[ -n "$(git -C "$REPO" status --porcelain)" ]] && git_dirty=1
    env_args=(PRISM_GIT_SHA="$git_sha" PRISM_GIT_DIRTY="$git_dirty"
              OUTDIR="$outdir" CONTEXT="$ctx" MODEL="$MODEL" MODEL_KEY="$MODEL_KEY"
              METHODS="$METHODS" RATIOS="$RATIOS" ATTN="$ATTN" DTYPE="$DTYPE" MAXLEN="$MAXLEN"
              DECODE_STEPS="$DECODE_STEPS" REPEATS="$REPEATS" WARMUP="$WARMUP"
              TRC="$TRC" DEQUANT="$DEQUANT")
    # --export=ALL only; never --export=ALL,VAR=val (that list splits on commas).
    cmd=(sbatch --parsable --job-name="$jobname" --time="$walltime"
         --begin="now+$((submitted * STAGGER))" --export=ALL "$SBATCH_SCRIPT")

    if [[ "$DRY_RUN" == "1" ]]; then
        printf "DRY: env"; printf " %q" "${env_args[@]}" "${cmd[@]}"; printf "\n"
    else
        jobid="$(env "${env_args[@]}" "${cmd[@]}")"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "$jobid" "$jobname" \
               "$MODEL_KEY" "$ctx" "$METHODS" "$RATIOS" "$outdir" >> "$MANIFEST"
        echo "submitted $jobid  $jobname  (walltime $walltime)"
    fi
    submitted=$((submitted+1))
done
done
echo "mode=$([[ "$SMOKE" == 1 ]] && echo smoke || echo full) submitted=$submitted skipped(done)=$skipped in-flight=$inflight"
