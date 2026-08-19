#!/bin/bash
# ============================================================================
# THE sweep command. Edit ./sweep.yaml, then run:
#
#     ./scripts/submit_sweep.sh                 # submit everything in sweep.yaml
#     ./scripts/submit_sweep.sh --dry-run       # preview the plan, submit nothing
#     ./scripts/submit_sweep.sh my_other.yaml   # use a different config file
#
# Submits one SLURM array job per (model x benchmark) pair; each array runs the
# method x ratio (x Ridge/Verified) grid. All logic lives in scripts/sweep.py.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

# First non-flag arg (if any) is the config path; the rest pass through.
CONFIG="sweep.yaml"
PASS=()
for a in "$@"; do
    case "$a" in
        -*) PASS+=("$a") ;;
        *)  CONFIG="$a" ;;
    esac
done

[ -x .venv/bin/python ] && PY=.venv/bin/python || PY=python
exec "$PY" scripts/sweep.py --config "$CONFIG" --submit "${PASS[@]}"
