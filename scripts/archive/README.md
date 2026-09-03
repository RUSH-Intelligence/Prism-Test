# scripts/archive/

Retired sweep launchers, superseded by the generic sweep tool at the repo root:

    ./scripts/submit_sweep.sh      # reads ./sweep.yaml, submits SLURM arrays
    scripts/sweep.py               # the generic engine
    scripts/sweep.sbatch           # the generic array template

What's here and why it was retired:

- **`longbench_sweep.py`** — the original per-benchmark sweep engine. Its cell
  grid, `--cell-index`, `--resume`, and manifest logic were reimplemented
  (generalized over model + benchmark, not longbench-specific) in
  `scripts/sweep.py`. Kept for reference / reproducing old manifests.
- **`slurm/`** — the eight hand-written per-experiment `.sbatch` files plus
  `launch_ridge_gamma_tune.sh`. Each was one experiment's array job with the
  method/ratio grid baked in; `sweep.yaml` + `sweep.sbatch` replace all of them.

Nothing in the live tree imports these. Paths inside them (and inside the
`scripts/reporting/*` docstrings that mention them) still point at the old
`scripts/` locations — update if you ever revive one. The spreadsheet/report
builders that read old sweep output live in `scripts/reporting/`.
