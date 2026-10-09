# Hidden-state KV recovery — how to test and reproduce it

Everything below runs from the repository root of the `feature/hidden-state-kv-recovery` branch. The
"distillation" is the hidden-state alignment fine-tuning of `eval_harness/kv_recovery/`: a *student* running with
a compressed KV cache is trained — on a small subset of its own weights — to reproduce the hidden states of the
*teacher*, the same model with a full cache. Each step says what to run, how long it takes, what it writes, and what
"correct" looks like, so the pipeline can be checked piece by piece before spending GPU hours.

| Step | What it checks | Hardware | Time |
|---|---|---|---|
| 1 | environment | CPU | 10 min |
| 2 | corpus (PG-19 + leakage filter) | CPU, network | 3 min |
| 3 | unit tests (every module on tiny models) | CPU | 3 min |
| 4 | GPU smoke: 15 end-to-end checks per model | 1 GPU | 7–11 min / model |
| 5 | layer sensitivity profiles + selection | 1 GPU | 5–6 min / model |
| 6 | one training run | 1 GPU | 7–12 min |
| 7 | three-way evaluation + recovery report | 1 GPU / cell | 0.8–2.5 h / cell |
| 8 | the pre-registered matrix | SLURM | 10 min / run, 4 h / run of evaluation |
| 9 | misalignment figures on RULER | 1 GPU + CPU | 15 min / model |

## 1. Environment

Two environments are used: a GPU environment for anything that loads the real models, and a light CPU environment for
the unit tests.

```bash
# GPU environment (what the SLURM launchers activate: /scratch/sj157/prism_env)
python3.13 -m venv prism_env && source prism_env/bin/activate
pip install -r requirements.txt            # pins torch 2.11 (CUDA 13) and transformers 5.10.2
pip install flash-linear-attention causal-conv1d      # Qwen3.5's GatedDeltaNet kernels (the torch fallback is ~10x slower)
# do NOT install `kernels` 0.17: it breaks transformers 5.10.2's hub_kernels import

# CPU test environment (.venv-test): any Python >= 3.11 with a CPU torch and transformers >= 5.10 works
python3.12 -m venv .venv-test && .venv-test/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv-test/bin/pip install transformers==5.10.2 pandas numpy scipy pyyaml tqdm safetensors datasets rouge jieba fuzzywuzzy
```

Before any GPU job: `export CUBLAS_WORKSPACE_CONFIG=:4096:8` (the `scripts/slurm/*.sbatch` files do this; deterministic
mode checks it) and point `HF_HOME` at a cache that holds `mistralai/Ministral-3-3B-Instruct-2512`,
`Qwen/Qwen3.5-4B`, `xAlg-AI/att-hub-ruler-16k`, `xAlg-AI/att-hub-ruler-32k` and `Xnhyacinth/LongBench` (the first
run of step 2 and step 7 downloads them; compute nodes without network need the cache pre-filled).

The launchers `scripts/slurm/kv_recovery_{train,eval,smoke,python}.sbatch` take their settings from the environment
(`CONFIG`, `RUN_NAME`, `SET`, `EXTRA`, `CONFIG_FILE`, `ARGS`) and default to `REPO=/scratch/sj157/Prism-Test-hsr`;
set `REPO` to your checkout. Every command below can also be run directly on a GPU node.

## 2. Corpus

```bash
python scripts/prepare_kv_recovery_data.py --num-train 256 --num-val 32 --seed 42        # login node, network
```

Writes `data/kv_recovery/pg19_{train,val}.jsonl` (256 + 32 excerpts of 200 000 characters from the `train` /
`test` books of `emozilla/pg19`), `pg19_manifest.json` (sha256 of each file; with seed 42 the training file hashes
to `7d19cbd8…`, the validation file to `5d454f04…`) and `pg19_leakage_report.json`. Correct: the log reports
5 750 benchmark contexts scanned and 13 train / 1 val candidates rejected (13-word-shingle hits, mostly LongBench
`narrativeqa`); `books_disjoint: true` in the manifest.

## 3. Unit tests (no GPU, no weights)

```bash
.venv-test/bin/python -m unittest discover eval_harness/tests                       # whole suite, ~2 min
.venv-test/bin/python -m unittest discover eval_harness/tests -p 'test_kv_recovery_*.py'   # the 100+ recovery tests
```

Correct: `OK` (a handful of skips are expected: the figure test without matplotlib, Mistral3/Qwen3.5 tests on a
transformers build without those models). The recovery tests build tiny config-initialised Llama / Mistral3 / Qwen3.5
models and exercise every module through the real prefill / segment path: the alignment losses, the sensitivity
formula and selection, trainable-subset selection, teacher/student bitwise identity without compression, gradient
flow, the training loop, delta checkpoints (write / apply / double-application refused), the three-way eval
configs and the recovery metrics. The suite passes on transformers 5.10.2 and on 5.19.0 (what CI installs).

## 4. GPU smoke (one model, one GPU, 15 checks)

```bash
CONFIG=configs/kv_recovery/smoke_ministral_3b.yaml EXTRA=--with-benchmark sbatch scripts/slurm/kv_recovery_smoke.sbatch
CONFIG=configs/kv_recovery/smoke_qwen35_4b.yaml    EXTRA=--with-benchmark sbatch scripts/slurm/kv_recovery_smoke.sbatch
# or directly: python scripts/kv_recovery_smoke.py --config configs/kv_recovery/smoke_ministral_3b.yaml --with-benchmark
```

Writes `outputs/kv_recovery/smoke_<model>/smoke_report.json`. Correct: every entry has `"passed": true` and the log
ends with `all_passed=True`. The checks, in order: S0 synthetic corpus, S1 environment, S2 load (FP8 → BF16 bitwise
for Ministral; 26 / 8 full-attention layers), S3 prompt shaping (no auto system block), S4 windows, S5 compressor
hooks on exactly the K/V-carrying layers, S6 cache budget `int(T(1−r))` per layer, S7 block-vs-full-forward
continuation (bf16 noise ≤ 2e-2), S14 sensitivity selection (deterministic, > 0 on K/V layers, exactly 0 before the
first K/V layer on Qwen3.5, 0 without a compressor), S8 teacher == student without compression (loss 0), S9 more
compression → more divergence, S10 gradients only on the selected projections, S11 two training steps + checkpoint
round trip on a fresh load, S12 identity delta reproduces generations bitwise, S13 three-way RULER plumbing
(3 subsets × 5 rows) through `eval_harness.cli` and `report`. Reference runs 2026-10-08: jobs 320576 (Ministral,
6:45) and 320577 (Qwen3.5, 10:56).

## 5. Layer sensitivity: measure the selection signal without training

```bash
ARGS="scripts/measure_layer_sensitivity.py --config configs/kv_recovery/ministral_3b.yaml --compressors knorm,cur --ratios 0.75,0.5" \
    sbatch --time=0:45:00 scripts/slurm/kv_recovery_python.sbatch
```

Writes `outputs/kv_recovery/sensitivity/<model>/summary.md` plus one JSON/CSV per (compressor, ratio) with the
per-layer `E_l = ‖H_dense − H_comp‖_F / (‖H_dense‖_F + 1e-6)` on 8 held-out 16K calibration windows (seed
`data.seed + 2`, disjoint from the training and validation windows), the ranking and the top-k selection. Correct
(seed 42 corpus): Ministral knorm @ 0.75 → `[12, 13, 14, 15]`, cur @ 0.75 → `[2, 3, 4, 5]`; Qwen3.5 knorm → `[15, 19,
27, 31]`, cur → `[11, 15, 19, 27]`, with layers 0–2 exactly 0 (they precede the first K/V layer). A training run with
`trainable.layers: sensitivity` reproduces exactly these layers (same windows → same `E_l`).

## 6. One training run (the distillation)

```bash
# card default: q_proj + o_proj on the top-4 compression-sensitive layers, knorm @ 0.75, 16K windows, 64 steps
python scripts/train_kv_recovery.py --config configs/kv_recovery/ministral_3b.yaml --run-name demo

# variants
python scripts/train_kv_recovery.py --config configs/kv_recovery/ministral_3b.yaml --run-name demo_cur_kv \
    --set kv_compression.kv_compressor=cur --trainable-attention-projections k_proj,v_proj --sensitivity-top-k 16
python scripts/train_kv_recovery.py --config configs/kv_recovery/ministral_3b.yaml --run-name demo_static \
    --trainable-layers last_n:4                      # the position heuristic instead of the measurement

# on SLURM (SET = newline-separated dotted overrides)
CONFIG=configs/kv_recovery/ministral_3b.yaml RUN_NAME=demo SET=$'kv_compression.kv_compressor=cur' \
    sbatch scripts/slurm/kv_recovery_train.sbatch
```

What happens, in order (`scripts/train_kv_recovery.py`): seeds + determinism flags → teacher and student loaded through
the production `ResearchAdapter` (bitwise identical weights asserted) → train / val windows → sensitivity measurement
on the calibration windows and top-k selection → freeze everything but the selected projections → same-model check
(teacher == uncompressed student, bitwise) → 64 AdamW steps (FP32 masters, BF16 forward/backward, grad-accum 4,
lr 1e-5, clip 1.0) with validation every 8 steps → frozen-weight check → delta checkpoint.

Outputs in `outputs/kv_recovery/<run_name>/` and what to verify:

| file | correct when |
|---|---|
| `layer_sensitivity.json` / `.csv` | `selected` equals the layers step 5 reported for the same compressor and ratio |
| `trainable_parameters.txt` | only `self_attn.{q,o}_proj.weight` (or `{k,v}_proj`) of the selected layers; 100.7 M (2.94 %) for 4 Ministral q+o layers, 125.8 M for 4 Qwen3.5 q+o layers |
| `sanity_checks.json` | every `passed` is `true`: `same_model`, `check3_gradients_only_on_trainable`, `frozen_sample_unchanged`, `all_frozen_bitwise_equal_to_teacher`, `layer_selection` |
| `val_loss.csv` | `val_loss` falls from step 0 to step 64 (reference: 0.107 → 0.087 for Ministral knorm q+o; 0.139 → 0.047 for Ministral cur k+v top-16) |
| `train_metrics.jsonl` | 64 rows, finite losses, no `train_metrics.unstable.jsonl` (the instability restart never fired) |
| `checkpoint/` | `adapted_weights.safetensors` (only the trained tensors + FP32 masters) and `metadata.json` with `layer_selection`, sha256 of original / adapted / frozen-sample tensors, the compression block, seeds, packages, git commit |

Cost: 7–10 min and 21–29 GiB on one H200 (both 3–4B models resident).

## 7. Three-way evaluation and the recovery report

```bash
python scripts/eval_kv_recovery.py run --run-name demo --dry-run          # shows the 9 cells and their barcodes
python scripts/eval_kv_recovery.py run --run-name demo --submit           # dense / compressed / compressed_recovered x RULER-16K, RULER-32K, LongBench-16
python scripts/eval_kv_recovery.py report --run-name demo --n-resamples 2000
python scripts/kv_recovery_pilot_summary.py --glob 'outputs/kv_recovery/demo*'
```

The driver builds every arm from the run's own `config.yaml` (identical compression block, prompt shaping, subsets,
seeds, `query_aware: false`, `deterministic: true`); `compressed` and `compressed_recovered` differ only in
`llm_kwargs.weight_delta`, which `HFAdapter` applies after loading while verifying the base tensors' sha256 (a
wrong base or a double application raises). Cells are barcode-named under `outputs/kv_recovery/eval/<model>/` and
shared, so dense and compressed cells run once per model and compressor. `report` re-scores `predictions.csv` with
each benchmark's own scorer, pairs rows across arms and writes `eval_results.{json,md}` with `compression_drop`,
`recovery`, `recovery_fraction` and paired task-stratified bootstrap CIs; the summary script tabulates runs.

Correct: `report` prints no `arms are not comparable` error, `comparability.ok` is true in `eval_results.json`, the
`dense` and `compressed` macro scores match the shared cells (Ministral 89.2 / 88.4 / 44.3 dense; 29.3 / 27.5 /
29.9 knorm-compressed on RULER-16K / RULER-32K / LongBench), and the recovered arm differs. Reference cell: Ministral
knorm q+o layers 12–15 → RULER-16K 31.6, recovery +2.3 [1.2, 3.3]. Cell cost on one H200: RULER-16K ≈ 50 min,
RULER-32K ≈ 70 min, LongBench ≈ 2.3 h.

To evaluate a delta from any ordinary research-backend config instead of the driver, add
`llm_kwargs: {weight_delta: {path: outputs/kv_recovery/demo/checkpoint, sha256: <adapted_weights digest>, strict: true}}`.

## 8. The pre-registered matrix

```bash
python scripts/kv_recovery_matrix.py --primary --dry-run                                   # 24 cells, done cells marked
python scripts/kv_recovery_matrix.py --primary --trainable qo_sens4,kv_sens16 --submit     # the sensitivity-selected cells
python scripts/kv_recovery_matrix.py --models ministral_3b --compressors cur --trainable kv_sens16 --submit
for r in outputs/kv_recovery/*_16k_*_r075_*; do python scripts/eval_kv_recovery.py run --run-name $(basename $r) --submit; done
```

`configs/kv_recovery/matrix.yaml` pre-registers the hyper-parameters; a cell is `<model>_<context>_<compressor>_r<ratio>_<subset>`
and is skipped when its `checkpoint/metadata.json` exists or its job is queued. Expected pilot results (16K, ratio
0.75) are tabulated in `hidden_state_recovery_plan.md` §7.

## 9. Misalignment figures on RULER-16K / RULER-32K

```bash
ARGS="scripts/measure_layer_sensitivity.py --config configs/kv_recovery/ministral_3b.yaml --sources ruler16k,ruler32k \
      --compressors knorm,cur --ratios 0.75,0.5 --rows-per-task 2" sbatch --time=1:30:00 scripts/slurm/kv_recovery_python.sbatch
python scripts/plot_layer_sensitivity.py --inputs outputs/kv_recovery/sensitivity --out-dir docs/figures   # CPU, needs matplotlib
```

Benchmark sources are analysis only (their output is never read by the trainer). Writes
`summary__ruler16k.md` / `summary__ruler32k.md` next to the PG-19 profiles and the figures
`docs/figures/<model>__profiles.{png,svg}` and `<model>__tasks__<compressor>_r075.{png,svg}`. Correct: on Ministral
the RULER profiles peak at layers 13–16 for both compressors and the 16K and 32K curves coincide; on Qwen3.5
layers 0–2 are exactly 0 and the profile climbs to the last K/V layer.

## Reproducibility notes

* Seeds: `seed: 42` (python / numpy / torch / CUDA), `data.seed: 42` for window selection (+1 validation, +2
  calibration), `optim.shuffle_seed: 0`; `deterministic: true` pins deterministic algorithms and the flash + math SDPA
  backends (cuDNN SDPA disabled; `FRAMEWORK_VERSION` 2). Evaluation generations are bitwise reproducible across jobs;
  training is reproducible up to SDPA kernel noise in the backward (not claimed bitwise).
* No benchmark data in training or selection: the corpus is leakage-filtered against every evaluated context, the
  calibration windows are PG-19 excerpts, and step 9's benchmark measurements are analysis only.
* Every run records its provenance (`metadata.json`: git commit, package versions, hardware, config digest, sha256 of
  the corpus files and of every trained tensor); `apply_delta` refuses a delta whose base tensors do not hash to the
  originals it was trained from.
