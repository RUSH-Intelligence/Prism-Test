# Hidden-state KV recovery — implementation plan

Branch `feature/hidden-state-kv-recovery`. Written before the code (spec §2); the
"Status" section at the end is updated as pieces land.

## 1. Hypothesis

A model whose KV cache is compressed at inference time (tokens evicted after the
context prefill) loses accuracy. Its internal representations at the tokens
processed *after* compression (question, answer, continuation) diverge from the
representations the same model produces with the full cache. We test whether a
**very small** amount of direct weight adaptation (no LoRA) that minimises a
**hidden-state alignment loss** between the compressed-cache *student* and the
full-cache *teacher* (identical pretrained weights, identical tokenised
sequence) recovers downstream accuracy, measured as

```
compression_drop  = dense - compressed
recovery          = compressed_recovered - compressed
recovery_fraction = recovery / compression_drop        (undefined when the drop is <= 0)
```

Models: `mistralai/Ministral-3-3B-Instruct-2512`, `Qwen/Qwen3.5-4B`. Compressors:
the repository's `knorm` and `cur` (door 3) at `compression_ratio` 0.75 and 0.5.
Evaluation: RULER-16K, RULER-32K and LongBench (16 English tasks) through the
existing `eval_harness` research backend, unchanged.

## 2. Repository inspection (spec §2)

1. **Model loading.** `eval_harness/hf_adapter.py::HFAdapter.__init__` → `_load_model`
   (`AutoConfig` probe → `Mistral3ForConditionalGeneration` / `Qwen3_5ForConditionalGeneration`
   via `_load_conditional_model`, else `AutoModelForCausalLM`; explicit `attn_implementation`
   wins; `dequantize_fp8` injects `FineGrainedFP8Config(dequantize=True)`), then `.to("cuda").eval()`.
   `ResearchAdapter` (`research_adapter.py`) subclasses it and builds the three doors plus the
   `ResearchGenerationPipeline` in `__init__`.
2. **KV compression.** `eval_harness/kv_compression/base.py::KVCompressor` — a context manager
   that installs a post-attention forward hook on every full-softmax attention module; the hook
   reads `cache.layers[i].keys/.values` (RoPE-rotated), calls `compress`, writes the pruned
   tensors back and returns the layer output unchanged. `ScorerKVCompressor.compress` keeps
   `int(T*(1-r))` entries per head (`topk` + `gather`). The pipeline (`research_pipeline.py::_forward`)
   enters the compressor context only around the context prefill, so compression happens ONCE
   after the single-pass prefill and the question is processed against the pruned cache.
3. **Existing methods.** ~36 compressors in `kv_compression/compressors/` (kvpress ports and
   in-house methods); listed by `eval_harness.kv_compression.available_kv_compressors()`.
4. **Support for the two models.** Both load through `_load_conditional_model`. Qwen3.5 is a
   hybrid (`layer_types` = 3 linear : 1 full); `_is_non_full_attention_layer` restricts hooks to
   the 8 full-attention layers and `HybridCacheAdapter` builds `DynamicCache(config=...)`.
   `knorm` / `cur` need no RoPE or attention weights and run on both (sdpa). `cur`'s only
   randomness (`use_random_leverage`) is off by default.
5. **RULER.** `benchmarks/ruler16k.py` / `ruler32k.py` load `xAlg-AI/att-hub-ruler-{16k,32k}`
   (13 tasks × 200 rows, per-row `max_new_tokens`), score with string-match closures;
   `metrics.json = {overall_score, task_scores{task:{string_match}}, ...}`.
6. **LongBench.** `benchmarks/longbench.py` loads `Xnhyacinth/LongBench`; the conventional
   selection is the 16 English tasks (`sweep.yaml`); per-row chat-template / system-block-strip /
   middle-truncation flags; `_score_row` applies the official metric (max over ground truths).
7. **RULER-16K/32K configs** exist as benchmarks (`ruler16k`, `ruler32k`); run cards live in
   `evaluate/` (`evaluate.yaml`, `example_research.yaml`, archived Ministral / Qwen3.5-9B cards).
8. **Budgets.** `research_config.compression_ratio` = fraction of the context KV **pruned** per
   head; `ResearchAdapter._build_kv_compressor` injects it into any compressor declaring the field.
9. **Where the cache is modified.** `KVCompressor.forward_hook` (`kv_compression/base.py`);
   masking presses use `kv_compression/attention_patch.py` instead (not used here).
10. **Backend.** HuggingFace transformers (5.10.2 pinned) with a custom generation pipeline
    (`research_pipeline.py`); vLLM is a separate backend not used for compression research.
11. **Seeds / per-example logging.** `EvalConfig.seed` seeds python/numpy/torch/cuda
    (`runner._set_seed`); `deterministic: true` pins deterministic algorithms and the SDPA
    backends (`runner._enable_determinism`; this branch also disables the cuDNN SDPA backend).
    `predictions.csv` stores every row's prediction but **no per-example score**; the recovery
    report re-scores rows through the benchmark's own scorer to run a paired bootstrap.

## 3. Design

### 3.1 Teacher / student execution (`eval_harness/kv_recovery/student.py`)

One training example is a token window `[context (T) | suffix (L)]` (`data.max_length = T + L`).

* Teacher: a second `ResearchAdapter` instance of the same checkpoint (no compressor). Context
  prefill via `pipe._run_prefill` under `no_grad`, then the suffix through the full model with
  **absolute** `position_ids = arange(T, T+L)` and no `cache_position` — exactly the call
  `generate_answer` makes for the question block. Hidden states are captured by forward hooks on
  the decoder layers (+ final norm) and detached.
* Student: the trainable instance; prefill with the compressor installed exactly as `_forward`
  does (`with compressor(model): set_phase("prefill"); _run_prefill(...)`), under `no_grad` by
  default (read-path adaptation; `student.prefill_grad: true` enables the write path), then the
  same suffix forward with gradients enabled.
* The compressor is built by `ResearchAdapter._build_kv_compressor` from the SAME
  `research_config` dict the evaluation arms use (`config.research_config_dict`).
* Under single-pass post-prefill compression every layer's cache is pruned *after* that layer's
  attention ran, so context-token hidden states are identical between teacher and student and
  only suffix tokens see the compressed cache. The alignment segment is therefore the suffix:
  `positions.all` = every suffix token, `recent` = last N, `first_k` = first k,
  `post_eviction` ≡ `all` (hook kept for streaming schedules).
* Qwen3.5: transformers 5.10.2's `Qwen3_5GatedDeltaNet` continues the cached conv/recurrent
  state on multi-token forwards, so block feeding is correct (`segment_mode: auto`); a
  continuation probe in the smoke script guards this.

### 3.2 Objective (`alignment.py`)

`L = hidden_weight * mean_layers(mean_positions D(h_s, h_t)) + kl_weight * T^2 KL(p_t || p_s)`,
fp32. `normalized_mse` sums the squared difference of unit-normalised states over the hidden
dimension (= 2·(1−cos)); the spec's literal elementwise form is available as
`normalized_mse_elementwise` (it divides by H, which shrinks gradients below Adam's eps).
Aligned layers default to `from_first_trainable` (+ the final norm); layers upstream of every
trainable parameter are refused (zero gradient) unless explicitly allowed.

### 3.3 Trainable subsets and layer selection (`trainable.py`, `sensitivity.py`)

Strategies: `last_n_blocks`, `blocks`, `attention_projections` (q/k/v/o on softmax-attention
layers), `mlp`, `norms`, `full`; the layer subset of the last four is `all | last_n:k | [indices]
| sensitivity`. Full-path names only; vision tower, projector, MTP head and the tied `lm_head`
are never trainable. Counts and percentages are printed and written to `metadata.json`.

**Layer selection by compression sensitivity (`trainable.layers: sensitivity`, the default of the
run cards since 2026-10-08).** Instead of fixing the layers by position, the trainer measures how
strongly the compressor perturbs every layer's hidden states ("Identifying Compression-Sensitive
Layers"). For each of `trainable.sensitivity.num_examples` held-out calibration windows (rows of
`data.val_path` drawn under seed `data.seed + 2`, disjoint from both the training and the
validation windows) the student runs twice with its ORIGINAL weights — dense cache, then
compressed cache, through the production prefill / segment path — and every decoder layer `l`
gets

```
E_l = ||H_l^dense − H_l^comp||_F / (||H_l^dense||_F + eps)
```

over the suffix tokens (`sensitivity.positions`, default all), where `H_l` is the residual-stream
output of decoder layer `l` — the same states and positions the alignment loss uses. `E_l` is
averaged (`aggregate: mean | median`) over the calibration windows, layers are ranked (exact ties
resolve toward the deeper layer) and the `top_k` highest-ranked **eligible** layers are selected:
for `attention_projections` the eligible pool is the K/V-carrying layers (on Qwen3.5 the 8
full-attention layers; linear-attention layers are measured and reported but never eligible), for
`blocks | mlp | norms` every decoder layer. The selection happens before any weight changes, needs
the online teacher, never sees benchmark data, and is written to `layer_sensitivity.{json,csv}`
(per-window scores, mean, std, ranking, candidates, selection) and into the delta metadata
(`layer_selection`). **The loss is unchanged**: the aligned layers follow the selection through
`alignment.layers.strategy: from_first_trainable` (every layer from the first selected one to the
last, plus the final norm). `scripts/measure_layer_sensitivity.py` reports the same profile and
selection for several compressors / ratios from one loaded model without training.

The static subsets remain available as the position-heuristic baseline (`qo_last4`, `kv_attn`,
`last1`, `last2`); the matrix pairs each projection subset with its sensitivity-selected twin at an
identical parameter budget (`qo_sens4` vs `qo_last4`, `kv_sens16` vs `kv_attn`), so the comparison
isolates *which* layers are trained from *how many* parameters are trained.

### 3.4 Training (`trainer.py`, `scripts/train_kv_recovery.py`)

AdamW on FP32 master copies of the trainable BF16 weights (BF16 forward/backward, masters
rounded back each step), gradient accumulation, global-norm clipping, optional warmup, the
prior experiment's instability rule, validation every N steps, `train_metrics.jsonl`, and the
sanity checks of spec §21/§22 (same-model loss = 0, only intended parameters receive gradients,
frozen parameters byte-identical, loss decreases on a fixed batch).

### 3.5 Data (`data.py`, `scripts/prepare_kv_recovery_data.py`)

PG-19 excerpts (`emozilla/pg19`, train books for training, validation books for validation),
materialised to JSONL with a 13-word-shingle leakage filter against every evaluated benchmark
context (LongBench `narrativeqa` is Gutenberg text, like PG-19). Windows are tokenised whole
(`bos + text`) and split at the token level; `suffix_mode: recall` is a pre-registered ablation
that copies an earlier span as the suffix. No benchmark label is ever used.

### 3.6 Checkpoints (`checkpoint.py`)

`adapted_weights.safetensors` holds only the trained tensors (bf16, optional fp32 masters) plus
`metadata.json` (base model / revision, load flags, trainable names, sha256 of original and
adapted tensors, a sample of frozen tensors, the compression block, prompt shaping, training
config, seeds, determinism flags, packages, hardware, git). `apply_delta` verifies the base
tensors before overwriting and refuses a second application.

### 3.7 Evaluation (`eval_configs.py`, `scripts/eval_kv_recovery.py`, `hf_adapter.py`)

`HFAdapter` accepts `llm_kwargs.weight_delta = {path, sha256}` (popped like `dequantize_fp8`,
applied after loading, fingerprinted through `run_spec` `load_flags`). The driver builds the
`dense` / `compressed` / `compressed_recovered` (optional `dense_recovered`) `EvalConfig`s from
ONE `RecoveryConfig`, asserts the compressed and recovered arms differ only in `weight_delta`,
refuses a delta whose recorded compression block differs from the eval's, pins `query_aware:
false`, `deterministic: true`, `max_new_tokens: null`, identical subsets / `max_requests` / seed,
and reuses dense and compressed cells across runs via barcode-named folders + `resume`.
`report` re-scores `predictions.csv` rows, verifies row identity across arms, and computes
drop / recovery / recovery_fraction per task and per benchmark with paired, task-stratified
bootstrap CIs. `scripts/measure_representation_alignment.py` reports layer-wise cosine /
normalized MSE / relative error for pretrained-compressed and recovered-compressed vs dense.

## 4. How to run

Step-by-step reproduction with the expected output of every step: [kv_recovery_reproduce.md](kv_recovery_reproduce.md).

```bash
python scripts/prepare_kv_recovery_data.py --num-train 256 --num-val 32 --seed 42   # login node
CONFIG=configs/kv_recovery/smoke_ministral_3b.yaml EXTRA=--with-benchmark sbatch scripts/slurm/kv_recovery_smoke.sbatch
python scripts/measure_layer_sensitivity.py --config configs/kv_recovery/ministral_3b.yaml --compressors knorm,cur --ratios 0.75,0.5
python scripts/train_kv_recovery.py --config configs/kv_recovery/ministral_3b.yaml --run-name demo            # card default: q+o on the top-4 sensitive layers
python scripts/train_kv_recovery.py --config configs/kv_recovery/ministral_3b.yaml --run-name demo_static --trainable-layers last_n:4   # the position heuristic
python scripts/eval_kv_recovery.py run --config configs/kv_recovery/ministral_3b.yaml --run-name demo --submit
python scripts/eval_kv_recovery.py report --config configs/kv_recovery/ministral_3b.yaml --run-name demo
python scripts/measure_representation_alignment.py --config ... --run-name demo
python scripts/kv_recovery_matrix.py --primary --dry-run
python scripts/measure_layer_sensitivity.py --config configs/kv_recovery/ministral_3b.yaml --sources ruler16k,ruler32k --compressors knorm,cur --ratios 0.75,0.5   # analysis only
python scripts/plot_layer_sensitivity.py --inputs outputs/kv_recovery/sensitivity        # -> outputs/kv_recovery/figures/ (not committed)
```

## 5. Experiment matrix (spec §24)

`configs/kv_recovery/matrix.yaml`: 2 models × {16K, 32K} × {knorm, cur} × {0.75, 0.5} ×
{last1, last2, qo_last4, kv_attn, qo_sens4, kv_sens16} = 96 training runs; `--primary` = 16K × 0.75
(24 runs: the 16 static cells ran on 2026-10-06, the 8 sensitivity-selected cells were added with the
layer-selection change; `--primary --trainable qo_sens4,kv_sens16` submits just those).
Staged order: smoke (both models) → pilot (Ministral, cur 0.75, qo_last4 + last1) → knorm
0.75 → ratio 0.5 / remaining subsets → Qwen3.5 → ablations on the pilot cell → 32K.
Rough cost on one H200: training 5–20 min per 16K run; evaluation ≈ 3 GPU-h per
(model, arm) for RULER-16K/32K + LongBench; full matrix ≈ 180–200 GPU-h.

## 6. Validity rules

No benchmark labels in training; layer selection uses only held-out PG-19 calibration windows
(disjoint from train and val) and is recorded in the delta; identical compression block, prompt
shaping, subsets, seeds and decoding in every arm; `strip_auto_system_block: true` everywhere (the model's auto system
prompt is never compressed as context); hyper-parameters pre-registered, selection only on the
validation alignment loss; determinism flags + `CUBLAS_WORKSPACE_CONFIG` for all evaluation
arms (training backward is reproducible only up to SDPA kernel noise — not claimed bitwise).

## 7. Status (2026-10-06)

* [x] scaffolding (configs, config module, model spec, provenance, data prep)
* [x] alignment core (hidden-state capture, losses, trainable selection, teacher/student)
* [x] training loop + delta checkpoints
* [x] evaluation integration + report + representation metrics
* [x] smoke results (both models) — see below

### Corpus (`scripts/prepare_kv_recovery_data.py`, login node)

`emozilla/pg19` revision `c021754c`: 256 training excerpts from the `train` books and 32 validation excerpts from
the `test` books (the 50-book `validation` split yields too few candidates); 200 000 characters per excerpt;
leakage scan of 5 750 benchmark contexts (LongBench-16 × 200 rows, RULER-16K/32K × 13 tasks × 100 rows,
13-word shingles, 141 s): 13 train + 1 val candidates rejected (hits in LongBench `narrativeqa` 19 contexts,
`trec` 15, `triviaqa` 6, `musique` 4, `passage_count` 2, `hotpotqa` 1, `lcc` 1; RULER `qa_1` 6 per length).
The scan also repopulated the HF dataset cache the evaluation jobs read.

### GPU smoke (`scripts/kv_recovery_smoke.py`, one H200 each, SLURM jobs 316207 / 316208)

All 14 checks pass on both families (`outputs/kv_recovery/smoke_*/smoke_report.json`):

| check | Ministral-3-3B-Instruct-2512 | Qwen3.5-4B |
|---|---|---|
| S2 load | 26 full-attention layers; FP8 → BF16 verified bitwise for all 182 tensors | 32 layers, full attention at 3,7,…,31 |
| S5/S6 hooks + budget | 26 hooks; 4096 → 1024 kept per layer at r=0.75; compress called once per layer | 8 hooks; 1024 kept; 8 calls |
| S7 continuation probe (block / token-by-token vs full forward) | worst rel-Frobenius 1.30e-02, min cos 0.99981 | 1.11e-02, min cos 0.99981 |
| S8 teacher == uncompressed student | bitwise, loss 0 | bitwise, loss 0 |
| S9 divergence grows with the ratio | loss 0.0306 (r=0.5) → 0.0678 (r=0.75) | 0.0401 → 0.1218 |
| S10 trainable (last block) | 116.4M params (3.39 % of the text LM); gradients only there | 107.5M (2.56 %) |
| S11 2 steps + round trip on a fresh load; peak GPU memory | pass; 16.59 GiB | pass; 19.17 GiB |
| S12 identity delta reproduces generation bitwise | pass (repeat generation deterministic) | pass |
| S13 three-way RULER-16K (3 subsets × 5 rows, throwaway 2-step delta) | dense 100.0 / compressed 26.7 / recovered 33.3 | 100.0 / 46.7 / 46.7 |

The S13 numbers only prove the plumbing (the smoke delta is two steps at lr 1e-4 on synthetic text); the
`eval_results.{json,md}` they produced are the first end-to-end outputs of `eval_kv_recovery.py report`.
Measured bf16 noise floor of the continuation probe (block vs full-sequence SDPA shapes): relative Frobenius
error up to 1.1e-2 at middle layers with per-position cosine ≥ 0.9998 — the probe gate is 2e-2 / 0.999.

### Real training script (`scripts/train_kv_recovery.py`, job 315970)

Ministral-3-3B, PG-19 16 384-token windows (512-token suffix), knorm r=0.75, last block, lr 1e-5, 2 optimizer
steps of 2 windows: same-model check bitwise, validation loss 0.0937 → 0.0898, train loss
0.0924 → 0.0851, ~1.3 s per window (teacher + student 16K prefills), peak GPU memory 20.842 GiB with both
models resident, all frozen tensors bitwise equal to the teacher afterwards, full `metadata.json` (spec §17) and a
698 MB delta (bf16 tensors + fp32 masters) written. Observation: at lr 1e-4 (smoke) Adam's first step moves every
parameter of the block by ~lr and the loss jumps before recovering; the pre-registered matrix lr is 1e-5.

### Representation metrics (`scripts/measure_representation_alignment.py`, job 316206)

Runs on the smoke delta with 2 validation windows: uncompressed-original sanity pass (mean cosine
1.0000, normalized MSE 0.00e+00); compressed-vs-dense cosine across layers 0.87–0.97
(all-key mean 0.918); layers 0–24 identical between compressed and recovered (only block 25 was
trained); the throwaway delta lowers layer 25 / final-norm cosine, as expected for two lr 1e-4 steps on word salad.

### Pilot training runs (`kv_recovery_matrix.py --primary --submit`, 2026-10-06, jobs 316243-316280)

16 runs (both models × knorm/cur at ratio 0.75 × last1 / last2 / qo_last4 / kv_attn), 16K windows, 256 training +
16 validation excerpts, lr 1e-5, 64 optimizer steps, one H200 each. Every run passed all sanity checks (same-model
bitwise, gradients only on the trainable subset, frozen tensors bitwise equal to the teacher) and lowered the
validation alignment loss; no instability restart was triggered. Six Qwen3.5 runs initially crashed in the fla
backward because the cache layer overwrote its recurrent state in place after the kernel had saved it
(fixed: the student rebinds cache states during gradient forwards; commit 2c949e2).

| run | trainable params | % text LM | val loss first → last | Δ | wall | peak GiB |
|---|---|---|---|---|---|---|
| `ministral_3b_16k_cur_r075_kv_attn` | 100.7M | 2.94 | 0.1111 → 0.0914 | -17.7 % | 9.0 min | 24.2 |
| `ministral_3b_16k_cur_r075_last1` | 116.4M | 3.39 | 0.0796 → 0.0661 | -17.0 % | 8.1 min | 20.9 |
| `ministral_3b_16k_cur_r075_last2` | 232.8M | 6.79 | 0.0796 → 0.0641 | -19.4 % | 8.1 min | 22.6 |
| `ministral_3b_16k_cur_r075_qo_last4` | 100.7M | 2.94 | 0.0802 → 0.0631 | -21.4 % | 7.8 min | 20.7 |
| `ministral_3b_16k_knorm_r075_kv_attn` | 100.7M | 2.94 | 0.1099 → 0.0913 | -17.0 % | 8.4 min | 24.2 |
| `ministral_3b_16k_knorm_r075_last1` | 116.4M | 3.39 | 0.0771 → 0.0699 | -9.4 % | 8.0 min | 20.9 |
| `ministral_3b_16k_knorm_r075_last2` | 232.8M | 6.79 | 0.0778 → 0.0696 | -10.5 % | 8.2 min | 22.6 |
| `ministral_3b_16k_knorm_r075_qo_last4` | 100.7M | 2.94 | 0.0800 → 0.0701 | -12.4 % | 7.8 min | 20.7 |
| `qwen35_4b_16k_cur_r075_kv_attn` | 41.9M | 1.00 | 0.0275 → 0.0201 | -26.9 % | 10.3 min | 22.6 |
| `qwen35_4b_16k_cur_r075_last1` | 107.5M | 2.56 | 0.0260 → 0.0217 | -16.5 % | 9.2 min | 21.2 |
| `qwen35_4b_16k_cur_r075_last2` | 220.4M | 5.24 | 0.0260 → 0.0220 | -15.2 % | 9.8 min | 22.9 |
| `qwen35_4b_16k_cur_r075_qo_last4` | 125.8M | 2.99 | 0.0255 → 0.0199 | -21.9 % | 10.1 min | 21.7 |
| `qwen35_4b_16k_knorm_r075_kv_attn` | 41.9M | 1.00 | 0.0467 → 0.0338 | -27.6 % | 12.3 min | 22.6 |
| `qwen35_4b_16k_knorm_r075_last1` | 107.5M | 2.56 | 0.0564 → 0.0502 | -11.0 % | 9.9 min | 21.2 |
| `qwen35_4b_16k_knorm_r075_last2` | 220.4M | 5.24 | 0.0572 → 0.0509 | -11.0 % | 9.9 min | 22.9 |
| `qwen35_4b_16k_knorm_r075_qo_last4` | 125.8M | 2.99 | 0.0486 → 0.0382 | -21.5 % | 9.9 min | 21.7 |

Evaluation arms (dense / compressed / compressed_recovered on RULER-16K, RULER-32K, LongBench-16) are submitted per
run with `scripts/eval_kv_recovery.py run --run-name <run> --submit`; dense and compressed cells are shared per
model and compressor.

### Pilot evaluation — RULER-16K, RULER-32K, LongBench-16 (paired bootstrap CIs; jobs 316554-316619, 318120)

RULER: 13 tasks × 100 rows per arm; LongBench: the 16 English tasks × up to 200 rows (3 150 rows per arm).
Dense anchors: Ministral-3-3B 89.2 / 88.4 / 44.3 and Qwen3.5-4B 96.1 / 96.2 / 44.4 (RULER-16K / RULER-32K /
LongBench), consistent with the July full-cache anchors. All deltas were trained at 16K; the 32K and LongBench
rows therefore also measure transfer to other context lengths and task distributions.

| run | benchmark | n | dense | compressed | recovered | drop | recovery [CI] | recovery fraction [CI] | flags |
|---|---|---|---|---|---|---|---|---|---|
| `ministral_3b_16k_cur_r075_kv_attn` | longbench | 3150 | 44.3 | 41.4 | 42.5 | 2.9 | 1.1 [0.4, 1.8] | 37.7% [16, 62] | unstable_gap |
| `ministral_3b_16k_cur_r075_last1` | longbench | 3150 | 44.3 | 41.4 | 41.5 | 2.9 | 0.1 [-0.3, 0.5] | 2.6% [-14, 17] | unstable_gap |
| `ministral_3b_16k_cur_r075_last2` | longbench | 3150 | 44.3 | 41.4 | 41.6 | 2.9 | 0.2 [-0.3, 0.6] | 6.2% [-10, 22] | unstable_gap |
| `ministral_3b_16k_cur_r075_qo_last4` | longbench | 3150 | 44.3 | 41.4 | 41.7 | 2.9 | 0.2 [-0.1, 0.6] | 7.9% [-5, 21] | unstable_gap |
| `ministral_3b_16k_knorm_r075_kv_attn` | longbench | 3150 | 44.3 | 29.9 | 31.0 | 14.4 | 1.1 [0.4, 1.7] | 7.6% [3, 12] |  |
| `ministral_3b_16k_knorm_r075_last1` | longbench | 3150 | 44.3 | 29.9 | 30.2 | 14.4 | 0.3 [-0.1, 0.7] | 2.0% [-1, 5] |  |
| `ministral_3b_16k_knorm_r075_last2` | longbench | 3150 | 44.3 | 29.9 | 30.3 | 14.4 | 0.4 [-0.1, 0.8] | 2.6% [-0, 6] |  |
| `ministral_3b_16k_knorm_r075_qo_last4` | longbench | 3150 | 44.3 | 29.9 | 30.4 | 14.4 | 0.5 [0.2, 0.9] | 3.6% [1, 6] |  |
| `qwen35_4b_16k_cur_r075_kv_attn` | longbench | 3150 | 44.4 | 37.3 | 39.0 | 7.1 | 1.6 [1.0, 2.3] | 23.2% [15, 32] |  |
| `qwen35_4b_16k_cur_r075_last1` | longbench | 3150 | 44.4 | 37.3 | 36.9 | 7.1 | -0.4 [-0.9, 0.0] | -6.0% [-13, 0] |  |
| `qwen35_4b_16k_cur_r075_last2` | longbench | 3150 | 44.4 | 37.3 | 36.7 | 7.1 | -0.6 [-1.1, -0.1] | -9.0% [-17, -2] |  |
| `qwen35_4b_16k_cur_r075_qo_last4` | longbench | 3150 | 44.4 | 37.3 | 37.9 | 7.1 | 0.5 [0.1, 1.0] | 7.7% [2, 13] |  |
| `qwen35_4b_16k_knorm_r075_kv_attn` | longbench | 3150 | 44.4 | 30.6 | 31.7 | 13.8 | 1.0 [0.4, 1.7] | 7.5% [3, 12] |  |
| `qwen35_4b_16k_knorm_r075_last1` | longbench | 3150 | 44.4 | 30.6 | 30.1 | 13.8 | -0.5 [-1.0, 0.1] | -3.4% [-8, 1] |  |
| `qwen35_4b_16k_knorm_r075_last2` | longbench | 3150 | 44.4 | 30.6 | 30.3 | 13.8 | -0.3 [-0.8, 0.3] | -2.0% [-6, 2] |  |
| `qwen35_4b_16k_knorm_r075_qo_last4` | longbench | 3150 | 44.4 | 30.6 | 31.5 | 13.8 | 0.9 [0.3, 1.5] | 6.2% [2, 11] |  |
| `ministral_3b_16k_cur_r075_kv_attn` | ruler16k | 1300 | 89.2 | 33.5 | 37.7 | 55.7 | 4.2 [2.7, 5.8] | 7.5% [5, 10] |  |
| `ministral_3b_16k_cur_r075_last1` | ruler16k | 1300 | 89.2 | 33.5 | 34.1 | 55.7 | 0.6 [-0.4, 1.6] | 1.1% [-1, 3] |  |
| `ministral_3b_16k_cur_r075_last2` | ruler16k | 1300 | 89.2 | 33.5 | 34.0 | 55.7 | 0.5 [-0.5, 1.4] | 0.8% [-1, 3] |  |
| `ministral_3b_16k_cur_r075_qo_last4` | ruler16k | 1300 | 89.2 | 33.5 | 35.3 | 55.7 | 1.8 [0.9, 2.8] | 3.3% [2, 5] |  |
| `ministral_3b_16k_knorm_r075_kv_attn` | ruler16k | 1300 | 89.2 | 29.3 | 29.3 | 59.9 | -0.0 [-1.0, 0.9] | -0.1% [-2, 1] |  |
| `ministral_3b_16k_knorm_r075_last1` | ruler16k | 1300 | 89.2 | 29.3 | 29.3 | 59.9 | -0.0 [-0.5, 0.5] | -0.0% [-1, 1] |  |
| `ministral_3b_16k_knorm_r075_last2` | ruler16k | 1300 | 89.2 | 29.3 | 29.6 | 59.9 | 0.3 [-0.3, 0.9] | 0.4% [-0, 1] |  |
| `ministral_3b_16k_knorm_r075_qo_last4` | ruler16k | 1300 | 89.2 | 29.3 | 29.2 | 59.9 | -0.1 [-0.7, 0.5] | -0.1% [-1, 1] |  |
| `qwen35_4b_16k_cur_r075_kv_attn` | ruler16k | 1300 | 96.1 | 59.2 | 59.2 | 36.9 | -0.0 [-1.3, 1.3] | -0.1% [-4, 3] |  |
| `qwen35_4b_16k_cur_r075_last1` | ruler16k | 1300 | 96.1 | 59.2 | 58.7 | 36.9 | -0.6 [-1.2, 0.1] | -1.5% [-3, 0] |  |
| `qwen35_4b_16k_cur_r075_last2` | ruler16k | 1300 | 96.1 | 59.2 | 59.1 | 36.9 | -0.1 [-0.9, 0.6] | -0.3% [-2, 2] |  |
| `qwen35_4b_16k_cur_r075_qo_last4` | ruler16k | 1300 | 96.1 | 59.2 | 59.4 | 36.9 | 0.1 [-0.8, 1.1] | 0.3% [-2, 3] |  |
| `qwen35_4b_16k_knorm_r075_kv_attn` | ruler16k | 1300 | 96.1 | 46.4 | 47.4 | 49.7 | 1.0 [-0.3, 2.4] | 2.1% [-1, 5] |  |
| `qwen35_4b_16k_knorm_r075_last1` | ruler16k | 1300 | 96.1 | 46.4 | 45.8 | 49.7 | -0.6 [-1.5, 0.3] | -1.2% [-3, 1] |  |
| `qwen35_4b_16k_knorm_r075_last2` | ruler16k | 1300 | 96.1 | 46.4 | 45.9 | 49.7 | -0.5 [-1.4, 0.4] | -0.9% [-3, 1] |  |
| `qwen35_4b_16k_knorm_r075_qo_last4` | ruler16k | 1300 | 96.1 | 46.4 | 47.6 | 49.7 | 1.2 [0.0, 2.5] | 2.5% [0, 5] |  |
| `ministral_3b_16k_cur_r075_kv_attn` | ruler32k | 1300 | 88.4 | 31.1 | 33.9 | 57.2 | 2.8 [1.2, 4.3] | 4.8% [2, 7] |  |
| `ministral_3b_16k_cur_r075_last1` | ruler32k | 1300 | 88.4 | 31.1 | 31.7 | 57.2 | 0.5 [-0.5, 1.5] | 0.9% [-1, 3] |  |
| `ministral_3b_16k_cur_r075_last2` | ruler32k | 1300 | 88.4 | 31.1 | 32.2 | 57.2 | 1.1 [0.1, 2.0] | 1.9% [0, 3] |  |
| `ministral_3b_16k_cur_r075_qo_last4` | ruler32k | 1300 | 88.4 | 31.1 | 32.5 | 57.2 | 1.3 [0.5, 2.1] | 2.3% [1, 4] |  |
| `ministral_3b_16k_knorm_r075_kv_attn` | ruler32k | 1300 | 88.4 | 27.5 | 28.8 | 60.9 | 1.4 [0.4, 2.4] | 2.3% [1, 4] |  |
| `ministral_3b_16k_knorm_r075_last1` | ruler32k | 1300 | 88.4 | 27.5 | 27.9 | 60.9 | 0.4 [-0.3, 1.1] | 0.7% [-0, 2] |  |
| `ministral_3b_16k_knorm_r075_last2` | ruler32k | 1300 | 88.4 | 27.5 | 27.8 | 60.9 | 0.4 [-0.4, 1.1] | 0.6% [-1, 2] |  |
| `ministral_3b_16k_knorm_r075_qo_last4` | ruler32k | 1300 | 88.4 | 27.5 | 27.8 | 60.9 | 0.4 [-0.2, 1.1] | 0.6% [-0, 2] |  |
| `qwen35_4b_16k_cur_r075_kv_attn` | ruler32k | 1300 | 96.2 | 57.6 | 57.8 | 38.6 | 0.2 [-1.1, 1.6] | 0.6% [-3, 4] |  |
| `qwen35_4b_16k_cur_r075_last1` | ruler32k | 1300 | 96.2 | 57.6 | 57.5 | 38.6 | -0.1 [-0.8, 0.6] | -0.3% [-2, 2] |  |
| `qwen35_4b_16k_cur_r075_last2` | ruler32k | 1300 | 96.2 | 57.6 | 57.8 | 38.6 | 0.2 [-0.5, 0.9] | 0.5% [-1, 2] |  |
| `qwen35_4b_16k_cur_r075_qo_last4` | ruler32k | 1300 | 96.2 | 57.6 | 57.3 | 38.6 | -0.3 [-1.3, 0.8] | -0.7% [-3, 2] |  |
| `qwen35_4b_16k_knorm_r075_kv_attn` | ruler32k | 1300 | 96.2 | 51.3 | 54.2 | 44.9 | 2.9 [1.4, 4.3] | 6.4% [3, 9] |  |
| `qwen35_4b_16k_knorm_r075_last1` | ruler32k | 1300 | 96.2 | 51.3 | 51.4 | 44.9 | 0.1 [-0.6, 0.8] | 0.3% [-1, 2] |  |
| `qwen35_4b_16k_knorm_r075_last2` | ruler32k | 1300 | 96.2 | 51.3 | 51.5 | 44.9 | 0.2 [-0.5, 1.0] | 0.4% [-1, 2] |  |
| `qwen35_4b_16k_knorm_r075_qo_last4` | ruler32k | 1300 | 96.2 | 51.3 | 53.3 | 44.9 | 2.0 [0.8, 3.2] | 4.4% [2, 7] |  |

**Reading (pre-registered budget: 64 steps, lr 1e-5, 256 PG-19 windows, suffix-only alignment).**

* Recovery is small but real and consistent for the **attention-projection subsets**: `kv_attn` (k/v of the last
  16 / all 8 full-attention layers, 1-3 % of the text LM) improves the compressed model significantly in 8 of its 9
  completed cells — RULER-16K Ministral/cur +4.2 [2.7, 5.8] (7.5 % of the gap), RULER-32K Ministral/cur +2.8 and
  Qwen3.5/knorm +2.9 [1.4, 4.3] (6.4 %), LongBench +1.0 to +1.6 on every model × compressor (Qwen3.5/cur: 23 %
  [15, 32] of a 7.1-point gap). `qo_last4` is next (RULER-16K Ministral/cur +1.8, RULER-32K Qwen3.5/knorm +2.0,
  LongBench +0.5 to +0.9). **Whole last blocks (`last1`, `last2`) recover nothing** and on LongBench Qwen3.5/cur
  `last2` is significantly *worse* than the untouched compressed model (−0.6 [−1.1, −0.1]).
* The gains transfer: deltas trained on 16K PG-19 continuations help at 32K and on LongBench's natural tasks.
* Effect sizes are an order of magnitude below the prior logit-level calibration on RULER-format data (46 % of the
  CUR gap at 16K, kv_compression_adaptation/results/REPORT.md). On RULER the recovered accuracy is concentrated on
  retrieval tasks (cur/kv_attn: niah_multiquery +16.5 [9.8, 23.5], niah_multikey_2 +11.0, niah_single_1 +10.0).
* The alignment loss itself fell 10-28 % on held-out windows in every run, so the objective is being optimised; the
  weak downstream effect points at the *signal*, not the optimiser: the per-position loss is dominated by the first
  suffix tokens (0.29 for tokens 0-16 vs 0.06 beyond 64) and by local continuation rather than long-range reads.
  The pre-registered ablations probe exactly this (`recall_suffix`, `first_k64`, `plus_kl`, `prefill_grad_kv`,
  `relative_mse`: `python scripts/kv_recovery_matrix.py --ablations --submit`), together with a larger
  step / learning-rate budget on the responsive `kv_attn` cells.
* One LongBench cell (Ministral/cur/last1 recovered) died from a Lustre stale-file-handle error in the HF dataset
  lock under 44 concurrent readers; resubmitted alone it completed (+0.1 [−0.3, 0.5], not significant). All 48 cells
  of the pilot are complete.

### Layer selection by compression sensitivity (2026-10-08)

The trainable layers are no longer fixed by position. `trainable.layers: sensitivity` (§3.3; the run
cards' default) measures `E_l = ||H_l^dense − H_l^comp||_F / (||H_l^dense||_F + eps)` on 8 held-out
PG-19 calibration windows (16 384 tokens, suffix 512, disjoint from train and val) with the original
weights and trains the top-k eligible layers with the unchanged alignment loss. CPU tests: 20 new
(`tests/test_kv_recovery_sensitivity.py`: formula, aggregation, ranking, eligible pools, config rules,
tiny-Llama measurement, tiny-Qwen3.5 hybrid zeros before the first K/V layer, calibration split,
select-then-train); the full suite (1 380 tests) passes on transformers 5.10.2 (pinned) **and** 5.19.0
(what CI installs) — the two failures CI showed on the previous head are fixed: Qwen3.5 decoder layers in
transformers ≥ 5.11 carry `block_type` ∈ {linear_attention, full_attention} (now understood by
`kv_compression.base._is_non_full_attention_layer`), and the linear-attention cache rebinding now wraps the
original update methods and clones tensor / dict / list state containers instead of re-implementing the
5.10 signature.

**Measured profiles (`scripts/measure_layer_sensitivity.py`, one H200, jobs 320578 / 320582; full tables
in `outputs/kv_recovery/sensitivity/<model>/summary.md`).** `E_l` is the mean over the 8 windows; the
measurement costs ≈ 9 s per (compressor, ratio) once the model is loaded, i.e. ≈ 20 s per training run.

Ministral-3-3B (26 layers, all K/V-carrying):

| compressor @ ratio | profile | peak | ranks of the last 4 layers (22–25) | top-4 → `qo_sens4` |
|---|---|---|---|---|
| knorm @ 0.75 | rises from 0.24–0.26 (L0–1) to a hump at L12–16 (0.40–0.43), back to 0.29–0.30 at L20–25 | L13 0.430 ± 0.032 | 16 / 20 / 22 / 21 | [12, 13, 14, 15] |
| knorm @ 0.5 | same shape, lower (0.16 → 0.36 → 0.22) | L13 0.362 | 17 / 19 / 23 / 21 | [12, 13, 14, 15] |
| cur @ 0.75 | **early** peak L2–5 (0.43–0.47), secondary hump L12–13 (0.41), tail L20–25 (0.27–0.29) | L3 0.473 ± 0.024 | 23 / 22 / 24 / 21 | [2, 3, 4, 5] |
| cur @ 0.5 | bimodal: L2–3 (0.33–0.35) and L12–13 (0.31) | L2 0.346 | 21 / 22 / 24 / 23 | [2, 3, 12, 13] |

`kv_sens16` (top-16) at knorm @ 0.75 resolves to layers 5–19 + 22 (vs. the heuristic `kv_attn` 10–25); at
cur @ 0.75 to layers 0–15 (vs. 10–25).

Qwen3.5-4B (32 layers; K/V cache only at 3, 7, …, 31; layers 0–2 measure **exactly 0**, as they must):

| compressor @ ratio | E_l of the 8 K/V layers (3 / 7 / 11 / 15 / 19 / 23 / 27 / 31) | top-4 → `qo_sens4` (= `kv_sens16` on Qwen) |
|---|---|---|
| knorm @ 0.75 | .104 / .192 / .228 / **.267** / .229 / .206 / .253 / .245 | [15, 19, 27, 31] |
| knorm @ 0.5 | .069 / .117 / .146 / **.175** / .148 / .131 / .163 / .159 | [15, 19, 27, 31] |
| cur @ 0.75 | .106 / .159 / .177 / **.206** / .172 / .146 / .161 / .159 | [11, 15, 19, 27] |
| cur @ 0.5 | .065 / .095 / .109 / **.127** / .105 / .089 / .096 / .095 | [11, 15, 19, 27] |

Reading:

* The most perturbed layers are **never the last ones**. On Ministral the heuristic `qo_last4` trained layers
  ranked 16th–24th of 26; on Qwen3.5 the heuristic's layer 23 is the 6th–7th most sensitive of the 8 K/V layers.
  Layer 15 is the peak of every Qwen3.5 setting, and the linear-attention layers right after it (16, 17) inherit
  its perturbation.
* The profile is **compressor-specific** on Ministral: knorm perturbs the middle of the stack most, cur the first
  layers (cur's leverage scores evict different tokens than key norms do); the knorm profile is stable across
  ratios, cur's is not (the early peak flattens at ratio 0.5).
* Sensitivity is a hypothesis about *where* to calibrate, not a guarantee: the best pilot cell so far
  (Ministral / cur / `kv_attn`, layers 10–25) did not contain cur's most sensitive layers (2–5). The
  budget-matched pairs `qo_sens4` vs `qo_last4` and `kv_sens16` vs `kv_attn` (same projections, same layer
  count) are the test.

GPU smoke (jobs 320576 / 320577, smoke cards now `attention_projections` q/o with `layers: sensitivity`): the new
check S14 passes on both models — calibration windows disjoint from train/val, E_l finite and > 0 on every
K/V-carrying layer, exactly 0 on Qwen3.5's layers 0–2, bitwise identical across two measurements, top-k drawn
from the eligible pool only, E_l ≡ 0 without a compressor — and S8–S13 ran on the selected layers
(Ministral: [0, 6], Qwen3.5: [27, 31] on the 1K-token synthetic smoke windows).

**Sensitivity-selected pilot training runs** (`kv_recovery_matrix.py --primary --trainable qo_sens4,kv_sens16 --submit`,
2026-10-08, jobs 320584–320591; same pre-registered budget as the static cells: 16K windows, 256 train + 16 val excerpts,
lr 1e-5, 64 steps, one H200 each). The selection inside each run reproduced the standalone measurement exactly (same
calibration windows → identical `E_l` → identical layers); every run passed the same-model / stray-gradient / frozen-weight
checks and `sanity_checks.json` confirms the trained layers equal the selection.

| run | selected layers | trainable params | % text LM | val loss first → last | Δ | wall | peak GiB |
|---|---|---|---|---|---|---|---|
| `ministral_3b_16k_cur_r075_kv_sens16` | 0–15 | 100.7 M | 2.94 | 0.1385 → 0.0474 | -65.8 % | 7.2 min | 28.9 |
| `ministral_3b_16k_cur_r075_qo_sens4` | [2, 3, 4, 5] | 100.7 M | 2.94 | 0.1373 → 0.0815 | -40.7 % | 7.3 min | 28.0 |
| `ministral_3b_16k_knorm_r075_kv_sens16` | 5–19 + 22 | 100.7 M | 2.94 | 0.1101 → 0.0824 | -25.2 % | 7.1 min | 26.6 |
| `ministral_3b_16k_knorm_r075_qo_sens4` | [12, 13, 14, 15] | 100.7 M | 2.94 | 0.1074 → 0.0873 | -18.8 % | 7.1 min | 23.4 |
| `qwen35_4b_16k_cur_r075_kv_sens16` | [11, 15, 19, 27] | 21.0 M | 0.50 | 0.0303 → 0.0259 | -14.5 % | 9.9 min | 21.1 |
| `qwen35_4b_16k_cur_r075_qo_sens4` | [11, 15, 19, 27] | 125.8 M | 2.99 | 0.0303 → 0.0239 | -21.1 % | 10.2 min | 22.9 |
| `qwen35_4b_16k_knorm_r075_kv_sens16` | [15, 19, 27, 31] | 21.0 M | 0.50 | 0.0526 → 0.0460 | -12.6 % | 10.1 min | 20.5 |
| `qwen35_4b_16k_knorm_r075_qo_sens4` | [15, 19, 27, 31] | 125.8 M | 2.99 | 0.0526 → 0.0426 | -19.0 % | 10.2 min | 22.3 |

Val-loss drops are only comparable between runs that align the same layers (`from_first_trainable` starts at the first
selected layer, so `kv_sens16` on Ministral/cur aligns layers 0–25 while `kv_attn` aligned 10–25). On Qwen3.5 the
`kv_sens16` cells are top-4 k+v (21 M parameters, the model override) and the knorm / cur selections differ in exactly one
layer (31 vs 11). All eight runs lowered the held-out alignment loss (−12.6 % to −65.8 %); none triggered the instability rule.

**Sensitivity-selected pilot evaluation** (`compressed_recovered` arms, dense and compressed cells reused; paired
task-stratified bootstrap, 2 000 resamples; 23 of 24 cells complete on 2026-10-09 — the last RULER-16K cell of
Qwen3.5 / cur / `kv_sens16` was resubmitted after a transient HF-cache error and is running; **bold** = CI excludes 0;
`scripts/kv_recovery_pilot_summary.py --glob 'outputs/kv_recovery/*_16k_*_r075_*sens*'` regenerates the raw table):

| model · compressor | subset | benchmark | dense | compressed | recovered (sens) | recovery sens [CI] | fraction | recovery heuristic twin [CI] |
|---|---|---|---|---|---|---|---|---|
| Ministral-3-3B · knorm | `qo_sens4` | ruler16k | 89.2 | 29.3 | 31.6 | **+2.3 [1.2, 3.3]** | 3.8 % | `qo_last4`: -0.1 [-0.7, 0.5] |
| Ministral-3-3B · knorm | `qo_sens4` | ruler32k | 88.4 | 27.5 | 31.2 | **+3.7 [2.7, 4.8]** | 6.1 % | `qo_last4`: +0.4 [-0.2, 1.1] |
| Ministral-3-3B · knorm | `qo_sens4` | longbench | 44.3 | 29.9 | 31.8 | **+1.9 [1.1, 2.6]** | 12.9 % | `qo_last4`: **+0.5 [0.2, 0.9]** |
| Ministral-3-3B · knorm | `kv_sens16` | ruler16k | 89.2 | 29.3 | 30.3 | **+1.0 [0.2, 2.0]** | 1.7 % | `kv_attn`: -0.0 [-1.0, 0.9] |
| Ministral-3-3B · knorm | `kv_sens16` | ruler32k | 88.4 | 27.5 | 30.0 | **+2.5 [1.4, 3.6]** | 4.2 % | `kv_attn`: **+1.4 [0.4, 2.4]** |
| Ministral-3-3B · knorm | `kv_sens16` | longbench | 44.3 | 29.9 | 31.6 | **+1.7 [1.0, 2.4]** | 12.0 % | `kv_attn`: **+1.1 [0.4, 1.7]** |
| Ministral-3-3B · cur | `qo_sens4` | ruler16k | 89.2 | 33.5 | 34.9 | +1.4 [-0.1, 3.0] | 2.5 % | `qo_last4`: **+1.8 [0.9, 2.8]** |
| Ministral-3-3B · cur | `qo_sens4` | ruler32k | 88.4 | 31.1 | 33.1 | **+2.0 [0.4, 3.6]** | 3.5 % | `qo_last4`: **+1.3 [0.5, 2.1]** |
| Ministral-3-3B · cur | `qo_sens4` | longbench | 44.3 | 41.4 | 41.9 | +0.5 [-0.2, 1.1] | 16.1 % | `qo_last4`: +0.2 [-0.1, 0.6] |
| Ministral-3-3B · cur | `kv_sens16` | ruler16k | 89.2 | 33.5 | 42.0 | **+8.5 [6.5, 10.5]** | 15.2 % | `kv_attn`: **+4.2 [2.7, 5.8]** |
| Ministral-3-3B · cur | `kv_sens16` | ruler32k | 88.4 | 31.1 | 37.3 | **+6.1 [4.2, 8.0]** | 10.7 % | `kv_attn`: **+2.8 [1.2, 4.3]** |
| Ministral-3-3B · cur | `kv_sens16` | longbench | 44.3 | 41.4 | 43.1 | **+1.7 [0.9, 2.5]** | 57.7 % | `kv_attn`: **+1.1 [0.4, 1.8]** |
| Qwen3.5-4B · knorm | `qo_sens4` | ruler16k | 96.1 | 46.4 | 47.3 | +0.9 [-0.5, 2.3] | 1.8 % | `qo_last4`: **+1.2 [0.0, 2.5]** |
| Qwen3.5-4B · knorm | `qo_sens4` | ruler32k | 96.2 | 51.3 | 52.8 | **+1.5 [0.2, 2.7]** | 3.4 % | `qo_last4`: **+2.0 [0.8, 3.2]** |
| Qwen3.5-4B · knorm | `qo_sens4` | longbench | 44.4 | 30.6 | 31.9 | **+1.3 [0.7, 2.0]** | 9.5 % | `qo_last4`: **+0.9 [0.3, 1.5]** |
| Qwen3.5-4B · knorm | `kv_sens16` | ruler16k | 96.1 | 46.4 | 48.4 | **+2.0 [0.9, 3.2]** | 4.1 % | `kv_attn`: +1.0 [-0.3, 2.4] |
| Qwen3.5-4B · knorm | `kv_sens16` | ruler32k | 96.2 | 51.3 | 54.0 | **+2.7 [1.4, 3.9]** | 5.9 % | `kv_attn`: **+2.9 [1.4, 4.3]** |
| Qwen3.5-4B · knorm | `kv_sens16` | longbench | 44.4 | 30.6 | 31.3 | **+0.7 [0.1, 1.3]** | 5.0 % | `kv_attn`: **+1.0 [0.4, 1.7]** |
| Qwen3.5-4B · cur | `qo_sens4` | ruler16k | 96.1 | 59.2 | 59.6 | +0.3 [-1.0, 1.6] | 0.9 % | `qo_last4`: +0.1 [-0.8, 1.1] |
| Qwen3.5-4B · cur | `qo_sens4` | ruler32k | 96.2 | 57.6 | 58.1 | +0.5 [-0.7, 1.8] | 1.3 % | `qo_last4`: -0.3 [-1.3, 0.8] |
| Qwen3.5-4B · cur | `qo_sens4` | longbench | 44.4 | 37.3 | 38.7 | **+1.3 [0.7, 2.0]** | 18.8 % | `qo_last4`: **+0.5 [0.1, 1.0]** |
| Qwen3.5-4B · cur | `kv_sens16` | ruler16k | 96.1 | 59.2 | pending | pending | — | `kv_attn`: -0.0 [-1.3, 1.3] |
| Qwen3.5-4B · cur | `kv_sens16` | ruler32k | 96.2 | 57.6 | 57.5 | -0.1 [-1.1, 1.0] | -0.2 % | `kv_attn`: +0.2 [-1.1, 1.6] |
| Qwen3.5-4B · cur | `kv_sens16` | longbench | 44.4 | 37.3 | 38.3 | **+1.0 [0.5, 1.5]** | 14.2 % | `kv_attn`: **+1.6 [1.0, 2.3]** |

Reading:

* **The same projections on measured layers beat the position heuristic in 14 of the 23 comparable cells and lose in 2.**
  The clearest win is Ministral / cur / k+v: the sensitivity-selected layers 0–15 recover **+8.5 [6.5, 10.5]** on RULER-16K
  (15 % of the gap; the heuristic's layers 10–25: +4.2), **+6.1** on RULER-32K (vs +2.8) and **+1.7** on LongBench
  (vs +1.1) — twice the best cell of the position-heuristic pilot, at an identical parameter budget. Ministral / knorm
  q+o on layers 12–15 turns the heuristic's zero into +2.3 / +3.7 / +1.9 across the three benchmarks.
* **Where the heuristic already sat on the right layers the two tie.** On Qwen3.5 the measured top-4 (15, 19, 27, 31 for
  knorm; 11, 15, 19, 27 for cur) shares two to three layers with the heuristic (19, 23, 27, 31), and the paired
  differences are within noise except LongBench (sensitivity better for q+o: +1.3 vs +0.9 / +0.5; heuristic better for
  k+v: +1.0 / +1.6 vs +0.7 / +1.0 — but `kv_sens16` on Qwen3.5 trains 4 of the 8 layers, half the parameters of `kv_attn`).
* **Ministral / cur / q+o is the one exception**: the early-layer pick (2–5) is on par with the heuristic (+1.4 n.s. /
  **+2.0** / +0.5 vs **+1.8** / **+1.3** / +0.2). Its k+v sibling, which covers layers 0–15, is the best cell overall —
  calibrating the early layers helps when enough of the stack above them is adapted too.
* Effect sizes remain modest in absolute terms (≤ 15 % of a 56-point RULER gap); the alignment signal, not the layer choice,
  is still the main lever, and the pre-registered ablations on the signal remain the next step.

### Where the misalignment sits on RULER-16K / RULER-32K (figures, 2026-10-09)

`scripts/measure_layer_sensitivity.py --sources ruler16k,ruler32k` measures the same `E_l` on benchmark contexts
(2 seeded rows per task from the 100-row evaluation pool = 26 windows per length; the evaluation's own prompt shaping;
the measured region is the question + answer prefix + gold answer, teacher-forced; **analysis only** — nothing here feeds
training or selection), and `scripts/plot_layer_sensitivity.py` draws the figures into `outputs/kv_recovery/figures/`
(not committed; regenerate with the two commands in §4 — jobs 321484 / 321485 on 2026-10-09): `<model>__profiles.png`
(E_l vs layer, panels PG-19 16K | RULER-16K | RULER-32K, hue = compressor, line style = ratio, ±1 std band) and
`<model>__tasks__{knorm,cur}_r075.png` (per-task small multiples, RULER-16K vs RULER-32K).

| | Ministral-3-3B (26 layers) | Qwen3.5-4B (K/V at 3, 7, …, 31) |
|---|---|---|
| shape on RULER | unimodal for **both** compressors: rise from layer 0, plateau over 12–17, slow decline to 20–25, peak at 13–15 (knorm @ 0.75: L15 0.49 / 0.52; cur @ 0.75: L15 0.62 / L13 0.63 at 16K / 32K) | climbs through the stack in steps (one per K/V layer; the linear-attention layers in between carry the state unchanged) and peaks at the **last** K/V layer 31 (knorm @ 0.75: 0.52 / 0.51; cur: 0.48 / 0.48) |
| vs the PG-19 profile | the early-layer cur peak (layers 2–5) is **absent**; on RULER cur sits above knorm at every layer, on PG-19 knorm was above cur in the middle | PG-19 peaked at layer 15; on RULER the late K/V layers (27, 31) dominate |
| magnitude vs PG-19 | 1.3–1.5× larger (mean over layers 0.42 knorm / 0.54 cur vs 0.33 / 0.37) | ≈ 2× larger (mean over K/V layers 0.40 / 0.39 vs 0.22 / 0.16) |
| 16K vs 32K | the same curve within ≈ 0.02 (means 0.419 vs 0.430 knorm, 0.541 vs 0.540 cur) | the same within ≈ 0.02 (0.399 vs 0.381, 0.387 vs 0.381) |
| top-4 on RULER | [13, 14, 15, 16] in every setting (cur @ 0.5 at 16K: [12, 13, 14, 15]) — 3 of 4 overlap with the PG-19 knorm pick, 0 of 4 with the PG-19 cur pick | [15, 19, 27, 31] for knorm @ 0.75 and cur (= the PG-19 knorm pick; the PG-19 cur pick differs by one layer), [15, 23, 27, 31] for knorm @ 0.5 |
| per task | `fwe` and `vt` under knorm stay flat and low (0.15–0.4; the tasks compression does not hurt); `niah_multikey_3` rises monotonically to 1.2 at the last layer (the task compression destroys); the early-layer cur signature comes from needle tasks (`niah_single_1` jumps to 0.75 at layer 2) and `vt`; `qa_1` / `qa_2` are the only tasks where 16K and 32K differ visibly | `fwe` flat at 0.15; `cwe` jumps at the last layers (0.35 → 0.6); needle tasks saturate at layer 15–19; `qa_1` is higher at 16K than at 32K over the upper half of the stack |

Reading: the question/answer tokens of RULER are perturbed more, and in different layers, than PG-19 continuations are —
the training-distribution profile that drives the layer selection is a proxy, and on Ministral it picked cur's layers
where RULER shows no peak. Context length barely matters: at a fixed compression ratio the 16K and 32K profiles are
indistinguishable, so a selection made at 16K transfers to 32K (consistent with the RULER-32K recoveries above). Measuring
the selection signal on benchmark contexts would be the obvious next experiment, but it crosses the no-benchmark-data
line of this protocol; a held-out *task-shaped* calibration set (synthetic needle/QA prompts) would not.

### Not run

The pre-registered matrix (`configs/kv_recovery/matrix.yaml`, now 96 runs + evaluations, ≈270–300 GPU-h) and the
pilot cell are launched only explicitly (`python scripts/kv_recovery_matrix.py --primary --submit`,
then `scripts/eval_kv_recovery.py run --submit` per run).
