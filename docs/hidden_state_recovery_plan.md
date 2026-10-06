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

### 3.3 Trainable subsets (`trainable.py`)

`last_n_blocks`, `attention_projections` (q/k/v/o on softmax-attention layers, layer subset
`all | last_n:k | [indices]`), `mlp`, `norms`, `full`. Full-path names only; vision tower,
projector, MTP head and the tied `lm_head` are never trainable. Counts and percentages are
printed and written to `metadata.json`. The matrix budget-matches subsets (q+o on the last 4
attention layers, k+v on the last 16 (Ministral) / all 8 full layers (Qwen3.5)).

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

```bash
python scripts/prepare_kv_recovery_data.py --num-train 256 --num-val 32 --seed 42   # login node
sbatch scripts/slurm/kv_recovery_smoke.sbatch   # CONFIG=configs/kv_recovery/smoke_ministral_3b.yaml
python scripts/train_kv_recovery.py --config configs/kv_recovery/ministral_3b.yaml --run-name demo
python scripts/eval_kv_recovery.py run --config configs/kv_recovery/ministral_3b.yaml --run-name demo --submit
python scripts/eval_kv_recovery.py report --config configs/kv_recovery/ministral_3b.yaml --run-name demo
python scripts/measure_representation_alignment.py --config ... --run-name demo
python scripts/kv_recovery_matrix.py --primary --dry-run
```

## 5. Experiment matrix (spec §24)

`configs/kv_recovery/matrix.yaml`: 2 models × {16K, 32K} × {knorm, cur} × {0.75, 0.5} ×
{last1, last2, qo_last4, kv_attn} = 64 training runs; `--primary` = 16K × 0.75 (16 runs).
Staged order: smoke (both models) → pilot (Ministral, cur 0.75, qo_last4 + last1) → knorm
0.75 → ratio 0.5 / remaining subsets → Qwen3.5 → ablations on the pilot cell → 32K.
Rough cost on one H200: training 5–20 min per 16K run; evaluation ≈ 3 GPU-h per
(model, arm) for RULER-16K/32K + LongBench; full matrix ≈ 180–200 GPU-h.

## 6. Validity rules

No benchmark labels in training; identical compression block, prompt shaping, subsets, seeds and
decoding in every arm; `strip_auto_system_block: true` everywhere (the model's auto system
prompt is never compressed as context); hyper-parameters pre-registered, selection only on the
validation alignment loss; determinism flags + `CUBLAS_WORKSPACE_CONFIG` for all evaluation
arms (training backward is reproducible only up to SDPA kernel noise — not claimed bitwise).

## 7. Status

* [x] scaffolding (configs, config module, model spec, provenance, data prep)
* [x] alignment core (hidden-state capture, losses, trainable selection, teacher/student)
* [x] training loop + delta checkpoints
* [x] evaluation integration + report + representation metrics
* [ ] smoke results (both models)
