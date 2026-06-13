## Benchmarking with Prism-Test

> If you're new to the repo, read [README.md](README.md) first for install, layout, and a working smoke run. This guide picks up from there.

This is the working guide for **doing benchmarking research** with Prism-Test: which backend to pick for which experiment, the conditions you need to hold fixed before reporting numbers, where to plug your own code in (and at what depth), and how to add a new benchmark.

---

### Contents

- [What questions is this harness built to answer?](#what-questions-is-this-harness-built-to-answer)
- [Pick a backend](#pick-a-backend)
- [Configure a run](#configure-a-run)
  - [Conditions to control before reporting numbers](#conditions-to-control-before-reporting-numbers)
  - [Output format](#output-format)
- [Plug in your own code](#plug-in-your-own-code)
  - [Layer 1 — a KV-cache compression sketch](#layer-1--a-kv-cache-compression-sketch)
  - [Layer 2 — a custom attention kernel](#layer-2--a-custom-attention-kernel)
  - [Layer 3 — modify the research adapter itself](#layer-3--modify-the-research-adapter-itself)
- [Research backend architecture](#research-backend-architecture)
- [Add a new benchmark](#add-a-new-benchmark)
- [RAG backend setup](#rag-backend-setup)
- [Conventions & invariants](#conventions--invariants)
- [Troubleshooting](#troubleshooting)

---

## What questions is this harness built to answer?

Long-context evaluation has questions short-context harnesses don't ask. The harness is laid out so that the answer to each of these is a single config change, not a rewrite:

- **Quality vs. context length** — does my method's accuracy survive 64K? 128K? 1M?
- **Quality vs. KV budget** — for a fixed cache size, which sketch comes closest to dense attention?
- **Cost of long context** — prefill latency, decode latency, peak VRAM, KV-cache size at length L.
- **Where recall actually fails** — global sink? mid-context? local window? task type?
- **Attention vs. retrieval** — when is a RAG baseline competitive with long-context attention?

If your experiment isn't a variation on one of these, double-check that this harness is the right fit before investing.

---

## Pick a backend

| Backend     | Use when                                              | You get                                                   | You give up                                            |
| ----------- | ----------------------------------------------------- | --------------------------------------------------------- | ------------------------------------------------------ |
| `vllm`      | Production-quality numbers; large-batch eval.         | Best throughput; prefix caching across same-context Qs.   | No attention-internals access.                         |
| `hf`        | Small-context debugging; profiling.                   | Clean `_prefill`/`_decode` split; native FA2 if present.  | Slow; no prefix caching.                               |
| `research`  | Context extension, KV sketches, custom kernels.       | Rotated K/V `DynamicCache`; single full-context prefill pass. | Defaults to `attn_implementation: sdpa` (the validated parity path). |
| `rag`       | Retrieval baselines.                                  | OnePassRAG (LanceDB + llm-embedder + Ollama llama3.1).    | Different architecture — not apples-to-apples.         |

Rules of thumb:

- **"I want a baseline number on RULER-64K"** → `vllm`.
- **"I'm comparing my sketch against `knorm`"** → `research`.
- **"I'm comparing attention vs. retrieval"** → run both `research` (or `vllm`) and `rag` on the same benchmark.
- **"My research-backend numbers don't match HF reference"** → re-run with `hf` to isolate whether it's a vLLM kernel difference or your sketch.

---

## Configure a run

Ready-made configs live in [evaluate/](evaluate/): `evaluate_vllm.yaml` /
`evaluate_hf.yaml` (clean no-method baselines), `evaluate_kv.yaml`
(KV-compression sketch only), `evaluate_dca.yaml` / `evaluate_reattention.yaml`
(verified paper baselines), and `evaluate_common.yaml` (the full research
surface, including the attention_method × kv_compressor compatibility matrix). Run with:

```bash
python -m eval_harness.cli run --config_file ./evaluate/evaluate_common.yaml
```

Or override any field on the CLI:

```bash
python -m eval_harness.cli run \
  --config_file ./evaluate/evaluate_common.yaml \
  --benchmark ruler64k \
  --subsets qa_1,qa_2 \
  --backend research \
  --model meta-llama/Llama-3.1-8B-Instruct
```

For the full field reference, see [eval_harness/config.py](eval_harness/config.py) — the `EvalConfig` dataclass validates every knob in `__post_init__`.

### Conditions to control before reporting numbers

Before publishing a comparison, confirm each of the following is **intentional**, not the default that happened to be in your YAML:

| Knob                       | Why it matters                                                                                  |
| -------------------------- | ----------------------------------------------------------------------------------------------- |
| `dtype`                    | `bfloat16` vs `float16` changes scores on some benchmarks beyond 32K.                           |
| `temperature` / `top_p`    | Default `0.0` (greedy). Change only when explicitly reporting a sampling experiment.            |
| `seed`                     | Default `42`; rows are random-subsampled when `fraction < 1.0`. Same seed → same rows.          |
| `enable_prefix_caching`    | vLLM only. On by default. Turn off only when measuring cold-prefill cost.                       |
| `attn_implementation`      | `sdpa` for `research` (the parity path the prefill-method hooks are validated against); `flash_attention_2` for `hf`. vLLM ignores it. |
| `max_model_len`            | Override the model's positional cap when running past its training window.                      |
| `query_aware`              | If `True`, the question is concatenated into the context — needed for some query-aware methods. Note it in your write-up. |
| `max_requests` / `max_requests_per_subset` | Subsampling caps. Same value across compared runs or numbers don't line up.    |
| `llm_kwargs.research_config`  | Sketch name, compression ratio, target size, and `prefill_method` (+ its kwargs). Pin all of these when comparing runs.  |

### Output format

Each run writes a directory:

```
results/<benchmark>__<model>__<backend>__t0__p1__subsets_qa_1/
├── predictions.csv     # per-row inputs + predicted_answer + ground truth
├── metrics.json        # aggregated scores
└── config.yaml         # the exact, fully-resolved config
```

The raw `context` column is **not** persisted (it can be hundreds of thousands of tokens — look it up from the source dataset). Run directories never overwrite; collisions get suffixed `/1`, `/2`, ...

---

## Plug in your own code

There are three layers, from least to most invasive. **Use the smallest one that fits your work.** Going deeper than necessary means more code to maintain and more places your method can drift from the reference path.

(A fourth option — Layer 0, a new context-extension *prefill method* — is covered in [Research backend architecture](#research-backend-architecture): subclass `PrefillMethod`, register with `@register_prefill_method`.)

### Layer 1 — a KV-cache compression sketch

Use this when your method is "decide which tokens to keep / drop / rescore *after* the keys and values exist." Sketches run as a `forward_hook` on each attention layer at the end of prefill.

1. Subclass [`KVCompressor`](eval_harness/kv_compression/base.py) and implement `compress`:

   ```python
   from eval_harness.kv_compression.base import KVCompressor

   class MySketch(KVCompressor):
       def compress(self, module, hidden_states, keys, values, attentions, kwargs):
           # return (keys_pruned, values_pruned) with the same trailing shape
           ...
   ```

2. Decorate the class with `@register_kv_compressor("my_sketch")` ([eval_harness/kv_compression/compressors/registry.py](eval_harness/kv_compression/compressors/registry.py)) and drop the file into [eval_harness/kv_compression/compressors/](eval_harness/kv_compression/compressors/) — sketch modules are auto-discovered on first lookup, so adding a sketch never requires editing shared files. `ResearchAdapter._build_kv_compressor` ([eval_harness/research_adapter.py](eval_harness/research_adapter.py)) resolves `kv_compressor` through the registry and passes `kv_compressor_kwargs` to the constructor; the adapter-level `compression_ratio` is injected as a default **only when the class declares a `compression_ratio` dataclass field** (sketches that expose it as a property — `think`, `simlayerkv`, `key_rerotation`, `dms` — must be configured via `kv_compressor_kwargs` or programmatically). Composite sketches whose arguments can't be expressed as flat config kwargs (`DecodingSketch`, `PrefillDecodingSketch`) remain named special cases in `_build_kv_compressor` instead of registry entries.

3. Select it in YAML:

   ```yaml
   llm_kwargs:
     research_config:
       kv_compressor: my_sketch
       compression_ratio: 0.5
       kv_compressor_kwargs: { my_param: 123 }   # extra constructor kwargs, forwarded verbatim
       target_size: 2048
       max_context_length: 65536
   ```

   List the live registry at any time:

   ```python
   from eval_harness.kv_compression import available_kv_compressors
   print(available_kv_compressors())
   ```

Reference implementations to read first: [knorm_sketch.py](eval_harness/kv_compression/compressors/knorm_sketch.py), [reattention_sketch.py](eval_harness/kv_compression/compressors/reattention_sketch.py) (scoring base class in [scorer_sketch.py](eval_harness/kv_compression/compressors/scorer_sketch.py)), [random_sketch.py](eval_harness/kv_compression/compressors/random_sketch.py), [decoding_sketch.py](eval_harness/kv_compression/compressors/decoding_sketch.py).

Shipped sketches (mostly faithful ports of kvpress 0.5.1 presses — each module's class docstring lists its parameters, the upstream quirks replicated on purpose, and its deviations from kvpress; read it before reporting numbers):

| Sketch                   | Trigger             | What it does                                                  |
| ------------------------ | ------------------- | ------------------------------------------------------------- |
| `none`                   | —                   | Pass-through; full KV cache.                                  |
| `knorm` (alias `knorm_sketch`) | Prefill       | Keep tokens with smallest ‖K‖ (largest-norm keys evicted).    |
| `reattention` (alias `reattention_sketch`) | Prefill | Re-score with QK relevance over middle tokens.    |
| `random` (alias `random_sketch`) | Prefill     | Random baseline (uniform random scores).                      |
| `streaming_llm`          | Prefill             | Keep `n_sink` sink tokens + most recent; prune the middle. Kept keys stay at their original RoPE phases (no rerotation wrapper, as in kvpress). |
| `keydiff`                | Prefill             | Evict keys most cosine-similar to the per-head mean key direction (keeps distinctive keys). |
| `lagkv`                  | Prefill             | Lag-relative scoring: each partition is range-normalized by the next partition's min/max; std-softmax scores. |
| `cur`                    | Prefill             | CurDKV: approximate leverage scores of keys (k2) and values (v2), combined per `leverage_type`. |
| `leverage`               | Prefill             | Approximate statistical leverage scores of pre-RoPE keys via Gaussian sketch + Cholesky (Compactor component). |
| `non_causal_attention`   | Prefill             | Compactor's non-causal chunked-attention column-sum scorer (component). |
| `compactor`              | Prefill             | Full Compactor: z-normalized blend of leverage scores + non-causal attention sums over the sink-protected interior. |
| `ridge`                  | Prefill             | Value-aware query-ridge scoring (research-fork `RidgePress`, not upstream kvpress); sink + local window always kept. |
| `random_sketch_press`    | Prefill             | Research-fork `RandomSketchPress`; upstream dead-code bug replicated faithfully, so it behaves identically to `ridge` (pinned by tests). |
| `expected_attention`     | Prefill             | Predicts future attention from pre-RoPE query mean/covariance rotated to averaged future positions; optional ‖V‖ rescale. |
| `expected_attention_stats` † | Prefill         | `expected_attention` with pre-computed per-layer calibration query statistics (HF hub repo or local `stats_folder`). |
| `snapkv`                 | Prefill             | Window attention of the last `window_size` tokens scores the rest (recomputed pre-RoPE queries re-rotated to absolute positions). |
| `pyramidkv` ‡            | Prefill             | SnapKV scoring with linearly decreasing per-layer budgets (more cache for lower layers); `uniform_budget=True` degenerates to uniform SnapKV. |
| `tova`                   | Prefill             | Last-token attention row (averaged across query heads) scores previous tokens. |
| `observed_attention` \*  | Prefill             | Mean attention weight each KV pair actually received during prefill (H2O-related). |
| `qfilter` †              | Prefill             | Dot products against learned per-model Q-filters (hub collection `nthngdy/q-filters-…`). |
| `kvzap` †                | Prefill             | Lightweight surrogate model (linear/MLP from `nvidia/KVzap-…`) predicts importance from hidden states; top-k variant only (= kvpress `kvzap_mlp_head`). |
| `finch`                  | Prefill             | Delimiter-driven SnapKV-style window scoring with per-row normalization; requires `context + delimiter + question` input and a `update_model_and_tokenizer` call before entering the context. |
| `think`                  | Prefill             | Channel-wise **key** compression: zeroes the lowest-scoring key channels in place (shapes unchanged — no memory savings, by design). |
| `simlayerkv` ‡           | Prefill             | Detects "lazy" layers via last-token attention concentration; lazy layers keep only sink + recent, others keep everything. |
| `criticalkv`             | Prefill (wrapper)   | Two-stage selection around an inner scorer: stage 1 by raw scores, stage 2 rescaled by ‖W_o·V‖₁. |
| `adakv` ◊                | Prefill (masking)   | Head-adaptive top-k across all heads with a per-head `alpha_safeguard` floor. |
| `critical_adakv` ◊       | Prefill (masking)   | CriticalKV's output-projection rescaling + AdaKV's head-wise budgets. |
| `dms` ◊                  | Prefill (+ decode if `decoding=True`) (masking) | Dynamic Memory Sparsification: evicts below a score `threshold` (adaptive ratio); sliding window protected. |
| `duo_attention` ◊ †      | Prefill (masking)   | Splits KV heads into retrieval heads (full cache) vs streaming heads (sink + recent) using pre-computed per-head patterns. |
| `kvzip` ◊                | Post-prefill scoring passes (masking) | Scores KV pairs by the cross-attention they receive when the model is prompted to *reconstruct* the context (costs ~2–3× prefill). |
| `fastkvzip` ◊ †          | Prefill (masking)   | Per-layer trained gate networks predict KVzip scores from hidden states alone (`Jang-Hyun/Fast-KVzip` gates; released e.g. for Qwen3). |
| `block`                  | Prefill (wrapper)   | KeyDiff-paper block processing: streaming top-k over fixed-size blocks with an inner scorer; programmatic construction only (nested sketch arg). |
| `chunk`                  | Prefill (wrapper)   | Applies an inner scorer independently per fixed-size chunk (FINCH-style uniform selection across the context). |
| `chunkkv`                | Prefill (wrapper)   | Keeps/drops whole chunks by mean head-summed inner score (inner defaults to `knorm`, so it is YAML-constructible). |
| `composed`               | Prefill (wrapper)   | Chains multiple sketches sequentially; members may be registry names / `(name, kwargs)` pairs, so it is reachable from flat YAML. |
| `key_rerotation`         | Prefill (wrapper)   | Re-rotates kept keys to contiguous positions `0..n_kept-1` after an inner scorer prunes. Caveat: this pipeline does not rebase question/decode position ids, leaving a positional gap (warning logged). |
| `per_layer_compression` ‡ | Prefill (wrapper)  | Applies per-layer `compression_ratios` through an inner scorer (registry name + `press_kwargs` accepted). |
| `decoding_knorm`         | Decode (periodic)   | Compress mid-decode every `compression_interval` steps. Not in the registry — named special case in `_build_kv_compressor`. |
| `prefill_decoding_knorm` | Prefill + decode    | Compress at prefill end and during decode. Not in the registry — named special case in `_build_kv_compressor`. |

\* **`observed_attention` requires `attn_implementation: eager`** — only the eager forward returns attention probabilities to the hook; under the default sdpa, `attentions` is `None` and the sketch asserts. Run it with `attention_method: none`.

† **External assets with injection hooks.** `qfilter`, `kvzap`, `duo_attention`, `expected_attention_stats`, and `fastkvzip` load pre-trained/pre-computed artifacts from the HF hub in `post_init_from_model` (published only for specific models). Each exposes a constructor injection hook for offline compute nodes and tests: `q_filters` (tensor), `model_name_override` (repo-id derivation for local snapshot dirs), `attention_pattern` / `pattern_dir`, `stats_folder`, and `gates` respectively.

◊ **Masking-based presses keep the cache full-length — no memory savings, faithful attention semantics.** `adakv`, `critical_adakv`, `dms`, `duo_attention`, `kvzip`, and `fastkvzip` never physically prune: they record evicted `(batch, head, seq)` indices on `module.masked_key_indices`, and the globally installed attention patch ([eval_harness/kv_compression/attention_patch.py](eval_harness/kv_compression/attention_patch.py), applied over `ALL_ATTENTION_FUNCTIONS` at `import eval_harness.kv_compression`) overwrites those key slots with fake keys such that `exp(⟨q, k_fake⟩) == 0` on every `q_len < k_len` forward, resetting on the next full prefill. Consequences: these are quality-only baselines (logged cache lengths stay at full context length); they require a **non-eager** attention implementation (the runner default sdpa is fine; eager bypasses `ALL_ATTENTION_FUNCTIONS`); and they are incompatible with prefill methods that replace `self_attn.forward` wholesale (`dca`, `reattention_exact`) — the mask would be silently ignored. (`think` also yields no memory savings, via zeroed channels rather than masking.)

‡ **Ragged-cache methods need `flash_attention_2`.** `pyramidkv`, `simlayerkv` (with `lazy_threshold < 1.0`), and `per_layer_compression` (with unequal ratios) retain different lengths per layer; under the pinned transformers, sdpa/eager build one decode mask sized from layer 0, so these sketches' `post_init_from_model` raises unless the model runs flash-attention-2 (or the uniform/no-op escape hatch is used).

Composition caveat: prefill-method hooks fire **before** sketch hooks, and most position-sensitive scorers (snapkv, tova, finch, compactor, qfilter, expected_attention / expected_attention_stats, simlayerkv, think, …) assume vanilla absolute-position RoPE-rotated cached keys — do not combine them with `attention_method: dca` (cyclic key positions) or cache-pruning hooks; the safe, validated combination is `attention_method: none` unless a sketch's docstring says otherwise.

### Layer 2 — a custom attention kernel

Use this when you need a different *kernel* (Triton, a FlashAttention variant, custom CUDA) underneath an attention method's scoring or attention path (Door 2).

There is no `_prefill_attn_impl` / `_decode_attn_impl` seam on the adapter. Kernels are called directly from the attention method that owns them. The two shipped seams to read and model your own on:

- [`einsum_topk_func`](eval_harness/kernels/einsum_topk.py) — ReAttention's Triton kernel for fused QK scoring + top-k token selection over the cached keys.
- [`flash_attn_with_lse`](eval_harness/kernels/dca_flash.py) — DCA's FlashAttention variant returning the log-sum-exp, used to merge the intra/successive/inter attention components with online softmax.

A new kernel lives in [eval_harness/kernels/](eval_harness/kernels/) and is invoked from your attention method's `self_attn.forward` replacement (or, for the legacy `PrefillMethod` subclasses, the post-attention `prefill_forward_hook`) — see [Layer 3](#layer-3--modify-the-research-adapter-itself) and [Research backend architecture](#research-backend-architecture).

Invariant you must respect:

- Position IDs are **absolute** (token's index in the full sequence), never chunk-relative — except where a method deliberately re-rotates keys at cyclic positions (DCA stores keys at `pos % chunk_len`).

### Layer 3 — modify the research adapter itself

Use this when none of the above fits — e.g., you want a new context-extension mechanism, or to change how the pipeline installs the three doors.

Files you'll touch:

- [eval_harness/research_adapter.py](eval_harness/research_adapter.py) — the thin `HFAdapter` subclass that builds the three doors (`positional_method`, `attention_method`, `kv_compressor`) from `ResearchConfig` and runs them through the pipeline.
- [eval_harness/research_pipeline.py](eval_harness/research_pipeline.py) — `SketchTextGenerationPipeline`; the chunked-prefill loop (`_forward`, single full-context pass by default) and per-token decode (`generate_answer`), plus the nested install of the doors: `positional_method` (outer) → `attention_method` → `kv_compressor` (inner).
- [eval_harness/kv_compression/cache_adapter.py](eval_harness/kv_compression/cache_adapter.py) — `DynamicCache` semantics (rotated K/V) and the length-based multi-question checkpoint/restore.

Tests that must keep passing:

- [eval_harness/tests/test_research_adapter.py](eval_harness/tests/test_research_adapter.py)
- [eval_harness/tests/test_prefill_methods.py](eval_harness/tests/test_prefill_methods.py)
- [eval_harness/tests/test_cache_adapter.py](eval_harness/tests/test_cache_adapter.py)
- [eval_harness/tests/test_positional_methods.py](eval_harness/tests/test_positional_methods.py), [test_chunked_prefill.py](eval_harness/tests/test_chunked_prefill.py), [test_three_doors_integration.py](eval_harness/tests/test_three_doors_integration.py)

If you find yourself working here, document *why* in your branch — Layer 3 changes affect the meaning of every other experiment run on the research backend.

---

## Research backend architecture

You only need this section if you're working at Layer 0, 2, or 3. `ResearchAdapter` ([eval_harness/research_adapter.py](eval_harness/research_adapter.py)) is a thin `HFAdapter` subclass. It does **not** install an identity-RoPE swap or an attention-function override; it builds the **three doors** — `positional_method` (Door 1), `attention_method` (Door 2), and `kv_compressor` (Door 3) — from `ResearchConfig` and runs everything through `SketchTextGenerationPipeline` ([eval_harness/research_pipeline.py](eval_harness/research_pipeline.py)).

#### Prefill pass (single, or chunked/streaming)

By default prefill is **one** full-context forward through the model's *normal* path: `pipeline._forward` calls `self.model.model(input_ids=context_ids, past_key_values=cache)`. The model's own layers apply RoPE, so HF's `DynamicCache` accumulates **RoPE-rotated** K/V (not raw). Methods that need position-agnostic keys recover them themselves (ReAttention un-rotates cached K on the fly; DCA replaces the attention forward and re-rotates at cyclic positions). Setting `prefill_chunk_size` to an integer instead loops the context in chunks with absolute per-chunk `position_ids` (`prefill_chunk_size: null` = single pass, byte-identical to the old behavior); `streaming`-scheduled compressors evict after each chunk so the cache stays memory-bounded.

Decode runs per-token in `pipeline.generate_answer` via `self.model(...)`.

#### Door 1 — positional method (RoPE frequency / position)

A *positional method* ([eval_harness/positional_methods/](eval_harness/positional_methods/)) changes **how token positions are stamped**. It wraps the model's shared `rotary_emb` (outermost context manager) so its forward emits modified `(cos, sin)`. Override `compute_inv_freq` (frequency scaling — NTK/YaRN) and/or `remap_position_ids` (position remap — Linear-PI); set `mscale` for YaRN's logit temperature. The base class is the identity transform. Shipped: `yarn`, `ntk`, `linear_pi`, `none` ([base.py](eval_harness/positional_methods/base.py), [yarn.py](eval_harness/positional_methods/yarn.py), [ntk.py](eval_harness/positional_methods/ntk.py), [linear_pi.py](eval_harness/positional_methods/linear_pi.py)). An attention method that computes its own RoPE (DCA) bypasses `rotary_emb` and overrides this door for its layers.

#### Door 2 — attention method (context extrapolation)

An *attention method* is how the research backend changes the attention math / extends context beyond the model's trained window. Methods live in [eval_harness/attention_methods/](eval_harness/attention_methods/):

- [base.py](eval_harness/attention_methods/base.py) — `AttentionMethod` base class; author writes one `attention_forward`, gated by `phase` ∈ {prefill, decode, both}.
- [dca.py](eval_harness/attention_methods/dca.py) — DCA (Dual Chunk Attention), a native `AttentionMethod` (`phase=both`).
- [_method_base.py](eval_harness/attention_methods/_method_base.py) — `PrefillMethod` base + RoPE helpers; the two legacy override seams are `prefill_forward_hook` (post-attention prune) and `__call__` (replace `self_attn.forward`).
- [reattention.py](eval_harness/attention_methods/reattention.py), [reattention_exact.py](eval_harness/attention_methods/reattention_exact.py) — the faithful ReAttention methods, kept as legacy `PrefillMethod` subclasses on the same door slot (reachable via `attention_method`).

The pipeline installs the door as a context manager that stays open across **both** prefill and decode, nested inside `positional_method` and outside `kv_compressor` — so forward hooks fire **method-then-compressor**. The two mechanisms:

1. **ReAttention — post-attention prune hook** (`PrefillMethod.prefill_forward_hook`). The hook fires *after* each full-attention layer **during prefill only** (it no-ops on decode). ReAttention un-rotates the cached K to score raw Q·K, selects `[global | top-k middle | local]` tokens, and prunes the cache contents. No decode-time selection. Per-layer top-k naturally retains a different count per layer, but HF's normal decode shares one causal mask/position grid across layers — so `uniform_retained` (default on) equalizes every layer to the same retained length (`uniform_budget` if set, else the first layer's selection size: shorter layers are padded with the most-recent unselected middle tokens, longer layers shrunk by the reference's frequency-clip rule). This is a Prism integration adaptation — the original ReAttention replaces each layer's attention forward and tolerates the ragged cache; `uniform_retained=False` restores per-layer selection but only decodes safely on single-layer models.

2. **DCA — full `self_attn.forward` replacement** (monkeypatch). For methods that must change *how* attention positions tokens. DCA stays active across **both prefill and decode**: it stores keys rotated at cyclic position `pos % chunk_len` and runs the intra/successive/inter decomposition merged by online softmax (decode recomputes cyclic query positions per step).

3. **ReAttention-exact — full `self_attn.forward` replacement** ([reattention_exact.py](eval_harness/attention_methods/reattention_exact.py), registered `reattention_exact`). Reproduces the *original* ReAttention computation, where mechanism 1's `reattention` only reproduces its retention policy: the `DynamicCache` stores **raw (pre-RoPE) K/V for the whole context and is never pruned**; prefill runs in `prefill_chunk_size` query chunks inside the replaced forward; each chunk (and, with the default `recall_option: whole`, each decode step) recalls a `[global | top-k middle | local]` view **before** attention, so the attention scope stays bounded during prefill itself; RoPE is applied after selection at original absolute positions (`pe_original=True`). Replicates the reference's unconditional 128-alignment quirk, the wrapper's chunk schedule (`chunk_schedule: reference` — engineered first chunk, last token as a `qlen==1` generate step; applied per forward call since this pipeline splits context/question), and the kernel gate (`einsum_topk` for multi-token chunks with `mid_size ∈ {1,4}`); `recall_option: full_attn` reduces it to the no-method baseline exactly (tested). Trade-off vs mechanism 1: real prefill-scope sparsity and decode-time re-selection, but **no KV-memory savings** (full cache retained, like the reference).

> Frequency/position methods (NTK / YaRN / Linear-PI) live in **Door 1** ([eval_harness/positional_methods/](eval_harness/positional_methods/)), which wraps the shared `rotary_emb` and invokes `compute_inv_freq` / `remap_position_ids` on every rotary call — see the Door 1 subsection above. (The legacy `PrefillMethod.compute_inv_freq` stub on the Door 2 base is unused.)

#### Multi-question checkpoint/restore

The runner feeds all questions for one context together. After the shared prefill, [eval_harness/kv_compression/cache_adapter.py](eval_harness/kv_compression/cache_adapter.py) records each layer's cache **sequence length** (`clone_or_checkpoint_for_multi_question`) and, after each question's decode, truncates `layer.keys` / `layer.values` back to that recorded length (`restore_after_question`) so the next question starts from the clean post-prefill cache.

#### Wiring config

`runner._setup_adapter` pops the `research_config` dict out of `EvalConfig.llm_kwargs`, converts it to `ResearchConfig`, and passes it to `ResearchAdapter`. The three independent, optional doors (each `none` = off):

- **Door 1** — `positional_method` (str, e.g. `"yarn"`, `"ntk"`, `"linear_pi"`, `"none"`) + `positional_method_kwargs` (dict). Resolved via `positional_methods.get_positional_method`.
- **Door 2** — `attention_method` (str, e.g. `"dca"`, `"reattention_exact"`, `"reattention"`, `"none"`) + `attention_method_kwargs` (dict) + `attention_phase` ∈ {prefill, decode, both}. `_build_attention_method` resolves the native `attention_methods` registry first (DCA), then falls back to the legacy `prefill_methods` registry (reattention / reattention_exact). ReAttention's `recall_type` defaults to `'qk'` (options: `qk` | `qkv` | `qkv2`) — a method kwarg, **not** an adapter-level selection mode.
- **Door 3** — `kv_compressor` (str, resolved through the `@register_kv_compressor` registry) + `kv_compressor_kwargs` (forwarded verbatim to the constructor) + `compression_schedule` ∈ {streaming, post_prefill, decode} + `compression_ratio` (injected as a default only when the compressor class declares that dataclass field).

`prefill_chunk_size` (`null` = single pass; int = streaming chunks) drives the chunked prefill the `streaming` schedule hooks into.

---

## Add a new benchmark

1. Drop a file into [eval_harness/benchmarks/](eval_harness/benchmarks/) (e.g. `my_bench.py`).
2. Subclass [`Benchmark`](eval_harness/benchmarks/base.py) and implement `info`, `load`, and `score`.
3. Decorate with `@register_benchmark`:

   ```python
   from .base import Benchmark, BenchmarkInfo
   from .registry import register_benchmark

   @register_benchmark("my_bench", aliases=["mb"])
   class MyBench(Benchmark):
       @property
       def info(self) -> BenchmarkInfo:
           return BenchmarkInfo(
               name="my_bench",
               description="...",
               default_subsets=["task_a"],
           )

       def load(self, subsets=None):
           # return a pd.DataFrame with required columns:
           #   context, question, answer_prefix (optional), max_new_tokens (optional),
           #   plus whatever ground-truth columns score() needs.
           ...

       def score(self, df):
           # return {"overall_score": ..., "task_scores": {...}, "total_samples": N}
           ...
   ```

The benchmark is auto-discovered by `ensure_benchmarks_loaded()` — no manual import. [`mock_benchmark.py`](eval_harness/benchmarks/mock_benchmark.py) is the simplest end-to-end example; [`ruler.py`](eval_harness/benchmarks/ruler.py) shows the HuggingFace-backed pattern.

Constraints to honor:

- `predictions.csv` will drop the `context` column at write-time — anything `score()` needs must be a *separate* column.
- The runner does `df.groupby("context", sort=False)` and feeds one context's questions at a time. The research backend additionally requires **one `answer_prefix` per context group** — fix your loader if you trip this.

---

## RAG backend setup

`OnePassRAG` indexes each document once via LanceDB + `BAAI/llm-embedder`, then answers all questions against the index using `llama3.1` served through Ollama. Matches the single-pass pattern used in the Prism benchmark paper.

#### 1. Install Ollama and pull the model (one-time per machine)

```bash
mkdir -p ~/.ollama
curl -fsSL https://ollama.com/download/ollama-linux-amd64.tar.zst | tar --zstd -x -C ~/.ollama
~/.ollama/bin/ollama pull llama3.1
```

#### 2. Start the Ollama server before each run

```bash
CUDA_VISIBLE_DEVICES=0 nohup ~/.ollama/bin/ollama serve > ollama.log 2>&1 &
```

> **GPU wake lock (optional):** if you hit Ollama timeouts due to GPU sleep between long eval runs, keep the GPU awake:
> ```bash
> nohup python -c "import torch, time; torch.zeros(1).cuda(); time.sleep(86400)" > wake_lock.log 2>&1 &
> ```

#### 3. Configure

```yaml
benchmark: longbench
subsets: qasper
backend: rag
```

Then run normally with `python -m eval_harness.cli run --config_file <your_config>.yaml` — none of the shipped [evaluate/](evaluate/) configs uses `backend: rag`; copy `evaluate/evaluate_vllm.yaml` and set `backend: rag` plus the YAML keys above.

#### 4. Tear down when done

```bash
pkill ollama
pkill -f wake_lock   # only if you started the wake lock
```

---

## Conventions & invariants

- **Determinism.** `seed` (default `42`) is applied to `random`, `numpy`, `torch`, and CUDA. Greedy decoding (`temperature=0.0`) is the default.
- **Per-context grouping.** The runner does `df.groupby("context", sort=False)` and the adapter receives all questions for a single context at once. Prefix caching (vLLM) or shared prefill (HF / research) makes this dramatically faster than per-row prompting.
- **Absolute position IDs by default.** Position IDs are the token's index in the *full* sequence, unless a prefill method deliberately re-rotates keys at cyclic positions (DCA stores keys at `pos % chunk_len`). New methods that introduce their own positioning scheme must document it.
- **No model loading in tests.** Use `object.__new__(Adapter)` plus fake `nn.Module` doubles. Never download weights from a test. See [eval_harness/tests/](eval_harness/tests/) for the pattern.
- **Config persistence.** The fully-resolved `EvalConfig` is written to `config.yaml` next to every results directory.
- **Run directories never overwrite.** Collisions append `/1`, `/2`, ... suffixes.

---

## Troubleshooting

| Symptom                                                              | Likely cause / fix                                                                                       |
| -------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `RuntimeError: CUDA out of memory` during prefill                    | Reduce `max_model_len`, set `gpu_memory_utilization: 0.85`, or switch to `dtype: bfloat16`.              |
| `Unknown benchmark '...'`                                            | Check `available_benchmarks()`; the benchmark module may have a syntax error preventing registry import. |
| `Inconsistent answer_prefix values within the same context group`    | The research backend requires one `answer_prefix` per context group — fix the benchmark loader.          |
| vLLM ignores `attn_implementation`                                   | vLLM has its own attention; `attn_implementation` only applies to `hf` and `research` backends.          |
| RAG: connection refused to Ollama                                    | Start the server first: `~/.ollama/bin/ollama serve`. Verify with `curl http://localhost:11434/api/tags`. |
| RAG: requests time out after long idle                               | GPU may have entered low-power state — start the wake-lock command shown above.                          |
| Research backend gives different numbers vs HF on the same prompt    | Confirm `attn_implementation: sdpa` — it is the path the prefill-method/sketch baselines were validated against; FA2 introduces numeric drift vs the SDPA-validated reference. |
| `observed_attention` asserts / `attentions` is `None`                | It needs eager attention probabilities: set `llm_kwargs: {attn_implementation: eager}` and `attention_method: none`. |
| Masking sketch (`adakv`/`dms`/`duo_attention`/`kvzip`/`fastkvzip`/`critical_adakv`) shows full-length cache / no memory savings | Expected — compression is virtual via `masked_key_indices` + the attention patch; these are quality-only baselines. If quality also looks like no compression at all, check you're not on eager attention or a `self_attn.forward`-replacing prefill method (`dca`, `reattention_exact`), both of which bypass the mask. |
| `ValueError` from `post_init_from_model` about flash attention       | `pyramidkv` / `simlayerkv` / `per_layer_compression` produce a cross-layer ragged cache that only decodes safely under `flash_attention_2`; switch implementation or use the sketch's uniform fallback. |
| Hub download fails on a compute node (qfilter/kvzap/duo_attention/expected_attention_stats/fastkvzip) | These sketches fetch external assets; pre-fetch and use the injection hook (`q_filters`, `model_name_override`, `attention_pattern`/`pattern_dir`, `stats_folder`, `gates`). |
| Tests download model weights unexpectedly                            | A test forgot `object.__new__` — file a bug.                                                             |
