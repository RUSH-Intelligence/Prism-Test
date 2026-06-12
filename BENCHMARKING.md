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
surface, including the prefill_method × sketch compatibility matrix). Run with:

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
| `llm_kwargs.cache_config`  | Sketch name, compression ratio, target size, and `prefill_method` (+ its kwargs). Pin all of these when comparing runs.  |

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

1. Subclass [`BaseSketch`](eval_harness/sketch/sketches/base_sketch.py) and implement `compress`:

   ```python
   from eval_harness.sketch.sketches.base_sketch import BaseSketch

   class MySketch(BaseSketch):
       def compress(self, module, hidden_states, keys, values, attentions, kwargs):
           # return (keys_pruned, values_pruned) with the same trailing shape
           ...
   ```

2. Drop the file into [eval_harness/sketch/sketches/](eval_harness/sketch/sketches/), export it from the sketch `__init__` modules, and add a name→class branch in `ResearchAdapter._build_sketch` ([eval_harness/research_adapter.py](eval_harness/research_adapter.py)).

3. Select it in YAML:

   ```yaml
   llm_kwargs:
     cache_config:
       sketch_name: my_sketch
       compression_ratio: 0.5
       target_size: 2048
       max_context_length: 65536
   ```

Reference implementations to read first: [knorm_sketch.py](eval_harness/sketch/sketches/knorm_sketch.py), [reattention_sketch.py](eval_harness/sketch/sketches/reattention_sketch.py) (scoring base class in [scorer_sketch.py](eval_harness/sketch/sketches/scorer_sketch.py)), [random_sketch.py](eval_harness/sketch/sketches/random_sketch.py), [decoding_sketch.py](eval_harness/sketch/sketches/decoding_sketch.py).

Shipped sketches:

| Sketch                   | Trigger             | What it does                                                  |
| ------------------------ | ------------------- | ------------------------------------------------------------- |
| `none`                   | —                   | Pass-through; full KV cache.                                  |
| `knorm`                  | Prefill             | Keep tokens with smallest ‖K‖ (largest-norm keys evicted).    |
| `reattention`            | Prefill             | Re-score with QK relevance over middle tokens.                |
| `random`                 | Prefill             | Random baseline.                                              |
| `decoding_knorm`         | Decode (periodic)   | Compress mid-decode every `compression_interval` steps.       |
| `prefill_decoding_knorm` | Prefill + decode    | Compress at prefill end and during decode.                    |

### Layer 2 — a custom attention kernel

Use this when you need a different *kernel* (Triton, a FlashAttention variant, custom CUDA) underneath a prefill method's scoring or attention path.

There is no `_prefill_attn_impl` / `_decode_attn_impl` seam on the adapter. Kernels are called directly from the prefill method that owns them. The two shipped seams to read and model your own on:

- [`einsum_topk_func`](eval_harness/kernels/einsum_topk.py) — ReAttention's Triton kernel for fused QK scoring + top-k token selection over the cached keys.
- [`flash_attn_with_lse`](eval_harness/kernels/dca_flash.py) — DCA's FlashAttention variant returning the log-sum-exp, used to merge the intra/successive/inter attention components with online softmax.

A new kernel lives in [eval_harness/kernels/](eval_harness/kernels/) and is invoked from your prefill method's hook (`prefill_forward_hook`) or `self_attn.forward` replacement (see [Layer 3](#layer-3--modify-the-research-adapter-itself) and [Research backend architecture](#research-backend-architecture)).

Invariant you must respect:

- Position IDs are **absolute** (token's index in the full sequence), never chunk-relative — except where a method deliberately re-rotates keys at cyclic positions (DCA stores keys at `pos % chunk_len`).

### Layer 3 — modify the research adapter itself

Use this when none of the above fits — e.g., you want a new context-extension mechanism, or to change how the pipeline installs the prefill method and sketch.

Files you'll touch:

- [eval_harness/research_adapter.py](eval_harness/research_adapter.py) — the thin `HFAdapter` subclass that builds the `sketch` and `prefill_method` from `CacheConfig` and runs them through the pipeline.
- [eval_harness/sketch/pipeline.py](eval_harness/sketch/pipeline.py) — `SketchTextGenerationPipeline`; single full-context prefill (`_forward`) and per-token decode (`generate_answer`), and the nested install of `prefill_method` (outer) and `sketch` (inner).
- [eval_harness/sketch/cache_adapter.py](eval_harness/sketch/cache_adapter.py) — `DynamicCache` semantics (rotated K/V) and the length-based multi-question checkpoint/restore.

Tests that must keep passing:

- [eval_harness/tests/test_research_adapter.py](eval_harness/tests/test_research_adapter.py)
- [eval_harness/tests/test_prefill_methods.py](eval_harness/tests/test_prefill_methods.py)
- [eval_harness/tests/test_cache_adapter.py](eval_harness/tests/test_cache_adapter.py)

If you find yourself working here, document *why* in your branch — Layer 3 changes affect the meaning of every other experiment run on the research backend.

---

## Research backend architecture

You only need this section if you're working at Layer 0, 2, or 3. `ResearchAdapter` ([eval_harness/research_adapter.py](eval_harness/research_adapter.py)) is a thin `HFAdapter` subclass. It does **not** install an identity-RoPE swap or an attention-function override; it builds a `sketch` (KV compression) and a `prefill_method` (context extension) from `CacheConfig` and runs everything through `SketchTextGenerationPipeline` ([eval_harness/sketch/pipeline.py](eval_harness/sketch/pipeline.py)).

#### Single full-context prefill pass

Prefill is **one** full-context forward through the model's *normal* path: `pipeline._forward` calls `self.model.model(input_ids=context_ids, past_key_values=cache)`. The model's own layers apply RoPE, so HF's `DynamicCache` accumulates **RoPE-rotated** K/V (not raw). There is no chunk loop and no `cache_config.chunk_size`. Methods that need position-agnostic keys recover them themselves (ReAttention un-rotates cached K on the fly; DCA replaces the attention forward and re-rotates at cyclic positions).

Decode runs per-token in `pipeline.generate_answer` via `self.model(...)`.

#### Layer 0 — prefill method (context extrapolation)

A *prefill method* is how the research backend extends context beyond the model's trained window. Methods live in [eval_harness/prefill_methods/](eval_harness/prefill_methods/):

- [base.py](eval_harness/prefill_methods/base.py) — `PrefillMethod` base class; the two override seams are `prefill_forward_hook` (post-attention prune) and `__call__` (replace `self_attn.forward`).
- [reattention.py](eval_harness/prefill_methods/reattention.py) — ReAttention.
- [dca.py](eval_harness/prefill_methods/dca.py) — DCA (Dual Chunk Attention).

The pipeline installs them as **nested context managers** — `prefill_method(model)` is the outer manager and `sketch(model)` is the inner one — so forward hooks fire **method-then-sketch**. The two mechanisms:

1. **ReAttention — post-attention prune hook** (`PrefillMethod.prefill_forward_hook`). The hook fires *after* each full-attention layer **during prefill only** (it no-ops on decode). ReAttention un-rotates the cached K to score raw Q·K, selects `[global | top-k middle | local]` tokens, and prunes the cache contents. No decode-time selection. Per-layer top-k naturally retains a different count per layer, but HF's normal decode shares one causal mask/position grid across layers — so `uniform_retained` (default on) equalizes every layer to the same retained length (`uniform_budget` if set, else the first layer's selection size: shorter layers are padded with the most-recent unselected middle tokens, longer layers shrunk by the reference's frequency-clip rule). This is a Prism integration adaptation — the original ReAttention replaces each layer's attention forward and tolerates the ragged cache; `uniform_retained=False` restores per-layer selection but only decodes safely on single-layer models.

2. **DCA — full `self_attn.forward` replacement** (monkeypatch). For methods that must change *how* attention positions tokens. DCA stays active across **both prefill and decode**: it stores keys rotated at cyclic position `pos % chunk_len` and runs the intra/successive/inter decomposition merged by online softmax (decode recomputes cyclic query positions per step).

3. **ReAttention-exact — full `self_attn.forward` replacement** ([reattention_exact.py](eval_harness/prefill_methods/reattention_exact.py), registered `reattention_exact`). Reproduces the *original* ReAttention computation, where mechanism 1's `reattention` only reproduces its retention policy: the `DynamicCache` stores **raw (pre-RoPE) K/V for the whole context and is never pruned**; prefill runs in `prefill_chunk_size` query chunks inside the replaced forward; each chunk (and, with the default `recall_option: whole`, each decode step) recalls a `[global | top-k middle | local]` view **before** attention, so the attention scope stays bounded during prefill itself; RoPE is applied after selection at original absolute positions (`pe_original=True`). Replicates the reference's unconditional 128-alignment quirk, the wrapper's chunk schedule (`chunk_schedule: reference` — engineered first chunk, last token as a `qlen==1` generate step; applied per forward call since this pipeline splits context/question), and the kernel gate (`einsum_topk` for multi-token chunks with `mid_size ∈ {1,4}`); `recall_option: full_attn` reduces it to the no-method baseline exactly (tested). Trade-off vs mechanism 1: real prefill-scope sparsity and decode-time re-selection, but **no KV-memory savings** (full cache retained, like the reference).

> ⚠️ Tier-1 "frequency-only" methods (NTK / YaRN / Linear-PI) are **not yet functional**: `PrefillMethod.compute_inv_freq` exists but **nothing calls it** — the pipeline only invokes `prefill_forward_hook`, `compute_question_position_ids`, and `on_prefill_start` / `on_prefill_end`. Implementing them needs a RoPE-level interceptor the framework currently lacks.

#### Multi-question checkpoint/restore

The runner feeds all questions for one context together. After the shared prefill, [eval_harness/sketch/cache_adapter.py](eval_harness/sketch/cache_adapter.py) records each layer's cache **sequence length** (`clone_or_checkpoint_for_multi_question`) and, after each question's decode, truncates `layer.keys` / `layer.values` back to that recorded length (`restore_after_question`) so the next question starts from the clean post-prefill cache.

#### Wiring config

`runner._setup_adapter` pulls the `cache_config` dict out of `EvalConfig.llm_kwargs`, converts it to `CacheConfig`, and passes it to `ResearchAdapter`. Relevant fields: `prefill_method` (str, e.g. `"dca"`, `"reattention"`, `"none"`) + `prefill_method_kwargs` (dict), plus the sketch fields (`sketch_name`, `compression_ratio`, …). `ResearchAdapter._build_prefill_method` resolves the name via `prefill_methods.get_prefill_method`. ReAttention's `recall_type` defaults to `'qk'` (options: `qk` | `qkv` | `qkv2`) — this is a method kwarg, **not** an adapter-level selection mode.

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
| Tests download model weights unexpectedly                            | A test forgot `object.__new__` — file a bug.                                                             |
