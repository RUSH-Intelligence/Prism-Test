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
| `research`  | Sparse attention, KV sketches, custom kernels.        | Raw pre-RoPE Q/K at the attention hook; chunked prefill.  | Requires `attn_implementation: sdpa`.                  |
| `rag`       | Retrieval baselines.                                  | OnePassRAG (LanceDB + llm-embedder + Ollama llama3.1).    | Different architecture — not apples-to-apples.         |

Rules of thumb:

- **"I want a baseline number on RULER-64K"** → `vllm`.
- **"I'm comparing my sketch against `knorm`"** → `research`.
- **"I'm comparing attention vs. retrieval"** → run both `research` (or `vllm`) and `rag` on the same benchmark.
- **"My research-backend numbers don't match HF reference"** → re-run with `hf` to isolate whether it's a vLLM kernel difference or your sketch.

---

## Configure a run

The full config lives in [evaluate_config.yaml](evaluate_config.yaml). Run with:

```bash
python -m eval_harness.cli run --config_file ./evaluate_config.yaml
```

Or override any field on the CLI:

```bash
python -m eval_harness.cli run \
  --config_file ./evaluate_config.yaml \
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
| `attn_implementation`      | `sdpa` for `research` (required for `ALL_ATTENTION_FUNCTIONS` dispatch); `flash_attention_2` for `hf`. vLLM ignores it. |
| `max_model_len`            | Override the model's positional cap when running past its training window.                      |
| `query_aware`              | If `True`, the question is concatenated into the context — needed for some query-aware methods. Note it in your write-up. |
| `max_requests` / `max_requests_per_subset` | Subsampling caps. Same value across compared runs or numbers don't line up.    |
| `llm_kwargs.cache_config`  | Sketch name, compression ratio, target size, chunk size. Pin all four when comparing sketches.  |

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

2. Drop the file into [eval_harness/sketch/sketches/](eval_harness/sketch/sketches/) and wire it into the sketch selector in [eval_harness/sketch/](eval_harness/sketch/).

3. Select it in YAML:

   ```yaml
   llm_kwargs:
     cache_config:
       sketch_name: my_sketch
       compression_ratio: 0.5
       target_size: 2048
       max_context_length: 65536
   ```

Reference implementations to read first: [knorm_sketch.py](eval_harness/sketch/sketches/knorm_sketch.py), [scorer_sketch.py](eval_harness/sketch/sketches/scorer_sketch.py) (reattention), [random_sketch.py](eval_harness/sketch/sketches/random_sketch.py), [decoding_sketch.py](eval_harness/sketch/sketches/decoding_sketch.py).

Shipped sketches:

| Sketch                   | Trigger             | What it does                                                  |
| ------------------------ | ------------------- | ------------------------------------------------------------- |
| `none`                   | —                   | Pass-through; full KV cache.                                  |
| `knorm`                  | Prefill             | Keep tokens with largest ‖K‖.                                 |
| `reattention`            | Prefill             | Re-score with QK relevance over middle tokens.                |
| `random`                 | Prefill             | Random baseline.                                              |
| `decoding_knorm`         | Decode (periodic)   | Compress mid-decode every `compression_interval` steps.       |
| `prefill_decoding_knorm` | Prefill + decode    | Compress at prefill end and during decode.                    |

### Layer 2 — a custom attention kernel

Use this when you need a different *kernel* (Triton, a FlashAttention variant, custom CUDA) but the same selection + RoPE pipeline that the research backend provides.

Subclass [`ResearchAdapter`](eval_harness/research_adapter.py) and override one or both of:

- `_prefill_attn_impl(query, key, value, attention_mask, ...)`
- `_decode_attn_impl(query, key, value, ...)`

**Do not** override `prefill_attention` / `decode_attention` themselves. Those are the *integration boundary* — they handle selection and RoPE; your algorithm lives below them.

Invariants you must respect (see [Research backend architecture](#research-backend-architecture) for details):

- Position IDs are **absolute** (token's index in the full sequence), never chunk-relative.
- Raw K/V is accumulated in HF's `DynamicCache` — no separate cache storage needed.
- `T_prev = keys.shape[-2] - queries.shape[-2]` is the absolute start of the current chunk.

### Layer 3 — modify the research adapter itself

Use this when none of the above fits — e.g., you want a different selection strategy, a different chunking scheme, or to replace the identity-RoPE intercept.

Files you'll touch:

- [eval_harness/research_adapter.py](eval_harness/research_adapter.py) — adapter, identity-RoPE interceptor, attention hooks, sparse selector wiring.
- [eval_harness/sketch/attention_patch.py](eval_harness/sketch/attention_patch.py) — `ALL_ATTENTION_FUNCTIONS` patching.
- [eval_harness/sketch/cache_adapter.py](eval_harness/sketch/cache_adapter.py) — `DynamicCache` semantics with raw K/V.

Tests that must keep passing:

- [eval_harness/tests/test_research_adapter.py](eval_harness/tests/test_research_adapter.py)
- [eval_harness/tests/test_identity_rope_equivalence.py](eval_harness/tests/test_identity_rope_equivalence.py)
- [eval_harness/tests/test_cache_adapter.py](eval_harness/tests/test_cache_adapter.py)

If you find yourself working here, document *why* in your branch — Layer 3 changes affect the meaning of every other experiment run on the research backend.

---

## Research backend architecture

You only need this section if you're working at Layer 2 or 3. The research backend exists because attention-compression research needs two things stock HF doesn't give you:

1. **Raw, pre-RoPE Q/K at the attention boundary** — so the kernel can choose tokens, then apply RoPE with original absolute positions.
2. **A `DynamicCache` that stores unrotated K/V** — so re-selection on later chunks doesn't have to un-rotate stale entries.

Two cooperating intercepts make this work:

#### 1. Identity-RoPE interceptor

`ResearchRotaryEmbedding` replaces every `rotary_emb` submodule (detected by an `inv_freq` buffer) with a module whose `forward()` returns `(cos=1, sin=0)`. The model's own `apply_rotary_pos_emb` then becomes a no-op, so Q/K arrive at the attention hook **unrotated**. The interceptor also stashes the `position_ids` it was called with, so the hook can recover absolute positions.

#### 2. Attention hook

`prefill_attention` and `decode_attention` are registered into `transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS` via `_with_attention`. Inside the hook:

1. Apply real RoPE manually (`rope.compute(...)`) using the stashed absolute positions.
2. Run `_prefill_attn_impl` or `_decode_attn_impl` — these are the override points for custom kernels.

Because identity RoPE is a no-op, HF's `DynamicCache` accumulates **raw** K/V — there's no separate cache storage to manage.

#### Chunked sparse prefill

`ResearchAdapter._prefill` loops over `cache_config.chunk_size`-sized chunks, calling the model with `past_key_values` and explicit absolute `position_ids` each iteration. Inside the hook:

- **First chunk / small context / `selection='full'`:** dense path. Apply RoPE to full Q/K, run dense causal SDPA.
- **Subsequent chunks:** sparse path:
  - `SparseSelector.select()` picks **global-sink + top-k-middle + local-window** tokens from history; the current chunk's K/V is kept verbatim.
  - Apply RoPE to Q (chunk positions), selected hist-K (original positions), and curr-K (chunk positions) — each with its true position ID.
  - Combine as `[selected_history | current_chunk]`.
  - Mask is zeros for history columns and upper-triangular `-inf` for current-chunk columns.

#### Sparse decode

`decode_attention` calls `SparseSelector.select` on the full raw K/V cache, applies RoPE with each token's **original absolute position**, then runs `_decode_attn_impl`.

#### Wiring config

`runner._setup_adapter` pulls the `cache_config` dict out of `EvalConfig.llm_kwargs`, converts it to `CacheConfig`, and passes it to `ResearchAdapter`. Selection modes (where applicable): `'qkv2'` (default, QK·‖V‖₂), `'qk'`, `'full'`.

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

Then run normally with `python -m eval_harness.cli run --config_file ./evaluate_config.yaml`.

#### 4. Tear down when done

```bash
pkill ollama
pkill -f wake_lock   # only if you started the wake lock
```

---

## Conventions & invariants

- **Determinism.** `seed` (default `42`) is applied to `random`, `numpy`, `torch`, and CUDA. Greedy decoding (`temperature=0.0`) is the default.
- **Per-context grouping.** The runner does `df.groupby("context", sort=False)` and the adapter receives all questions for a single context at once. Prefix caching (vLLM) or shared prefill (HF / research) makes this dramatically faster than per-row prompting.
- **Absolute position IDs everywhere.** Every position ID in the codebase is the token's index in the *full* sequence, never the index inside a chunk. New kernels must respect this or sparse attention silently breaks.
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
| Research backend gives different numbers vs HF on the same prompt    | Confirm `attn_implementation: sdpa` is set; SDPA is the parity path. FA2 won't exercise the hook.        |
| Tests download model weights unexpectedly                            | A test forgot `object.__new__` — file a bug.                                                             |
