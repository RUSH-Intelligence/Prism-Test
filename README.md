# Prism-Test

> A long-context inference evaluation framework for research on extending transformer context windows beyond their original training limits.

Prism-Test is a unified, reproducible harness for benchmarking large language models on **long-context** tasks — including contexts that vastly exceed a model's native training window. It standardizes evaluation across popular long-context suites (RULER, LOFT, LongBench, InfiniteBench, GSM-Infinite, AIME, Loogle, and the Prism-1M dataset), while exposing first-class extension points for **inference-time context compression** research: sparse attention, KV-cache eviction sketches, hybrid sparse/dense prefill kernels, and RAG-style retrieval.

It supports four interchangeable inference backends — `vllm`, `hf`, `research`, and `rag` — and ships with detailed quality + systems metrics (accuracy, retrieval recall, latency, throughput, memory, KV-cache size, prefill/decode efficiency).

---

## Table of contents

- [Why Prism-Test](#why-prism-test)
- [Key features](#key-features)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Configuration](#configuration)
- [Backends](#backends)
  - [vLLM backend](#vllm-backend)
  - [HuggingFace backend](#huggingface-backend)
  - [Research backend](#research-backend)
  - [RAG backend](#rag-backend)
- [Supported benchmarks](#supported-benchmarks)
- [Output format](#output-format)
- [The Research backend in depth](#the-research-backend-in-depth)
- [Sketch / KV-cache compression API](#sketch--kv-cache-compression-api)
- [Adding a new benchmark](#adding-a-new-benchmark)
- [Adding a new attention kernel](#adding-a-new-attention-kernel)
- [Testing](#testing)
- [Reproducibility & conventions](#reproducibility--conventions)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Citation](#citation)

---

## Why Prism-Test

Most evaluation harnesses are built around **short-context throughput**. Long-context research has different requirements:

1. **Inputs that exceed `max_position_embeddings`** — the model has never seen 128K, 250K, or 1M tokens during training.
2. **Per-context grouping** — many long-context benchmarks ask several questions about the same massive document; the harness must amortize the prefill across questions.
3. **Attention/KV-cache surgery** — research methods need raw, pre-RoPE Q/K, custom attention dispatch, and control over which tokens remain in the cache.
4. **Apples-to-apples comparison** — the same prompt, the same scorer, swappable backends.

Prism-Test gives you all four with a single config file.

---

## Key features

- **Four backends, one runner.** `vllm` for production throughput, `hf` for clean prefill/decode debugging, `research` for compression experiments, `rag` for retrieval baselines.
- **Per-context batching.** The runner groups by shared context so a 1M-token document is prefilled once and reused across all of its questions.
- **Custom attention via `ALL_ATTENTION_FUNCTIONS`.** The `research` backend swaps the model's RoPE with an identity-RoPE module, allowing the attention hook to receive raw Q/K with absolute position IDs intact — the foundation for sparse selection.
- **Pluggable sketches.** `knorm`, `reattention`, `random`, and decoding-time variants compress the KV cache during prefill or decode.
- **A standalone benchmark registry.** Drop a file into `eval_harness/benchmarks/`, decorate with `@register_benchmark`, and it's runnable from the CLI.
- **Deterministic runs.** Seeded RNG, `temperature=0.0` by default, configs persisted alongside outputs.
- **Tested without GPUs.** Unit tests bypass model loading via `object.__new__` + fake modules; CI is cheap.

---

## Repository layout

```
Prism-Test/
├── eval_harness/
│   ├── cli.py                  # argparse entry point
│   ├── config.py               # EvalConfig dataclass
│   ├── runner.py               # dataset → adapter → groupby(context) → score
│   ├── vllm_adapter.py         # vLLM backend
│   ├── hf_adapter.py           # HuggingFace backend (clean prefill/decode)
│   ├── research_adapter.py     # HF subclass: identity-RoPE + sparse prefill
│   ├── rag_adapter.py          # OnePassRAG backend wrapper
│   ├── rag/                    # LanceDB + llm-embedder + Ollama
│   ├── sketch/                 # KV-cache compression sketches
│   │   ├── attention_patch.py
│   │   ├── cache_adapter.py
│   │   ├── pipeline.py
│   │   └── sketches/           # knorm, reattention, random, ...
│   ├── benchmarks/             # one module per benchmark
│   │   ├── base.py
│   │   ├── registry.py         # get_benchmark(), @register_benchmark
│   │   ├── ruler*.py           # ruler / ruler16k / ruler32k / ...
│   │   ├── longbench*.py
│   │   ├── infinite_bench.py
│   │   ├── loft.py
│   │   ├── loogle.py
│   │   ├── prism1m.py
│   │   ├── aime*.py
│   │   ├── gsm_infinite_128k.py
│   │   ├── zero_scrolls.py
│   │   └── mock_benchmark.py
│   └── tests/                  # unittest — no model loading
├── datasets/Prism-Data/        # Prism-1M long-context QA jsonl shards
│   ├── 128K/{Easy,Medium}/
│   ├── 250K/{Easy,Medium}/
│   └── 1M/{Easy,Medium}/
├── results/                    # per-run output directories
├── evaluate_config.yaml        # default config
├── run_eval.py                 # thin wrapper over CliEntryPoint
├── pyproject.toml             # project metadata + loose dep constraints
├── uv.lock                    # uv-native pinned lock (reproducible installs)
├── requirements.txt           # pinned pip export (autogenerated from uv.lock)
├── EVAL_HARNESS.md             # user-facing setup notes (RAG, Ollama)
├── CLAUDE.md                   # internal AI assistant guide
├── CONTRIBUTING.md
└── README.md                   # you are here
```

---

## Installation

### Requirements

- Python 3.10+
- CUDA-capable GPU (for `vllm`, `hf`, `research` backends)
- ~24 GB+ VRAM recommended for 7-8B models at 64K context
- (Optional) Ollama for the `rag` backend

### Install

Clone the repo:

```bash
git clone https://github.com/<your-org>/Prism-Test.git
cd Prism-Test
```

**Option A — loose install (recommended when integrating into your own
research environment).** Uses the constraints declared in `pyproject.toml`,
so it co-exists with the torch / transformers / vllm versions you already
have:

```bash
pip install .
# or, for an editable install:
pip install -e .
```

**Option B — reproducible install (for matching official benchmark numbers
exactly).** Uses the pinned `uv.lock`:

```bash
# with uv (fastest):
uv sync

# or with pip:
pip install -r requirements.txt
```

Key packages and their supported ranges (from `pyproject.toml`):

| Package              | Constraint        | Purpose                            |
| -------------------- | ----------------- | ---------------------------------- |
| `vllm`               | `>=0.5.5,<1.0`    | Production inference backend       |
| `transformers`       | `>=4.45,<6`       | HF / research backends             |
| `torch`              | `>=2.1,<3`        | Tensor ops, CUDA                   |
| `datasets`           | `>=2.20`          | Benchmark loading                  |
| `pandas`, `numpy`    | latest            | Per-row dataframes, scoring        |
| `lancedb`            | latest            | RAG vector store                   |
| `llama-index-*`      | latest            | RAG embeddings + Ollama LLM client |
| `sentence-transformers` | latest         | Embedding models                   |
| `ninja`, `setuptools` | latest           | Build-from-source dependencies     |

Conda users: create a conda env for Python, then use either option above —
no separate `environment.yml` is required.

```bash
conda create -n prism python=3.11 && conda activate prism
pip install .   # or `pip install -r requirements.txt` for the pinned env
```

To regenerate the lock and pip export after editing dependencies in
`pyproject.toml`:

```bash
uv lock
uv export --format requirements-txt --no-hashes --no-emit-project -o requirements.txt
```

Optional but recommended:

```bash
pip install flash-attn --no-build-isolation
```

---

## Quickstart

### 1. Pick a model and a benchmark

Edit [`evaluate_config.yaml`](evaluate_config.yaml):

```yaml
benchmark: ruler16k
subsets: qa_1                          # comma-separated list, or null for defaults
backend: research                       # vllm | hf | rag | research
model: mistralai/Ministral-3-3B-Instruct-2512

# Runtime
tensor_parallel_size: 1
dtype: bfloat16
max_model_len: 65536
gpu_memory_utilization: 0.9
trust_remote_code: true
enable_prefix_caching: true

# Generation
max_new_tokens: 128
temperature: 0.0
top_p: 1.0
seed: 42

# Evaluation limits
max_requests: 200
max_requests_per_subset:
  qa_1: 200
fraction: 1.0
query_aware: false

output_dir: ./results

llm_kwargs:
  attn_implementation: flash_attention_2
  cache_config:
    sketch_name: none
    compression_ratio: 0
    max_context_length: 65536
    log_cache_seq_len: true
```

### 2. Run

```bash
python -m eval_harness.cli run --config_file ./evaluate_config.yaml
```

Or override any field on the CLI:

```bash
python -m eval_harness.cli run \
  --config_file ./evaluate_config.yaml \
  --benchmark longbench \
  --subsets narrativeqa,hotpotqa \
  --backend vllm \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --max_new_tokens 128 \
  --max_requests 50
```

### 3. Read results

```
results/<benchmark>__<model>__<backend>__t0__p1__subsets_qa_1/
├── predictions.csv     # per-row inputs + predicted_answer + ground truth
├── metrics.json        # aggregated benchmark scores
└── config.yaml         # the exact config that produced this run
```

If the output directory already exists, a numeric suffix (`/1`, `/2`, ...) is appended — runs are never silently overwritten.

---

## Configuration

All knobs live on [`EvalConfig`](eval_harness/config.py). The CLI loads YAML, overlays CLI flags, and constructs the dataclass.

| Field                       | Type                    | Default                              | Description                                                                  |
| --------------------------- | ----------------------- | ------------------------------------ | ---------------------------------------------------------------------------- |
| `benchmark`                 | `str`                   | `"ruler32k"`                         | Registered benchmark name (see `eval_harness/benchmarks/registry.py`)        |
| `subsets`                   | `Optional[str]`         | `None`                               | Comma-separated subset list (`"qa_1,qa_2"`), or `None` for defaults          |
| `backend`                   | `str`                   | `"vllm"`                             | One of `vllm`, `hf`, `rag`, `research`                                       |
| `model`                     | `str`                   | `"meta-llama/Llama-3.1-8B-Instruct"` | Any HF model ID                                                              |
| `tensor_parallel_size`      | `int`                   | `1`                                  | vLLM tensor-parallel degree                                                  |
| `dtype`                     | `str`                   | `"auto"`                             | `auto` / `bfloat16` / `float16` / `float32`                                  |
| `max_model_len`             | `Optional[int]`         | `None`                               | Override the model's positional cap                                          |
| `gpu_memory_utilization`    | `float`                 | `0.9`                                | vLLM-only                                                                    |
| `trust_remote_code`         | `bool`                  | `True`                               | Allow custom HF code (`Mistral-3`, etc.)                                     |
| `enable_prefix_caching`     | `bool`                  | `True`                               | vLLM-only; major speedup for per-context groups                              |
| `max_new_tokens`            | `Optional[int]`         | `None`                               | Override benchmark-provided per-row token budget                             |
| `temperature`               | `float`                 | `0.0`                                | Greedy by default                                                            |
| `top_p`                     | `float`                 | `1.0`                                |                                                                              |
| `system_prompt`             | `Optional[str]`         | `None`                               |                                                                              |
| `seed`                      | `int`                   | `42`                                 | Seeds `random`, `numpy`, `torch`, CUDA                                       |
| `fraction`                  | `float`                 | `1.0`                                | Random subsample of dataset rows (in `(0, 1]`)                               |
| `max_requests`              | `Optional[int]`         | `None`                               | Global cap                                                                   |
| `max_requests_per_subset`   | `Optional[Dict[str,int]]` | `{}`                                 | Per-subset cap (overrides `max_requests` for matching subsets)               |
| `query_aware`               | `bool`                  | `False`                              | If `True`, concatenates question into context (for query-aware compression)  |
| `output_dir`                | `str`                   | `"./results"`                        |                                                                              |
| `llm_kwargs`                | `Optional[Dict[str,Any]]` | `{}`                                 | Passthrough to backend; includes `cache_config` for the research backend     |

Validation is performed in `__post_init__` — invalid values raise `ValueError` before the model is loaded.

---

## Backends

### vLLM backend

**When to use:** production-quality throughput, large-batch evaluation, prefix caching across questions sharing a context.

```yaml
backend: vllm
tensor_parallel_size: 2
gpu_memory_utilization: 0.9
enable_prefix_caching: true
```

Prefix caching is the main reason this backend is preferred for benchmarks like RULER where 200 questions share one 64K context.

### HuggingFace backend

**When to use:** small-context debugging, profiling, or when you need a clean `_prefill` / `_decode` split that you can step through.

```yaml
backend: hf
llm_kwargs:
  attn_implementation: flash_attention_2     # native FA2 if installed
```

The HF backend installs **no hooks** into attention dispatch — what you see is what the model does.

### Research backend

**When to use:** sparse attention experiments, KV-cache compression sketches, custom prefill/decode kernels.

```yaml
backend: research
llm_kwargs:
  attn_implementation: sdpa    # forced; required for ALL_ATTENTION_FUNCTIONS dispatch
  cache_config:
    sketch_name: knorm         # none | knorm | reattention | random | decoding_knorm | ...
    compression_ratio: 0.5
    compression_interval: 512
    target_size: 2048
    max_context_length: 65536
    log_cache_seq_len: true
```

See [The Research backend in depth](#the-research-backend-in-depth) for the architecture.

### RAG backend

**When to use:** retrieval baselines, comparisons against non-attention long-context methods.

`OnePassRAG` indexes each document once via LanceDB + `BAAI/llm-embedder`, then answers all questions for that document against the index using `llama3.1` served through Ollama.

```yaml
backend: rag
benchmark: longbench
subsets: qasper
```

**Ollama setup (one-time, per machine):**

```bash
mkdir -p ~/.ollama
curl -fsSL https://ollama.com/download/ollama-linux-amd64.tar.zst | tar --zstd -x -C ~/.ollama
~/.ollama/bin/ollama pull llama3.1
```

**Start the server before each run:**

```bash
CUDA_VISIBLE_DEVICES=0 nohup ~/.ollama/bin/ollama serve > ollama.log 2>&1 &
```

> If you hit Ollama timeouts due to GPU sleep between long evals, keep the GPU awake:
> ```bash
> nohup python -c "import torch, time; torch.zeros(1).cuda(); time.sleep(86400)" > wake_lock.log 2>&1 &
> ```

**Tear down when done:**

```bash
pkill ollama
pkill -f wake_lock
```

See [EVAL_HARNESS.md](EVAL_HARNESS.md) for the full RAG walkthrough.

---

## Supported benchmarks

| Benchmark         | Tasks / subsets                              | Notes                                                              |
| ----------------- | -------------------------------------------- | ------------------------------------------------------------------ |
| `ruler`           | Configurable via subsets                     | Generic RULER loader                                               |
| `ruler16k`        | `qa_1`, `qa_2`, ...                          | 16K context                                                        |
| `ruler32k`        | same                                         | 32K context                                                        |
| `ruler64k`        | same                                         | 64K context                                                        |
| `ruler128k`       | same                                         | 128K context                                                       |
| `longbench`       | `narrativeqa`, `qasper`, `hotpotqa`, ...     | Standard LongBench English subsets                                 |
| `longbenchv2`     | LongBench v2 tasks                           |                                                                    |
| `infinite_bench`  | `passkey`, `kv_retrieval`, ...               | Extreme-length retrieval                                           |
| `loft`            | LOFT subtasks                                | Retrieval / reasoning                                              |
| `loft_rag`        | LOFT RAG variant                             |                                                                    |
| `loogle`          | Loogle subtasks                              |                                                                    |
| `gsm_infinite_128k` | GSM-Infinite at 128K                       | Math reasoning at long context                                     |
| `aime` / `aime2024` / `aime2025` | AIME problems                 | Math reasoning                                                     |
| `zero_scrolls`    | ZeroSCROLLS                                  |                                                                    |
| `prism1m`         | `128K`, `250K`, `1M` × `Easy`, `Medium`      | Ships with this repo under `datasets/Prism-Data/`                  |
| `mock_benchmark`  | Tiny synthetic                               | Smoke-test the harness without downloading data                    |

List the live registry at any time:

```python
from eval_harness.benchmarks.registry import available_benchmarks
print(available_benchmarks())
```

---

## Output format

Each run produces three files in its own directory.

### `predictions.csv`

One row per request. Columns vary by benchmark, but you'll always get:

- `question`
- `answer_prefix`
- `predicted_answer`
- Ground-truth columns (e.g. `answer`, `gold`, `references`) the scorer used
- `task` / `subset` (when applicable)

The raw `context` column is **omitted** from the CSV (it can be hundreds of thousands of tokens). It lives in the original dataset.

### `metrics.json`

Benchmark-specific aggregate scores — typically a dict of `{metric_name: float}` plus per-subset breakdowns.

### `config.yaml`

The exact, fully-resolved `EvalConfig` that produced this run — useful for reproducing results months later.

---

## The Research backend in depth

This is the non-obvious part of the codebase. It exists because attention-compression research needs two things HF doesn't give you out of the box:

1. **Raw, pre-RoPE Q/K at the attention boundary** — so the kernel can choose tokens, then apply RoPE with original absolute positions.
2. **A `DynamicCache` that stores unrotated K/V** — so re-selection on later chunks doesn't have to un-rotate stale entries.

Two cooperating intercepts make this work:

### 1. Identity-RoPE interceptor

`ResearchRotaryEmbedding` replaces every `rotary_emb` submodule (detected by an `inv_freq` buffer) with a module whose `forward()` returns `(cos=1, sin=0)`. The model's own `apply_rotary_pos_emb` then becomes a no-op, so Q/K arrive at the attention hook **unrotated**. The interceptor also stashes the `position_ids` it was called with, so the hook can recover absolute positions.

### 2. Attention hook

`prefill_attention` and `decode_attention` are registered into `transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS` via `_with_attention`. Inside the hook:

1. Apply real RoPE manually (`rope.compute(...)`) using the stashed absolute positions.
2. Run `_prefill_attn_impl` or `_decode_attn_impl` — **these are the override points** for custom kernels (Triton, FlashAttention variants, custom CUDA, etc.).

Because identity RoPE is a no-op, HF's `DynamicCache` ends up accumulating **raw** K/V — there's no separate cache storage to manage.

### Chunked sparse prefill

`ResearchAdapter._prefill` loops over `cache_config.chunk_size`-sized chunks, calling the model with `past_key_values` and explicit absolute `position_ids` each iteration. Inside the hook:

- **First chunk / small context / `selection='full'`:** dense path. Apply RoPE to full Q/K, run dense causal SDPA.
- **Subsequent chunks:** sparse path:
  - `SparseSelector.select()` picks **global-sink + top-k-middle + local-window** tokens from history; the current chunk's K/V is kept verbatim.
  - Apply RoPE to Q (chunk positions), selected hist-K (original positions), and curr-K (chunk positions) — each with its true position ID.
  - Combine as `[selected_history | current_chunk]`.
  - Mask is zeros for history columns and upper-triangular `-inf` for current-chunk columns.

**Key invariant:** `T_prev = keys.shape[-2] - queries.shape[-2]` is the absolute start of the current chunk.

### Sparse decode

`decode_attention` calls `SparseSelector.select` on the full raw K/V cache, applies RoPE with each token's **original absolute position**, then runs `_decode_attn_impl`.

### Wiring config

`runner._setup_adapter` pulls the `cache_config` dict out of `EvalConfig.llm_kwargs`, converts it to `CacheConfig`, and passes it to `ResearchAdapter`. Selection modes (where applicable): `'qkv2'` (default, QK·‖V‖₂), `'qk'`, `'full'`.

---

## Sketch / KV-cache compression API

The `eval_harness/sketch/` package implements KVPress-style sketches that compress the KV cache during prefill or decode.

| Sketch                  | Class                  | Trigger                         | What it does                                                 |
| ----------------------- | ---------------------- | ------------------------------- | ------------------------------------------------------------ |
| `none`                  | _(no sketch)_          | —                               | Pass-through; full KV cache                                  |
| `knorm`                 | `KnormSketch`          | Prefill                         | Keep tokens with largest ‖K‖ norm                            |
| `reattention`           | `ReAttentionSketch`    | Prefill                         | Re-score with QK relevance over middle tokens                |
| `random`                | `RandomSketch`         | Prefill                         | Random baseline                                              |
| `decoding_knorm`        | `DecodingSketch`       | Decode (periodic)               | Compress mid-decode at every `compression_interval` steps    |
| `prefill_decoding_knorm`| `PrefillDecodingSketch`| Prefill + decode                | Compress at prefill end and during decode                    |

Configure via `llm_kwargs.cache_config` in your YAML:

```yaml
llm_kwargs:
  cache_config:
    sketch_name: knorm
    compression_ratio: 0.5
    compression_interval: 512
    target_size: 2048
    hidden_states_buffer_size: 256
    max_context_length: 65536
    log_cache_seq_len: true
```

---

## Adding a new benchmark

1. Create a file in `eval_harness/benchmarks/` (e.g. `my_bench.py`).
2. Subclass `base.Benchmark` and implement `load()` and `score()`.
3. Register it:

```python
from .base import Benchmark
from .registry import register_benchmark

@register_benchmark(name="my_bench", aliases=["mb"])
class MyBench(Benchmark):
    def load(self, subsets=None):
        # return a pandas.DataFrame with required columns:
        # context, question, answer_prefix (optional), max_new_tokens (optional),
        # plus whatever ground-truth columns score() needs.
        ...

    def score(self, df):
        # return {"accuracy": ..., "by_subset": {...}}
        ...
```

The benchmark is auto-discovered by `ensure_benchmarks_loaded()` the next time the registry is queried — no manual import.

---

## Adding a new attention kernel

Subclass `ResearchAdapter` and override either or both of:

- `_prefill_attn_impl(query, key, value, attention_mask, ...)`
- `_decode_attn_impl(query, key, value, ...)`

**Do not** override `prefill_attention` / `decode_attention` themselves unless you also intend to change selection or RoPE handling — those are the integration boundary, not the algorithm boundary.

Position IDs everywhere are **absolute** (each token's index in the full sequence), not chunk-relative. Honor this invariant or sparse attention breaks silently.

---

## Testing

The test suite is intentionally GPU-free.

```bash
python -m unittest discover eval_harness/tests -v
```

Tests bypass model loading via `object.__new__(Adapter)` plus fake `nn.Module` doubles — so they exercise the prefill/decode plumbing, sketch wiring, and benchmark loaders without ever touching CUDA or downloading weights.

Highlights:

- `test_hf_adapter.py` — prefill/decode boundaries, position ID accounting
- `test_research_adapter.py` — identity-RoPE interceptor, chunked sparse prefill
- `test_cache_adapter.py` — `DynamicCache` semantics with raw K/V
- `test_benchmarks_*.py` — registry, RULER, LongBench, Prism-1M loaders

---

## Reproducibility & conventions

- **Determinism.** `seed` (default `42`) is applied to `random`, `numpy`, `torch`, and CUDA. Greedy decoding (`temperature=0.0`) is the default.
- **Per-context grouping.** The runner does `df.groupby("context", sort=False)` and the adapter receives all questions for a single context at once — prefix caching (vLLM) or shared prefill (HF/Research) makes this dramatically faster than per-row prompting.
- **Absolute position IDs.** Everywhere in the codebase, position IDs refer to the token's index in the full sequence — never the index inside a chunk. New kernels must respect this.
- **No model loading in tests.** Use `object.__new__(Adapter)` plus fake modules; never download weights from a test.
- **Config persistence.** The exact `EvalConfig` is written to `config.yaml` next to every results dir.
- **Run directories are never overwritten.** Collisions append `/1`, `/2`, ... suffixes.

---

## Troubleshooting

| Symptom                                                              | Likely cause / fix                                                                                       |
| -------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `RuntimeError: CUDA out of memory` during prefill                    | Reduce `max_model_len`, set `gpu_memory_utilization: 0.85`, or switch to `dtype: bfloat16`.              |
| `Unknown benchmark '...'`                                            | Check `available_benchmarks()`; benchmark module may have a syntax error preventing registry import.     |
| `Inconsistent answer_prefix values within the same context group`    | Research backend requires one `answer_prefix` per context group — fix the benchmark loader.              |
| vLLM ignores `attn_implementation`                                   | vLLM has its own attention; `attn_implementation` only applies to `hf` and `research` backends.          |
| RAG: connection refused to Ollama                                    | Start the server first: `~/.ollama/bin/ollama serve`. Verify with `curl http://localhost:11434/api/tags`. |
| RAG: requests time out after long idle                               | GPU may have entered low-power state — start the wake-lock command shown in the RAG section.             |
| Research backend gives different numbers vs HF on the same prompt    | Confirm `attn_implementation: sdpa` is set; SDPA is the parity path. FA2 may not exercise the hook.      |
| Tests download model weights unexpectedly                            | A test forgot `object.__new__` — file a bug.                                                             |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Quick notes:

- Pre-commit hooks live in `.pre-commit-config.yaml`; install with `pre-commit install`.
- Keep unit tests GPU-free.
- New benchmarks go in `eval_harness/benchmarks/` and are auto-discovered.
- For new sketches, follow the `BaseSketch` interface in `eval_harness/sketch/sketches/base_sketch.py`.

---

## Citation

If you use Prism-Test in academic work, please cite the repository:

```bibtex
@software{prism_test,
  title        = {Prism-Test: A Long-Context Inference Evaluation Framework},
  author       = {{Prism-Test Contributors}},
  year         = {2026},
  url          = {https://github.com/<your-org>/Prism-Test}
}
```
