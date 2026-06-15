# Prism-Test

> A long-context inference evaluation framework for research on extending transformer context windows beyond their original training limits.

Prism-Test is a unified, reproducible harness for benchmarking large language models on **long-context** tasks — including contexts that vastly exceed a model's native training window. It standardizes evaluation across popular long-context suites (RULER, LOFT, LongBench, InfiniteBench, GSM-Infinite, AIME, Loogle, and the Prism-1M dataset), while exposing first-class extension points for **inference-time context compression** research: sparse attention, KV-cache eviction sketches, hybrid sparse/dense prefill kernels, and RAG-style retrieval.

It supports four interchangeable inference backends — `vllm`, `hf`, `research`, and `rag` — and ships with detailed quality + systems metrics (accuracy, retrieval recall, latency, throughput, memory, KV-cache size, prefill/decode efficiency).

> **If you're here to run benchmarks, design compression experiments, or plug in your own attention code, read [BENCHMARKING.md](BENCHMARKING.md).** This README covers what's in the repo and how to get a first run working; BENCHMARKING goes into adapter selection, where to plug your code in at several layers of depth, the research-backend architecture, and how to add a new benchmark.

---

## Contents

- [Why Prism-Test](#why-prism-test)
- [Key features](#key-features)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Backends at a glance](#backends-at-a-glance)
- [Supported benchmarks](#supported-benchmarks)
- [Testing](#testing)
- [Contributing](#contributing)
- [Citation](#citation)

---

## Why Prism-Test

Most evaluation harnesses are built around **short-context throughput**. Long-context research has different requirements:

1. **Inputs that exceed `max_position_embeddings`** — the model has never seen 128K, 250K, or 1M tokens during training.
2. **Per-context grouping** — many long-context benchmarks ask several questions about the same massive document; the harness must amortize the prefill across questions.
3. **Attention/KV-cache surgery** — research methods recover/restructure positional info and choose which tokens stay in the cache.
4. **Apples-to-apples comparison** — same prompt, same scorer, swappable backends.

Prism-Test gives you all four with a single config file.

---

## Key features

- **Four backends, one runner.** `vllm` for production throughput, `hf` for clean prefill/decode debugging, `research` for compression experiments, `rag` for retrieval baselines.
- **Per-context batching.** The runner groups by shared context so a 1M-token document is prefilled once and reused across all of its questions.
- **Three pluggable doors.** The `research` backend exposes three independent extension points, installed as nested context managers (positional → attention → KV): **Door 1 — positional** (RoPE frequency/position: YaRN, NTK, Linear-PI; wraps the shared `rotary_emb`), **Door 2 — attention** (the attention math: DCA replaces `self_attn.forward`, ReAttention prunes via a post-attention hook), and **Door 3 — KV compression** (below). There is no identity-RoPE swap, so HF's `DynamicCache` stores RoPE-rotated K/V; methods that need position-agnostic K recover it on the fly (ReAttention un-rotates cached K, DCA re-rotates keys at cyclic positions).
- **Pluggable KV compressors (Door 3).** A registry (`@register_kv_compressor`, auto-discovered) of ~36 KV-compression baselines — mostly faithful kvpress 0.5.1 ports (`snapkv`, `pyramidkv`, `tova`, `expected_attention`, `streaming_llm`, `adakv`, `duo_attention`, `kvzip`, `qfilter`, wrappers like `chunk`/`composed`/`per_layer_compression`, ...) alongside `knorm`, `reattention`, `random`, and decoding-time variants — compressing the KV cache during prefill or decode (`compression_schedule` ∈ streaming | post_prefill | decode). List them with `from eval_harness.kv_compression import available_kv_compressors`; see [BENCHMARKING.md](BENCHMARKING.md#layer-1--a-kv-cache-compression-sketch) for the full roster and per-compressor constraints.
- **A standalone benchmark registry.** Drop a file into [eval_harness/benchmarks/](eval_harness/benchmarks/), decorate with `@register_benchmark`, and it's runnable from the CLI.
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
│   ├── research_adapter.py     # HF subclass: builds the three doors (positional/attention/KV) and runs them through ResearchGenerationPipeline
│   ├── research_pipeline.py    # ResearchGenerationPipeline: chunked prefill + decode; installs the doors as nested context managers
│   ├── positional_methods/     # DOOR 1 (RoPE freq/position): base.py, registry.py, yarn.py, ntk.py, linear_pi.py
│   ├── attention_methods/      # DOOR 2 (attention math): base.py, registry.py, dca.py + legacy reattention.py, reattention_exact.py
│   ├── kernels/                # Triton einsum-topk + bitonic merge (ReAttention), flash-attn-with-LSE (DCA)
│   ├── rag_adapter.py          # OnePassRAG backend wrapper
│   ├── rag/                    # LanceDB + llm-embedder + Ollama
│   ├── mlp_methods/            # DOOR 4 (reserved seam only — MoE/activation sparsity; not implemented)
│   ├── kv_compression/         # DOOR 3 (KV compression)
│   │   ├── attention_patch.py  # global masking patch (adakv/dms/duo_attention/kvzip/fastkvzip)
│   │   ├── cache_adapter.py
│   │   ├── base.py             # KVCompressor / ScorerKVCompressor
│   │   ├── registry.py
│   │   └── compressors/        # @register_kv_compressor registry: knorm, snapkv, pyramidkv, tova, ... (~36 baselines)
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
├── evaluate/                   # ready-made run configs: evaluate_{vllm,hf,kv,positional,dca,reattention,common}.yaml
├── run_eval.py                 # thin wrapper over CliEntryPoint
├── pyproject.toml              # project metadata + loose dep constraints
├── uv.lock                     # uv-native pinned lock (reproducible installs)
├── requirements.txt            # pinned pip export (autogenerated from uv.lock)
├── BENCHMARKING.md             # practical guide for benchmarking & research
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
- (Optional) Ollama for the `rag` backend — see [BENCHMARKING.md](BENCHMARKING.md#rag-backend-setup)

### Install

```bash
git clone https://github.com/<your-org>/Prism-Test.git
cd Prism-Test
```

**Option A — loose install (recommended when integrating into your own research environment).** Uses the constraints declared in [pyproject.toml](pyproject.toml), so it co-exists with the torch / transformers / vllm versions you already have:

```bash
pip install .
# or, for an editable install:
pip install -e .
```

**Option B — reproducible install (for matching official benchmark numbers exactly).** Uses the pinned [uv.lock](uv.lock):

```bash
# with uv (fastest):
uv sync

# or with pip:
pip install -r requirements.txt
```

Key packages and their supported ranges (from [pyproject.toml](pyproject.toml)):

| Package                 | Constraint     | Purpose                            |
| ----------------------- | -------------- | ---------------------------------- |
| `vllm`                  | `>=0.5.5,<1.0` | Production inference backend       |
| `transformers`          | `>=4.45,<6`    | HF / research backends             |
| `torch`                 | `>=2.1,<3`     | Tensor ops, CUDA                   |
| `datasets`              | `>=2.20`       | Benchmark loading                  |
| `pandas`, `numpy`       | latest         | Per-row dataframes, scoring        |
| `lancedb`               | latest         | RAG vector store                   |
| `llama-index-*`         | latest         | RAG embeddings + Ollama LLM client |
| `sentence-transformers` | latest         | Embedding models                   |
| `ninja`, `setuptools`   | latest         | Build-from-source dependencies     |

Conda users: create a conda env for Python, then use either option above — no separate `environment.yml` is required.

```bash
conda create -n prism python=3.11 && conda activate prism
pip install .   # or `pip install -r requirements.txt` for the pinned env
```

To regenerate the lock and pip export after editing dependencies in [pyproject.toml](pyproject.toml):

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

Pick a starter config from [evaluate/](evaluate/) (e.g. [evaluate/evaluate_vllm.yaml](evaluate/evaluate_vllm.yaml)) and edit:

```yaml
benchmark: ruler16k
subsets: qa_1                          # comma-separated list, or null for defaults
backend: vllm                          # vllm | hf | rag | research
model: meta-llama/Llama-3.1-8B-Instruct

tensor_parallel_size: 1
dtype: bfloat16
max_model_len: 65536
gpu_memory_utilization: 0.9
enable_prefix_caching: true

max_new_tokens: 128
temperature: 0.0
seed: 42

max_requests: 200
output_dir: ./results
```

### 2. Run

```bash
python -m eval_harness.cli run --config_file ./evaluate/evaluate_vllm.yaml
```

Or override any field on the CLI:

```bash
python -m eval_harness.cli run \
  --config_file ./evaluate/evaluate_vllm.yaml \
  --benchmark longbench \
  --subsets narrativeqa,hotpotqa \
  --backend vllm \
  --max_new_tokens 128
```

### 3. Read results

```
results/<benchmark>__<model>__<backend>__t0__p1__subsets_qa_1/
├── predictions.csv     # per-row inputs + predicted_answer + ground truth
├── metrics.json        # aggregated benchmark scores
└── config.yaml         # the exact config that produced this run
```

If the output directory already exists, a numeric suffix (`/1`, `/2`, ...) is appended — runs are never silently overwritten.

For **picking the right backend, controlling experimental conditions, plugging in your own sketches or kernels, and adding new benchmarks**, see [BENCHMARKING.md](BENCHMARKING.md).

---

## Backends at a glance

| Backend     | Use when                                              | What you get                                              |
| ----------- | ----------------------------------------------------- | --------------------------------------------------------- |
| `vllm`      | Production-quality numbers; large-batch eval.         | Best throughput; prefix caching across same-context Qs.   |
| `hf`        | Small-context debugging; profiling.                   | Clean `_prefill`/`_decode` split; native FA2 if present.  |
| `research`  | Sparse attention, KV compression, custom kernels.     | Three doors — positional (YaRN/NTK/Linear-PI), attention (DCA / ReAttention), KV compression (~36 baselines); chunked or single-pass prefill; cache stores rotated K/V. |
| `rag`       | Retrieval baselines.                                  | OnePassRAG (LanceDB + llm-embedder + Ollama llama3.1).    |

Backend selection criteria, tradeoffs, and the research-backend architecture live in [BENCHMARKING.md](BENCHMARKING.md#pick-a-backend).

---

## Supported benchmarks

| Benchmark           | Tasks / subsets                              | Notes                                              |
| ------------------- | -------------------------------------------- | -------------------------------------------------- |
| `ruler`             | Configurable via subsets                     | Generic RULER loader                               |
| `ruler16k`          | `qa_1`, `qa_2`, ...                          | 16K context                                        |
| `ruler32k`          | same                                         | 32K context                                        |
| `ruler64k`          | same                                         | 64K context                                        |
| `ruler128k`         | same                                         | 128K context                                       |
| `longbench`         | `narrativeqa`, `qasper`, `hotpotqa`, ...     | Standard LongBench English subsets                 |
| `longbenchv2`       | LongBench v2 tasks                           |                                                    |
| `infinite_bench`    | `passkey`, `kv_retrieval`, ...               | Extreme-length retrieval                           |
| `loft`              | LOFT subtasks                                | Retrieval / reasoning                              |
| `loft_rag`          | LOFT RAG variant                             |                                                    |
| `loogle`            | Loogle subtasks                              |                                                    |
| `gsm_infinite_128k` | GSM-Infinite at 128K                         | Math reasoning at long context                     |
| `aime` / `aime2024` / `aime2025` | AIME problems                   | Math reasoning                                     |
| `zero_scrolls`      | ZeroSCROLLS                                  |                                                    |
| `prism1m`           | `128K`, `250K`, `1M` × `Easy`, `Medium`      | Ships with this repo under [datasets/Prism-Data/](datasets/Prism-Data/) |
| `mock_benchmark`    | Tiny synthetic                               | Smoke-test the harness without downloading data    |

List the live registry at any time:

```python
from eval_harness.benchmarks.registry import available_benchmarks
print(available_benchmarks())
```

To **add a new benchmark**, see the walkthrough in [BENCHMARKING.md](BENCHMARKING.md#add-a-new-benchmark).

---

## Testing

The test suite is intentionally GPU-free.

```bash
python -m unittest discover eval_harness/tests -v
```

Tests bypass model loading via `object.__new__(Adapter)` plus fake `nn.Module` doubles — so they exercise the prefill/decode plumbing, three-door wiring, and benchmark loaders without ever touching CUDA or downloading weights.

Highlights:

- [test_hf_adapter.py](eval_harness/tests/test_hf_adapter.py) — prefill/decode boundaries, position ID accounting
- [test_research_adapter.py](eval_harness/tests/test_research_adapter.py) — door selection, pipeline wiring, generation through `ResearchGenerationPipeline`
- [test_prefill_methods.py](eval_harness/tests/test_prefill_methods.py) — attention-method (Door 2) wiring into the research adapter and post-attention `prefill_forward_hook` ordering (ReAttention prune, DCA `self_attn.forward` replacement)
- [test_three_doors_integration.py](eval_harness/tests/test_three_doors_integration.py) / [test_positional_methods.py](eval_harness/tests/test_positional_methods.py) / [test_chunked_prefill.py](eval_harness/tests/test_chunked_prefill.py) — three-door composition, Door 1 RoPE math, chunked-prefill equivalence
- [test_cache_adapter.py](eval_harness/tests/test_cache_adapter.py) — `DynamicCache` checkpoint/restore semantics over rotated K/V
- `test_benchmarks_*.py` — registry, RULER, LongBench, Prism-1M loaders

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Quick notes:

- Pre-commit hooks live in [.pre-commit-config.yaml](.pre-commit-config.yaml); install with `pre-commit install`.
- Keep unit tests GPU-free.
- New benchmarks go in [eval_harness/benchmarks/](eval_harness/benchmarks/) and are auto-discovered.
- For new KV compressors, follow the `KVCompressor` interface in [eval_harness/kv_compression/base.py](eval_harness/kv_compression/base.py).

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
