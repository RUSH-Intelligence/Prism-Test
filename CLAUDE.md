# Prism-Test

Long-context inference evaluation framework for research on extending transformer context windows beyond their original training limits.

The framework evaluates language models on context lengths far beyond their training context, enabling systematic research on inference-time context compression methods. It provides a unified interface for implementing and benchmarking custom inference strategies while remaining compatible with standard HuggingFace and vLLM backends.

The framework supports standardized evaluation across benchmarks such as RULER, LOFT, LongBench, InfiniteBench, GSM-Infinite, AIME, and custom long-context tasks, with detailed reporting of both quality and systems metrics including accuracy, retrieval performance, latency, throughput, memory usage, KV-cache size, and prefill/decode efficiency.

User-facing docs: see [EVAL_HARNESS.md](EVAL_HARNESS.md) for setup, supported benchmarks, and the RAG backend (Ollama).

## Repo layout

```
eval_harness/
  cli.py                 # argparse entry; `python -m eval_harness.cli run --config_file ...`
  config.py              # EvalConfig dataclass; backend ∈ {vllm, hf, rag, research}
  runner.py              # load dataset → setup adapter → groupby(context) generation → score
  vllm_adapter.py        # vLLM backend
  hf_adapter.py          # HF backend — clean prefill/decode split, native flash attention
  research_adapter.py    # HF subclass with identity-RoPE intercept + chunked sparse prefill
  rag_adapter.py, rag/   # OnePassRAG (LanceDB + llm-embedder + Ollama llama3.1)
  benchmarks/            # one module per benchmark; registry.py exposes get_benchmark()
  tests/                 # unittest — no model loading; uses object.__new__ + fake models
run_eval.py              # thin wrapper over CliEntryPoint
evaluate_config.yaml     # default config
```

## Running

```bash
# Eval
python -m eval_harness.cli run --config_file ./evaluate_config.yaml
# or override on CLI: --benchmark, --subsets, --backend, --model, --max_new_tokens, ...

# Tests (from repo root)
python -m unittest discover eval_harness/tests -v
```

Results land in `results/<benchmark>__<model>__<backend>__.../{predictions.csv,metrics.json,config.yaml}`.

## Backend notes

- **`vllm`** — production path. Prefix caching on by default.
- **`hf`** — small-context / debugging path. `_prefill` and `_decode` are direct model calls; flash-attn-2 loads natively if available (no override hooks installed).
- **`research`** — `ResearchAdapter` (subclass of `HFAdapter`) for context compression experiments.
- **`rag`** — OnePassRAG; requires a running Ollama server (see EVAL_HARNESS.md).

## ResearchAdapter — architecture

This is the non-obvious part of the codebase. Two intercepts cooperate to give the attention hook raw, pre-RoPE Q/K:

1. **Identity-RoPE interceptor** (`ResearchRotaryEmbedding`): every `rotary_emb` submodule (detected by `inv_freq` buffer) is swapped for a module whose `forward()` returns `(cos=1, sin=0)`. The model's own `apply_rotary_pos_emb` then becomes a no-op, so Q/K arrive at the attention hook unrotated. The interceptor also stashes the `position_ids` it was called with so the hook can recover them.
2. **Attention hook** (`prefill_attention` / `decode_attention`): registered into `transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS` via `_with_attention`. Applies real RoPE manually (`rope.compute()`), runs `_prefill_attn_impl` / `_decode_attn_impl` (override these for custom kernels — Triton, FlashAttention, etc.).

Because identity RoPE is a no-op, HF's `DynamicCache` accumulates raw K/V — no separate cache storage needed.

### Chunked sparse prefill

`ResearchAdapter._prefill` loops over `cache_config.chunk_size`-sized chunks, calling the model with `past_key_values` and explicit absolute `position_ids` each iteration. Inside the hook:

- **First chunk / small context / `selection='full'`**: dense path. Apply RoPE to full Q/K, dense causal SDPA.
- **Subsequent chunks**: sparse path. `SparseSelector.select()` picks global-sink + top-k-middle + local-window tokens from history; current chunk K/V kept verbatim. Apply RoPE to Q (chunk positions), selected hist-K (original positions), curr-K (chunk positions) — each with its true position ID. Combine as `[selected_history | current_chunk]`. Mask is zeros for history columns and upper-triangular `-inf` for current-chunk columns.

Key invariant: `T_prev = keys.shape[-2] - queries.shape[-2]` is the absolute start of the current chunk.

### Sparse decode

`decode_attention` calls `SparseSelector.select` on the full raw K/V cache, applies RoPE with each token's *original* absolute position, runs `_decode_attn_impl`.

### Wiring config

`runner._setup_adapter` pulls `cache_config` dict out of `EvalConfig.llm_kwargs`, converts to `CacheConfig`, passes to `ResearchAdapter`. Selection modes: `'qkv2'` (default, QK·‖V‖₂), `'qk'`, `'full'`.

## Conventions

- Tests bypass model loading via `object.__new__(Adapter)` plus fake modules — never load real weights in unit tests.
- Position IDs everywhere are *absolute* (token's position in the full sequence), not chunk-relative.
- New benchmarks: drop into `eval_harness/benchmarks/`, subclass `base.Benchmark`, register in `registry.py`.
- The `research` backend's `_prefill_attn_impl` / `_decode_attn_impl` are the override points for new attention kernels; don't override `prefill_attention` / `decode_attention` unless you also want to change selection or RoPE handling.
