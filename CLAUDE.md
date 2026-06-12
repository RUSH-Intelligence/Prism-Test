# Prism-Test

Long-context inference evaluation framework for research on extending transformer context windows beyond their original training limits.

The framework evaluates language models on context lengths far beyond their training context, enabling systematic research on inference-time context compression methods. It provides a unified interface for implementing and benchmarking custom inference strategies while remaining compatible with standard HuggingFace and vLLM backends.

The framework supports standardized evaluation across benchmarks such as RULER, LOFT, LongBench, InfiniteBench, GSM-Infinite, AIME, and custom long-context tasks, with detailed reporting of both quality and systems metrics including accuracy, retrieval performance, latency, throughput, memory usage, KV-cache size, and prefill/decode efficiency.

User-facing docs: [README.md](README.md) for setup/overview, [BENCHMARKING.md](BENCHMARKING.md) for the benchmarker's guide (adapter selection, where to plug code in, research backend internals, RAG/Ollama setup).

## Repo layout

```
eval_harness/
  cli.py                 # argparse entry; `python -m eval_harness.cli run --config_file ...`
  config.py              # EvalConfig dataclass; backend ∈ {vllm, hf, rag, research}
  runner.py              # load dataset → setup adapter → groupby(context) generation → score
  vllm_adapter.py        # vLLM backend
  hf_adapter.py          # HF backend — clean prefill/decode split, native flash attention
  research_adapter.py    # HF subclass: wires prefill_methods (context extension) + sketches (KV compression) into SketchTextGenerationPipeline
  prefill_methods/       # Layer-0 context-extrapolation methods (base.py, registry.py, reattention.py, reattention_exact.py, dca.py)
  sketch/                # KV-compression sketches + SketchTextGenerationPipeline (pipeline.py)
  kernels/               # Triton einsum-topk + bitonic-merge (ReAttention) + flash-attn-with-LSE (DCA)
  rag_adapter.py, rag/   # OnePassRAG (LanceDB + llm-embedder + Ollama llama3.1)
  benchmarks/            # one module per benchmark; registry.py exposes get_benchmark()
  tests/                 # unittest — no model loading; uses object.__new__ + fake models
run_eval.py              # thin wrapper over CliEntryPoint
evaluate/                # ready-made run configs: evaluate_{vllm,hf,kv,dca,reattention,common}.yaml
```

## Running

```bash
# Eval
python -m eval_harness.cli run --config_file ./evaluate/evaluate_common.yaml   # or evaluate_{vllm,hf,kv,dca,reattention}.yaml
# or override on CLI: --benchmark, --subsets, --backend, --model, --max_new_tokens, ...

# Tests (from repo root)
python -m unittest discover eval_harness/tests -v
```

Results land in `results/<benchmark>__<model>__<backend>__.../{predictions.csv,metrics.json,config.yaml}`.

## Backend notes

- **`vllm`** — production path. Prefix caching on by default.
- **`hf`** — small-context / debugging path. `_prefill` and `_decode` are direct model calls; flash-attn-2 loads natively if available (no override hooks installed).
- **`research`** — `ResearchAdapter` (subclass of `HFAdapter`) for context compression experiments.
- **`rag`** — OnePassRAG; requires a running Ollama server (see BENCHMARKING.md).

## ResearchAdapter — architecture

> **Note (2026-06):** earlier revisions of this section described an identity-RoPE
> interceptor (`ResearchRotaryEmbedding`), an `ALL_ATTENTION_FUNCTIONS` hook with
> `_prefill_attn_impl`/`_decode_attn_impl`, a `SparseSelector`, a `cache_config.chunk_size`
> chunked-prefill loop, and a raw-K/V cache. **None of that exists in the code.** It was an
> earlier design plan (the "Raw-K Invariant" proposal) that was *not* the path actually shipped.
> The description below reflects the current implementation.

`ResearchAdapter` (`research_adapter.py`) is a thin `HFAdapter` subclass. It **deletes**
`rope_method`/`rope_scale_factor` (`research_adapter.py:60`) and installs **no** identity-RoPE
swap and **no** attention-function override. It builds a `sketch` (KV compression) and a
`prefill_method` (context extension) from `CacheConfig` and runs everything through
`SketchTextGenerationPipeline` (`sketch/pipeline.py`).

Key consequence: **prefill is a single full-context pass** through the model's *normal* forward
(`pipeline._forward` → `self.model.model(input_ids=context_ids, past_key_values=cache)`), so the
model's own layers apply RoPE and HF's `DynamicCache` stores **RoPE-rotated K/V** (not raw).
Methods that need position-agnostic K recover it themselves (ReAttention un-rotates on the fly;
DCA replaces the attention forward and re-rotates at cyclic positions).

### Extension points — two mechanisms

Context-extension / compression behavior is supplied by `eval_harness/prefill_methods/` and
`eval_harness/sketch/sketches/`, installed by the pipeline as **nested context managers**
(`prefill_method(model)` outer, `sketch(model)` inner) so forward hooks fire method-then-sketch:

1. **Post-attention prune hook** (`PrefillMethod.prefill_forward_hook`, `base.py:97`): fires
   *after* each full-attention layer during prefill, may return `(keys, values)` to replace the
   cache contents. No-ops on decode (`base.py:_is_decoding_step`). **ReAttention** uses this:
   it un-rotates cached K to score raw Q·K, selects `[global | top-k middle | local]`, and
   prunes the cache. Prefill-only. Because HF's normal decode shares ONE causal mask/position
   grid across layers (sized from layer 0), per-layer selection must not leave a *ragged*
   cache — `uniform_retained` (default on) equalizes the post-span-expansion selection to a
   per-prefill target (`uniform_budget`, else the first hooked layer's selection size; shorter
   layers recency-padded, longer layers shrunk by the frequency-clip rule). Set
   `uniform_retained=False` only for single-layer models / custom decode paths.
2. **Full `self_attn.forward` replacement** (monkeypatch): for methods that must change *how*
   attention scores positions. **DCA** uses this (`DCAMethod.__call__`, `dca.py:142`), staying active across
   **both prefill and decode** — it stores keys rotated at cyclic position `pos % chunk_len`
   and runs the 3-component intra/successive/inter decomposition merged by online-softmax
   (`kernels/dca_flash.py`). **ReAttention-exact** (`reattention_exact.py`, registered
   `reattention_exact`) also uses this mechanism to reproduce the *original* ReAttention
   computation: raw (pre-RoPE) K/V cached for the entire context (never pruned), prefill
   processed in `prefill_chunk_size` query chunks inside the forward, pre-attention recall
   per chunk/decode-step (`recall_option='whole'` default = decode-time re-selection too),
   RoPE applied after selection at original absolute positions, reference 128-alignment
   quirk replicated, `einsum_topk` kernel gated exactly as upstream (`qlen != 1`,
   `mid_size ∈ {1,4}`). `recall_option='full_attn'` reduces it to the no-method baseline
   (tested bitwise). Contrast with mechanism 1's `reattention`: that baseline keeps prefill
   attention exact and prunes the decode-facing cache; this one bounds the attention scope
   during prefill itself, like the paper.

> ⚠️ Tier-1 frequency-only methods (NTK/YaRN/Linear-PI) are **not yet functional**:
> `PrefillMethod.compute_inv_freq` exists but **nothing calls it** (the pipeline only invokes
> `prefill_forward_hook`, `compute_question_position_ids`, `on_prefill_start/end`). Implementing
> them needs a RoPE-level interceptor that the framework currently lacks.

### Decode

Decode runs per-token in `pipeline.generate_answer` via `self.model(...)`. ReAttention does no
decode-time selection (its hook no-ops on decode). DCA keeps its `self_attn.forward` replacement
active and recomputes cyclic query positions per step (`dca._dca_decode_attention`).

### Wiring config

`runner._setup_adapter` builds a `CacheConfig` from `EvalConfig.llm_kwargs`. Relevant fields:
`prefill_method` (str, e.g. `"dca"`, `"reattention"`, `"none"`) + `prefill_method_kwargs` (dict),
and the sketch fields (`sketch_name`, `compression_ratio`, …). `ResearchAdapter._build_prefill_method`
resolves the name via `prefill_methods.get_prefill_method`. ReAttention's `recall_type` defaults
to `'qk'` (options: `qk` | `qkv` | `qkv2`) — this is a method kwarg, **not** an adapter-level
selection mode.

## Conventions

- Tests bypass model loading via `object.__new__(Adapter)` plus fake modules — never load real weights in unit tests.
- Position IDs everywhere are *absolute* (token's position in the full sequence), not chunk-relative.
- New benchmarks: drop into `eval_harness/benchmarks/`, subclass `base.Benchmark`, register in `registry.py`.
- New prefill methods: drop into `eval_harness/prefill_methods/`, subclass `PrefillMethod`, decorate with `@register_prefill_method`. Override `prefill_forward_hook` for a post-attention prune (ReAttention-style) or override `__call__` to replace `self_attn.forward` (DCA-style). Custom kernels live in `eval_harness/kernels/`.
- The HF `DynamicCache` on the research path stores **RoPE-rotated** K/V. `KnormSketch` norm-scoring is still valid (RoPE is orthogonal, norms preserved), but any consumer that assumes contiguous absolute positions must account for DCA's cyclic-rotated keys.
