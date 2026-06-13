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
  research_adapter.py    # HF subclass: builds the THREE DOORS from ResearchConfig and runs them through ResearchGenerationPipeline (research_pipeline.py)
  research_pipeline.py   # ResearchGenerationPipeline: chunked prefill + decode, installs the three door context managers (positional → attention → kv) nested
  positional_methods/    # DOOR 1 (RoPE freq/position): base.py (PositionalMethod), registry.py, yarn.py, ntk.py, linear_pi.py
  attention_methods/     # DOOR 2 (attention math): base.py (AttentionMethod + AttentionPhase), registry.py, dca.py; plus the faithful ReAttention methods (reattention.py prune, reattention_exact.py) as legacy PrefillMethod subclasses on the same method slot — _method_base.py (RoPE helpers + PrefillMethod) + _method_registry.py (register_prefill_method)
  kv_compression/        # DOOR 3 (KV compression): base.py (KVCompressor/ScorerKVCompressor + CompressionSchedule/Operation), registry.py (@register_kv_compressor), cache_adapter.py, utils.py, attention_patch.py, compressors/ (~36 KV baselines, mostly kvpress 0.5.1 ports)
  mlp_methods/           # DOOR 4 (reserved seam only — MoE/activation-sparsity; not implemented)
  kernels/               # Triton einsum-topk + bitonic-merge (ReAttention) + flash-attn-with-LSE (DCA)
  rag_adapter.py, rag/   # OnePassRAG (LanceDB + llm-embedder + Ollama llama3.1)
  benchmarks/            # one module per benchmark; registry.py exposes get_benchmark()
  tests/                 # unittest — no model loading; uses object.__new__ + fake models
run_eval.py              # thin wrapper over CliEntryPoint
evaluate/                # ready-made run configs: evaluate_{vllm,hf,kv,positional,dca,reattention,common}.yaml
```

## Running

```bash
# Eval
python -m eval_harness.cli run --config_file ./evaluate/evaluate_common.yaml   # or evaluate_{vllm,hf,kv,positional,dca,reattention}.yaml
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
`rope_method`/`rope_scale_factor` and installs **no** identity-RoPE
swap and **no** attention-function override. It builds the three doors — a positional
method (door 1), an attention method (door 2), and a KV compressor (door 3) — from
`ResearchConfig` and runs everything through `ResearchGenerationPipeline`
(`research_pipeline.py`).

Key consequence: **prefill is a single full-context pass** through the model's *normal* forward
(`pipeline._forward` → `self.model.model(input_ids=context_ids, past_key_values=cache)`), so the
model's own layers apply RoPE and HF's `DynamicCache` stores **RoPE-rotated K/V** (not raw).
Methods that need position-agnostic K recover it themselves (ReAttention un-rotates on the fly;
DCA replaces the attention forward and re-rotates at cyclic positions).

### Extension points — two mechanisms

Context-extension / compression behavior is supplied by `eval_harness/attention_methods/` and
`eval_harness/kv_compression/`, installed by the pipeline as **nested context managers**
(attention method outer, KV compressor inner) so forward hooks fire method-then-compressor:

1. **Post-attention prune hook** (`PrefillMethod.prefill_forward_hook`, `base.py`): fires
   *after* each full-attention layer during prefill, may return `(keys, values)` to replace the
   cache contents. No-ops on decode (the pipeline declares the phase explicitly via
   `KVCompressor.set_phase`, falling back to the `_is_decoding_step` cache_position
   heuristic; `kv_compression/base.py`). **ReAttention** uses this:
   it un-rotates cached K to score raw Q·K, selects `[global | top-k middle | local]`, and
   prunes the cache. Prefill-only. Because HF's normal decode shares ONE causal mask/position
   grid across layers (sized from layer 0), per-layer selection must not leave a *ragged*
   cache — `uniform_retained` (default on) equalizes the post-span-expansion selection to a
   per-prefill target (`uniform_budget`, else the first hooked layer's selection size; shorter
   layers recency-padded, longer layers shrunk by the frequency-clip rule). Set
   `uniform_retained=False` only for single-layer models / custom decode paths.
2. **Full `self_attn.forward` replacement** (monkeypatch): for methods that must change *how*
   attention scores positions. **DCA** uses this (`DCAMethod.__call__`, `dca.py`), staying active across
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

> **Door 1 (positional) is functional.** The pipeline installs `positional_method(model)` as
> the outermost context manager (`research_pipeline.py`), which wraps the shared `rotary_emb`
> so its forward emits modified `(cos, sin)`. `PositionalMethod.compute_inv_freq` (frequency
> scaling — NTK/YaRN) and `remap_position_ids` (position remap — Linear-PI) are both invoked
> per rotary call (`positional_methods/base.py`); `mscale` applies YaRN's logit temperature.
> An attention method that computes its own RoPE (DCA) bypasses `rotary_emb` and therefore
> overrides this door for its layers.

### Decode

Decode runs per-token in `pipeline.generate_answer` via `self.model(...)`. ReAttention does no
decode-time selection (its hook no-ops on decode). DCA keeps its `self_attn.forward` replacement
active and recomputes cyclic query positions per step (`dca._dca_decode_attention`).

### Wiring config — the three doors

`runner._setup_adapter` builds a `ResearchConfig` from `EvalConfig.llm_kwargs["research_config"]`.
The three doors are independent, optional config keys (`none` = off):

- **Door 1 — `positional_method`** (+ `positional_method_kwargs`): `yarn` | `ntk` | `linear_pi`.
  `ResearchAdapter._build_positional_method` → `positional_methods.get_positional_method`.
- **Door 2 — `attention_method`** (+ `attention_method_kwargs`, `attention_phase` ∈
  prefill|decode|both): `dca` | `reattention_exact` | `reattention`.
  `_build_attention_method` resolves the **new `attention_methods` registry first** (DCA), then
  falls back to the **legacy `prefill_methods` registry** (reattention / reattention_exact). Both
  install via the pipeline's method slot; `attention_phase` is applied to native `AttentionMethod`
  instances. ReAttention's `recall_type` defaults to `'qk'` (`qk` | `qkv` | `qkv2`) — a method kwarg.
- **Door 3 — `kv_compressor`** (+ `kv_compressor_kwargs`, `compression_schedule` ∈
  streaming|post_prefill|decode, `compression_ratio`): any registered compressor.
  `_build_kv_compressor` resolves the `kv_compression` registry.

`prefill_chunk_size` (`None` = single pass) drives the chunked prefill the `streaming` schedule
hooks into. The pipeline installs the doors as nested context managers:
`positional_method` (outermost) → `attention_method` → `kv_compressor`.

### Door 3 — KV compressors (registry & roster)

Compressors live in `eval_harness/kv_compression/compressors/`, decorate their class with
`@register_kv_compressor("name")` (`kv_compression/registry.py`), and are auto-discovered on first
lookup — adding one never edits shared files. `ResearchAdapter._build_kv_compressor` resolves
`ResearchConfig.kv_compressor` through the registry, forwards `kv_compressor_kwargs` to the
constructor, injects the adapter-level `compression_ratio` as a default **only when the class
declares a `compression_ratio` dataclass field** (property-based ones — `think`, `simlayerkv`,
`key_rerotation`, `dms` — take it via `kv_compressor_kwargs`/the wrapped compressor), and passes
`compression_schedule` into the compressor's `schedule` field when given. `DecodingSketch` /
`PrefillDecodingSketch` (`decoding_knorm`, `prefill_decoding_knorm`) stay as named special cases
because their nested-compressor args aren't flat kwargs. Live list:
`from eval_harness.kv_compression import available_kv_compressors`. (Base classes: `KVCompressor`,
`ScorerKVCompressor`, renamed from `BaseSketch`/`ScorerSketch`; compressor class names keep their
`…Sketch` suffix.)

The roster (kv_baselines branch) is mostly faithful kvpress 0.5.1 ports — each class docstring
documents params, replicated upstream quirks, and deviations: scorers `knorm`, `random`,
`reattention`, `streaming_llm`, `keydiff`, `lagkv`, `cur`, `leverage`, `non_causal_attention`,
`compactor`, `ridge`, `random_sketch_press` (research-fork; dead-code bug replicated ⇒ ≡ `ridge`),
`expected_attention`, `expected_attention_stats`, `snapkv`, `pyramidkv`, `tova`,
`observed_attention`, `qfilter`, `kvzap`, `finch`, `think` (key-channel zeroing), `simlayerkv`;
masking-based `adakv`, `critical_adakv`, `dms`, `duo_attention`, `kvzip`, `fastkvzip`; wrappers
`criticalkv`, `block`, `chunk`, `chunkkv`, `composed`, `key_rerotation`, `per_layer_compression`.

Constraints to keep in mind when wiring runs or reviewing changes:

- **`observed_attention` needs `attn_implementation: eager`** (only eager returns attention
  probabilities to the hook; sdpa passes `attentions=None` and the sketch asserts).
- **External assets + injection hooks**: `qfilter` (`q_filters`), `kvzap`
  (`model_name_override`), `duo_attention` (`attention_pattern`/`pattern_dir`),
  `expected_attention_stats` (`stats_folder`), `fastkvzip` (`gates`) download model-specific
  artifacts from the HF hub in `post_init_from_model`; use the injection hook on offline nodes
  and in tests.
- **Masking-based presses keep the cache full-length** — no memory savings, faithful attention
  semantics: they record pruned indices on `module.masked_key_indices`, consumed by
  `kv_compression/attention_patch.py` (patches `ALL_ATTENTION_FUNCTIONS` at
  `import eval_harness.kv_compression`; fake keys with `exp(⟨q,k⟩)=0` on every `q_len < k_len`
  forward, reset at next full prefill). They require non-eager attention (sdpa default OK) and are
  incompatible with `self_attn.forward`-replacing attention methods (`dca`, `reattention_exact`).
- **Ragged-cache sketches need flash_attention_2**: `pyramidkv`, `simlayerkv`
  (`lazy_threshold < 1`), `per_layer_compression` (unequal ratios) — `post_init_from_model`
  raises otherwise (sdpa/eager share one decode mask sized from layer 0).
- Most position-sensitive scorers assume vanilla absolute-position rotated keys: do **not**
  combine with `attention_method: dca` (cyclic positions); validated combo is
  `attention_method: none` unless the docstring says otherwise.

## Conventions

- Tests bypass model loading via `object.__new__(Adapter)` plus fake modules — never load real weights in unit tests.
- Position IDs everywhere are *absolute* (token's position in the full sequence), not chunk-relative.
- New benchmarks: drop into `eval_harness/benchmarks/`, subclass `base.Benchmark`, register in `registry.py`.
- New positional methods (door 1): drop into `eval_harness/positional_methods/`, subclass `PositionalMethod`, decorate with `@register_positional_method`. Override `compute_inv_freq` (frequency scaling) and/or `remap_position_ids` (position remap); set `mscale` for a logit temperature.
- New attention methods (door 2): drop into `eval_harness/attention_methods/`, subclass `AttentionMethod`, decorate with `@register_attention_method`. Implement `attention_forward` (one impl; framework gates it by `phase` ∈ prefill|decode|both) and `setup(model)`. Custom kernels live in `eval_harness/kernels/`. (Legacy faithful methods — reattention / reattention_exact — now live in `attention_methods/` as `PrefillMethod` subclasses and are reachable via `attention_method`.)
- New KV compressors (door 3): drop into `eval_harness/kv_compression/compressors/`, subclass `KVCompressor` (or `ScorerKVCompressor` for score-and-topk methods), decorate with `@register_kv_compressor` — auto-discovered, selected via `kv_compressor` + `kv_compressor_kwargs`; `schedule` ∈ streaming|post_prefill|decode gates firing. Ports of kvpress presses replicate upstream quirks on purpose and document deviations in the class docstring; keep that contract when editing.
- The HF `DynamicCache` on the research path stores **RoPE-rotated** K/V. `KnormSketch` norm-scoring is still valid (RoPE is orthogonal, norms preserved), but any consumer that assumes contiguous absolute positions must account for DCA's cyclic-rotated keys.
