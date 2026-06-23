# Prism-Test — Project Progress

*Last updated: 2026-06-22. Adds Mamba-attention hybrid (NemotronH) KV-compression
support and refreshes the architecture description to the current **three-door**
model. Note: earlier revisions called Door 2 "prefill methods"
(`eval_harness/prefill_methods/`) and Door 3 "sketches"; those now live in
`eval_harness/attention_methods/` and `eval_harness/kv_compression/`. The
faithful DCA/ReAttention narrative below is unchanged in substance.*

## Goal

Enable research on custom **prefill**, **KV-cache compression**, and **decode
attention** methods — letting them run and be tested in an error-free way across
standard models *and* hybrid (Mamba-attention) models, on the long-context
benchmark suite (RULER, LOFT, LongBench, InfiniteBench, GSM-Infinite, …).

## Where the project stands

Prism-Test is a long-context evaluation harness with four interchangeable
backends (`vllm`, `hf`, `rag`, `research`). The `research` backend is the
extension surface: it wires three independent, optional **doors** into a shared
generation pipeline (nested context managers, outer → inner):

- **Door 1 — positional** (`positional_methods/`): RoPE frequency/position
  remap — `yarn`, `ntk`, `linear_pi`.
- **Door 2 — attention** (`attention_methods/`): how attention scores positions
  — `dca`, and the faithful `reattention` / `reattention_exact` (legacy
  `PrefillMethod` subclasses on the same slot).
- **Door 3 — KV compression** (`kv_compression/`): what stays in the cache —
  ~36 compressors (mostly faithful kvpress 0.5.1 ports), auto-discovered from
  `kv_compression/compressors/` via `@register_kv_compressor`.

so inference-time long-context techniques can be benchmarked under one roof
against clean baselines.

### Context-extension methods (Door 2, `eval_harness/attention_methods/`)

Three methods are implemented, registered, and verified against their
reference implementations:

| Method | Mechanism | What it measures |
|---|---|---|
| `reattention` (hook port) | Post-attention forward hook; prunes the decode-facing cache to `[global \| top-k middle \| local]`. Prefill attention stays exact. | The ReAttention *retention policy* in isolation (pruning, not extension — position-OOD beyond the native window). |
| `reattention_exact` | Full `self_attn.forward` replacement reproducing the original ReAttention *computation*: raw-KV cache, chunked prefill, pre-attention recall per chunk and per decode step, RoPE after selection. `pe_original: false` re-rotates the recall view contiguously — the published-eval setting that actually extends context. | The paper's method, end to end. |
| `dca` (ChunkLlama) | Full `self_attn.forward` replacement, active across prefill *and* decode: 3-component dual-chunk attention (intra / successive / inter) with cyclic key positions and LSE merge. | Training-free length extrapolation by attention re-positioning. |

Faithfulness is pinned by in-suite oracles: verbatim transcriptions of the
upstream ChunkLlama (`chunkllama_attn_replace.py`) and ReAttention
(`RECacheV2.update`) reference code run against our ports inside the test
suite, so any future drift fails CI. Known, deliberate deviations (fp32
matmuls, GQA head-pairing fix, absolute-position causal masks) are documented
in the module docstrings.

Two notable correctness fixes made during verification:

- **Ragged-cache decode** (hook port): per-layer pruning left layers with
  different cache lengths; HF decode sizes one causal mask from layer 0, which
  either crashes or silently slices the mask (causality leak).
  `uniform_retained` (default on) equalizes per-layer retention.
- **DCA multi-token decode NaN**: the LSE decomposition assigned every query in
  a block the chunk components of the *last* key — a question block straddling
  a chunk boundary produced empty softmax rows (NaN → silent token-0 argmax)
  and leaked future keys. Multi-token blocks now use the reference's
  concat-scores + single fp32 softmax decode branch.

### KV-cache compression (Door 3) & hybrid-model support

Door 3 installs a compressor as a post-attention forward hook on the
**full-softmax attention layers only**, rewriting that layer's cached K/V. The
roster is ~36 baselines (knorm, ridge, snapkv, pyramidkv, tova, finch, adakv,
duo_attention, kvzip, …); see `available_kv_compressors()` and CLAUDE.md for the
full list and per-method quirks.

**Mixed-attention models** are handled by hooking only the layers that carry a
standard K/V cache and skipping the rest:

| Family | Non-cache layers | How they're detected |
|---|---|---|
| Qwen3.5 / Qwen3-Next | DeltaNet linear-attention | `layer.layer_type` / config `layer_types` contains `linear` |
| Gemma3 | sliding-window (still cached) | `config.sliding_window` — *not* skipped (it has K/V) |
| **NemotronH** (new) | Mamba2 + MLP | `block.block_type` / `config.layers_block_type` / `hybrid_override_pattern` |

**NemotronH (`nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`) — Mamba2-attention hybrid.**
The 4B model is 42 layers with only **four** full-attention layers
(`hybrid_override_pattern` positions 12/17/24/32); the rest are Mamba2/MLP and
hold no K/V cache. Differences from Llama-style models, and how the framework
adapts:

- Attention lives under `block.mixer` (no `block.self_attn`) — the install loop
  resolves it via `_resolve_attention_module` (base.py) and hooks only the four
  attention mixers.
- Blocks are typed by `block_type` / `config.layers_block_type` — added to
  `_is_non_full_attention_layer` (skip mamba/mlp) and to `_is_hybrid_model`
  (selects `HybridCacheAdapter`, which checkpoints/restores attention layers
  only).
- Attention applies **no RoPE** — the `rotary_emb` install assignment is guarded,
  and SnapKV/PyramidKV fall back to raw (un-rotated) queries when
  `position_embeddings` is absent (exactly what the model computes).
- `NemotronHForCausalLM` is registered in `SUPPORTED_MODELS`.

Verified by `test_nemotron_h_kv_compression.py` (faithful fake NemotronH; no
weights, no GPU): hooks land on exactly the four attention layers, and knorm /
ridge / snapkv / pyramidkv / compactor / expected_attention / keydiff each shrink
only those four caches (mamba/mlp slots untouched). `compactor` and
`expected_attention` reduce their RoPE step to identity on this no-RoPE model.
Config: `evaluate/evaluate_nemotron_kv.yaml`.

**End-to-end on H200 (RULER-16k, transformers 5.9 native `nemotron_h`, fast Mamba
kernels):** single-pass prefill at the full ~15.6k context works and retains
`(1-ratio)·prompt` on the attention layers (e.g. ratio 0.5 → ~7.8k retained,
`per_layer_min=0` for the Mamba/MLP layers). A 6-method × 4-ratio sweep + full-cache
baseline is in [`nemotron_ruler16k_kv_report.md`](nemotron_ruler16k_kv_report.md)
(baseline 92.7; robustness keydiff ≳ compactor > knorm > snapkv ≈ pyramidkv >
expected_attention). Two issues found and fixed getting there:

- **Chunked prefill ⇒ per-chunk over-eviction.** The post-prefill compressor's
  hook fires after *every* prefill chunk, so a ratio applied per 512-token chunk
  collapses geometrically (retained ≈ chunk size, not `(1-ratio)·prompt`). Use
  **single-pass** prefill (`prefill_chunk_size: null`, the default) for correct
  KV-compression semantics; chunking is only for the `streaming` schedule.
- **Mamba decode assumes `q_len==1`.** The pipeline forwarded the question as one
  multi-token block after the context prefill; NemotronH's cached Mamba forward
  does `hidden_states.squeeze(1)` and crashes (`weight must have shape
  (dim, width)`) on a multi-token block. `generate_answer` now feeds the question
  **token-by-token** for Mamba models (`_model_has_mamba_layers`) — identical to
  the block forward for the attention layers, required for the Mamba layers.

> **Runtime requirements.** CUDA GPU; a transformers build with the *native*
> `nemotron_h` architecture (the older trust_remote_code modeling file does not
> thread attention K/V through a hookable forward); single-pass 16k needs the fast
> Mamba kernels (`use_mamba_kernels` default True → `kernels-community/causal-conv1d`
> + `mamba-ssm`; the pure-torch SSD path OOMs), so run **online** (or pre-warm the
> kernel cache) for the publisher-trust check. PyramidKV's ragged cache decodes
> correctly only under `flash_attention_2`.

### Custom kernels (`eval_harness/kernels/`)

- Fused Triton einsum+top-k (+ bitonic merge) for ReAttention recall
  (top-1/top-4, head_dim 128, 128-aligned, fp16/bf16, CUDA; dense fallback
  with `auto`, fail-fast with `force`). Note the kernel is a block-max
  approximation of dense selection — paper runs should pin `force` so a
  mid-run fallback can't mix selection semantics.
- `flash_attn_with_lse` for DCA: real flash-attn when available, pure-torch
  CPU fallback (also used by the GPU-free tests). Triton and flash-attn are
  both optional dependencies — everything degrades gracefully without them.

### Ready-made configs (`evaluate/`)

Self-documenting configs, each verified to construct end-to-end through the
harness's own loading path (`EvalConfig` parse):

- `evaluate_vllm.yaml` / `evaluate_hf.yaml` — clean no-method baselines.
- `evaluate_kv.yaml` — KV-compression (Door 3) only (knorm 50%, Llama-3.1-8B).
- `evaluate_nemotron_kv.yaml` — **new**: KV compression on NemotronH (Mamba-attention
  hybrid); compresses the four full-attention layers only.
- `evaluate_positional.yaml` — Door 1 (YaRN / NTK / linear-PI).
- `evaluate_dca.yaml` — DCA with faithful ChunkLlama defaults
  (chunk_size 6144 / local_window 1024 for an 8K Llama-3, no PI).
- `evaluate_reattention.yaml` — the paper-faithful `reattention_exact` config
  (`recall_clip: 127` is load-bearing: 32 + 127·32 + 4096 = 8192 keeps the
  re-rotated view inside the native window).
- `evaluate_common.yaml` — the full research surface, including the
  attention-method × compressor compatibility matrix (`dca` and
  `reattention_exact` must keep the KV compressor at `none`; the `reattention`
  hook port composes with any compressor).

### Reproduction status (RULER-16k, Meta-Llama-3-8B base, fp/bf16, greedy)

The ReAttention paper numbers reproduce with `reattention_exact`:

- `niah_single_1`: **100.0** (200/200, clean EOS-terminated answers)
- `niah_single_3`: **96.0** (paper ≈ 94)
- `niah_multikey_3`: paper ≈ 15 (near-floor is the *correct* reproduction —
  the method genuinely fails there; don't chase higher)

The DCA baseline (`evaluate_dca.yaml`) is configured and verified; its RULER
run is the immediate next experiment.

### Test suite

**All 988 tests green** (11 environment-conditional skips), including:
reference-transcription oracles for both DCA and ReAttention, per-compressor
kvpress oracles (~36 compressors), reduce-to-dense baselines
(`recall_option: full_attn` is bitwise-identical to no-method), multi-layer
ragged-cache regressions, chunk-boundary straddle tests, hybrid-layer-detection
regressions (Qwen3.5 + NemotronH), and integration tests driving the real
pipeline on a tiny CPU Llama. The suite also passes with triton, flash-attn,
and vLLM all absent (verified by import-hiding simulation).

The NemotronH hybrid path is pinned by `tests/test_nemotron_h_kv_compression.py`
(24 tests): attention-only layer hooking (a direct hook-count assertion plus the
detection helpers), mamba/mlp skip, no-RoPE handling for every scorer, the
`_can_slice_attention_kv` None-guard on mamba cache slots, explicit-phase
decode no-op, and the token-by-token Mamba question feed.

### Environment

- **Python ≥ 3.10 is required** (`pyproject.toml`; the code uses
  `dataclasses.field(kw_only=...)`). Under 3.9 the entire `kv_compression`
  package fails to import (`field() got an unexpected keyword argument
  'kw_only'`), cascading to dozens of test-collection errors — those are an
  interpreter mismatch, not code bugs.
- Unit tests are **weight-free and CPU-only** (fake modules + `object.__new__`),
  so they run anywhere with torch + transformers installed. Real model
  evaluations need a CUDA GPU (and, for NemotronH, native `nemotron_h` support
  — see the KV-compression section).

### Hardening landed alongside

- A missing `--config_file` now raises `FileNotFoundError` (previously it
  silently ran a pure-default vllm/ruler32k eval to completion).
- `num_logits_to_keep` → `logits_to_keep` (transformers 5.x rename; the old
  kwarg was silently ignored, materializing full-vocab logits over every
  question block).
- scipy demoted from a hard import-time dependency (via benchmark autoload) to
  a lazy LOFT-scoring dependency.
- Per-method `output_dir`s in the configs: the auto-named run directory does
  not encode the method/sketch, so same-model configs previously collided.

## Where to take this further

### Near-term (experiments)

1. **Run the DCA baseline** on RULER-16k with `evaluate_dca.yaml` and compare
   against the ReAttention anchors above; then extend both to 32k/64k
   (`evaluate_reattention.yaml` documents the 64k local_size/chunk
   constraints).
2. **Method × compression studies** via `evaluate_common.yaml`: the hook port
   composes with any sketch (both are cache-pruning hooks), giving a clean
   extension-plus-compression axis on a 128k-native model.
3. **Prism-1M / longer benchmarks**: the harness already registers prism1m and
   gsm_infinite at 128k; nothing has been run beyond RULER yet.
4. **NemotronH real-run validation — DONE** on H200 (see
   [`nemotron_ruler16k_kv_report.md`](nemotron_ruler16k_kv_report.md)): 6 methods ×
   4 ratios + baseline on RULER-16k, single-pass, correct `(1-ratio)·prompt`
   retention on the 4 attention layers. Follow-ups: (a) investigate
   `expected_attention`'s weakness on this no-RoPE model (its averaged-RoPE score
   reduces to identity — likely ill-suited); (b) scale to the full 13-subset ×
   200-sample RULER; (c) extend to larger Nemotron Nano / Nano-2 hybrids (same
   `hybrid_override_pattern` machinery).

### Medium-term (framework)

5. **Validate Door 1 (positional) at length**: `yarn` / `ntk` / `linear_pi`
   are implemented and wired (the pipeline wraps `rotary_emb`;
   `compute_inv_freq` / `remap_position_ids` fire per rotary call) — but they
   have not been swept on long-context benchmarks. Run the positional door
   against the no-method baseline on RULER-32k/64k. (Door 1 is a no-op for
   NemotronH, which uses no RoPE — note this when composing doors.)
6. **Pre-attention sparsity + repositioning as a first-class frame**:
   `reattention_exact` (pe_original=false) is the architecturally-complete
   instance of the "select, then re-rotate contiguously, then attend" pattern.
   Generalizing that into a reusable `AttentionMethod` base would let new
   selection policies inherit the verified chunked-prefill/recall machinery
   instead of re-porting it.
7. **Decode-side selection for the `reattention` hook**: the hook no-ops on
   decode; a decode-time re-selection hook would close the gap to
   `reattention_exact` (`recall_option: whole`) at much lower memory cost.
8. **Mamba-state compression**: Door 3 only touches attention K/V. NemotronH-4B's
   non-attention memory is the 21 Mamba2 layers' conv/ssm state (the other 17
   non-attention layers are MLP and hold none) — a separate "Door 3.5" for
   SSM-state compression is the natural next hybrid axis.

### Housekeeping

9. The reposition mode on the `reattention` hook (`reposition: true`) is shipped
   but default-off and unexplored experimentally.
10. `test_identity_rope_equivalence.py` no longer tests identity-RoPE (it pins
    legacy CacheConfig field removal) — rename when convenient.
11. CI installs unpinned latest transformers; the lock pins 5.10.2 and the
    suite is verified on 5.9.0. Consider pinning a floor in CI to avoid
    surprise breakage from future transformers majors. **Pin a Python ≥ 3.10
    floor too** — under 3.9 the `kv_compression` package fails to import.
