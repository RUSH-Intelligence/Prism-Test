# Prism-Test — Project Progress

*Last updated: 2026-06-11. Supersedes the original prefill-methods progress
report (the approximate ReAttention/DCA ports it described have since been
replaced by faithful, reference-verified implementations).*

## Where the project stands

Prism-Test is a long-context evaluation harness with four interchangeable
backends (`vllm`, `hf`, `rag`, `research`). The `research` backend is the
extension surface: it wires **prefill methods** (context extension) and
**sketches** (KV-cache compression) into a shared generation pipeline, so
inference-time long-context techniques can be benchmarked under one roof
against clean baselines.

### Context-extension methods (`eval_harness/prefill_methods/`)

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

Six self-documenting configs, each verified to construct end-to-end through
the harness's own loading path:

- `evaluate_vllm.yaml` / `evaluate_hf.yaml` — clean no-method baselines.
- `evaluate_kv.yaml` — KV-compression sketch only (knorm 50%).
- `evaluate_dca.yaml` — DCA with faithful ChunkLlama defaults
  (chunk_size 6144 / local_window 1024 for an 8K Llama-3, no PI).
- `evaluate_reattention.yaml` — the paper-faithful `reattention_exact` config
  (`recall_clip: 127` is load-bearing: 32 + 127·32 + 4096 = 8192 keeps the
  re-rotated view inside the native window).
- `evaluate_common.yaml` — the full research surface, including the
  prefill-method × sketch compatibility matrix (`dca` and `reattention_exact`
  must keep `sketch_name: none`; the hook port composes with any sketch).

### Reproduction status (RULER-16k, Meta-Llama-3-8B base, fp/bf16, greedy)

The ReAttention paper numbers reproduce with `reattention_exact`:

- `niah_single_1`: **100.0** (200/200, clean EOS-terminated answers)
- `niah_single_3`: **96.0** (paper ≈ 94)
- `niah_multikey_3`: paper ≈ 15 (near-floor is the *correct* reproduction —
  the method genuinely fails there; don't chase higher)

The DCA baseline (`evaluate_dca.yaml`) is configured and verified; its RULER
run is the immediate next experiment.

### Test suite

**193 tests, all green** (2 environment-conditional skips), including:
reference-transcription oracles for both DCA and ReAttention, reduce-to-dense
baselines (`recall_option: full_attn` is bitwise-identical to no-method),
multi-layer ragged-cache regressions, chunk-boundary straddle tests, and
integration tests driving the real pipeline on a tiny CPU Llama. The suite
also passes in a CI-like environment with triton, flash-attn, and vLLM all
absent (verified by import-hiding simulation).

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

### Medium-term (framework)

4. **Tier-1 RoPE methods (NTK, YaRN, linear PI)**: `compute_inv_freq` exists
   on `PrefillMethod` but nothing calls it — implementing these needs a
   RoPE-level interceptor in the pipeline. This is the single biggest missing
   piece of the original design.
5. **Pre-attention sparsity + repositioning as a first-class frame**:
   `reattention_exact` (pe_original=false) is the architecturally-complete
   instance of the "select, then re-rotate contiguously, then attend" pattern.
   Generalizing that into a reusable base (the way `PrefillMethod` generalizes
   post-attention pruning) would let new selection policies inherit the
   verified chunked-prefill/recall machinery instead of re-porting it.
6. **Attention-plugin seam** (`eval_harness/attention/`): a registry +
   Protocol for swappable attention implementations exists but is wired into
   nothing (only its own tests). Decide whether to grow it into the Tier-1
   interceptor above or drop it.
7. **Decode-side selection for the hook port**: the hook no-ops on decode;
   a decode-time re-selection hook would close the gap to `reattention_exact`
   (`recall_option: whole`) at much lower memory cost.

### Housekeeping

8. The reposition mode on the hook port (`reposition: true`) is shipped but
   default-off and unexplored experimentally.
9. `test_identity_rope_equivalence.py` no longer tests identity-RoPE (it pins
   legacy CacheConfig field removal) — rename when convenient.
10. CI installs unpinned latest transformers; the lock pins 5.10.2 and the
    suite is verified on 5.9.0. Consider pinning a floor in CI to avoid
    surprise breakage from future transformers majors.
