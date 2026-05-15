# Long-Context Baseline — Working Doc

As of 2026-05-15.

**Scope:** define hardware, model, context, and evaluation constraints for a
**non-subquadratic** long-context baseline. Custom subquadratic attention work
comes *after* this baseline is built and understood.

**Working approach (incremental):** we answer only the **Step 1** questions now —
the decisions that block every other decision. **Step 2** questions depend on
Step 1 answers and are not worth debating yet. The **Parked** list holds
later-phase questions so nothing is lost, but we deliberately do not answer them
until their step arrives. The goal is to take a first concrete step, learn from
it, then re-open the next batch — not to answer everything up front.

## The Three-Way Tradeoff (read this first)

Every long-context attention design solves one problem under one constraint.
Three properties matter:

- **Efficient (subquadratic)** — compute grows slower than n² as context grows
  (ideally linear). Without it, 1M+ tokens is unaffordable.
- **Content-routed** — the model picks *which* tokens to attend to by *meaning*,
  not by a fixed positional rule. A fixed rule ("attend to the last 4K tokens")
  decides where to look *before* knowing what it needs, and misses anything
  outside that window.
- **Exact recall from any position** — a specific fact stays individually
  recoverable however far back it sits. The enemy is compression: squeeze the
  past into a fixed-size state and old facts blur together ("feature collision").

**The constraint: you can have two, not three.** Efficiency forces you to *do
less* — keep less (compress state → lose exact recall) or compare less via a
fixed rule (→ lose content-routing).

| Attention family | Efficient | Content-routed | Exact recall | Sacrifice |
|---|:--:|:--:|:--:|---|
| Full attention — incl. MLA, GQA, iRoPE | ✗ | ✓ | ✓ | pays n² + full KV cache |
| Pure linear / state-space (Mamba-style) | ✓ | ✓ | ✗ | fixed state blurs old facts |
| Sliding-window / fixed-sparse | ✓ | ✗ | ✓¹ | blind outside the window |
| Hybrid (Qwen DeltaNet+attn, MiniMax Lightning) | ~ | ✓ | ✓ | still n² in the few full-attn layers |
| Learned sparse (SubQ SSA) | ✓ | ✓ | ✓² | recall holds only if the selector picks right |

¹ exact only *within* the window. ² conditional on a content-aware selector
being both cheap and accurate — the unproven part.

**Example — one fact in a haystack.** A 2M-token context contains, at token
40,000, the line `API_TIMEOUT = 30`. At the end you ask: *"what's the timeout?"*

- **Full attention** — kept every token; finds token 40,000 exactly →
  **correct**, but paid n² compute and a huge KV cache.
- **Pure linear / state-space** — compressed 2M tokens into a fixed state; `30`
  blurred with a million other tokens → **wrong / fuzzy**.
- **Sliding-window** — only looked at the last 4K tokens; token 40,000 was never
  in view → **fails outright**.
- **Hybrid** — cheap linear layers carry an approximate summary; a periodic
  full-attention layer pulls token 40,000 exactly → **correct, at ~¼ the cost**.
- **Learned sparse (SSA)** — a selector tries to pick the block holding token
  40,000. Picks it → exact answer, cheap. Misses it → the needle is gone.

**Why this is the lens for the matrix below.** The "Attention family" column is
really telling you *which two properties a model bought and which one it
sacrificed*. The open models split into **full-attention** (exact but expensive)
and **hybrid** (cheap, with full-attention layers as a recall backstop). SubQ's
entire pitch is that SSA gets all three at once — by moving the problem into a
content-aware selector. That selector is the unproven part, and reproducing or
beating it is the actual task.

## Candidate Models — Baseline Matrix

Wider comparison incl. architecture and benchmark scores. `n/p` = not published on that exact
benchmark. Cross-model scores are **indicative, not apples-to-apples** (different
harnesses/scaffolds, RULER "@128K" vs "average", MRCR needle configs differ).

| Model | Weights / License | Total / Active | Context (native→max) | Attention family | RULER @128K | MRCR v2 8-needle @1M | SWE-Bench Verified |
|---|---|---|---|---|---|---|---|
| DeepSeek V4-Flash | Open · MIT | 284B / 13B | 1M | MoE · MLA lineage | n/p | ~49% | 79.0 |
| DeepSeek V4-Pro | Open · MIT | 1.6T / 49B | 1M | MoE · MLA lineage | n/p | ~59% ¹ | 80.6 |
| Llama 4 Scout | Open · Meta custom | 109B / 17B | ~256K → 10M | MoE · iRoPE | n/p | n/p | ~68% ² |
| Qwen3-Next-80B-A3B | Open · Apache-2.0 | 80B / 3B | 262K → ~1M | Hybrid · Gated DeltaNet + full attn (3:1) | 91.8% (avg) ³ | n/p | base n/p; Coder-Next >70% |
| Qwen3-Coder-Next | Open · Apache-2.0 | 80B / 3B | 262K native; ~1M unverified | Hybrid · Gated DeltaNet + full attn (3:1) | n/p | n/p | >70% reported |
| Qwen3.5-397B-A17B | Open · Apache-2.0 | 397B / 17B (512 exp.) | 262K → 1M | Hybrid · Gated DeltaNet + full attn (3:1) | n/p | n/p | 76.2 |
| MiniMax-Text-01 | Open · MiniMax license **(verify)** | 456B / 45.9B | 1M train → 4M infer | Hybrid · Lightning Attention (linear), 7:1 | ~94.7% ⁴ | n/p | n/p |
| MiniMax M2.5 | Open · Modified MIT | ~229–230B / ~10B | 196K config | MoE · GQA full attention | n/p | n/p | agent/coding focus; verify |
| Kimi K2.6 | Open · Modified MIT | ~1T / 32B (384 exp.) | 256K (262K) | MoE · MLA | n/p | n/p | **80.2** |
| GLM-5.1 | Open · MIT | ~744–754B / 40B | 200K | MoE · full attention | n/p | n/p | n/p ⁵ |
| **SubQ (1M-Preview)** — *target, closed* | Closed · no weights | Undisclosed | 1M → 12M | Subquadratic Sparse Attention | **95.6%** | **86.2 latest Appen / 65.9 earlier prod** | **81.8** |

Footnotes:
1. DeepSeek V4-Pro MRCR is cited ~59% (Pro-Max) to ~66% depending on config/harness.
2. Llama 4 Scout SWE-Bench is an approximate secondary-source figure, not official.
3. Qwen3-Next's 91.8% is a RULER **average across lengths**, not the @128K point.
4. MiniMax-Text-01 holds RULER 0.947→0.910 across the 128K–1M band; ~94.7% is the 128K end.
5. GLM-5.1 publishes SWE-Bench **Pro** 58.4 (claimed SOTA); a clean SWE-Bench *Verified* number for 5.1 was not found. Predecessor GLM-5 reported 77.8 on Verified.

Notes:
- **DeepSeek V4-Pro** is not a RoPE-only long-context model. Its relevant
  mechanism is the CSA/HCA compressed-attention design plus compressed KV path.
- **Llama 4 Scout** is the closest open-weight "extreme context" probe because
  Meta claims 10M context, but it was trained/post-trained around 256K and
  length-generalized via iRoPE. Treat the 10M claim as something to validate,
  not as a finished baseline.
- **Qwen3-Next / Qwen3-Coder-Next** use the same broad `qwen3_next` family:
  48 layers, high-sparsity MoE, and a 3:1 pattern of Gated DeltaNet layers to
  Gated Attention layers. Qwen validates the Instruct model to ~1M with YaRN;
  the Coder-Next 1M path should be treated as an experiment, not assumed.
- **MiniMax** appears twice on purpose. `MiniMax M2.5` is the agentic/coding
  model — ~196K config context, full GQA attention, `rope_theta=5000000`.
  `MiniMax-Text-01` is the long-context one — Lightning/linear attention and
  4M inference context. For a long-context baseline, **Text-01 is the relevant
  MiniMax**.
- Architecturally, the open models split two ways: **hybrid linear-attention**
  (Qwen3-Next, Qwen3.5-397B, MiniMax-Text-01) vs **full-attention MoE**
  (DeepSeek V4, Llama 4 Scout, Kimi K2.6, GLM-5.1). The hybrids are the closer
  analogs to SubQ's design and the cheaper starting point for later
  subquadratic work.
- Deployment weight class: DeepSeek V4-Pro / Kimi K2.6 / GLM-5.1 are
  multi-node-class (8× B200-tier); Qwen3-Next-80B-A3B is the lightest to run
  (3B active).
- SubQ's MRCR number needs version clarity. Launch coverage discussed an older
  production number around 65.9, while the later Appen brief reports 86.2 on
  the latest evaluated release.

Reference links to keep close:
[DeepSeek V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro),
[DeepSeek V4 docs](https://huggingface.co/docs/transformers/model_doc/deepseek_v4),
[Llama 4 Scout](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E),
[Meta Llama 4 blog](https://ai.meta.com/blog/llama-4-multimodal-intelligence/),
[Qwen3-Next](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct),
[Qwen3-Coder-Next](https://huggingface.co/Qwen/Qwen3-Coder-Next),
[MiniMax M2.5](https://huggingface.co/MiniMaxAI/MiniMax-M2.5),
[MiniMax-M2 docs](https://huggingface.co/docs/transformers/model_doc/minimax_m2),
[Appen SubQ benchmark](https://www.appen.com/whitepapers/benchmarking-subquadratics-latest-model-ssa-kernel).

---

## Step 1 — Decide now (blocks everything else)

1. **Which SubQ axis are we targeting first** — context length, cost, speed, or
   quality? The first baseline should be judged on **one** axis; pick it.
2. **First milestone context length.** What is the smallest *still-meaningful*
   length we get working end-to-end first — recommend **128K or 256K** to prove
   the pipeline. 1M / 4M / 12M are later milestones, not the first.
3. **Retrieval or reasoning.** Does the first milestone need real long-context
   *reasoning*, or only retrieval / passkey behavior? This decides which eval we
   run first.
4. **Hardware in hand.** What GPUs can we actually use *this week* — count,
   model, VRAM, interconnect (NVLink vs PCIe)? This is the real constraint;
   "what's ideal" is not the question.
5. **De-risk small first.** Should experiment #1 run on a small model (8B–30B)
   purely to validate the whole flow — load, long-context inference, eval
   harness, VRAM measurement — before spending GPU budget on a 200B+ base?
6. **Model shortlist + license gate.** Which 2 models do we pull and run first?
   And: do we need a fully permissive license (Apache/MIT)? If yes, Llama 4
   Scout's custom community license is out for commercial serving.
7. **Experiment #1 definition.** State it in one sentence with one pass/fail
   number — e.g. "Run model X at 256K on our node; pass = RULER retrieval ≥ N%
   AND peak VRAM fits AND decode ≥ M tok/s."

## Step 2 — After the base model, hardware, and first milestone are fixed

1. **KV-cache math.** What is the KV-cache size per token for the chosen model,
   and does the first-milestone context fit in HBM on our hardware — or do we
   need KV quantization / CPU-NVMe offload?
2. **Position-mechanism fork.** Is the chosen base a *positional-extrapolation*
   model (YaRN-extended, e.g. Qwen3-Next) or an *architecturally-long* model
   (iRoPE, compressed attention, hybrid linear)? This decides whether the first
   change is a config tweak or a serving-stack validation.
3. **RoPE state.** What `rope_theta` / scaling is baked into the checkpoint, and
   does extrapolation alone reach the milestone, or is continued long-context
   training needed?
4. **Serving stack.** Does vLLM / SGLang / TensorRT-LLM / Transformers support
   the chosen model at the milestone length *today*?
5. **Quantization.** If we quantize to fit, does it hold RULER/MRCR at long
   context?
6. **Short-context regression.** If we change position scaling, how do we
   measure (and bound) the regression on shorter inputs?

## Position Scaling — Hidden Constraints

For Qwen-style 262K native context, the raw arithmetic is simple but misleading:

- `factor=2` -> ~524K tokens.
- `factor=4` -> ~1.05M tokens. This is the level Qwen has publicly validated
  for Qwen3-Next-Instruct.
- `factor=8` -> ~2.1M tokens.
- `factor=16` -> ~4.2M tokens.
- `factor=32` -> ~8.4M tokens.
- `factor≈46` -> ~12M tokens.

The hidden issue: YaRN/RoPE scaling changes position interpretation, not the
attention or KV memory problem. Large factors can make the server accept longer
inputs while the model's retrieval, ordering, and reasoning quality collapse.
Static YaRN can also hurt shorter prompts because the same scaling is applied
even when the prompt is not long.

So the position-scaling ladder should be experimental:

1. Native context first: 196K / 256K / 262K depending on model.
2. `factor=2` or equivalent next.
3. `factor=4` only after native and `factor=2` pass.
4. Higher factors only with explicit quality gates, not just "server did not
   crash."

Track at each rung: peak HBM, CPU RAM if offloaded, prefill time, decode tok/s,
repetition/degeneration, RULER, MRCR, and a real codebase/document QA eval.

## KV Offload — Useful but Not Magic

CPU KV offload is worth testing, but it converts the bottleneck from HBM
capacity into PCIe/NVLink/host-memory bandwidth and scheduling.

Rough Qwen3-Next-style KV order of magnitude:

- 1M tokens: ~25 GB KV.
- 4M tokens: ~100 GB KV.
- 8M tokens: ~200 GB KV.
- 12M tokens: ~300 GB KV.

This can fit in CPU RAM on a serious box, but decode may become slow if every
new token needs broad access to old KV blocks. Offload is most useful for:

- long static contexts reused across many queries;
- prefix caching / shared repo or document prefixes;
- offline and batch runs where latency is secondary;
- experiments that would otherwise crash from HBM pressure.

It is weak for:

- one cold 12M-token prompt followed by one answer;
- low-latency interactive serving;
- high-concurrency serving;
- dense attention over all previous tokens at every decode step.

Tools to test, in likely order:

- **LMCache** for CPU RAM / disk / remote KV cache with vLLM/SGLang.
- **vLLM KV offloading connector** for explicit GPU/CPU KV movement.
- **NVIDIA Dynamo + LMCache** if we need a more production-like cache layer.
- **TensorRT-LLM KV offloading** if we move into NVIDIA's optimized stack.

Experiment rule: KV offload can help prove feasibility, but it should not be
counted as a SubQ-like cost/speed result unless latency and throughput are
measured end-to-end.

## Multi-GPU Serving — Qwen3-Next 1M Starting Point

For Qwen3-Next at ~1M, the cleanest first serving experiment is **vLLM tensor
parallelism + decode context parallelism**, plus YaRN via HF config overrides.

Parallelism terms:

- **TP / tensor parallelism:** splits model layers and attention heads across
  GPUs. This is the first lever for fitting/running the model.
- **DCP / decode context parallelism:** shards KV cache along the context
  dimension during decode. This helps long-context KV memory pressure.
- **PCP / prefill context parallelism:** splits long prompt prefill. This is
  relevant for TTFT, but support is still more model/backend-specific.
- **DP / data parallelism:** duplicates the whole serving replica for
  throughput. It does not help one giant prompt fit.

Important vLLM version note: recent vLLM docs say the older `--rope-scaling`
flag is no longer supported; use `--hf-overrides` with `rope_parameters`.

Qwen3-Next has 2 KV heads in Gated Attention. vLLM guidance for DCP is roughly:

```text
1 <= dcp <= tp_size / kv_heads
```

So:

```text
tp=4 -> max useful dcp ~= 2
tp=8 -> max useful dcp ~= 4
tp=16 -> max useful dcp ~= 8
```

### vLLM 4-GPU Example

```bash
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 1010000 \
  --tensor-parallel-size 4 \
  --decode-context-parallel-size 2 \
  --hf-overrides '{"rope_parameters":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":262144}}' \
  --enable-chunked-prefill \
  --max-num-batched-tokens 131072
```

### vLLM 8-GPU Example

```bash
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 1010000 \
  --tensor-parallel-size 8 \
  --decode-context-parallel-size 4 \
  --hf-overrides '{"rope_parameters":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":262144}}' \
  --enable-chunked-prefill \
  --max-num-batched-tokens 131072
```

### vLLM Multi-Node Skeleton

Node 0:

```bash
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --nnodes 2 \
  --node-rank 0 \
  --master-addr NODE0_IP \
  --master-port 29501 \
  --tensor-parallel-size 16 \
  --decode-context-parallel-size 8 \
  --max-model-len 1010000 \
  --hf-overrides '{"rope_parameters":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":262144}}'
```

Node 1 uses the same command with `--node-rank 1`.

### SGLang Starting Point

For Qwen3-Next, treat SGLang as a second experiment after vLLM. Useful knobs:

- `--tp-size`
- `--context-length`
- `--chunked-prefill-size`
- `--json-model-override-args`
- `--pp-size` for pipeline parallelism
- `--attn-cp-size` for attention context parallelism where supported

Skeleton:

```bash
SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-Next-80B-A3B-Instruct \
  --tp-size 8 \
  --context-length 1010000 \
  --chunked-prefill-size 131072 \
  --json-model-override-args '{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":262144}}'
```

SGLang has explicit long-sequence prefill context-parallel flags documented for
DeepSeek V3.2 NSA/DSA, for example:

```bash
--enable-nsa-prefill-context-parallel
--attn-cp-size 4
--nsa-prefill-cp-mode round-robin-split
```

Do not assume those Qwen3-Next paths work without testing; treat them as
backend-specific.

Serving experiment success criteria:

- server starts cleanly;
- can ingest native 262K, then YaRN 512K, then YaRN ~1M;
- no silent truncation;
- peak HBM and CPU RAM are logged;
- prefill time and decode tok/s are logged;
- RULER/MRCR or a private retrieval eval passes at each rung.

## Context Parallelism vs REFRAG vs SubQ-Like SSA

These are different levers:

```text
REFRAG = reduce/compress what the model sees.
Context parallelism = distribute a long sequence across GPUs.
SubQ-like SSA = change attention itself.
```

### Where REFRAG Can Fail

- **Wrong chunk selection:** if the selector misses a crucial chunk, the
  decoder never sees the exact evidence.
- **Loss of exact details:** compressed embeddings may preserve gist but lose
  identifiers, numbers, clauses, function signatures, or small constraints.
- **Cross-reference chains:** if the answer depends on many small facts spread
  across distant chunks, REFRAG may miss some or need to expand too many chunks.
- **Not true native attention:** REFRAG is an external compression/selection
  layer before the decoder; SubQ claims model-internal sparse attention.
- **RAG-shaped assumption:** REFRAG works best when context is retrieved
  passages with sparse relevance. It may be weaker on full codebases, long
  conversations, traces, or books where relevance is structural.
- **Compression training burden:** a normal Qwen model cannot automatically
  consume arbitrary chunk embeddings. The encoder/projection/decoder interface
  must be trained or aligned.
- **Expansion policy errors:** selective expansion introduces a second model or
  policy whose mistakes produce incomplete but plausible answers.
- **Conditional scaling:** if many chunks must be expanded, effective context
  shrinks and latency rises.

### Why Context Parallelism Helps but Does Not Solve 12M

Context parallelism can split a long sequence across GPUs:

```text
12M tokens / 16 GPUs = 750K tokens per GPU
```

This helps capacity and prefill parallelism, but:

- it does not reduce total dense-attention work; it distributes it;
- GPU communication becomes expensive;
- KV cache still exists, just sharded;
- decode may still be slow because each new token needs access to distributed
  old KV;
- it does not fix RoPE/YaRN extrapolation failure;
- it does not make Qwen understand 12M if the model was only trained/validated
  around 1M.

Best practical stance: use context parallelism and KV sharding to push serving
limits, use REFRAG-style compression for effective raw-corpus coverage, and
reserve SubQ-like SSA for the architecture-research phase.

## Long-Context Data / Training Assets

Public data exists, but most directly usable datasets are 64K-128K class, not
true 1M-12M reasoning data. Treat them as ingredients, not the full recipe.

Useful public assets:

- **LongAlign-10k**: long instruction data, roughly 8K-64K. Good for alignment
  and data-format bootstrapping.
- **LongMIT-128K**: multi-hop long-context QA, useful for 128K-style training
  and evaluation.
- **LongAlpaca / LongQA**: older long-instruction/QA assets from LongLoRA-era
  work; still useful as low-cost baselines.
- **LongWriter-6k**: useful for long-output behavior, not mainly long-input
  reasoning.
- **ChatQA2-Long-SFT-data**: long-context / RAG SFT data; license and usage
  terms need verification before commercial use.
- **LongCodeBench**: better as evaluation for long code-context behavior than
  as primary training data.

Likely training recipe for 1M+ awareness:

1. Start with a model whose architecture already tolerates long context.
2. Progressive long-context continued pretraining: 128K -> 256K -> 512K -> 1M.
3. Mix real long documents/codebases with synthetic retrieval, multi-hop, and
   ordering tasks.
4. Keep short-context data in the mixture to reduce regression.
5. Use long-context SFT only after continued pretraining has made the model
   stable at the target length.
6. Evaluate every length rung; do not wait until 1M/4M to discover collapse.

The Qwen2.5-1M recipe is the closest public template: long-data synthesis,
progressive pretraining, multi-stage SFT, length extrapolation, sparse/chunked
attention, and chunked prefill. For our work, public datasets alone are not
enough; we will probably need synthetic data generation and private workload
evals.

## Parked — later phases (do not answer yet)

Kept so the questions are not lost; do not spend effort here until the step arrives.

- **Fine-tuning / continued pretraining:** instruction-only vs continued
  pretraining; real long docs/codebases vs synthetic needle data (overfit
  risk); training sequence-length schedule (128K→1M); optimizer-state and
  checkpoint cost; LoRA/QLoRA vs full-parameter.
- **Evaluation at scale:** freezing exact harness versions; full 128K–12M
  benchmark matrix; MRCR needle-count configs (1/4/8); private real-workload
  eval (full-repo bugfix, legal-corpus QA, multi-doc agent memory); success
  bar vs GPT/Claude/Gemini and vs SubQ.
- **12M-specific:** long-context training recipe; prefill-speed and decode
  targets at 1M / 12M; multi-node inference; sustained KV offload.
- **Subquadratic attention (the eventual goal):** attention architecture
  choice; custom CUDA/Triton kernels; reproducing SubQ-style cost claims
  without custom kernels.
- **Availability-gap research:** is there any public open-weight model with an
  independently validated 12M context; any public 12M long-context training
  recipe; any open benchmark that stresses 12M *reasoning* (not just retrieval).
