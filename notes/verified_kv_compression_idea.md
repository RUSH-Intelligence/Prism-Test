# Verified KV Compression — Idea and Methodology

Inspired by [vAttention: Verified Sparse Attention (arXiv:2510.05688)](https://arxiv.org/abs/2510.05688). The pivot is away from "beat the compactor on accuracy" and toward a new axis: **provable per-prompt error bounds on the compressed cache**.

## The problem with current KV compression

Every method in `eval_harness/kv_compression/compressors/` (Ridge, Compactor, knorm, SnapKV, PyramidKV, ...) is doing the same thing at heart:

1. Score every token with some heuristic.
2. Keep the top-k highest-scoring tokens.
3. Evict the rest.

At decode time, attention runs over the retained set only. This is fast, but:

- The evicted tokens are gone forever.
- We have **no idea** how much any specific decode query would have cared about them.
- Reported quality is empirical / benchmark-averaged — no per-prompt guarantee.

Nobody in the KV compression space offers a mathematical bound of the form "our compressed output is within ε of the true output with probability ≥ 1−δ." That's the gap.

## The vAttention insight (in one paragraph)

Attention is a weighted average of value vectors: `out = Σ softmax(q·k_i) · v_i`. Top-k works when the softmax is spiky (a few tokens dominate). Random sampling (with 1/probability rescaling) gives an *unbiased* estimate of a weighted average, and concentration inequalities (Hoeffding / Bernstein) tell you how tight that estimate is for a given sample count. vAttention combines both: exact contribution from top-k heavy hitters + unbiased estimate over a random tail from the rest → a certified (ε, δ) approximation of full attention, per query per head.

vAttention operates at **sparse attention time**: all K/V are still in the cache and it peeks at the full softmax to decide how many samples to draw. **KV compression can't peek** — the evicted tokens are physically gone. So the trick has to be adapted.

## The methodology (verified KV compression)

Same total memory budget as any existing top-k compressor. The change is only in *what* fills the retained slots.

### At compression time (after prefill)

Given a compression ratio r (keep fraction 1−r of N prompt tokens = M slots total):

1. Score all N tokens using an **existing compressor method** (Ridge, Compactor, knorm, ...). This is the deterministic head.
2. Keep the top **M − s** highest-scoring tokens as usual.
3. From the remaining `N − (M − s)` evicted candidates, uniformly randomly sample **s** tokens. Tag each with a scale factor equal to `(N − (M − s)) / s` — this is the Horvitz-Thompson weight that makes their contribution unbiased.

Cache still holds M tokens total. Composition: `(M − s)` deterministic top-k + `s` random ambassadors carrying scale factors.

### At decode time

Standard flow — no change to the query path:

- One query `q`.
- Score against all M retained keys.
- Softmax over the M scores.
- Compute `out = (exact contribution from the top-(M−s) tokens) + (Σ scale_i · softmax_i · v_i over the s random ambassadors)`.

Each ambassador's contribution is scaled up by its 1/probability factor before being added into the output.

### The bound

Because the s ambassadors are drawn uniformly at random from the evicted set, their scaled sum is an **unbiased estimator** of the true evicted-set contribution. Hoeffding's inequality (or Bernstein for tighter bounds when variance is small) gives:

> P(‖compressed_out − true_out‖ ≤ ε) ≥ 1 − δ

where the required sample count `s` for a target (ε, δ) is a closed-form function of ε, δ, and the value-vector norm range.

Users get a knob:

| Mode | Split (top / random) | Bound |
|------|----------------------|-------|
| Fast | 9 / 1 | Loose |
| Verified | 6 / 4 | Tight |
| Certified | 4 / 6 | Ironclad |

No existing KV method offers a *quality-guarantee* knob — only a memory knob.

## Concrete toy example

100-token context, ratio 0.9 (keep 10). Split 8 + 2:

**Compression:**
- Ridge scores all 100 tokens.
- Keep top 8 by Ridge score.
- Uniformly sample 2 from the remaining 92. Scale factor per ambassador = 92 / 2 = 46.

**Decode:**
- Query attends over all 10 retained tokens.
- Softmax normally.
- Contribution of the 8 deterministic tokens: as-is.
- Contribution of the 2 random tokens: each multiplied by 46 before adding to `out`.

Result: an unbiased estimate of the full 100-token attention, with a Hoeffding bound on the error.

## Why this is a paper, not just another baseline

- **New axis.** Every KV compression paper competes on accuracy-vs-ratio. This paper adds *provable* as a third axis.
- **Method-agnostic.** Turns any top-k compressor into "verified-X." Broad claim, easy to demonstrate across the roster (Ridge, Compactor, knorm, SnapKV, ...).
- **Same memory, same decode speed.** The overhead is entirely at compression time (scoring + sampling) and a tiny extra per-decode compute (M unchanged).
- **Transfer story is clean.** vAttention proved the math works at sparse attention time; this is the KV-cache analogue. Same framework, different regime (persistent eviction, prefill-time commitment, long-context RULER/LongBench workloads).

## Open design decisions (flagged for later, not blockers)

1. **Budget split.** How to divide M between deterministic top-k and random ambassadors. 80/20 is a reasonable starting sweep; expect a per-benchmark sweet spot.
2. **Uniform vs. weighted random sampling.** Uniform is simplest and gives standard Horvitz-Thompson. Sampling with probability proportional to compressor score (importance sampling) reduces variance → tighter bound at same sample count. Same math, better constant.
3. **Which bound.**
   - Per-token output bound: `‖compressed_out − true_out‖ ≤ ε` (vAttention's flavor — strongest, hardest).
   - Downstream logit-KL bound: `KL(compressed_logits ‖ true_logits) ≤ ε` (ties more directly to generation quality, may sell better as "verified generation").
4. **Denominator handling.** Full attention's softmax denominator over evicted tokens is also unknown. Standard fix: estimate it from the same random sample → ratio estimator (biased in finite samples but consistent; there are standard bounds). Worth reading vAttention Sec 3–4 for their exact treatment.
5. **Base compressor.** Ridge and Compactor are the strongest candidates to bolt this on to first, since they're the current focus of the RULER16k sweeps. Every method in `eval_harness/kv_compression/compressors/` is potentially convertible.

## Relationship to existing work

- **vAttention (2510.05688):** verified *sparse attention* — full K/V present, peeks at softmax per decode step. This work is the *KV cache* analogue: verified *once* at eviction time, guarantee carried by ambassador tokens.
- **All existing KV compressors:** deterministic top-k with no bound. This work is a wrapper that upgrades any of them into a verified variant.
