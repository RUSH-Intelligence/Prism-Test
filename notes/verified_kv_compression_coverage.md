# Verified KV Compression — Measuring Coverage (v2 design)

Companion to [`verified_kv_compression_idea.md`](verified_kv_compression_idea.md).
That note explains the *pitch* (turn any top-k compressor into a "verified"
variant with an error bound). This note nails down the part that was fuzzy:
**how do we actually measure how much we might have thrown away, and how does
that decide the token split — all while keeping a fixed memory budget.**

Everything here is written in plain language on purpose. It's the design we
converged on; the math/certificate details come later.

---

## The current code vs. what we want

`verified_sketch.py` (v1) does a **fixed** split: `det_fraction=0.75` → 75% of
the budget is Ridge's smart-picked tokens, 25% is uniform-random from the
evicted pool. It never checks how much it actually captured, and the random
tokens carry no reweighting, so they aren't yet an honest estimate — just
"top-k plus some noise."

**v2 goal:** stop guessing the split. *Measure* how much meaning the kept
tokens reproduce, and let that decide how many random tokens to add — without
ever changing the total budget.

---

## Rule 1 — How we keep a fixed budget while being "adaptive"

There are two knobs you could make adaptive. Only one is safe.

- **Total tokens kept (M):** DO NOT make this adaptive. The cache is a
  fixed-size rectangle; if M floats, memory blows up and the "same cost as
  top-k" story dies.
- **The split of M between smart-picked and random:** make *this* adaptive.

So: **M stays hard-locked. Coverage only slides the dial between "smart tokens"
and "random tokens" inside that fixed M.**

Example, M = 100 slots:
- Head captures 99% already → 98 smart + 2 random (barely need insurance).
- Head captures 60% → 70 smart + 30 random (need lots of insurance).

Both keep exactly 100. If even 100 smart tokens can't hit the target, the method
does **not** grab more memory — it **reports a weaker guarantee** for that head.
The bound bends; the budget doesn't.

Per-head note: each head can use a different split (head A 90/10, head B 60/40)
and still keep exactly M, so the rectangular cache is preserved. Only *different
totals* per head would need the ragged-cache machinery — we don't do that.

---

## Rule 2 — "Coverage" is two different jobs, not one bucket

The biggest source of confusion: bundling everything into one word "coverage."
There are **two separate jobs**, and different ideas belong to different jobs.

- **Job A — PICKING** which tokens to keep.
- **Job B — CHECKING** how much we might have missed.

| Idea | Job A (picking) | Job B (checking) |
|------|:---:|:---:|
| Uniqueness / unique directions | ✅ | |
| Attention weight | ✅ | (weak) |
| Value norm | ✅ | |
| **Output-error (see below)** | | ✅ **best** |

Key realization: **uniqueness is a *picking* method, not a checking method.** It
was confusing to compare it head-to-head with output-error — they answer
different questions.

---

## Rule 3 — The two things that actually matter (the two axes)

When you ask "did I keep what matters?", you're secretly assuming what the cache
is *for*. There are two different answers:

- **Breadth** — did I keep *a piece of every distinct thing* in the document?
  (The needle-in-a-haystack concern.)
- **Intensity** — did I keep the tokens the query is *loud about* and that carry
  *big* values? (The "one heavy token wrecks the answer" concern.)

These get protected through **two different doors**:

- **Breadth → handled at PICKING time.** Ridge's uniqueness term keeps one token
  from every distinct direction *regardless of attention*, so the needle
  survives before we ever check anything.
- **Intensity → handled at CHECKING time.** Output-error + random insurance
  (below).

**Why breadth can't be handled by checking (important):** a breadth miss is a
token *no current query looks at*. Any query-based check measures importance by
asking questions — so it is structurally blind to a token every question
ignores. You cannot measure the importance of something using questions that
ignore it. Therefore breadth **must** be built in at pick time; it can never be
caught after the fact. Trying to make the check catch breadth is the mistake.

---

## Rule 4 — What "output-error" means (the best checking metric)

The output of attention is a **weighted blend**: `out = Σ (attention_i × value_i)`.
Output-error asks: *after dropping tokens, how far is the blended answer from the
true one?* — not "how much attention did we keep."

### Worked example (why it beats the simpler metrics)

Document = 4 tokens; each carries a "value" (pretend it's a number):

| Token | Value | Query looks at it |
|-------|-------|-------------------|
| A | 10 | 30% |
| B | 10 | 30% |
| C | 10 | 30% |
| D | **1000** | 10% |

True output = `0.3×10 + 0.3×10 + 0.3×10 + 0.1×1000` = 3+3+3+**100** = **109**.

Drop D, keep A/B/C → output = **10**. Error = 109 → 10. **Enormous** — even
though D had only 10% attention, because its *value* was huge.

### The punchline: output-error accounts for BOTH norm and query relation

Because the output is `attention × value`, the damage from dropping a token
depends on **both, multiplied**:

- **Query relation (attention)** — how much the query looks at it.
- **Norm (value size)** — how big the thing it carries is.

A token is only dangerous to drop when **both** are non-trivial:

- Query ignores it (≈0% attention) → contributes ≈nothing → safe.
- Query looks at it but value is tiny → contributes ≈nothing → safe.
- Query looks at it **and** value is big (token D) → dropping it wrecks the
  answer → dangerous.

So the two "half" metrics each miss a case:

- **Attention-weight only** sees the query relation, blind to norm → misses D.
- **Norm only** sees the size, blind to the query → over-keeps big tokens the
  query never looks at.

**Output-error = attention × norm, fused automatically** — not because we glued
them together, but because that's literally how the attention output is
computed. You don't have to choose how to weigh attention vs. norm; the math
already does it in the right proportion. This is why it's the north star, and
why attention-weight and norm are just cheap approximations of it.

The one thing it needs is *a query* to define "attention" — that's the proxy
limitation in Rule 5.

---

## Rule 5 — The proxy-query problem (honest scope)

We compress **before the real question exists** (post-prefill, single-pass, the
compressor fires before the question tokens — pinned by
`tests/test_compression_schedule.py`). So we cannot measure output-error against
the real question; we only have the document's own queries as a stand-in.

Don't trust a single query — that's fragile (what if the question hits an
earlier part of the document?). Use **worst-of-several**:

> Take a handful of prefill queries (weight the *last* tokens heaviest — they sit
> nearest where the real question will land — but don't use only them). Measure
> output-error for each. Report the **worst** one, so if *any* plausible query
> cared about a dropped region, we buy more insurance.

**State this scope out loud in the paper:** the guarantee is "verified against
the document's own spotlight," not "verified against whatever gets asked." That
is weaker than vAttention (which peeks at the real decode query with all tokens
present) — but it is a number **nobody else reports**, and it is the same
stand-in every existing compressor already silently trusts. We're just the first
to measure it.

---

## The six-step recipe (v2)

Base = **Ridge**, because its two internal terms already produce both axes for
free (tau ≈ breadth/uniqueness, omega ≈ intensity/attention). Budget M is
hard-locked.

1. **Ridge picks the smart tokens** — ask it for slightly fewer than M, leaving a
   few slots open (e.g. 80 of 100, 20 open). *Breadth is guaranteed here.*
2. **Measure the leftover** — output-error (Rule 4) of the kept tokens, against
   the worst of several prefill queries (Rule 5). Get one number, e.g. "these 80
   reproduce 92% of the answer" → 8% leftover.
3. **Turn leftover into a count** — small leftover → few randoms (give the spare
   slots back to Ridge for more smart tokens); big leftover → many randoms.
4. **Grab the randoms** — sample that many from the tokens Ridge threw away.
5. **Reweight them (the honest part, missing in v1)** — each random token stands
   in for many evicted tokens, so scale its contribution up accordingly
   (Horvitz-Thompson weight). This is what makes the leftover an *unbiased
   estimate* instead of "top-k plus noise." Caveat: a large scale-up can distort
   the softmax, so cap/soften it in practice — a known wrinkle, not a blocker.
6. **Done** — kept = smart + reweighted randoms = exactly M. Same cache size as
   any compressor. New output: a per-head confidence number.

### Division of labor (the one-line mental model)

We keep the **needle by choosing well** (Ridge, up front — breadth) and we keep
the **loud-heavy tokens by checking the answer and buying insurance** (Steps 2–5
— intensity). Two different problems, two different fixes, and **neither can do
the other's job.**

---

## What's already done vs. new work

- Ridge exists and emits both axes → **Step 1 is free.**
- v1 already does "smart tokens + random fill" → **Steps 1, 4, 6 basically done.**
- Genuinely new for v2: **Step 2 (measure output-error leftover), Step 3 (count
  from leftover), Step 5 (reweight).** Lands in `verified_sketch.py` plus this
  note.

---

## Open questions carried forward

- Exact formula from leftover → sample count (Hoeffding/Bernstein constant, value
  norm range) — deferred with the certificate; see
  [`verified_kv_compression_idea.md`](verified_kv_compression_idea.md) §"Open
  design decisions".
- How to cap/soften the Horvitz-Thompson reweight so it doesn't distort softmax.
- Whether to certify per-token output error or downstream logit-KL (idea note
  decision #3).
- Importance sampling (draw randoms proportional to compressor score) instead of
  uniform → tighter bound at the same count (idea note decision #2).
