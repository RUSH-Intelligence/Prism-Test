# Ridge KV Compression — Plain-English Walkthrough

This is a from-scratch explanation of how the `ridge` KV compressor works in this
repo. No fancy words. The goal is that after reading this, you can look at a
ridge run's numbers and have a sense of what every knob is doing.

Code lives in [ridge_sketch.py](../eval_harness/kv_compression/compressors/ridge_sketch.py).

---

## 1. What the KV cache even is

When the model reads your prompt, every token produces two things at every
attention layer:

- A **key vector K** — like an "address" the token uses to advertise itself.
- A **value vector V** — the actual content the model wants to remember.

These get stored in a big bucket called the **KV cache**, one (K, V) pair per
token per layer. Later tokens look back at this cache to "attend" to earlier
parts of the prompt. The cache grows linearly with prompt length — at 16,000
tokens it eats a ton of memory.

**KV compression** is just answering one question: "we don't have enough memory
to keep all of these. Which ones do we throw away?"

Different methods (knorm, snapkv, ridge, …) are different answers to that
question.

---

## 2. The core idea of ridge — "the cloud of keys"

For one attention layer, picture every key as an arrow in space. If
`head_dim = 2`, each arrow lives on a piece of paper. In reality it's 128 dims,
but the picture works the same.

Say the layer has these 4 keys:

```
k1 = (1.00, 0.00)
k2 = (1.00, 0.01)   ← basically a copy of k1
k3 = (0.99, 0.00)   ← also basically a copy of k1
k4 = (0.00, 1.00)   ← points a totally different direction
```

Plot them. You'd see a **fat clump along the x-axis** (k1, k2, k3 piled on
top of each other) and **one lonely arrow up the y-axis** (k4). That's "the
cloud."

Ridge asks: **"which arrows carry information the others don't already cover?"**

- k1, k2, k3 are near-duplicates → throw any of them out and you barely lose
  anything, because the others still point that direction.
- k4 is the only arrow in its direction → throw it out and you lose that
  direction entirely.

So ridge would say: keep k4 for sure, and you can afford to drop two of
{k1, k2, k3}.

### How it measures "sticking out"

It uses a formula called the **ridge leverage score**:

```
tau_i = k_i · (K^T K + lambda·I)^(-1) · k_i
```

Don't memorize this. Plain English:

1. `K^T K` is a "shape matrix" that describes how the cloud is stretched.
   Wide-in-x, thin-in-y cloud → big number in the x-slot, small in the y-slot.
2. Inverting it flips those weights — wide directions get cheap, thin
   directions get expensive. It becomes a **ruler** that says "1 unit of
   stretch in a popular direction is no big deal; 1 unit in a rare direction is
   special."
3. `k_i · (inverse) · k_i` measures how much `k_i` sticks out **with that
   ruler**. Big number = points in a rare, expensive direction = unique. Small
   number = points in a popular, cheap direction = redundant.

For our example, `tau` works out to roughly:

```
tau_1 ≈ 0.33   (popular x-direction, just another one)
tau_2 ≈ 0.33
tau_3 ≈ 0.33
tau_4 ≈ 1.00   (only one in y-direction, unique)
```

Big tau = keep me, I'm unique. Small tau = I'm a copy of others.

That's pure ridge.

### A deeper look at what "wide" actually means

It's worth being precise here, because the picture is easy to misread.

"Wide along the x-direction" does **NOT** mean "many keys point along x." It
means **"the total squared x-energy across all keys, summed up, is big."**

The x-x entry of `K^T K` is literally:

```
(x-x entry) = sum over all keys of (k_i's x-component)^2
```

For our 4 keys:

```
k1 = (1.00, 0.00) → x² = 1.00
k2 = (1.00, 0.01) → x² = 1.00
k3 = (0.99, 0.00) → x² ≈ 0.98
k4 = (0.00, 1.00) → x² = 0
                   ────────
sum               ≈ 2.98 ≈ 3
```

Same for y → roughly 1. So the shape matrix ends up:

```
[ 3   0 ]
[ 0   1 ]
```

The number 3 could come from:

- **Many keys each contributing a little** — 100 keys with x = 0.1 each gives
  total x-mass = 100 × 0.01 = 1.
- **A few keys contributing a lot** — 1 key with x = 1 also gives x-mass = 1.

The matrix can't tell those apart. It only sees the sum.

### Sign doesn't matter

A key at `(-1, 0)` looks **identical** to a key at `(+1, 0)` as far as the
shape matrix is concerned, because:

```
(-1)² = (+1)² = 1
```

Both contribute exactly 1 unit of x-mass. The matrix has no way to tell which
direction along the x-axis a key points.

### Consequence — outliers in the opposite direction get LOW tau too

Imagine the cloud has 1000 keys clustered near `(1, 0)`, and you add one lone
outlier at `(-1, 0)`. You'd think the outlier is super "unique," right? It's
on the far side of the axis, all alone.

Ridge does **not** see it that way. Its tau works out to roughly the same
small value as any of the 1000 cluster keys. Why? Because the x-axis has
already been "spent" by the cluster, and the outlier just adds one more vote
for that same already-spent axis. Ridge thinks "you're on the x-line, the
x-line is taken, you're redundant."

Here's a quick table for the original 3-key x-cluster:

| New key | tau | Why |
|---|---|---|
| `(1, 0)` | 0.33 | Duplicate of the cluster |
| `(-1, 0)` | 0.33 | Opposite direction, but same axis → still "spent" |
| `(0.5, 0)` | ~0.08 | Smaller magnitude on the same spent axis |
| `(0, 1)` | 1.00 | Uses a fresh direction (y) → unique |
| `(1, 1)` | ~1.33 | Partly old (the x part), partly fresh (the y part) |
| `(2, 0)` | ~1.33 | Same direction as cluster, but bigger → "pushes further" along the spent axis, which ridge weirdly rewards |

That last row is another quirk: ridge gives **higher tau to keys with bigger
magnitude along the popular direction**, because they "stretch" further along
it. So a really long vector along x can score high even though it's not in a
new direction. People sometimes L2-normalize keys before computing ridge
leverage to avoid this, but this code doesn't — it operates on raw cached
keys.

### How magnitude exactly maps to tau

It's not a hard "in-range vs out-of-range" threshold — it's a smooth gradient.
For a key on the x-axis, tau is roughly:

```
tau = x² / (total x-energy of the cloud)
```

So tau scales with the **square of the magnitude**, regardless of sign:

| Cluster sits at | New key | tau | Comment |
|---|---|---|---|
| x ≈ 1 | x = +1 | 0.33 | "normal" magnitude, normal score |
| x ≈ 1 | x = -1 | 0.33 | same magnitude, same score (sign invisible) |
| x ≈ 1 | x = +0.5 | 0.08 | smaller magnitude → smaller tau |
| x ≈ 1 | x = +1.5 | 0.75 | bigger magnitude → bigger tau |
| x ≈ 1 | x = +2 | 1.33 | even bigger → even bigger tau |
| x ≈ 1 | x = -3 | 3.00 | far out (either sign) → high tau |

So:

- **Within the typical magnitude of the cluster** (`|x|` ≈ 1) → all keys get
  roughly the same tau, sign-blind.
- **Below the typical magnitude** (`|x|` < 1) → score drops smoothly.
- **Above the typical magnitude** (`|x|` > 1) → score rises smoothly.

There's no cliff you cross — it's just `magnitude²` scaling. A key at
`|x| = 1.01` would score very slightly higher than one at `|x| = 1.00`.

### Why this is weird

Conceptually you'd want "uniqueness" to mean "points in a direction nobody
else points." But ridge actually rewards two different things together:

1. **Direction uniqueness** — using an axis the cluster doesn't span (the
   thing you'd hope for).
2. **Magnitude on a popular axis** — being unusually far out along an
   already-used axis (the quirk).

These get blended into one number. So a key that's a far outlier along the
spent x-axis can score similarly to a key on a totally fresh y-axis, even
though intuitively only the second one is "uniquely informative."

The pure-math fix is to L2-normalize keys before computing leverage (everyone
is the same length, only direction matters). This code doesn't — it scores
raw cached keys, magnitude and all.

### TL;DR on the cloud

- "Wide along axis X" = "total squared X-component across all keys, summed up."
  Not a count, not a range — a sum of squares.
- Sign is invisible. `(-1, 0)` and `(+1, 0)` look identical to the shape matrix.
- A key gets a low tau if it points along a direction other keys already use,
  regardless of which way along that direction.
- A key gets a high tau only if it uses a direction no other keys touch — or
  if it stretches unusually far along a direction.

---

## 3. The catch — ridge alone doesn't know what the model cares about

Ridge picks unique keys, but unique ≠ useful. A key can be one of a kind AND
totally irrelevant to whatever the user is asking. So the code adds a second
signal: **does the prompt actually care about this key?**

This is called `omega`:

```
omega_i = ||Q · k_i||
```

`Q` is a matrix of "queries" — we'll explain in Section 5 exactly where these
queries come from. For now, think of them as "all the things the prompt was
looking up."

`Q · k_i` is a list of dot products — one per query, measuring how strongly
that query points at key `i`. Taking the norm (length) gives **total attention
energy key `i` receives across all queries**.

### How omega is actually computed, step by step

For **one** key `k_i`:

1. Dot it with every query → get a list of T numbers.
2. Square each of those numbers.
3. Add them all up.
4. Take the square root.

That's `omega_i`. You repeat the whole thing for every key — each key gets
its own omega. The set of queries Q stays the same across all keys; only
`k_i` changes.

The matrix shorthand `omega_i = ||Q · k_i||` just packages those four steps
into one expression. `Q · k_i` produces the list of T dot products as a
vector, and `|| · ||` is the "length" of that vector, which is
square-root-of-sum-of-squares (the L2 norm).

### Worked example for all 4 keys

Queries:

```
q1 = (5.0, 0.0)
q2 = (0.1, 0.0)
```

Keys:

```
k1 = (1.00, 0.00)
k2 = (1.00, 0.01)
k3 = (0.99, 0.00)
k4 = (0.00, 1.00)
```

For each key, dot with every query, square, sum, sqrt:

```
omega_1:
  q1 · k1 = 5*1 + 0*0     = 5.0     → squared = 25.00
  q2 · k1 = 0.1*1 + 0*0   = 0.1     → squared =  0.01
  sum = 25.01,  sqrt ≈ 5.00

omega_2:
  q1 · k2 = 5*1 + 0*0.01  = 5.0     → squared = 25.00
  q2 · k2 = 0.1*1 + 0*0.01= 0.1     → squared =  0.01
  sum = 25.01,  sqrt ≈ 5.00

omega_3:
  q1 · k3 = 5*0.99 + 0*0  = 4.95    → squared = 24.50
  q2 · k3 = 0.1*0.99 + 0*0= 0.099   → squared =  0.0098
  sum ≈ 24.51, sqrt ≈ 4.95

omega_4:
  q1 · k4 = 5*0 + 0*1     = 0       → squared = 0
  q2 · k4 = 0.1*0 + 0*1   = 0       → squared = 0
  sum = 0,    sqrt = 0
```

### What omega is really measuring

Each dot product `q_j · k_i` is the **raw attention score** that query j
would have given to key i (before softmax). Squaring and summing across all
queries is just "total attention pressure key i collects across the whole
set of queries." The sqrt at the end is mostly cosmetic — keeps units
reasonable. The ranking would be the same without the sqrt (sqrt is
monotonic for non-negative numbers).

The code computes it via a tiny algebraic shortcut for speed: build the
matrix `Q^T Q` once, then for each key compute
`omega_i² = k_i · (Q^T Q) · k_i`. Mathematically identical. See
`_compute_query_key_interaction` at
[ridge_sketch.py:380](../eval_harness/kv_compression/compressors/ridge_sketch.py#L380).

One small detail: by default `query_gram_normalization="mean"` divides
`Q^T Q` by `T` (the number of queries), so omega values are scaled down by
`1/sqrt(T)`. Global rescaling — doesn't change the **ranking** of keys,
just the absolute numbers.

### Same toy example, with queries

Suppose two queries:

```
q1 = (5.0, 0.0)    ← strongly pointing along x
q2 = (0.1, 0.0)    ← weakly pointing along x
```

Then:

```
omega_1 = sqrt(5^2 + 0.1^2) ≈ 5.00    ← queries hammer it
omega_2 ≈ 5.00
omega_3 ≈ 4.95
omega_4 = sqrt(0 + 0) = 0             ← nothing points at it
```

Now compare to tau:

| Key | tau (unique?) | omega (queries care?) |
|---|---|---|
| k1 | 0.33 (no) | **5.00 (yes!)** |
| k2 | 0.33 (no) | **5.00 (yes!)** |
| k3 | 0.33 (no) | **4.95 (yes!)** |
| k4 | **1.00 (yes!)** | 0.00 (no) |

These two signals **completely disagree** in this example. Pure ridge would
throw away k1/k2/k3 (the ones the model actually wants). Pure query would throw
away k4 (the only key covering the y-direction).

The default ridge combo `max(tau, omega)` says "keep it if EITHER signal says
keep" → keeps k4 from ridge's vote, keeps one of k1/k2/k3 from omega's vote.
Best of both worlds.

---

## 4. The third signal — value norm `||v_i||`

Remember each token has a value vector V too. At attention time, the model's
output is roughly `sum over i of (attention_weight_i × v_i)`. If a value
vector is tiny, that token barely contributes to the output even if it gets
high attention. So keeping it is pointless.

Ridge multiplies the final score by `||v_i||`, the length of the value vector.
Tokens with bigger, more "substantial" values get a boost.

The exponent on this is `value_norm_power`:

- 0 → ignore values entirely
- 1 (default) → linear weighting, twice the value = twice the boost
- 2 → squared, very biased toward big values
- 0.5 → mild bias

---

## 5. When compression actually happens, and which queries are used

This is the part that confuses people most. Let me walk through one full
inference run, step by step.

### The user sends two things

1. A **context** — a long document, say 15,000 tokens.
2. A **question** — say 100 tokens.

### Step 1 — prefill the context

The model reads all 15,000 context tokens in one pass. For each token, at each
attention layer, it computes K and V. These get stored in the cache. No
compression happens here — we're just loading the document.

After this, the cache has 15,000 (K, V) pairs per attention layer. Big.

### Step 2 — compress, once per layer

This is the moment ridge fires. It fires **exactly once per attention layer,
right after the document prefill finishes**. The setting that controls this is
`compression_schedule = post_prefill` (the default).

At the moment compress() runs, it gets handed:
- `keys` — all 15,000 cached K vectors for this layer
- `values` — same for V
- `hidden_states` — the model's internal representation at each of the 15,000
  context positions (one vector per token, fed into this layer)

**Where do the queries Q come from?** Inside compress(), there's this line:

```python
q = module.q_proj(hidden_states)
```

It runs the cached hidden states through the same projection the layer would
have used during prefill, giving you a query vector for every one of the
15,000 context tokens. So `Q` is a "every context position acted as a query"
matrix.

**These queries are NOT the user's question.** The question hasn't even been
processed yet. The queries are the **context's own self-attention queries** —
the queries each context token used while it was reading the document.

So `omega_i = ||Q · k_i||` is asking: "during the document's own prefill, how
much attention did key i pull from all the other context tokens combined?" It's
a self-attention popularity score.

### Step 3 — score and pick

For each key in the middle zone (we'll explain zones in Section 6), compute
the score and pick the top-k. Throw the rest out of the cache.

### Step 4 — decode the question

Now the question gets fed in one token at a time. Each token attends to the
**compressed** cache plus whatever it has generated so far. **No more
compression** — ridge only fired once back in Step 2.

So whatever the user asks, it sees the same compressed cache. The question
itself doesn't influence which keys got kept.

### "Do we compress multiple times for multiple question tokens?"

No. Default ridge = `post_prefill` = one compression per layer, right after
the document loads, before the question is touched.

If you set `compression_schedule = streaming`, it would fire after every
prefill chunk. There's a footgun there — if you chunk the prefill into pieces,
the compression ratio applies per chunk, so retained tokens collapse
geometrically. The default `post_prefill` with single-pass prefill is what you
usually want.

### Important caveat — "document then question" is a benchmarking choice, not a universal rule

The clean split above (prefill the document, compress, THEN start the
question) is specific to this framework's research backend. The pipeline
intentionally tokenizes context and question into two separate tensors
(`context_ids`, `questions_ids` — see
[research_pipeline.py:204](../eval_harness/research_pipeline.py#L204)) so
that compression sees only the document. This isolates the effect of
compression for evaluation purposes.

**In most real-world LLM systems (ChatGPT, Claude, vLLM serving, …), there is
no such split.** The user's full prompt — system message + context + question —
is one concatenated sequence. That whole thing gets prefilled together in one
pass, and if any compression is applied, it sees the question tokens too.

What this means for ridge's omega in a real system:

- The "queries Q" used to score keys would include queries from the question
  positions, not just the document positions.
- Those question-position queries are exactly the ones whose attention the
  model is about to act on → they're more informative about which keys
  actually matter for the answer.
- So omega in a real chat setting is generally a stronger signal than it is in
  this framework's benchmark setting.

Why does this framework split anyway? Two reasons:
1. **Apples-to-apples benchmarking** — every method sees the same compressed
   document, regardless of how clever its question-aware scoring might be. If
   the question were in the prefill, methods could "cheat" by leaning on it.
2. **Realistic for RAG / multi-turn** — in setups where one long document is
   reused across many user queries (RAG, document-grounded chat, prefix
   caching), the document genuinely IS prefilled and compressed once, before
   any specific question arrives. The split mirrors that pattern.

---

## 6. The three zones — sink, middle, local

Not every token is even eligible for scoring. The cache is split:

```
[ SINK (first N tokens) | MIDDLE (scored, some pruned) | LOCAL (last M tokens) ]
```

- **Sink** (`sink_size`, default 8) — always kept. The first few tokens are
  "attention sinks." Models lean on them a lot. Removing them hurts.
- **Local** (`local_size`, default 64) — always kept. The most recent tokens
  matter for decoding the answer.
- **Middle** — the only zone ridge scores and prunes.

Target keep count = `int(T × (1 - compression_ratio))`. Subtract sink + local,
the rest is what gets picked from the middle by score.

### Heads-up on defaults

`sink=8, local=64` in this repo **differ from the upstream RidgePress
reference**, which used `sink=4, local=28`. If you're comparing this codebase's
numbers against a saved baseline, that difference alone can move results. To
reproduce the upstream: pass `sink_size=4, local_size=28`.

### `compression_ratio` is fraction PRUNED, not kept

`compression_ratio = 0.5` means **drop half**, keep half.
`compression_ratio = 0.8` means drop 80%, keep 20%.

Single most common confusion. Double-check this when reading numbers.

---

## 7. RoPE, and the rotate_queries quirk

RoPE is genuinely confusing the first time you meet it. This section uses a
clock-hands picture instead of math.

### The problem RoPE is trying to solve

When a model reads "the cat sat on the mat," the math of attention doesn't
naturally know that "cat" came before "mat." Without something extra, the
model would treat the words like a bag — order is invisible.

So we need to give every token a **fingerprint** that says "I am the 5th
token" or "I am the 1000th token." That fingerprint = RoPE.

### The trick — clock hands

Picture every token's key vector as a **clock hand** sitting at 12:00 to
start with.

For each token, we **spin the hand by an amount that depends on its
position**:

```
Token at position 0    → no spin, hand still at 12:00
Token at position 1    → spin a tiny bit, hand at 12:01
Token at position 5    → spin a bit more, hand at 12:05
Token at position 30   → hand at 12:30
Token at position 60   → hand at 1:00
Token at position 720  → went all the way around once
```

Same trick for the query vectors — each query's "hand" gets spun by the
query's position.

That's it. Every token now has a unique hand orientation purely based on
where it sits in the sequence.

### Why this is useful

When the model does attention, it compares two tokens by
**dot-producting their vectors**. Dot product is basically:

- Two vectors pointing the **same** direction → BIG number → "these match"
- Two vectors pointing **different** directions → small number → "these
  don't match"

With RoPE in place, the dot product between two tokens automatically
reflects **how far apart their clock hands are**, which is **how far apart
they are in the sequence**.

- A query at position 6 and a key at position 5 → hands 1 minute apart →
  still mostly aligned → big dot product → "these are close"
- A query at position 6 and a key at position 1000 → hands way far apart on
  the dial → smaller dot product → "these are far"

Distance-awareness for free, just from spinning the hands.

### What "rotating a vector" actually looks like

If a vector is just two numbers like `(1, 0)`:

- It's an arrow pointing right (east)
- Spin it 90° → now points up (north) → `(0, 1)`
- Spin it 180° → now points left (west) → `(-1, 0)`
- Spin it 360° → back to where it started

That's literally all "rotation" means. Spin the arrow around the origin
like a clock hand.

For a real 128-number key, it does the spin in **pairs** (numbers 0+1 spin
as one little hand, 2+3 as another, etc.) — so a 128-dim key is really 64
little clock hands, each spinning at its own rate. But each one is just
doing the same simple "spin a hand" trick.

### Wait, doesn't rotating destroy the semantic info?

Natural concern: cat and kitten start out near each other in vector space
(semantically similar). If we rotate cat by 5° (position 5) and kitten by
1000° (position 1000), they end up in totally different parts of space. So
a query searching for "cat-like stuff" would fail to find kitten. Right?

**No — because the query is also rotated.**

Key fact about rotation: **if you rotate two arrows by the SAME amount,
their alignment with each other is preserved.** Both moved, but they
moved together.

Picture two arrows pointing NE:

```
↗ ↗   ← both pointing NE
```

Rotate both by 90°:

```
↘ ↘   ← both moved, still match each other
```

Their dot product is unchanged.

Now rotate one by 0° and the other by 90°:

```
↗     ← arrow 1 unchanged
↘     ← arrow 2 spun 90°
```

They don't match anymore.

**It's the DIFFERENCE in rotation that matters, not the absolute rotation.**

### Applying that to attention

When the model dot-products a rotated query with a rotated key, the math
works out so the answer depends on:

1. **How aligned q and k were ORIGINALLY** — the semantic similarity. Was
   this query about cats, was this key about kittens?
2. **The DIFFERENCE in their rotations** — i.e., the difference in their
   positions. Were they close in the sequence, or far apart?

Three concrete cases:

| Case | Query | Key | Rotation diff | Semantic match | Final dot product |
|---|---|---|---|---|---|
| A | cat-query @ pos 6 | kitten-key @ pos 5 | 1° (tiny) | high (cat ~ kitten) | high → "match, and close" |
| B | cat-query @ pos 6 | kitten-key @ pos 1000 | 994° (big) | high (cat ~ kitten) | medium → "matches semantically, but far" |
| C | cat-query @ pos 6 | dog-key @ pos 5 | 1° (tiny) | low (cat ≠ dog) | low → "not cat-like, position doesn't save it" |

Both signals coexist in the same number. RoPE doesn't choose between
semantic-info and position-info — it weaves them together.

### Why the intuition felt off

You were thinking of vectors as **fixed points in space** — cat is here,
kitten is there. From that view, rotating each by a different amount
scatters everything.

But attention doesn't read vectors as fixed points. It only ever asks
**"how aligned are these two specific vectors?"** And that alignment
depends on how much they were rotated **relative to each other**, not on
where each one ended up.

### What about accidental alignment? Couldn't rotation make two random vectors line up?

Another good concern: if cat and dog point at right angles originally (dot
product ≈ 0), and rotation introduces a 90° relative spin, now they point
the same direction → artificial high score. Wouldn't that lie about
similarity?

In a hypothetical 2D world with a single rotation, yes. RoPE protects
against this by using **many clock hands at different rotation speeds**.

The 64 little clock hands inside a 128-dim key each spin at their own rate:

- Hand 1 spins fast — maybe 1° per position
- Hand 2 a bit slower — maybe 0.5° per position
- Hand 3 slower — 0.25°
- …
- Hand 64 glacially — maybe 0.0001° per position

At position-distance 994:

- Hand 1 has rotated 994° (wrapped multiple times)
- Hand 2 has rotated 497°
- Hand 3 at 248.5°
- Hand 64 has barely moved (~0.1°)

Each hand contributes its own modulation. The dot product is the sum
across all 64. For two random vectors to "accidentally align," they'd need
to line up on **every single one of the 64 hands at the same
position-distance** — essentially impossible.

What actually happens: the modulations from different hands smear together
and partially cancel. The net result is a smooth-ish curve that **decays
with distance**:

```
  modulation
  ▲
1 ┤●
  │ ●
  │  ●
  │   ●●
  │     ●●●
  │        ●●●●●●●●●●●●●●●●●●●●
0 ┼────────────────────────────────────────► distance
  0       50       100      500
```

Strong at distance 0, smoothly decays as distance grows. No weird spikes,
no accidental alignments at trained distances.

### BUT — at very long distances, this protection breaks down

The model is trained on sequences up to some length (say 8K). It learns to
interpret modulation patterns for distances 0 through ~8000. Push it to
distance 50,000:

- The fast-spinning hands have cycled around so many times they produce
  patterns the model has never seen.
- The slow-spinning hands enter regimes they were never exposed to in
  training.
- The smooth-decay picture above gets noisy and lumpy.
- The model gets genuinely confused — semantically distant things can
  occasionally look close, similar things can look distant.

This is **the central problem of long-context inference**. It's exactly
what positional methods like YaRN, NTK scaling, and Linear-PI exist to
fix — they stretch RoPE so it works gracefully at longer-than-trained
distances. (That's "Door 1" in the Prism-Test framework —
`positional_methods/`.)

### What this means for the KV cache

When the model puts a key in the cache, it puts it in **already spun**. So
`cache[token_5]` is the key for token 5 **after** it's been spun by an
amount proportional to position 5. The original un-spun version is gone —
only the spun one is stored.

### Now the ridge quirk, finally

Inside ridge's compress() we need queries Q and keys K. Here's what
happens:

- It grabs **keys** from the cache → already spun ✓
- It computes **fresh queries** inside compress() via
  `q = q_proj(hidden_states)` → these are **NOT spun** ✗

So queries and keys use **different clock conventions**. The omega number
still gets computed, but it's not quite what the real attention layer
would compute when the model actually runs. It's a slightly off proxy.

**Why default to the mismatch?** The upstream reference code did it that
way and this port is faithful to upstream bit-for-bit.

**`rotate_queries=True`** = "also spin the queries to match the keys."
Now both clocks agree → omega faithfully matches real attention. Off by
default; opt-in deviation from upstream.

Effect on numbers: usually small but real. Flipping it changes which
tokens end up at the top of the score and which get cut.

On NemotronH this flag is irrelevant — NemotronH doesn't use RoPE at all,
so its cached keys aren't spun to begin with.

### TL;DR on RoPE

- The model needs to know token positions.
- RoPE = spin each token's key/query vector like a clock hand, by an
  amount = the token's position.
- Spinning preserves alignment between two vectors **if they're spun by
  the same amount**. The dot product after RoPE depends on the
  **difference** in rotation (i.e., position difference) and the
  **original** alignment (i.e., semantic similarity) — both at once.
- Many clock hands at different speeds prevent accidental alignments at
  trained-on distances. Past trained distances, this breaks down — and
  that's what context-extension methods exist to fix.
- The cache stores already-spun keys.
- Ridge's omega accidentally mixes spun keys with unspun queries unless
  you flip `rotate_queries=True`.

---

## 8. `ridge_lambda` — the safety regularizer

The formula uses `(K^T K + λ·I)^(-1)`. The `λI` is `ridge_lambda` times the
identity matrix.

### Why it's needed

`K^T K` can be uninvertible (singular) — e.g. if some keys are exactly
parallel to others, or if there are fewer keys than dimensions. Inverting a
near-zero eigenvalue gives a near-infinity number → numerics blow up.

`λI` adds a tiny floor. The smallest eigenvalue of `K^T K + λI` is at least
`λ`, so the inverse stays bounded.

### What changing λ does

- **Small λ (1e-6)** → ridge stays very sensitive to fine differences between
  keys. Numerically riskier.
- **Default 1e-4** → gentle smoothing, safe.
- **Big λ (1.0)** → heavy smoothing. The cloud looks "rounder" to the
  algorithm — tau values get pulled toward each other, ridge becomes a worse
  discriminator. Starts behaving like everyone is equally unique.

Analogy: measuring how unusual someone's height is. Small λ → real
standard deviation, 6'5" looks rare. Big λ → fake cushion added, 6'5" looks
average.

You almost never need to touch this. Default fine. Only raise it if you see
NaNs.

---

## 9. `normalize_score_components` — keeping the units sane

`tau` and `omega` live on completely different scales:

- `tau_i` is usually 0.001 to 0.1
- `omega_i` can be 10, 100, even 1000+

If you naively do `max(tau, gamma · omega)`, omega wins **every single token**
just because it's numerically bigger. The max becomes meaningless.

**`normalize_score_components=True`** (default) → divide each by its sum
across all keys:

```
tau_normalized   = tau   / sum(all tau)
omega_normalized = omega / sum(all omega)
```

Both now sum to 1 across keys (proper probability distributions). The biggest
entry of each is roughly comparable. Now `max(tau, omega)` is a fair fight.

**`normalize_score_components=False`** → use raw values. Whichever is
naturally bigger dominates. Almost never what you want.

Mental model: comparing percentages, not raw numbers. "tau is 5% of the tau
budget vs omega is 8% of the omega budget" is a real comparison. "tau = 0.01
vs omega = 150" is a unit mismatch.

---

## 10. How tau, omega, and value norm combine — `combine_mode`

Once you have all three signals, you mash them together into one score. There
are 5 modes:

| Mode | Formula | Vibe |
|---|---|---|
| `additive` | `α · tau + (1-α) · omega` | Weighted average. α dials between them. |
| `multiplicative` | `tau^α · omega^(1-α)` | Geometric mean. Token has to be good on BOTH, not great on one. |
| `envelope` | `max(tau, omega)` | "Good on EITHER is enough." |
| `fixed_envelope` (default) | `max(tau, γ · omega)` | Same as envelope, with γ tilt. |
| `weighted_envelope` | `max(tau, γ · omega)`, γ depends on overlap | If tau-top-k and omega-top-k disagree a lot, boost omega. If they agree, leave γ at 1. |

Then multiply by `||v_i||^p` (value norm) for the final score.

### `envelope_gamma` (γ)

Only matters in `fixed_envelope` (the default). It tilts the max:

- γ = 1.0 (default) → balanced
- γ > 1 (say 2.0) → omega side wins more ties → "trust queries more"
- γ < 1 (say 0.5) → ridge side wins more ties → "trust uniqueness more"

### `alpha` (α)

Only matters in `additive` and `multiplicative`. Default 0.8 = lean toward
ridge. α = 1 → pure tau. α = 0 → pure omega.

---

## 11. `alpha_selection` — auto-tuning alpha

Only matters in `additive` and `multiplicative`. The default `fixed_envelope`
ignores alpha entirely. But here are the strategies if you do use those modes:

### `fixed`

Just use the config value of `alpha`. No adaptation. Simplest.

### `entropy` — "how confident are the queries?"

Look at omega across all keys.

**Flat omega** (all keys get similar omega) → queries are vague, they don't
know what they want → trust ridge → push α high.

**Peaky omega** (a few keys hog all the omega) → queries clearly know what
they want → trust queries → push α low.

The code computes Shannon entropy of normalized omega and maps:
- high entropy → α near `alpha_max`
- low entropy → α near `alpha_min`

Cheap, adaptive.

### `tail_risk` — minimax over a grid of alphas

Try a list of candidate α values (the `alpha_grid`, e.g. 0.0, 0.1, …, 1.0).
For each:

1. Compute scores with that α
2. Pick the top-k
3. Measure ridge_tail (how much tau mass got dropped) and query_tail (how much
   omega mass got dropped)
4. Take the **worse** of the two

Pick the α that minimizes the worst case.

Analogy: packing for a trip when you don't know if you're going to Alaska or
Hawaii — pack things that survive both, even if neither is optimal.

### `query_constrained` — "queries first, but don't kill ridge"

Lean toward omega aggressively, but with a safety leash.

1. Compute the pure-ridge ridge_tail (at α = 1) — call this the reference.
2. For each α, measure ridge_tail and query_tail.
3. Pick the α that minimizes `query_tail + huge_penalty × max(0, ridge_tail -
   reference - ridge_slack)`.

It will go query-heavy as long as ridge doesn't get destroyed beyond
`ridge_slack` (default 0.20 = 20% extra ridge mass lost).

### `gated_query_constrained` (default for additive/multiplicative)

Two-stage. The full `query_constrained` search is expensive AND not always
appropriate. So gate first:

1. Are queries peaky enough? Compute `max(omega) / mean(omega)`. If ≥
   `query_peakiness_threshold` (default 4.0), queries are opinionated → safe.
2. Would going all the way to α = 0 destroy ridge? Compute ridge_excess at α = 0.
   If ≤ `gate_ridge_excess_threshold` (default 0.10), even extreme query-leaning
   is tolerable.

If **both** pass → run `query_constrained`'s grid search.
If **either** fails → just use `fallback_alpha` (default 1.0 = pure ridge).

Cheap most of the time, careful when it counts.

---

## 12. Remaining knobs

### `query_aware` (default True)
If False, omega is skipped entirely → score becomes `tau × ||v||`. Pure-ridge
with a value-norm weight.

### `query_position_mode`
- `matching_keys` (default) → only the queries at the middle-zone positions
  are used to compute omega
- `all_prefill` → use queries from every prefill position, including sinks
  and locals

### `selection_method`
Once you have a score per token, how to pick:
- `topk` (default) → take the K highest scores, deterministic
- `multinomial` → sample K with probability proportional to score, stochastic
  (no seed parameter — uses global torch RNG, set seed globally for
  reproducibility)

### `min_tokens_to_compress` (default 64)
If the cache has fewer than this many tokens, don't bother compressing.
Avoids fiddly behavior on tiny caches.

### Knobs that only matter for some alpha_selection modes
- `alpha_grid` — list of α candidates to try (used by tail_risk,
  query_constrained, gated_query_constrained)
- `ridge_slack`, `ridge_penalty` — the safety leash strength for
  query_constrained
- `fallback_alpha`, `query_peakiness_threshold`,
  `gate_ridge_excess_threshold` — the gate for gated_query_constrained

---

## 13. Quick lookup — what to check when numbers surprise you

Two ridge runs disagree? Check, in order:

1. Different `sink_size` / `local_size` — repo default 8/64 vs upstream 4/28
   moves results.
2. `compression_ratio` confusion — it's fraction **pruned**, not kept.
3. Different `combine_mode` — envelope vs additive picks different winners.
4. `rotate_queries` flipped — changes omega values.
5. `query_aware` flipped — without it, you're pure-ridge.
6. `multinomial` selection — non-deterministic, will differ across re-runs
   without a seed.
7. `compression_schedule` — `streaming` over chunked prefill collapses retained
   tokens geometrically. Default `post_prefill` over single-pass is the safe
   path.

---

## 14. One-sentence summary

Ridge keeps the keys that are simultaneously (a) unique within the cloud of
all keys and (b) actually getting hit hard by the prompt's own queries,
weighted by how big their value vectors are — always preserving the first 8
sink tokens and last 64 local tokens, and firing exactly once per layer right
after the document prefill, before the user's question even starts decoding.
