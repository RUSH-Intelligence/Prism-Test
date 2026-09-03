# Ridge — Streaming & Speedup, and the Linear Algebra Behind It

Companion to [ridge_explained.md](./ridge_explained.md). That note explains *what*
Ridge does conceptually (the cloud of keys, tau, omega, RoPE). This note explains
the **math machinery** — enough linear-algebra recap to actually follow it — and
uses that to explain the two claims we make about Ridge vs Compactor:

1. **Ridge can compress while streaming; Compactor cannot.**
2. **Ridge should be faster**, because all its hard math runs on a tiny
   fixed-size matrix that doesn't grow with the document.

Both claims come from one fact: Ridge's core quantity, the Gram matrix `KᵀK`, is
a **running sum over tokens**. Everything below builds up to why that's true and
why it matters.

---

## Part A — The overarching picture (no math yet)

### What "streaming" means

- **Streaming** = compress the KV cache *as tokens arrive*, in chunks, never
  needing the whole sequence in memory at once. Like reading a book and keeping
  running notes — you never need all 500 pages open on the desk simultaneously.
- **Non-streaming (one-shot)** = you must wait until the *entire* context is
  loaded, look at all of it together, and only then decide what to keep. All 500
  pages spread on the desk at the same time.

This distinction is the whole argument for Ridge as a method, even when it scores
a point or two below Compactor on LongBench quality: Ridge works in a setting
Compactor fundamentally can't touch.

### Why Compactor can't stream

Compactor scores every token by blending three things, and each needs the full
sequence:

1. **Leverage scores** via an **SVD** (a heavy matrix factorization) over the
   keys, then **z-scores** each token *relative to all other tokens*.
2. **Non-causal attention** — every token "looks at" every other token,
   including ones *after* it. You can't compute this until every token exists.
3. **Blend + z-score again** — combine, and normalize again against the whole
   sequence.

The z-scoring and "every token sees every token" steps are the killers: a
token's score isn't a property of that token alone, it's defined *relative to the
whole batch*. If tokens are still streaming in, the score is undefined. Our code
hard-asserts Compactor is **prefill-only**.

### Why Ridge can stream

Ridge scores each token from two quantities:

- **tau** = "how unique is this key's direction?" — from the Gram matrix `KᵀK`.
- **omega** = "how much do the queries care about this key?" — from `QᵀQ`.

Both are Gram matrices, and a Gram matrix is a **sum, one small piece per token**.
You can build it a little at a time as tokens stream in, discarding each token
after adding its piece. That is exactly what streaming needs, and it's exactly
what Compactor's SVD + relative z-scoring cannot be turned into.

The rest of this note proves the "sum, one piece per token" claim from scratch.

---

## Part B — Linear algebra recap (with real numbers)

Everything here uses one tiny running example: **3 tokens, each key is 2 numbers**
(so `N = 3` tokens, `D = 2` dimensions). In the real model it's `N ≈ 16000` and
`D = 128`, but the rules are identical.

```
k1 = (2, 0)
k2 = (1, 1)
k3 = (0, 3)
```

### B0 — The only cost unit: a "multiply-add"

When we say a computer is "fast" or "slow" at this, we're counting
**multiply-adds**: multiply two numbers, add to a running total. That's the atom.
"Ridge is faster than Compactor" just means "Ridge needs fewer multiply-adds."

### B1 — Vector

An ordered list of numbers. `k1 = (2, 0)` is a length-2 vector. Picture it as an
arrow from the origin: 2 right, 0 up.

### B2 — Dot product (multiply matching slots, add them up)

```
k1 · k2 = (2, 0) · (1, 1) = (2·1) + (0·1) = 2 + 0 = 2
```

One number out. Cost = `D` multiply-adds (one per dimension). Reused later:

```
k1 · k1 = (2·2) + (0·0) = 4
k2 · k2 = (1·1) + (1·1) = 2
k3 · k3 = (0·0) + (3·3) = 9
```

### B3 — Matrix (a grid of numbers)

Stack the 3 keys as **rows** of a matrix `K`:

```
      col1  col2
     ┌           ┐
row1 │   2    0  │   ← k1
K =  │   1    1  │   ← k2      (3 rows, 2 columns → "3×2")
row3 │   0    3  │   ← k3
     └           ┘
```

`K` is 3 rows tall (one per token), 2 columns wide (one per dimension). Real
model: `16000 × 128` — tall and skinny. The **columns** also mean something:

- Column 1 = `(2, 1, 0)` = "the 1st coordinate of every token."
- Column 2 = `(0, 1, 3)` = "the 2nd coordinate of every token."

### B4 — Transpose `Kᵀ` (flip rows ↔ columns)

`K` is 3×2; `Kᵀ` is 2×3. The rows of `Kᵀ` are the columns of `K`:

```
      ┌           ┐              ┌               ┐
      │   2    0  │              │   2    1    0 │   ← was column 1 of K
K  =  │   1    1  │     Kᵀ  =    │   0    1    3 │   ← was column 2 of K
      │   0    3  │              └               ┘
      └           ┘                (2 rows, 3 columns)
```

### B5 — Matrix multiplication (the one rule)

> Entry in row `i`, column `j` of the answer = (row `i` of the left matrix) ·
> (column `j` of the right matrix).

Size rule: `(2×3) × (3×2) = (2×2)`. The inner 3's must match and get "eaten"; the
outer 2 and 2 become the answer's shape.

---

## Part C — Computing `KᵀK` by hand

We want `Kᵀ K = (2×3) × (3×2) = a 2×2 answer`.

```
        Kᵀ                 K
   ┌            ┐      ┌       ┐
   │  2  1  0   │      │  2  0 │
   │  0  1  3   │      │  1  1 │
   └            ┘      │  0  3 │
                       └       ┘
```

Each answer entry = (a row of `Kᵀ`) · (a column of `K`):

```
(1,1): (2,1,0)·(2,1,0) = 4 + 1 + 0 = 5
(1,2): (2,1,0)·(0,1,3) = 0 + 1 + 0 = 1
(2,1): (0,1,3)·(2,1,0) = 0 + 1 + 0 = 1
(2,2): (0,1,3)·(0,1,3) = 0 + 1 + 9 = 10
```

```
        ┌         ┐
KᵀK  =  │  5   1  │
        │  1  10  │
        └         ┘
```

### What each entry represents

- `(1,1) = 5` = column 1 dotted with itself = **total energy in coordinate 1**
  over all tokens (2² + 1² + 0²).
- `(2,2) = 10` = **total energy in coordinate 2** over all tokens (0² + 1² + 3²).
- `(1,2) = (2,1) = 1` = how much coordinates 1 and 2 **move together** across
  tokens.

This is the "shape of the cloud" from `ridge_explained.md`, now as arithmetic:
diagonal = how stretched along each axis, off-diagonal = how tilted. Crucially,
**it's 2×2 even though we had 3 tokens** — the token count got summed away. With
16,000 tokens it's still `128×128`.

---

## Part D — The punchline: `KᵀK` is a SUM over tokens

Look at how entry `(2,2) = 10` was built:

```
(2,2) = (0·0) + (1·1) + (3·3)
         └k1's┘   └k2's┘   └k3's┘
        piece    piece    piece
```

Each term came from **one token**. That's true for every entry. So instead of
grouping by entry, group by token: each token contributes a whole little 2×2
grid, its **outer product** `kᵢ ⊗ kᵢ`, built by:

> `(kᵢ ⊗ kᵢ)` at position `[a, b]` = `kᵢ[a] · kᵢ[b]`.

Build all three by hand:

```
        [ 2·2  2·0 ]   [ 4  0 ]
k1⊗k1 = [ 0·2  0·0 ] = [ 0  0 ]

        [ 1·1  1·1 ]   [ 1  1 ]
k2⊗k2 = [ 1·1  1·1 ] = [ 1  1 ]

        [ 0·0  0·3 ]   [ 0  0 ]
k3⊗k3 = [ 3·0  3·3 ] = [ 0  9 ]
```

Add the three grids slot by slot:

```
  [ 4  0 ]   [ 1  1 ]   [ 0  0 ]     [ 5   1 ]
  [ 0  0 ] + [ 1  1 ] + [ 0  9 ]  =  [ 1  10 ]
```

**Exactly the `KᵀK` from Part C.** ✓ So:

```
KᵀK  =  (k1 ⊗ k1)  +  (k2 ⊗ k2)  +  (k3 ⊗ k3)
```

One clean 2×2 contribution per token, added up. **That is what "summable" means**
— not a metaphor, a literal fact.

---

## Part E — Summable ⇒ streamable

Because it's a running total, build it one token at a time and discard each token
after adding it:

```
running = [ 0  0 ]      ← start empty
          [ 0  0 ]

k1 arrives → += k1⊗k1 → [ 4  0 ]   → discard k1
                        [ 0  0 ]

k2 arrives → += k2⊗k2 → [ 5  1 ]   → discard k2
                        [ 1  1 ]

k3 arrives → += k3⊗k3 → [ 5   1 ]  → discard k3
                        [ 1  10 ]
```

Never more than one token plus the tiny 2×2 scratchpad in memory. Token 3 didn't
care that tokens 1–2 are gone; their contribution is already baked into the
total. **That's streaming** — process a 16,000-token (or 16-million-token)
document while only ever holding a `128×128` grid.

Compactor's scores can't be written as "token1's bit + token2's bit + …" (they
compare every token against every other and normalize against the whole batch),
so there's no running total to accumulate — you're stuck holding everything at
once.

---

## Part F — The cost math (where the speedup comes from)

Two numbers control everything:

- **`N`** = number of tokens. **Huge** (16,000+).
- **`D`** = head dimension = length of each key. **Small and fixed** (128).

Keep the gap in mind: **N gigantic, D tiny.** The speedup is entirely about
routing the hard math onto `D`, not `N`.

Ridge's leverage formula:

```
tau_i = k_i · (KᵀK + λI)⁻¹ · k_i
```

Cost of each piece:

| Piece | Cost | Grows with document length `N`? |
|---|---|---|
| Build `KᵀK` (the Gram) | `N · D²` | Yes, but only **linearly** |
| Invert it, `(·)⁻¹` | `D³` (fixed, ~2M for D=128) | **No — same for any length** |
| Score every token | `N · D²` | Yes, linearly |

### The inverse

`⁻¹` = **matrix inverse**, the matrix version of "divide by" — the grid that
*undoes* `KᵀK`. (`λI` is a tiny cushion on the diagonal so the undo never blows
up; see `ridge_explained.md` §8.) You never compute it by hand. Two facts:

1. Inverting an `m × m` matrix costs ~`m³`.
2. The matrix we invert is `KᵀK`, which is **`128 × 128`** — the small one — NOT
   `16000 × 16000`.

So the one genuinely hard step runs on a tiny fixed grid whose size **never grows
with the document.** Everything touching `N` is just cheap linear addition. This
is "the Gram-matrix formulation": funnel all the hard math onto the small `D × D`
matrix.

---

## Part G — Why Compactor costs more (the contrast)

Compactor computes the same *idea* (leverage — which keys are unique) with a
heavier tool, plus extra passes:

1. **SVD is pricier than an inverse.** Even when big-O looks similar, SVD has a
   much larger constant factor and runs as a slow iterative routine GPUs dislike.
   Ridge's Gram-build + solve is a couple of fast, GPU-friendly matrix multiplies.
2. **Compactor does a second expensive pass Ridge doesn't.** Its score is
   `attn_z + 0.5·lev_z` — the leverage (SVD) **and** a full **non-causal
   attention pass** over the sequence, then blended. Two heavy scans. Ridge's
   second signal (omega) is just *another cheap Gram matrix* `QᵀQ` — same `D × D`
   trick, not a full attention pass.
3. **Compactor can't stream, so it can't amortize.** Its z-scoring normalizes
   every token against the whole sequence and its attention lets every token see
   every other — neither is a running sum, so it must hold the entire context at
   once and do the work in one big batch. Ridge's Gram dribbles in token-by-token.

```
RIDGE leverage:                        COMPACTOR leverage:
  ┌─────────────────────┐               ┌───────────────────────────┐
  │ cheap linear scan    │  build Gram   │ chunked SVD (slow,         │
  │ over N tokens        │──────────────▶│ iterative, GPU-unfriendly) │
  │      ↓               │               │           +                │
  │ ONE tiny 128×128     │  the hard     │ full non-causal            │
  │ inverse (fixed cost, │  part, but    │ attention pass over N      │
  │ independent of N)    │  tiny         │           +                │
  │      ↓               │               │ z-score vs whole seq       │
  │ cheap linear scan    │  score tokens │ (forces all-at-once)       │
  └─────────────────────┘               └───────────────────────────┘
   can run token-by-token                 must see everything at once
```

**The claim to test in their repo**
([compactor-vllm](https://github.com/vnchari/compactor-vllm)): swap Compactor's
SVD-based leverage for Ridge's Gram-based leverage, time both. If Ridge's
`KᵀK`-and-solve beats their SVD-and-attention on wall-clock, that's the speedup
result — and streaming falls out for free from the same running-sum property.

---

## The three sentences to remember

1. **`KᵀK` is a `D×D` grid where each entry is a dot product of two
   coordinate-columns** — it summarizes the *shape* of all keys, and its size
   doesn't depend on how many tokens you have.
2. **It equals a sum of one small `kᵢ⊗kᵢ` grid per token**, so you can build it by
   adding tokens one at a time and discarding each — that's why Ridge streams and
   Compactor can't.
3. **The one hard step (the inverse) acts on that tiny `128×128` grid**, fixed
   cost regardless of document length — that's where the speedup comes from.
