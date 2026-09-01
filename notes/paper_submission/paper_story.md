# Ridge Paper — The Story (plain English)

This is the simple version of what the paper is trying to say. No jargon.
If someone read only this file, they should understand the whole arc of the
paper in a few minutes. Details, tables, and math go elsewhere — this is just
the *spine*.

Related: [ridge_explained.md](../ridge_explained.md) (how the method actually
works), [venue_and_submission_guide.md](venue_and_submission_guide.md) (where
we submit and how).

---

## The one-sentence version

> Ridge is our KV-cache compressor that decides which tokens to keep using the
> *geometry* of the keys plus a query-importance signal — so it stays competitive
> with attention-based compressors **without ever needing the attention matrix**,
> which means it works with FlashAttention (the thing everyone actually uses).

That last clause — "without needing attention scores" — is the headline. Lead
with it.

---

## The problem (why anyone should care)

When a model reads a long prompt, it stores a Key and a Value for every token,
at every layer. This is the **KV cache**. It grows linearly with prompt length.
At long context (say 16k–128k tokens) it becomes a huge chunk of the memory and
bandwidth cost of inference.

**KV compression** = "we can't keep all of these tokens; which do we throw
away?" Every method is just a different answer to that one question.

---

## The idea (our answer)

Two intuitions, combined.

1. **Keep tokens that are geometrically unique.** Picture every key as an arrow.
   Some arrows point in crowded directions (lots of near-duplicates); some point
   in lonely directions nobody else covers. Ridge leverage scores measure exactly
   this "how much does this key stick out from the crowd." Keep the ones that
   stick out — dropping a near-duplicate costs almost nothing.

2. **...but unique ≠ useful.** A token can be one-of-a-kind and totally irrelevant
   to the prompt. So we add a second signal, `omega`: how hard the prompt's own
   queries actually hit this key. High omega = the model was really paying
   attention to this token.

We combine the two (keep a token if *either* signal says keep), and weight the
whole thing by how "substantial" the token's value vector is.

---

## The key argument — why not just use ridge leverage alone?

This is the intellectual core of the paper, and it needs **evidence**, not just
assertion. The plan:

- Show that **ridge leverage alone is not enough** — geometrically-unique keys
  are not the same as the keys the model actually attends to. (Experiment:
  correlation between ridge-leverage scores and true attention scores; or show
  cases where they sharply disagree.)
- That disagreement is *why* we pull in the Gram matrix / query signal (`omega`)
  and the `gamma` tilt. Each added component should earn its place.
- **The ablation table is the backbone.** Strip each component (leverage only,
  +omega, +value weighting, +gamma) and show what each one buys in accuracy.
  This single table simultaneously explains the method and proves each piece
  matters.

---

## Where we sit vs. what exists (positioning + baselines)

The closest neighbor is **Compactor** — it also mixes a leverage-style signal
with attention. **But Compactor uses materialized attention scores.** That's a
real limitation: materializing the attention matrix is incompatible with
FlashAttention, which is what long-context inference actually runs on. Ridge is
query-*aware* but never materializes attention — so it composes with FlashAttention.
**This is our differentiator; say it clearly.**

### The honest framing: "as good as the best, but practical"

We know Ridge is **roughly equal to (or slightly below) Compactor on accuracy.**
That's fine — it just means "most accurate" is NOT our claim, and we don't
pretend it is. Our claim is the two-axis one:

> Ridge **matches** the best method (Compactor) on accuracy **while** working with
> FlashAttention, which Compactor cannot.

Consequences of this framing (important):
- **Always include Compactor in the plots.** Hiding your strongest competitor
  reads as hiding a loss → instant reject. Showing an honest tie + a practical
  win is *stronger* than a suspicious 0.3% lead.
- **Because we tie on accuracy, the FlashAttention advantage is the whole
  contribution — so we must SHOW it, not just assert it.** That promotes the
  cost/compatibility evidence from "nice-to-have" to a co-equal must-have (see
  the two-axis result plan below).
- If we *can't* yet demonstrate a real memory/speed/compatibility win (kernels
  are future work), the intellectual contribution leans instead on the
  **ablation** + the **leverage-vs-true-attention insight**. Be clear-eyed about
  which we can actually show at submission time.

Baselines to compare against (span the space so we're not cherry-picking):
- attention-based: **Compactor**, **SnapKV**
- norm / geometry based: **knorm**, a **CUR / leverage** method
- (round out with whatever the sweeps already cover)

All comparisons on **one axis**: accuracy at a matched compression ratio
(same kept-token budget per head) on a long-context benchmark (RULER / LongBench).
Matched-budget discipline is what reviewers check — keep it apples-to-apples.

---

## Bonus evidence — it generalizes to hybrid models

Ridge also works on hybrid attention/Mamba models (NemotronH), where only a few
layers even have a KV cache. This shows the method isn't tied to one
architecture. **Keep this short** — a paragraph or an appendix, not a headline
section. Generality is a nice bonus, not the main claim.

---

## Future work (honest open ends)

- Efficiency: the scoring can be pushed down into custom **Triton kernels** so the
  compression itself is cheap. (We're not claiming the systems win yet — we're
  claiming the *selection quality*, and flagging the kernel work as the path to
  the full efficiency story.)

---

## The four-beat outline (what goes where)

1. **Problem + idea.** KV cache is the long-context bottleneck → ridge-leverage
   intuition → why leverage alone fails (the correlation evidence) → so we add
   the Gram/query term and gamma. *This is the depth of the paper.*
2. **Positioning + baselines.** Closest work = Compactor; its attention-score
   dependence is the gap we fill (FlashAttention compatibility). Compare against
   Compactor / SnapKV / knorm / CUR at matched ratios.
3. **Results.** Main accuracy-vs-ratio comparison + the ablation table. Hybrid-model
   generality as a short bonus.
4. **Future work.** Triton kernels for the efficiency story.

---

## What we need to run (each run = one claim it proves)

Start with **ONE model + ONE benchmark** (Llama-3.1-8B on RULER 16k — the
`ridge-ruler16k-ablations` branch is already here). More models/benchmarks are
*enrichment for later*, not required for a first workshop paper.

**Lock these 4 knobs once and keep them fixed** so everything is comparable:
model = Llama-3.1-8B · benchmark = RULER 16k · ratios = ~3 points (keep
75% / 50% / 25%) · metric = accuracy at matched kept-token budget per head.

### The two-axis result (both are must-haves)

| # | Run / table | Claim it proves | Output |
|---|---|---|---|
| **1. Accuracy** | Ridge vs **Compactor**, SnapKV, knorm, CUR — accuracy at each ratio | "We lose nothing on quality — Ridge sits on top of Compactor" | accuracy-vs-ratio line plot |
| **2. Cost / compatibility** | Ridge vs Compactor: needs attention matrix? memory? latency at long ctx? | "...and we're the practical one (FlashAttention-compatible, Compactor isn't)" | table (needs-attn Y/N) + memory/latency numbers if feasible |
| **3. Ablation** | Ridge pieces on/off: leverage → +omega → +value-norm → +gamma | "every component earns its place" (backbone; doubles as the method explanation) | ablation table |
| **4. Insight** | leverage score vs **true** attention received, per key/layer | "unique ≠ attended — which is *why* we combine both signals" | scatter / correlation figure |

Run 2 is the one that used to be "nice-to-have" — because we tie Compactor on
accuracy, it's now core. If it can't be shown yet, lean the contribution on 3+4.

### Bonus (add if time allows)

- [ ] NemotronH hybrid-model run — shows generality. One paragraph / appendix.
- [ ] A second benchmark (LongBench) or second model — answers "does it generalize?"

### Honest minimum for a real paper

Runs **1 + 3** on one model + one benchmark = a publishable workshop paper on
their own. Run **4** makes it *smart*; run **2** makes the FlashAttention claim
*real*; the bonuses make it *complete*.

Sweeps are still running as of writing — the paper's timing follows the sweeps,
not the calendar.

---

## How we'll actually write this (workflow + AI ground rules)

### The order of operations (writing is the LAST step)

Nobody writes dense paper-prose top-to-bottom. A paper is *assembled*, and the
fancy writing is the last 10%:

1. **Figures & tables first.** Make the plots that carry the claim — the
   Ridge-vs-baselines accuracy-vs-ratio curve and the component ablation table —
   *before* writing prose. If the figures tell the story, the paper is basically
   done and the text just narrates them. If they don't, no writing saves it.
2. **One-sentence claim, then a skeleton.** Section headings + 2–3 bullets each.
   (This file already is that skeleton.)
3. **Ugly first draft.** Fill the bullets with plain sentences — write it like
   you're explaining Ridge to a labmate out loud. Nobody sees this draft; getting
   the *logic* right and plain is the whole job here.
4. **Revise for flow, then polish for tone last.** "Make it sound like a paper"
   is a separate, mechanical, final pass — not something to worry about while
   drafting.

The density in published papers is a **learnable dialect** ("we ablate", "at
matched budget", "we observe that"), not a talent. Clarity beats fancy — tired
reviewers reward the paper they understand in one pass. `ridge_explained.md` is
already good technical writing; the paper is that, compressed, costume on.

### AI-use ground rules (safe line for a first-time author)

AI is an **editor and translator, not the author.** It makes the draft read
better; it does not know if the draft is *true*.

- ✅ Polish plain drafts into paper tone; tighten for the page limit; suggest
  structure; explain conventions.
- ✅ **Disclose** AI use if the venue asks (usually a one-line statement); check
  the specific workshop's call — workshops can set stricter rules than the main
  conference.
- ❌ Never let AI invent **results, numbers, or citations** (it fabricates
  plausible-but-fake papers — verify every citation by hand).
- ❌ Never auto-generate a **review** if assigned to review others' papers
  (reciprocal-review duty; this is what the ICLR 2026 crackdown actually
  targeted — flagged by watermark traps).

Core policy everywhere: *you may use LLMs, but you disclose and you are fully
responsible for every claim.* The ideas and correctness are yours. See
[venue_and_submission_guide.md](venue_and_submission_guide.md) for the policy
sources.
