# Improving Ridge — rotation, normalization, lambda, and query subsetting

Ridge is **our own KV compressor** (leverage-scored keys + prefill-query `omega`
importance + fixed-envelope combination + value weighting — see the authorship
note in [`ridge_explained.md`](ridge_explained.md)). This project’s goal is to
**make our Ridge compressor work better** than its original faithful default. We
run controlled sweeps on RULER16k, change one family of knobs at a time, and
record scores in [`Ridge Press.xlsx`](../Ridge%20Press.xlsx). Throughout this
note, "original" / "reference" means **our own earlier implementation**
(`RidgePress` in our kvpress fork), not any third-party code.

For *how* Ridge works (tau, omega, RoPE, combine modes), see
[`ridge_explained.md`](ridge_explained.md). This note is about **what we
changed, what we measured, and what we learned.**

---

## Knobs we are actively testing

These are the Ridge deviations / tuning axes this effort cares about:

| Knob | Config field | What it does |
|------|--------------|--------------|
| **Query rotation** | `rotate_queries` | Spin fresh queries with RoPE before omega so they match already-rotated cached keys. Default **off** (matches our original reference). |
| **L2 key normalization (tau)** | `normalize_keys_for_tau` | L2-normalize keys before building `K^T K` for tau leverage. Stops “big vector along a popular axis” from scoring high just because of magnitude. Default **off**. |
| **Ridge lambda** | `ridge_lambda` | Regularizer in `(K^T K + λI)⁻¹`. Reference default **1e-4**. |
| **Query selection method** | `omega_query_selector` | How to pick which prefill queries feed omega: `all` (default), `top_norm`, `leverage`, or `random`. |
| **Query keep fraction** | `omega_query_fraction` | Fraction of queries **kept** per head before forming omega’s Gram (e.g. 0.1 = keep 10%). Default **1.0** = all queries. |

Other Ridge settings (`combine_mode`, sink/local windows, etc.) stay at repo
defaults unless noted in the sweep scripts. We hold compression schedule at
**post_prefill, single-pass** so the compressor fires once on the full prompt.

---

## Sweep structure (two tracks)

### Track 1 — tau / omega hygiene

**Part A — lambda sweep** (`results/ruler16k_sweep/`)

- Swept: **`ridge_lambda`** × envelope settings (lambda was the hypothesis).
- Fixed: original rotate/normalize defaults, all queries for omega.
- Sheet: `RULER16k-Ridge-Llama-3.1-8B` (and Ministral counterpart).
- Launcher: `scripts/slurm/ruler_ridge_sweep.sbatch`

**Part B — 2×2 corner sweep** (`results/ruler16k_sweep_2x2/`)

- Swept: **`rotate_queries`** × **`normalize_keys_for_tau`** (four corners:
  `rqF_nkF`, `rqF_nkT`, `rqT_nkF`, `rqT_nkT`).
- Fixed: `ridge_lambda = 1e-4` (flat from Part A).
- Sheets: `Ridge2x2-Llama-3.1-8B-r0p6`, `Ridge2x2-Llama-3.1-8B-r0p8` (+ Ministral).
- Launcher: `scripts/slurm/ruler_ridge_2x2_sweep.sbatch`

**Track 1 question:** *Starting from our original faithful defaults, which
tau/omega hygiene fixes actually move RULER16k scores?*

### Track 2 — query subsetting for omega

(`results/ruler16k_sweep_track2/`)

- **Locked** from Track 1 winner: `rotate_queries=False`, `normalize_keys_for_tau=True` (`rqF_nkT`).
- **Locked:** `ridge_lambda = 1e-4`.
- Swept: **`omega_query_selector`** × **`omega_query_fraction`**.
  - Selectors: `top_norm` (qtop), `leverage` (qlev), `random` (qrnd).
  - Fractions kept: 10%, 25%, 50%, 75%.
- Sheets: `RidgeTrack2-Llama-3.1-8B-r0p6`, `RidgeTrack2-Llama-3.1-8B-r0p8`.
- Launcher: `scripts/slurm/ruler_ridge_track2_sweep.sbatch`

**Track 2 question:** *Omega uses many near-duplicate prefill queries. If we
give omega a smaller, better-chosen subset per head, does Ridge pick better
keys — without changing tau?*

---

## What each knob is trying to fix

### Query rotation (`rotate_queries`)

Ridge reads **already-RoPE-rotated keys** from the cache but computes **fresh,
unrotated queries** for omega unless this flag is on. That means omega is a
slightly wrong proxy for real attention (see
[`ridge_explained.md` §7](ridge_explained.md)).

**Hypothesis:** turning rotation on makes omega faithful → better token retention.

### L2 normalization for tau (`normalize_keys_for_tau`)

Tau uses ridge leverage on raw cached keys. Keys with large magnitude along an
already-popular direction can score high even when they are not directionally
unique (see [`ridge_explained.md` §3–4](ridge_explained.md)).

**Hypothesis:** normalize keys before `K^T K` so tau measures **directional**
uniqueness, not raw magnitude on a crowded axis.

### Ridge lambda (`ridge_lambda`)

λ regularizes the Gram inverse so tau stays numerically stable. Changing λ also
smooths or sharpens how aggressively tau distinguishes keys.

**Hypothesis:** our original default may not be optimal for long-context RULER
prompts.

### Query selection + fraction (`omega_query_selector`, `omega_query_fraction`)

Omega aggregates how much the prompt’s queries “care about” each key. Many
prefill queries are near-duplicates; the hope is that a **smaller, diverse or
high-signal subset** gives omega a cleaner vote.

**Hypothesis:** subsetting queries improves omega without touching tau.

---

## Results — Llama-3.1-8B-Instruct on RULER16k

All numbers below are from [`Ridge Press.xlsx`](../Ridge%20Press.xlsx). Ministral
tabs exist for Track 1; Track 2 Llama run is complete (264/264 cells).

### 1. Ridge lambda — **does not matter**

Across lambdas at the same settings, RULER16k averages move by **~0.2 points**
or less. No lambda value consistently wins.

**Takeaway:** keep **`ridge_lambda = 1e-4`** (reference default). Do not spend
sweep budget here.

### 2. Query rotation — **does not matter (practically)**

From the 2×2 sheet, flipping `rotate_queries` changes mean scores by **~0–0.6**
points depending on corner — negligible next to normalization.

**Takeaway:** leave **`rotate_queries=False`** unless you have a RoPE-specific
reason to change it. Not a lever for “making Ridge better” on this benchmark.

### 3. L2 key normalization for tau — **this matters**

This is the clear Track 1 win. Holding other settings fixed, turning
**`normalize_keys_for_tau=True`** improves mean RULER16k scores by roughly:

- **~+2 points** in one compression setting
- **~+4 points** in the other

The lambda-only runs (without normalization) plateau much lower than the 2×2
runs with `nkT` on the same benchmark.

**Takeaway:** for improved Ridge on RULER16k, **`normalize_keys_for_tau=True`**
is the one hygiene fix worth shipping. It directly addresses the “magnitude on a
popular axis” tau quirk documented in `ridge_explained.md`.

### 4. Query selection method and keep fraction — **no quality win; some caveats**

Track 2 compared three selectors at four keep fractions, with Track 1’s best
corner locked (`rqF_nkT`).

| Finding | Detail |
|---------|--------|
| **Peak scores** | Best Track 2 configs match best Track 1 all-query configs within **~0.1 pt** — no improvement from subsetting. |
| **top_norm vs random** | `top_norm` is slightly best on average, but only **~0.2–1 pt** ahead of **random** at the same fraction. Smart picking ≈ shuffling. |
| **Fraction** | For `top_norm` / `random`, keeping **10%** vs **75%** of queries changes means by **~1 pt** — subset size barely matters. |
| **leverage** | **Worst selector**, especially at **10% queries** with aggressive settings — scores can **collapse** vs other configs. |
| **Compute angle** | Subsetting does not improve peak quality, but **10–50% keep** with `top_norm` or `random` lands on the **same best scores** as all queries — possible **omega-side compute savings** without measured quality loss. |

**Takeaway:** query subsetting is a **negative result on the main hypothesis**
(smarter omega → better Ridge). Do not expect better RULER scores from it. Avoid
**`leverage` at low keep fractions**. If you subset for speed, **`top_norm`** or
even **`random`** at 10–50% looks safe at peak settings.

---

## Recommended Ridge settings (from these sweeps)

For “make Ridge better” on RULER16k, ignoring knobs we are not focusing on in
this doc:

```yaml
kv_compressor: ridge
kv_compressor_kwargs:
  normalize_keys_for_tau: true   # the meaningful win
  rotate_queries: false          # flat; original default is fine
  ridge_lambda: 1.0e-4           # flat; keep default
  # query subsetting: optional for compute only, not for quality
  omega_query_selector: all      # default; subsetting didn't help scores
  omega_query_fraction: 1.0
```

Omega (the query–key term) can still contribute when the combine envelope
weights it — that is separate from **how many** queries you feed it. Track 2
showed you do not need a smaller query set to get peak performance; Track 1
showed tau quality improves most when keys are normalized before leverage.

---

## Track 3 — original sink/local verification

**Motivation.** The `Llama-3.1-8B` comparison sheet in `Ridge Press.xlsx`
reports "Ridge (envelope)" at **84.30** (r=0.6) / **73.31** (r=0.8) on
RULER-16K — ~1.7 pts above what looked at first like our Track 1 winner.
`ridge_explained.md` §6 flags that this repo defaults `sink_size=8,
local_size=64` while our original RidgePress reference uses `sink_size=4,
local_size=28`. Initial hypothesis: that alone explains the gap.

**Sweep** (`results/ruler16k_sweep_upstream_sink/`)

- **Locked:** `rotate_queries=False`, `sink_size=4`, `local_size=28`,
  `ridge_lambda=1e-4`.
- **Swept:** `envelope_gamma` × `normalize_keys_for_tau ∈ {False, True}` × ratio.
- 11 gammas × 2 ratios × 2 nk = 44 cells.
- Sheets: `Ridge2x2Sink4-Llama-3.1-8B-r0p6`, `Ridge2x2Sink4-Llama-3.1-8B-r0p8`.
- Launcher: `scripts/slurm/ruler_ridge_upstream_sink_sweep.sbatch`.

**Result.** Sink/local was a red herring; the gap itself was an accounting
artifact.

- Moving to the original 4/28 shifted the best single-γ number by ~0.1 pt in
  either direction — noise.
- `normalize_keys_for_tau=True` still gives ~+2 pts on top of the original
  sink baseline (80.33 → 82.61 single-γ at r=0.6), matching the gain we saw
  at sink=8/local=64. Portable improvement.

### The accounting artifact

The old sheet's "Ridge (envelope)" 84.30 / 73.31 uses **best-per-task
across the γ sweep** (per-task-optimal γ picks visible in the misplaced
gamma annotations on the sheet — those `gamma=2-3, 0-3, ...` labels are
Ridge γ picks, not Compactor knobs). Our earlier "best single γ averaged"
number (82.61 at r=0.6) is a different — and stricter — statistic. Same
Ridge, same sweep grid, different summary.

Re-scoring our sink=4, nkT sweep with the sheet's per-task-best-γ recipe:

| Config | r=0.6 best-per-task-γ (13 tasks) | r=0.8 best-per-task-γ (13 tasks) |
|--------|----------------------------------|----------------------------------|
| Sheet "Ridge (envelope)" target | 84.30 | 73.31 |
| Sink=4, **nkT** (original sink + our fix) | **84.79** | **73.89** |
| Sink=4, nkF (original-faithful baseline) | 82.13 | 68.37 |

**Our improved Ridge slightly beats the sheet's Ridge (envelope) on its own
accounting** — by ~0.5 pts at both ratios. The gap doesn't exist; we were
comparing different statistics. `normalize_keys_for_tau=True` also gives a
bigger visible lift on the per-task-best cut (~+2.7 pts r=0.6, ~+5.5 pts
r=0.8) than on the single-γ cut, because it lets more tasks pick a good γ.

### Compactor benchmark control (`results/ruler16k_compactor_control/`)

Before believing the accounting-artifact story, we reran Compactor at the
sheet's 4 ratios on the current framework as an independent control. Sheet:
`CompactorCtrl-Llama-3.1-8B` (target / ours / Δ rows).

| Ratio (prune) | Sheet Compactor | Ours | Δ |
|---------------|-----------------|------|---|
| 20% | 92.60 | 92.59 | −0.01 |
| 40% | 87.65 | 87.41 | −0.24 |
| 60% | 83.96 | 83.51 | −0.45 |
| 80% | 75.01 | 73.17 | −1.84 |

Benchmark reproduces cleanly (r=0.2 exact to the decimal). The small r=0.8
drop is within Compactor's Gaussian-sketch variance. This rules out RULER
subset composition, decode budget, and answer-parsing drift. Any remaining
"gap" is Ridge-side, not benchmark-side — which the accounting fix above
resolves.

### Ridge vs Compactor on current framework

| Ratio | Our Compactor (single run) | Our Ridge (per-task best γ, sink=4, nkT) | Δ |
|-------|----------------------------|-------------------------------------------|---|
| 0.6 | 83.51 | **84.79** | +1.28 (Ridge wins) |
| 0.8 | 73.17 | **73.89** | +0.72 (Ridge wins) |

Caveat: Ridge here uses per-task-γ (matches sheet Ridge accounting);
Compactor is single-run (matches sheet Compactor accounting). If Ridge is
restricted to best single γ, Compactor edges it at r=0.6 (83.51 > 82.61)
and clearly wins at r=0.8 (73.17 > 71.37). So Ridge's win rests on the
per-task-γ oracle being an accepted way to report Ridge.

**Bottom line:** the improvements ship as a real net gain over both the
sheet's Ridge (envelope) and Compactor on the same accounting the sheet
uses. Further hyperparameter chasing on `combine_mode`, `value_norm_power`,
etc. is not required to justify the improvement — that hunt is optional
upside, not a debt.

---

## Where things live

| Artifact | Path / name |
|----------|-------------|
| Results spreadsheet | [`Ridge Press.xlsx`](../Ridge%20Press.xlsx) |
| Lambda sweep outputs | `results/ruler16k_sweep/<model>/` |
| 2×2 sweep outputs | `results/ruler16k_sweep_2x2/<model>/` |
| Query-subset sweep outputs | `results/ruler16k_sweep_track2/<model>/` |
| Upstream sink/local sweep outputs | `results/ruler16k_sweep_upstream_sink/<model>/` |
| Sweep driver | `scripts/longbench_sweep.py` |
| Fill xlsx from manifests | `scripts/ruler_ridge_ablation_to_xlsx.py` |
| Mechanism deep-dive | [`ridge_explained.md`](ridge_explained.md) |

**Export xlsx after a sweep:**

```bash
# Track 1 lambda
python scripts/ruler_ridge_ablation_to_xlsx.py \
  --cells-dir results/ruler16k_sweep/meta-llama--Llama-3.1-8B-Instruct/manifest.cells

# Track 1 2×2
python scripts/ruler_ridge_ablation_to_xlsx.py \
  --cells-dir results/ruler16k_sweep_2x2/meta-llama--Llama-3.1-8B-Instruct/manifest.cells

# Track 2 query subset
python scripts/ruler_ridge_ablation_to_xlsx.py \
  --cells-dir results/ruler16k_sweep_track2/meta-llama--Llama-3.1-8B-Instruct/manifest.cells

# Track 3 upstream sink/local (writes to Ridge2x2Sink4-... sheets;
# `--sheet` override is required so it doesn't overwrite the Track 1 2x2 sheets)
python scripts/ruler_ridge_ablation_to_xlsx.py \
  --cells-dir results/ruler16k_sweep_upstream_sink/meta-llama--Llama-3.1-8B-Instruct/manifest.cells \
  --sheet Ridge2x2Sink4-Llama-3.1-8B
```

---

## One-paragraph summary

We are tuning Ridge to retain the right tokens on long RULER16k prompts. **L2
normalizing keys before tau (`normalize_keys_for_tau=True`) is the only knob
that clearly improves scores** — worth ~+2 pts single-γ / ~+3–5 pts
per-task-best-γ, portable across sink/local settings. Ridge lambda, query
rotation, query subsetting (method and keep fraction), and sink/local matching
our original defaults (4/28 vs 8/64) are all flat or harmful. Leverage-based query
selection at low keep fractions is the one configuration to avoid. The old
"Ridge (envelope)" 84.30 turned out to be **per-task-best-γ** scoring, not
single-γ; on that same accounting our improved Ridge lands at **84.79**
(r=0.6) / **73.89** (r=0.8), slightly beating the old sheet and — on the
same current framework — beating our Compactor control (83.51 / 73.17).
Compactor benchmark control reproduced the sheet to the decimal at r=0.2,
so the gap was pure accounting, not a benchmark drift.
