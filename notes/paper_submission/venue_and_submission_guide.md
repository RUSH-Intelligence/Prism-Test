# Venue & Submission Guide (workshop paper, first-timer notes)

Everything we figured out about *where* to submit and *how* it works. Written for
someone submitting their first paper. Companion to
[paper_story.md](paper_story.md).

---

## The plan in one line

Aim Ridge at an **efficiency / long-context / ML-systems workshop**, deadline
around **February 2027** (ICLR 2027 cycle, or a NeurIPS/ICML 2026 workshop if a
good one lands sooner). It's a **non-archival** venue — low-risk, and we can
expand the same work into a full conference paper later.

**Timing follows the sweeps, not the calendar:** pick a deadline ~3–4 weeks after
we expect clean numbers, so there's a writing buffer.

---

## Timing reality

ICLR **2026** already happened (April 2026, Rio de Janeiro). Its workshop paper
deadlines were ~Feb 5, 2026. That cycle is closed. We're aiming at the **next
cycle (ICLR 2027)**. The yearly rhythm is stable, so expect roughly:

| Milestone | Rough timing (ICLR 2027, by analogy to 2026) |
|---|---|
| Workshop list announced | ~Dec 2026 / Jan 2027 |
| Workshop paper deadlines | ~Feb 2027 (AOE) |
| Author notification | ~early March 2027 |
| Conference + workshop days | ~April 2027 |

The exact ICLR 2027 workshop list isn't out yet. But the *categories* repeat
every year (efficient ML, long-context/memory, ML systems), so we plan around
those now and slot into the specific workshop once the list drops.

---

## What a "workshop paper" even is

- A **workshop** is a one-day themed event inside the main conference (e.g.
  "Workshop on Efficient ML"). Each workshop accepts **many** papers — often
  50–150. (The ICLR "virtual" event pages list *individual accepted papers*, which
  is why it looked like one link = one paper; each of those belongs to a parent
  workshop with dozens more. Each workshop also has its **own website** with the
  call for papers, dates, and organizers — that's where you actually submit.)
- Workshop papers are **short** (ICLR requires every workshop to accept **4-page**
  submissions; some allow up to 6–9). References and appendix usually don't count
  toward the limit.
- Most are **non-archival** = *not* an official publication. This is the point for
  us: you get reviews + a talk/poster, and you can still submit the expanded work
  to a real conference afterward. No-regret move for a first paper.
- Deadlines are **AOE** ("Anywhere on Earth", UTC-12) — effectively end-of-day
  somewhere, buys ~a day.

The content bar for a workshop: **one clear, defensible claim backed by clean
experiments.** Not state-of-the-art, not a complete story. Interesting +
correct wins.

---

## Blind review — the three flavors

"Blind" = who knows whose identity during reviewing (to reduce bias).

- **Double-blind:** reviewers don't know the authors, authors don't know the
  reviewers. The PDF must be **anonymized** — no names, no "our lab at UMD," no
  direct GitHub link, and refer to your own prior work in third person
  ("Singhal et al." not "we previously..."). Most common at competitive venues.
- **Single-blind:** reviewers know who you are; you don't know them. No
  anonymization needed. Some workshops use this.
- **Non-blind / open:** identities visible to all. Rare, small workshops.

Each workshop's call states which. Practical move: **write it so it's easy to
anonymize** (don't bake your name into the prose); de-anonymize later if the
venue turns out single-blind.

---

## OpenReview — the submission website

- **OpenReview** (openreview.net) is just the portal where you upload the PDF and
  where reviewing happens. Not scary — it's plumbing.
- Flow: make an account → find the specific workshop → click submit → upload PDF +
  fill a form (title, abstract, authors) → done. Later, reviews appear there and
  you can post responses.

---

## Do we present it somewhere?

Yes. If accepted, you (or a co-author) present at the workshop, which is a real
session at the conference (in person, usually with a virtual option). Almost
always a **poster** (stand next to a printed poster, people come chat); a smaller
number get a short **spotlight talk**. This is the best part — direct feedback
from people who work on exactly this problem. That feedback loop is the whole
reason we're starting at a workshop.

---

## Format / template

- Papers use a **fixed LaTeX template** (the venue's style file, usually available
  as an Overleaf template). You fill it in — you don't design the layout.
- ICLR-family workshops typically use the ICLR style. The specific workshop's page
  links the exact template + page limit. Grab it once the venue is chosen.
- Page limit is a **hard rule** (references/appendix usually excluded). Check the
  exact number per workshop.

### Templates — start now, don't wait

- **You don't have to wait for a "2027" template.** The ICLR LaTeX style barely
  changes year to year — grab the **current ICLR template today** and draft in it;
  swapping to the official 2027 file later is a find-and-replace, not a rewrite.
- **The *look* is per-conference; the *page limit / fine print* is per-workshop.**
  All ICLR workshops share the ICLR visual style, but each workshop sets its own
  page limit (4 vs 6 vs 9) and may add a banner or tweak the header. ICLR ≠
  NeurIPS ≠ ICML — each conference family has its own separate style.
- **Default to the tightest limit (4 pages).** Easier to expand a tight draft into
  6 pages later than to hack 6 down to 4 under deadline — and writing short is the
  discipline workshop reviewers reward.

### Where to get the LaTeX

1. **Overleaf gallery (easiest):** [overleaf.com/latex/templates](https://www.overleaf.com/latex/templates)
   → search "ICLR" → **"Open as Template"** copies the latest (currently ICLR 2026)
   into your account, editable in-browser, no LaTeX install. This is what most
   people do.
2. **Official style files:** linked from the [ICLR Call-for-Papers / author kit](https://iclr.cc/Conferences/2026/CallForPapers)
   page — use if you want the canonical `.sty`/`.zip` or write locally.
3. **The chosen workshop's own page:** once the ICLR 2027 workshop list is out, the
   workshop links *its* template (with the right page limit baked in) — use that
   for the final submission.

For now: option 1 with the current ICLR 2026 template. It's ~identical to 2027.

---

## Next steps (when sweeps firm up)

1. Pick the exact workshop from the live list (check its blind-review type,
   page limit, template, deadline).
2. Lock Ridge's single main claim (see [paper_story.md](paper_story.md)).
3. Turn the four-beat outline into a section-by-section plan with the specific
   figures/tables each part needs.

---

## Sources

- [ICLR 2026 Dates & Deadlines](https://iclr.cc/Conferences/2026/Dates)
- [ICLR 2026 Workshops (virtual event list)](https://iclr.cc/virtual/2026/events/workshop)
- [ICLR 2026 workshops blog announcement](https://blog.iclr.cc/2026/01/13/iclr2026-workshops/)
- [SPOT @ ICLR 2026 (example workshop site)](https://spoticlr.github.io/)
