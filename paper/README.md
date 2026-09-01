# Ridge paper — LaTeX skeleton

ICLR-workshop skeleton wired to the four-beat outline in
[../notes/paper_submission/paper_story.md](../notes/paper_submission/paper_story.md).
Venue/logistics: [../notes/paper_submission/venue_and_submission_guide.md](../notes/paper_submission/venue_and_submission_guide.md).

## Files
- `main.tex` — the paper. Sections map 1:1 to the four beats; each Result has a
  placeholder figure/table with a `TODO` and its run status.
- `references.bib` — placeholder keys. **Every entry must be verified by hand**
  (AI must not invent citations).

## Getting it onto Overleaf
1. Overleaf gallery → search **ICLR** → **Open as Template** (ICLR 2025 is fine;
   the style barely changes year to year). That project ships
   `iclr2025_conference.sty`, the `.bst`, `fancyhdr.sty`, and `math_commands.tex`.
2. Replace the template's `main.tex` with this one; add `references.bib`.
3. The `\usepackage{...}` / `\bibliographystyle{...}` lines say
   `iclr2025_conference` — confirm that matches the `.sty` filename in the
   template's file list.
4. Open the PR for Sahil's review (meeting action item #3).

## Status of the four results (as of 2026-08-31)
| Result | In `main.tex` | Data status |
|---|---|---|
| 1. Accuracy vs baselines | `sec:res-accuracy` | **Not run** — baselines commented out in `sweep.yaml` (critical path) |
| 2. Cost / FlashAttn | `sec:res-cost` | Not started — kernel/H100 work |
| 3. Ablation | `sec:res-ablation` | Partial — γ sweep done (LongBench); other rungs pending |
| 4. Insight (leverage vs attn) | `sec:res-insight` | Coverage logging wired; figure not produced |

Main benchmark = **RULER 16k**; LongBench data becomes the appendix
(`app:longbench`).
