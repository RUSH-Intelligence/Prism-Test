# NemotronH KV-Cache Compression on RULER-16k

**Model:** `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` — a Mamba2–attention hybrid (42 layers:
**4 full-attention**, 21 Mamba2, 17 MLP). Only the 4 attention layers hold a K/V cache, so
compression applies to them alone (Mamba conv/SSM state is untouched).

**Setup:** RULER-16k, **all 13 subsets · 100 samples/subset (1,300 per cell)** · research backend,
**single-pass prefill**, `flash_attention_2`, fast Mamba kernels (`use_mamba_kernels`), greedy,
`max_new_tokens=128`, `max_model_len=18000`, H200 · `compression_ratio` = **fraction pruned**
(0.2 ⇒ keep 80% of the cache). The longest subsets (`qa_1`, `qa_2`, `cwe`) template above 18k
tokens and are truncated to 18,000 — uniformly across every method/ratio, so the *relative*
comparison is unaffected.

> Full 6-method × 4-ratio (+ full-cache baseline) sweep; **25/25 cells complete**.

## Accuracy (overall, %) vs compression ratio

| Method | 0% (full) | 20% | 40% | 60% | 80% |
|---|---|---|---|---|---|
| **compactor** | 93.43 | **91.49** | **87.08** | **84.41** | **77.02** |
| **keydiff** | 93.43 | 90.89 | 86.61 | 81.95 | 76.67 |
| knorm | 93.43 | 82.66 | 78.71 | 75.46 | 59.70 |
| snapkv | 93.43 | 84.10 | 72.23 | 56.61 | 38.55 |
| pyramidkv | 93.43 | 78.87 | 59.68 | 41.85 | 33.47 |
| expected_attention | 93.43 | 46.95 | 37.54 | 28.39 | 17.37 |

Full-cache baseline (no compression): **93.43** overall.

## Per-subset accuracy — full-cache baseline (%)

| cwe | fwe | nm_1 | nm_2 | nm_3 | nmq | nmv | ns_1 | ns_2 | ns_3 | qa_1 | qa_2 | vt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 96.80 | 99.33 | 100 | 91.0 | 85.0 | 100 | 100 | 100 | 100 | 100 | 78.0 | 66.0 | 98.40 |

(`nm`=niah_multikey, `nmq`=niah_multiquery, `nmv`=niah_multivalue, `ns`=niah_single.) The full
per-subset × ratio matrix for every method is in the generated artifact
`/scratch/sj157/nemotron_ruler16k_kv_report.md`.

## Retained attention-cache length

`per_layer_min = 0` for every Mamba/MLP layer — compression touches the 4 attention layers only.
Per attention layer the retained length is ≈ `(1−ratio)·min(prompt, 18000)` (e.g. for an
18k-capped prompt: 20%→14,400, 40%→10,800, 60%→7,200, 80%→3,600). PyramidKV retains more than the
flat target: its per-layer pyramid budget, sampled at the 4 attention layers' absolute depths
(12/17/24/32 of 42), averages above the uniform line.

## Observations

- **Ranking (robustness to pruning): compactor ≳ keydiff > knorm > snapkv > pyramidkv > expected_attention.**
- **compactor and keydiff are clearly best** — both stay above **76%** even at 80% pruning (≈0.82× the
  full-cache score) and lose <3 points at 20%. compactor edges keydiff at every ratio.
- **knorm is the most robust of the cheap scorers at high pruning**: it trails snapkv at 20%
  (82.66 vs 84.10) but overtakes it as pruning increases (59.70 vs 38.55 at 80%) — key-norm scoring
  degrades gracefully.
- **snapkv → pyramidkv → expected_attention** degrade progressively faster. PyramidKV's pyramid over
  only 4 sparse attention layers is more aggressive than its nominal budget suggests.
- **`expected_attention` is anomalously weak** (46.95 already at 20%). Its score rotates query
  statistics by an averaged-RoPE matrix that reduces to identity on this no-RoPE model; that
  adaptation looks ill-suited to NemotronH and is worth a closer look (possible follow-up).
- Compression is correct and **attention-only**: retained ≈ `(1−ratio)·prompt`, Mamba/MLP untouched.

## Reproduce

Config: `evaluate/evaluate_nemotron_kv.yaml` (single-pass; set `kv_compressor` + `compression_ratio`).
Fleet launcher: `/scratch/sj157/launch_nemotron_fleet.sh` (6 methods × 4 ratios + baseline = 25 H200
jobs; staggered starts to avoid the HF-`datasets` FileLock race; one cell per `nb_<method>_r<ratio>`
SLURM job over `/scratch/sj157/nemotron_bench.sbatch`). Aggregate with
`/scratch/sj157/aggregate_nemotron.sh`. Raw metrics + logs under
`/scratch/sj157/results_nemo_bench/<method>_r<ratio>/`.
