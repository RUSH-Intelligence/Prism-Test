# NemotronH KV-Cache Compression on RULER-16k

**Model:** `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` — a Mamba2–attention hybrid (42 layers:
**4 full-attention**, 21 Mamba2, 17 MLP). Only the 4 attention layers hold a K/V cache, so
compression applies to them alone (Mamba conv/SSM state is untouched).

**Setup:** RULER-16k (`niah_single_1`, `niah_single_3`, `niah_multikey_3`; 50 samples/subset;
prompts ≈ 15.6–16.2k tokens) · research backend, **single-pass prefill**, `flash_attention_2`,
fast Mamba kernels (`use_mamba_kernels`), greedy, `max_new_tokens=64`, H200 ·
`compression_ratio` = **fraction pruned** (0.2 ⇒ keep 80% of the cache).

> Numbers are from a 50-sample/subset run (indicative, not the full 200/subset × 13-subset RULER).

## Accuracy (overall, %) vs compression ratio

| Method | 0% (full) | 20% | 40% | 60% | 80% |
|---|---|---|---|---|---|
| **keydiff** | 92.67 | **88.00** | **77.33** | **72.00** | **68.67** |
| **compactor** | 92.67 | 84.67 | 72.00 | 68.00 | 66.67 |
| knorm | 92.67 | 75.33 | 71.33 | 68.00 | 40.00 |
| snapkv | 92.67 | 74.00 | 46.00 | 35.33 | 34.00 |
| pyramidkv | 92.67 | 63.33 | 35.33 | 34.00 | 34.00 |
| expected_attention | 92.67 | 41.33 | 33.33 | 30.67 | 28.67 |

Full-cache baseline (no compression): **92.67** (per-subset 100 / 100 / 78).

## Retained attention-cache length (median tokens; prompt ≈ 16.2k)

`per_layer_min = 0` for every Mamba/MLP layer — compression touches the attention layers only.
Retained ≈ `(1−ratio)·prompt` exactly (PyramidKV is higher: its per-layer pyramid budget over the
4 sparse attention layers averages above the flat target).

| Method | 0% (full) | 20% | 40% | 60% | 80% |
|---|---|---|---|---|---|
| knorm / snapkv / compactor / keydiff / expected_attention | 16240 | 12992 | 9744 | 6496 | 3247 |
| pyramidkv | 16240 | 14312 | 12410 | 9054 | 4527 |

## Per-subset accuracy

| Config | niah_single_1 | niah_single_3 | niah_multikey_3 |
|---|---|---|---|
| baseline (full) | 100 | 100 | 78 |
| knorm 20/40/60/80 | 100/100/100/100 | 100/100/96/16 | 26/14/8/4 |
| snapkv 20/40/60/80 | 100/100/100/100 | 88/26/2/2 | 34/12/4/0 |
| pyramidkv 20/40/60/80 | 100/100/100/100 | 68/4/2/2 | 22/2/0/0 |
| compactor 20/40/60/80 | 100/100/100/100 | 100/100/100/100 | 54/16/4/0 |
| keydiff 20/40/60/80 | 100/100/100/100 | 100/100/100/100 | 64/32/16/6 |
| expected_attention 20/40/60/80 | 100/100/92/86 | 0/0/0/0 | 24/0/0/0 |

## Observations

- **Ranking (robustness to pruning): keydiff ≳ compactor > knorm > snapkv ≈ pyramidkv > expected_attention.**
- `niah_single_1` (single needle, near start) survives even 80% pruning for every method — the easy case.
- `niah_single_3` separates the field: **keydiff and compactor hold 100% through 80% pruning**; knorm
  holds to ~60%; snapkv/pyramidkv collapse by 40%.
- `niah_multikey_3` (multiple needles) is hardest and degrades for all; keydiff/compactor lead at low ratios.
- **`expected_attention` is anomalously weak** (0% on `niah_single_3` even at 20%). Its score rotates
  query statistics by an averaged-RoPE matrix that reduces to identity on this no-RoPE model — that
  adaptation looks ill-suited here and is worth a closer look (possible follow-up).
- Compression is correct and attention-only: retained = `(1−ratio)·prompt`, Mamba/MLP layers untouched.

## Reproduce

Config: `evaluate/evaluate_nemotron_kv.yaml` (single-pass; set `kv_compressor` + `compression_ratio`).
Sweep launcher used here: `/scratch/sj157/nemotron_bench.sbatch` (per-config `OUTDIR` from the SLURM
job name; `--export=ALL` + env-prefix for method/ratio/subsets). Raw metrics + logs under
`/scratch/sj157/results_nemo_bench/<method>_r<ratio>/`.
