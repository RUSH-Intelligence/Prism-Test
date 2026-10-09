#!/usr/bin/env python
"""Layer-wise compression sensitivity on held-out windows — the layer-selection signal of
``trainable.layers: sensitivity`` — WITHOUT training anything.

  python scripts/measure_layer_sensitivity.py --config configs/kv_recovery/ministral_3b.yaml \
      [--set a.b.c=value ...] [--compressors knorm,cur] [--ratios 0.75,0.5] [--num-examples 8] \
      [--top-k 4] [--strategy attention_projections] [--split val] [--out DIR]

For every (compressor, ratio) the ONE loaded model instance runs each calibration window twice —
dense cache, then compressed cache (original weights) — and reports per layer

    E_l = ||H_l^dense - H_l^comp||_F / (||H_l^dense||_F + eps)

over the suffix tokens (mean +- std across windows), the ranking, the eligible layers for the
trainable strategy and the resulting top-k selection, exactly as ``train_kv_recovery.py`` would
select them for that configuration. Calibration windows are the ``trainable.sensitivity`` rows:
``num_examples`` windows of ``split`` (seed ``data.seed + 2``) that are NOT validation windows.

Writes ``<out>/<model>__<compressor>_r<ratio>.{json,csv}`` and ``<out>/summary.md`` (one table
per setting plus a cross-setting rank table). Default ``--out`` is
``<output.root>/sensitivity/<model_slug>``.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eval_harness.kv_recovery.config import LAYER_SELECTED_STRATEGIES, load_config  # noqa: E402


def _ratio_tag(r: float) -> str:
    return f"r{int(round(float(r) * 100)):03d}"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--compressors", help="comma list; default: the config's kv_compressor")
    ap.add_argument("--ratios", help="comma list; default: the config's compression_ratio")
    ap.add_argument("--num-examples", type=int, help="override trainable.sensitivity.num_examples")
    ap.add_argument("--top-k", type=int, help="override trainable.sensitivity.top_k")
    ap.add_argument("--strategy", choices=list(LAYER_SELECTED_STRATEGIES),
                    help="eligible-layer pool (default: the config's trainable.strategy, else attention_projections)")
    ap.add_argument("--split", choices=["val", "train"], help="override trainable.sensitivity.split")
    ap.add_argument("--out")
    args = ap.parse_args(argv)

    shortcuts: Dict[str, Any] = {}
    if args.num_examples is not None:
        shortcuts["trainable.sensitivity.num_examples"] = args.num_examples
    if args.top_k is not None:
        shortcuts["trainable.sensitivity.top_k"] = args.top_k
    if args.split:
        shortcuts["trainable.sensitivity.split"] = args.split
    cfg = load_config(args.config, overrides=args.set, shortcuts=shortcuts)
    strategy = args.strategy or (cfg.trainable.strategy if cfg.trainable.strategy in LAYER_SELECTED_STRATEGIES
                                 else "attention_projections")
    compressors = [c.strip() for c in args.compressors.split(",")] if args.compressors else [cfg.kv_compression.kv_compressor]
    ratios = [float(r) for r in args.ratios.split(",")] if args.ratios else [float(cfg.kv_compression.compression_ratio)]
    out_dir = Path(args.out) if args.out else Path(cfg.output.root) / "sensitivity" / cfg.model.name.replace("/", "--")
    out_dir.mkdir(parents=True, exist_ok=True)

    from eval_harness.kv_recovery.data import assert_disjoint, describe_split, load_split
    from eval_harness.kv_recovery.model_spec import inspect_model
    from eval_harness.kv_recovery.provenance import configure_determinism, provenance, seed_everything
    from eval_harness.kv_recovery.sensitivity import select_layers_by_sensitivity
    from eval_harness.kv_recovery.student import build_compressor, load_adapter, resolve_segment_mode

    seed_everything(cfg.seed)
    configure_determinism(cfg.deterministic)
    adapter = load_adapter(cfg, compressed=True)      # one instance serves both passes (weights untouched)
    model = adapter._model
    spec = inspect_model(model)
    mode = resolve_segment_mode(model, cfg.student)
    tokenizer = adapter._tokenizer
    val_ex, _ = load_split(cfg, tokenizer, "val", model=model, pipeline=adapter._pipe)
    exclude = {e.id for e in val_ex}
    if cfg.trainable.sensitivity.split == "train":
        train_ex, _ = load_split(cfg, tokenizer, "train", model=model, pipeline=adapter._pipe)
        exclude |= {e.id for e in train_ex}
    calib, cstats = load_split(cfg, tokenizer, "calibration", model=model, pipeline=adapter._pipe, exclude_ids=exclude)
    assert_disjoint(val_ex, calib)
    print(f"{len(calib)} calibration windows ({cfg.data.max_length} tokens, suffix {cfg.data.suffix_length}); "
          f"layers {spec.n_layers}, K/V-carrying {list(spec.full_attention_layers)}; strategy {strategy}", flush=True)

    summary_rows: List[Dict[str, Any]] = []
    md: List[str] = [f"# Layer-wise compression sensitivity — `{cfg.model.name}`", "",
                     f"{len(calib)} held-out calibration windows of {cfg.data.max_length} tokens (suffix {cfg.data.suffix_length}, "
                     f"positions `{cfg.trainable.sensitivity.positions.strategy}`), `E_l = ||H_dense - H_comp||_F / "
                     f"(||H_dense||_F + {cfg.trainable.sensitivity.eps:g})`, {cfg.trainable.sensitivity.aggregate} over windows. "
                     f"Eligible pool: `{strategy}` -> layers {list(spec.full_attention_layers) if strategy == 'attention_projections' else 'all'}.", ""]
    rank_table: Dict[int, Dict[str, int]] = {}
    for comp_name in compressors:
        for ratio in ratios:
            c2 = copy.deepcopy(cfg)
            c2.kv_compression.kv_compressor = comp_name
            c2.kv_compression.compression_ratio = float(ratio)
            compressor = build_compressor(c2)
            if compressor is None:
                raise SystemExit(f"{comp_name}: no compressor built")
            t0 = time.time()
            rep = select_layers_by_sensitivity(c2, adapter, adapter, compressor, calib, spec=spec, mode=mode, strategy=strategy)
            tag = f"{comp_name}_{_ratio_tag(ratio)}"
            print(f"\n=== {comp_name} @ ratio {ratio}  ({time.time() - t0:.0f}s)")
            print(rep.table(), flush=True)
            payload = {"schema_version": 1, "model": cfg.model.name, "compressor": comp_name, "compression_ratio": ratio,
                       "strategy": strategy, "protocol": {"split": cfg.trainable.sensitivity.split, "n_examples": len(calib),
                                                          "max_length": cfg.data.max_length, "suffix_length": cfg.data.suffix_length,
                                                          "format": cfg.data.format, "suffix_mode": cfg.data.suffix_mode,
                                                          "segment_mode": mode,
                                                          "calibration": describe_split(calib, cstats, cfg.data.val_path if cfg.trainable.sensitivity.split == "val" else cfg.data.path)},
                       "report": rep.to_dict(), "provenance": provenance(REPO_ROOT)}
            # NOTE: plain string concatenation, not Path.with_suffix — model slugs contain dots ("Qwen3.5-4B").
            base = str(out_dir / f"{cfg.model.name.replace('/', '--')}__{tag}")
            Path(base + ".json").write_text(json.dumps(payload, indent=2, default=str))
            rows = rep.rows()
            with Path(base + ".csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            md += [f"## `{comp_name}` @ ratio {ratio}", "", "| layer | E_l (mean) | std | rank | K/V cache | eligible | selected |",
                   "|---|---|---|---|---|---|---|"]
            for r in rows:
                md.append(f"| {r['layer']} | {r['sensitivity']:.4f} | {r['std']:.4f} | {r['rank']} | "
                          f"{'yes' if r['hooked'] else '-'} | {'yes' if r['candidate'] else '-'} | {'**yes**' if r['selected'] else ''} |")
            md += ["", f"Selected top-{rep.top_k}: `{rep.selected}`" + (f" — {'; '.join(rep.notes)}" if rep.notes else ""), ""]
            for r in rows:
                rank_table.setdefault(r["layer"], {})[tag] = r["rank"]
            summary_rows.append({"compressor": comp_name, "ratio": ratio, "selected": rep.selected, "ranking": rep.ranking,
                                 "seconds": rep.seconds})
    if len(summary_rows) > 1:
        tags = [f"{r['compressor']}_{_ratio_tag(r['ratio'])}" for r in summary_rows]
        md += ["## Rank of each layer across settings (1 = most sensitive)", "",
               "| layer | " + " | ".join(tags) + " |", "|---|" + "---|" * len(tags)]
        for layer in sorted(rank_table):
            md.append(f"| {layer} | " + " | ".join(str(rank_table[layer].get(t, "")) for t in tags) + " |")
        md.append("")
        md += ["Selections: " + "; ".join(f"`{t}` -> {r['selected']}" for t, r in zip(tags, summary_rows)), ""]
    (out_dir / "summary.md").write_text("\n".join(md))
    (out_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2))
    print(f"\nwrote {out_dir}/summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
