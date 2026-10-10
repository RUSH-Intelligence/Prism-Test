#!/usr/bin/env python
"""Representation metrics (spec §16): on held-out windows, how close are the compressed
student's hidden states to the dense teacher's — before (pretrained) and after (recovered)
training — layer by layer.

  python scripts/measure_representation_alignment.py --config configs/kv_recovery/ministral_3b.yaml \
      --run-name demo [--checkpoint DIR] [--num-examples 32] [--split val] [--no-all-layers] \
      [--sanity] [--dense-recovered] [--out FILE]

ONE model instance: pass 1 = dense teacher states (no compressor) kept on the CPU; pass 2 =
compressed student with the ORIGINAL weights; apply_delta; pass 3 = compressed student with
the RECOVERED weights; optional pass 4 = recovered weights WITHOUT compression (dense drift);
``--sanity`` adds pass 0 = uncompressed original student, which must match the teacher exactly.
Writes representation_metrics.{json,csv}: per layer (+ final norm) cosine / normalized MSE /
relative error, per-position buckets, and for hybrid models an identity check on the layers
before the first full-attention layer (untouched by K/V pruning).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eval_harness.kv_recovery.config import load_config  # noqa: E402

BUCKETS = ((0, 16), (16, 64), (64, None))


def _bucket_metrics(teacher: Dict[Any, "torch.Tensor"], student: Dict[Any, "torch.Tensor"], positions) -> Dict[str, Dict[str, float]]:
    import torch
    import torch.nn.functional as F

    out: Dict[str, Dict[str, float]] = {}
    pos = positions.cpu()
    for lo, hi in BUCKETS:
        mask = (pos >= lo) if hi is None else ((pos >= lo) & (pos < hi))
        if not bool(mask.any()):
            continue
        cos, nm = [], []
        for k, t in teacher.items():
            s = student[k]
            tf, sf = t.float()[mask], s.float()[mask]
            cos.append(float(F.cosine_similarity(sf, tf, dim=-1).mean()))
            nm.append(float((F.normalize(sf, dim=-1) - F.normalize(tf, dim=-1)).pow(2).sum(-1).mean()))
        out[f"{lo}-{hi if hi is not None else 'end'}"] = {"cosine": sum(cos) / len(cos), "normalized_mse": sum(nm) / len(nm)}
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--run-name")
    ap.add_argument("--run-dir")
    ap.add_argument("--checkpoint")
    ap.add_argument("--num-examples", type=int, default=32)
    ap.add_argument("--split", default="val", choices=["val", "train"])
    ap.add_argument("--all-layers", dest="all_layers", action="store_true", default=True)
    ap.add_argument("--no-all-layers", dest="all_layers", action="store_false")
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--dense-recovered", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    shortcuts = {"run_name": args.run_name} if args.run_name else {}
    cfg = load_config(args.config, overrides=args.set, shortcuts=shortcuts)
    run_dir = Path(args.run_dir) if args.run_dir else cfg.run_dir
    ckpt = Path(args.checkpoint) if args.checkpoint else run_dir / "checkpoint"

    import torch

    from eval_harness.kv_recovery.alignment import alignment_keys_for, position_index
    from eval_harness.kv_recovery.checkpoint import apply_delta, checkpoint_digest, load_metadata
    from eval_harness.kv_recovery.data import build_examples, read_jsonl
    from eval_harness.kv_recovery.hidden_states import FINAL_NORM_KEY, gather_positions
    from eval_harness.kv_recovery.metrics import representation_metrics
    from eval_harness.kv_recovery.model_spec import inspect_model
    from eval_harness.kv_recovery.provenance import configure_determinism, seed_everything
    from eval_harness.kv_recovery.student import build_compressor, load_adapter, resolve_segment_mode, run_segment
    from eval_harness.kv_recovery.trainable import first_trainable_layer, select_trainable

    seed_everything(cfg.seed)
    configure_determinism(cfg.deterministic)
    adapter = load_adapter(cfg, compressed=True)
    model = adapter._model
    spec = inspect_model(model)
    compressor = build_compressor(cfg)
    # A sensitivity-selected run stores its resolved layers in the checkpoint metadata; re-use them
    # (never re-measure here: the delta must be compared on the layers it was actually trained on).
    resolved = None
    if cfg.trainable.layers == "sensitivity":
        sel = (load_metadata(ckpt).get("layer_selection") or {}).get("selected")
        if not sel:
            raise SystemExit(f"{ckpt}: trainable.layers=sensitivity but the checkpoint metadata records no layer_selection.selected")
        resolved = [int(i) for i in sel]
        print(f"sensitivity-selected trainable layers (from checkpoint metadata): {resolved}", flush=True)
    names = select_trainable(model, spec, cfg.trainable, resolved_layers=resolved)
    first_tl = first_trainable_layer(names, spec)
    aligned = alignment_keys_for(cfg.alignment, spec.n_layers, first_trainable_layer=first_tl)
    keys: List[Any] = (list(range(spec.n_layers)) + ([FINAL_NORM_KEY] if spec.has_final_norm else [])) if args.all_layers else list(aligned)
    layer_indices = [k for k in keys if isinstance(k, int)]
    include_norm = FINAL_NORM_KEY in keys
    mode = resolve_segment_mode(model, cfg.student)
    path = cfg.data.val_path if args.split == "val" else cfg.data.path
    seed = cfg.data.seed + (1 if args.split == "val" else 0)
    examples, stats = build_examples(read_jsonl(path), adapter._tokenizer, cfg.data, n_examples=args.num_examples, seed=seed,
                                     model=model, pipeline=adapter._pipe)
    print(f"{len(examples)} {args.split} windows, keys {keys}, segment mode {mode}", flush=True)
    ratio = float(cfg.kv_compression.compression_ratio)

    def pass_states(ex, comp, ratio_):
        out = run_segment(adapter, ex, comp, layer_indices, include_final_norm=include_norm, grad=False, want_logits=False,
                          mode=mode, spec=spec, compression_ratio=ratio_)
        pos = position_index(cfg.alignment.positions, ex.suffix_len, device=next(iter(out.states.values())).device)
        return {k: v.cpu() for k, v in gather_positions(out.states, pos).items()}, pos.cpu()

    t0 = time.time()
    teacher, positions = {}, {}
    for ex in examples:
        teacher[ex.id], positions[ex.id] = pass_states(ex, None, 0.0)
    print(f"teacher states: {time.time() - t0:.0f}s", flush=True)

    conditions: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    buckets: Dict[str, Dict[str, Dict[str, List[float]]]] = {}

    def collect(name: str, comp, ratio_):
        per_ex = {}
        for ex in examples:
            st, pos = pass_states(ex, comp, ratio_)
            per_ex[ex.id] = representation_metrics(teacher[ex.id], st)
            for b, m in _bucket_metrics(teacher[ex.id], st, pos).items():
                for metric, val in m.items():
                    buckets.setdefault(name, {}).setdefault(b, {}).setdefault(metric, []).append(val)
        conditions[name] = per_ex
        print(f"{name}: {time.time() - t0:.0f}s", flush=True)

    if args.sanity:
        collect("sanity_uncompressed_original", None, 0.0)
    collect("compressed", compressor, ratio)
    delta_info = apply_delta(model, ckpt, strict=True)
    collect("recovered", compressor, ratio)
    if args.dense_recovered:
        collect("dense_recovered", None, 0.0)

    def mean_over_examples(name: str, key: str, metric: str) -> float:
        vals = [conditions[name][ex.id][str(key)][metric] for ex in examples]
        return sum(vals) / len(vals)

    layers_out = []
    for k in keys:
        row: Dict[str, Any] = {"layer": str(k), "aligned": k in aligned, "trainable_block": isinstance(k, int) and first_tl is not None and k >= first_tl,
                               "hooked": isinstance(k, int) and k in spec.full_attention_layers}
        for name in conditions:
            for metric in ("cosine", "normalized_mse", "relative_error"):
                row[f"{name}_{metric}"] = mean_over_examples(name, k, metric)
        layers_out.append(row)
    identity = None
    if spec.is_hybrid and spec.full_attention_layers:
        first_full = spec.first_full_attention_layer
        pre = [k for k in layer_indices if k < first_full]
        if pre:
            worst = max(1.0 - mean_over_examples("compressed", k, "cosine") for k in pre)
            identity = {"layers_expected_identical": pre, "max_one_minus_cosine": worst, "ok": worst <= 1e-6}

    def summary(name: str, subset: List[Any]) -> Dict[str, float]:
        return {metric: sum(mean_over_examples(name, k, metric) for k in subset) / len(subset)
                for metric in ("cosine", "normalized_mse", "relative_error")}

    results = {
        "schema_version": 1, "run_name": cfg.run_name, "model": cfg.model.name, "kv_compression": cfg.kv_compression.__dict__,
        "checkpoint": {"path": str(ckpt), "sha256": checkpoint_digest(ckpt), "applied": len(delta_info["applied"])},
        "protocol": {"split": args.split, "n_examples": len(examples), "max_length": cfg.data.max_length,
                     "suffix_length": cfg.data.suffix_length, "positions": cfg.alignment.positions.__dict__,
                     "format": cfg.data.format, "suffix_mode": cfg.data.suffix_mode, "segment_mode": mode,
                     "data_path": path, "window_stats": stats.as_dict()},
        "aligned_layers": [str(k) for k in aligned], "trainable_first_layer": first_tl,
        "trainable_layers_resolved": resolved,
        "hooked_layers": list(spec.full_attention_layers), "layers": layers_out,
        "per_position_bucket": {name: {b: {m: sum(v) / len(v) for m, v in mm.items()} for b, mm in bb.items()}
                                for name, bb in buckets.items()},
        "identity_prefix_check": identity,
        "summary": {name: {"aligned_layers_mean": summary(name, aligned), "all_keys_mean": summary(name, keys)} for name in conditions},
    }
    out = Path(args.out) if args.out else run_dir / "representation_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str))
    with out.with_suffix(".csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(layers_out[0].keys()))
        w.writeheader()
        w.writerows(layers_out)
    print(f"layer | compressed_cosine | recovered_cosine | compressed_nmse | recovered_nmse")
    for r in layers_out:
        print(f"{r['layer']:>5} | {r['compressed_cosine']:.4f} | {r['recovered_cosine']:.4f} | "
              f"{r['compressed_normalized_mse']:.5f} | {r['recovered_normalized_mse']:.5f}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
