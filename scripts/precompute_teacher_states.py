#!/usr/bin/env python
"""Offline teacher: precompute the full-cache teacher's hidden states at the aligned layers
and positions for every training / validation window (spec §11, optional mode).

  python scripts/precompute_teacher_states.py --config configs/kv_recovery/ministral_3b.yaml \
      [--set ...] [--out outputs/kv_recovery/teacher_states/<digest>] [--max-gb 50] [--dtype bfloat16]

Stores one safetensors file per example (``layer_<k>`` / ``norm`` tensors ``[P, H]`` plus
``positions``) and ``manifest.json`` with the teacher digest the trainer checks. Logits are NOT
stored, so ``loss.kl_weight`` must be 0 when training offline. Refuses to write more than
``--max-gb`` (estimated as n_examples x |keys| x P x H x bytes).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eval_harness.kv_recovery.config import load_config  # noqa: E402


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--out")
    ap.add_argument("--max-gb", type=float, default=50.0)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args(argv)
    cfg = load_config(args.config, overrides=args.set)

    import torch

    from eval_harness.kv_recovery.alignment import alignment_keys_for
    from eval_harness.kv_recovery.data import load_split
    from eval_harness.kv_recovery.model_spec import inspect_model
    from eval_harness.kv_recovery.provenance import configure_determinism, seed_everything
    from eval_harness.kv_recovery.student import load_adapter, resolve_segment_mode
    from eval_harness.kv_recovery.trainable import first_trainable_layer, select_trainable
    from eval_harness.kv_recovery.trainer import AlignmentSetup, TeacherStore, teacher_digest, teacher_states_for

    seed_everything(cfg.seed)
    configure_determinism(cfg.deterministic)
    teacher = load_adapter(cfg, compressed=False)
    model = teacher._model
    spec = inspect_model(model)
    names = select_trainable(model, spec, cfg.trainable)
    keys = alignment_keys_for(cfg.alignment, spec.n_layers, first_trainable_layer=first_trainable_layer(names, spec))
    mode = resolve_segment_mode(model, cfg.student)
    setup = AlignmentSetup(keys=keys, layer_indices=[k for k in keys if isinstance(k, int)],
                           include_final_norm=cfg.alignment.include_final_norm, positions_cfg=cfg.alignment.positions,
                           loss_name=cfg.alignment.loss, layer_weights=cfg.alignment.layer_weights, loss_cfg=cfg.loss,
                           want_logits=False, mode=mode, spec=spec, compression_ratio=0.0,
                           prefill_chunk_size=cfg.kv_compression.prefill_chunk_size, prefill_grad=False)
    train_ex, _ = load_split(cfg, teacher._tokenizer, "train", model=model, pipeline=teacher._pipe)
    val_ex, _ = load_split(cfg, teacher._tokenizer, "val", model=model, pipeline=teacher._pipe)
    examples = train_ex + val_ex
    digest = teacher_digest(cfg)
    out = Path(args.out) if args.out else Path(cfg.output.root) / "teacher_states" / digest
    out.mkdir(parents=True, exist_ok=True)

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    P = len(examples[0].suffix_ids[0]) if cfg.alignment.positions.strategy == "all" else min(cfg.alignment.positions.n, cfg.data.suffix_length)
    est_gb = len(examples) * len(keys) * P * spec.hidden_size * torch.tensor([], dtype=dtype).element_size() / 2**30
    if est_gb > args.max_gb:
        raise SystemExit(f"estimated {est_gb:.1f} GiB > --max-gb {args.max_gb}; reduce examples/layers/positions")
    print(f"writing ~{est_gb:.2f} GiB of teacher states for {len(examples)} examples to {out}", flush=True)
    t0 = time.time()
    for i, ex in enumerate(examples):
        ts = teacher_states_for(teacher, ex, setup)
        TeacherStore.save(out, ex.id, ts, dtype=dtype)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(examples)} ({time.time() - t0:.0f}s)", flush=True)
    manifest = {"teacher_digest": digest, "keys": [str(k) for k in keys], "dtype": args.dtype, "n": len(examples),
                "ids": [e.id for e in examples], "model": cfg.model.name, "positions": cfg.alignment.positions.__dict__,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (out / TeacherStore.MANIFEST).write_text(json.dumps(manifest, indent=2))
    print(f"done in {time.time() - t0:.0f}s: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
