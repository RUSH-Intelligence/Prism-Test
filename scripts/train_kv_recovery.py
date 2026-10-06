#!/usr/bin/env python
"""Train a hidden-state KV-recovery delta (spec §6-§11, §17-§20).

  python scripts/train_kv_recovery.py --config configs/kv_recovery/ministral_3b.yaml \
      [--run-name NAME] [--set a.b.c=value ...] [--learning-rate 1e-5]
      [--trainable-last-n-blocks 1 | --trainable-attention-projections k_proj,v_proj]
      [--kv-compressor knorm] [--compression-ratio 0.75 | --kv-budget-ratio 0.25]
      [--max-length 16384] [--suffix-length 512] [--num-train-examples 256] [--seed 42]
      [--epochs 1] [--max-steps N] [--kl-weight 0.0] [--teacher-states DIR]

Outputs (outputs/kv_recovery/<run_name>/): config.yaml, metadata.json, train_metrics.jsonl,
val_loss.csv, weight_update_norms.csv, trainable_parameters.txt, sanity_checks.json,
checkpoint/{adapted_weights.safetensors, metadata.json, config.yaml}, logs/.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

from eval_harness.kv_recovery.config import RecoveryConfig, kv_budget_to_ratio, load_config, training_identity  # noqa: E402

logger = logging.getLogger("kv_recovery.train")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="dotted override, YAML value")
    ap.add_argument("--run-name")
    ap.add_argument("--learning-rate", type=float)
    ap.add_argument("--trainable-last-n-blocks", type=int)
    ap.add_argument("--trainable-attention-projections", help="comma list, e.g. k_proj,v_proj")
    ap.add_argument("--trainable-layers", help="all | last_n:<k> | comma list of layer indices (attention_projections)")
    ap.add_argument("--kv-compressor")
    ap.add_argument("--compression-ratio", type=float, help="fraction of the context KV pruned")
    ap.add_argument("--kv-budget-ratio", type=float, help="fraction KEPT (spec wording) = 1 - compression_ratio")
    ap.add_argument("--max-length", type=int)
    ap.add_argument("--suffix-length", type=int)
    ap.add_argument("--num-train-examples", type=int)
    ap.add_argument("--num-val-examples", type=int)
    ap.add_argument("--data")
    ap.add_argument("--val-data")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--kl-weight", type=float)
    ap.add_argument("--teacher-states", help="directory from scripts/precompute_teacher_states.py (offline teacher)")
    ap.add_argument("--overwrite", action="store_true")
    return ap


def shortcuts_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    s: Dict[str, Any] = {}
    if args.run_name: s["run_name"] = args.run_name
    if args.learning_rate is not None: s["optim.learning_rate"] = args.learning_rate
    if args.trainable_last_n_blocks is not None:
        s["trainable.strategy"] = "last_n_blocks"; s["trainable.n"] = args.trainable_last_n_blocks
    if args.trainable_attention_projections:
        s["trainable.strategy"] = "attention_projections"
        s["trainable.modules"] = [m.strip() for m in args.trainable_attention_projections.split(",") if m.strip()]
    if args.trainable_layers:
        v = args.trainable_layers
        s["trainable.layers"] = v if v == "all" or v.startswith("last_n:") else [int(x) for x in v.split(",")]
    if args.kv_compressor: s["kv_compression.kv_compressor"] = args.kv_compressor
    if args.compression_ratio is not None and args.kv_budget_ratio is not None:
        raise SystemExit("pass either --compression-ratio or --kv-budget-ratio, not both")
    if args.compression_ratio is not None: s["kv_compression.compression_ratio"] = args.compression_ratio
    if args.kv_budget_ratio is not None:
        s["kv_compression.compression_ratio"] = kv_budget_to_ratio(args.kv_budget_ratio)
        print(f"kv budget ratio {args.kv_budget_ratio} -> compression_ratio {s['kv_compression.compression_ratio']:.4f}")
    if args.max_length is not None: s["data.max_length"] = args.max_length
    if args.suffix_length is not None: s["data.suffix_length"] = args.suffix_length
    if args.num_train_examples is not None: s["data.num_train_examples"] = args.num_train_examples
    if args.num_val_examples is not None: s["data.num_val_examples"] = args.num_val_examples
    if args.data: s["data.path"] = args.data
    if args.val_data: s["data.val_path"] = args.val_data
    if args.seed is not None: s["seed"] = args.seed
    if args.epochs is not None: s["optim.epochs"] = args.epochs
    if args.max_steps is not None: s["optim.max_steps"] = args.max_steps
    if args.kl_weight is not None: s["loss.kl_weight"] = args.kl_weight
    if args.teacher_states:
        s["teacher.mode"] = "offline"; s["teacher.states_dir"] = args.teacher_states
    if args.overwrite: s["output.overwrite"] = True
    return s


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in r.items()})


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, overrides=args.set, shortcuts=shortcuts_from_args(args))
    run_dir = cfg.run_dir
    if run_dir.exists() and any(run_dir.iterdir()) and not cfg.output.overwrite:
        raise SystemExit(f"run dir {run_dir} is not empty (use --overwrite or a new --run-name)")
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoint").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(run_dir / "logs" / "train.log")])
    config_yaml = yaml.safe_dump(cfg.to_dict(), sort_keys=False)
    (run_dir / "config.yaml").write_text(config_yaml)
    t_start = time.time()

    # --- environment, seeds, determinism (before any model load) ---------------------
    from eval_harness.kv_recovery.provenance import configure_determinism, provenance, seed_everything

    seed_everything(cfg.seed)
    det_flags = configure_determinism(cfg.deterministic)
    prov = provenance(REPO_ROOT)
    logger.info("git %s branch %s dirty=%s", prov["git"].get("git_commit"), prov["git"].get("git_branch"), prov["git"].get("git_dirty"))

    import torch

    from eval_harness.kv_recovery.alignment import alignment_keys_for, check_alignment_has_gradient
    from eval_harness.kv_recovery.checkpoint import frozen_sample_names, hashes_of, write_delta
    from eval_harness.kv_recovery.data import assert_disjoint, describe_split, load_split
    from eval_harness.kv_recovery.model_spec import inspect_model, resolve_base_revision
    from eval_harness.kv_recovery.student import build_compressor, load_adapter, resolve_segment_mode
    from eval_harness.kv_recovery.trainable import (assert_trainable, changed_parameters, format_parameter_summary,
                                                    freeze_all_but, parameter_summary, select_trainable,
                                                    snapshot_parameters, trainable_parameters)
    from eval_harness.kv_recovery.trainer import (AlignmentSetup, TeacherStore, TrainingUnstable, check_same_model,
                                                  teacher_digest, train, weight_update_norms)

    # --- models -----------------------------------------------------------------------
    student = load_adapter(cfg, compressed=True)
    model = student._model
    teacher = None
    store = None
    if cfg.teacher.mode == "online":
        teacher = load_adapter(cfg, compressed=False)
    else:
        if not cfg.teacher.states_dir:
            raise SystemExit("teacher.mode=offline needs teacher.states_dir (--teacher-states)")
        store = TeacherStore(cfg.teacher.states_dir, device=next(model.parameters()).device)
    spec = inspect_model(model)
    compressor = build_compressor(cfg)
    if compressor is None:
        raise SystemExit("kv_compression.kv_compressor is 'none': nothing to recover from")
    logger.info("model family=%s layers=%d hidden=%d full_attention_layers=%s hybrid=%s",
                spec.family, spec.n_layers, spec.hidden_size, list(spec.full_attention_layers), spec.is_hybrid)

    # --- trainable subset --------------------------------------------------------------
    names = select_trainable(model, spec, cfg.trainable)
    expected = freeze_all_but(model, names)
    assert_trainable(model, expected, spec)
    summary = parameter_summary(model, spec)
    text = format_parameter_summary(summary)
    print(text, flush=True)
    (run_dir / "trainable_parameters.txt").write_text(text + "\n")
    trainable = trainable_parameters(model, names)
    if teacher is not None:
        tp = dict(teacher._model.named_parameters())
        mismatch = [n for n, p in model.named_parameters() if not torch.equal(p.detach(), tp[n].detach())]
        if mismatch:
            raise SystemExit(f"teacher and student weights differ at load time: {mismatch[:5]}")
        logger.info("teacher and student parameters bitwise identical at start (%d tensors)", len(tp))

    # --- alignment setup ----------------------------------------------------------------
    first_tl = summary["first_trainable_layer"]
    keys = alignment_keys_for(cfg.alignment, spec.n_layers, first_trainable_layer=first_tl)
    dead = check_alignment_has_gradient(keys, first_tl, allow=cfg.alignment.allow_dead_terms)
    if dead:
        logger.warning("aligned keys %s carry no gradient (allowed by config)", dead)
    mode = resolve_segment_mode(model, cfg.student)
    setup = AlignmentSetup(keys=keys, layer_indices=[k for k in keys if isinstance(k, int)],
                           include_final_norm=cfg.alignment.include_final_norm, positions_cfg=cfg.alignment.positions,
                           loss_name=cfg.alignment.loss, layer_weights=cfg.alignment.layer_weights, loss_cfg=cfg.loss,
                           want_logits=cfg.loss.kl_weight > 0, mode=mode, spec=spec,
                           compression_ratio=float(cfg.kv_compression.compression_ratio),
                           prefill_chunk_size=cfg.kv_compression.prefill_chunk_size, prefill_grad=cfg.student.prefill_grad,
                           deterministic_backward=cfg.optim.deterministic_backward)
    logger.info("alignment keys=%s positions=%s loss=%s segment_mode=%s", keys, cfg.alignment.positions, cfg.alignment.loss, mode)

    # --- data ---------------------------------------------------------------------------
    tokenizer = student._tokenizer
    train_ex, train_stats = load_split(cfg, tokenizer, "train", model=model, pipeline=student._pipe)
    val_ex, val_stats = load_split(cfg, tokenizer, "val", model=model, pipeline=student._pipe)
    assert_disjoint(train_ex, val_ex)
    logger.info("data: %d train / %d val windows (%s)", len(train_ex), len(val_ex), train_stats.as_dict())
    if store is not None:
        store.check(cfg, keys)
        missing = [e.id for e in train_ex + val_ex if not store.has(e.id)]
        if missing:
            raise SystemExit(f"offline teacher states missing for {len(missing)} examples (e.g. {missing[:3]})")

    # --- sanity: same-model loss == 0 (spec §22) ------------------------------------------
    sanity: Dict[str, Any] = {}
    if teacher is not None:
        same = check_same_model(teacher, student, train_ex[:2], setup)
        sanity["same_model"] = same
        logger.info("same-model check: %s", same)
        if not same["passed"]:
            raise SystemExit("same-model sanity check failed: teacher and uncompressed student differ")

    # --- snapshots for the frozen-weight check and the delta -----------------------------
    originals = snapshot_parameters(model, names)
    original_sha = hashes_of(model, names)
    frozen_names = frozen_sample_names(model, names, spec)
    frozen_sha = hashes_of(model, frozen_names)
    frozen_snapshot = snapshot_parameters(model, frozen_names)

    # --- train (with the pre-registered instability restart) -----------------------------
    log_path = run_dir / "train_metrics.jsonl"
    log_path.write_text("")
    restart_note = None
    try:
        state = train(teacher, student, compressor, trainable, train_ex, val_ex, cfg, setup, log_path=log_path, store=store)
    except TrainingUnstable as exc:
        if not cfg.optim.instability.restart:
            raise
        restart_note = f"{exc}; restarted with warmup_fraction={cfg.optim.instability.warmup_fraction}"
        logger.warning(restart_note)
        with torch.no_grad():
            for n, p in trainable.items():
                p.copy_(originals[n].to(p.device))
        log_path.rename(run_dir / "train_metrics.unstable.jsonl")
        log_path.write_text("")
        state = train(teacher, student, compressor, trainable, train_ex, val_ex, cfg, setup, log_path=log_path, store=store,
                      warmup_fraction=cfg.optim.instability.warmup_fraction)

    # --- post-training checks -------------------------------------------------------------
    sanity["check3_gradients_only_on_trainable"] = state.check3
    changed_frozen = changed_parameters(model, frozen_snapshot)
    sanity["frozen_sample_unchanged"] = {"passed": not changed_frozen, "names": frozen_names, "changed": changed_frozen}
    if teacher is not None:
        tp = dict(teacher._model.named_parameters())
        drift = [n for n, p in model.named_parameters() if n not in trainable and not torch.equal(p.detach(), tp[n].detach())]
        sanity["all_frozen_bitwise_equal_to_teacher"] = {"passed": not drift, "drifted": drift[:10]}
        if drift:
            raise SystemExit(f"frozen parameters changed during training: {drift[:5]}")
    sanity["trainable_changed"] = {"n_changed": len(changed_parameters(model, originals)), "n_trainable": len(names)}
    (run_dir / "sanity_checks.json").write_text(json.dumps(sanity, indent=2, default=str))
    write_csv(run_dir / "weight_update_norms.csv",
              weight_update_norms(originals, trainable, state.masters, state.optimizer_steps, cfg.optim.learning_rate, spec))
    write_csv(run_dir / "val_loss.csv", [{k: v for k, v in r.items() if k != "val_per_layer"} for r in state.val_logs])

    # --- metadata + delta (spec §17 / §19) ------------------------------------------------
    metadata: Dict[str, Any] = {
        "run_name": cfg.run_name,
        "base_model": cfg.model.name,
        "base_revision": cfg.model.revision or resolve_base_revision(model),
        "base_dtype": str(next(model.parameters()).dtype).replace("torch.", ""),
        **training_identity(cfg),
        "trainable": cfg.trainable.__dict__,
        "parameter_summary": {k: v for k, v in summary.items() if k != "trainable_names"},
        "alignment": {"keys": [str(k) for k in keys], "dead_keys": [str(k) for k in dead], "positions": cfg.alignment.positions.__dict__,
                      "loss": cfg.alignment.loss, "layer_weights": cfg.alignment.layer_weights,
                      "include_final_norm": cfg.alignment.include_final_norm},
        "loss": cfg.loss.__dict__,
        "optim": {**{k: v for k, v in cfg.optim.__dict__.items() if k != "instability"},
                  "instability": cfg.optim.instability.__dict__, "optimizer": "AdamW",
                  "precision": "bf16 forward/backward, fp32 master weights + AdamW state" if cfg.optim.master_weights_fp32 else "bf16",
                  "steps": state.optimizer_steps, "micro_batches": state.micro_batches, "instability_restart": restart_note},
        "data": {"train": describe_split(train_ex, train_stats, cfg.data.path), "val": describe_split(val_ex, val_stats, cfg.data.val_path),
                 "max_length": cfg.data.max_length, "suffix_length": cfg.data.suffix_length, "format": cfg.data.format,
                 "suffix_mode": cfg.data.suffix_mode, "seed": cfg.data.seed},
        "sequence_length": cfg.data.max_length,
        "segment_mode": mode,
        "teacher": {"mode": cfg.teacher.mode, "states_dir": cfg.teacher.states_dir, "digest": teacher_digest(cfg)},
        "student": cfg.student.__dict__,
        "seeds": {"global": cfg.seed, "data": cfg.data.seed, "shuffle": cfg.optim.shuffle_seed},
        "determinism": det_flags,
        "provenance": prov,
        "compute": {"wall_clock_seconds": round(time.time() - t_start, 1), "seconds_train_loop": round(state.seconds_train_loop, 1),
                    "seconds_validation": round(state.seconds_validation, 1), "context_tokens_teacher": state.context_tokens_teacher,
                    "context_tokens_student": state.context_tokens_student, "suffix_tokens_aligned": state.suffix_tokens_aligned,
                    "peak_gpu_memory_gib": state.peak_gpu_memory_gib, "first_train_loss": state.first_train_loss,
                    "final_train_loss": state.final_train_loss, "val_first": state.val_logs[0], "val_last": state.val_logs[-1]},
        "sanity_checks": {k: (v.get("passed") if isinstance(v, dict) and "passed" in v else v) for k, v in sanity.items()},
    }
    ckpt = write_delta(run_dir / "checkpoint", model, names, original_sha, metadata, masters=state.masters,
                       frozen_sample_sha256=frozen_sha, config_yaml=config_yaml)
    shutil.copy(ckpt / "metadata.json", run_dir / "metadata.json")
    logger.info("done: %d optimizer steps, val_loss %.6f -> %.6f, checkpoint %s", state.optimizer_steps,
                state.val_logs[0]["val_loss"], state.val_logs[-1]["val_loss"], ckpt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
