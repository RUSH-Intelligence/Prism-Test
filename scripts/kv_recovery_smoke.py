#!/usr/bin/env python
"""GPU smoke test for one model family (spec §21 / §22), ~15 min on one H200.

  python scripts/kv_recovery_smoke.py --config configs/kv_recovery/smoke_ministral_3b.yaml [--with-benchmark]

Checks (each recorded in <run_dir>/smoke_report.json; the exit code is non-zero if any fails):
  S0  synthetic corpus written when the configured JSONL is absent
  S1  environment (versions, CUBLAS_WORKSPACE_CONFIG, GPU, determinism flags)
  S2  model loads (teacher + student), attention implementation as requested, parameters bitwise
      identical; FP8 checkpoints: dequantised weights == fp8 * scale exactly, plain bf16 Linear
  S3  prompt shaping: no auto system block after strip; raw window = bos + text
  S4  data windows: shapes, train/val disjoint
  S5  compressor hooks on exactly the full-attention layers, none on the teacher
  S6  budget: int(T(1-r)) per hooked layer after prefill, +L after the suffix, compress once per layer
  S7  segment-continuation: prefill(ctx)+segment == forward(ctx+seg), block and token-by-token
      (relative Frobenius <= 2e-2 and per-position cosine >= 0.999; bf16 kernel noise is ~1e-2)
  S8  teacher == student without compression (loss 0)
  S9  compression increases divergence (ratio 0.75 > 0.5 > 0)
  S10 trainable selection / stray gradients / dead terms / tied lm_head
  S11 2 training steps + checkpoint round trip on a fresh load; frozen tensors untouched;
      optimisation check (8 steps on a fixed batch lower the loss)
  S12 identity delta reproduces compressed generations bitwise
  S13 (--with-benchmark) three-way eval on 3 RULER subsets x 5 rows through eval_harness.cli + report
  S14 compression-sensitivity layer selection (runs before S10): calibration windows disjoint from
      train/val, E_l finite and > 0 on every K/V-carrying layer, exactly 0 on layers before the first
      full-attention layer (hybrids), bitwise deterministic across two measurements, top-k drawn from
      the eligible layers only, E_l == 0 without a compressor; the selection feeds S10-S13 when the
      card uses ``trainable.layers: sensitivity``
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eval_harness.kv_recovery.config import RecoveryConfig, load_config  # noqa: E402

WORDS = ("the quick brown fox jumps over lazy dog river mountain whisper ancient library candle "
         "garden winter summer letter journey silence morning evening harbour castle meadow story "
         "window bridge forest lantern market village thunder shadow mirror copper silver golden").split()


class Report:
    def __init__(self, path: Path):
        self.path = path
        self.checks: Dict[str, Dict[str, Any]] = {}

    def run(self, name: str, fn: Callable[[], Any]) -> bool:
        t0 = time.time()
        try:
            detail = fn()
            self.checks[name] = {"passed": True, "detail": detail, "seconds": round(time.time() - t0, 1)}
            print(f"[PASS] {name}: {json.dumps(detail, default=str)[:400]}", flush=True)
            ok = True
        except Exception as exc:  # noqa: BLE001
            self.checks[name] = {"passed": False, "error": f"{type(exc).__name__}: {exc}",
                                 "traceback": traceback.format_exc()[-2000:], "seconds": round(time.time() - t0, 1)}
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}", flush=True)
            ok = False
        self.path.write_text(json.dumps(self.checks, indent=2, default=str))
        return ok

    @property
    def all_passed(self) -> bool:
        return all(c["passed"] for c in self.checks.values())


def synthetic_rows(n: int, n_words: int, seed: int) -> List[dict]:
    rng = random.Random(seed)
    return [{"id": f"smoke-{seed}-{i:03d}", "source": "synthetic", "split": "smoke",
             "text": " ".join(rng.choice(WORDS) for _ in range(n_words))} for i in range(n)]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--with-benchmark", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    cfg: RecoveryConfig = load_config(args.config, overrides=args.set, shortcuts={"output.overwrite": True})
    run_dir = Path(args.out) if args.out else cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    rep = Report(run_dir / "smoke_report.json")

    # ---- S0 synthetic corpus ----------------------------------------------------------------
    def s0():
        for path, n, seed in ((cfg.data.path, 8, 1), (cfg.data.val_path, 4, 2)):
            p = Path(path)
            if not p.exists():
                p.parent.mkdir(parents=True, exist_ok=True)
                rows = synthetic_rows(n, n_words=cfg.data.max_length * 3, seed=seed)
                p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        return {"train": cfg.data.path, "val": cfg.data.val_path}
    rep.run("S0_synthetic_corpus", s0)

    import torch

    from eval_harness.kv_recovery.provenance import configure_determinism, device_info, package_versions, seed_everything

    seed_everything(cfg.seed)
    det = configure_determinism(cfg.deterministic)

    def s1():
        info = {"packages": package_versions(), "device": device_info(), "determinism": det}
        assert torch.cuda.is_available(), "no CUDA device"
        assert os.environ.get("CUBLAS_WORKSPACE_CONFIG"), "CUBLAS_WORKSPACE_CONFIG not exported"
        try:
            import fla  # noqa: F401
            info["fla"] = True
        except Exception:
            info["fla"] = False
        return info
    rep.run("S1_environment", s1)

    from eval_harness.hf_adapter import HFGenerateConfig
    from eval_harness.kv_compression import KnormSketch
    from eval_harness.kv_recovery.alignment import alignment_keys_for, check_alignment_has_gradient, hidden_loss, position_index
    from eval_harness.kv_recovery.checkpoint import apply_delta, frozen_sample_names, hashes_of, write_delta, write_identity_delta
    from eval_harness.kv_recovery.data import assert_disjoint, load_split
    from eval_harness.kv_recovery.hidden_states import gather_positions
    from eval_harness.kv_recovery.model_spec import attention_module_of, decoder_layers, inspect_model
    from eval_harness.kv_recovery.student import (Example, assert_budget, assert_no_hooks, build_compressor, load_adapter,
                                                  prefill_context, probe_block_continuation, resolve_segment_mode, run_segment,
                                                  segment_forward)
    from eval_harness.kv_recovery.trainable import (assert_no_stray_grads, assert_trainable, changed_parameters, first_trainable_layer,
                                                    freeze_all_but, parameter_summary, select_trainable, snapshot_parameters,
                                                    trainable_parameters)
    from eval_harness.kv_recovery.trainer import AlignmentSetup, check_same_model, train
    from eval_harness.research_adapter import ResearchAdapter

    state: Dict[str, Any] = {}

    # ---- S2 load ------------------------------------------------------------------------------
    def s2():
        student = load_adapter(cfg, compressed=True)
        teacher = load_adapter(cfg, compressed=False)
        state.update(student=student, teacher=teacher, model=student._model, spec=inspect_model(student._model))
        tp = dict(teacher._model.named_parameters())
        mism = [n for n, p in student._model.named_parameters() if not torch.equal(p, tp[n])]
        assert not mism, f"teacher/student differ: {mism[:3]}"
        detail = {"family": state["spec"].family, "n_layers": state["spec"].n_layers,
                  "full_attention_layers": list(state["spec"].full_attention_layers),
                  "attn_implementation": getattr(student._model.config, "_attn_implementation", None),
                  "dtype": str(next(student._model.parameters()).dtype), "n_params": sum(p.numel() for p in student._model.parameters())}
        if cfg.model.dequantize_fp8:
            from safetensors import safe_open
            from transformers.utils.hub import cached_file

            path = cached_file(cfg.model.name, "model.safetensors")
            params = dict(student._model.named_parameters())
            checked, bad = 0, []
            with safe_open(path, framework="pt") as f:
                keys = set(f.keys())
                for k in keys:
                    if not k.endswith(".weight") or f"{k}_scale_inv" not in keys:
                        continue
                    w = f.get_tensor(k); s = f.get_tensor(f"{k}_scale_inv")
                    name = k
                    for pre in ("language_model.model.", "language_model."):
                        if name.startswith(pre):
                            name = "model.language_model." + name[len(pre):]
                            break
                    if name not in params:
                        continue
                    ref = (w.float() * s.float()).to(params[name].dtype).to(params[name].device)
                    if not torch.equal(ref, params[name].detach()):
                        bad.append(name)
                    checked += 1
            assert checked > 0, "no FP8 tensors found to verify"
            assert not bad, f"{len(bad)} dequantised tensors differ from fp8*scale, e.g. {bad[:3]}"
            left = [n for n, _ in student._model.named_buffers() if "scale_inv" in n]
            assert not left, f"leftover FP8 scale buffers {left[:3]}"
            detail["fp8_tensors_verified_bitwise"] = checked
        return detail
    if not rep.run("S2_load_and_dequant", s2):
        print(json.dumps(rep.checks, indent=2, default=str)); return 1
    student, teacher, model, spec = state["student"], state["teacher"], state["model"], state["spec"]
    tokenizer = student._tokenizer

    # ---- S3 prompt shaping -----------------------------------------------------------------------
    def s3():
        enc = student._pipe.preprocess("Alpha beta gamma.", questions=["Q?"], answer_prefix="", max_context_length=int(1e10),
                                       use_chat_template=True, strip_auto_system_block=True)
        prefix = tokenizer.decode(enc["context_ids"][0][:40], skip_special_tokens=False)
        assert "[SYSTEM_PROMPT]" not in prefix and "Cutting Knowledge" not in prefix, f"system block present: {prefix!r}"
        raw = student._pipe.preprocess("Alpha beta gamma.", questions=["Q?"], answer_prefix="", max_context_length=int(1e10),
                                       use_chat_template=False)
        return {"chat_prefix_ids": enc["context_ids"][0][:16].tolist(), "chat_prefix_text": prefix[:120],
                "raw_first_ids": raw["context_ids"][0][:4].tolist(), "bos_token_id": getattr(tokenizer, "bos_token_id", None)}
    rep.run("S3_prompt_shaping", s3)

    # ---- S4 data ---------------------------------------------------------------------------------
    def s4():
        tr, trs = load_split(cfg, tokenizer, "train", model=model, pipeline=student._pipe)
        va, vas = load_split(cfg, tokenizer, "val", model=model, pipeline=student._pipe)
        assert_disjoint(tr, va)
        for e in tr + va:
            assert e.context_len == cfg.data.max_length - cfg.data.suffix_length and e.suffix_len == cfg.data.suffix_length
        state.update(train=tr, val=va)
        return {"train": trs.as_dict(), "val": vas.as_dict()}
    if not rep.run("S4_data_windows", s4):
        print(json.dumps(rep.checks, indent=2, default=str)); return 1
    train_ex, val_ex = state["train"], state["val"]
    compressor = build_compressor(cfg)
    ratio = float(cfg.kv_compression.compression_ratio)
    mode = resolve_segment_mode(model, cfg.student)

    # ---- S5 hooks ----------------------------------------------------------------------------------
    def s5():
        with compressor(model):
            hooked = [i for i, l in enumerate(decoder_layers(model))
                      if (attention_module_of(l) is not None and attention_module_of(l)._forward_hooks)]
        assert hooked == list(spec.full_attention_layers), f"hooked {hooked} != {list(spec.full_attention_layers)}"
        assert_no_hooks(model); assert_no_hooks(teacher._model)
        return {"hooked_layers": hooked}
    rep.run("S5_compressor_hooks", s5)

    # ---- S6 budget ---------------------------------------------------------------------------------
    def s6():
        T, L = 4096, 64
        g = torch.Generator().manual_seed(0)
        vocab = int(spec.hidden_size and model.config.get_text_config().vocab_size)
        ctx = torch.randint(5, min(vocab, 30000), (1, T), generator=g)
        seg = torch.randint(5, min(vocab, 30000), (1, L), generator=g)
        calls = {"n": 0}
        orig = compressor.compress

        def spy(*a, **k):
            calls["n"] += 1
            return orig(*a, **k)
        compressor.compress = spy
        try:
            cache = prefill_context(student, ctx, compressor)
        finally:
            del compressor.compress
        lens = assert_budget(cache, spec, T, ratio)
        assert calls["n"] == len(spec.full_attention_layers), f"compress called {calls['n']} times"
        with torch.no_grad():
            segment_forward(model, cache, seg, T, logits_to_keep=1, mode=mode)
        assert_budget(cache, spec, T, ratio, suffix_appended=L)
        tcache = prefill_context(teacher, ctx, None)
        assert_budget(tcache, inspect_model(teacher._model), T, 0.0)
        return {"T": T, "kept_per_layer": lens[spec.full_attention_layers[0]], "compress_calls": calls["n"]}
    rep.run("S6_budget", s6)

    # ---- S7 continuation probe -----------------------------------------------------------------------
    def s7():
        # bf16 noise floor measured on H200 (block vs full-sequence SDPA shapes, sdpa): relative Frobenius
        # error up to 1.1e-2 at middle layers with per-position cosine >= 0.9998 on both Ministral-3-3B
        # and Qwen3.5-4B; a dropped cache / recurrent state gives O(1) error and cosine << 0.999.
        probe = probe_block_continuation(model, student._cache_adapter, T=1024, L=64, rtol=2e-2, min_cos=0.999)
        bad = {m: {k: v for k, v in probe["layers"][m].items() if not v["ok"]} for m in ("block", "token_by_token")}
        assert probe["block_ok"], f"block continuation differs from the full forward: {bad['block']}"
        assert probe["token_by_token_ok"], f"token-by-token continuation differs: {bad['token_by_token']}"
        worst = max(v["rel_frobenius"] for v in probe["layers"]["block"].values())
        return {"segment_mode": mode, "worst_rel_frobenius_block": worst,
                "min_cos_block": min(v["min_cos"] for v in probe["layers"]["block"].values()),
                "min_cos_token_by_token": min(v["min_cos"] for v in probe["layers"]["token_by_token"].values())}
    rep.run("S7_segment_continuation", s7)

    # ---- S14 compression-sensitivity layer selection -----------------------------------------------------
    sens: Dict[str, Any] = {}

    def s14():
        from eval_harness.kv_recovery.config import LAYER_SELECTED_STRATEGIES
        from eval_harness.kv_recovery.sensitivity import candidate_layers, measure_layer_sensitivity, select_layers_by_sensitivity
        used = {e.id for e in train_ex} | {e.id for e in val_ex}
        calib, cstats = load_split(cfg, tokenizer, "calibration", model=model, pipeline=student._pipe, exclude_ids=used)
        assert_disjoint(train_ex, calib); assert_disjoint(val_ex, calib)
        strategy = cfg.trainable.strategy if cfg.trainable.strategy in LAYER_SELECTED_STRATEGIES else "attention_projections"
        r1 = select_layers_by_sensitivity(cfg, teacher, student, compressor, calib, spec=spec, mode=mode, strategy=strategy)
        r2 = select_layers_by_sensitivity(cfg, teacher, student, compressor, calib, spec=spec, mode=mode, strategy=strategy)
        assert r1.scores == r2.scores and r1.selected == r2.selected, "sensitivity measurement is not deterministic"
        assert all(math.isfinite(v) and v >= 0.0 for v in r1.scores.values()), r1.scores
        hooked = list(spec.full_attention_layers)
        assert all(r1.scores[l] > 0.0 for l in hooked), {l: r1.scores[l] for l in hooked}
        pre = [l for l in r1.layers if l < spec.first_full_attention_layer]
        assert all(r1.scores[l] == 0.0 for l in pre), {l: r1.scores[l] for l in pre}      # untouched by K/V pruning
        cands = candidate_layers(strategy, spec)
        assert set(r1.selected) <= set(cands), (r1.selected, cands)
        assert len(r1.selected) == min(int(cfg.trainable.sensitivity.top_k), len(cands)), r1.selected
        assert r1.ranking == sorted(r1.layers, key=lambda l: (-r1.scores[l], -l)), "ranking inconsistent with scores"
        zero = measure_layer_sensitivity(teacher, student, None, calib[:1], spec=spec, mode=mode)
        assert all(v == 0.0 for sc in zero.values() for v in sc.values()), zero       # no compressor -> no divergence
        assert_no_hooks(model); assert_no_hooks(teacher._model)
        print(r1.table(), flush=True)
        (run_dir / "layer_sensitivity.json").write_text(json.dumps(r1.to_dict(), indent=2))
        sens.update(report=r1, calibration=calib)
        return {"selected": r1.selected, "ranking": r1.ranking, "scores": {str(l): round(v, 6) for l, v in r1.scores.items()},
                "n_calibration": len(calib), "calibration_stats": cstats.as_dict(), "seconds": r1.seconds, "notes": r1.notes}
    rep.run("S14_layer_sensitivity_selection", s14)
    resolved = sens["report"].selected if "report" in sens else None
    if cfg.trainable.layers == "sensitivity" and resolved is None:
        print(json.dumps(rep.checks, indent=2, default=str)); return 1

    # ---- trainable + alignment setup -------------------------------------------------------------------
    names = select_trainable(model, spec, cfg.trainable, resolved_layers=resolved)
    expected = freeze_all_but(model, names)
    summary = parameter_summary(model, spec)
    first_tl = summary["first_trainable_layer"]
    keys = alignment_keys_for(cfg.alignment, spec.n_layers, first_trainable_layer=first_tl)
    setup = AlignmentSetup(keys=keys, layer_indices=[k for k in keys if isinstance(k, int)],
                           include_final_norm=cfg.alignment.include_final_norm, positions_cfg=cfg.alignment.positions,
                           loss_name=cfg.alignment.loss, layer_weights=None, loss_cfg=cfg.loss, want_logits=cfg.loss.kl_weight > 0,
                           mode=mode, spec=spec, compression_ratio=ratio, prefill_chunk_size=None, prefill_grad=cfg.student.prefill_grad)

    # ---- S8 same model ----------------------------------------------------------------------------------
    def s8():
        res = check_same_model(teacher, student, train_ex[:2], setup)
        assert res["passed"], res
        return {"bitwise": [r["bitwise_equal"] for r in res["examples"]], "max_abs": max(r["max_abs_diff"] for r in res["examples"])}
    rep.run("S8_same_model_loss_zero", s8)

    # ---- S9 compression increases divergence ----------------------------------------------------------------
    def s9():
        from eval_harness.kv_recovery.config import build_research_config
        losses = {}
        for r in (0.5, 0.75):
            c2 = copy.deepcopy(cfg); c2.kv_compression.compression_ratio = r
            comp = ResearchAdapter._build_kv_compressor(build_research_config(c2, compressed=True))
            vals = []
            for ex in val_ex[:2]:
                t = run_segment(teacher, ex, None, setup.layer_indices, include_final_norm=True, grad=False, want_logits=False, mode=mode, spec=spec)
                s = run_segment(student, ex, comp, setup.layer_indices, include_final_norm=True, grad=False, want_logits=False, mode=mode, spec=spec, compression_ratio=r)
                pos = position_index(cfg.alignment.positions, ex.suffix_len, device=next(iter(s.states.values())).device)
                loss, _, _ = hidden_loss(gather_positions(s.states, pos), gather_positions(t.states, pos), "normalized_mse", keys)
                vals.append(float(loss))
            losses[str(r)] = sum(vals) / len(vals)
        assert losses["0.75"] > losses["0.5"] > 0, losses
        return losses
    rep.run("S9_compression_increases_divergence", s9)

    # ---- S10 trainable -----------------------------------------------------------------------------------
    def s10():
        assert_trainable(model, expected, spec)
        dead = check_alignment_has_gradient(keys, first_tl, allow=cfg.alignment.allow_dead_terms)
        ex = train_ex[0]
        from eval_harness.kv_recovery.trainer import student_loss, teacher_states_for
        ts = teacher_states_for(teacher, ex, setup, compressor=compressor)
        loss, info = student_loss(student, compressor, ex, ts, setup, grad=True)
        assert torch.isfinite(loss), "non-finite loss"
        loss.backward()
        assert_no_stray_grads(model, names)
        for p in trainable_parameters(model, names).values():
            assert torch.isfinite(p.grad).all()
            p.grad = None
        return {"trainable_parameters": summary["trainable_parameters"], "percent_of_text_lm": summary["percent_trainable_of_text_lm"],
                "first_trainable_layer": first_tl, "aligned_keys": [str(k) for k in keys], "dead_keys": [str(k) for k in dead],
                "loss": float(loss.detach()), "per_layer": info["per_layer"]}
    rep.run("S10_trainable_and_gradients", s10)

    # ---- S11 train + checkpoint round trip ------------------------------------------------------------------
    originals = snapshot_parameters(model, names)
    original_sha = hashes_of(model, names)
    frozen_names = frozen_sample_names(model, names, spec)
    frozen_snap = snapshot_parameters(model, frozen_names)

    def s11():
        trainable = trainable_parameters(model, names)
        log = run_dir / "smoke_train_metrics.jsonl"; log.write_text("")
        st = train(teacher, student, compressor, trainable, train_ex, val_ex, cfg, setup, log_path=log, print_fn=lambda *a, **k: None)
        assert st.optimizer_steps == cfg.optim.max_steps, st.optimizer_steps
        assert all(torch.isfinite(torch.tensor(r["loss"])) for r in st.step_logs)
        assert not changed_parameters(model, frozen_snap), "frozen tensors changed"
        tp = dict(teacher._model.named_parameters())
        drift = [n for n, p in model.named_parameters() if n not in trainable and not torch.equal(p, tp[n])]
        assert not drift, f"frozen drift {drift[:3]}"
        changed = changed_parameters(model, originals)
        assert changed, "no trainable tensor changed"
        ckpt = run_dir / "checkpoint"
        if ckpt.exists():
            shutil.rmtree(ckpt)
        from eval_harness.kv_recovery.config import training_identity
        from eval_harness.kv_recovery.model_spec import resolve_base_revision
        write_delta(ckpt, model, names, original_sha,
                    {"base_model": cfg.model.name, "base_revision": resolve_base_revision(model), "smoke": True,
                     **training_identity(cfg)},
                    masters=st.masters, frozen_sample_sha256=hashes_of(teacher._model, frozen_names))
        trained = {n: p.detach().cpu().clone() for n, p in trainable.items()}
        fresh = load_adapter(cfg, compressed=True)
        info = apply_delta(fresh._model, ckpt, strict=True)
        fp = dict(fresh._model.named_parameters())
        assert all(torch.equal(fp[n].cpu(), trained[n]) for n in names), "reloaded delta differs from the trained tensors"
        assert all(torch.equal(fp[n], tp[n]) for n in fp if n not in trainable), "fresh load + delta changed frozen tensors"
        del fresh; torch.cuda.empty_cache()
        # optimisation sanity (spec §22): fixed 2-example batch, 8 steps at lr 1e-4 -> loss decreases
        with torch.no_grad():
            for n, p in trainable.items():
                p.copy_(originals[n].to(p.device))
        c2 = copy.deepcopy(cfg); c2.optim.max_steps = 8; c2.optim.grad_accum = 1; c2.optim.epochs = 8; c2.optim.learning_rate = 1e-4; c2.optim.val_every_steps = 8
        fixed = train_ex[:2]
        val2 = [Example(id="v" + e.id, ctx_ids=e.ctx_ids, suffix_ids=e.suffix_ids) for e in fixed]
        log2 = run_dir / "smoke_fixed_batch.jsonl"; log2.write_text("")
        st2 = train(teacher, student, compressor, trainable, fixed, val2, c2, setup, log_path=log2, print_fn=lambda *a, **k: None)
        assert st2.step_logs[-1]["loss"] < st2.step_logs[0]["loss"], (st2.step_logs[0]["loss"], st2.step_logs[-1]["loss"])
        return {"steps": st.optimizer_steps, "train_loss": [r["loss"] for r in st.step_logs], "val": st.val_logs,
                "applied": len(info["applied"]), "n_changed": len(changed), "peak_gib": st.peak_gpu_memory_gib,
                "fixed_batch_loss_first_last": (st2.step_logs[0]["loss"], st2.step_logs[-1]["loss"])}
    rep.run("S11_train_checkpoint_roundtrip", s11)

    # ---- S12 identity delta --------------------------------------------------------------------------------------
    def s12():
        trainable = trainable_parameters(model, names)
        with torch.no_grad():
            for n, p in trainable.items():
                p.copy_(originals[n].to(p.device))
        ctx_text = " ".join(random.Random(7).choice(WORDS) for _ in range(600))
        gen = HFGenerateConfig(max_tokens=12)
        before = student.generate_for_context(ctx_text, ["What is the last word?"], "", gen)
        ident = run_dir / "identity_checkpoint"
        if ident.exists():
            shutil.rmtree(ident)
        write_identity_delta(ident, model, names, {"base_model": cfg.model.name})
        apply_delta(model, ident, strict=True)
        after = student.generate_for_context(ctx_text, ["What is the last word?"], "", gen)
        again = student.generate_for_context(ctx_text, ["What is the last word?"], "", gen)
        assert before == after == again, {"before": before, "after": after, "again": again}
        return {"generation": before[0][:80], "deterministic_repeat": before == again}
    rep.run("S12_identity_delta_bitwise", s12)

    # ---- S13 benchmark three-way (optional) -----------------------------------------------------------------------
    if args.with_benchmark:
        def s13():
            del state["student"], state["teacher"]
            nonlocal student, teacher
            student = teacher = None
            import gc
            gc.collect(); torch.cuda.empty_cache()
            import subprocess
            cmd = [sys.executable, str(REPO_ROOT / "scripts" / "eval_kv_recovery.py"), "run", "--config", args.config,
                   "--run-dir", str(run_dir), "--checkpoint", str(run_dir / "checkpoint"), "--local"]
            out = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
            assert out.returncode == 0, out.stderr[-2000:]
            cmd = [sys.executable, str(REPO_ROOT / "scripts" / "eval_kv_recovery.py"), "report", "--config", args.config,
                   "--run-dir", str(run_dir), "--n-resamples", "500"]
            out = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
            assert out.returncode == 0, out.stderr[-2000:]
            res = json.loads((run_dir / "eval_results.json").read_text())
            assert "ruler16k" in res["benchmarks"], res.get("skipped")
            o = res["benchmarks"]["ruler16k"]["overall"]
            assert res["benchmarks"]["ruler16k"]["comparability"]["ok"]
            return {"dense": o["dense"], "compressed": o["compressed"], "compressed_recovered": o["compressed_recovered"],
                    "recovery_fraction": o["recovery_fraction"], "flags": o["flags"]}
        rep.run("S13_three_way_benchmark", s13)

    print(f"\nsmoke report: {rep.path}  all_passed={rep.all_passed}")
    return 0 if rep.all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
