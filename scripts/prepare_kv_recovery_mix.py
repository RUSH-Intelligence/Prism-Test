#!/usr/bin/env python
"""Curate a MIXED distillation corpus for hidden-state KV recovery — benchmark-shaped rows that are
outside the evaluated pool plus long natural text — as one JSONL per split with a manifest.

  python scripts/prepare_kv_recovery_mix.py --name mix16k --context 16k            # login node, network
  python scripts/prepare_kv_recovery_mix.py --name mix32k --context 32k

Sources (every count is a CLI knob; the manifest records the final composition):

* RULER (``kind: qa``): rows of ``ruler16k`` (16K) / ``ruler32k`` (32K) OUTSIDE the evaluated pool. The
  evaluation scores rows 0-99 of every task (``max_requests: 100``); training rows come from
  ``--ruler-train-rows`` (default 120-199, the last 80) and validation / calibration rows from
  ``--ruler-val-rows`` (default 100-119). Row-index disjointness is the leakage guarantee here —
  RULER haystacks share their essay text across rows, so a shingle filter would reject everything.
* LongBench (``kind: qa``): rows ``--longbench-rows`` (default 100-199) of the 16 English tasks, which
  requires the LongBench EVALUATION to use rows 0-99 (``eval.benchmarks[longbench].max_requests: 100``,
  as the ``*_mix.yaml`` cards do). Contexts longer than ``--qa-max-chars`` are skipped (the trainer
  applies the exact token cap ``data.max_context_tokens``).
* PG-19 (``kind: text``): the excerpts written by ``prepare_kv_recovery_data.py`` (already
  leakage-filtered), ``--pg19-n`` train / ``--pg19-val-n`` val rows.
* FineWeb-Edu (``kind: text``): documents of ``HuggingFaceFW/fineweb-edu`` (``sample-10BT``, streamed,
  seeded shuffle) packed back to back with blank lines until ``--fineweb-chars`` characters, so a
  packed row tokenises to at least ``max_length`` tokens; train and val rows use disjoint documents.
  Packed text rows are shingle-checked against the evaluated contexts like the PG-19 excerpts.

qa rows store ``id, kind, source, task, row, context, question, answer_prefix, answer`` (answers as a
list); text rows store ``id, kind, source, text`` (+ ``doc_ids`` for packed rows). Validation rows are
disjoint from training rows by construction (different row ranges / documents / books).

Outputs (``--out-dir``, default ``data/kv_recovery``): ``<name>_train.jsonl``, ``<name>_val.jsonl``,
``<name>_manifest.json`` (composition, row ranges, eval pool definition, sha256, leakage report).
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eval_harness.kv_recovery.config import LONGBENCH_16  # noqa: E402
from scripts.prepare_kv_recovery_data import RULER_TASKS, leakage_scan, sha256_file, write_jsonl  # noqa: E402

CONTEXT_TOKENS = {"16k": 16384, "32k": 32768, "64k": 65536, "128k": 131072}
RULER_FOR_CONTEXT = {"16k": "ruler16k", "32k": "ruler32k"}
EVAL_POOL_ROWS = 100          # the evaluation scores rows [0, 100) of every task
CHARS_PER_TOKEN_QA = 4.2      # char prefilter for qa contexts (the trainer applies the exact token cap)


# ---------------------------------------------------------------------------
# helpers (pure, unit-tested)
# ---------------------------------------------------------------------------
def parse_range(text: str) -> Tuple[int, int]:
    """``'120-199'`` -> (120, 200) half-open; ``'137'`` -> (137, 138)."""
    text = str(text).strip()
    if "-" in text:
        a, b = text.split("-", 1)
        lo, hi = int(a), int(b) + 1
    else:
        lo, hi = int(text), int(text) + 1
    if lo < 0 or hi <= lo:
        raise ValueError(f"bad row range {text!r}")
    return lo, hi


def check_row_ranges(train: Tuple[int, int], val: Tuple[int, int], eval_rows: int = EVAL_POOL_ROWS) -> None:
    """Training / validation ranges must stay out of the evaluated pool and out of each other."""
    for name, (lo, hi) in (("train", train), ("val", val)):
        if lo < eval_rows:
            raise ValueError(f"{name} rows {lo}-{hi - 1} overlap the evaluated pool 0-{eval_rows - 1}")
    if max(train[0], val[0]) < min(train[1], val[1]):
        raise ValueError(f"train rows {train} and val rows {val} overlap")


def sample_rows(n_available: int, lo: int, hi: int, k: int, seed: str) -> List[int]:
    """``k`` seeded row indices from ``[lo, min(hi, n_available))`` (all of them when fewer exist)."""
    pool = list(range(lo, min(hi, n_available)))
    rng = random.Random(seed)
    return sorted(rng.sample(pool, min(k, len(pool))))


def pack_documents(docs: Iterable[Tuple[str, str]], target_chars: int, min_doc_chars: int = 200) -> Iterable[Dict[str, Any]]:
    """Pack ``(doc_id, text)`` pairs back to back (blank line between) into rows of >= ``target_chars``."""
    buf: List[str] = []
    ids: List[str] = []
    size = 0
    for doc_id, text in docs:
        text = (text or "").strip()
        if len(text) < min_doc_chars:
            continue
        buf.append(text)
        ids.append(str(doc_id))
        size += len(text) + 2
        if size >= target_chars:
            yield {"text": "\n\n".join(buf), "doc_ids": ids, "n_docs": len(ids), "chars": size}
            buf, ids, size = [], [], 0


def context_key(row: Dict[str, Any]) -> str:
    return hashlib.sha256((row.get("context") or row.get("text") or "").encode("utf-8")).hexdigest()


def split_by_context(rows: List[dict], k_val: int, *, seed: str) -> Tuple[List[dict], List[dict]]:
    """Hold out about ``k_val`` rows WITHOUT splitting a shared context (LongBench asks several questions
    about one document): contexts are grouped, groups are shuffled, and whole groups go to validation until
    ``k_val`` rows are reached. Returns ``(val_rows, train_rows)``; at least one group stays in training."""
    groups: Dict[str, List[dict]] = {}
    for r in rows:
        groups.setdefault(context_key(r), []).append(r)
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    val: List[dict] = []
    train: List[dict] = []
    for i, k in enumerate(keys):
        if len(val) < k_val and i < len(keys) - 1:
            val += groups[k]
        else:
            train += groups[k]
    return val, train


def drop_shared_contexts(val: List[dict], train: List[dict]) -> Tuple[List[dict], List[str]]:
    """Remove validation rows whose context also occurs in training (returns the kept rows and the dropped ids)."""
    keys = {context_key(r) for r in train}
    kept = [r for r in val if context_key(r) not in keys]
    dropped = [r["id"] for r in val if context_key(r) in keys]
    return kept, dropped


def qa_row(bench: str, task: str, row_idx: int, row: Dict[str, Any]) -> Dict[str, Any]:
    from eval_harness.benchmarks.common import parse_answers

    answers = parse_answers(row.get("answer", row.get("answers", "")))
    return {"id": f"{bench}/{task}/row{int(row_idx)}", "kind": "qa", "source": bench, "task": task, "row": int(row_idx),
            "context": str(row["context"]), "question": str(row["question"]),
            "answer_prefix": str(row.get("answer_prefix", "") or ""), "answer": answers}


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------
def benchmark_qa_rows(bench: str, tasks: List[str], row_range: Tuple[int, int], per_task: int, *, seed: int,
                      max_chars: Optional[int], tag: str) -> Tuple[List[dict], Dict[str, Any]]:
    from eval_harness.benchmarks.registry import get_benchmark

    df = get_benchmark(bench).load(tasks)
    out: List[dict] = []
    report: Dict[str, Any] = {}
    for task in tasks:
        sub = df[df["task"].astype(str) == task].reset_index(drop=True)
        lo, hi = row_range
        candidates = sample_rows(len(sub), lo, hi, len(sub), seed=f"{seed}:{bench}:{task}:{tag}")   # the whole eligible range, shuffled below
        rng = random.Random(f"{seed}:{bench}:{task}:{tag}:order")
        rng.shuffle(candidates)
        picked, skipped_long = [], 0
        for i in candidates:
            row = sub.iloc[i].to_dict()
            if max_chars is not None and len(str(row["context"])) > max_chars:
                skipped_long += 1
                continue
            picked.append(i)
            if len(picked) >= per_task:
                break
        for i in sorted(picked):
            out.append(qa_row(bench, task, i, sub.iloc[i].to_dict()))
        report[task] = {"rows_available": int(len(sub)), "range": [lo, hi - 1], "picked": sorted(picked),
                        "skipped_long": skipped_long}
    return out, report


def pg19_rows(path: Path, n: int, seed: int, tag: str) -> List[dict]:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    rng = random.Random(f"{seed}:pg19:{tag}")
    rng.shuffle(rows)
    return [{"id": r["id"], "kind": "text", "source": "pg19", "title": r.get("title"), "url": r.get("url"), "text": r["text"]}
            for r in rows[:n]]


def fineweb_rows(n_train: int, n_val: int, target_chars: int, *, seed: int, name: str = "HuggingFaceFW/fineweb-edu",
                 config: str = "sample-10BT", buffer_size: int = 2000) -> Tuple[List[dict], List[dict]]:
    from datasets import load_dataset

    ds = load_dataset(name, name=config, split="train", streaming=True).shuffle(seed=seed, buffer_size=buffer_size)
    docs = ((r["id"], r["text"]) for r in ds)
    packed = pack_documents(docs, target_chars)
    rows = []
    for i, p in enumerate(itertools.islice(packed, n_train + n_val)):
        rows.append({"id": f"fineweb_edu-{i:05d}", "kind": "text", "source": "fineweb_edu", "text": p["text"],
                     "doc_ids": p["doc_ids"], "n_docs": p["n_docs"]})
    if len(rows) < n_train + n_val:
        raise RuntimeError(f"fineweb-edu stream yielded only {len(rows)} packed rows of {n_train + n_val}")
    return rows[:n_train], rows[n_train:]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="corpus name, e.g. mix16k")
    ap.add_argument("--context", required=True, choices=list(CONTEXT_TOKENS), help="window length the corpus targets")
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "data" / "kv_recovery"))
    ap.add_argument("--seed", type=int, default=42)
    # RULER
    ap.add_argument("--ruler-benchmark", help="default by --context: ruler16k | ruler32k")
    ap.add_argument("--ruler-train-rows", default="120-199")
    ap.add_argument("--ruler-val-rows", default="100-119")
    ap.add_argument("--ruler-rows-per-task", type=int, default=60)
    ap.add_argument("--ruler-val-rows-per-task", type=int, default=8)
    ap.add_argument("--no-ruler", action="store_true")
    # LongBench
    ap.add_argument("--longbench-rows", default="100-199", help="rows outside the (100-row) evaluated pool")
    ap.add_argument("--longbench-rows-per-task", type=int, default=30)
    ap.add_argument("--longbench-val-rows-per-task", type=int, default=3)
    ap.add_argument("--no-longbench", action="store_true")
    # PG-19
    ap.add_argument("--pg19-train", default=str(REPO_ROOT / "data" / "kv_recovery" / "pg19_train.jsonl"))
    ap.add_argument("--pg19-val", default=str(REPO_ROOT / "data" / "kv_recovery" / "pg19_val.jsonl"))
    ap.add_argument("--pg19-n", type=int, default=256)
    ap.add_argument("--pg19-val-n", type=int, default=16)
    # FineWeb-Edu
    ap.add_argument("--fineweb-n", type=int, default=256)
    ap.add_argument("--fineweb-val-n", type=int, default=16)
    ap.add_argument("--fineweb-chars", type=int, help="packed characters per row (default 7 x context tokens)")
    ap.add_argument("--no-fineweb", action="store_true")
    # caps / leakage
    ap.add_argument("--qa-max-chars", type=int, help=f"skip qa contexts longer than this (default {CHARS_PER_TOKEN_QA} x context tokens)")
    ap.add_argument("--leakage-check", dest="leakage_check", action="store_true", default=True)
    ap.add_argument("--no-leakage-check", dest="leakage_check", action="store_false")
    ap.add_argument("--shingle-words", type=int, default=13)
    args = ap.parse_args(argv)

    n_tokens = CONTEXT_TOKENS[args.context]
    qa_max_chars = args.qa_max_chars or int(CHARS_PER_TOKEN_QA * n_tokens)
    fineweb_chars = args.fineweb_chars or 7 * n_tokens
    ruler_bench = args.ruler_benchmark or RULER_FOR_CONTEXT.get(args.context)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    train: List[dict] = []
    val: List[dict] = []
    manifest: Dict[str, Any] = {"name": args.name, "context": args.context, "context_tokens": n_tokens, "seed": args.seed,
                                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                "eval_pool": {"ruler16k": f"rows 0-{EVAL_POOL_ROWS - 1} of every task",
                                              "ruler32k": f"rows 0-{EVAL_POOL_ROWS - 1} of every task",
                                              "longbench": f"rows 0-{EVAL_POOL_ROWS - 1} of every task (eval.benchmarks[longbench].max_requests: 100)"},
                                "sources": {}}

    # --- RULER ------------------------------------------------------------------------
    if not args.no_ruler and ruler_bench:
        tr, va = parse_range(args.ruler_train_rows), parse_range(args.ruler_val_rows)
        check_row_ranges(tr, va)
        rows_tr, rep_tr = benchmark_qa_rows(ruler_bench, RULER_TASKS, tr, args.ruler_rows_per_task, seed=args.seed,
                                            max_chars=None, tag="train")
        rows_va, rep_va = benchmark_qa_rows(ruler_bench, RULER_TASKS, va, args.ruler_val_rows_per_task, seed=args.seed,
                                            max_chars=None, tag="val")
        train += rows_tr; val += rows_va
        manifest["sources"][ruler_bench] = {"kind": "qa", "train_rows": [tr[0], tr[1] - 1], "val_rows": [va[0], va[1] - 1],
                                            "n_train": len(rows_tr), "n_val": len(rows_va), "per_task_train": rep_tr, "per_task_val": rep_va}
        print(f"{ruler_bench}: {len(rows_tr)} train / {len(rows_va)} val qa rows ({time.time() - t0:.0f}s)", flush=True)

    # --- LongBench -------------------------------------------------------------------
    if not args.no_longbench:
        lo, hi = parse_range(args.longbench_rows)
        if lo < EVAL_POOL_ROWS:
            raise SystemExit(f"--longbench-rows must start at {EVAL_POOL_ROWS} or later (the evaluation uses rows 0-{EVAL_POOL_ROWS - 1})")
        per = args.longbench_rows_per_task + args.longbench_val_rows_per_task
        rows_all, rep = benchmark_qa_rows("longbench", LONGBENCH_16, (lo, hi), per, seed=args.seed, max_chars=qa_max_chars, tag="all")
        by_task: Dict[str, List[dict]] = {}
        for r in rows_all:
            by_task.setdefault(r["task"], []).append(r)
        rows_tr, rows_va = [], []
        for task, rows in by_task.items():
            rows_va_t, rows_tr_t = split_by_context(rows, args.longbench_val_rows_per_task, seed=f"{args.seed}:longbench:{task}:split")
            rows_va += rows_va_t; rows_tr += rows_tr_t
        train += rows_tr; val += rows_va
        manifest["sources"]["longbench"] = {"kind": "qa", "rows": [lo, hi - 1], "qa_max_chars": qa_max_chars, "n_train": len(rows_tr),
                                            "n_val": len(rows_va), "per_task": rep,
                                            "val_ids": sorted(r["id"] for r in rows_va)}
        print(f"longbench: {len(rows_tr)} train / {len(rows_va)} val qa rows (contexts <= {qa_max_chars} chars) ({time.time() - t0:.0f}s)", flush=True)

    # --- PG-19 ------------------------------------------------------------------------
    if args.pg19_n > 0:
        rows_tr = pg19_rows(Path(args.pg19_train), args.pg19_n, args.seed, "train")
        rows_va = pg19_rows(Path(args.pg19_val), args.pg19_val_n, args.seed, "val")
        train += rows_tr; val += rows_va
        manifest["sources"]["pg19"] = {"kind": "text", "train_file": args.pg19_train, "val_file": args.pg19_val,
                                       "n_train": len(rows_tr), "n_val": len(rows_va)}
        print(f"pg19: {len(rows_tr)} train / {len(rows_va)} val text rows", flush=True)

    # --- FineWeb-Edu -----------------------------------------------------------------
    if not args.no_fineweb and args.fineweb_n > 0:
        rows_tr, rows_va = fineweb_rows(args.fineweb_n, args.fineweb_val_n, fineweb_chars, seed=args.seed)
        train += rows_tr; val += rows_va
        manifest["sources"]["fineweb_edu"] = {"kind": "text", "dataset": "HuggingFaceFW/fineweb-edu:sample-10BT",
                                              "packed_chars": fineweb_chars, "n_train": len(rows_tr), "n_val": len(rows_va),
                                              "mean_docs_per_row": sum(r["n_docs"] for r in rows_tr + rows_va) / max(len(rows_tr) + len(rows_va), 1)}
        print(f"fineweb_edu: {len(rows_tr)} train / {len(rows_va)} val packed text rows ({time.time() - t0:.0f}s)", flush=True)

    # --- leakage check for TEXT rows against the evaluated contexts ----------------------
    report: Dict[str, Any] = {"enabled": bool(args.leakage_check)}
    if args.leakage_check:
        text_rows = [r for r in train + val if r.get("kind", "text") == "text"]
        if text_rows:
            scan = leakage_scan(text_rows, ["longbench", "ruler16k", "ruler32k"],
                                {"longbench": EVAL_POOL_ROWS, "ruler16k": EVAL_POOL_ROWS, "ruler32k": EVAL_POOL_ROWS}, args.shingle_words)
            hit_ids = {text_rows[i]["id"] for i in scan["candidate_hits"]}
            before = len(train) + len(val)
            train = [r for r in train if r["id"] not in hit_ids]
            val = [r for r in val if r["id"] not in hit_ids]
            report.update({k: v for k, v in scan.items() if k != "candidate_hits"})
            report["rejected_ids"] = sorted(hit_ids)
            print(f"leakage scan: {scan['n_benchmark_contexts']} contexts in {scan['seconds']}s; rejected {before - len(train) - len(val)} text rows", flush=True)

    # --- disjointness + write ---------------------------------------------------------------
    ids_tr, ids_va = {r["id"] for r in train}, {r["id"] for r in val}
    if ids_tr & ids_va:
        raise SystemExit(f"train/val share ids: {sorted(ids_tr & ids_va)[:5]}")
    val, dropped = drop_shared_contexts(val, train)          # e.g. a RULER haystack reused across row ranges
    if dropped:
        print(f"dropped {len(dropped)} validation rows whose context also occurs in training: {dropped[:4]}", flush=True)
    manifest["val_rows_dropped_shared_context"] = dropped
    if {context_key(r) for r in train} & {context_key(r) for r in val}:
        raise SystemExit("train/val share a context/text")
    rng = random.Random(args.seed)
    rng.shuffle(train); rng.shuffle(val)
    train_path, val_path = out_dir / f"{args.name}_train.jsonl", out_dir / f"{args.name}_val.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(val_path, val)
    comp = lambda rows: {s: sum(1 for r in rows if r["source"] == s) for s in sorted({r["source"] for r in rows})}  # noqa: E731
    manifest.update({"composition": {"train": comp(train), "val": comp(val)}, "n_train": len(train), "n_val": len(val),
                     "files": {train_path.name: sha256_file(train_path), val_path.name: sha256_file(val_path)},
                     "leakage_report": report, "seconds": round(time.time() - t0, 1)})
    (out_dir / f"{args.name}_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"wrote {train_path} ({len(train)}: {manifest['composition']['train']}) and {val_path} ({len(val)}: {manifest['composition']['val']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
