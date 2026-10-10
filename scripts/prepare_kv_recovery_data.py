#!/usr/bin/env python
"""Materialise the KV-recovery training corpus: seeded long-text excerpts in JSONL.

Default source is PG-19 via the parquet mirror ``emozilla/pg19`` (fields
``short_book_title, publication_date, url, text``; splits train 13,684 /
validation 50 / test 100 books). Training excerpts come from ``train`` and
validation excerpts from ``validation`` — disjoint BOOKS by construction.

Each row stores an EXCERPT (``--excerpt-chars`` characters from a seeded offset)
rather than the whole book, so the trainer tokenises deterministically from the
excerpt start (it needs ``data.max_length`` tokens; ~6 chars/token leaves ample
margin) and so the leakage filter below covers exactly the text that can be used.

Leakage filter (``--leakage-check``, default on): every evaluated benchmark context
(LongBench 16 English tasks x 200 rows; RULER-16K/32K rows 0-99 of the 13 tasks) is
scanned for 13-word shingles shared with any candidate excerpt. Candidates with a
hit are dropped (LongBench ``narrativeqa`` contexts are Project-Gutenberg books, as
is PG-19). The scan also repopulates the HF dataset cache the eval jobs need.

Outputs (``--out-dir``, default ``data/kv_recovery``):
  <name>_train.jsonl, <name>_val.jsonl                rows {id, source, split, title, url, offset, text}
  <name>_manifest.json                                seeds, counts, sha256 of each JSONL, source revision
  <name>_leakage_report.json                          hits per benchmark/task, rejected candidates

Example (login node, network available):
  python scripts/prepare_kv_recovery_data.py --num-train 256 --num-val 32 --seed 42
  python scripts/prepare_kv_recovery_data.py --source my_corpus.jsonl --name mine --no-leakage-check
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eval_harness.kv_recovery.config import LONGBENCH_16  # noqa: E402

RULER_TASKS = ["cwe", "fwe", "niah_multikey_1", "niah_multikey_2", "niah_multikey_3", "niah_multiquery",
               "niah_multivalue", "niah_single_1", "niah_single_2", "niah_single_3", "qa_1", "qa_2", "vt"]
_WORD_RE = re.compile(r"\w+")


# ---------------------------------------------------------------------------
# shingles
# ---------------------------------------------------------------------------
def words_of(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


def shingle_hashes(text: str, k: int) -> np.ndarray:
    """uint64 hashes of every k-word shingle of ``text`` (blake2b-8, deterministic)."""
    w = words_of(text)
    if len(w) < k:
        return np.zeros(0, dtype=np.uint64)
    out = np.empty(len(w) - k + 1, dtype=np.uint64)
    for i in range(len(w) - k + 1):
        h = hashlib.blake2b(" ".join(w[i:i + k]).encode("utf-8"), digest_size=8).digest()
        out[i] = int.from_bytes(h, "little")
    return out


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------
def iter_hf_books(source: str, split: str, seed: int, buffer_size: int) -> Iterator[dict]:
    from datasets import load_dataset

    ds = load_dataset(source, split=split, streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=buffer_size)
    for row in ds:
        yield row


def iter_jsonl_books(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def source_revision(source: str) -> Optional[str]:
    try:
        from huggingface_hub import HfApi

        return HfApi().dataset_info(source).sha
    except Exception:
        return None


def take_excerpt(text: str, excerpt_chars: int, rng: random.Random) -> tuple[int, str]:
    if len(text) <= excerpt_chars:
        return 0, text
    offset = rng.randint(0, len(text) - excerpt_chars)
    return offset, text[offset:offset + excerpt_chars]


def collect_candidates(source: str, split: str, n: int, *, seed: int, min_chars: int, excerpt_chars: int,
                       buffer_size: int, name: str) -> List[dict]:
    rng = random.Random(seed)
    rows: List[dict] = []
    books = iter_jsonl_books(Path(source)) if Path(source).exists() else iter_hf_books(source, split, seed, buffer_size)
    for raw in books:
        text = raw.get("text") or ""
        if len(text) < min_chars:
            continue
        offset, excerpt = take_excerpt(text, excerpt_chars, rng)
        rows.append({
            "id": f"{name}-{split}-{len(rows):05d}",
            "source": source,
            "split": split,
            "title": raw.get("short_book_title") or raw.get("title"),
            "url": raw.get("url"),
            "publication_date": raw.get("publication_date"),
            "offset": offset,
            "book_chars": len(text),
            "text": excerpt,
        })
        if len(rows) >= n:
            break
    return rows


# ---------------------------------------------------------------------------
# leakage scan
# ---------------------------------------------------------------------------
def iter_benchmark_contexts(benchmarks: List[str], max_requests: Dict[str, int]) -> Iterator[tuple[str, str, int, str]]:
    from eval_harness.benchmarks.registry import get_benchmark

    for bench in benchmarks:
        b = get_benchmark(bench)
        subsets = list(LONGBENCH_16) if bench == "longbench" else list(RULER_TASKS)
        for subset in subsets:
            df = b.load([subset])
            n = max_requests.get(bench)
            if n is not None:
                df = df.iloc[:n]
            for i, ctx in enumerate(df["context"].tolist()):
                yield bench, subset, i, str(ctx)


def leakage_scan(candidates: List[dict], benchmarks: List[str], max_requests: Dict[str, int], k: int) -> dict:
    per_cand = [shingle_hashes(c["text"], k) for c in candidates]
    all_hashes = np.concatenate([h for h in per_cand if len(h)]) if per_cand else np.zeros(0, dtype=np.uint64)
    order = np.argsort(all_hashes, kind="stable")
    sorted_hashes = all_hashes[order]
    owner = np.concatenate([np.full(len(h), i, dtype=np.int64) for i, h in enumerate(per_cand)])[order]
    hits: Dict[int, List[dict]] = {}
    per_bench: Dict[str, int] = {}
    n_contexts = 0
    t0 = time.time()
    for bench, subset, row, ctx in iter_benchmark_contexts(benchmarks, max_requests):
        n_contexts += 1
        h = shingle_hashes(ctx, k)
        if len(h) == 0 or len(sorted_hashes) == 0:
            continue
        pos = np.searchsorted(sorted_hashes, h)
        pos = np.minimum(pos, len(sorted_hashes) - 1)
        matched = sorted_hashes[pos] == h
        if matched.any():
            cands = np.unique(owner[pos[matched]])
            for ci in cands.tolist():
                hits.setdefault(int(ci), []).append({"benchmark": bench, "task": subset, "row": row,
                                                    "n_shared_shingles": int(matched.sum())})
            per_bench[f"{bench}/{subset}"] = per_bench.get(f"{bench}/{subset}", 0) + 1
    return {"k": k, "n_benchmark_contexts": n_contexts, "n_candidate_shingles": int(len(all_hashes)),
            "contexts_with_hits_per_task": per_bench, "candidate_hits": hits, "seconds": round(time.time() - t0, 1)}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="emozilla/pg19", help="HF dataset id (streaming) or a local JSONL with a 'text' field")
    ap.add_argument("--name", default="pg19")
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "data" / "kv_recovery"))
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--val-split", default="validation")
    ap.add_argument("--num-train", type=int, default=256)
    ap.add_argument("--num-val", type=int, default=32)
    ap.add_argument("--spare-fraction", type=float, default=0.5, help="extra candidates sampled for leakage rejection")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-chars", type=int, default=250_000, help="books shorter than this are skipped")
    ap.add_argument("--excerpt-chars", type=int, default=200_000, help="chars stored per row (~32K tokens at 6 chars/token)")
    ap.add_argument("--shuffle-buffer", type=int, default=400)
    ap.add_argument("--leakage-check", dest="leakage_check", action="store_true", default=True)
    ap.add_argument("--no-leakage-check", dest="leakage_check", action="store_false")
    ap.add_argument("--leakage-benchmarks", default="longbench,ruler16k,ruler32k")
    ap.add_argument("--leakage-max-requests", default="longbench=200,ruler16k=100,ruler32k=100")
    ap.add_argument("--shingle-words", type=int, default=13)
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_train_cand = int(args.num_train * (1 + args.spare_fraction)) + 1
    n_val_cand = int(args.num_val * (1 + args.spare_fraction)) + 1
    print(f"collecting {n_train_cand} train / {n_val_cand} val candidates from {args.source} ...", flush=True)
    train_c = collect_candidates(args.source, args.train_split, n_train_cand, seed=args.seed, min_chars=args.min_chars,
                                 excerpt_chars=args.excerpt_chars, buffer_size=args.shuffle_buffer, name=args.name)
    val_c = collect_candidates(args.source, args.val_split, n_val_cand, seed=args.seed + 1, min_chars=args.min_chars,
                               excerpt_chars=args.excerpt_chars, buffer_size=args.shuffle_buffer, name=args.name)
    if Path(args.source).exists():
        # A local corpus has a single split: carve val from the tail, disjoint rows.
        rows = train_c
        val_c = rows[-n_val_cand:] if len(rows) > n_val_cand else []
        train_c = rows[:-n_val_cand] if len(rows) > n_val_cand else rows
        for r in val_c:
            r["split"] = "validation"
    print(f"  got {len(train_c)} train / {len(val_c)} val candidates", flush=True)

    report = {"enabled": bool(args.leakage_check)}
    rejected_train, rejected_val = [], []
    if args.leakage_check:
        benches = [b.strip() for b in args.leakage_benchmarks.split(",") if b.strip()]
        maxreq = {kv.split("=")[0]: int(kv.split("=")[1]) for kv in args.leakage_max_requests.split(",") if "=" in kv}
        print(f"leakage scan against {benches} ({maxreq}) with {args.shingle_words}-word shingles ...", flush=True)
        scan = leakage_scan(train_c + val_c, benches, maxreq, args.shingle_words)
        hit_idx = set(scan["candidate_hits"].keys())
        n_tr = len(train_c)
        rejected_train = [train_c[i]["id"] for i in sorted(hit_idx) if i < n_tr]
        rejected_val = [val_c[i - n_tr]["id"] for i in sorted(hit_idx) if i >= n_tr]
        train_c = [c for i, c in enumerate(train_c) if i not in hit_idx]
        val_c = [c for i, c in enumerate(val_c) if (i + n_tr) not in hit_idx]
        report.update({k: v for k, v in scan.items() if k != "candidate_hits"})
        report["candidate_hits"] = {str(k): v for k, v in scan["candidate_hits"].items()}
        report["rejected_train_ids"] = rejected_train
        report["rejected_val_ids"] = rejected_val
        print(f"  scanned {scan['n_benchmark_contexts']} contexts in {scan['seconds']}s; "
              f"rejected {len(rejected_train)} train / {len(rejected_val)} val candidates", flush=True)

    if len(train_c) < args.num_train or len(val_c) < args.num_val:
        print(f"ERROR: not enough clean candidates (train {len(train_c)}/{args.num_train}, val {len(val_c)}/{args.num_val}); "
              f"raise --spare-fraction", file=sys.stderr)
        return 2
    train_rows, val_rows = train_c[:args.num_train], val_c[:args.num_val]
    train_path = out_dir / f"{args.name}_train.jsonl"
    val_path = out_dir / f"{args.name}_val.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)
    report_path = out_dir / f"{args.name}_leakage_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    manifest = {
        "name": args.name, "source": args.source, "source_revision": source_revision(args.source),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "seed": args.seed,
        "train_split": args.train_split, "val_split": args.val_split,
        "min_chars": args.min_chars, "excerpt_chars": args.excerpt_chars, "shuffle_buffer": args.shuffle_buffer,
        "files": {
            train_path.name: {"n": len(train_rows), "sha256": sha256_file(train_path),
                              "ids": [r["id"] for r in train_rows]},
            val_path.name: {"n": len(val_rows), "sha256": sha256_file(val_path),
                            "ids": [r["id"] for r in val_rows]},
        },
        "leakage_report": report_path.name,
        "leakage_report_sha256": sha256_file(report_path),
        "books_disjoint": sorted(set(r.get("url") or r["id"] for r in train_rows)).__len__() + len(val_rows)
        == len(set((r.get("url") or r["id"]) for r in train_rows + val_rows)),
    }
    (out_dir / f"{args.name}_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {train_path} ({len(train_rows)}), {val_path} ({len(val_rows)}), manifest + leakage report", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
