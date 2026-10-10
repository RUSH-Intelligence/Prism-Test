"""Training data: long-text JSONL rows -> ``[context | suffix]`` token windows (spec §12).

Rows need an ``id`` and either a ``text`` field (``kind: text``, the default — PG-19 / FineWeb-Edu excerpts
written by ``scripts/prepare_kv_recovery_data.py`` / ``scripts/prepare_kv_recovery_mix.py``) or the
benchmark-shaped fields ``context / question / answer_prefix / answer`` (``kind: qa`` — rows of RULER or
LongBench OUTSIDE the evaluated pool, written by ``prepare_kv_recovery_mix.py``). qa rows are shaped exactly
as the evaluation shapes them and aligned on the question (+ answer prefix) [+ gold answer] region
(``data.qa_region``); the evaluated rows themselves never enter a corpus (row-index disjointness is
enforced by the corpus builder and recorded in its manifest).

Formats
* ``raw``  : ids = [bos] + tokenize(text); the window is the first ``max_length`` ids,
             split at the token level into ``context`` (first ``T = max_length - suffix_length``)
             and ``suffix`` (next ``L``). Mirrors the pipeline's no-chat path (``bos + context``).
* ``chat`` : the context text goes through ``ResearchGenerationPipeline.preprocess`` (chat
             prefix + context, question = continuation + chat suffix), i.e. the exact eval
             shaping; token counts then differ slightly from ``max_length``.

Suffix modes
* ``continuation`` : the suffix is the text that follows the context (default).
* ``recall``       : the suffix is a verbatim span copied from an earlier offset of the same
                     window — a read-from-cache probe (pre-registered ablation).
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

from .config import DataCfg, RecoveryConfig
from .student import Example


@dataclass
class WindowStats:
    n_rows: int = 0
    n_used: int = 0
    n_skipped_short: int = 0
    n_skipped_empty: int = 0
    n_skipped_excluded: int = 0
    n_skipped_long: int = 0          # qa rows whose context exceeds data.max_context_tokens
    context_tokens: int = 0
    suffix_tokens: int = 0
    format: str = "raw"
    suffix_mode: str = "continuation"
    ids: List[str] = field(default_factory=list)
    by_source: Dict[str, int] = field(default_factory=dict)   # used windows per row ``source``
    by_kind: Dict[str, int] = field(default_factory=dict)     # used windows per row ``kind``

    def as_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if k != "ids"} | {"n_ids": len(self.ids)}


def read_jsonl(path: str | Path) -> List[dict]:
    rows: List[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row.setdefault("id", f"{Path(path).stem}-{i:06d}")
            rows.append(row)
    return rows


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def bos_id_for(tokenizer, model=None) -> Optional[int]:
    """``bos_token_id`` of the tokenizer, else the model's generation config, else None
    (Qwen3.5 has no BOS; the pipeline's no-chat path prepends ``bos_token or ""``)."""
    bos = getattr(tokenizer, "bos_token_id", None)
    if bos is None and model is not None:
        gen = getattr(model, "generation_config", None)
        bos = getattr(gen, "bos_token_id", None)
    return int(bos) if bos is not None else None


def encode_text(tokenizer, text: str, *, max_chars: Optional[int] = None) -> List[int]:
    if max_chars is not None:
        text = text[:max_chars]
    return list(tokenizer.encode(text, add_special_tokens=False))


def split_window(ids: List[int], max_length: int, suffix_length: int, *, suffix_mode: str,
                 rng: random.Random) -> Tuple[List[int], List[int]]:
    T = int(max_length) - int(suffix_length)
    L = int(suffix_length)
    if len(ids) < max_length:
        raise ValueError("window too short")
    ctx = ids[:T]
    if suffix_mode == "continuation":
        suffix = ids[T:T + L]
    elif suffix_mode == "recall":
        lo, hi = max(0, T // 8), max(0, T - L)   # skip the very first tokens (sinks)
        o = rng.randint(lo, hi) if hi > lo else 0
        suffix = ids[o:o + L]
    else:
        raise ValueError(f"unknown suffix_mode {suffix_mode!r}")
    return ctx, suffix


def _raw_example(row: dict, tokenizer, dcfg: DataCfg, *, bos_id: Optional[int], rng: random.Random) -> Optional[Example]:
    text = row.get("text") or ""
    if not text.strip():
        return None
    ids = encode_text(tokenizer, text, max_chars=int(dcfg.max_length) * 12)
    if bos_id is not None:
        ids = [bos_id] + ids
    if len(ids) < dcfg.max_length:
        return None
    ctx, suffix = split_window(ids, dcfg.max_length, dcfg.suffix_length, suffix_mode=dcfg.suffix_mode, rng=rng)
    return Example(id=str(row["id"]), ctx_ids=torch.tensor([ctx], dtype=torch.long),
                   suffix_ids=torch.tensor([suffix], dtype=torch.long),
                   meta={"format": "raw", "suffix_mode": dcfg.suffix_mode, "source": row.get("source"),
                         "title": row.get("title")})


def _chat_example(row: dict, tokenizer, dcfg: DataCfg, *, pipeline, rng: random.Random) -> Optional[Example]:
    """Context inside the user turn exactly as evaluation builds it; the continuation plays the
    role of the question (answer_prefix empty)."""
    text = row.get("text") or ""
    if not text.strip():
        return None
    ids = encode_text(tokenizer, text, max_chars=int(dcfg.max_length) * 12)
    if len(ids) < dcfg.max_length:
        return None
    ctx_ids, suf_ids = split_window(ids, dcfg.max_length, dcfg.suffix_length, suffix_mode=dcfg.suffix_mode, rng=rng)
    context_text = tokenizer.decode(ctx_ids, skip_special_tokens=True)
    suffix_text = tokenizer.decode(suf_ids, skip_special_tokens=True)
    enc = pipeline.preprocess(context_text, questions=[suffix_text], answer_prefix="",
                              max_context_length=int(1e10), use_chat_template=True,
                              strip_auto_system_block=bool(dcfg.strip_auto_system_block))
    ctx = enc["context_ids"]
    suffix = enc["questions_ids"][0]
    return Example(id=str(row["id"]), ctx_ids=ctx.to(torch.long), suffix_ids=suffix.to(torch.long),
                   meta={"format": "chat", "suffix_mode": dcfg.suffix_mode, "source": row.get("source"),
                         "title": row.get("title")})


def _qa_example(row: dict, tokenizer, dcfg: DataCfg, *, pipeline) -> Optional[Example]:
    """A ``kind: qa`` corpus row -> benchmark-shaped window (see :func:`benchmark_example`); ``None`` when the
    context exceeds ``data.max_context_tokens``."""
    if pipeline is None:
        raise ValueError("qa rows need the ResearchGenerationPipeline (pass pipeline=)")
    r = dict(row)
    r.setdefault("_task", r.get("task", "?"))
    r.setdefault("_row", r.get("row", 0))
    ex = benchmark_example(r, tokenizer, pipeline=pipeline, bench_name=str(r.get("source", r.get("benchmark", "qa"))),
                           region=dcfg.qa_region, use_chat_template=True,
                           strip_auto_system_block=bool(dcfg.strip_auto_system_block), example_id=str(row["id"]))
    if dcfg.max_context_tokens is not None and ex.context_len > int(dcfg.max_context_tokens):
        return None
    ex.meta["source"] = row.get("source")
    ex.meta["kind"] = "qa"
    return ex


def build_examples(rows: Iterable[dict], tokenizer, dcfg: DataCfg, *, n_examples: int, seed: int,
                   model=None, pipeline=None, exclude_ids: Optional[Iterable[str]] = None
                   ) -> Tuple[List[Example], WindowStats]:
    """The first ``n_examples`` usable windows of ``rows`` in seeded order (rows whose id is in
    ``exclude_ids`` are skipped — used to carve a calibration set disjoint from train / val). Text rows
    become ``[context | continuation]`` windows, qa rows benchmark-shaped ``[context | question (+ answer)]``."""
    rows = list(rows)
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    stats = WindowStats(n_rows=len(rows), format=dcfg.format, suffix_mode=dcfg.suffix_mode)
    bos_id = bos_id_for(tokenizer, model)
    excluded = set(str(i) for i in (exclude_ids or ()))
    examples: List[Example] = []
    for idx in order:
        if len(examples) >= n_examples:
            break
        row = rows[idx]
        if str(row.get("id", idx)) in excluded:
            stats.n_skipped_excluded += 1
            continue
        rng = random.Random(f"{seed}:{row.get('id', idx)}")
        kind = str(row.get("kind") or "text")
        if kind == "qa":
            ex = _qa_example(row, tokenizer, dcfg, pipeline=pipeline)
            if ex is None:
                stats.n_skipped_long += 1
                continue
        elif kind == "text" and dcfg.format == "raw":
            ex = _raw_example(row, tokenizer, dcfg, bos_id=bos_id, rng=rng)
        elif kind == "text" and dcfg.format == "chat":
            if pipeline is None:
                raise ValueError("data.format=chat needs the ResearchGenerationPipeline (pass pipeline=)")
            ex = _chat_example(row, tokenizer, dcfg, pipeline=pipeline, rng=rng)
        elif kind == "text":
            raise ValueError(f"unknown data.format {dcfg.format!r}")
        else:
            raise ValueError(f"row {row.get('id', idx)}: unknown kind {kind!r} (text | qa)")
        if ex is None:
            if (row.get("text") or "").strip():
                stats.n_skipped_short += 1
            else:
                stats.n_skipped_empty += 1
            continue
        ex.meta.setdefault("source", row.get("source"))
        ex.meta.setdefault("kind", kind)
        examples.append(ex)
        stats.n_used += 1
        stats.context_tokens += ex.context_len
        stats.suffix_tokens += ex.suffix_len
        stats.ids.append(ex.id)
        src = str(ex.meta.get("source") or "?")
        stats.by_source[src] = stats.by_source.get(src, 0) + 1
        stats.by_kind[kind] = stats.by_kind.get(kind, 0) + 1
    if len(examples) < n_examples:
        raise ValueError(f"only {len(examples)} usable windows of {n_examples} requested "
                         f"(rows={len(rows)}, too short={stats.n_skipped_short}, empty={stats.n_skipped_empty}, "
                         f"excluded={stats.n_skipped_excluded}, long={stats.n_skipped_long}); prepare more / longer rows, "
                         f"lower data.max_length or raise data.max_context_tokens"
                         + (" or lower trainable.sensitivity.num_examples / data.num_val_examples" if excluded else ""))
    return examples, stats


def load_split(cfg: RecoveryConfig, tokenizer, which: str, *, model=None, pipeline=None,
               exclude_ids: Optional[Iterable[str]] = None) -> Tuple[List[Example], WindowStats]:
    """``train`` / ``val`` windows, or the ``calibration`` windows of the sensitivity layer selection
    (``trainable.sensitivity``: ``num_examples`` rows of ``split`` under seed ``data.seed + 2``, skipping
    ``exclude_ids`` — pass the train and val window ids so the three sets are disjoint)."""
    d = cfg.data
    if which == "train":
        path, n, seed = d.path, d.num_train_examples, d.seed
    elif which == "val":
        if d.val_path is None:
            return [], WindowStats(format=d.format, suffix_mode=d.suffix_mode)
        path, n, seed = d.val_path, d.num_val_examples, d.seed + 1
    elif which == "calibration":
        s = cfg.trainable.sensitivity
        path = d.val_path if s.split == "val" else d.path
        if path is None:
            raise ValueError("trainable.sensitivity.split='val' needs data.val_path")
        n, seed = s.num_examples, d.seed + 2
    else:
        raise ValueError(which)
    if n == 0:
        return [], WindowStats(format=d.format, suffix_mode=d.suffix_mode)
    rows = read_jsonl(path)
    return build_examples(rows, tokenizer, d, n_examples=n, seed=seed, model=model, pipeline=pipeline,
                          exclude_ids=exclude_ids)


# ---------------------------------------------------------------------------
# Benchmark-context windows — ANALYSIS ONLY (never for training or layer selection)
# ---------------------------------------------------------------------------
def benchmark_rows(bench_name: str, subsets: Optional[List[str]], *, pool_rows: int = 100, rows_per_task: int = 2,
                   seed: int = 0, request_offset: int = 0) -> List[dict]:
    """A seeded sample of ``rows_per_task`` rows per task from the first ``pool_rows`` rows of each
    subset — the same pool the evaluation protocol scores (``max_requests`` rows from
    ``request_offset``). Rows are plain dicts with the benchmark's columns plus ``_task`` / ``_row``."""
    from eval_harness.benchmarks.registry import get_benchmark

    bench = get_benchmark(bench_name)
    df = bench.load(subsets)
    if "task" not in df.columns:
        raise ValueError(f"{bench_name}: rows carry no 'task' column")
    out: List[dict] = []
    for task in list(dict.fromkeys(df["task"].astype(str).tolist())):
        sub = df[df["task"].astype(str) == task].iloc[int(request_offset):int(request_offset) + int(pool_rows)]
        idx = list(range(len(sub)))
        rng = random.Random(f"{seed}:{bench_name}:{task}")
        pick = sorted(rng.sample(idx, min(int(rows_per_task), len(idx))))
        for i in pick:
            row = sub.iloc[i].to_dict()
            row["_task"], row["_row"] = task, int(i) + int(request_offset)
            out.append(row)
    return out


def benchmark_example(row: dict, tokenizer, *, pipeline, bench_name: str, region: str = "question_answer",
                      use_chat_template: bool = True, strip_auto_system_block: bool = True,
                      example_id: Optional[str] = None) -> Example:
    """One benchmark row -> ``[context | measured region]`` shaped EXACTLY as the evaluation
    shapes it (``ResearchGenerationPipeline.preprocess``: chat template, stripped auto system block,
    ``query_aware: false`` so the question never enters the context). The measured region is the
    question (+ chat suffix + the benchmark's answer prefix) — ``region='question'`` — optionally
    followed by the gold answer, teacher-forced (``'question_answer'``, the paper's question/answer
    region). Nothing here feeds training or selection; it exists for the layer-wise analysis."""
    from eval_harness.benchmarks.common import parse_answers

    if region not in ("question", "question_answer"):
        raise ValueError(f"region must be 'question' or 'question_answer', got {region!r}")
    enc = pipeline.preprocess(str(row["context"]), questions=[str(row["question"])],
                              answer_prefix=str(row.get("answer_prefix", "") or ""), max_context_length=int(1e10),
                              use_chat_template=use_chat_template, strip_auto_system_block=strip_auto_system_block)
    ctx = enc["context_ids"].to(torch.long)
    suffix = enc["questions_ids"][0].to(torch.long)
    n_question = int(suffix.shape[1])
    n_answer = 0
    if region == "question_answer":
        answers = parse_answers(row.get("answer", ""))
        text = ", ".join(a for a in answers if a) if len(answers) > 1 and str(row.get("_task", "")).startswith(("cwe", "niah_multi")) \
            else (answers[0] if answers else "")
        if text:
            ans = tokenizer.encode(" " + text, add_special_tokens=False)
            if ans:
                suffix = torch.cat([suffix, torch.tensor([ans], dtype=torch.long)], dim=1)
                n_answer = len(ans)
    task = str(row.get("_task", row.get("task", "?")))
    return Example(id=example_id or f"{bench_name}/{task}/row{int(row.get('_row', 0))}", ctx_ids=ctx, suffix_ids=suffix,
                   meta={"format": "benchmark", "benchmark": bench_name, "task": task, "row": int(row.get("_row", 0)),
                         "region": region, "n_question_tokens": n_question, "n_answer_tokens": n_answer,
                         "context_tokens": int(ctx.shape[1])})


def benchmark_examples(bench_name: str, subsets: Optional[List[str]], tokenizer, *, pipeline, rows_per_task: int = 2,
                       pool_rows: int = 100, seed: int = 0, region: str = "question_answer",
                       use_chat_template: bool = True, strip_auto_system_block: bool = True,
                       request_offset: int = 0) -> Tuple[List[Example], WindowStats]:
    rows = benchmark_rows(bench_name, subsets, pool_rows=pool_rows, rows_per_task=rows_per_task, seed=seed,
                          request_offset=request_offset)
    stats = WindowStats(n_rows=len(rows), format="benchmark", suffix_mode=region)
    examples: List[Example] = []
    for row in rows:
        ex = benchmark_example(row, tokenizer, pipeline=pipeline, bench_name=bench_name, region=region,
                               use_chat_template=use_chat_template, strip_auto_system_block=strip_auto_system_block)
        examples.append(ex)
        stats.n_used += 1
        stats.context_tokens += ex.context_len
        stats.suffix_tokens += ex.suffix_len
        stats.ids.append(ex.id)
    return examples, stats


def _context_key(e: Example) -> str:
    return hashlib.sha256(e.ctx_ids[0].to(torch.int64).numpy().tobytes()).hexdigest()


def assert_disjoint(train: List[Example], val: List[Example]) -> None:
    """Validation windows must come from different rows AND different text than training (ids and the full
    context token sequence; benchmark rows of one task legitimately share their instruction prefix)."""
    tid = {e.id for e in train}
    overlap = [e.id for e in val if e.id in tid]
    if overlap:
        raise AssertionError(f"validation ids also in training: {overlap[:5]}")
    keys = {_context_key(e) for e in train}
    dup = [e.id for e in val if _context_key(e) in keys]
    if dup:
        raise AssertionError(f"validation windows repeat a training context: {dup[:5]}")


def describe_split(examples: List[Example], stats: WindowStats, path: Optional[str]) -> Dict[str, Any]:
    return {
        "path": path,
        "sha256": sha256_file(path) if path and Path(path).exists() else None,
        "n_examples": len(examples),
        "ids": [e.id for e in examples],
        **stats.as_dict(),
    }
