"""Training data: long-text JSONL rows -> ``[context | suffix]`` token windows (spec §12).

Rows need a ``text`` field (and an ``id``); ``scripts/prepare_kv_recovery_data.py`` writes
them from PG-19. Nothing here touches benchmark data or labels.

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
    context_tokens: int = 0
    suffix_tokens: int = 0
    format: str = "raw"
    suffix_mode: str = "continuation"
    ids: List[str] = field(default_factory=list)

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


def build_examples(rows: Iterable[dict], tokenizer, dcfg: DataCfg, *, n_examples: int, seed: int,
                   model=None, pipeline=None, exclude_ids: Optional[Iterable[str]] = None
                   ) -> Tuple[List[Example], WindowStats]:
    """The first ``n_examples`` usable windows of ``rows`` in seeded order (rows whose id is in
    ``exclude_ids`` are skipped — used to carve a calibration set disjoint from train / val)."""
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
        if dcfg.format == "raw":
            ex = _raw_example(row, tokenizer, dcfg, bos_id=bos_id, rng=rng)
        elif dcfg.format == "chat":
            if pipeline is None:
                raise ValueError("data.format=chat needs the ResearchGenerationPipeline (pass pipeline=)")
            ex = _chat_example(row, tokenizer, dcfg, pipeline=pipeline, rng=rng)
        else:
            raise ValueError(f"unknown data.format {dcfg.format!r}")
        if ex is None:
            if (row.get("text") or "").strip():
                stats.n_skipped_short += 1
            else:
                stats.n_skipped_empty += 1
            continue
        examples.append(ex)
        stats.n_used += 1
        stats.context_tokens += ex.context_len
        stats.suffix_tokens += ex.suffix_len
        stats.ids.append(ex.id)
    if len(examples) < n_examples:
        raise ValueError(f"only {len(examples)} usable windows of {n_examples} requested "
                         f"(rows={len(rows)}, too short={stats.n_skipped_short}, empty={stats.n_skipped_empty}, "
                         f"excluded={stats.n_skipped_excluded}); prepare more / longer rows, lower data.max_length"
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


def assert_disjoint(train: List[Example], val: List[Example]) -> None:
    """Validation windows must come from different rows AND different text than training."""
    tid = {e.id for e in train}
    overlap = [e.id for e in val if e.id in tid]
    if overlap:
        raise AssertionError(f"validation ids also in training: {overlap[:5]}")
    heads = {tuple(e.ctx_ids[0, :64].tolist()) for e in train}
    dup = [e.id for e in val if tuple(e.ctx_ids[0, :64].tolist()) in heads]
    if dup:
        raise AssertionError(f"validation windows share their first 64 tokens with training windows: {dup[:5]}")


def describe_split(examples: List[Example], stats: WindowStats, path: Optional[str]) -> Dict[str, Any]:
    return {
        "path": path,
        "sha256": sha256_file(path) if path and Path(path).exists() else None,
        "n_examples": len(examples),
        "ids": [e.id for e in examples],
        **stats.as_dict(),
    }
