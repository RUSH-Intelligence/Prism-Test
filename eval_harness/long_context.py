from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Sequence


@dataclass
class LongContextCompressionConfig:
    enabled: bool = False
    max_context_len: Optional[int] = None
    sink_tokens: int = 32
    local_tokens: int = 4096
    top_k_tokens: Optional[int] = None
    span_tokens: int = 0


@dataclass
class CompressionResult:
    token_ids: List[int]
    was_compressed: bool
    original_length: int
    compressed_length: int
    kept_indices: List[int]


@dataclass
class CompressionBudget:
    sink_indices: List[int]
    local_indices: List[int]
    candidate_indices: List[int]
    top_k_budget: int


def _pick_topk_by_attention_proxy(
    token_ids: Sequence[int],
    candidate_indices: Sequence[int],
    query_indices: Sequence[int],
    top_k: int,
) -> List[int]:
    if top_k <= 0 or not candidate_indices:
        return []

    # Lightweight proxy: token overlap with recent query region + recency prior.
    query_freq = Counter(token_ids[idx] for idx in query_indices)
    max_idx = max(len(token_ids) - 1, 1)

    scored = []
    for idx in candidate_indices:
        token_id = token_ids[idx]
        semantic_score = float(query_freq.get(token_id, 0))
        recency_score = float(idx) / float(max_idx)
        score = semantic_score + (0.01 * recency_score)
        scored.append((score, idx))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    chosen = [idx for _, idx in scored[:top_k]]
    chosen.sort()
    return chosen


def _compute_budget(
    token_ids: Sequence[int],
    cfg: LongContextCompressionConfig,
) -> CompressionBudget:
    original_len = len(token_ids)
    max_context_len = max(int(cfg.max_context_len or 0), 0)

    sink_keep = min(max(int(cfg.sink_tokens), 0), original_len, max_context_len)
    remaining_budget = max_context_len - sink_keep

    local_keep = min(max(int(cfg.local_tokens), 0), max(original_len - sink_keep, 0), remaining_budget)
    remaining_budget -= local_keep

    sink_indices = list(range(sink_keep))
    local_start = original_len - local_keep
    local_indices = list(range(local_start, original_len))

    candidate_start = sink_keep
    candidate_end = local_start
    candidate_indices = list(range(candidate_start, max(candidate_end, candidate_start)))

    if cfg.top_k_tokens is None:
        top_k_budget = remaining_budget
    else:
        top_k_budget = min(max(int(cfg.top_k_tokens), 0), remaining_budget)

    return CompressionBudget(
        sink_indices=sink_indices,
        local_indices=local_indices,
        candidate_indices=candidate_indices,
        top_k_budget=top_k_budget,
    )


def expand_indices_with_span(
    indices: Sequence[int],
    span_tokens: int,
    start: int,
    end: int,
) -> List[int]:
    if not indices:
        return []
    if span_tokens <= 0:
        return sorted(set(indices))

    half = max(span_tokens // 2, 0)
    expanded = set()
    for idx in indices:
        for pos in range(idx - half, idx + half + 1):
            if start <= pos < end:
                expanded.add(pos)
    return sorted(expanded)


def select_topk_indices_from_scores(
    candidate_indices: Sequence[int],
    scores: Sequence[float],
    top_k: int,
) -> List[int]:
    if top_k <= 0 or not candidate_indices:
        return []
    if len(candidate_indices) != len(scores):
        raise ValueError(
            "candidate_indices and scores must have the same length, "
            f"got {len(candidate_indices)} and {len(scores)}"
        )

    ranked = sorted(zip(scores, candidate_indices), key=lambda x: (x[0], x[1]), reverse=True)
    chosen = [idx for _, idx in ranked[:top_k]]
    chosen.sort()
    return chosen


def merge_budgeted_indices(
    token_count: int,
    budget: CompressionBudget,
    topk_indices: Sequence[int],
    span_tokens: int,
) -> List[int]:
    expanded_topk = expand_indices_with_span(
        topk_indices,
        span_tokens=span_tokens,
        start=min(budget.candidate_indices) if budget.candidate_indices else 0,
        end=max(budget.candidate_indices) + 1 if budget.candidate_indices else token_count,
    )
    keep_mask = set(budget.sink_indices)
    keep_mask.update(expanded_topk)
    keep_mask.update(budget.local_indices)
    kept_indices = sorted(i for i in keep_mask if 0 <= i < token_count)

    max_context_len = len(budget.sink_indices) + len(budget.local_indices) + budget.top_k_budget
    if max_context_len <= 0:
        return []
    if len(kept_indices) <= max_context_len:
        return kept_indices

    keep_list = list(kept_indices)
    while len(keep_list) > max_context_len:
        removed = False
        for idx in reversed(keep_list):
            if idx in budget.sink_indices or idx in budget.local_indices:
                continue
            keep_list.remove(idx)
            removed = True
            break
        if not removed:
            keep_list = keep_list[:max_context_len]
            break
    keep_list.sort()
    return keep_list


def compress_token_ids(
    token_ids: Sequence[int],
    cfg: LongContextCompressionConfig,
) -> CompressionResult:
    original_len = len(token_ids)
    if not cfg.enabled or cfg.max_context_len is None or original_len <= cfg.max_context_len:
        return CompressionResult(
            token_ids=list(token_ids),
            was_compressed=False,
            original_length=original_len,
            compressed_length=original_len,
            kept_indices=list(range(original_len)),
        )

    max_context_len = max(int(cfg.max_context_len), 0)
    if max_context_len == 0:
        return CompressionResult(
            token_ids=[],
            was_compressed=True,
            original_length=original_len,
            compressed_length=0,
            kept_indices=[],
        )

    budget = _compute_budget(token_ids, cfg)

    query_start = max(0, original_len - max(int(cfg.local_tokens), 1))
    query_indices = list(range(query_start, original_len))
    topk_indices = _pick_topk_by_attention_proxy(
        token_ids=token_ids,
        candidate_indices=budget.candidate_indices,
        query_indices=query_indices,
        top_k=budget.top_k_budget,
    )
    kept_indices = merge_budgeted_indices(
        token_count=original_len,
        budget=budget,
        topk_indices=topk_indices,
        span_tokens=cfg.span_tokens,
    )
    compressed_ids = [token_ids[idx] for idx in kept_indices]

    return CompressionResult(
        token_ids=compressed_ids,
        was_compressed=True,
        original_length=original_len,
        compressed_length=len(compressed_ids),
        kept_indices=kept_indices,
    )