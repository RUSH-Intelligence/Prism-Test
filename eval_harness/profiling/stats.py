"""Pure-stdlib statistics for the performance benchmark.

No torch, no numpy — so these are trivially testable and cannot drag CUDA into
an import.  ``percentile`` uses numpy's ``linear`` (interpolated closest ranks)
convention, spelled out here so a future reader never has to guess which of the
nine definitions is in play.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class LatencySummary:
    n: int
    mean: float
    median: float
    p50: float
    p90: float
    p95: float
    p99: float
    std: float
    cv: float
    min: float
    max: float

    def to_dict(self) -> dict:
        return asdict(self)


def percentile(xs: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (numpy 'linear' convention). ``q`` in [0, 100]."""
    if not xs:
        raise ValueError("percentile of an empty sequence")
    if not 0.0 <= q <= 100.0:
        raise ValueError(f"q must be in [0, 100], got {q}")
    ordered = sorted(xs)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (q / 100.0) * (len(ordered) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(ordered[lo])
    return float(ordered[lo] + (pos - lo) * (ordered[hi] - ordered[lo]))


def summarize(xs: Sequence[float]) -> Optional[LatencySummary]:
    """Summarize latencies. Returns ``None`` for an empty input (never 0.0)."""
    if not xs:
        return None
    vals = [float(x) for x in xs]
    mean = statistics.fmean(vals)
    # Sample stdev (statistics.stdev), not population: switching these silently
    # changes every reported error bar.
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return LatencySummary(
        n=len(vals),
        mean=mean,
        median=statistics.median(vals),
        p50=percentile(vals, 50),
        p90=percentile(vals, 90),
        p95=percentile(vals, 95),
        p99=percentile(vals, 99),
        std=std,
        cv=(std / mean) if mean else 0.0,
        min=min(vals),
        max=max(vals),
    )


def decode_throughput_total(step_ms: Sequence[float]) -> Optional[float]:
    """Total tokens / total time (tok/s) — what a user actually experiences.

    Algebraically ``1000 / mean(step_ms)``.  This is the headline number: unlike
    ``1000 / p50`` it does not discard the tail, which is exactly where allocator
    stalls and clock throttling live.
    """
    if not step_ms:
        return None
    total = sum(step_ms)
    return (1000.0 * len(step_ms) / total) if total > 0 else None


def decode_throughput_median(step_ms: Sequence[float]) -> Optional[float]:
    """``1000 / p50`` — robust to a single stall, but hides the tail.

    Reported alongside :func:`decode_throughput_total`, never instead of it.
    """
    if not step_ms:
        return None
    p50 = percentile(step_ms, 50)
    return (1000.0 / p50) if p50 > 0 else None


def prefill_throughput(n_tokens: int, prefill_ms: float) -> Optional[float]:
    """Context tokens per second. Excludes the question block by construction."""
    if prefill_ms is None or prefill_ms <= 0:
        return None
    return n_tokens / (prefill_ms / 1000.0)


def speedup(anchor_ms: Optional[float], cell_ms: Optional[float]) -> Optional[float]:
    """anchor / cell. ``None`` when either side is missing — NEVER 1.0.

    Rendering a missing anchor as 1.00x would silently assert 'no difference'
    for a comparison that was never made.
    """
    if anchor_ms is None or cell_ms is None or cell_ms <= 0:
        return None
    return anchor_ms / cell_ms


def linfit(xs: Sequence[float], ys: Sequence[float]) -> Optional[tuple]:
    """Ordinary least squares. Returns ``(slope, intercept)`` or ``None``."""
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return slope, my - slope * mx
