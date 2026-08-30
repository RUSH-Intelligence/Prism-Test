"""Self-audit: the checks that make a timing number trustworthy.

A wrong accuracy score looks wrong.  A wrong tok/s looks fine.  So every cell is
gated before it is allowed into a table.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

H200_PEAK_BW_GBPS = 4800.0     # HBM3e, GB/s
H200_PEAK_BF16_TFLOPS = 989.0


def expected_budget(n_tokens: int, ratio: float) -> int:
    """``int(T * (1 - r))`` -- the exact rule in ScorerKVCompressor.compress:527.

    Note the float: 1 - 0.9 == 0.09999999999999998, so 8192 -> 819, not 820.
    """
    return int(n_tokens * (1.0 - ratio))


def roofline_step_ms(weight_bytes: int, kv_bytes_per_token: int, seq_len: int,
                     bandwidth_gbps: float = H200_PEAK_BW_GBPS,
                     dyncache: bool = True) -> float:
    """Lower bound on decode step time from memory traffic.

    ``dyncache=True`` models THIS harness: transformers' ``DynamicCache`` grows by
    ``torch.cat`` every step (cache_utils.py:143-144), so per step the cache is
    read for the concat, written for the concat, and read again by attention --
    about 3x the naive traffic.  ``dyncache=False`` is the ideal paged engine.
    """
    factor = (3 * seq_len + 1) if dyncache else seq_len
    total = weight_bytes + kv_bytes_per_token * factor
    return 1000.0 * total / (bandwidth_gbps * 1e9)


def achieved_bandwidth_gbps(weight_bytes: int, kv_bytes_per_token: int, seq_len: int,
                            step_ms: float, dyncache: bool = True) -> Optional[float]:
    if not step_ms or step_ms <= 0:
        return None
    factor = (3 * seq_len + 1) if dyncache else seq_len
    total = weight_bytes + kv_bytes_per_token * factor
    return total / (step_ms / 1000.0) / 1e9


def audit_cell(cell, payload: Dict[str, Any], *, jitter_tol: float = 1.15,
               budget_rule: str = "strict") -> Dict[str, List[str]]:
    """Return ``{"problems": [...], "warnings": [...]}``. Problems disqualify a cell."""
    problems: List[str] = []
    warnings: List[str] = []
    kv = payload.get("kv_cache") or {}
    dec = payload.get("decode") or {}
    got = kv.get("seq_len_max", 0)
    ctx = (payload.get("prefill") or {}).get("tokens", cell.context_tokens)

    # 1. Budget: the cache must have shrunk to exactly int(T*(1-r)).
    if cell.is_anchor:
        if got != ctx:
            problems.append(
                f"anchor cache was evicted: seq_len {got} != context {ctx}; "
                "the full-KV reference is not full")
    else:
        exp = expected_budget(ctx, cell.compression_ratio)
        if budget_rule == "strict" and got != exp:
            problems.append(f"budget: cache seq_len {got} != expected {exp} "
                            f"(int({ctx}*(1-{cell.compression_ratio})))")
        elif budget_rule == "ragged_mean":
            per = kv.get("per_layer_seq_len") or []
            mean = sum(per) / len(per) if per else 0
            if abs(mean - exp) > max(2.0, 0.05 * exp):
                problems.append(f"ragged budget: mean per-layer {mean:.1f} != expected {exp}")

        # 2. Masking-press detector. This is the silent-wrongness case: presses
        #    that keep the cache full-length record pruned indices instead of
        #    evicting, so they pay FULL KV traffic while looking compressed.
        if got == ctx and cell.compression_ratio > 0:
            problems.append(
                f"masking-based press: cache did not shrink ({got} == {ctx}). There is no "
                "memory or bandwidth benefit; this method must not appear in a throughput table")

    # 3. Ragged cache under a strict rule -> a scalar cache length is a lie.
    if kv.get("ragged") and budget_rule == "strict":
        warnings.append(f"ragged cache (min {kv.get('seq_len_min')}, max {kv.get('seq_len_max')}); "
                        "report the distribution, not a scalar")

    # 4. Step count must be identical across cells or tok/s denominators differ.
    # N forwards yield N-1 token-to-token intervals (the last has no successor).
    expected_samples = (cell.decode_steps - 1) * cell.repeats
    if dec.get("n_samples") != expected_samples:
        problems.append(f"decode produced {dec.get('n_samples')} samples, expected "
                        f"{expected_samples} (({cell.decode_steps}-1) x {cell.repeats})")

    # 5. Jitter, measured on p90 rather than p99. Each repeat's FIRST decode step
    #    pays allocator growth, so ~1% of samples are structural outliers and
    #    p99/median flags every healthy cell. p90/median tests what we actually
    #    care about: is the bulk of the distribution tight?
    per_step = dec.get("per_step") or {}
    if per_step.get("median") and per_step.get("p90"):
        ratio = per_step["p90"] / per_step["median"]
        if ratio > jitter_tol:
            warnings.append(f"jitter p90/median = {ratio:.2f} > {jitter_tol}; noisy node")
    if per_step.get("cv", 0) > 0.10:
        problems.append(f"step-latency CV {per_step['cv']:.1%} > 10%: not a steady-state measurement")

    # 6. Allocator stalls land directly in p99.
    mem = payload.get("memory") or {}
    if mem.get("num_alloc_retries"):
        problems.append(f"num_alloc_retries={mem['num_alloc_retries']}: the allocator did a "
                        "cudaFree+retry, a multi-ms stall inside the measurement")
    return {"problems": problems, "warnings": warnings}


def audit_group(cells_and_payloads) -> List[str]:
    """Cross-cell checks: anchor pairing and environment homogeneity."""
    problems: List[str] = []
    anchors = {c.anchor_key for c, _ in cells_and_payloads if c.is_anchor}
    for c, _ in cells_and_payloads:
        if not c.is_anchor and c.anchor_key not in anchors:
            problems.append(f"no full-KV anchor for {c.anchor_key}: {c.cell_id} has no denominator")
    for field in ("gpu_name", "git_sha", "torch_version"):
        vals = {(p.get("environment") or {}).get(field) for _, p in cells_and_payloads}
        vals.discard(None)
        if len(vals) > 1:
            problems.append(f"cells span multiple {field} values {sorted(vals)} -- "
                            "they are not mutually comparable")
    return problems
