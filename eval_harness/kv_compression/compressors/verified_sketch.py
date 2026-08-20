import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import nn

from eval_harness.kv_compression.base import KVCompressor
from eval_harness.kv_compression.registry import get_kv_compressor, register_kv_compressor

logger = logging.getLogger(__name__)


@register_kv_compressor("verified")
@dataclass
class VerifiedSketch(KVCompressor):
    """
    Verified KV compression (v1) — deterministic head + uniform random tail.

    Motivation and full methodology: ``notes/verified_kv_compression_idea.md``,
    inspired by vAttention: Verified Sparse Attention (arXiv:2510.05688). The
    long-term goal is a *provable* per-prompt error bound on the compressed
    cache (Horvitz-Thompson reweighted random "ambassadors" + a Hoeffding /
    Bernstein certificate on the evicted-set contribution). **This class is only
    the v1 skeleton toward that** and deliberately ships none of the bound
    machinery yet.

    What v1 does
    ------------
    Same total budget as any top-k compressor. Given ``compression_ratio`` the
    cache keeps ``M = int(T * (1 - compression_ratio))`` tokens. That budget is
    split by a single knob ``det_fraction`` (default 0.75):

    - **Deterministic head (``det_fraction`` of M).** Run the wrapped ``inner``
      compressor (Ridge by default, but *any* registered key-preserving
      compressor) at a reduced budget so it selects its own top ``≈ det_fraction·M``
      tokens.
    - **Random tail (the remaining slots).** Uniformly sample the rest of the
      budget from the tokens the inner compressor did **not** keep (the evicted
      pool), per (batch, kv-head), so the two sets are disjoint.

    Kept = deterministic head ∪ random tail, sorted back into temporal order.
    Total kept == M, constant across heads (rectangular cache), so decode is
    completely unchanged — this is why v1 is cheap: it only changes *which*
    tokens fill the slots.

    ``det_fraction`` is a clean dial: ``1.0`` degenerates to the plain inner
    compressor; ``0.0`` degenerates to pure uniform-random eviction (a useful
    sanity baseline).

    Deferred to v2 (NOT in this class)
    ----------------------------------
    - No Horvitz-Thompson scale factors on the random tail — the ambassadors sit
      in the cache as ordinary tokens, so the decode output is *not* yet an
      unbiased estimator of full attention.
    - No (ε, δ) certificate / adaptive sample count. ``det_fraction`` is a fixed
      ratio, not derived from a target error.

    How the deterministic indices are recovered
    -------------------------------------------
    ``KVCompressor.compress`` returns gathered ``(keys, values)`` — not indices —
    so to identify the evicted pool we recover the inner's kept positions by a
    random-projection fingerprint (``f = <k_i, w>`` for a fixed random ``w``) and
    an exact-match ``searchsorted`` back into the full key tensor. This is O(T log T)
    and needs no changes to any existing compressor. It assumes the inner
    **gathers keys unmodified** (true for Ridge / SnapKV / Compactor / Knorm /
    KeyDiff / …); value-reweighting coreset methods (``balancekv``) are therefore
    unsupported as the inner.

    Constraints
    -----------
    - ``post_prefill`` schedule (default): fires once on the full prompt cache.
      Do not use ``streaming`` (the inner would fire per chunk → geometric
      over-eviction, and pool bookkeeping assumes the full sequence).
    - Keep ``attention_method: none`` — like most position-sensitive scorers this
      assumes vanilla absolute-position rotated keys, not DCA's cyclic positions.
    - ``inner`` must expose a settable ``compression_ratio`` (Ridge and every
      ``ScorerKVCompressor`` do).

    Parameters
    ----------
    inner : str, default="ridge"
        Registry name of the wrapped compressor supplying the deterministic head.
    inner_kwargs : dict, default={}
        Constructor kwargs forwarded to the inner compressor.
    det_fraction : float, default=0.75
        Fraction of the kept budget filled by the inner compressor; the rest is
        uniform random. Must be in [0, 1].
    min_tokens_to_compress : int, default=0
        Skip compression (return the cache unchanged) when ``T`` is below this.
    sample_seed : int | None, default=None
        If set, the random tail is drawn from a seeded generator for
        reproducibility; otherwise the global torch RNG is used (seed externally).
    """

    compression_ratio: float = 0.0
    inner: str = "ridge"
    inner_kwargs: dict = field(default_factory=dict)
    det_fraction: float = 0.75
    min_tokens_to_compress: int = 0
    sample_seed: Optional[int] = None
    # v2 (Step 1): MEASURE-ONLY output-error of the chosen keep-set. Purely
    # observational — logs the per-layer "leftover" and never changes what is
    # kept. Enable to collect the signal the adaptive split will later consume.
    measure_coverage: bool = False
    coverage_n_probe: int = 8

    def __post_init__(self):
        super().__post_init__()
        assert 0.0 <= self.compression_ratio < 1.0, "compression_ratio must be in [0, 1)"
        assert 0.0 <= self.det_fraction <= 1.0, "det_fraction must be in [0, 1]"
        assert self.min_tokens_to_compress >= 0, "min_tokens_to_compress must be >= 0"

        # Accumulates per-layer coverage readings across the whole run (one per
        # compress() call when measure_coverage is on); drained to coverage.json
        # by the runner. Kept tiny (a few floats per compressed layer per prompt).
        self._coverage_records: list = []
        # Set by the runner (begin_prompt_group) before each context group so
        # every reading is tagged with the df rows it belongs to — this is what
        # lets us line coverage up against per-question correctness. None when
        # the runner hasn't stamped one (e.g. direct unit-test calls).
        self._current_group: Optional[tuple] = None

        # Resolve the wrapped compressor once (auto-discovery happens on lookup).
        self._inner: KVCompressor = get_kv_compressor(self.inner, **dict(self.inner_kwargs))
        if not isinstance(self._inner, KVCompressor):
            raise TypeError(f"inner '{self.inner}' did not resolve to a KVCompressor")
        if not hasattr(self._inner, "compression_ratio"):
            raise TypeError(
                f"inner '{self.inner}' has no settable compression_ratio; "
                "VerifiedSketch needs it to size the deterministic head."
            )

    def post_init_from_model(self, model) -> None:
        # Forward so inners that download / build model-specific artifacts
        # (qfilter, kvzap, expected_attention_stats, ...) initialise.
        self._inner.post_init_from_model(model)

    def begin_prompt_group(self, row_indices) -> None:
        """Stamp the df rows whose context is about to be prefilled.

        Called by the runner before each context group. Compression fires once
        per context (shared by all questions in the group), so every coverage
        reading until the next call belongs to exactly these rows.
        """
        self._current_group = tuple(int(i) for i in row_indices)

    @staticmethod
    def _recover_indices(full_keys: torch.Tensor, kept_keys: torch.Tensor) -> torch.Tensor:
        """Map each kept key row back to its position in ``full_keys``.

        full_keys: [B, H, S, D]; kept_keys: [B, H, n, D] of exact row copies
        (the inner gathers keys unmodified). Returns [B, H, n] long indices.

        Uses a random-projection fingerprint + exact-match searchsorted so this
        stays O(S log S) instead of the O(S·n) broadcast-equality used in tests.
        """
        B, H, S, D = full_keys.shape
        n = kept_keys.shape[2]
        if n == 0:
            return torch.zeros(B, H, 0, dtype=torch.long, device=full_keys.device)

        w = torch.randn(D, device=full_keys.device, dtype=torch.float32)
        f_full = (full_keys.float() * w).sum(-1)  # [B, H, S]
        f_kept = (kept_keys.float() * w).sum(-1)  # [B, H, n]

        order = f_full.argsort(dim=-1)                     # ascending positions
        f_sorted = f_full.gather(-1, order)                # sorted fingerprints
        pos = torch.searchsorted(f_sorted, f_kept)         # left match (exact)
        pos = pos.clamp_(max=S - 1)
        return order.gather(-1, pos)                       # -> original positions

    def _log_coverage(self, module, hidden_states, keys, values, keep_idx, kwargs):
        """MEASURE-ONLY: log the output-error leftover of ``keep_idx``.

        Wrapped so a measurement failure can never break a run — this path is
        observational and must stay strictly side-effect-free w.r.t. the cache.
        """
        if not self.measure_coverage:
            return
        try:
            from eval_harness.kv_compression.compressors.verified_coverage import (
                measure_output_error,
            )

            report = measure_output_error(
                module, hidden_states, keys, values, keep_idx, kwargs,
                n_probe=self.coverage_n_probe,
            )
            if report is None:
                return
            layer_idx = getattr(module, "layer_idx", "?")
            self._coverage_records.append({
                "layer": layer_idx,
                "group": self._current_group,
                "worst": report.worst,
                "mean": report.mean,
                "n_keep": report.n_keep,
                "n_total": report.n_total,
            })
            logger.info(
                "[verified/coverage] layer=%s keep=%d/%d n_probe=%d "
                "worst_rel_err=%.4f mean_rel_err=%.4f",
                layer_idx, report.n_keep, report.n_total, report.n_probe,
                report.worst, report.mean,
            )
        except Exception as exc:  # never let measurement break a run
            logger.warning("[verified/coverage] measurement skipped: %s", exc)

    def drain_coverage(self) -> Optional[dict]:
        """Aggregate accumulated coverage readings into a compact summary.

        Returns ``None`` when nothing was measured (measure_coverage off, or no
        compression fired). The runner writes this to ``coverage.json`` next to
        ``metrics.json``. Aggregates over EVERY compress() call in the run —
        i.e. all prompts and (if a run bundles several) all subsets mixed. For
        per-subset coverage, run subsets as separate cells so each gets its own
        coverage.json.
        """
        records = self._coverage_records
        if not records:
            return None

        def _mean(xs):
            xs = list(xs)
            return sum(xs) / len(xs) if xs else 0.0

        per_layer: dict = {}
        for r in records:
            per_layer.setdefault(r["layer"], []).append(r)
        per_layer_summary = {
            str(layer): {
                "worst_rel_err_mean": _mean(x["worst"] for x in rs),
                "mean_rel_err_mean": _mean(x["mean"] for x in rs),
                "n": len(rs),
            }
            for layer, rs in sorted(per_layer.items(), key=lambda kv: str(kv[0]))
        }
        # Per-prompt: group readings by the df rows they were stamped with, then
        # average across layers → one coverage number per question. Every row in
        # a shared-context group gets that group's number (they share the cache).
        # This is the column to line up against per-question correctness.
        per_group: dict = {}
        for r in records:
            if r.get("group"):
                per_group.setdefault(r["group"], []).append(r)
        per_prompt: dict = {}
        for group_rows, rs in per_group.items():
            worst = _mean(x["worst"] for x in rs)
            mean = _mean(x["mean"] for x in rs)
            for row_idx in group_rows:
                per_prompt[str(row_idx)] = {
                    "worst_rel_err": worst,
                    "mean_rel_err": mean,
                    "n_layers": len(rs),
                }

        summary = {
            "method": "verified",
            "inner": self.inner,
            "det_fraction": self.det_fraction,
            "n_records": len(records),
            "n_layers": len(per_layer),
            # Run-level headline: pair THIS with the benchmark score.
            "worst_rel_err_mean": _mean(r["worst"] for r in records),
            "mean_rel_err_mean": _mean(r["mean"] for r in records),
            "per_layer": per_layer_summary,
        }
        if per_prompt:
            # Keyed by df row index → joins directly onto predictions.csv.
            summary["per_prompt"] = per_prompt
        return summary

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ):
        if self.compression_ratio == 0:
            return keys, values

        B, H, T, D = keys.shape
        if T < self.min_tokens_to_compress:
            return keys, values

        M = int(T * (1.0 - self.compression_ratio))
        M = max(0, min(M, T))
        if M <= 0 or M >= T:
            return keys, values

        # 1. Deterministic head: run the inner at the reduced budget. Its exact
        #    kept count self-corrects the split (we read it back below), so an
        #    off-by-one in the derived ratio is harmless.
        inner_ratio = 1.0 - self.det_fraction * (1.0 - self.compression_ratio)
        inner_ratio = float(min(max(inner_ratio, 0.0), 1.0 - 1e-6))
        saved_ratio = self._inner.compression_ratio
        try:
            self._inner.compression_ratio = inner_ratio
            det_keys, det_values = self._inner.compress(
                module, hidden_states, keys, values, attentions, kwargs
            )
        finally:
            self._inner.compression_ratio = saved_ratio

        n_det = det_keys.shape[2]
        n_rand = max(0, min(M - n_det, T - n_det))

        # 2. Random tail: sample n_rand tokens uniformly from the evicted pool
        #    (positions the inner did NOT keep), per (batch, kv-head).
        det_idx = self._recover_indices(keys, det_keys)  # [B, H, n_det]

        if n_rand == 0:
            # Inner already fills (or overfills) the budget — nothing to sample.
            self._log_coverage(module, hidden_states, keys, values, det_idx, kwargs)
            return det_keys.contiguous(), det_values.contiguous()
        avail = torch.ones(B, H, T, dtype=torch.bool, device=keys.device)
        avail.scatter_(-1, det_idx, False)

        if self.sample_seed is not None:
            gen = torch.Generator(device=keys.device).manual_seed(int(self.sample_seed))
            noise = torch.rand(B, H, T, generator=gen, device=keys.device)
        else:
            noise = torch.rand(B, H, T, device=keys.device)
        noise = noise.masked_fill(~avail, -1.0)
        rand_idx = noise.topk(n_rand, dim=-1).indices  # [B, H, n_rand]

        # 3. Union, restore temporal order, gather from the ORIGINAL cache.
        keep_idx = torch.cat([det_idx, rand_idx], dim=-1).sort(dim=-1).values
        self._log_coverage(module, hidden_states, keys, values, keep_idx, kwargs)
        gather_idx = keep_idx.unsqueeze(-1).expand(-1, -1, -1, D)
        out_keys = keys.gather(2, gather_idx).contiguous()
        out_values = values.gather(2, gather_idx).contiguous()
        return out_keys, out_values
