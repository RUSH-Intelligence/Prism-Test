import logging
from dataclasses import dataclass
from typing import Literal, Optional

import torch
from torch import nn

from eval_harness.kv_compression.base import KVCompressor
from eval_harness.kv_compression.registry import register_kv_compressor

logger = logging.getLogger(__name__)


@register_kv_compressor("ridge")
@dataclass
class RidgeSketch(KVCompressor):
    """
    Value-aware query-ridge KV compression.

    Port of ``RidgePress`` (kvpress/presses/ridge_press.py; a research-fork
    addition in the local kvpress 0.5.1 checkout, not upstream NVIDIA kvpress).

    Main scores:
      tau_i   = key-space ridge leverage score
      omega_i = query-key interaction ||Q k_i||_2
      ||v_i|| = value norm

    Envelope modes:
      envelope:
          score_i = max(p_ridge_i, p_query_i)

      fixed_envelope:
          score_i = max(p_ridge_i, envelope_gamma * p_query_i), envelope_gamma >= 0

      weighted_envelope:
          score_i = max(p_ridge_i, gamma * p_query_i)
          gamma = 1 + query_boost_strength * (1 - topk_overlap)

    The first ``sink_size`` and last ``local_size`` tokens are always kept.
    Scoring and selection apply to the middle region only; per (batch, kv-head)
    row, ``keep_mid = int(T * (1 - compression_ratio)) - sink - local`` middle
    tokens are kept (indices sorted ascending so temporal order is preserved).
    The kept count is a pure function of ``T`` and the hyperparameters, so the
    cache stays rectangular across heads and layers; only the kept token
    positions differ per head.

    Notes
    -----
    Upstream quirks replicated faithfully:
    - ``compression_ratio=None`` raises at compress time (the research adapter
      injects the adapter-level float when built from config).
    - When ``keep_total < sink + local`` the ``[sink | local]`` concatenation
      is returned, so the kept count can EXCEED the nominal budget
      ``int(T * (1 - compression_ratio))`` (under-compression).
    - Queries come from ``module.q_proj`` WITHOUT rotary applied while the
      cached keys are RoPE-rotated: omega mixes unrotated Q with rotated K and
      tau is ridge leverage of rotated keys (not RoPE-invariant). This matches
      the reference bit-for-bit; do not "fix" it with un-rotation. Under DCA's
      cyclic-rotated keys the semantics shift further, so this sketch is
      intended for ``prefill_method: none``.
    - ``selection_method='multinomial'`` draws from the global torch RNG
      (no seed parameter); seed globally for reproducibility.
    - The alpha-selection machinery (entropy / tail_risk / query_constrained /
      gated_query_constrained) is bypassed whenever ``combine_mode`` is one of
      the envelope modes (alpha is set to NaN and ignored), which includes the
      default configuration; it remains reachable via config.
    - ``_matrix_sqrt_psd`` and ``_compute_query_metric_ridge_tau`` are dormant
      upstream ablation helpers, never called from ``compress``.

    Deviations from kvpress
    -----------------------
    - ``_get_all_queries`` skips the query-aware path (warning + tau-only
      fallback) when ``hidden_states`` and ``keys`` cover different numbers of
      tokens. Upstream assumes they match and would misalign (or crash on the
      reshape) otherwise; in Prism-Test an outer prefill-method hook (e.g.
      reattention) fires before the sketch hook and can leave the cached keys
      shorter than ``hidden_states``.
    - The commented-out duplicate of ``_scores_from_tau_omega_and_values``
      present in the reference source is omitted.
    """

    compression_ratio: Optional[float] = None
    ridge_lambda: float = 1e-4
    sink_size: int = 4
    local_size: int = 28
    min_tokens_to_compress: int = 64

    # Query-aware options.
    query_aware: bool = True
    normalize_queries: bool = False
    normalize_keys_for_query_metric: bool = False
    query_gram_normalization: Literal["mean", "sum"] = "mean"
    query_position_mode: Literal["matching_keys", "all_prefill"] = "matching_keys"
    log_prefill_selection_stats: bool = False

    # Selection options.
    selection_method: Literal["multinomial", "topk"] = "topk"

    # Score combination.
    combine_mode: Literal[
        "multiplicative",
        "additive",
        "envelope",
        "weighted_envelope",
        "fixed_envelope",
    ] = "fixed_envelope"

    # If True:
    #   tau_component   = tau / sum(tau)
    #   omega_component = omega / sum(omega)
    # If False:
    #   tau_component   = tau
    #   omega_component = omega
    normalize_score_components: bool = True

    # Alpha selection. Ignored by envelope / weighted_envelope / fixed_envelope.
    alpha_selection: Literal[
        "fixed",
        "entropy",
        "tail_risk",
        "query_constrained",
        "gated_query_constrained",
    ] = "gated_query_constrained"
    alpha: float = 0.8
    alpha_min: float = 0.0
    alpha_max: float = 1.0
    alpha_grid: str = "0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"

    # Query-constrained alpha selection.
    ridge_slack: float = 0.20
    ridge_penalty: float = 10.0

    # Gate for gated_query_constrained.
    fallback_alpha: float = 1.0
    query_peakiness_threshold: float = 4.0
    gate_ridge_excess_threshold: float = 0.10

    # Weighted envelope option.
    # gamma = 1 + query_boost_strength * (1 - overlap).
    query_boost_strength: float = 2.0

    # Fixed envelope option.
    # score_i = max(p_ridge_i, envelope_gamma * p_query_i), envelope_gamma >= 0.
    # Try < 1 to downweight query side and recover FWE.
    envelope_gamma: float = 1.0

    # Tail-risk / query-constrained validation options.
    alpha_validation_split: Literal["none", "even_odd"] = "even_odd"

    value_norm_power: float = 1.0
    eps: float = 1e-8

    def __post_init__(self):
        if self.compression_ratio is not None:
            assert 0.0 <= self.compression_ratio < 1.0, "compression_ratio must be in [0, 1)"
        assert self.ridge_lambda > 0, "ridge_lambda must be > 0"
        assert self.sink_size >= 0, "sink_size must be >= 0"
        assert self.local_size >= 0, "local_size must be >= 0"
        assert self.min_tokens_to_compress >= 0, "min_tokens_to_compress must be >= 0"
        assert 0.0 <= self.alpha <= 1.0, "alpha must be in [0, 1]"
        assert 0.0 <= self.alpha_min <= 1.0, "alpha_min must be in [0, 1]"
        assert 0.0 <= self.alpha_max <= 1.0, "alpha_max must be in [0, 1]"
        assert self.alpha_min <= self.alpha_max, "alpha_min must be <= alpha_max"
        assert self.value_norm_power >= 0, "value_norm_power must be >= 0"
        assert self.query_gram_normalization in {"mean", "sum"}
        assert self.query_position_mode in {"matching_keys", "all_prefill"}
        assert self.combine_mode in {
            "multiplicative",
            "additive",
            "envelope",
            "weighted_envelope",
            "fixed_envelope",
        }
        assert self.alpha_selection in {
            "fixed",
            "entropy",
            "tail_risk",
            "query_constrained",
            "gated_query_constrained",
        }
        assert self.ridge_slack >= 0.0, "ridge_slack must be >= 0"
        assert self.ridge_penalty >= 0.0, "ridge_penalty must be >= 0"
        assert 0.0 <= self.fallback_alpha <= 1.0, "fallback_alpha must be in [0, 1]"
        assert self.query_peakiness_threshold >= 0.0, "query_peakiness_threshold must be >= 0"
        assert self.gate_ridge_excess_threshold >= 0.0, "gate_ridge_excess_threshold must be >= 0"
        assert self.query_boost_strength >= 0.0, "query_boost_strength must be >= 0"
        assert self.envelope_gamma >= 0.0, "envelope_gamma must be >= 0"
        assert self.alpha_validation_split in {"none", "even_odd"}
        _ = self._parse_alpha_grid()

    def _parse_alpha_grid(self) -> list[float]:
        vals = [float(x.strip()) for x in str(self.alpha_grid).split(",") if x.strip()]
        vals = [a for a in vals if self.alpha_min <= a <= self.alpha_max]
        if not vals:
            vals = [float(self.alpha)]
        return sorted(set(vals))

    def _compute_key_ridge_tau(self, keys: torch.Tensor) -> torch.Tensor:
        """tau_i = k_i^T (K^T K + lambda I)^(-1) k_i."""
        B, H, N, D = keys.shape
        if N == 0:
            return torch.zeros(B, H, 0, device=keys.device, dtype=keys.dtype)

        k = keys.float()
        eye = torch.eye(D, device=k.device, dtype=k.dtype).view(1, 1, D, D)
        gram = k.transpose(-2, -1) @ k
        reg = gram + self.ridge_lambda * eye

        try:
            inv_reg = torch.linalg.inv(reg)
        except torch.linalg.LinAlgError:
            inv_reg = torch.linalg.pinv(reg)

        tau = ((k @ inv_reg) * k).sum(dim=-1).clamp_min(0.0)
        return tau.to(keys.dtype)

    def _get_all_queries(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Return prefill queries aligned with KV heads: [B, H_kv, T, D]."""
        if hidden_states is None or not hasattr(module, "q_proj"):
            return None

        B, H_kv, T, D = keys.shape

        if hidden_states.shape[1] != T:
            logger.warning(
                "Query/key token mismatch (hidden_states has %s tokens, keys have %s); "
                "skipping query-aware scoring.",
                hidden_states.shape[1],
                T,
            )
            return None

        try:
            q = module.q_proj(hidden_states)
        except Exception as exc:
            logger.warning("Could not compute queries from q_proj: %s", exc)
            return None

        if q.shape[-1] % D != 0:
            logger.warning("q_proj output dim %s is not divisible by head_dim %s", q.shape[-1], D)
            return None

        H_q = q.shape[-1] // D
        q = q.view(B, T, H_q, D).transpose(1, 2).contiguous()

        if H_q == H_kv:
            pass
        elif H_q > H_kv and H_q % H_kv == 0:
            group = H_q // H_kv
            q = q.view(B, H_kv, group, T, D).mean(dim=2)
        elif H_kv > H_q and H_kv % H_q == 0:
            repeat = H_kv // H_q
            q = q.repeat_interleave(repeat, dim=1)
        else:
            logger.warning("Incompatible query/KV head counts: H_q=%s, H_kv=%s", H_q, H_kv)
            return None

        q = q.to(keys.dtype)
        if self.normalize_queries:
            q = q / q.float().norm(p=2, dim=-1, keepdim=True).clamp_min(self.eps).to(q.dtype)
        return q.contiguous()

    def _matrix_sqrt_psd(self, gram: torch.Tensor) -> torch.Tensor:
        gram_f = 0.5 * (gram.float() + gram.float().transpose(-2, -1))
        evals, evecs = torch.linalg.eigh(gram_f)
        evals = evals.clamp_min(0.0).sqrt()
        return (evecs * evals.unsqueeze(-2)) @ evecs.transpose(-2, -1)

    def _compute_query_metric_ridge_tau(
        self,
        keys_mid: torch.Tensor,
        queries: torch.Tensor,
    ) -> torch.Tensor:
        """Optional ablation: ridge leverage under query metric G_Q = Q^T Q."""
        B, H, N, D = keys_mid.shape
        T = queries.shape[2]
        if N == 0:
            return torch.zeros(B, H, 0, device=keys_mid.device, dtype=keys_mid.dtype)
        if T == 0:
            return self._compute_key_ridge_tau(keys_mid)

        k = keys_mid.float()
        q = queries.float()
        if self.normalize_keys_for_query_metric:
            k = k / k.norm(p=2, dim=-1, keepdim=True).clamp_min(self.eps)

        G_q = q.transpose(-2, -1) @ q
        if self.query_gram_normalization == "mean":
            G_q = G_q / max(T, 1)

        G_q_sqrt = self._matrix_sqrt_psd(G_q)
        k_tilde = k @ G_q_sqrt

        eye = torch.eye(D, device=k_tilde.device, dtype=k_tilde.dtype).view(1, 1, D, D)
        gram_tilde = k_tilde.transpose(-2, -1) @ k_tilde
        reg = gram_tilde + self.ridge_lambda * eye

        try:
            inv_reg = torch.linalg.inv(reg)
        except torch.linalg.LinAlgError:
            inv_reg = torch.linalg.pinv(reg)

        tau = ((k_tilde @ inv_reg) * k_tilde).sum(dim=-1).clamp_min(0.0)
        return tau.to(keys_mid.dtype)

    def _compute_query_key_interaction(
        self,
        keys_mid: torch.Tensor,
        queries: torch.Tensor,
    ) -> torch.Tensor:
        """omega_i = ||Q k_i||_2 = sqrt(k_i^T Q^T Q k_i)."""
        B, H, N, _ = keys_mid.shape
        T = queries.shape[2]
        if N == 0:
            return torch.zeros(B, H, 0, device=keys_mid.device, dtype=keys_mid.dtype)
        if T == 0:
            return torch.ones(B, H, N, device=keys_mid.device, dtype=keys_mid.dtype)

        k = keys_mid.float()
        q = queries.float()
        if self.normalize_keys_for_query_metric:
            k = k / k.norm(p=2, dim=-1, keepdim=True).clamp_min(self.eps)

        G_q = q.transpose(-2, -1) @ q
        if self.query_gram_normalization == "mean":
            G_q = G_q / max(T, 1)

        omega = ((k @ G_q) * k).sum(dim=-1).clamp_min(0.0).sqrt()
        return omega.to(keys_mid.dtype)

    def _normalize_distribution(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float().clamp_min(self.eps)
        return x / x.sum(dim=-1, keepdim=True).clamp_min(self.eps)

    def _score_components(
        self,
        tau_f: torch.Tensor,
        omega_f: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return ridge/query components used by additive, multiplicative, and envelope modes.

        normalize_score_components=True:
            tau_c   = tau / sum(tau)
            omega_c = omega / sum(omega)

        normalize_score_components=False:
            tau_c   = tau
            omega_c = omega
        """
        if self.normalize_score_components:
            tau_c = self._normalize_distribution(tau_f)
            omega_c = self._normalize_distribution(omega_f)
        else:
            tau_c = tau_f
            omega_c = omega_f

        return tau_c, omega_c

    def _compute_entropy_alpha(self, omega: torch.Tensor) -> tuple[float, float]:
        """Entropy fallback: low entropy -> lower alpha, high entropy -> higher alpha."""
        q = omega.float().clamp_min(self.eps)
        p = q / q.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        entropy = -(p * p.log()).sum(dim=-1)
        log_n = torch.log(torch.tensor(float(q.shape[-1]), device=q.device, dtype=q.dtype))
        normalized_entropy = entropy / log_n.clamp_min(self.eps)
        entropy_scalar = normalized_entropy.mean().item()
        alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * entropy_scalar
        alpha = float(max(self.alpha_min, min(self.alpha_max, alpha)))
        return alpha, entropy_scalar

    def _scores_from_tau_omega_and_values(
        self,
        tau: torch.Tensor,
        values: torch.Tensor,
        omega: Optional[torch.Tensor] = None,
        alpha: Optional[float] = None,
        n_keep: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute value-aware query-ridge scores.

        If normalize_score_components=True:
            tau_c   = tau / sum(tau)
            omega_c = omega / sum(omega)

        If normalize_score_components=False:
            tau_c   = tau
            omega_c = omega

        additive:
            score_i = [alpha * tau_c_i + (1-alpha) * omega_c_i] * ||v_i||^p

        multiplicative:
            score_i = tau_c_i^alpha * omega_c_i^(1-alpha) * ||v_i||^p

        fixed_envelope:
            score_i = max(tau_c_i, envelope_gamma * omega_c_i) * ||v_i||^p
        """
        if alpha is None:
            alpha = self.alpha

        tau_f = tau.float().clamp_min(self.eps)

        value_norms = values.float().norm(p=2, dim=-1).clamp_min(self.eps)
        if self.value_norm_power > 0:
            vweight = value_norms.pow(self.value_norm_power)
        else:
            vweight = torch.ones_like(value_norms)

        if omega is None:
            if self.normalize_score_components:
                tau_c = self._normalize_distribution(tau_f)
            else:
                tau_c = tau_f
            return tau_c * vweight

        omega_f = omega.float().clamp_min(self.eps)
        tau_c, omega_c = self._score_components(tau_f, omega_f)

        if self.combine_mode in {"envelope", "weighted_envelope", "fixed_envelope"}:
            p1 = tau_c
            p2 = omega_c

            if self.combine_mode == "envelope":
                gamma = 1.0

            elif self.combine_mode == "fixed_envelope":
                gamma = float(self.envelope_gamma)

            elif self.combine_mode == "weighted_envelope":
                B, H, N = p1.shape
                k = N if n_keep is None else max(1, min(int(n_keep), N))

                tau_top = torch.topk(p1, k=k, dim=-1).indices
                omega_top = torch.topk(p2, k=k, dim=-1).indices

                mask_tau = torch.zeros_like(p1, dtype=torch.bool)
                mask_omega = torch.zeros_like(p2, dtype=torch.bool)
                mask_tau.scatter_(dim=-1, index=tau_top, value=True)
                mask_omega.scatter_(dim=-1, index=omega_top, value=True)

                overlap = (mask_tau & mask_omega).float().sum(dim=-1) / float(k)
                gamma = 1.0 + self.query_boost_strength * (1.0 - overlap).clamp(0.0, 1.0)
                gamma = gamma.unsqueeze(-1)

            else:
                raise ValueError(f"Unknown envelope combine_mode: {self.combine_mode}")

            scores = torch.maximum(p1, gamma * p2) * vweight

        elif self.combine_mode == "additive":
            scores = alpha * tau_c + (1.0 - alpha) * omega_c
            scores = scores * vweight

        elif self.combine_mode == "multiplicative":
            scores = tau_c.pow(alpha) * omega_c.pow(1.0 - alpha)
            scores = scores * vweight

        else:
            raise ValueError(f"Unknown combine_mode: {self.combine_mode}")

        return scores

    @staticmethod
    def _gather_1d(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Gather x:[B,H,N] using indices:[B,H,K] -> [B,H,K]."""
        return x.gather(dim=2, index=indices)

    def _gather_by_token_indices(self, x: torch.Tensor, token_indices: torch.Tensor) -> torch.Tensor:
        gather_idx = token_indices.unsqueeze(-1).expand(-1, -1, -1, x.shape[-1])
        return x.gather(dim=2, index=gather_idx).contiguous()

    def _select_indices_from_scores(self, scores: torch.Tensor, n_keep: int) -> torch.Tensor:
        B, H, N = scores.shape
        if n_keep <= 0:
            return torch.zeros(B, H, 0, device=scores.device, dtype=torch.long)
        if n_keep >= N:
            return torch.arange(N, device=scores.device, dtype=torch.long).view(1, 1, N).expand(B, H, N)

        flat = scores.float().clamp_min(0.0).reshape(B * H, N)
        row_sums = flat.sum(dim=-1, keepdim=True)
        zero_rows = row_sums.squeeze(-1) <= 0
        if zero_rows.any():
            flat = flat.clone()
            flat[zero_rows] = 1.0

        if self.selection_method == "topk":
            selected = torch.topk(flat, k=n_keep, dim=-1).indices
        elif self.selection_method == "multinomial":
            selected = torch.multinomial(flat, n_keep, replacement=False)
        else:
            raise ValueError(f"Unknown selection_method: {self.selection_method}")

        selected = selected.view(B, H, n_keep)
        return selected.sort(dim=-1).values

    def _choose_alpha_by_tail_risk(
        self,
        tau: torch.Tensor,
        omega_score: torch.Tensor,
        omega_val: torch.Tensor,
        values: torch.Tensor,
        n_keep: int,
    ) -> tuple[float, float, float, float]:
        """Choose alpha by minimizing empirical tail-risk certificate."""
        if n_keep <= 0:
            return float(self.alpha), float("nan"), float("nan"), float("nan")

        vnorm = values.float().norm(p=2, dim=-1).clamp_min(self.eps)
        if self.value_norm_power != 1.0:
            vweight = vnorm.pow(self.value_norm_power)
        else:
            vweight = vnorm

        tau_mass = tau.float().clamp_min(self.eps) * vweight
        omega_val_mass = omega_val.float().clamp_min(self.eps) * vweight

        tau_total = tau_mass.sum(dim=-1).clamp_min(self.eps)
        omega_total = omega_val_mass.sum(dim=-1).clamp_min(self.eps)

        best_alpha = float(self.alpha)
        best_risk = float("inf")
        best_ridge_tail = float("nan")
        best_query_tail = float("nan")

        for a in self._parse_alpha_grid():
            scores = self._scores_from_tau_omega_and_values(
                tau=tau,
                values=values,
                omega=omega_score,
                alpha=float(a),
            )
            idx = self._select_indices_from_scores(scores, n_keep)

            kept_tau = self._gather_1d(tau_mass, idx).sum(dim=-1)
            kept_omega = self._gather_1d(omega_val_mass, idx).sum(dim=-1)

            ridge_tail = (1.0 - kept_tau / tau_total).clamp_min(0.0)
            query_tail = (1.0 - kept_omega / omega_total).clamp_min(0.0)
            risk = torch.maximum(ridge_tail, query_tail)

            risk_scalar = risk.mean().item()
            if risk_scalar < best_risk:
                best_risk = risk_scalar
                best_alpha = float(a)
                best_ridge_tail = ridge_tail.mean().item()
                best_query_tail = query_tail.mean().item()

        return best_alpha, best_risk, best_ridge_tail, best_query_tail

    def _choose_alpha_query_constrained(
        self,
        tau: torch.Tensor,
        omega_score: torch.Tensor,
        omega_val: torch.Tensor,
        values: torch.Tensor,
        n_keep: int,
    ) -> tuple[float, float, float, float, float]:
        """Query-first alpha selection with a ridge-tail constraint."""
        if n_keep <= 0:
            return float(self.alpha), float("nan"), float("nan"), float("nan"), float("nan")

        vnorm = values.float().norm(p=2, dim=-1).clamp_min(self.eps)
        if self.value_norm_power != 1.0:
            vweight = vnorm.pow(self.value_norm_power)
        else:
            vweight = vnorm

        tau_mass = tau.float().clamp_min(self.eps) * vweight
        omega_val_mass = omega_val.float().clamp_min(self.eps) * vweight

        tau_total = tau_mass.sum(dim=-1).clamp_min(self.eps)
        omega_total = omega_val_mass.sum(dim=-1).clamp_min(self.eps)

        def tails_for_alpha(a: float) -> tuple[float, float]:
            scores = self._scores_from_tau_omega_and_values(
                tau=tau,
                values=values,
                omega=omega_score,
                alpha=float(a),
            )
            idx = self._select_indices_from_scores(scores, n_keep)
            kept_tau = self._gather_1d(tau_mass, idx).sum(dim=-1)
            kept_omega = self._gather_1d(omega_val_mass, idx).sum(dim=-1)

            ridge_tail = (1.0 - kept_tau / tau_total).clamp_min(0.0)
            query_tail = (1.0 - kept_omega / omega_total).clamp_min(0.0)
            return ridge_tail.mean().item(), query_tail.mean().item()

        ridge_ref, _ = tails_for_alpha(1.0)

        best_alpha = float(self.alpha)
        best_obj = float("inf")
        best_ridge_tail = float("nan")
        best_query_tail = float("nan")

        for a in self._parse_alpha_grid():
            ridge_tail, query_tail = tails_for_alpha(float(a))
            violation = max(0.0, ridge_tail - ridge_ref - self.ridge_slack)
            obj = query_tail + self.ridge_penalty * violation

            if obj < best_obj or (abs(obj - best_obj) <= 1e-12 and float(a) < best_alpha):
                best_obj = obj
                best_alpha = float(a)
                best_ridge_tail = ridge_tail
                best_query_tail = query_tail

        return best_alpha, best_obj, best_ridge_tail, best_query_tail, ridge_ref

    def _query_peakiness(self, omega: torch.Tensor) -> float:
        omega_f = omega.float().clamp_min(self.eps)
        peak = omega_f.max(dim=-1).values / omega_f.mean(dim=-1).clamp_min(self.eps)
        return float(peak.mean().detach().cpu().item())

    def _ridge_excess_for_alpha(
        self,
        tau: torch.Tensor,
        omega_score: torch.Tensor,
        values: torch.Tensor,
        n_keep: int,
        alpha: float,
    ) -> tuple[float, float, float]:
        vnorm = values.float().norm(p=2, dim=-1).clamp_min(self.eps)
        vweight = vnorm.pow(self.value_norm_power) if self.value_norm_power != 1.0 else vnorm
        tau_mass = tau.float().clamp_min(self.eps) * vweight
        tau_total = tau_mass.sum(dim=-1).clamp_min(self.eps)

        def ridge_tail_for(a: float) -> float:
            scores = self._scores_from_tau_omega_and_values(
                tau=tau,
                values=values,
                omega=omega_score,
                alpha=float(a),
            )
            idx = self._select_indices_from_scores(scores, n_keep)
            kept_tau = self._gather_1d(tau_mass, idx).sum(dim=-1)
            ridge_tail = (1.0 - kept_tau / tau_total).clamp_min(0.0)
            return float(ridge_tail.mean().detach().cpu().item())

        ridge_tail_alpha = ridge_tail_for(alpha)
        ridge_tail_ridge = ridge_tail_for(1.0)
        return ridge_tail_alpha, ridge_tail_ridge, ridge_tail_alpha - ridge_tail_ridge

    def _make_query_split(self, queries_for_metric: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (queries_score, queries_val) for alpha self-validation."""
        if self.alpha_validation_split == "even_odd" and queries_for_metric.shape[2] >= 2:
            q_score = queries_for_metric[:, :, 0::2, :]
            q_val = queries_for_metric[:, :, 1::2, :]
            if q_score.shape[2] > 0 and q_val.shape[2] > 0:
                return q_score, q_val
        return queries_for_metric, queries_for_metric

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del attentions, kwargs

        if self.compression_ratio is None:
            raise ValueError("compression_ratio must be set before RidgeSketch.compress is called")
        if self.compression_ratio == 0:
            return keys, values

        _, _, T, _ = keys.shape
        if T < self.min_tokens_to_compress:
            return keys, values

        sink = min(self.sink_size, T)
        local = min(self.local_size, max(0, T - sink))
        mid_start = sink
        mid_end = T - local
        mid_len = max(0, mid_end - mid_start)
        if mid_len == 0:
            return keys, values

        keep_total = int(T * (1.0 - self.compression_ratio))
        keep_total = max(0, min(keep_total, T))
        keep_mid = min(max(keep_total - sink - local, 0), mid_len)

        if keep_mid <= 0:
            return (
                torch.cat([keys[:, :, :sink, :], keys[:, :, mid_end:, :]], dim=2).contiguous(),
                torch.cat([values[:, :, :sink, :], values[:, :, mid_end:, :]], dim=2).contiguous(),
            )
        if keep_mid == mid_len:
            return keys, values

        keys_mid = keys[:, :, mid_start:mid_end, :]
        values_mid = values[:, :, mid_start:mid_end, :]
        tau = self._compute_key_ridge_tau(keys_mid)

        omega_score = None
        omega_val = None
        omega_final = None
        if self.query_aware:
            queries = self._get_all_queries(module=module, hidden_states=hidden_states, keys=keys)
            if queries is not None:
                if self.query_position_mode == "matching_keys":
                    queries_for_metric = queries[:, :, mid_start:mid_end, :]
                elif self.query_position_mode == "all_prefill":
                    queries_for_metric = queries
                else:
                    raise ValueError(f"Unknown query_position_mode: {self.query_position_mode}")

                q_score, q_val = self._make_query_split(queries_for_metric)
                omega_score = self._compute_query_key_interaction(keys_mid, q_score)
                omega_val = self._compute_query_key_interaction(keys_mid, q_val)
                omega_final = self._compute_query_key_interaction(keys_mid, queries_for_metric)
            else:
                logger.warning("Query-aware reweighting skipped because queries are unavailable.")

        alpha = self.alpha
        entropy_scalar = float("nan")
        risk_scalar = float("nan")
        ridge_tail_scalar = float("nan")
        query_tail_scalar = float("nan")
        ridge_ref_scalar = float("nan")
        query_peakiness_scalar = float("nan")
        gate_excess_scalar = float("nan")
        gate_used_fallback = False

        if omega_score is not None:
            if self.combine_mode in {"envelope", "weighted_envelope", "fixed_envelope"}:
                alpha = float("nan")
            elif self.alpha_selection == "fixed":
                alpha = self.alpha
            elif self.alpha_selection == "entropy":
                alpha, entropy_scalar = self._compute_entropy_alpha(omega_score)
            elif self.alpha_selection == "tail_risk":
                alpha, risk_scalar, ridge_tail_scalar, query_tail_scalar = self._choose_alpha_by_tail_risk(
                    tau=tau,
                    omega_score=omega_score,
                    omega_val=omega_val if omega_val is not None else omega_score,
                    values=values_mid,
                    n_keep=keep_mid,
                )
            elif self.alpha_selection == "query_constrained":
                alpha, risk_scalar, ridge_tail_scalar, query_tail_scalar, ridge_ref_scalar = self._choose_alpha_query_constrained(
                    tau=tau,
                    omega_score=omega_score,
                    omega_val=omega_val if omega_val is not None else omega_score,
                    values=values_mid,
                    n_keep=keep_mid,
                )
            elif self.alpha_selection == "gated_query_constrained":
                query_peakiness_scalar = self._query_peakiness(omega_score)
                _, _, gate_excess_scalar = self._ridge_excess_for_alpha(
                    tau=tau,
                    omega_score=omega_score,
                    values=values_mid,
                    n_keep=keep_mid,
                    alpha=0.0,
                )
                use_query_rule = (
                    query_peakiness_scalar >= self.query_peakiness_threshold
                    and gate_excess_scalar <= self.gate_ridge_excess_threshold
                )
                if use_query_rule:
                    alpha, risk_scalar, ridge_tail_scalar, query_tail_scalar, ridge_ref_scalar = self._choose_alpha_query_constrained(
                        tau=tau,
                        omega_score=omega_score,
                        omega_val=omega_val if omega_val is not None else omega_score,
                        values=values_mid,
                        n_keep=keep_mid,
                    )
                else:
                    alpha = self.fallback_alpha
                    gate_used_fallback = True
            else:
                raise ValueError(f"Unknown alpha_selection: {self.alpha_selection}")

        omega_for_final_score = omega_final if omega_final is not None else omega_score
        scores = self._scores_from_tau_omega_and_values(
            tau=tau,
            values=values_mid,
            omega=omega_for_final_score,
            alpha=alpha,
            n_keep=keep_mid,
        )

        if self.log_prefill_selection_stats:
            omega_mean = omega_for_final_score.float().mean().item() if omega_for_final_score is not None else float("nan")
            omega_max = omega_for_final_score.float().max().item() if omega_for_final_score is not None else float("nan")
            print(
                f"RidgeSketch stats: "
                f"combine_mode={self.combine_mode} "
                f"alpha_selection={self.alpha_selection} "
                f"alpha={alpha:.4f} "
                f"envelope_gamma={self.envelope_gamma:.4f} "
                f"entropy={entropy_scalar:.4f} "
                f"tail_risk={risk_scalar:.4f} "
                f"ridge_tail={ridge_tail_scalar:.4f} "
                f"query_tail={query_tail_scalar:.4f} "
                f"ridge_ref={ridge_ref_scalar:.4f} "
                f"ridge_slack={self.ridge_slack:.4f} "
                f"query_boost_strength={self.query_boost_strength:.4f} "
                f"query_peakiness={query_peakiness_scalar:.4f} "
                f"gate_excess={gate_excess_scalar:.4f} "
                f"gate_fallback={int(gate_used_fallback)} "
                f"keep_mid={keep_mid} "
                f"mid_len={mid_len} "
                f"tau_mean={tau.float().mean().item():.4e} "
                f"tau_max={tau.float().max().item():.4e} "
                f"omega_mean={omega_mean:.4e} "
                f"omega_max={omega_max:.4e}",
                flush=True,
            )

        keep_idx_mid = self._select_indices_from_scores(scores, keep_mid)
        kept_mid_keys = self._gather_by_token_indices(keys_mid, keep_idx_mid)
        kept_mid_values = self._gather_by_token_indices(values_mid, keep_idx_mid)

        out_keys = torch.cat([keys[:, :, :sink, :], kept_mid_keys, keys[:, :, mid_end:, :]], dim=2)
        out_values = torch.cat([values[:, :, :sink, :], kept_mid_values, values[:, :, mid_end:, :]], dim=2)
        return out_keys.contiguous(), out_values.contiguous()


@register_kv_compressor("random_sketch_press")
@dataclass
class RandomSketchRidgeSketch(RidgeSketch):
    """
    Prefill-time KV compression baseline mirroring RidgeSketch, intended to
    sample with uniform random scores instead of ridge leverage scores.

    Port of ``RandomSketchPress`` (kvpress/presses/random_sketch_press.py; a
    research-fork addition in the local kvpress 0.5.1 checkout, not upstream
    NVIDIA kvpress). Unrelated to upstream ``RandomPress``, which is ported
    separately as ``RandomSketch`` (registry name "random").

    Upstream bug, replicated faithfully: the single override ``_compute_tau``
    is DEAD CODE. ``RidgePress.compress`` calls ``_compute_key_ridge_tau`` and
    nothing in the kvpress checkout ever calls ``_compute_tau``, so as written
    the press is bitwise-identical to ``RidgePress`` under the same
    configuration and no randomness ever executes. This port preserves the
    dead override and the resulting RidgeSketch-equivalent behavior (pinned by
    tests) rather than wiring the documented intent into the live scoring
    path.
    """

    def _compute_tau(self, keys: torch.Tensor) -> torch.Tensor:
        """
        Return uniform random scores in [0, 1) with shape [B, H, N].

        keys shape: [B, H, N, D]
        tau shape:    [B, H, N]
        """
        B, H, N, _ = keys.shape
        if N == 0:
            return torch.zeros(B, H, 0, device=keys.device, dtype=keys.dtype)
        return torch.rand(B, H, N, device=keys.device, dtype=keys.dtype)
