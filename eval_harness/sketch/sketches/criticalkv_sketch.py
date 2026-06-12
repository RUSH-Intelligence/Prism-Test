import logging
from dataclasses import dataclass

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.models.llama.modeling_llama import repeat_kv

from eval_harness.sketch.sketches.base_sketch import BaseSketch
from eval_harness.sketch.sketches.registry import register_sketch
from eval_harness.sketch.sketches.scorer_sketch import ScorerSketch

logger = logging.getLogger(__name__)


@register_sketch("criticalkv")
class CriticalKVSketch(ScorerSketch):
    """
    CriticalKV: Two-stage compression with output projection weighting.

    Enhances existing scoring methods by rescaling scores using the L1 norm
    of output projection applied to values (Wo @ values). Stage 1 locks in
    ``int((1 - compression_ratio) * k_len * first_stage_ratio)`` tokens by the
    raw inner scores; the remaining keep-budget goes to the highest
    ``(scores + epsilon) * ||Wo V||_1`` rescaled scores. Selection and physical
    pruning are the inherited uniform ``ScorerSketch.compress``.

    Based on CriticalKV (https://arxiv.org/abs/2502.03805). Port of kvpress
    ``CriticalKVPress`` (kvpress/presses/criticalkv_press.py).

    Parameters
    ----------
    press : ScorerSketch
        Base scoring method to enhance with output projection weighting.
    epsilon : float, default=1e-4
        Small value for numerical stability in score rescaling.
    first_stage_ratio : float, default=0.5
        Fraction of compression budget allocated to first stage selection.
        Remaining budget used in second stage with output projection weighting.

    Deviations from kvpress
    -----------------------
    - ``vwl1norm`` reads the GQA group count from ``module.num_key_value_groups``
      everywhere (upstream mixes ``config.num_attention_heads // H_kv`` and
      ``module.num_key_value_groups``) and uses ``module.head_dim`` instead of
      ``config.head_dim`` (absent on some configs).
    - The upstream ``ExpectedAttentionPress.use_vnorm`` warning is duck-typed on
      a ``use_vnorm`` attribute of the inner sketch instead of an isinstance
      check.
    """

    def __init__(self, press: ScorerSketch, epsilon: float = 1e-4, first_stage_ratio: float = 0.5):
        self.press = press
        self.epsilon = epsilon
        self.first_stage_ratio = first_stage_ratio

        assert isinstance(self.press, ScorerSketch), "CriticalKVSketch requires a ScorerSketch as input"
        if getattr(self.press, "use_vnorm", False):
            logger.warning("use_vnorm should be disabled for CriticalKVSketch")

    def post_init_from_model(self, model: PreTrainedModel):
        self.press.post_init_from_model(model)

    @property  # type: ignore[misc]
    def compression_ratio(self) -> float:
        return self.press.compression_ratio

    @compression_ratio.setter
    def compression_ratio(self, value: float):
        self.press.compression_ratio = value

    @staticmethod
    def vwl1norm(values: torch.Tensor, module: nn.Module) -> torch.Tensor:
        bsz, num_key_value_heads, k_len, _ = values.shape
        num_key_value_groups = module.num_key_value_groups
        Wo = module.o_proj.weight.transpose(0, 1)
        Wo = Wo.view(num_key_value_heads * num_key_value_groups, module.head_dim, module.config.hidden_size)
        V = repeat_kv(values, num_key_value_groups)

        # We use head-wise computation instead of direct matmul to reduce the memory usage of WoV.
        head_WoV_norm_list = []
        for head in range(V.size(1)):
            head_WoV = V[:, head, :, ...].matmul(Wo[head, ...].unsqueeze(0))
            head_WoV_norm = torch.norm(head_WoV, p=1, dim=-1)
            head_WoV_norm_list.append(head_WoV_norm)

        WoV_norm = torch.stack(head_WoV_norm_list, dim=1)
        WoV_norm = WoV_norm.view(bsz, num_key_value_heads, num_key_value_groups, k_len).mean(dim=2)
        return WoV_norm

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        # Stage 1
        scores = self.press.score(module, hidden_states, keys, values, attentions, kwargs)
        k_len = keys.shape[2]
        selection_budget = int((1 - self.compression_ratio) * k_len * self.first_stage_ratio)
        top_k_index = torch.topk(scores, selection_budget, sorted=True, dim=-1).indices

        # Stage 2
        projected_norm = self.vwl1norm(values, module)
        scores = (scores + self.epsilon) * projected_norm

        # Merge the two stages
        scores.scatter_(-1, top_k_index, torch.finfo(scores.dtype).max)

        return scores


@register_sketch("critical_adakv")
@dataclass
class CriticalAdaKVSketch(BaseSketch):
    """
    CriticalAdaKV: Combined two-stage compression with adaptive head-wise selection.

    Combines output projection weighting from CriticalKV with adaptive head-wise
    compression from AdaKV. Provides both accurate importance estimation and
    head-specific compression adaptation.

    Based on CriticalAdaKV (https://arxiv.org/abs/2502.03805). Port of kvpress
    ``CriticalAdaKVPress`` (kvpress/presses/criticalkv_press.py).

    Like ``AdaKVSketch``, the cache is never physically pruned: ``compress``
    returns keys/values unchanged and records ``module.masked_key_indices``,
    enforced by the globally installed attention patch
    (``eval_harness/sketch/attention_patch.py``) on every ``q_len < k_len``
    forward — zero memory savings, non-eager attention required, and
    incompatible with prefill methods that replace ``self_attn.forward``
    wholesale (``dca``, ``reattention_exact``).

    Parameters
    ----------
    press : ScorerSketch
        The underlying scoring method used to evaluate token importance.
    alpha_safeguard : float, default=0.20
        Minimum fraction of KV pairs that each head must retain. (see AdaKVSketch)
    epsilon : float, default=1e-4
        Small value for numerical stability in score rescaling.
    first_stage_ratio : float, default=0.5
        Fraction of compression budget allocated to first stage selection.

    Deviations from kvpress
    -----------------------
    - ``compress`` asserts ``bsz == 1``: the upstream per-head budget
      accumulation (``head_budgets.scatter_add_``) sums winning indices over the
      whole batch into a single (H_kv,) tensor, silently mis-allocating budgets
      for B > 1; the eval pipeline is batch-1, so the port makes the limit
      explicit instead of inheriting the bug.
    - The upstream ``ExpectedAttentionPress.use_vnorm`` warning is duck-typed on
      a ``use_vnorm`` attribute of the inner sketch.
    - Replicated upstream quirks: ``post_init_from_model`` is NOT delegated to
      the inner press (kvpress's ``CriticalAdaKVPress`` omits it, unlike
      ``CriticalKVPress``); the redundant ``budget_scores`` re-scatter before
      budget allocation is kept verbatim.
    """

    press: ScorerSketch = None
    alpha_safeguard: float = 0.20
    epsilon: float = 1e-4
    first_stage_ratio: float = 0.5

    def __post_init__(self):
        assert 0 <= self.alpha_safeguard <= 1, "alpha_safeguard should be in 0, 1]"
        assert isinstance(self.press, ScorerSketch), "CriticalAdaKVSketch requires a ScorerSketch as input"
        if getattr(self.press, "use_vnorm", False):
            logger.warning("use_vnorm should be disabled for CriticalAdaKVSketch")

    @property
    def compression_ratio(self) -> float:
        return self.press.compression_ratio

    @compression_ratio.setter
    def compression_ratio(self, value: float):
        self.press.compression_ratio = value

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if self.compression_ratio == 0:
            return keys, values

        assert module.config._attn_implementation != "eager", "eager mode not supported"

        # Compute scores
        scores = self.press.score(module, hidden_states, keys, values, attentions, kwargs)
        bsz, num_key_value_heads, k_len = scores.shape
        assert bsz == 1, "CriticalAdaKVSketch only supports batch size 1 (head budgets are summed across the batch)"

        # Make sure to keep at least alpha * (1 - compression_ratio) KV pairs per head
        n_kept = int(k_len * (1 - self.compression_ratio))  # ScorerSketch definition
        n_safe = int(n_kept * self.alpha_safeguard)
        top_indices = torch.topk(scores, n_safe, dim=-1).indices
        scores.scatter_(-1, top_indices, torch.finfo(scores.dtype).max)

        ############################
        # Start of CriticalKV code #
        ############################

        # Budget allocation
        budget_scores = scores.scatter(-1, top_indices, torch.finfo(scores.dtype).max)
        budget_scores = budget_scores.reshape(bsz, -1)
        top_indices = torch.topk(budget_scores, n_kept * num_key_value_heads, dim=-1).indices
        top_indices_head_idx = top_indices // k_len
        head_budgets = torch.zeros(num_key_value_heads, device=keys.device, dtype=torch.int64)
        head_budgets.scatter_add_(0, top_indices_head_idx.flatten(), torch.ones_like(top_indices_head_idx.flatten()))

        # Stage 1
        head_selection_budget_1st = (head_budgets * self.first_stage_ratio).to(torch.int64).tolist()
        top_k_index = torch.topk(scores, max(head_selection_budget_1st), sorted=True, dim=-1).indices
        for head_idx in range(num_key_value_heads):
            phase1_budget = head_selection_budget_1st[head_idx]
            scores[:, head_idx, :].scatter_(-1, top_k_index[:, head_idx, :phase1_budget], torch.finfo(scores.dtype).max)

        # Stage 2
        projected_norm = CriticalKVSketch.vwl1norm(values, module)
        scores = (scores + self.epsilon) * projected_norm
        top_k_index = torch.topk(scores, max(head_budgets), sorted=True, dim=-1).indices
        for head_idx in range(num_key_value_heads):
            budget = head_budgets[head_idx]
            scores[:, head_idx, :].scatter_(-1, top_k_index[:, head_idx, :budget], torch.finfo(scores.dtype).max)

        ##########################
        # End of CriticalKV code #
        ##########################

        # Compute bottom-k across heads
        n_pruned = num_key_value_heads * (k_len - n_kept)
        indices = torch.topk(-scores.reshape(bsz, -1), n_pruned, dim=1).indices.flatten()

        # Save indices to mask during the attention mechanism. Please refer to attention_patch.py for more details
        batch_indices = torch.arange(bsz).repeat_interleave(n_pruned)
        head_indices = indices // k_len
        seq_indices = indices % k_len
        module.masked_key_indices = (batch_indices, head_indices, seq_indices)
        return keys, values
