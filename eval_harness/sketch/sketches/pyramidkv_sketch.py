from dataclasses import dataclass

import torch
from torch import nn
from transformers import PreTrainedModel

from eval_harness.sketch.sketches.registry import register_sketch
from eval_harness.sketch.sketches.snapkv_sketch import SnapKVSketch


@register_sketch("pyramidkv")
@dataclass
class PyramidKVSketch(SnapKVSketch):
    """
    PyramidKV: Layer-wise adaptive KV cache allocation with pyramid structure.

    Dynamically adjusts KV cache sizes across transformer layers, allocating
    more tokens to lower layers and fewer to higher layers. Scoring is
    inherited from SnapKV; only the per-layer retained count changes: layer 0
    keeps ``max_num`` entries, the last layer ``~min_num``, decreasing
    linearly with depth so the mean over layers is ``q_len * (1 -
    compression_ratio)`` — the global ratio is honored exactly while
    individual layers deviate.

    Based on PyramidKV (https://arxiv.org/abs/2406.02069).
    Port of kvpress ``PyramidKVPress`` (kvpress/presses/pyramidkv_press.py);
    budget formula from
    https://github.com/Zefan-Cai/KVCache-Factory/blob/main/pyramidkv/pyramidkv_utils.py#L197.

    Parameters
    ----------
    compression_ratio : float, default=0.0
        Global fraction of key-value pairs to remove; per-layer budgets
        average to ``q_len * (1 - compression_ratio)``.
    window_size : int, default=64
        Number of recent tokens whose attention scores the rest; the window
        is always kept (max-padded scores) and is the floor of every layer
        budget.
    kernel_size : int, default=5
        Size of the pooling kernel for attention smoothing (inherited from
        SnapKV; must be odd).
    beta : int, default=20
        Pyramid steepness: ``min_num = q_len * (1 - compression_ratio) /
        beta``; larger beta creates steeper top-vs-bottom layer budget
        differences.
    uniform_budget : bool, default=False
        Escape hatch for non-flash-attention decode paths: every layer keeps
        ``round(q_len * (1 - compression_ratio))`` (the kvpress short-prompt
        fallback formula), keeping the cache rectangular across layers. This
        degenerates PyramidKV to uniform SnapKV and is a safety fallback, not
        the method.

    Deviations from kvpress
    -----------------------
    - Ragged-decode gate: the per-layer pyramid budgets leave the cache
      cross-layer ragged. Under the pinned transformers 5.9, sdpa/eager build
      ONE decode mask sized from layer 0 and never slice it to the per-layer
      key length, so any shorter layer hits a broadcast ``RuntimeError`` on
      the multi-token question forward; only flash_attention_2 (per-layer
      bottom-right-aligned causal handling) decodes a ragged cache correctly.
      ``post_init_from_model`` therefore raises ``ValueError`` unless the
      model runs ``flash_attention_2``, ``uniform_budget=True``, or
      ``compression_ratio == 0`` (kvpress installs no such guard).
    - ``uniform_budget`` flag (see above); not present in kvpress.
    - Inherits SnapKV's deviations (duck-typed pre-RoPE query extraction,
      odd-``kernel_size`` assert, dead ``attentions`` branch).

    Quirks kept for kvpress parity: budgets use Python ``round()`` (banker's
    rounding, e.g. ``round(62.5) == 62``), not ``int()`` truncation; the
    short-prompt fallback branch (whenever ``min_num < window_size``, i.e.
    roughly ``q_len < beta * window_size / (1 - compression_ratio)``) silently
    degenerates to a uniform SnapKV budget for ALL layers; a model with
    ``num_hidden_layers == 1`` raises ``ZeroDivisionError`` in the steps
    division; ``beta >= 1`` is asserted inside ``get_layer_budget``; kept KV
    pairs stay in score-descending topk order. Do not combine with the DCA
    prefill method (see SnapKVSketch).
    """

    compression_ratio: float = 0.0
    window_size: int = 64
    kernel_size: int = 5
    beta: int = 20
    uniform_budget: bool = False

    def post_init_from_model(self, model: PreTrainedModel):
        if self.uniform_budget or self.compression_ratio == 0:
            return
        attn_impl = getattr(model.config, "_attn_implementation", None)
        if attn_impl != "flash_attention_2":
            raise ValueError(
                f"PyramidKVSketch produces a cross-layer ragged KV cache, which decodes correctly "
                f"only under flash_attention_2 (model uses attn_implementation={attn_impl!r}): "
                f"transformers 5.x sizes the decode mask from layer 0 and sdpa/eager raise a "
                f"broadcast RuntimeError on the multi-token question forward. Load the model with "
                f"attn_implementation='flash_attention_2' or set uniform_budget=True (degenerates "
                f"to a SnapKV-uniform budget)."
            )

    def get_layer_budget(
        self,
        module: nn.Module,
        q_len: int,
    ) -> int:
        """
        Compute the budget for each layer based on the pyramid shape.

        Transcribed from kvpress ``PyramidKVPress.get_layer_budget``: this
        always applies ``compression_ratio`` (instead of disabling compression
        or keeping a fixed budget for short queries like the original code),
        via ``max_capacity_prompt = window_size + q_len * (1 -
        compression_ratio)`` so that ``total_kvcache_size =
        (max_capacity_prompt - window_size) * num_layers = q_len * num_layers
        * (1 - compression_ratio)``.
        """
        assert self.beta >= 1, "Beta should >= 1"

        if self.uniform_budget:
            return round(q_len * (1 - self.compression_ratio))

        max_capacity_prompt = self.window_size + q_len * (1 - self.compression_ratio)

        min_num = (max_capacity_prompt - self.window_size) / self.beta
        max_num = (max_capacity_prompt - self.window_size) * 2 - min_num

        if max_num >= q_len - self.window_size:
            max_num = q_len - self.window_size
            min_num = (max_capacity_prompt - self.window_size) * 2 - max_num

        if not (q_len >= max_num >= min_num >= self.window_size):
            return round(q_len * (1 - self.compression_ratio))

        steps = (max_num - min_num) / (module.config.num_hidden_layers - 1)
        return round(max_num - module.layer_idx * steps)

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

        scores = self.score(module, hidden_states, keys, values, attentions, kwargs)

        k_len = keys.shape[2]
        n_kept = self.get_layer_budget(module, k_len)
        indices = scores.topk(n_kept, dim=-1).indices
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)

        keys = keys.gather(2, indices).contiguous()
        values = values.gather(2, indices).contiguous()

        return keys, values
