from dataclasses import dataclass
from functools import cache
from typing import Optional

import torch
from huggingface_hub import PyTorchModelHubMixin, get_collection
from torch import nn
from transformers import PreTrainedModel

from eval_harness.sketch.sketches.registry import register_sketch
from eval_harness.sketch.sketches.scorer_sketch import ScorerSketch


class QFilters(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(self, num_layers: int, num_kv_heads: int, kv_head_dim: int):
        super().__init__()
        self.q_filters = torch.nn.Parameter(torch.randn(num_layers, num_kv_heads, kv_head_dim))


@register_sketch("qfilter")
@dataclass
class QFilterSketch(ScorerSketch):
    """Q-Filter: learned filter-based KV cache compression.

    Port of kvpress ``QFilterPress`` (kvpress/presses/qfilter_press.py).

    This method uses pre-trained learned filters (Q-filters) to score and
    compress key-value pairs. Unlike heuristic-based methods, Q-filters are
    vectors that identify important tokens for specific model architectures.

    The method works by:
    1. Loading pre-trained Q-filter parameters for the specific model
    2. Computing dot products between keys and the learned filters
    3. Using these dot products as importance scores for compression
    4. Pruning tokens with the lowest filter response scores

    The Q-filters are automatically loaded based on the model name and are
    expected to be available in the Hugging Face collection
    ``nthngdy/q-filters-67a4994dcb302a3d37f3d119`` (e.g. there is a filter for
    Llama-3.1-8B-Instruct but NOT for Meta-Llama-3-8B).

    Based on Q-Filter (https://arxiv.org/abs/2503.02812).

    Scoring uses the RoPE-rotated cached keys, matching kvpress (the published
    filters are trained on post-RoPE key distributions). Validated only with
    ``prefill_method: none``: DCA stores keys rotated at cyclic positions
    (a distribution shift vs. filter training), and combining with the
    ReAttention hook double-prunes the cache.

    Deviations from kvpress
    -----------------------
    - ``q_filters`` is a regular constructor argument (kvpress uses
      ``field(init=False, default=None)``) so pre-built filter tensors of shape
      ``(num_layers, num_kv_heads, kv_head_dim)`` can be injected for offline
      runs and tests. When already set, ``post_init_from_model`` is a no-op
      (injected filters are used as-is, with no ``model.dtype`` cast).
    - ``post_init_from_model`` skips the hub download when
      ``compression_ratio == 0`` (kvpress downloads regardless, even though
      ``compress`` short-circuits before ``score`` is ever reached).

    Parameters
    ----------
    compression_ratio : float, default=0.0
        Fraction of key-value pairs to remove during compression.
    q_filters : torch.Tensor, optional
        Pre-built filter tensor ``(num_layers, num_kv_heads, kv_head_dim)``.
        When ``None``, filters are downloaded from ``nthngdy/<model>_qfilt`` in
        ``post_init_from_model``.
    """

    compression_ratio: float = 0.0
    q_filters: Optional[torch.Tensor] = None

    def post_init_from_model(self, model: PreTrainedModel):
        if self.q_filters is not None or self.compression_ratio == 0:
            return
        model_name = model.config.name_or_path.split("/")[-1]
        self.q_filters = self.load_q_filters(model_name)
        self.q_filters = self.q_filters.to(model.dtype)

    @staticmethod
    @cache
    def load_q_filters(model_name: str) -> torch.Tensor:
        model_name = model_name if "Meta-Llama-3.1-405B" in model_name else model_name.replace("Meta-Llama", "Llama")
        try:
            return QFilters.from_pretrained(f"nthngdy/{model_name}_qfilt").q_filters
        except TypeError:
            raise ValueError(
                f"Could not load Q-filters for {model_name}. Available models: {QFilterSketch.available_qfilters()}"
            )

    @staticmethod
    def available_qfilters() -> list[str]:
        collection = get_collection("nthngdy/q-filters-67a4994dcb302a3d37f3d119", token=False)
        return [x.item_id.split("/")[-1][:-6] for x in collection.items]

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        if self.q_filters is None:
            raise ValueError(
                "Q-filters not loaded. If you are using a wrapper press, make sure to call post_init_from_model."
            )
        q_filter = self.q_filters[module.layer_idx][None, :, None]
        q_filter = q_filter.to(keys.device)
        scores = -(q_filter * keys).sum(dim=-1)
        return scores
