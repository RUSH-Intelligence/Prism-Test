from dataclasses import dataclass, field
from typing import Literal, Optional

import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel

from eval_harness.kv_compression.registry import register_kv_compressor
from eval_harness.kv_compression.base import ScorerKVCompressor


class KVzapConfig(PretrainedConfig):
    model_type: str = "kvzap"
    input_dim: int
    output_dim: int
    hidden_dim: Optional[int] = None
    n_modules: int


class KVzapModel(PreTrainedModel):
    config_class = KVzapConfig  # type: ignore[assignment]

    def __init__(self, config):
        super().__init__(config)
        self.all_tied_weights_keys = {}
        if config.hidden_dim is None:
            # Linear model
            self.layers = nn.ModuleList(
                [nn.Linear(config.input_dim, config.output_dim) for _ in range(config.n_modules)]
            )
        else:
            # 2-layer MLP model
            self.layers = nn.ModuleList(
                nn.Sequential(
                    nn.Linear(config.input_dim, config.hidden_dim),
                    nn.GELU(),
                    nn.Linear(config.hidden_dim, config.output_dim),
                )
                for _ in range(config.n_modules)
            )

    def forward(self, x):
        return torch.stack([module(x[:, i, :]) for i, module in enumerate(self.layers)], dim=1)


@register_kv_compressor("kvzap")
@dataclass
class KVzapSketch(ScorerKVCompressor):
    """
    KVzap (https://arxiv.org/abs/2601.07891) is a fast approximation of KVzip that works
    in both prefilling and decoding. It applies a lightweight surrogate model to the hidden
    states to predict importance scores for every KV pair.
    model_type can be "linear" or "mlp"; the surrogate is loaded from the hub repo
    ``nvidia/KVzap-{model_type}-{model-name}`` on first context entry.

    Port of ``kvpress/presses/kvzap_press.py`` (KVzapPress).

    Deviations from kvpress:
    - Standalone top-k variant only, equivalent to kvpress's ``kvzap_mlp_head`` registry
      entry. kvpress's flagship ``kvzap_mlp`` (DMSPress wrapper, threshold-based
      ``masked_key_indices`` eviction) and ``kvzap_mlp_layer`` (AdaKVPress wrapper) are not
      ported: their per-(head, token) masking requires head-ragged caches that Prism's
      rectangular ``DynamicCache`` cannot represent. Compare eval numbers against
      kvpress's ``kvzap_mlp_head`` rows.
    - Added ``model_name_override``: replaces ``model.config.name_or_path.split("/")[-1]``
      in the surrogate repo-id derivation, so models loaded from local snapshot
      directories (whose dir name does not match the hub model name) can still resolve
      the published surrogate. ``None`` keeps the kvpress derivation verbatim.
    - Added a shape-alignment assert in ``score``: scores follow ``hidden_states`` length,
      so a pre-populated cache (keys longer than the current pass) would silently restrict
      selection to the suffix. Unreachable in the single-shot prefill pipeline.
    - Do not wrap in ``DecodingSketch``/``PrefillDecodingSketch``: score length follows
      the step's ``hidden_states`` while decode-time wrappers pass full-cache-length
      keys (kvpress likewise excludes KVzapPress from DecodingPress tests).
    """

    model_type: Literal["linear", "mlp"] = "mlp"
    model_name_override: Optional[str] = None
    kvzap_model_name: Optional[str] = field(default=None, init=False)

    def post_init_from_model(self, model: PreTrainedModel):
        base_name = self.model_name_override or model.config.name_or_path.split("/")[-1]
        kvzap_model_name = f"nvidia/KVzap-{self.model_type}-{base_name}"
        if kvzap_model_name != self.kvzap_model_name:
            self.kvzap_model_name = kvzap_model_name
            self.kvzap_model = KVzapModel.from_pretrained(self.kvzap_model_name)

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> torch.Tensor:
        kvzap_module = self.kvzap_model.layers[module.layer_idx]
        kvzap_module = kvzap_module.to(hidden_states.device, dtype=hidden_states.dtype).eval()
        scores = kvzap_module(hidden_states).transpose(1, 2)
        assert scores.shape[-1] == keys.shape[2], (
            "KVzap scores follow hidden_states length; a pre-populated cache "
            f"is unsupported (scores cover {scores.shape[-1]} positions, cache holds {keys.shape[2]})"
        )
        return scores
