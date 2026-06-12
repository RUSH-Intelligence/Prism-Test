import logging
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.cache_utils import QuantizedCache

from eval_harness.sketch.sketches.adakv_sketch import AdaKVSketch
from eval_harness.sketch.sketches.base_sketch import BaseSketch
from eval_harness.sketch.sketches.scorer_sketch import ScorerSketch
from eval_harness.sketch.utils import extract_keys_and_values

logger = logging.getLogger(__name__)


@dataclass
class DecodingSketch(BaseSketch):
    base_sketch: ScorerSketch | AdaKVSketch
    compression_interval: int = 512
    target_size: int = 2048
    hidden_states_buffer_size: int = 256

    def __post_init__(self):
        assert isinstance(self.base_sketch, (ScorerSketch, AdaKVSketch)), "DecodingSketch requires a ScorerSketch as input"
        self.hidden_states_buffer = defaultdict(list)
        self.layer_step_counts = defaultdict(int)

        assert self.compression_interval > 0, "compression_interval must be greater than 0"
        assert self.target_size > 0, "target_size must be greater than 0"

    def post_init_from_model(self, model):
        self.base_sketch.post_init_from_model(model)

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k_len = keys.shape[2]
        target_compression_ratio = self._find_target_compression_ratio(k_len, self.target_size)

        original_compression_ratio = self.base_sketch.compression_ratio
        self.base_sketch.compression_ratio = target_compression_ratio
        result = self.base_sketch.compress(module, hidden_states, keys, values, attentions, kwargs)
        self.base_sketch.compression_ratio = original_compression_ratio
        return result

    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        hidden_states = kwargs["hidden_states"]
        cache = kwargs["past_key_values"]
        q_len = hidden_states.shape[1]
        layer_idx = module.layer_idx

        if not self._is_decoding_step(module, kwargs, q_len):
            return output

        self.hidden_states_buffer[layer_idx].append(hidden_states.detach().clone())
        self.layer_step_counts[layer_idx] += 1

        if (self.layer_step_counts[layer_idx] >= self.compression_interval) or (q_len >= self.target_size):
            cache_layer = cache.layers[module.layer_idx]
            keys, values = extract_keys_and_values(cache, module.layer_idx)
            attentions = output[1] if len(output) > 1 and output[1] is not None else None
            buffered_hidden_states = torch.cat(self.hidden_states_buffer[layer_idx], dim=1)
            keys, values = self.compress(module, buffered_hidden_states, keys, values, attentions, kwargs)

            if isinstance(cache, QuantizedCache):
                cache_layer._quantized_keys = cache_layer._quantize(keys, axis=cache_layer.axis_key)
                cache_layer._quantized_values = cache_layer._quantize(values, axis=cache_layer.axis_value)
                cache_layer.keys = torch.zeros(0, dtype=keys.dtype, device=keys.device)
                cache_layer.values = torch.zeros(0, dtype=keys.dtype, device=keys.device)
                cache_layer.cumulative_length = keys.shape[2]
            else:
                cache_layer.keys = keys
                cache_layer.values = values

            self.layer_step_counts[layer_idx] = 0
            self.hidden_states_buffer[layer_idx] = []

        self.hidden_states_buffer[layer_idx] = (
            self.hidden_states_buffer[layer_idx][-self.hidden_states_buffer_size :]
            if self.hidden_states_buffer_size > 0
            else []
        )
        return output

    def reset(self):
        self.hidden_states_buffer = defaultdict(list)
        self.layer_step_counts = defaultdict(int)

    @contextmanager
    def __call__(self, model: PreTrainedModel):
        try:
            with super().__call__(model):
                yield
        finally:
            self.reset()

    def _find_target_compression_ratio(self, q_len: int, target_tokens: int) -> float:
        if q_len <= target_tokens:
            return 0.0

        ratio = 1.0 - (target_tokens / q_len)
        low, high = 0.0, 1.0
        max_iterations = 20
        iteration = 0

        while iteration < max_iterations:
            n_kept = int(q_len * (1 - ratio))
            if n_kept == target_tokens:
                break
            if n_kept > target_tokens:
                low = ratio
                ratio = (ratio + high) / 2
            else:
                high = ratio
                ratio = (low + ratio) / 2
            iteration += 1
        return ratio
