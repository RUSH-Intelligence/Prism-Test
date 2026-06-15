from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from eval_harness.kv_compression.base import KVCompressor
from eval_harness.kv_compression.compressors.decoding_sketch import DecodingSketch


@dataclass
class PrefillDecodingSketch(KVCompressor):
    prefilling_sketch: Optional[KVCompressor] = None
    decoding_sketch: Optional[DecodingSketch] = None

    def post_init_from_model(self, model):
        if self.prefilling_sketch is not None:
            self.prefilling_sketch.post_init_from_model(model)
        if self.decoding_sketch is not None:
            self.decoding_sketch.post_init_from_model(model)

    def set_phase(self, phase) -> None:
        # The pipeline sets the phase on this wrapper; forward it to the inner
        # sketches so they don't fall back to the cache_position heuristic
        # (which misclassifies non-first chunked-prefill chunks as decode).
        super().set_phase(phase)
        if self.prefilling_sketch is not None:
            self.prefilling_sketch.set_phase(phase)
        if self.decoding_sketch is not None:
            self.decoding_sketch.set_phase(phase)

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_len = hidden_states.shape[1]
        is_decoding_step = self._is_decoding_step(module, kwargs, q_len)
        if not is_decoding_step and self.prefilling_sketch is not None:
            return self.prefilling_sketch.compress(module, hidden_states, keys, values, attentions, kwargs)
        if self.decoding_sketch is not None:
            return self.decoding_sketch.compress(module, hidden_states, keys, values, attentions, kwargs)
        return keys, values

    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        hidden_states = kwargs["hidden_states"]
        q_len = hidden_states.shape[1]
        is_decoding_step = self._is_decoding_step(module, kwargs, q_len)
        if not is_decoding_step and self.prefilling_sketch is not None:
            return self.prefilling_sketch.forward_hook(module, input, kwargs, output)
        if self.decoding_sketch is not None:
            return self.decoding_sketch.forward_hook(module, input, kwargs, output)
        return output

    @contextmanager
    def __call__(self, model: PreTrainedModel):
        try:
            with super().__call__(model):
                yield
        finally:
            if self.decoding_sketch is not None:
                self.decoding_sketch.reset()
