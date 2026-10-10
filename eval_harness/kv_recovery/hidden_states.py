"""Forward-hook capture of decoder-layer outputs (the residual stream after each block)
and, optionally, the final-norm output (the exact ``lm_head`` input).

``output_hidden_states=True`` is deliberately NOT used: its last entry is post-final-norm
while the others are pre-norm, it materialises every layer, and hybrid wrappers differ in
what they return. Hooks are explicit, cheap and model-agnostic
(ported from ``kv_compression_adaptation/src/analysis/sensitivity.py``).
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterable, Iterator, List, Sequence, Union

import torch
from torch import nn

from .model_spec import decoder_layers, final_norm

FINAL_NORM_KEY = "norm"
StateKey = Union[int, str]


class CapturedStates:
    """Accumulates per-call outputs per key; ``states()`` concatenates along the sequence axis
    (so token-by-token feeding yields one ``[1, L, H]`` tensor per key)."""

    def __init__(self, detach: bool):
        self.detach = detach
        self._buffers: Dict[StateKey, List[torch.Tensor]] = {}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

    def hook_for(self, key: StateKey):
        def hook(module, args, output):  # noqa: ARG001
            h = output[0] if isinstance(output, (tuple, list)) else output
            if not isinstance(h, torch.Tensor):
                raise TypeError(f"layer {key}: unexpected output type {type(h).__name__}")
            if self.detach:
                h = h.detach()
            self._buffers.setdefault(key, []).append(h)
        return hook

    def states(self) -> Dict[StateKey, torch.Tensor]:
        out: Dict[StateKey, torch.Tensor] = {}
        for key, chunks in self._buffers.items():
            out[key] = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=1)
        return out

    def clear(self) -> None:
        self._buffers.clear()

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


@contextmanager
def capture_layer_outputs(model: nn.Module, layer_indices: Sequence[int], *, detach: bool,
                          include_final_norm: bool = False) -> Iterator[CapturedStates]:
    """Register output hooks on ``decoder_layers(model)[i]`` for ``i in layer_indices`` (and on the
    final norm when requested); hooks are removed on exit, captured tensors stay available."""
    layers = decoder_layers(model)
    cap = CapturedStates(detach=detach)
    try:
        for i in layer_indices:
            i = int(i)
            if not (0 <= i < len(layers)):
                raise IndexError(f"layer index {i} out of range [0, {len(layers)})")
            cap._handles.append(layers[i].register_forward_hook(cap.hook_for(i)))
        if include_final_norm:
            norm = final_norm(model)
            if norm is None:
                raise ValueError("model has no final norm module to capture")
            cap._handles.append(norm.register_forward_hook(cap.hook_for(FINAL_NORM_KEY)))
        yield cap
    finally:
        cap.remove()


def gather_positions(states: Dict[StateKey, torch.Tensor], positions: torch.Tensor) -> Dict[StateKey, torch.Tensor]:
    """``{key: [1, L, H]}`` -> ``{key: [P, H]}`` at the given suffix positions."""
    out: Dict[StateKey, torch.Tensor] = {}
    for key, h in states.items():
        if h.dim() != 3 or h.shape[0] != 1:
            raise ValueError(f"{key}: expected [1, L, H], got {tuple(h.shape)}")
        out[key] = h[0].index_select(0, positions.to(h.device))
    return out


def state_keys(layer_indices: Iterable[int], include_final_norm: bool) -> List[StateKey]:
    keys: List[StateKey] = [int(i) for i in layer_indices]
    if include_final_norm:
        keys.append(FINAL_NORM_KEY)
    return keys
