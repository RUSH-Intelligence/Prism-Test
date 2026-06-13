from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Union

import torch
from torch import nn
from transformers import PreTrainedModel

from eval_harness.kv_compression.base import KVCompressor
from eval_harness.kv_compression.registry import get_kv_compressor, register_kv_compressor

SketchSpec = Union[KVCompressor, str, Sequence]


def _resolve_member(spec: SketchSpec) -> KVCompressor:
    if isinstance(spec, KVCompressor):
        return spec
    if isinstance(spec, str):
        return get_kv_compressor(spec)
    if (
        isinstance(spec, Sequence)
        and len(spec) == 2
        and isinstance(spec[0], str)
        and (spec[1] is None or isinstance(spec[1], Mapping))
    ):
        return get_kv_compressor(spec[0], **dict(spec[1] or {}))
    raise TypeError(
        "ComposedSketch members must be KVCompressor instances, registry names, "
        f"or (name, kwargs) pairs, got {spec!r}"
    )


@register_kv_compressor("composed")
@dataclass
class ComposedSketch(KVCompressor):
    """Composed compression: chain multiple compression methods sequentially.

    Port of kvpress 0.5.1 ``ComposedPress`` (kvpress/presses/composed_press.py).

    Applies multiple compression methods in sequence, with each method
    operating on the output of the previous one: each member's
    ``forward_hook`` re-extracts K/V from the cache the previous member just
    rewrote and prunes them further. The hook ``output`` tuple (post-attention
    hidden states, attention weights) is passed through unchanged, so it is
    stale relative to the pruned cache — members that consume attention
    weights see the original, un-pruned weights (the kvpress-documented
    hazard; moot on this pipeline, where ``output[1]`` is always ``None``
    under sdpa). Decode gating is delegated entirely to the members: the
    wrapper never gates, each delegated hook no-ops on decode steps, and the
    static ratio product is harmlessly recomputed (parity with kvpress).

    ``compression_ratio`` is a derived attribute (``None`` until the first
    hook fires), not a dataclass field, exactly as in kvpress — after each
    hook it is recomputed as ``1 - prod(1 - r_i)`` over the members. The
    realized kept count is the nested truncation
    ``k_n = int(k_{n-1} * (1 - r_n))``, which can differ from
    ``int(S * (1 - compression_ratio))``. Members lacking a numeric
    ``compression_ratio`` (e.g. ``DecodingSketch``) raise
    ``TypeError``/``AttributeError`` when the hook fires, mirroring kvpress's
    lack of validation.

    Deviations from kvpress
    -----------------------
    - The ``__post_init__`` assertion excluding ``AdaKVPress``/``KVzipPress``
      is dropped: neither class is ported to Prism-Test.
    - Members may be given as registry names or ``(name, kwargs)`` pairs in
      addition to sketch instances; they are resolved through the sketch
      registry in ``__post_init__``. This makes the wrapper reachable from
      flat YAML config (``kv_compressor_kwargs={"presses": [["knorm",
      {"compression_ratio": 0.5}], ...]}``) without adapter plumbing; kvpress
      only accepts instances.

    Parameters
    ----------
    presses : list[KVCompressor | str | (str, dict)]
        Compression methods applied sequentially, each operating on the
        compressed output of the previous one. Final compression ratio is
        ``1 - prod(1 - r_i)``.
    """

    presses: list[SketchSpec]

    def __post_init__(self):
        self.compression_ratio = None
        self.presses = [_resolve_member(press) for press in self.presses]

    def post_init_from_model(self, model: PreTrainedModel):
        for press in self.presses:
            press.post_init_from_model(model)

    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        retained_fraction = 1.0
        for press in self.presses:
            output = press.forward_hook(module, input, kwargs, output)
            retained_fraction *= 1 - press.compression_ratio
        self.compression_ratio = 1 - retained_fraction
        return output
