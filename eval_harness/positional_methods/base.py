"""Door 1 — Positional methods.

A *positional method* changes **how token positions are stamped** — i.e. the
RoPE frequencies and/or the position indices fed into the rotation.  This is the
home for frequency-scaling context-extension methods (YaRN, NTK) and
position-remapping ones (Linear-PI, SelfExtend), which the legacy code declared
(``PrefillMethod.compute_inv_freq``) but never wired in.

Door 1 works by wrapping the model's shared ``rotary_emb`` so it emits modified
``(cos, sin)``.  It is active in **both** phases automatically (the same
rotation everyone downstream sees), so it sits *outermost* in the door stack::

    positional_method(model)        # door 1  ← THIS
      → attention_method(model)     # door 2
        → kv_compressor(model)      # door 3

Override points
---------------
* :meth:`compute_inv_freq` — frequency scaling (NTK / YaRN).  The model's
  ``inv_freq`` is temporarily swapped for the returned tensor so HF's own
  cos/sin construction (including any baked-in ``attention_scaling`` amplitude)
  is reused exactly.
* :meth:`remap_position_ids` — position remap (Linear-PI / SelfExtend).
* :attr:`mscale` — optional logit-temperature scalar applied to ``(cos, sin)``
  (YaRN's attention-temperature trick).

The base class is the identity transform: it forwards positions unchanged, keeps
the native frequencies, and uses ``mscale == 1`` — so installing the base
``PositionalMethod`` is byte-for-byte equivalent to not installing one.

Interaction with Door 2
-----------------------
An attention method that computes its own RoPE (DCA) bypasses the model's
``rotary_emb`` entirely, so it overrides this door for the layers it owns.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, ClassVar, Generator, Optional, Set, Tuple

import torch

logger = logging.getLogger(__name__)


@dataclass
class PositionalMethod:
    """Base class for Door-1 positional methods (identity by default)."""

    mscale: float = 1.0

    # Whether :meth:`compute_inv_freq` actually varies with ``seq_len``.  The
    # shipped methods (NTK, YaRN) ignore ``seq_len`` — their frequencies are a
    # function of the config alone — so the interceptor computes the result
    # once and reuses it for every rotary forward (decode runs one forward per
    # token).  A method whose frequencies depend on the running length (e.g. a
    # *dynamic* NTK) must set this ``True`` so the cache keys on ``seq_len``.
    inv_freq_depends_on_seq_len: ClassVar[bool] = False

    # ------------------------------------------------------------------
    # Override points
    # ------------------------------------------------------------------

    def compute_inv_freq(
        self,
        original_inv_freq: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        """Return modified RoPE inverse frequencies.

        Default: the native frequencies unchanged.

        Parameters
        ----------
        original_inv_freq : Tensor [head_dim // 2]
            The model's native ``inv_freq``.
        seq_len : int
            Total sequence length being processed (lets YaRN/NTK pick a scale).
        """
        return original_inv_freq

    def remap_position_ids(
        self,
        position_ids: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        """Return remapped position IDs (e.g. Linear-PI divides by a factor).

        Default: positions unchanged.
        """
        return position_ids

    # ------------------------------------------------------------------
    # Capability flags
    # ------------------------------------------------------------------

    def supported_backends(self) -> Set[str]:
        return {"research"}

    # ------------------------------------------------------------------
    # Context manager: wrap the shared rotary embedding
    # ------------------------------------------------------------------

    @contextmanager
    def __call__(self, model: Any) -> Generator:
        from eval_harness.attention_methods._method_base import get_rotary_emb

        rotary = get_rotary_emb(model)
        if rotary is None:
            logger.warning(
                "%s: model exposes no rotary_emb; running as a no-op.",
                type(self).__name__,
            )
            yield
            return

        saved_forward = rotary.forward
        original_inv_freq = getattr(rotary, "inv_freq", None)
        method = self

        # Cache the (inv_freq, swap) decision so per-token decode forwards don't
        # redo compute_inv_freq (recover_base_and_dim's .item() GPU syncs) and
        # torch.equal on every call.  Keyed on seq_len only when the method's
        # frequencies actually depend on it; otherwise a single shared entry.
        _SEQ_LEN_AGNOSTIC = object()
        freq_cache: dict = {}

        def _resolve_swap(seq_len: int):
            key = seq_len if method.inv_freq_depends_on_seq_len else _SEQ_LEN_AGNOSTIC
            if key not in freq_cache:
                new_inv_freq = (
                    method.compute_inv_freq(original_inv_freq, seq_len)
                    if original_inv_freq is not None
                    else None
                )
                swap = (
                    new_inv_freq is not None
                    and original_inv_freq is not None
                    and new_inv_freq is not original_inv_freq
                    and not torch.equal(new_inv_freq, original_inv_freq)
                )
                freq_cache[key] = (new_inv_freq if swap else None, swap)
            return freq_cache[key]

        def wrapped(x: torch.Tensor, position_ids: torch.Tensor, **kw: Any):
            seq_len = _infer_seq_len(position_ids)
            position_ids = method.remap_position_ids(position_ids, seq_len)

            new_inv_freq, swap = _resolve_swap(seq_len)
            if swap:
                rotary.inv_freq = new_inv_freq
                try:
                    cos, sin = saved_forward(x, position_ids, **kw)
                finally:
                    rotary.inv_freq = original_inv_freq
            else:
                cos, sin = saved_forward(x, position_ids, **kw)

            if method.mscale != 1.0:
                cos = cos * method.mscale
                sin = sin * method.mscale
            return cos, sin

        rotary.forward = wrapped
        try:
            yield
        finally:
            rotary.forward = saved_forward


def _infer_seq_len(position_ids: torch.Tensor) -> int:
    """Largest absolute position + 1 (so frequency scaling can size itself)."""
    if position_ids is None or position_ids.numel() == 0:
        return 0
    return int(position_ids.max().item()) + 1


def recover_base_and_dim(inv_freq: torch.Tensor) -> Tuple[float, int]:
    """Recover the RoPE base ``theta`` and rotary ``dim`` from ``inv_freq``.

    The native frequencies are ``inv_freq[j] = base ** (-(2j)/dim)`` for
    ``j = 0 … dim/2 - 1``, so ``dim = 2 * len(inv_freq)`` and, from ``j = 1``,
    ``base = inv_freq[1] ** (-dim/2)``.  Methods that rebuild frequencies (NTK,
    YaRN) use this so they need no extra config to recover ``theta``.
    """
    inv_freq = inv_freq.detach().to(torch.float64)
    half = inv_freq.shape[-1]
    dim = 2 * half
    if half < 2:
        raise ValueError("inv_freq must have at least 2 entries to recover base")
    base = float(inv_freq[1].item() ** (-dim / 2.0))
    return base, dim
