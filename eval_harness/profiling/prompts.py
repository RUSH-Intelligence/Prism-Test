"""Exact-length prompt construction for the performance benchmark.

Timing is essentially content-independent (shapes are fixed), but the *token
count* is not: a cell labelled "128K" that silently head-truncates to 120K is a
wrong number, not a noisy one.  So:

* the returned ``context_ids`` has EXACTLY ``n_tokens`` columns, and
* requesting more than the model window is an **error, not a truncation**.

The trim happens AFTER any chat templating.  Trimming first and templating after
ships an ``N + k``-token cell labelled ``N``.
"""

from __future__ import annotations

import random
from typing import Optional, Tuple

import torch

# A fixed, offline, deterministic word pool.  Real English so the tokenizer
# produces a realistic tokens-per-word ratio; no asset, no download.
_WORDS = (
    "the quick brown fox jumps over a lazy dog while distant thunder rolls across "
    "empty fields and travellers pause beneath old stone arches counting minutes "
    "until the storm passes leaving wet roads glittering under lamplight as birds "
    "return to hedgerows and the village clock strikes seven in the cool evening air"
).split()


def synthetic_text(n_words: int, seed: int = 42) -> str:
    rng = random.Random(seed)
    return " ".join(rng.choice(_WORDS) for _ in range(n_words))


def build_exact_prompt(
    tokenizer,
    n_tokens: int,
    *,
    seed: int = 42,
    max_model_len: Optional[int] = None,
    reserve: int = 0,
    source_text: Optional[str] = None,
) -> Tuple[str, torch.Tensor]:
    """Return ``(text, ids)`` where ``ids.shape[1] == n_tokens`` exactly.

    ``reserve`` is the number of positions the caller still needs (question +
    decode steps); it is checked against ``max_model_len`` so absolute RoPE
    positions stay inside the trained window.
    """
    if n_tokens < 1:
        raise ValueError(f"n_tokens must be >= 1, got {n_tokens}")
    if max_model_len is not None and n_tokens + reserve > max_model_len:
        raise ValueError(
            f"context {n_tokens} + reserve {reserve} = {n_tokens + reserve} exceeds "
            f"max_model_len {max_model_len}. Refusing to silently truncate: lower "
            f"--context-lengths or --decode-steps."
        )

    def encode(text: str) -> torch.Tensor:
        return tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]

    # Overshoot, then trim to the exact count. Word count is a lower bound on
    # token count for this pool, so 1.2x plus a floor converges in one pass.
    text = source_text or synthetic_text(int(n_tokens * 1.2) + 64, seed=seed)
    ids = encode(text)
    while ids.shape[1] < n_tokens:
        text = text + " " + synthetic_text(n_tokens - ids.shape[1] + 64, seed=seed + 1)
        ids = encode(text)

    ids = ids[:, :n_tokens].contiguous()
    return tokenizer.decode(ids[0], skip_special_tokens=True), ids
