"""Door 2 — attention methods (forward replacement, phase-gated)."""

from .base import AttentionMethod, AttentionPhase
from .registry import (
    available_attention_methods,
    get_attention_method,
    register_attention_method,
)

__all__ = [
    "AttentionMethod",
    "AttentionPhase",
    "available_attention_methods",
    "get_attention_method",
    "register_attention_method",
]
