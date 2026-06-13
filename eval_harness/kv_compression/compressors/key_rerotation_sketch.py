import logging
from dataclasses import dataclass

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.models.llama.modeling_llama import rotate_half

from eval_harness.kv_compression.base import KVCompressor
from eval_harness.kv_compression.registry import register_kv_compressor
from eval_harness.kv_compression.base import ScorerKVCompressor

logger = logging.getLogger(__name__)


@register_kv_compressor("key_rerotation")
@dataclass
class KeyRerotationSketch(KVCompressor):
    """
    Key Rerotation: RoPE-aware compression wrapper for maintaining positional encoding.

    Enhances any ScorerKVCompressor by applying key rerotation after compression to maintain
    proper RoPE (Rotary Position Embedding) representations. When tokens are pruned,
    remaining tokens need positional encodings adjusted for their new positions: kept
    indices are sorted ascending and the kept keys are re-rotated to the canonical
    contiguous positions ``0..n_kept-1``. This method is used in several key-value
    cache compression methods, such as
    - SinkCache implementation in Hugging Face's transformers library
    - FINCH: Prompt-guided Key-Value Cache Compression for Large Language Models

    Port of kvpress ``KeyRerotationPress`` (kvpress/presses/key_rerotation_press.py).

    The re-rotation trig is built from the raw ``module.rotary_emb.inv_freq`` at the
    position deltas ``new_pos - old_pos``, so it has unit magnitude (no
    ``attention_scaling``). Cached keys on scaled-RoPE models carry
    ``s * R(old_pos) * k_raw`` and block rotations compose, so the output is
    ``s * R(new_pos) * k_raw`` — exactly one factor of the rope ``attention_scaling``
    survives, matching the key the model would have produced natively at ``new_pos``
    (no s^2 undo/redo defect; see ``prefill_methods/base.py::undo_rotary_pos_emb``).

    Parameters
    ----------
    press : ScorerKVCompressor
        The underlying scoring method to enhance with key rerotation.
        Rerotation is applied after the press determines which tokens to keep.
        ``compression_ratio`` is a get/set property proxying to this wrapped sketch.

    Deviations from kvpress
    -----------------------
    - kvpress's pipeline special-cases ``KeyRerotationPress`` and rebases
      question/decode position ids to the compressed cache length (kvpress
      pipeline.py:233-234). Prism's ``SketchTextGenerationPipeline`` has no such hook
      for sketches, so question/decode positions continue from the original context
      length, leaving a positional gap of ``S - n_kept`` between the re-rotated keys
      (at positions ``0..n_kept-1``) and the first question token. A warning is
      logged at construction.
    - ``compression_ratio`` is a property, not a dataclass field, so
      ``ResearchAdapter._build_sketch``'s ``fields()``-based ratio injection does not
      fire for the registry name ``key_rerotation``; construct programmatically with
      the ratio set on the wrapped sketch, e.g.
      ``KeyRerotationSketch(press=KnormSketch(compression_ratio=0.5))``, or add an
      adapter special case (``DecodingSketch`` precedent).

    Quirks kept for kvpress parity: ``compress`` names ``keys.shape[2]`` ``q_len``
    though it is the key length (identical during single-pass prefill);
    ``n_kept = int(q_len * (1 - compression_ratio))`` may reach 0 for tiny sequences
    at extreme ratios, yielding an empty cache that breaks decode (not clamped).
    """

    press: ScorerKVCompressor

    def __post_init__(self):
        assert isinstance(self.press, ScorerKVCompressor)
        logger.warning(
            "KeyRerotationSketch compacts kept keys to positions 0..n_kept-1, but "
            "SketchTextGenerationPipeline does not rebase question/decode position ids "
            "to the compressed cache length (kvpress's pipeline special-cases "
            "KeyRerotationPress for this); decode will see a positional gap between "
            "the last key and the first question token."
        )

    def post_init_from_model(self, model: PreTrainedModel):
        self.press.post_init_from_model(model)

    @property
    def compression_ratio(self):
        return self.press.compression_ratio

    @compression_ratio.setter
    def compression_ratio(self, value):
        self.press.compression_ratio = value

    @staticmethod
    def _rerotate_cos_sin(
        x: torch.Tensor,
        inv_freq: torch.Tensor,
        selected_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(cos, sin) of the delta rotation ``new_pos - old_pos`` for each kept key,
        transcribed from kvpress ``KeyRerotationPress._rerotate_cos_sin``.

        ``x`` provides dtype/device, ``inv_freq`` is ``module.rotary_emb.inv_freq``
        of shape ``(d//2,)``, ``selected_positions`` are the sorted kept indices of
        shape ``(bsz, num_key_value_heads, n_kept)``. Returns cos/sin of shape
        ``(bsz, num_key_value_heads, n_kept, d)``.
        """
        bsz, num_key_value_heads, n_kept = selected_positions.shape
        device = selected_positions.device
        device_type = x.device.type
        dtype = x.dtype
        # New positional indices
        idx = torch.arange(0, n_kept, device=device)  # (n_kept,)
        idx = idx.unsqueeze(0)  # (1, n_kept)
        inv_freq = inv_freq[None, None, :, None].float().expand(bsz, num_key_value_heads, -1, 1)
        idx = idx[:, None, :].float().expand(bsz, num_key_value_heads, n_kept)
        # Compute delta between new and selected positions
        delta_pos = idx - selected_positions  # (bsz, num_key_value_heads, n_kept)
        delta_pos = delta_pos.unsqueeze(2)  # (bsz, num_key_value_heads, 1, n_kept)

        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"

        with torch.autocast(device_type=device_type, enabled=False):
            freqs = delta_pos.float() * inv_freq.float()  # (bsz, num_key_value_heads, d//2, n_kept)
            freqs = freqs.transpose(2, 3)  # (bsz, num_key_value_heads, n_kept, d//2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos().contiguous()
            sin = emb.sin().contiguous()
        return cos.to(dtype=dtype), sin.to(dtype=dtype)

    @staticmethod
    def rerotate_keys(
        module: nn.Module,
        indices: torch.Tensor,
        keys: torch.Tensor,
    ) -> torch.Tensor:
        """Gather kept keys and re-rotate them to the contiguous positions
        ``0..n_kept-1``, transcribed from kvpress ``KeyRerotationPress.rerotate_keys``."""
        new_cos, new_sin = KeyRerotationSketch._rerotate_cos_sin(keys, module.rotary_emb.inv_freq, indices)
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)
        keys = keys.gather(2, indices).contiguous()
        return (keys * new_cos) + (rotate_half(keys) * new_sin)

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.press.compression_ratio == 0:
            return keys, values

        scores = self.press.score(module, hidden_states, keys, values, attentions, kwargs)

        q_len = keys.shape[2]
        n_kept = int(q_len * (1 - self.press.compression_ratio))
        indices = scores.topk(n_kept, dim=-1).indices
        indices = torch.sort(indices, dim=2).values
        keys = self.rerotate_keys(module, indices, keys)
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)
        values = values.gather(2, indices).contiguous()
        return keys, values
