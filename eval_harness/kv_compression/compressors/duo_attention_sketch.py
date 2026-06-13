import json
import logging
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import torch
from transformers import PreTrainedModel
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from eval_harness.kv_compression.base import KVCompressor
from eval_harness.kv_compression.registry import register_kv_compressor

logger = logging.getLogger(__name__)

PATTERNS_DICT = {
    "togethercomputer/Llama-2-7B-32K-Instruct": "Llama-2-7B-32K-Instruct/lr%3D0.02-reg%3D0.05-ctx%3D1000_32000-multi_passkey10",  # noqa: E501
    "gradientai//Llama-3-8B-Instruct-Gradient-1048k": "Llama-3-8B-Instruct-Gradient-1048k/lr%3D0.02-reg%3D0.05-ctx%3D1000_32000-multi_passkey10",  # noqa: E501
    "gradientai//Llama-3-8B-Instruct-Gradient-4194k": "Llama-3-8B-Instruct-Gradient-4194k/lr%3D0.02-reg%3D0.05-ctx%3D1000_32000-multi_passkey10",  # noqa: E501
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "Meta-Llama-3.1-8B-Instruct/lr=0.02-reg=0.05-ctx=1000_128000-multi_passkey10",  # noqa: E501
    # Prism adaptation: HF renamed the checkpoint; alias the post-rename name to the same pattern.
    "meta-llama/Llama-3.1-8B-Instruct": "Meta-Llama-3.1-8B-Instruct/lr=0.02-reg=0.05-ctx=1000_128000-multi_passkey10",  # noqa: E501
    "mistralai/Mistral-7B-Instruct-v0.2": "Mistral-7B-Instruct-v0.2/lr%3D0.02-reg%3D0.05-ctx%3D1000_32000-multi_passkey10",  # noqa: E501
    "mistralai/Mistral-7B-Instruct-v0.3": "Mistral-7B-Instruct-v0.3/lr%3D0.02-reg%3D0.05-ctx%3D1000_32000-multi_passkey10",  # noqa: E501
}

_BASE_URL = "https://raw.githubusercontent.com/mit-han-lab/duo-attention/refs/heads/main/attn_patterns"


def duo_attention_on_the_fly(model: PreTrainedModel, num_samples: int = 50, q_len: int = 500) -> np.ndarray:
    """
    New experimental method to quickly compute DuoAttention scores
    (port of kvpress ``duo_attention_on_the_fly``):
    - Compute the mean query and key on num_samples random samples from BookSum
    - Repeat the mean query and key q_len times and apply RoPE to get (Q, K)
    - Compute the attention weights for (Q[-1], K) and compute the "area under the cumulated attention curve"
    These scores could also be saved to avoid recomputing them but this method is still experimental.

    Requires network access (BookSum dataset + tokenizer download) and a Llama-family
    module layout (``model.model.layers``, ``input_layernorm``, ``model.model.rotary_emb``).
    Must not be combined with prefill methods that replace ``self_attn.forward`` (e.g. DCA):
    the scoring forwards run inside the prefill-method context and would be corrupted.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model.config.name_or_path)
    num_heads = model.config.num_attention_heads
    num_key_value_heads = model.config.num_key_value_heads
    num_key_value_groups = num_heads // num_key_value_heads

    dataset = load_dataset("kmfoda/booksum", split="train").to_pandas()
    texts = dataset.sample(num_samples, random_state=42)["chapter"].tolist()

    position_ids = torch.arange(q_len).unsqueeze(0)
    scores = torch.zeros((model.config.num_hidden_layers, num_key_value_heads))

    for text in texts:
        with torch.no_grad():
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            hidden_states = list(model(**inputs, output_hidden_states=True).hidden_states[:-1])

            for layer_idx, h in enumerate(hidden_states):
                module = model.model.layers[layer_idx]
                d = module.self_attn.head_dim
                h = module.input_layernorm(h)

                q = module.self_attn.q_proj(h)
                q = q.view(1, q.shape[1], -1, d)
                q_norm = getattr(module.self_attn, "q_norm", None)
                if q_norm is not None:
                    q = q_norm(q)
                q = q.mean(dim=1, keepdim=True)
                q = q.repeat(1, q_len, 1, 1).transpose(1, 2)

                k = module.self_attn.k_proj(h)
                k = k.view(1, k.shape[1], -1, d)
                k_norm = getattr(module.self_attn, "k_norm", None)
                if k_norm is not None:
                    k = k_norm(k)
                k = k.mean(dim=1, keepdim=True)
                k = k.repeat(1, q_len, 1, 1).transpose(1, 2)

                cos, sin = model.model.rotary_emb(h, position_ids.to(h.device))
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
                k = k.repeat_interleave(num_key_value_groups, dim=1)

                attn_weights = torch.matmul(q[:, :, -1:, :], k.transpose(2, 3)) / (d**0.5)
                attn_weights = attn_weights.softmax(dim=-1, dtype=torch.float32).squeeze()

                s = torch.cumsum(attn_weights, dim=1).mean(1)
                s = s.view(-1, num_key_value_groups).mean(1)

                scores[layer_idx] += s.cpu() / num_samples
    return scores.numpy()


@register_kv_compressor("duo_attention")
@dataclass
class DuoAttentionSketch(KVCompressor):
    """
    DuoAttention: Hybrid attention with retrieval and streaming heads.

    Splits attention heads into two types:
        - retrieval heads (use full KV cache) and
        - streaming heads (use only sink + recent tokens).
    Different heads have different attention patterns - some benefit from full context while others work well with
    limited context.

    Uses pre-computed attention patterns for supported models, falls back to
    on-the-fly computation for unsupported models.

    Based on DuoAttention (https://arxiv.org/abs/2410.10819).
    Port of kvpress 0.5.1 ``DuoAttentionPress`` (presses/duo_attention_press.py).

    Compression is virtual: ``compress`` returns keys/values unchanged and only sets
    ``module.masked_key_indices``; the globally patched attention functions
    (``eval_harness/sketch/attention_patch.py``) overwrite the masked cache entries with
    fake keys at decode time such that ``exp(<q, k>) == 0``. The physical ``DynamicCache``
    stays full-length and rectangular at every layer (no memory savings;
    ``compression_ratio_`` is reporting-only).

    Parameters
    ----------
    head_compression_ratio : float, default=0.0
        Fraction of (layer, kv-head) pairs converted to streaming heads
        (the globally lowest-scored ones, so per-layer budgets are uneven).
    on_the_fly_scoring : bool, default=False
        Whether to compute attention patterns on-the-fly using random samples.
        If True, computes patterns instead of loading pre-computed ones.
    attention_pattern : tuple, optional
        Direct injection of ``(sink_size, recent_size, head_scores)`` where ``head_scores``
        is array-like of shape ``[num_layers, num_kv_heads]``. Takes precedence over both
        the download path and on-the-fly scoring (offline runs and unit tests).
    pattern_dir : str, optional
        Local directory containing pre-fetched ``config.json`` and
        ``full_attention_heads.tsv`` (vendored DuoAttention pattern files), used instead
        of downloading from the DuoAttention repo (compute nodes may lack internet).
    compression_ratio_ : float
        Actual compression ratio achieved (computed during forward pass).
    recent_size : int
        Size of recent token window for streaming heads (determined automatically).
    sink_size : int
        Number of initial tokens preserved for streaming heads (determined automatically).
    streaming_mask : torch.Tensor
        Binary mask ``[num_layers, num_kv_heads]`` indicating which heads are streaming heads.

    Deviations from kvpress
    -----------------------
    - ``PATTERNS_DICT`` gains an alias for the renamed ``meta-llama/Llama-3.1-8B-Instruct``
      checkpoint (kvpress only lists the pre-rename ``Meta-Llama-3.1-8B-Instruct``).
    - ``attention_pattern`` / ``pattern_dir`` injection paths added so patterns can be
      supplied without network access; the kvpress download path is kept otherwise.
    - kvpress memoizes ``load_attention_pattern`` / ``duo_attention_on_the_fly`` with a
      cachetools LRU; here ``post_init_from_model`` early-returns when the pattern was
      already initialized for the same ``config.name_or_path`` (``KVCompressor.__call__``
      re-enters the sketch context once per sample). ``load_attention_pattern`` is an
      instance method (it reads ``pattern_dir``), not a cached staticmethod.
    - ``recent_size > 0`` is asserted after pattern loading: ``recent_size == 0`` would make
      the ``sink_size:-recent_size`` slice silently empty (never happens with real patterns).
    - On-the-fly qk-norm uses duck-typing on ``module.self_attn.q_norm/k_norm`` (the
      framework convention). Upstream's ``isinstance`` check targets the decoder *layer*
      (never an attention class), so the norm branch is dead code in kvpress; the
      duck-typed form applies the norms as intended for Qwen3/Gemma3-style models.
    - ``compression_ratio`` is a read-only property (not a dataclass field), so
      ``ResearchAdapter._build_sketch`` does not inject the adapter-level ratio; configure
      via ``sketch_kwargs: {head_compression_ratio: ...}``.

    Upstream quirks kept verbatim (do not "fix")
    --------------------------------------------
    - ``masked_key_indices`` is frozen at prefill: the recent window does NOT slide during
      decode and new decode tokens are never masked (kvpress deviation from the paper).
    - ``compression_ratio_`` goes negative when ``k_len < sink_size + recent_size``.
    - The decode-time wrapper writes fake keys in place, destroying the real keys at masked
      positions after the first post-prefill forward (those positions are permanently dead).
    - ``eager`` attention is rejected: eager dispatch bypasses the patched
      ``ALL_ATTENTION_FUNCTIONS``, so masking would be silently dropped (the research
      backend defaults to ``sdpa``, which is wrapped). ``hf_adapter._with_attention``
      override hooks register unwrapped functions and are likewise incompatible.
    """

    head_compression_ratio: float = 0.0
    on_the_fly_scoring: bool = False
    attention_pattern: Optional[Tuple[int, int, Any]] = None
    pattern_dir: Optional[str] = None
    compression_ratio_: Optional[float] = field(init=False, default=None)
    recent_size: Optional[int] = field(init=False, default=None)
    sink_size: Optional[int] = field(init=False, default=None)
    streaming_mask: Optional[torch.Tensor] = field(init=False, default=None)
    _pattern_key: Optional[str] = field(init=False, default=None, repr=False)

    def post_init_from_model(self, model: PreTrainedModel):
        """
        Initialize sink_size, recent_size, and streaming_mask from a model.
        """
        key = getattr(getattr(model, "config", None), "name_or_path", None)
        if self.streaming_mask is not None and self._pattern_key == key:
            return

        if self.attention_pattern is not None:
            sink_size, recent_size, head_scores = self.attention_pattern
            self.sink_size, self.recent_size = int(sink_size), int(recent_size)
            head_scores = np.asarray(head_scores, dtype=float)
        elif self.on_the_fly_scoring:
            self.sink_size, self.recent_size, head_scores = 128, 256, duo_attention_on_the_fly(model)
        else:
            self.sink_size, self.recent_size, head_scores = self.load_attention_pattern(model)

        assert self.recent_size > 0, "recent_size must be positive (a 0 window would mask through the sequence end)"

        n_pruned = round(head_scores.size * self.head_compression_ratio)
        self.streaming_mask = torch.zeros(head_scores.shape, dtype=bool, device=model.device)
        if n_pruned > 0:
            indices = np.argsort(head_scores, axis=None)[:n_pruned]
            self.streaming_mask[np.unravel_index(indices, head_scores.shape)] = True
        self._pattern_key = key

    @property
    def compression_ratio(self) -> float:
        assert self.compression_ratio_ is not None, "Forward pass must be run to compute the compression ratio"
        return self.compression_ratio_

    @compression_ratio.setter
    def compression_ratio(self, value):
        raise AttributeError(f"compression ratio cannot be set for {type(self).__name__}")

    def compress(
        self,
        module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        assert module.config._attn_implementation != "eager", "eager mode not supported"
        if self.streaming_mask is None:
            raise ValueError(
                "Streaming mask not initialized. Make sure to call post_init_from_model to initialize this press."
            )
        k_len = keys.shape[2]

        if (self.head_compression_ratio > 0) or (k_len > (self.sink_size + self.recent_size)):
            # Save indices to mask during the attention mechanism (see attention_patch.py).
            masked_keys = torch.zeros_like(keys[..., 0], dtype=torch.bool)
            masked_keys[:, self.streaming_mask[module.layer_idx], self.sink_size : -self.recent_size] = True
            module.masked_key_indices = torch.nonzero(masked_keys, as_tuple=True)

        # Compute the compression ratio
        self.compression_ratio_ = self.streaming_mask.float().mean().item()
        self.compression_ratio_ *= 1 - (self.sink_size + self.recent_size) / k_len

        return keys, values

    def load_attention_pattern(self, model: PreTrainedModel) -> Tuple[int, int, np.ndarray]:
        """
        Load the attention pattern from ``pattern_dir`` if set, else from the DuoAttention repo.
        """
        if self.pattern_dir is not None:
            pattern_dir = Path(self.pattern_dir)
            config = json.loads((pattern_dir / "config.json").read_text())
            head_scores = np.loadtxt(pattern_dir / "full_attention_heads.tsv", dtype=float, delimiter="\t")
        else:
            name = model.config.name_or_path
            assert name in PATTERNS_DICT, f"Checkpoint {name} not in {list(PATTERNS_DICT.keys())}"

            # Imported only on the download path: `requests` is not a dependency of the
            # minimal test environment, and the unknown-checkpoint assert must fire
            # without it.
            import requests  # type: ignore[import-untyped]

            url = f"{_BASE_URL}/{PATTERNS_DICT[name]}/"
            config = requests.get(url + "config.json").json()
            text = requests.get(url + "full_attention_heads.tsv").text
            head_scores = np.loadtxt(StringIO(text), dtype=float, delimiter="\t")

        # Clip as in duo_attn.utils.load_attn_pattern
        head_scores = np.clip(head_scores, 0, 1)

        return config["sink_size"], config["recent_size"], head_scores
