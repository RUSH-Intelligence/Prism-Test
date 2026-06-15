import logging
import math
from contextlib import contextmanager
from dataclasses import dataclass
from types import MethodType
from typing import Generator, List, Optional

import torch
from torch import nn
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase, QuantizedCache
from transformers.models.llama.modeling_llama import rotate_half

try:
    from transformers import Gemma3PreTrainedModel
except ImportError:
    Gemma3PreTrainedModel = None  # type: ignore[assignment]

from eval_harness.kv_compression.base import (
    SUPPORTED_MODELS,
    KVCompressor,
    _is_non_full_attention_layer,
)
from eval_harness.kv_compression.registry import register_kv_compressor
from eval_harness.kv_compression.utils import extract_keys_and_values

logger = logging.getLogger(__name__)


def _get_prerope_query_states(module: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    """Extract pre-RoPE query states ``[B, H_q, S, D]``.

    Duck-typed port of kvpress ``utils.get_prerope_query_states``: fused
    ``qkv_proj`` slice (Phi3-style) when present, ``q_proj`` otherwise, with an
    optional ``q_norm`` applied after the head reshape (Qwen3/Gemma3 qk-norm
    families). Kept module-local because ``eval_harness/sketch/utils.py`` is a
    shared file.
    """
    bsz, q_len, _ = hidden_states.shape
    num_heads = module.config.num_attention_heads
    head_dim = module.head_dim

    qkv_proj = getattr(module, "qkv_proj", None)
    if qkv_proj is not None:
        query_states = qkv_proj(hidden_states)[..., : num_heads * head_dim]
    elif hasattr(module, "q_proj"):
        query_states = module.q_proj(hidden_states)
    else:
        raise NotImplementedError(f"Sketch not yet implemented for {module.__class__}.")

    query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)

    q_norm = getattr(module, "q_norm", None)
    if q_norm is not None:
        query_states = q_norm(query_states)

    return query_states


@register_kv_compressor("kvzip")
@dataclass
class KVzipSketch(KVCompressor):
    """
    KVzip identifies the importance of KV pairs through context reconstruction,
    enabling effective query-agnostic KV cache compression.

    Port of kvpress ``KVzipPress`` (kvpress/presses/kvzip_press.py), itself the
    minimal-change variant of the original repository
    (https://github.com/snu-mllab/KVzip). Based on KVzip
    (https://arxiv.org/abs/2505.23416). Distinct from the already-registered
    ``kvzap`` sketch (KVzapSketch, https://arxiv.org/abs/2601.07891), which is a
    fast surrogate-model approximation of these scores.

    Lifecycle: the context manager wraps the normal context prefill (capturing
    ``input_ids`` and the cache pointer via a transient ``model.model.forward``
    wrapper). On exit, it chunks the context and prompts the model to
    *reconstruct* (repeat) each chunk against the prefilled cache; a gate-free
    forward hook scores every cached KV pair by the maximum cross-attention it
    receives from the repeat queries, then trims the cache back to the original
    context length after every layer of every pass. Finally ``compress_post``
    selects the bottom-scored pairs (globally across layers by default) and
    records them in ``module.masked_key_indices`` — *fake* compression consumed
    by ``eval_harness/kv_compression/attention_patch.py`` (fake keys with
    ``exp(<q, k>) = 0`` substituted on every ``q_len < k_len`` forward, reset on
    the next full prefill). The cache is never physically shrunk: zero memory
    savings (quality-only baseline), and logged cache lengths stay at the full
    context length. Requires a non-eager attention implementation (runner
    default is sdpa) and batch size 1 (``score_val`` hard-codes it). Scoring
    costs ``ceil(ctx / chunk_size)`` extra full-model forwards (the documented
    2-3x prefill overhead). Use with ``prefill_method: none``: methods that
    re-rotate keys (DCA, cyclic positions) or prune the cache during prefill
    (ReAttention-style) break the absolute-position queries and the
    ``start_idx``/``end_idx``/trim bookkeeping.

    Parameters
    ----------
    compression_ratio : float, default=0.0
        Fraction of key-value pairs to remove during compression.
    layerwise : bool, default=False
        Whether to enable uniform compression ratios across layers.
        When False, while the overall KV cache compression ratio is maintained,
        each layer has a different compression ratio.
    n_sink : int, default=4
        Number of initial tokens to preserve as attention sinks.
    kvzip_plus_normalization : bool, default=False
        Whether to enable KVzip+ normalization (attention rows divided by
        ``||h||``, columns multiplied by ``||Wo V||``).
    chunk_size : int, default=2048
        Reconstruction chunk length (kvpress hard-codes it as a ``prepare()``
        default).
    tokenizer : PreTrainedTokenizerBase, optional
        Injection hook for tests/offline runs. When ``None``, loaded once from
        ``model.config.name_or_path`` and cached on the instance.

    Deviations from kvpress
    -----------------------
    - ``num_logits_to_keep=1`` -> ``logits_to_keep=1`` (transformers 5.x kwarg).
    - ``get_prerope_query_states`` is inlined with duck-typing (``qkv_proj``
      presence, ``getattr(module, "q_norm", None)``) instead of isinstance
      checks against transformers attention classes.
    - The Gemma3 ``ValueError`` is extended to any model with non-full-attention
      layers (e.g. Qwen3.5 sliding/linear layers): ``score_val`` assumes every
      layer ``0..n_layers-1`` is hooked and scored, and an unscored layer would
      keep all-zero scores, so global mode would mask its entire cache.
    - Hook registration and ``compress_post`` resolve
      ``model.model.language_model`` when present (kvpress iterates
      ``model.model.layers``, which breaks on ConditionalGeneration wrappers).
    - The tokenizer is loaded once and cached on the instance (kvpress reloads
      from the hub on every context), with an injectable ``tokenizer`` field.
    - ``module.num_key_value_groups`` (KVzip+ path) is replaced with the locally
      computed group count for fake-module testability (same value).
    - ``model.model.forward`` is restored in a ``finally`` block (kvpress leaks
      the wrapper if prefill raises).
    - ``chunk_size`` is promoted from a ``prepare()`` keyword default to a
      dataclass field (default unchanged).

    Quirks kept for kvpress parity: the scoring softmax has no fp32 upcast and
    ``score_val`` lives in the model dtype (bf16 on GPU); ``prefix_length``
    comes from encoding the chat-template prefix alone and can differ by one
    BPE merge from the pipeline's single-pass encoding, leaving positions
    ``n_sink..prefix_length`` at score 0 (masked first in global mode); sink
    protection is not absolute under KVzip+ (scores can exceed 1.0); the
    attention patch overwrites pruned keys in place with fake keys (their
    originals are destroyed, harmless since they are pruned forever); kept
    indices are emitted in bottom-k order; ``int()`` truncation everywhere.
    """

    compression_ratio: float = 0.0
    layerwise: bool = False
    n_sink: int = 4
    kvzip_plus_normalization: bool = False
    chunk_size: int = 2048
    tokenizer: Optional[PreTrainedTokenizerBase] = None

    def __post_init__(self):
        super().__post_init__()
        assert 0 <= self.compression_ratio < 1, "Compression ratio must be between 0 and 1"
        logger.warning(
            "KVzipSketch requires multiple forward passes for chunked context reconstruction, "
            "resulting in a computational overhead of 2-3 times the initial prefilling cost. "
            "This significantly increases the overall prefilling time compared to other compression methods, "
            "which is inherent to the KVzip algorithm design."
        )
        self._reset_internal_parameters()

    def _reset_internal_parameters(self):
        self.context_length = 0
        self.prefix_length = 0

        self._suffix_ids = None
        self._context_ids = None
        self._cache = None

        self.score_val = None
        self.causal_mask_score = None
        self.start_idx = 0
        self.end_idx = 0

    def _resolve_tokenizer(self, model: PreTrainedModel) -> PreTrainedTokenizerBase:
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model.config.name_or_path)
        return self.tokenizer

    def _extract_prefix_suffix(self, tokenizer: PreTrainedTokenizerBase):
        if tokenizer.chat_template is None:
            prefix_text = ""
            suffix_text = "\n"  # Default suffix for models without chat template
        else:
            # Use a dummy context to extract the question suffix from the chat template
            dummy_context = "dummy context"
            separator = "\n" + "#" * len(dummy_context)
            temp_context = tokenizer.apply_chat_template(
                [{"role": "user", "content": dummy_context + separator}],
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            )
            context, suffix_text = temp_context.split(separator)
            prefix_text = context.split(dummy_context)[0]

        self.prefix_length = tokenizer.encode(prefix_text, return_tensors="pt", add_special_tokens=False).shape[-1]
        self._suffix_ids = tokenizer.encode(suffix_text, return_tensors="pt", add_special_tokens=False)

    @contextmanager
    def __call__(self, model: PreTrainedModel) -> Generator:
        """
        Context manager that handles both initial prefilling and KVzip scoring/compression.

        1. First yield: allows initial prefilling with the context.
        2. After yield: performs KVzip scoring and compression using context reconstruction.
        """
        if not isinstance(model, SUPPORTED_MODELS):
            logger.warning(f"Model {type(model)} not tested, supported models: {SUPPORTED_MODELS}")

        if Gemma3PreTrainedModel is not None and isinstance(model, Gemma3PreTrainedModel):
            raise ValueError("KVzipSketch is not supported for Gemma3 models")

        language_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
        if any(_is_non_full_attention_layer(layer) for layer in language_model.layers):
            raise ValueError(
                "KVzipSketch requires every layer to be a full-attention layer: score_val "
                "covers all layers, and an unhooked (sliding/linear) layer would keep "
                "all-zero scores, so the global budget would mask its entire cache."
            )

        tokenizer = self._resolve_tokenizer(model)
        self._extract_prefix_suffix(tokenizer)

        # Wrap the inner forward to capture the prefill input_ids and the cache pointer.
        original_forward = model.model.forward

        def wrapped_forward(model_self, *args, **kwargs):
            self._context_ids = kwargs["input_ids"]
            self._cache = kwargs["past_key_values"]
            return original_forward(*args, **kwargs)

        model.model.forward = MethodType(wrapped_forward, model.model)

        hooks = []
        try:
            try:
                yield
            finally:
                model.model.forward = original_forward  # Restore original

            # After yield: KVzip scoring and compression phase
            if self.compression_ratio > 0 and self._context_ids is not None:
                for layer in language_model.layers:
                    layer.self_attn.rotary_emb = language_model.rotary_emb
                    hooks.append(layer.self_attn.register_forward_hook(self.forward_hook, with_kwargs=True))

                self._perform_kvzip_compression(model, tokenizer)
        finally:
            for hook in hooks:
                hook.remove()
            self._reset_internal_parameters()

    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        """
        Override the forward_hook of KVCompressor.

        Deliberately gate-free (no ``_is_decoding_step``): the reconstruction
        passes look like decode steps (``cache_position[-1] > q_len``) and would
        be misclassified; the hooks only exist during the scoring phase, so a
        gate is also unnecessary. KVzip only calculates importance scores here
        and trims the cache back to the original context length; compression
        happens later in ``compress_post``.
        """
        hidden_states = kwargs["hidden_states"]
        cache = kwargs.get("past_key_values", None) or kwargs.get("past_key_value", None)
        cache_layer = cache.layers[module.layer_idx]

        keys, values = extract_keys_and_values(cache, module.layer_idx)

        # Compute importance scores for KV pairs in the prefilled context,
        # retaining only the originally prefilled KV pairs.
        keys, values = self.score_kvzip(module, hidden_states, keys, values, output[1], kwargs)

        if isinstance(cache, QuantizedCache):
            cache_layer._quantized_keys = cache_layer._quantize(keys, axis=cache_layer.axis_key)
            cache_layer._quantized_values = cache_layer._quantize(values, axis=cache_layer.axis_value)
            cache_layer.keys = torch.zeros(0, dtype=keys.dtype, device=keys.device)
            cache_layer.values = torch.zeros(0, dtype=keys.dtype, device=keys.device)
            cache_layer.cumulative_length = keys.shape[2]
        else:
            cache_layer.keys = keys
            cache_layer.values = values

        return output

    def _perform_kvzip_compression(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase):
        """Perform the KVzip scoring and compression algorithm."""
        # Prepare chunked inputs for context reconstruction
        self.context_length = self._context_ids.shape[1]
        chunked_context_pairs = self.prepare(model, tokenizer)

        # Perform scoring through context reconstruction,
        # using the cache stored during the initial forward pass.
        self.start_idx = self.prefix_length
        for prefill_ids, repeat_ids in chunked_context_pairs:
            self.end_idx = self.start_idx + prefill_ids.shape[1]
            model(
                input_ids=repeat_ids.to(model.device),
                past_key_values=self._cache,
                logits_to_keep=1,
            )
            self.start_idx = self.end_idx

        # Perform final compression
        self.compress_post(model)

    def _chunk_fn(self, ctx_ids: torch.Tensor, chunk_size: int) -> List[torch.Tensor]:
        """Chunk input tokens."""
        ctx_len = ctx_ids.shape[1]
        if ctx_len > chunk_size:
            chunk_num = (ctx_len - 1) // chunk_size + 1

            chunked_input_ids = []
            for i in range(chunk_num):
                start = i * chunk_size
                end = (i + 1) * chunk_size
                a_ids = ctx_ids[:, start:end]
                if a_ids.shape[1] == 0:
                    continue
                chunked_input_ids.append(a_ids)
        else:
            chunked_input_ids = [ctx_ids]

        return chunked_input_ids

    def prepare(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        prev_postfix_size: int = 8,
    ) -> List[tuple[torch.Tensor, torch.Tensor]]:
        """Prepare chunked inputs for KV importance scoring with context reconstruction."""
        ctx_ids = self._context_ids[:, self.prefix_length :].to("cpu")

        # initialize score values
        self.score_val = torch.zeros(
            (
                model.config.num_hidden_layers,
                1,
                model.config.num_key_value_heads,
                self.context_length,
            ),  # only support batch size of 1
            dtype=model.dtype,
            device=model.device,
        )
        self.score_val[..., : self.n_sink] = 1.0

        chunked_context_pairs = []
        chunked_input_ids = self._chunk_fn(ctx_ids, self.chunk_size)
        for i, a_ids in enumerate(chunked_input_ids):
            if i == 0:
                prompt = "\n\nRepeat the previous context exactly."
                q_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)
            else:
                prompt = "\n\nRepeat the part of the previous context exactly, starting with"
                q_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)
                postfix_prev = chunked_input_ids[i - 1][:, -prev_postfix_size:]
                q_ids = torch.cat([q_ids, postfix_prev], dim=1)

            chunked_context_pairs.append((a_ids, torch.cat([q_ids, self._suffix_ids, a_ids], dim=1)))

        return chunked_context_pairs

    def _make_mask(self, attn_weights: torch.Tensor, window_size: int):
        """Define causal mask shared across layers."""
        mask = torch.full((window_size, window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        self.causal_mask_score = mask[None, None, None, :, :]

    def _mask_causal(self, attn_weights: torch.Tensor, window_size: int):
        """Apply causal masking to the trailing window_size x window_size block."""
        if self.causal_mask_score is None:
            self._make_mask(attn_weights, window_size)
        elif self.causal_mask_score.size(-1) != window_size:
            self._make_mask(attn_weights, window_size)

        self.causal_mask_score = self.causal_mask_score.to(attn_weights.device)
        attn_weights[..., -window_size:, -window_size:] += self.causal_mask_score

    def score_kvzip(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the maximum cross-attention scores during context reconstruction,
        and return slices of the keys and values containing only the originally
        prefilled KV pairs, i.e., excluding KV pairs from repeated contexts.
        The computed scores are stored in ``self.score_val``.
        """
        bsz, q_len, _ = hidden_states.shape
        num_heads = module.config.num_attention_heads
        num_heads_kv = module.config.num_key_value_heads
        head_dim = module.head_dim
        num_key_value_groups = num_heads // num_heads_kv

        queries = _get_prerope_query_states(module, hidden_states)

        # Apply RoPE
        cos, sin = kwargs["position_embeddings"]
        queries = (queries * cos.unsqueeze(1)) + (rotate_half(queries) * sin.unsqueeze(1))
        queries = queries.view(bsz, num_heads_kv, num_key_value_groups, q_len, head_dim)

        # Subsample keys
        sink = min(self.n_sink, self.start_idx)
        ctx_len = self.end_idx - self.start_idx
        keys_subsampled = torch.cat(
            [
                keys[:, :, :sink],  # attention sink tokens (generally system prompt)
                keys[:, :, self.start_idx : self.end_idx],  # KV chunk in the cache
                keys[:, :, -q_len:],  # KV repeat chunk
            ],
            dim=2,
        )
        keys_subsampled = keys_subsampled.unsqueeze(2).transpose(-2, -1).contiguous()

        # Compute attention (no fp32 upcast: kvpress parity)
        attn_weights = torch.matmul(queries, keys_subsampled) / math.sqrt(head_dim)
        self._mask_causal(attn_weights, q_len)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if self.kvzip_plus_normalization:
            # Divide by ||h|| (by row)
            h_norm = torch.norm(hidden_states, dim=-1)
            attn_weights = torch.einsum("b h g t i, b t -> b h g t i", attn_weights, 1 / h_norm)

            # Multiply by ||WoV|| (by column)
            Wo = module.o_proj.weight.transpose(0, 1)
            Wo = Wo.view(num_heads_kv, num_key_value_groups, module.head_dim, module.config.hidden_size)
            values_subsampled = torch.cat(
                [values[:, :, :sink], values[:, :, self.start_idx : self.end_idx], values[:, :, -q_len:]], dim=2
            )
            values_subsampled = values_subsampled.unsqueeze(2).transpose(-2, -1).contiguous()
            V = values_subsampled.repeat_interleave(num_key_value_groups, dim=2)
            WoV_norm = torch.einsum("h g i j, b h g i t -> b h g t j", Wo, V).norm(dim=-1)
            attn_weights = torch.einsum("b h g i t, b h g t -> b h g i t", attn_weights, WoV_norm)

        attn_weights = attn_weights[..., sink : sink + ctx_len]
        scores = attn_weights.amax(dim=(-3, -2))  # max over group, q

        layer_idx = int(module.layer_idx)
        self.score_val[layer_idx][..., self.start_idx : self.end_idx] = scores  # update score

        # Retain the originally prefilled context KV pairs and exclude KV pairs from the repeated context
        keys, values = keys[:, :, : self.context_length], values[:, :, : self.context_length]
        return keys, values

    def compress_post(self, model: PreTrainedModel):
        """
        Obtain the indices of KV pairs to be evicted.
        Adopted from adakv_press.compress (fake compression). KVzip does not rely on safeguards.
        """
        if self.compression_ratio > 0:
            n_layer, bsz, num_key_value_heads, ctx_len = self.score_val.shape

            # calculate the pruned KV pairs across layers
            if self.layerwise:
                nl = int(bsz * num_key_value_heads * ctx_len * self.compression_ratio)
                n_pruned_layers = nl * torch.ones(n_layer, device=self.score_val.device, dtype=torch.int)
            else:
                n_pruned_indices = int(self.score_val.numel() * self.compression_ratio)
                pruned_indices = torch.topk(-self.score_val.reshape(-1), n_pruned_indices).indices
                n_tokens_per_layer = bsz * num_key_value_heads * ctx_len
                n_pruned_layers = torch.bincount(pruned_indices // n_tokens_per_layer, minlength=n_layer).int()

            language_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
            for layer in language_model.layers:
                module = layer.self_attn
                layer_idx = int(module.layer_idx)

                assert module.config._attn_implementation != "eager", "eager mode not supported"

                scores = self.score_val[layer_idx]

                # Compute bottom-k across heads
                n_pruned = n_pruned_layers[layer_idx].cpu()
                indices = torch.topk(-scores.reshape(bsz, -1), n_pruned, dim=1).indices.flatten().cpu()

                # Save indices to mask during the attention mechanism. See attention_patch.py for details
                batch_indices = torch.arange(bsz, device=n_pruned.device).repeat_interleave(n_pruned)
                head_indices = indices // ctx_len
                seq_indices = indices % ctx_len
                module.masked_key_indices = (batch_indices, head_indices, seq_indices)
