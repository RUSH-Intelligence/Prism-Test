from __future__ import annotations

import logging
from dataclasses import dataclass, fields
from typing import Any, List, Optional

from .attention_methods import (
    AttentionMethod,
    AttentionPhase,
    available_attention_methods,
    get_attention_method,
)
from .hf_adapter import HFAdapter, HFGenerateConfig
from .kv_compression import (
    DecodingSketch,
    KnormSketch,
    KVCompressor,
    PrefillDecodingSketch,
    get_kv_compressor_class,
)
from .kv_compression.cache_adapter import CacheAdapter, create_cache_adapter
from .positional_methods import PositionalMethod, get_positional_method
from .prefill_methods import get_prefill_method
from .research_pipeline import SketchTextGenerationPipeline

logger = logging.getLogger(__name__)


@dataclass
class ResearchConfig:
    """Three-door research configuration.

    Each door is an independent, optional lever (``none`` = off):

    * **Door 1 — positional** (``positional_method``): RoPE frequency / position
      remap — ``yarn`` | ``ntk`` | ``linear_pi``.
    * **Door 2 — attention** (``attention_method``): attention-math replacement —
      ``dca`` | ``reattention_exact``.  ``attention_phase`` (prefill | decode |
      both) gates when it is active.
    * **Door 3 — KV compression** (``kv_compressor``): post-attention cache
      rewrite — any registered compressor (``knorm``, ``random``, ``snapkv``, …).
      ``compression_schedule`` (streaming | post_prefill | decode) gates when it
      fires; ``compression_ratio`` is the convenience prune fraction.

    ``prefill_chunk_size`` (``None`` = single pass) drives the chunked,
    memory-bounded prefill that ``streaming`` compressors hook into.
    """

    # Door 1 — positional.
    positional_method: str = "none"
    positional_method_kwargs: Optional[dict] = None

    # Door 2 — attention.
    attention_method: str = "none"
    attention_method_kwargs: Optional[dict] = None
    attention_phase: str = "both"

    # Door 3 — KV compression.
    kv_compressor: str = "none"
    kv_compressor_kwargs: Optional[dict] = None
    compression_ratio: float = 0.0
    compression_schedule: Optional[Any] = None  # str | list[str]; None = compressor default
    compression_interval: int = 512       # decoding-timing compressors
    target_size: int = 2048               # decoding-timing compressors
    hidden_states_buffer_size: int = 256  # decoding-timing compressors

    # Prefill / runtime.
    prefill_chunk_size: Optional[int] = None
    max_context_length: Optional[int] = None
    log_cache_seq_len: bool = False


class ResearchAdapter(HFAdapter):
    """Three-door research backend.

    Flow:
    1. Prefill the context (optionally chunked) through the model's forward,
       with the positional method (door 1) wrapping the rotary embedding, the
       attention method (door 2) replacing ``self_attn.forward``, and the KV
       compressor (door 3) rewriting the cache after each hooked layer.
    2. Decode greedily from the resulting cache.
    """

    def __init__(
        self,
        model: str,
        dtype: str = "auto",
        trust_remote_code: bool = True,
        seed: int = 42,
        max_model_len: Optional[int] = None,
        research_config: Optional[ResearchConfig] = None,
        **model_kwargs: Any,
    ) -> None:
        self._cfg = research_config or ResearchConfig()

        requested_ctx = self._cfg.max_context_length or max_model_len

        super().__init__(
            model=model,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            seed=seed,
            max_model_len=max_model_len,
            **model_kwargs,
        )

        self._max_context_length = requested_ctx
        self._positional_method: Optional[PositionalMethod] = self._build_positional_method(self._cfg)
        self._attention_method = self._build_attention_method(self._cfg)
        self._kv_compressor: Optional[KVCompressor] = self._build_kv_compressor(self._cfg)
        self._cache_adapter: CacheAdapter = create_cache_adapter(self._model)
        self._pipe = SketchTextGenerationPipeline(model=self._model, tokenizer=self._tokenizer)

        logger.info(
            "ResearchAdapter initialized: positional=%s attention=%s(phase=%s) "
            "kv_compressor=%s compression_ratio=%.3f prefill_chunk_size=%s "
            "max_context_length=%s",
            self._cfg.positional_method,
            self._cfg.attention_method,
            self._cfg.attention_phase,
            self._cfg.kv_compressor,
            self._cfg.compression_ratio,
            self._cfg.prefill_chunk_size,
            self._max_context_length,
        )

    # ------------------------------------------------------------------
    # Door builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_positional_method(cfg: ResearchConfig) -> Optional[PositionalMethod]:
        name = (cfg.positional_method or "none").strip().lower()
        if name in {"none", "default", "standard", "native"}:
            return None
        return get_positional_method(name, **dict(cfg.positional_method_kwargs or {}))

    @staticmethod
    def _build_attention_method(cfg: ResearchConfig):
        """Resolve the attention method (door 2).

        Tries the new ``attention_methods`` registry first (DCA), then falls
        back to the legacy ``prefill_methods`` registry (reattention_exact, the
        ReAttention-prune method) — both install via the pipeline's method slot.
        ``attention_phase`` is applied to native :class:`AttentionMethod`
        instances.
        """
        name = (cfg.attention_method or "none").strip().lower()
        if name in {"none", "default", "standard"}:
            return None
        kw = dict(cfg.attention_method_kwargs or {})
        if name in available_attention_methods():
            method = get_attention_method(name, **kw)
            method.phase = AttentionPhase.coerce(cfg.attention_phase)
            return method
        # Legacy faithful methods still living in prefill_methods/.
        return get_prefill_method(name, **kw)

    def _build_kv_compressor(self, cfg: ResearchConfig) -> Optional[KVCompressor]:
        name = (cfg.kv_compressor or "none").strip().lower()
        if name in {"none", "no_sketch", "no_press"}:
            return None
        if name in {"decoding_knorm", "decoding"}:
            return DecodingSketch(
                base_sketch=KnormSketch(compression_ratio=cfg.compression_ratio),
                compression_interval=cfg.compression_interval,
                target_size=cfg.target_size,
                hidden_states_buffer_size=cfg.hidden_states_buffer_size,
            )
        if name in {"prefill_decoding_knorm", "prefill_decoding"}:
            return PrefillDecodingSketch(
                prefilling_sketch=KnormSketch(compression_ratio=cfg.compression_ratio),
                decoding_sketch=DecodingSketch(
                    base_sketch=KnormSketch(compression_ratio=cfg.compression_ratio),
                    compression_interval=cfg.compression_interval,
                    target_size=cfg.target_size,
                    hidden_states_buffer_size=cfg.hidden_states_buffer_size,
                ),
            )
        cls = get_kv_compressor_class(name)
        kw = dict(cfg.kv_compressor_kwargs or {})
        field_names = {f.name for f in fields(cls)}
        if "compression_ratio" in field_names:
            kw.setdefault("compression_ratio", cfg.compression_ratio)
        if cfg.compression_schedule is not None and "schedule" in field_names:
            kw.setdefault("schedule", cfg.compression_schedule)
        return cls(**kw)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, prompts: List[str], gen_cfg: HFGenerateConfig) -> List[str]:
        texts: List[str] = []
        cache_adapter = getattr(self, "_cache_adapter", None)
        for prompt in prompts:
            cache = cache_adapter.initialize_cache(None) if (self._cfg.log_cache_seq_len and cache_adapter) else None
            output = self._pipe(
                prompt,
                question="",
                positional_method=self._positional_method,
                prefill_method=self._attention_method,
                sketch=self._kv_compressor,
                prefill_chunk_size=self._cfg.prefill_chunk_size,
                max_new_tokens=gen_cfg.max_tokens,
                max_context_length=self._max_context_length,
                cache=cache,
                cache_adapter=cache_adapter,
            )
            texts.append(output["answer"])

            if cache is not None and cache_adapter is not None:
                self._log_cache_seq_lengths(cache, cache_adapter)

        return texts

    def generate_for_context(
        self,
        context: str,
        questions: List[str],
        answer_prefix: str,
        gen_cfg: HFGenerateConfig,
    ) -> List[str]:
        cache_adapter = getattr(self, "_cache_adapter", None)
        cache = cache_adapter.initialize_cache(None) if (self._cfg.log_cache_seq_len and cache_adapter) else None
        output = self._pipe(
            context,
            questions=questions,
            answer_prefix=answer_prefix,
            positional_method=self._positional_method,
            prefill_method=self._attention_method,
            sketch=self._kv_compressor,
            prefill_chunk_size=self._cfg.prefill_chunk_size,
            max_new_tokens=gen_cfg.max_tokens,
            max_context_length=self._max_context_length,
            cache=cache,
            cache_adapter=cache_adapter,
        )
        answers = output["answers"]

        if cache is not None and cache_adapter is not None:
            self._log_cache_seq_lengths(cache, cache_adapter)

        return answers

    @staticmethod
    def _log_cache_seq_lengths(cache, cache_adapter: CacheAdapter) -> None:
        layer_seq_lengths = []
        for layer_idx in range(len(cache)):
            try:
                layer_seq_lengths.append(cache.get_seq_length(layer_idx))
            except Exception:
                # Hybrid caches include linear-attention layers without seq-length semantics.
                continue

        logger.info(
            "Cache sequence length: global=%s, per_layer_min=%s, per_layer_max=%s",
            cache_adapter.get_seq_length(cache),
            min(layer_seq_lengths) if layer_seq_lengths else 0,
            max(layer_seq_lengths) if layer_seq_lengths else 0,
        )

    @property
    def research_config(self) -> ResearchConfig:
        return self._cfg

    @research_config.setter
    def research_config(self, cfg: ResearchConfig) -> None:
        self._cfg = cfg
        self._positional_method = self._build_positional_method(cfg)
        self._attention_method = self._build_attention_method(cfg)
        self._kv_compressor = self._build_kv_compressor(cfg)
        self._max_context_length = cfg.max_context_length or self._max_context_length
