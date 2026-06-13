from __future__ import annotations

import logging
from dataclasses import dataclass, fields
from typing import Any, List, Optional

from .hf_adapter import HFAdapter, HFGenerateConfig
from .prefill_methods import PrefillMethod, get_prefill_method
from .kv_compression import (
    KVCompressor,
    DecodingSketch,
    KnormSketch,
    PrefillDecodingSketch,
    get_kv_compressor_class,
)
from .kv_compression.cache_adapter import CacheAdapter, create_cache_adapter
from .research_pipeline import SketchTextGenerationPipeline

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    # Sketch/KV-compression controls.
    sketch_name: str = "none"
    sketch_kwargs: Optional[dict] = None
    compression_ratio: float = 0.0
    compression_interval: int = 512
    target_size: int = 2048
    hidden_states_buffer_size: int = 256
    max_context_length: Optional[int] = None
    log_cache_seq_len: bool = False

    # Prefill attention method (context extrapolation).
    prefill_method: str = "none"
    prefill_method_kwargs: Optional[dict] = None


class ResearchAdapter(HFAdapter):
    """KVPress-style adapter using Sketch pipeline semantics.

    Flow:
    1. Prefill full context (with optional sketch compression hooks)
    2. Decode greedy answers using the resulting cache
    3. Optionally compress during decoding when using decoding sketches
    """

    def __init__(
        self,
        model: str,
        dtype: str = "auto",
        trust_remote_code: bool = True,
        seed: int = 42,
        max_model_len: Optional[int] = None,
        cache_config: Optional[CacheConfig] = None,
        rope_method: str = "native",
        rope_scale_factor: float = 1.0,
        **model_kwargs: Any,
    ) -> None:
        del rope_method, rope_scale_factor

        self._cache_cfg = cache_config or CacheConfig()

        requested_ctx = self._cache_cfg.max_context_length or max_model_len

        super().__init__(
            model=model,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            seed=seed,
            max_model_len=max_model_len,
            **model_kwargs,
        )

        self._max_context_length = requested_ctx
        self._sketch: Optional[KVCompressor] = self._build_sketch(self._cache_cfg)
        self._prefill_method: PrefillMethod = self._build_prefill_method(self._cache_cfg)
        self._cache_adapter: CacheAdapter = create_cache_adapter(self._model)
        self._pipe = SketchTextGenerationPipeline(model=self._model, tokenizer=self._tokenizer)

        logger.info(
            "ResearchAdapter initialized: sketch=%s compression_ratio=%.3f "
            "prefill_method=%s max_context_length=%s",
            self._cache_cfg.sketch_name,
            self._cache_cfg.compression_ratio,
            self._cache_cfg.prefill_method,
            self._max_context_length,
        )

    @staticmethod
    def _build_prefill_method(cfg: CacheConfig) -> PrefillMethod:
        name = (cfg.prefill_method or "none").strip().lower()
        kw = dict(cfg.prefill_method_kwargs or {})
        return get_prefill_method(name, **kw)

    def _build_sketch(self, cfg: CacheConfig) -> Optional[KVCompressor]:
        name = (cfg.sketch_name or "none").strip().lower()
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
        # Everything else resolves through the sketches registry; the
        # adapter-level compression_ratio is applied unless the sketch does
        # not declare that field or sketch_kwargs overrides it.
        cls = get_kv_compressor_class(name)
        kw = dict(cfg.sketch_kwargs or {})
        if "compression_ratio" in {f.name for f in fields(cls)}:
            kw.setdefault("compression_ratio", cfg.compression_ratio)
        return cls(**kw)

    def generate(self, prompts: List[str], gen_cfg: HFGenerateConfig) -> List[str]:
        texts: List[str] = []
        cache_adapter = getattr(self, "_cache_adapter", None)
        for prompt in prompts:
            cache = cache_adapter.initialize_cache(None) if (self._cache_cfg.log_cache_seq_len and cache_adapter) else None
            output = self._pipe(
                prompt,
                question="",
                sketch=self._sketch,
                prefill_method=self._prefill_method,
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
        cache = cache_adapter.initialize_cache(None) if (self._cache_cfg.log_cache_seq_len and cache_adapter) else None
        output = self._pipe(
            context,
            questions=questions,
            answer_prefix=answer_prefix,
            sketch=self._sketch,
            prefill_method=self._prefill_method,
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
            "Sketch cache sequence length: global=%s, per_layer_min=%s, per_layer_max=%s",
            cache_adapter.get_seq_length(cache),
            min(layer_seq_lengths) if layer_seq_lengths else 0,
            max(layer_seq_lengths) if layer_seq_lengths else 0,
        )

    @property
    def cache_config(self) -> CacheConfig:
        return self._cache_cfg

    @cache_config.setter
    def cache_config(self, cfg: CacheConfig) -> None:
        self._cache_cfg = cfg
        self._sketch = self._build_sketch(cfg)
        self._prefill_method = self._build_prefill_method(cfg)
        self._max_context_length = cfg.max_context_length or self._max_context_length
