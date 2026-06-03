from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional

from transformers import DynamicCache

from .hf_adapter import HFAdapter, HFGenerateConfig
from .sketch import (
    BaseSketch,
    DecodingSketch,
    KnormSketch,
    PrefillDecodingSketch,
    ReAttentionSketch,
    RandomSketch,
    SketchTextGenerationPipeline,
)

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    # Legacy fields retained for compatibility with existing configs.
    global_size: int = 32
    local_size: int = 512
    mid_budget: int = 256
    span_size: int = 16
    selection: str = "qkv2"
    chunk_size: int = 512

    # Sketch/KV-compression controls.
    sketch_name: str = "none"
    compression_ratio: float = 0.0
    compression_interval: int = 512
    target_size: int = 2048
    hidden_states_buffer_size: int = 256
    max_context_length: Optional[int] = None
    log_cache_seq_len: bool = False


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
        self._sketch: Optional[BaseSketch] = self._build_sketch(self._cache_cfg)
        self._pipe = SketchTextGenerationPipeline(model=self._model, tokenizer=self._tokenizer)

        logger.info(
            "ResearchAdapter initialized: sketch=%s compression_ratio=%.3f max_context_length=%s",
            self._cache_cfg.sketch_name,
            self._cache_cfg.compression_ratio,
            self._max_context_length,
        )

    def _build_sketch(self, cfg: CacheConfig) -> Optional[BaseSketch]:
        name = (cfg.sketch_name or "none").strip().lower()
        if name in {"none", "no_sketch", "no_press"}:
            return None
        if name in {"knorm", "knorm_sketch"}:
            return KnormSketch(compression_ratio=cfg.compression_ratio)
        if name in {"reattention", "reattention_sketch"}:
            return ReAttentionSketch(compression_ratio=cfg.compression_ratio)
        if name in {"random", "random_sketch"}:
            return RandomSketch(compression_ratio=cfg.compression_ratio)
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
        raise ValueError(f"Unknown sketch_name '{cfg.sketch_name}'.")

    def generate(self, prompts: List[str], gen_cfg: HFGenerateConfig) -> List[str]:
        texts: List[str] = []
        for prompt in prompts:
            cache = DynamicCache() if self._cache_cfg.log_cache_seq_len else None
            output = self._pipe(
                prompt,
                question="",
                sketch=self._sketch,
                max_new_tokens=gen_cfg.max_tokens,
                max_context_length=self._max_context_length,
                cache=cache,
            )
            texts.append(output["answer"])

            if cache is not None:
                layer_seq_lengths = [cache.get_seq_length(layer_idx) for layer_idx in range(len(cache))]
                logger.info(
                    "Sketch cache sequence length: global=%s, per_layer_min=%s, per_layer_max=%s",
                    cache.get_seq_length(),
                    min(layer_seq_lengths) if layer_seq_lengths else 0,
                    max(layer_seq_lengths) if layer_seq_lengths else 0,
                )

        return texts

    def generate_for_context(
        self,
        context: str,
        questions: List[str],
        answer_prefix: str,
        gen_cfg: HFGenerateConfig,
    ) -> List[str]:
        cache = DynamicCache() if self._cache_cfg.log_cache_seq_len else None
        output = self._pipe(
            context,
            questions=questions,
            answer_prefix=answer_prefix,
            sketch=self._sketch,
            max_new_tokens=gen_cfg.max_tokens,
            max_context_length=self._max_context_length,
            cache=cache,
        )
        answers = output["answers"]

        if cache is not None:
            layer_seq_lengths = [cache.get_seq_length(layer_idx) for layer_idx in range(len(cache))]
            logger.info(
                "Sketch cache sequence length: global=%s, per_layer_min=%s, per_layer_max=%s",
                cache.get_seq_length(),
                min(layer_seq_lengths) if layer_seq_lengths else 0,
                max(layer_seq_lengths) if layer_seq_lengths else 0,
            )

        return answers

    @property
    def cache_config(self) -> CacheConfig:
        return self._cache_cfg

    @cache_config.setter
    def cache_config(self, cfg: CacheConfig) -> None:
        self._cache_cfg = cfg
        self._sketch = self._build_sketch(cfg)
        self._max_context_length = cfg.max_context_length or self._max_context_length
