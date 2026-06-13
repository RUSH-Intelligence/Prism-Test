import contextlib
import logging
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, Cache, Pipeline
from transformers.pipelines import PIPELINE_REGISTRY
from transformers.pipelines.base import GenericTensor

from eval_harness.attention_methods._method_base import PrefillMethod
from eval_harness.kv_compression.cache_adapter import CacheAdapter, create_cache_adapter
from eval_harness.kv_compression.base import KVCompressor
from eval_harness.kv_compression.compressors.decoding_sketch import DecodingSketch
from eval_harness.kv_compression.compressors.prefill_decoding_sketch import PrefillDecodingSketch

logger = logging.getLogger(__name__)


class ResearchGenerationPipeline(Pipeline):
    def _sanitize_parameters(
        self,
        question: Optional[str] = None,
        questions: Optional[list[str]] = None,
        answer_prefix: Optional[str] = None,
        kv_compressor: Optional[KVCompressor] = None,
        attention_method: Optional[PrefillMethod] = None,
        positional_method=None,
        max_new_tokens: int = 50,
        max_context_length: Optional[int] = None,
        prefill_chunk_size: Optional[int] = None,
        enable_thinking: bool = False,
        cache: Optional[Cache] = None,
        cache_adapter: Optional[CacheAdapter] = None,
        **kwargs,
    ):
        answer_prefix = answer_prefix or ""
        postprocess_kwargs = {"single_question": questions is None}
        assert question is None or questions is None, "Either question or questions should be provided, not both."
        questions = questions or ([question] if question else [""])
        if max_context_length is None:
            max_context_length = min(self.tokenizer.model_max_length, int(1e10))
        preprocess_kwargs = {
            "questions": questions,
            "answer_prefix": answer_prefix,
            "max_context_length": max_context_length,
            "enable_thinking": enable_thinking,
        }
        forward_kwargs = {
            "kv_compressor": kv_compressor,
            "attention_method": attention_method,
            "positional_method": positional_method,
            "max_new_tokens": max_new_tokens,
            "prefill_chunk_size": prefill_chunk_size,
            "cache": cache,
            "cache_adapter": cache_adapter,
        }
        return preprocess_kwargs, forward_kwargs, postprocess_kwargs

    def preprocess(
        self,
        context: str,
        questions: list[str],
        answer_prefix: str,
        max_context_length: int,
        enable_thinking: bool = False,
    ):
        if self.tokenizer.chat_template is None:
            bos_token = getattr(self.tokenizer, "bos_token", "")
            context = bos_token + context
            question_suffix = "\n"
        else:
            separator = "#" * (len(context) + 10)
            context = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": context + separator}],
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=enable_thinking,
            )
            context, question_suffix = context.split(separator)

        questions = [question + question_suffix + answer_prefix for question in questions]

        context_ids = self.tokenizer.encode(context, return_tensors="pt", add_special_tokens=False)
        question_ids = [
            self.tokenizer.encode(question, return_tensors="pt", add_special_tokens=False) for question in questions
        ]

        if context_ids.shape[1] > max_context_length:
            logger.warning(
                f"Context length has been truncated from {context_ids.shape[1]} to {max_context_length} tokens."
            )
            context_ids = context_ids[:, :max_context_length]

        return {"context_ids": context_ids, "questions_ids": question_ids}

    def _forward(
        self,
        input_tensors: dict[str, GenericTensor],
        max_new_tokens: int = 50,
        kv_compressor: Optional[KVCompressor] = None,
        attention_method: Optional[PrefillMethod] = None,
        positional_method=None,
        prefill_chunk_size: Optional[int] = None,
        cache: Optional[Cache] = None,
        cache_adapter: Optional[CacheAdapter] = None,
    ):
        if isinstance(kv_compressor, (DecodingSketch, PrefillDecodingSketch)) and len(input_tensors["questions_ids"]) > 1:
            raise ValueError("DecodingSketch is not compatible with multiple questions. Please specify one question.")

        context_ids = input_tensors["context_ids"].to(self.model.device)
        context_length = context_ids.shape[1]

        cache_adapter = cache_adapter or create_cache_adapter(self.model)
        cache = cache_adapter.initialize_cache(cache)

        # Determine whether to use a non-trivial attention method (door 2).
        use_attention_method = (
            attention_method is not None
            and type(attention_method) is not PrefillMethod  # not the base no-op
        )

        # Door 1 (positional) wraps the rotary embedding for the WHOLE run
        # (prefill + decode), so it is the outermost context.  An attention
        # method that computes its own RoPE (DCA) overrides it for its layers.
        positional_ctx = (
            positional_method(self.model)
            if positional_method is not None
            else contextlib.nullcontext()
        )

        perform_prefill_compression = kv_compressor is not None and not isinstance(kv_compressor, DecodingSketch)

        # Hook ordering: attention-method hooks install FIRST (outermost context
        # manager), then KV-compressor hooks install SECOND.  Since forward hooks
        # fire in registration order, method hooks run before compressor hooks.
        #
        #   model forward → method hook (select/restructure) → compressor hook (compress)
        attention_ctx = attention_method(self.model) if use_attention_method else contextlib.nullcontext()
        compressor_ctx = kv_compressor(self.model) if perform_prefill_compression else contextlib.nullcontext()

        # The attention-method context stays open across BOTH prefill and decode.
        # Methods that intercept the attention computation (e.g. DCA, which
        # replaces self_attn.forward) must remain active during decode; methods
        # that only prune on prefill (e.g. ReAttention) no-op on decode steps.
        with positional_ctx, attention_ctx:
            if use_attention_method:
                attention_method.on_prefill_start(context_length)
            with compressor_ctx:
                # Tell the compressor we are prefilling so it does not rely on
                # the cache_position heuristic, which mislabels non-first
                # chunked-prefill chunks as decode (and would then skip a
                # ``streaming`` compressor after the first chunk).
                if kv_compressor is not None:
                    kv_compressor.set_phase("prefill")
                self._run_prefill(
                    context_ids=context_ids,
                    cache=cache,
                    prefill_chunk_size=prefill_chunk_size,
                )
                cache_adapter.maybe_slice_prefill(cache)

                logger.debug(f"Context Length: {context_length}")
                logger.debug(f"Compressed Context Length: {cache_adapter.get_seq_length(cache)}")
            if use_attention_method:
                attention_method.on_prefill_end()

            perform_decoding_compression = kv_compressor is not None and isinstance(kv_compressor, (DecodingSketch, PrefillDecodingSketch))
            if kv_compressor is not None:
                kv_compressor.set_phase("decode")
            with kv_compressor(self.model) if perform_decoding_compression else contextlib.nullcontext():
                answers = []
                for question_ids in input_tensors["questions_ids"]:
                    checkpoint = cache_adapter.clone_or_checkpoint_for_multi_question(cache)

                    # Allow the attention method to override question position IDs.
                    if use_attention_method:
                        question_position_ids = attention_method.compute_question_position_ids(
                            context_length, question_ids.shape[1], self.model.device,
                        )
                    else:
                        question_position_ids = None

                    answer = self.generate_answer(
                        question_ids=question_ids.to(self.model.device),
                        cache=cache,
                        context_length=context_length,
                        max_new_tokens=max_new_tokens,
                        question_position_ids=question_position_ids,
                    )
                    cache_adapter.restore_after_question(cache, checkpoint)
                    answers.append(answer)
        return answers

    def _run_prefill(
        self,
        context_ids: torch.Tensor,
        cache: Cache,
        prefill_chunk_size: Optional[int] = None,
    ) -> None:
        """Prefill the context into ``cache``.

        ``prefill_chunk_size`` is ``None`` (or ``>=`` the context length) →
        a SINGLE full-context pass, byte-identical to the original pipeline
        (HF derives ``position_ids``/``cache_position`` from the empty cache).

        Otherwise the context is processed in ``prefill_chunk_size`` chunks,
        each fed its true **absolute** ``position_ids`` and ``cache_position``
        so the model places keys at the correct positions and builds the
        correct causal mask.  A plain causal forward is invariant to this
        chunking (each token still attends over keys ``[0, i]``), so the
        post-prefill cache and final logits match the single-pass path.  This
        is the memory-bounded path that ``streaming`` KV compressors hook into
        (they evict after each chunk's forward).

        .. note::
            The cache_position heuristic (``cache_position[-1] > q_len``) would
            misclassify later prefill chunks as decode (they start past
            ``q_len``).  ``_forward`` therefore declares the phase explicitly via
            ``KVCompressor.set_phase("prefill")`` before this loop, so a
            ``streaming`` compressor fires after *every* chunk, not just the
            first.
        """
        context_length = context_ids.shape[1]
        if prefill_chunk_size is None or prefill_chunk_size >= context_length:
            self.model.model(
                input_ids=context_ids,
                past_key_values=cache,
            )
            return

        if prefill_chunk_size <= 0:
            raise ValueError(
                f"prefill_chunk_size must be a positive int or None, "
                f"got {prefill_chunk_size}"
            )

        device = self.model.device
        for start in range(0, context_length, prefill_chunk_size):
            end = min(start + prefill_chunk_size, context_length)
            # ``position_ids`` carry the ABSOLUTE RoPE position; ``cache_position``
            # is the PHYSICAL slot in the cache.  These coincide until a
            # ``streaming`` compressor evicts mid-prefill — then the cache is
            # shorter than ``start``, so the physical slots must follow the
            # actual (post-eviction) cache length, or HF's causal mask (built
            # from ``cache_position``) would let queries attend across the gap.
            position_ids = torch.arange(start, end, device=device).unsqueeze(0)
            past_len = cache.get_seq_length()
            cache_position = torch.arange(past_len, past_len + (end - start), device=device)
            self.model.model(
                input_ids=context_ids[:, start:end],
                past_key_values=cache,
                position_ids=position_ids,
                cache_position=cache_position,
            )

    def generate_answer(
        self,
        question_ids: torch.Tensor,
        cache: Cache,
        context_length: int,
        max_new_tokens: int,
        question_position_ids: Optional[torch.Tensor] = None,
    ) -> str:
        if question_position_ids is not None:
            position_ids = question_position_ids
        else:
            position_ids = torch.arange(
                context_length, context_length + question_ids.shape[1], device=self.model.device
            ).unsqueeze(0)

        outputs = self.model(
            input_ids=question_ids.to(self.model.device),
            past_key_values=cache,
            position_ids=position_ids,
            logits_to_keep=1,
        )

        position_ids = position_ids[:, -1:] + 1
        generated_ids = [outputs.logits[0, -1].argmax()]

        should_stop_token_ids = self.model.generation_config.eos_token_id
        if not isinstance(should_stop_token_ids, list):
            should_stop_token_ids = [should_stop_token_ids]

        for i in range(max_new_tokens - 1):
            outputs = self.model(
                input_ids=generated_ids[-1].unsqueeze(0).unsqueeze(0),
                past_key_values=cache,
                position_ids=position_ids + i,
            )
            new_id = outputs.logits[0, -1].argmax()
            generated_ids.append(new_id)
            if new_id.item() in should_stop_token_ids:
                break

        return str(self.tokenizer.decode(torch.stack(generated_ids), skip_special_tokens=True))

    def postprocess(self, model_outputs, single_question):
        if single_question:
            return {"answer": model_outputs[0]}
        return {"answers": model_outputs}


PIPELINE_REGISTRY.register_pipeline(
    "research-text-generation",
    pipeline_class=ResearchGenerationPipeline,
    pt_model=AutoModelForCausalLM,
)
