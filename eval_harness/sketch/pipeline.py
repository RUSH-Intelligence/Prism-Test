import contextlib
import logging
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, Cache, Pipeline
from transformers.pipelines import PIPELINE_REGISTRY
from transformers.pipelines.base import GenericTensor

from eval_harness.sketch.cache_adapter import CacheAdapter, create_cache_adapter
from eval_harness.sketch.sketches.base_sketch import BaseSketch
from eval_harness.sketch.sketches.decoding_sketch import DecodingSketch
from eval_harness.sketch.sketches.prefill_decoding_sketch import PrefillDecodingSketch

logger = logging.getLogger(__name__)


class SketchTextGenerationPipeline(Pipeline):
    def _sanitize_parameters(
        self,
        question: Optional[str] = None,
        questions: Optional[list[str]] = None,
        answer_prefix: Optional[str] = None,
        sketch: Optional[BaseSketch] = None,
        max_new_tokens: int = 50,
        max_context_length: Optional[int] = None,
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
            "sketch": sketch,
            "max_new_tokens": max_new_tokens,
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
        sketch: Optional[BaseSketch] = None,
        cache: Optional[Cache] = None,
        cache_adapter: Optional[CacheAdapter] = None,
    ):
        if isinstance(sketch, (DecodingSketch, PrefillDecodingSketch)) and len(input_tensors["questions_ids"]) > 1:
            raise ValueError("DecodingSketch is not compatible with multiple questions. Please specify one question.")

        context_ids = input_tensors["context_ids"].to(self.model.device)
        context_length = context_ids.shape[1]

        cache_adapter = cache_adapter or create_cache_adapter(self.model)
        cache = cache_adapter.initialize_cache(cache)

        perform_prefill_compression = sketch is not None and not isinstance(sketch, DecodingSketch)
        with sketch(self.model) if perform_prefill_compression else contextlib.nullcontext():
            self.model.model(
                input_ids=context_ids,
                past_key_values=cache,
            )
            cache_adapter.maybe_slice_prefill(cache)

            logger.debug(f"Context Length: {context_length}")
            logger.debug(f"Compressed Context Length: {cache_adapter.get_seq_length(cache)}")

        perform_decoding_compression = sketch is not None and isinstance(sketch, (DecodingSketch, PrefillDecodingSketch))
        with sketch(self.model) if perform_decoding_compression else contextlib.nullcontext():
            answers = []
            for question_ids in input_tensors["questions_ids"]:
                checkpoint = cache_adapter.clone_or_checkpoint_for_multi_question(cache)
                answer = self.generate_answer(
                    question_ids=question_ids.to(self.model.device),
                    cache=cache,
                    context_length=context_length,
                    max_new_tokens=max_new_tokens,
                )
                cache_adapter.restore_after_question(cache, checkpoint)
                answers.append(answer)
        return answers

    def generate_answer(self, question_ids: torch.Tensor, cache: Cache, context_length: int, max_new_tokens: int) -> str:
        position_ids = torch.arange(
            context_length, context_length + question_ids.shape[1], device=self.model.device
        ).unsqueeze(0)

        outputs = self.model(
            input_ids=question_ids.to(self.model.device),
            past_key_values=cache,
            position_ids=position_ids,
            num_logits_to_keep=1,
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
    "sketch-text-generation",
    pipeline_class=SketchTextGenerationPipeline,
    pt_model=AutoModelForCausalLM,
)
