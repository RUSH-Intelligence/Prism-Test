import contextlib
import logging
import re
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, Cache, Pipeline
from transformers.pipelines import PIPELINE_REGISTRY
from transformers.pipelines.base import GenericTensor

# Llama-3.1-Instruct's default chat template ALWAYS emits a system header with
# "Cutting Knowledge Date / Today Date", even when no system message is passed
# and even when an empty system message is passed. Official LongBench / kvpress
# eval scripts use raw concatenation (no system block) — leaving the auto block
# in shifts every cell's numbers vs the published baselines. Strip it.
_AUTO_SYSTEM_BLOCK_RE = re.compile(
    r"(<\|begin_of_text\|>)"
    r"<\|start_header_id\|>system<\|end_header_id\|>\n\n"
    r"Cutting Knowledge Date:[^<]*?<\|eot_id\|>"
)

# Mistral-3 / Ministral-3 chat templates auto-inject a "[SYSTEM_PROMPT]You are
# Ministral-3-..., a Large Language Model created by Mistral AI...[/SYSTEM_PROMPT]"
# block even when no system message is passed. Same parity concern as Llama's
# Cutting-Knowledge-Date block — strip it so LongBench / kvpress numbers line
# up with the published baselines and our prior Llama runs. DOTALL because the
# auto block spans multiple lines.
_MISTRAL_AUTO_SYSTEM_BLOCK_RE = re.compile(
    r"(<s>)\[SYSTEM_PROMPT\].*?\[/SYSTEM_PROMPT\]",
    re.DOTALL,
)

# Both substitutions reduce to "keep the BOS capture group, drop the rest" —
# applied in sequence so each model family hits exactly one matching pattern.
_AUTO_SYSTEM_BLOCK_PATTERNS = (_AUTO_SYSTEM_BLOCK_RE, _MISTRAL_AUTO_SYSTEM_BLOCK_RE)

from eval_harness.attention_methods._method_base import PrefillMethod
from eval_harness.kv_compression.cache_adapter import CacheAdapter, create_cache_adapter
from eval_harness.kv_compression.base import KVCompressor
from eval_harness.kv_compression.compressors.decoding_sketch import DecodingSketch
from eval_harness.kv_compression.compressors.prefill_decoding_sketch import PrefillDecodingSketch

logger = logging.getLogger(__name__)


def _model_has_mamba_layers(model) -> bool:
    """True if the model has Mamba/SSM decoder layers (e.g. NemotronH).

    Such layers' cached forward assumes a single query token, so the question
    must be fed token-by-token after the context prefill (see generate_answer).
    Detected via ``hybrid_override_pattern`` ('M' = mamba) or ``layers_block_type``
    / ``layer_types`` containing "mamba".
    """
    cfg = getattr(model, "config", None)
    if cfg is None:
        return False
    try:
        cfg = cfg.get_text_config(decoder=True)
    except Exception:
        pass
    pattern = getattr(cfg, "hybrid_override_pattern", None)
    if isinstance(pattern, str) and "M" in pattern:
        return True
    layer_types = getattr(cfg, "layers_block_type", None) or getattr(cfg, "layer_types", None) or []
    return any("mamba" in str(t).lower() for t in layer_types)


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
        use_chat_template: bool = True,
        strip_auto_system_block: bool = False,
        middle_truncation: bool = False,
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
            "use_chat_template": use_chat_template,
            "strip_auto_system_block": strip_auto_system_block,
            "middle_truncation": middle_truncation,
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

    def _get_text_decoder(self):
        # Matches the multimodal-aware ``model.model.language_model`` lookup used
        # across attention_methods/ and kv_compression/ (e.g. _method_base.py,
        # finch_sketch.py, kvzip_sketch.py). For CausalLM (Llama/Mistral),
        # ``model.model`` already IS the text decoder.
        inner = self.model.model
        return inner.language_model if hasattr(inner, "language_model") else inner

    def preprocess(
        self,
        context: str,
        questions: list[str],
        answer_prefix: str,
        max_context_length: int,
        enable_thinking: bool = False,
        use_chat_template: bool = True,
        strip_auto_system_block: bool = False,
        middle_truncation: bool = False,
    ):
        # MistralCommonBackend exposes ``apply_chat_template`` but intentionally
        # has no ``chat_template`` attribute (mistral-common bakes the format in).
        # Default the missing case to a truthy sentinel so we take the chat path
        # rather than crashing on AttributeError.
        chat_template = getattr(self.tokenizer, "chat_template", "<builtin>")
        if chat_template is None or not use_chat_template:
            bos_token = getattr(self.tokenizer, "bos_token", "")
            context = bos_token + context
            question_suffix = "\n"
        else:
            separator = "#" * (len(context) + 10)
            # Standard HF tokenizers (Llama, Qwen, ...) have a Jinja
            # ``chat_template`` and silently ignore unused vars like
            # ``enable_thinking``. ``MistralCommonBackend`` has no
            # ``chat_template`` attr and validates kwargs strictly via
            # Pydantic, so we only forward the Jinja-only kwarg when the
            # tokenizer is the Jinja kind.
            chat_kwargs = {"add_generation_prompt": True, "tokenize": False}
            if hasattr(self.tokenizer, "chat_template"):
                chat_kwargs["enable_thinking"] = enable_thinking
            context = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": context + separator}],
                **chat_kwargs,
            )
            # Opt-in: strip the model's auto-injected system header. Off by
            # default to preserve prior RULER/KV/positional/DCA numerics — the
            # strip changes the prompt prefix and shifts established baselines.
            # LongBench enables it per-row for parity with the official pred.py.
            if strip_auto_system_block:
                for pattern in _AUTO_SYSTEM_BLOCK_PATTERNS:
                    context = pattern.sub(r"\1", context)
            context, question_suffix = context.split(separator)

        questions = [question + question_suffix + answer_prefix for question in questions]

        context_ids = self.tokenizer.encode(context, return_tensors="pt", add_special_tokens=False)
        question_ids = [
            self.tokenizer.encode(question, return_tensors="pt", add_special_tokens=False) for question in questions
        ]

        # Overflow truncation. Default is head-only truncation (preserves prior
        # RULER NIAH semantics — which specific needles survive). Opt-in
        # middle-truncation (first half + last half) matches official LongBench
        # pred.py; LongBench sets this per-row. The question is appended
        # separately AFTER context_ids in _forward, so it is always preserved.
        if context_ids.shape[1] > max_context_length:
            original_len = context_ids.shape[1]
            longest_question = max((q.shape[1] for q in question_ids), default=0)
            overflow = original_len - max_context_length
            if middle_truncation:
                half = max_context_length // 2
                context_ids = torch.cat(
                    [context_ids[:, :half], context_ids[:, -half:]], dim=1
                )
                logger.warning(
                    "TRUNCATION TRIGGERED: context=%d tokens > cap=%d (overflow=%d). "
                    "Applied middle-truncation (first %d + last %d tokens). "
                    "Question (%d tokens) is appended in full after the truncated context.",
                    original_len, max_context_length, overflow,
                    half, half, longest_question,
                )
            else:
                context_ids = context_ids[:, :max_context_length]
                logger.warning(
                    "TRUNCATION TRIGGERED: context=%d tokens > cap=%d (overflow=%d). "
                    "Applied head-truncation (first %d tokens). "
                    "Question (%d tokens) is appended in full after the truncated context.",
                    original_len, max_context_length, overflow,
                    max_context_length, longest_question,
                )

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
                    kv_compressor=kv_compressor,
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
        kv_compressor: Optional[KVCompressor] = None,
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
        # Reset the per-call "final chunk" flag so a prior chunked prefill that
        # crashed mid-loop can't leak a stale ``False`` into this call's
        # POST_PREFILL gate.  The chunked branch below overrides this per chunk.
        if kv_compressor is not None:
            kv_compressor.set_prefill_is_final(True)
        text_decoder = self._get_text_decoder()
        if prefill_chunk_size is None or prefill_chunk_size >= context_length:
            text_decoder(
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
            # Gate POST_PREFILL on "is this the final chunk?".  STREAMING
            # ignores the flag and still fires every chunk.  Single-pass
            # prefill never enters this loop, so its default ``True`` stands
            # and POST_PREFILL behavior is unchanged.
            if kv_compressor is not None:
                kv_compressor.set_prefill_is_final(end == context_length)
            text_decoder(
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

        question_ids = question_ids.to(self.model.device)
        if _model_has_mamba_layers(self.model) and question_ids.shape[1] > 1:
            # Mamba/SSM layers' cached (decode) forward assumes a single query token
            # (it does ``hidden_states.squeeze(1)``); a multi-token question block
            # after the context prefill would hit that path with q_len>1 and crash
            # (e.g. NemotronH: "weight must have shape (dim, width)"). Feed the
            # question one token at a time — identical to the block forward for the
            # attention layers (causal, KV-cached), and the only correct path for the
            # mamba layers. Compression already happened on the context prefill; no
            # compressor hooks are active here.
            last_logits = None
            for j in range(question_ids.shape[1]):
                outputs = self.model(
                    input_ids=question_ids[:, j : j + 1],
                    past_key_values=cache,
                    position_ids=position_ids[:, j : j + 1],
                    logits_to_keep=1,
                )
                last_logits = outputs.logits
            logits = last_logits
        else:
            outputs = self.model(
                input_ids=question_ids,
                past_key_values=cache,
                position_ids=position_ids,
                logits_to_keep=1,
            )
            logits = outputs.logits

        position_ids = position_ids[:, -1:] + 1
        generated_ids = [logits[0, -1].argmax()]

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
