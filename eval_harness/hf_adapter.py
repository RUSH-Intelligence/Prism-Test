from __future__ import annotations

import logging
import random
import string
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Generator, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

try:
    from transformers import Gemma3ForConditionalGeneration
except ImportError:
    Gemma3ForConditionalGeneration = None  # type: ignore[assignment]

try:
    from transformers import Mistral3ForConditionalGeneration
except ImportError:
    Mistral3ForConditionalGeneration = None  # type: ignore[assignment]

try:
    from transformers import FineGrainedFP8Config
except ImportError:
    FineGrainedFP8Config = None  # type: ignore[assignment]

try:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
except ImportError:
    ALL_ATTENTION_FUNCTIONS = None

logger = logging.getLogger(__name__)


@dataclass
class HFGenerateConfig:
    max_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0


class HFAdapter:
    """
    HuggingFace inference backend with explicit prefill and decode phases.

    Prefill:  all prompt tokens are processed in one forward pass, building the KV cache.
    Decode:   tokens are generated one at a time using the cached KV state.

    Override prefill_attention() and/or decode_attention() in subclasses to plug
    in custom attention kernels.  Both receive:

        module        – the nn.Module for the current attention layer
        queries       – [B, H_q,  S_q, D]  RoPE already applied by the model
        keys          – [B, H_kv, S_kv, D]  RoPE already applied
        values        – [B, H_kv, S_kv, D]
        attention_mask– [B, 1,    S_q, S_kv] or None
        scaling       – float (= 1 / sqrt(head_dim))

    Must return: (attn_output [B, S_q, H_q, D], attn_weights or None)
    """

    def __init__(
        self,
        model: str,
        dtype: str = "auto",
        trust_remote_code: bool = True,
        seed: int = 42,
        max_model_len: Optional[int] = None,
        **model_kwargs: Any,
    ) -> None:
        del seed, max_model_len  # unused at HF load time

        tokenizer_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if "mistral" in model.lower() or "ministral" in model.lower():
            tokenizer_kwargs["fix_mistral_regex"] = True

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(model, **tokenizer_kwargs)
        except TypeError:
            # Older tokenizer classes may not accept fix_mistral_regex.
            tokenizer_kwargs.pop("fix_mistral_regex", None)
            self._tokenizer = AutoTokenizer.from_pretrained(model, **tokenizer_kwargs)

        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        load_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code, **model_kwargs}
        torch_dtype = _resolve_dtype(dtype)
        if torch_dtype is not None:
            load_kwargs["dtype"] = torch_dtype

        # Opt-in: dequantize FP8 checkpoints to floating-point weights at load
        # time (FineGrainedFP8 only). Lets Mistral3 / other FP8-shipped models
        # run with BF16 weights for fair comparison vs BF16 baselines (Llama).
        # Off by default — FP8 stays FP8.
        if load_kwargs.pop("dequantize_fp8", False):
            _apply_fp8_dequantize(model, load_kwargs)

        self._model = _load_model(model, load_kwargs)
        if torch.cuda.is_available():
            self._model = self._model.to("cuda")
        self._model.eval()

    # -------------------------------------------------------------------------
    # Override points
    # -------------------------------------------------------------------------

    def prefill_attention(
        self,
        module: torch.nn.Module,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float = 1.0,
        dropout: float = 0.0,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, None]:
        return _sdpa(queries, keys, values, attention_mask, scaling, dropout)

    def decode_attention(
        self,
        module: torch.nn.Module,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float = 1.0,
        dropout: float = 0.0,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, None]:
        return _sdpa(queries, keys, values, attention_mask, scaling, dropout)

    # -------------------------------------------------------------------------
    # Prefill / decode
    # -------------------------------------------------------------------------

    @contextmanager
    def _with_attention(self, attention_fn: Callable) -> Generator:
        if ALL_ATTENTION_FUNCTIONS is None:
            logger.warning("ALL_ATTENTION_FUNCTIONS unavailable; running default attention.")
            yield
            return

        name = _unique_attention_name()
        ALL_ATTENTION_FUNCTIONS.register(name, attention_fn)
        saved: dict[str, str] = {}
        for mod_name, mod in self._model.named_modules():
            if hasattr(mod, "config") and hasattr(mod.config, "_attn_implementation"):
                saved[mod_name] = mod.config._attn_implementation
                mod.config._attn_implementation = name
        try:
            yield
        finally:
            for mod_name, mod in self._model.named_modules():
                if mod_name in saved:
                    mod.config._attn_implementation = saved[mod_name]
            ALL_ATTENTION_FUNCTIONS._global_mapping.pop(name, None)

    def _prefill(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        with torch.no_grad():
            outputs = self._model(input_ids=input_ids, use_cache=True, return_dict=True)
        return outputs.logits[:, -1, :], outputs.past_key_values

    def _decode(
        self,
        last_logits: torch.Tensor,
        past_key_values: Any,
        gen_cfg: HFGenerateConfig,
    ) -> List[int]:
        generated: List[int] = []
        next_logits = last_logits
        eos_token_id = self._tokenizer.eos_token_id

        for _ in range(gen_cfg.max_tokens):
            next_token_id = _sample(next_logits, gen_cfg)
            generated.append(next_token_id)
            if next_token_id == eos_token_id:
                break
            next_input = torch.tensor(
                [[next_token_id]], dtype=torch.long, device=self._model.device
            )
            with torch.no_grad():
                outputs = self._model(
                    input_ids=next_input,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
            next_logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

        return generated

    def generate(self, prompts: List[str], gen_cfg: HFGenerateConfig) -> List[str]:
        texts: List[str] = []
        for prompt in prompts:
            token_ids = self._tokenizer.encode(prompt, add_special_tokens=False)
            input_ids = torch.tensor([token_ids], dtype=torch.long, device=self._model.device)
            last_logits, past_key_values = self._prefill(input_ids)
            generated_ids = self._decode(last_logits, past_key_values, gen_cfg)
            texts.append(self._tokenizer.decode(generated_ids, skip_special_tokens=True))
        return texts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sdpa(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float,
) -> Tuple[torch.Tensor, None]:
    """Scaled dot-product attention with GQA support. Returns [B, S_q, H_q, D]."""
    keys, values = _expand_kv(queries, keys, values)
    q_len, k_len = queries.shape[-2], keys.shape[-2]
    is_causal = attention_mask is None and q_len == k_len
    out = F.scaled_dot_product_attention(
        queries * scaling, keys, values,
        attn_mask=attention_mask, dropout_p=dropout, is_causal=is_causal,
    )
    return out.transpose(1, 2), None


def _expand_kv(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Repeat KV heads to match query heads (GQA/MQA)."""
    h_q, h_kv = queries.shape[1], keys.shape[1]
    if h_q == h_kv:
        return keys, values
    assert h_q % h_kv == 0
    rep = h_q // h_kv
    keys   = keys  [:, :, None, :, :].expand(-1, h_kv, rep, -1, -1).reshape(keys.shape[0],   h_q, keys.shape[2],   keys.shape[3])
    values = values[:, :, None, :, :].expand(-1, h_kv, rep, -1, -1).reshape(values.shape[0], h_q, values.shape[2], values.shape[3])
    return keys, values


def _resolve_dtype(dtype: str) -> Optional[torch.dtype]:
    return {
        "float16": torch.float16, "fp16": torch.float16,
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
        "float32": torch.float32, "fp32": torch.float32,
    }.get(dtype)


def _apply_fp8_dequantize(model_name: str, load_kwargs: dict[str, Any]) -> None:
    """Override a model's ``config.json`` FP8 quantization with a dequantize-on-load
    variant. Mutates ``load_kwargs`` in place to inject ``quantization_config``.

    Streams FP8 weights off disk and dequantizes each tensor to ``load_kwargs['dtype']``
    (BF16/FP16) before they land in GPU memory — no FP8 VRAM spike, no post-hoc pass.
    Silently no-ops for non-FP8 checkpoints (the override is harmless when the model
    has no quantization to begin with — the quantizer just has nothing to attach).
    """
    if FineGrainedFP8Config is None:
        logger.warning(
            "dequantize_fp8=True requested but FineGrainedFP8Config is unavailable "
            "in this transformers version; loading FP8 weights as-is."
        )
        return
    try:
        cfg = AutoConfig.from_pretrained(
            model_name, trust_remote_code=load_kwargs.get("trust_remote_code", True)
        )
    except Exception:
        cfg = None
    qcfg = getattr(cfg, "quantization_config", None) if cfg is not None else None
    quant_method = (qcfg or {}).get("quant_method") if isinstance(qcfg, dict) else getattr(qcfg, "quant_method", None)
    if quant_method != "fp8":
        logger.info(
            "dequantize_fp8=True ignored: model %s is not FP8-quantized (quant_method=%s).",
            model_name, quant_method,
        )
        return
    load_kwargs["quantization_config"] = FineGrainedFP8Config(dequantize=True)
    logger.info("Loading %s with FP8 → %s dequantization at load time.", model_name, load_kwargs.get("dtype"))


def _load_model(model_name: str, load_kwargs: dict[str, Any]) -> PreTrainedModel:
    explicit_impl = load_kwargs.pop("attn_implementation", None)
    use_gemma3_conditional = False
    use_mistral3_conditional = False

    try:
        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=load_kwargs.get("trust_remote_code", True))
        architectures = [a.lower() for a in (getattr(cfg, "architectures", None) or [])]
        model_type = str(getattr(cfg, "model_type", "")).lower()
        use_gemma3_conditional = Gemma3ForConditionalGeneration is not None and (
            "gemma3forconditionalgeneration" in architectures
        )
        use_mistral3_conditional = Mistral3ForConditionalGeneration is not None and (
            "mistral3forconditionalgeneration" in architectures or model_type in {"mistral3", "ministral3"}
        )
    except Exception:
        # If config probing fails, stay on AutoModelForCausalLM for compatibility.
        pass

    if use_gemma3_conditional:
        return _load_conditional_model(
            Gemma3ForConditionalGeneration,
            model_name,
            load_kwargs,
            explicit_impl,
            family_label="Gemma3 conditional",
        )

    if use_mistral3_conditional:
        return _load_conditional_model(
            Mistral3ForConditionalGeneration,
            model_name,
            load_kwargs,
            explicit_impl,
            family_label="Mistral3 conditional",
        )

    if explicit_impl is not None:
        m = AutoModelForCausalLM.from_pretrained(
            model_name, attn_implementation=explicit_impl, **load_kwargs
        )
        logger.info("Loaded model with attn_implementation=%s", explicit_impl)
        return m

    for attn_impl in ("flash_attention_2", "sdpa"):
        try:
            m = AutoModelForCausalLM.from_pretrained(
                model_name, attn_implementation=attn_impl, **load_kwargs
            )
            logger.info("Loaded model with attn_implementation=%s", attn_impl)
            return m
        except (ImportError, ValueError, RuntimeError):
            continue
    return AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)


def _load_conditional_model(
    model_cls,
    model_name: str,
    load_kwargs: dict[str, Any],
    explicit_impl: Optional[str],
    family_label: str,
) -> PreTrainedModel:
    if explicit_impl is not None:
        m = model_cls.from_pretrained(
            model_name, attn_implementation=explicit_impl, **load_kwargs
        )
        logger.info("Loaded %s model with attn_implementation=%s", family_label, explicit_impl)
        return m

    for attn_impl in ("flash_attention_2", "sdpa"):
        try:
            m = model_cls.from_pretrained(
                model_name, attn_implementation=attn_impl, **load_kwargs
            )
            logger.info("Loaded %s model with attn_implementation=%s", family_label, attn_impl)
            return m
        except (ImportError, ValueError, RuntimeError):
            continue

    return model_cls.from_pretrained(model_name, **load_kwargs)


def _unique_attention_name() -> str:
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
    return f"prism_{suffix}"


def _sample(logits: torch.Tensor, gen_cfg: HFGenerateConfig) -> int:
    if gen_cfg.temperature <= 0.0:
        return int(logits.argmax(dim=-1).item())
    scaled = logits / max(gen_cfg.temperature, 1e-5)
    probs = torch.softmax(scaled, dim=-1)
    if gen_cfg.top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        mask = (cumsum - sorted_probs) > gen_cfg.top_p
        sorted_probs = sorted_probs.masked_fill(mask, 0.0)
        probs = torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    return int(torch.multinomial(probs, num_samples=1).item())
