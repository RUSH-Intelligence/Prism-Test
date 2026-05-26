from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .long_context import (
    LongContextCompressionConfig,
    merge_budgeted_indices,
    select_topk_indices_from_scores,
)


@dataclass
class HFGenerateConfig:
    max_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class HFAdapter:
    def __init__(
        self,
        model: str,
        dtype: str = "auto",
        trust_remote_code: bool = True,
        seed: int = 42,
        max_model_len: Optional[int] = None,
        enable_long_context_compression: bool = False,
        compression_sink_tokens: int = 32,
        compression_local_tokens: int = 4096,
        compression_top_k_tokens: Optional[int] = None,
        compression_span_tokens: int = 32,
        hf_naive_reattn_query_tokens: int = 128,
        **model_kwargs: Any,
    ) -> None:
        del seed

        self._tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=trust_remote_code)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        torch_dtype = None
        if dtype in {"float16", "fp16"}:
            torch_dtype = torch.float16
        elif dtype in {"bfloat16", "bf16"}:
            torch_dtype = torch.bfloat16
        elif dtype in {"float32", "fp32"}:
            torch_dtype = torch.float32

        load_kwargs = {
            "trust_remote_code": trust_remote_code,
            **model_kwargs,
        }
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype

        self._model = AutoModelForCausalLM.from_pretrained(model, **load_kwargs)
        if torch.cuda.is_available():
            self._model = self._model.to("cuda")
        self._model.eval()

        resolved_max_len = max_model_len
        if resolved_max_len is None:
            resolved_max_len = getattr(self._model.config, "max_position_embeddings", None)

        self._compression_cfg = LongContextCompressionConfig(
            enabled=enable_long_context_compression,
            max_context_len=resolved_max_len,
            sink_tokens=compression_sink_tokens,
            local_tokens=compression_local_tokens,
            top_k_tokens=compression_top_k_tokens,
            span_tokens=compression_span_tokens,
        )
        self._naive_query_tokens = max(int(hf_naive_reattn_query_tokens), 1)

    def _build_budget(self, seq_len: int) -> tuple[list[int], list[int], list[int], int]:
        max_context_len = int(self._compression_cfg.max_context_len or 0)
        sink_keep = min(max(int(self._compression_cfg.sink_tokens), 0), seq_len, max_context_len)
        remaining_budget = max_context_len - sink_keep

        local_keep = min(max(int(self._compression_cfg.local_tokens), 0), max(seq_len - sink_keep, 0), remaining_budget)
        remaining_budget -= local_keep

        sink_indices = list(range(sink_keep))
        local_start = seq_len - local_keep
        local_indices = list(range(local_start, seq_len))
        candidate_indices = list(range(sink_keep, max(local_start, sink_keep)))

        if self._compression_cfg.top_k_tokens is None:
            top_k_budget = remaining_budget
        else:
            top_k_budget = min(max(int(self._compression_cfg.top_k_tokens), 0), remaining_budget)
        return sink_indices, local_indices, candidate_indices, top_k_budget

    def _naive_layerwise_qk_scores(self, input_ids: torch.Tensor, candidate_indices: Sequence[int]) -> List[float]:
        if not candidate_indices:
            return []

        with torch.no_grad():
            outputs = self._model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("Model did not return hidden_states; cannot compute naive ReAttention scores")

        layers = getattr(self._model, "model", None)
        if layers is None or not hasattr(layers, "layers"):
            raise RuntimeError("Unsupported architecture for naive ReAttention scorer: missing model.layers")

        per_layer_scores: List[torch.Tensor] = []
        candidate_tensor = torch.tensor(candidate_indices, device=input_ids.device, dtype=torch.long)

        for layer_idx, layer in enumerate(layers.layers):
            hs = hidden_states[layer_idx]
            attn = layer.self_attn
            q_proj = attn.q_proj
            k_proj = attn.k_proj

            q_start = max(0, hs.shape[1] - self._naive_query_tokens)
            query_hs = hs[:, q_start:, :]
            key_hs = hs.index_select(1, candidate_tensor)

            q = q_proj(query_hs)
            k = k_proj(key_hs)

            num_heads = getattr(attn, "num_heads")
            num_kv_heads = getattr(attn, "num_key_value_heads", num_heads)
            head_dim = getattr(attn, "head_dim")

            q = q.view(1, -1, num_heads, head_dim).transpose(1, 2)
            k = k.view(1, -1, num_kv_heads, head_dim).transpose(1, 2)
            k = _repeat_kv(k, max(num_heads // num_kv_heads, 1))

            layer_scores = torch.einsum("bhqd,bhkd->bhqk", q, k).mean(dim=(1, 2)).squeeze(0)
            per_layer_scores.append(layer_scores.to(torch.float32))

        stacked = torch.stack(per_layer_scores, dim=0).mean(dim=0)
        return stacked.detach().cpu().tolist()

    def _compress_token_ids(self, token_ids: Sequence[int]) -> List[int]:
        seq_len = len(token_ids)
        if (
            not self._compression_cfg.enabled
            or self._compression_cfg.max_context_len is None
            or seq_len <= self._compression_cfg.max_context_len
        ):
            return list(token_ids)

        sink_indices, local_indices, candidate_indices, top_k_budget = self._build_budget(seq_len)
        if top_k_budget <= 0 or not candidate_indices:
            kept = sorted(set(sink_indices + local_indices))
            return [token_ids[idx] for idx in kept]

        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self._model.device)
        scores = self._naive_layerwise_qk_scores(input_ids=input_ids, candidate_indices=candidate_indices)
        topk_indices = select_topk_indices_from_scores(
            candidate_indices=candidate_indices,
            scores=scores,
            top_k=top_k_budget,
        )

        from .long_context import CompressionBudget

        budget = CompressionBudget(
            sink_indices=sink_indices,
            local_indices=local_indices,
            candidate_indices=candidate_indices,
            top_k_budget=top_k_budget,
        )
        kept = merge_budgeted_indices(
            token_count=seq_len,
            budget=budget,
            topk_indices=topk_indices,
            span_tokens=self._compression_cfg.span_tokens,
        )
        return [token_ids[idx] for idx in kept]

    def generate(self, prompts: List[str], gen_cfg: HFGenerateConfig) -> List[str]:
        texts: List[str] = []
        for prompt in prompts:
            token_ids = self._tokenizer.encode(prompt, add_special_tokens=False)
            token_ids = self._compress_token_ids(token_ids)

            input_ids = torch.tensor([token_ids], dtype=torch.long, device=self._model.device)
            attention_mask = torch.ones_like(input_ids)
            do_sample = gen_cfg.temperature > 0.0

            gen_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_new_tokens": gen_cfg.max_tokens,
                "do_sample": do_sample,
                "top_p": gen_cfg.top_p,
                "pad_token_id": self._tokenizer.pad_token_id,
                "eos_token_id": self._tokenizer.eos_token_id,
            }
            if do_sample:
                gen_kwargs["temperature"] = max(gen_cfg.temperature, 1e-5)

            with torch.no_grad():
                outputs = self._model.generate(**gen_kwargs)

            generated_ids = outputs[0, input_ids.shape[1] :]
            texts.append(self._tokenizer.decode(generated_ids, skip_special_tokens=True))
        return texts