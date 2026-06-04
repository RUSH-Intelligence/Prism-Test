from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from vllm import LLM, SamplingParams


@dataclass
class VLLMGenerateConfig:
    max_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0


class VLLMAdapter:
    def __init__(
        self,
        model: str,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        max_model_len: Optional[int] = None,
        gpu_memory_utilization: float = 0.9,
        trust_remote_code: bool = True,
        enable_prefix_caching: bool = True,
        seed: int = 42,
        **llm_kwargs,
    ) -> None:
        kwargs = {
            "model": model,
            "tensor_parallel_size": tensor_parallel_size,
            "dtype": dtype,
            "gpu_memory_utilization": gpu_memory_utilization,
            "trust_remote_code": trust_remote_code,
            "enable_prefix_caching": enable_prefix_caching,
            "seed": seed,
        }
        if max_model_len is not None:
            kwargs["max_model_len"] = max_model_len
        kwargs.update(llm_kwargs)
        self._llm = LLM(**kwargs)
        # Store tokenizer for direct tokenization (bypasses chat template)
        self._tokenizer = self._llm.get_tokenizer()

    def generate(self, prompts: List[str], gen_cfg: VLLMGenerateConfig) -> List[str]:
        sampling = SamplingParams(
            temperature=gen_cfg.temperature,
            top_p=gen_cfg.top_p,
            max_tokens=gen_cfg.max_tokens,
        )

        # Tokenize prompts directly to bypass automatic chat template application
        # This follows the pattern from sparse-attention-hub adapter
        prompt_token_ids = []
        for prompt in prompts:
            token_ids = self._tokenizer.encode(prompt)
            prompt_token_ids.append(token_ids)

        # Pass token IDs positionally to stay compatible with vLLM API keyword changes.
        outputs = self._llm.generate(
            prompt_token_ids,
            sampling_params=sampling,
            use_tqdm=False,
        )

        texts: List[str] = []
        for output in outputs:
            if not output.outputs:
                texts.append("")
                continue
            texts.append(output.outputs[0].text)
        return texts
