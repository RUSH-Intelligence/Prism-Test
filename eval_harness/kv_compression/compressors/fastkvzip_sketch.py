import logging
import math
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator, Optional

import torch
from torch import nn
from transformers import AutoConfig, PreTrainedModel
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from eval_harness.kv_compression.base import KVCompressor
from eval_harness.kv_compression.registry import register_kv_compressor

logger = logging.getLogger(__name__)


class FastKVzipGate(nn.Module):
    """
    Fast KVzip gate architecture (https://arxiv.org/abs/2601.17668).

    Transcribed from kvpress ``FastKVzipGate`` (kvpress/presses/fastkvzip_press.py).
    ``q_proj``/``k_proj``/``b`` are created in the checkpoint dtype while
    ``q_norm``/``k_norm``/``k_base`` keep the default (fp32) dtype, exactly as in
    kvpress: the RMSNorm fp32-weight multiply promotes activations to fp32, which
    makes the fp32 ``k_base`` matmul type-consistent.
    """

    def __init__(
        self,
        index: int,
        input_dim: int,
        nhead: int,
        ngroup: int,
        dtype: torch.dtype,
        output_dim: int = 16,
        sink: int = 16,
    ):
        super().__init__()
        self.index = index
        self.output_dim = output_dim
        self.nhead = nhead
        self.ngroup = ngroup
        self.sink = sink

        self.q_proj = nn.Linear(input_dim, nhead * ngroup * output_dim, bias=True, dtype=dtype)
        self.k_proj = nn.Linear(input_dim, nhead * output_dim, bias=False, dtype=dtype)
        self.q_norm = Qwen3RMSNorm(output_dim)
        self.k_norm = Qwen3RMSNorm(output_dim)
        self.k_base = nn.Parameter(torch.zeros([nhead, 1, sink, output_dim]))
        self.b = nn.Parameter(torch.zeros([nhead, 1, ngroup], dtype=dtype))

        self.d = math.sqrt(self.output_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.squeeze(0)  # bsz = 1
        nseq = hidden_states.shape[0]  # sequence x dim
        hidden_shape = (nseq, self.nhead, -1, self.output_dim)

        queries = self.q_norm(self.q_proj(hidden_states).view(hidden_shape))
        keys = self.k_norm(self.k_proj(hidden_states).view(hidden_shape))
        queries = queries.transpose(0, 1).transpose(-1, -2)
        keys = keys.transpose(0, 1)

        # head x seq x 1 x group
        logit = torch.matmul(keys, queries) / self.d + self.b.unsqueeze(2)
        # head x 1 x sink x group
        logit_base = torch.matmul(self.k_base, queries) / self.d
        score = 1 / (1 + torch.exp(logit_base - logit).sum(2, keepdim=True))

        score = score.mean(-1)  # n_head, seq, 1
        return score.squeeze(-1).unsqueeze(0)  # bsz x n_head x seq

    def extra_repr(self) -> str:
        repr_str = f"index={self.index}, output_dim={self.output_dim}, nhead={self.nhead}, ngroup={self.ngroup}\n"
        if self.sink != 0:
            repr_str += f"k_base shape: {self.k_base.shape}\n"
        repr_str += f"b shape: {self.b.shape}\n"
        return repr_str


def get_gate_id(model_name: str) -> str:
    """Get the gate id from model names."""
    config = AutoConfig.from_pretrained(model_name)
    if hasattr(config, "text_config"):
        config = config.text_config
    ngroup = config.num_attention_heads // config.num_key_value_heads
    file_name = f"q{ngroup}_dim16_sink16"

    model_name = model_name.split("/")[-1].lower()
    gate_id = os.path.join(model_name, file_name + ".pt")
    return gate_id


def get_gate_weight(model_name: str):
    """Load trained gate weights from HuggingFace."""
    from huggingface_hub import hf_hub_download

    gate_id = get_gate_id(model_name)
    file_path = hf_hub_download(repo_id="Jang-Hyun/Fast-KVzip", filename=gate_id, repo_type="model")

    weights = torch.load(file_path, weights_only=False)["module"]
    return weights, gate_id


def load_fastkvzip(model_name: str = "Qwen/Qwen3-8B", device="cuda") -> list[nn.Module]:
    """Load trained gate weights and rebuild per-layer ``FastKVzipGate`` modules."""
    if not model_name:
        raise AssertionError("Model_name is empty. Please check load_gate.")
    state_dict, gate_id = get_gate_weight(model_name)

    dtype = state_dict[0]["q_proj.weight"].dtype
    head_group_outdim, input_dim = state_dict[0]["q_proj.weight"].shape
    head_outdim, _ = state_dict[0]["k_proj.weight"].shape
    output_dim = state_dict[0]["q_norm.weight"].shape[-1]
    nhead = head_outdim // output_dim
    ngroup = head_group_outdim // head_outdim

    m = re.search(r"sink(\d+)", gate_id)
    sink = int(m.group(1)) if m else 0

    modules = []
    for idx, weight in enumerate(state_dict):
        module = FastKVzipGate(idx, input_dim, nhead, ngroup, dtype, output_dim, sink).to(device)
        module.load_state_dict(weight)
        modules.append(module)

    logger.info(f"load gate {gate_id} ({module})")
    return modules


@register_kv_compressor("fastkvzip")
@dataclass
class FastKVzipSketch(KVCompressor):
    """
    Fast KVzip estimates KV importance scores using gates trained on KVzip scores.

    Port of kvpress ``FastKVzipPress`` (kvpress/presses/fastkvzip_press.py).
    Based on Fast KVzip (https://arxiv.org/abs/2601.17668).
    Authors: Jang-Hyun Kim, Dongyoon Han, Sangdoo Yun (NAVER AI Lab).

    Per-layer trained gate networks predict KV importance from ``hidden_states``
    alone (no queries, keys, or attention weights): each gate scores a token as
    the mini-attention probability of the token's own gate-key against ``sink``
    learned base keys, averaged over query groups, yielding scores strictly in
    (0, 1). The first ``n_sink`` positions and a recency window are then
    force-protected (score := 1.0). Scoring happens in the prefill forward hook;
    compression is deferred until the sketch context exits (right after prefill,
    before the question/decode passes) and is realized as *fake compression*:
    the bottom-scored KV pairs are recorded in ``module.masked_key_indices``,
    consumed by ``eval_harness/kv_compression/attention_patch.py`` (active for all
    non-eager attention implementations), which substitutes fake keys such that
    ``exp(<q, k>) = 0`` at question/decode time. The cache stays physically
    full-length and rectangular, so the global cross-layer budget
    (``layerwise=False``) and the per-head ragged within-layer bottom-k are both
    decode-safe, exactly as in kvpress.

    Gates are auto-downloaded from the HF hub repo ``Jang-Hyun/Fast-KVzip``
    (file ``{model_basename}/q{ngroup}_dim16_sink16.pt``; released only for
    specific models, e.g. the Qwen3 family) unless injected via ``gates``.

    Requires a non-eager attention implementation (the runner default is sdpa).
    Only batch size 1 is supported (gate architecture constraint). Validated
    only with ``prefill_method: none``: methods that prune or reposition the
    cache during prefill (ReAttention, DCA) would invalidate the recorded
    masked sequence indices.

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
    window_size : int, default=4096
        Number of tokens in the local window retained for long contexts
        (context length >= 32000; threshold hardcoded as in kvpress).
    window_ratio : float, default=0.02
        Fraction of the context length used to calculate the local window size
        retained during short-context prefilling.
    gates : list[nn.Module], optional
        Per-layer ``FastKVzipGate`` modules indexed by ``layer_idx``. When
        ``None``, gates are downloaded in ``post_init_from_model``.

    Deviations from kvpress
    -----------------------
    - ``gates`` is a regular constructor argument (kvpress uses
      ``field(init=False, default=None)``) so gate modules can be injected for
      offline runs and tests; when set, ``post_init_from_model`` skips the hub
      download (same guard as kvpress).
    - Window-override guard: when the computed window size is 0 (context shorter
      than ``1 / window_ratio``), kvpress executes ``scores[:, :, -0:] = 1.0``,
      which sets ALL scores to 1.0 and degenerates selection to arbitrary ties;
      this port only applies the override when ``window_size > 0`` (spec-
      recommended guard), so short contexts keep their gate scores.
    - Prefill/decode gating uses ``KVCompressor._is_decoding_step`` instead of the
      inline ``cache_position`` comparison (framework idiom, same semantics).
    - Hook installation reuses ``KVCompressor.__call__`` (skips all non-full-
      attention layers, not only Gemma3 sliding ones), and ``compress_post``
      operates on the layers that actually scored instead of iterating
      ``model.model.layers`` — this avoids two kvpress latent bugs (``None``
      entries crashing the score stack on skipped layers; broken iteration on
      ``ForConditionalGeneration`` wrappers). Identical behavior on homogeneous
      models. ``compress_post`` consequently runs right after the hooks are
      removed rather than just before — the same point in the pipeline (after
      prefill, before the question/decode passes).
    - Explicit ``batch size = 1`` assertion before gate scoring (kvpress fails
      later with an opaque shape/view error).
    - ``__post_init__`` validates ``0 <= compression_ratio < 1`` (framework
      convention; kvpress does not validate this press).

    Upstream quirks kept verbatim
    -----------------------------
    - The 32000-token threshold switching between ``window_ratio`` and
      ``window_size`` is hardcoded.
    - ``n_sink`` larger than the context length silently truncates.
    - Eager attention is rejected (``assert`` in ``compress_post``): the
      masked-key substitution only runs for implementations dispatched through
      ``ALL_ATTENTION_FUNCTIONS``.
    """

    compression_ratio: float = 0.0
    layerwise: bool = False

    n_sink: int = 4
    window_size: int = 4096  # for long contexts
    window_ratio: float = 0.02

    gates: Optional[list[nn.Module]] = None

    def __post_init__(self):
        super().__post_init__()
        assert 0 <= self.compression_ratio < 1, "Compression ratio must be between 0 and 1"
        self.score_val: dict[int, torch.Tensor] = {}
        self._scored_modules: list[nn.Module] = []

    def post_init_from_model(self, model: PreTrainedModel):
        """Automatically load gates for the model."""
        if self.gates is None:
            try:
                self.gates = load_fastkvzip(model_name=model.config.name_or_path, device=model.device)
            except Exception as e:
                raise RuntimeError(
                    "The gates for the given model are not released! "
                    "Please check the available models at: "
                    "https://huggingface.co/Jang-Hyun/Fast-KVzip/tree/main"
                ) from e

    @contextmanager
    def __call__(self, model: PreTrainedModel) -> Generator:
        """
        Context manager that handles both initial prefilling and Fast KVzip scoring/compression.

        1. Inside the context: prefilling with the context and KV importance scoring via gates.
        2. On exit: fake KV eviction (``masked_key_indices``) based on the importance scores.
        """
        self.score_val = {}
        self._scored_modules = []
        with super().__call__(model):
            yield
        self.compress_post()

    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        """During prefill, only calculate and store importance scores; never touch the cache."""
        hidden_states = kwargs["hidden_states"]
        q_len = hidden_states.shape[1]

        if self._is_decoding_step(module, kwargs, q_len):
            return output

        self._score_fast(module, hidden_states)
        return output

    def _score_fast(self, module: nn.Module, hidden_states: torch.Tensor):
        """Calculate the KV importance scores."""
        assert hidden_states.shape[0] == 1, (
            f"FastKVzipSketch only supports batch size 1, got {hidden_states.shape[0]}"
        )
        if self.gates is None:
            raise ValueError("Gates not loaded. Make sure post_init_from_model was called or inject gates.")
        layer_idx = int(module.layer_idx)

        self.gates[layer_idx] = self.gates[layer_idx].to(hidden_states.device)
        scores = self.gates[layer_idx](hidden_states)
        scores[:, :, : self.n_sink] = 1.0

        ctx_len = scores.size(-1)
        if ctx_len < 32000:
            window_size = int(ctx_len * self.window_ratio)
        else:
            window_size = self.window_size
        if window_size > 0:
            scores[:, :, -window_size:] = 1.0

        if layer_idx not in self.score_val:
            self._scored_modules.append(module)
        self.score_val[layer_idx] = scores

    def compress_post(self):
        """
        Obtain the indices of KV pairs to be evicted and write them to
        ``module.masked_key_indices`` (fake compression, consumed by the
        attention patch). Transcribed from kvpress ``FastKVzipPress.compress_post``.
        """
        modules = self._scored_modules
        if self.compression_ratio == 0 or not modules:
            return

        score_val = torch.stack([self.score_val[int(module.layer_idx)] for module in modules], dim=0)
        n_layer, bsz, num_key_value_heads, ctx_len = score_val.shape

        # calculate the pruned KV pairs across layers
        if self.layerwise:
            nl = int(bsz * num_key_value_heads * ctx_len * self.compression_ratio)
            n_pruned_layers = nl * torch.ones(n_layer, device=score_val.device, dtype=torch.int)
        else:
            n_pruned_indices = int(score_val.numel() * self.compression_ratio)
            pruned_indices = torch.topk(-score_val.reshape(-1), n_pruned_indices).indices
            n_tokens_per_layer = bsz * num_key_value_heads * ctx_len
            n_pruned_layers = torch.bincount(pruned_indices // n_tokens_per_layer, minlength=n_layer).int()

        for stack_idx, module in enumerate(modules):
            assert module.config._attn_implementation != "eager", "eager mode not supported"

            scores = score_val[stack_idx]

            # Compute bottom-k across heads
            n_pruned = n_pruned_layers[stack_idx].cpu()
            indices = torch.topk(-scores.reshape(bsz, -1), n_pruned, dim=1).indices.flatten().cpu()

            # Save indices to mask during the attention mechanism. See attention_patch.py for details
            batch_indices = torch.arange(bsz, device=n_pruned.device).repeat_interleave(n_pruned)
            head_indices = indices // ctx_len
            seq_indices = indices % ctx_len
            module.masked_key_indices = (batch_indices, head_indices, seq_indices)
