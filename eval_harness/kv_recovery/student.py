"""Teacher / student execution through the PRODUCTION research path.

Both passes reuse ``ResearchAdapter`` (loader), ``ResearchGenerationPipeline._run_prefill``
(context prefill) and the exact segment call ``generate_answer`` makes for the question
block — ``model(input_ids, past_key_values=cache, position_ids=arange(T, T+L), logits_to_keep)``
with ABSOLUTE positions and no ``cache_position`` (HF derives the physical slots from the
pruned cache) — so training and evaluation share one code path.

* Teacher: a separate model instance, full cache, ``no_grad``, states detached.
* Student: the trainable instance; the context prefill runs inside ``with compressor(model)``
  exactly as ``_forward`` does (``set_phase("prefill")`` -> ``_run_prefill`` ->
  ``maybe_slice_prefill``), under ``no_grad`` unless ``student.prefill_grad``; the suffix
  forward runs with gradients.

Compressor hooks live on the model object (and graft ``rotary_emb`` permanently), so the
teacher must be a DIFFERENT instance that never enters the compressor context
(``assert_no_hooks`` guards this).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch import nn

from .config import RecoveryConfig, StudentCfg, build_research_config, model_llm_kwargs
from .hidden_states import StateKey, capture_layer_outputs
from .model_spec import ModelSpec, attention_module_of, decoder_layers, inspect_model, language_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# data types
# ---------------------------------------------------------------------------
@dataclass
class Example:
    id: str
    ctx_ids: torch.Tensor          # [1, T]
    suffix_ids: torch.Tensor       # [1, L]
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def context_len(self) -> int:
        return int(self.ctx_ids.shape[1])

    @property
    def suffix_len(self) -> int:
        return int(self.suffix_ids.shape[1])


@dataclass
class SegmentOutput:
    states: Dict[StateKey, torch.Tensor]   # key -> [1, L, H]
    logits: Optional[torch.Tensor]         # [1, k, V] or None
    cache_len_after_prefill: int
    cache_len_after_segment: int
    per_layer_cache_len: Dict[int, int]


@dataclass
class PairOutput:
    teacher: SegmentOutput
    student: SegmentOutput
    positions: torch.Tensor                # suffix positions used by the loss


# ---------------------------------------------------------------------------
# loading / building
# ---------------------------------------------------------------------------
def expected_budget(context_len: int, compression_ratio: float) -> int:
    """Mirror of ``ScorerKVCompressor.compress``: ``int(k_len * (1 - r))`` per head."""
    r = float(compression_ratio)
    return int(context_len) if r <= 0 else int(int(context_len) * (1 - r))


def load_adapter(cfg: RecoveryConfig, *, compressed: bool):
    """A ``ResearchAdapter`` loaded through the production loader; refuses a silent
    attention-implementation fallback (mirrors ``profiling/runner.load_runtime``)."""
    from eval_harness.research_adapter import ResearchAdapter

    adapter = ResearchAdapter(
        model=cfg.model.name,
        dtype=cfg.model.dtype,
        trust_remote_code=cfg.model.trust_remote_code,
        seed=cfg.seed,
        max_model_len=cfg.model.max_model_len,
        research_config=build_research_config(cfg, compressed=compressed),
        **model_llm_kwargs(cfg),
    )
    model = adapter._model
    actual = getattr(model.config, "_attn_implementation", None)
    if actual is None:
        tcfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
        actual = getattr(tcfg, "_attn_implementation", None)
    if actual != cfg.model.attn_implementation:
        raise RuntimeError(
            f"attn_implementation fell back: requested {cfg.model.attn_implementation!r}, got {actual!r}"
        )
    model.eval()
    model.requires_grad_(False)
    return adapter


def build_compressor(cfg: RecoveryConfig):
    """The student's compressor from the SAME research_config dict the eval arms use."""
    from eval_harness.kv_compression.compressors.decoding_sketch import DecodingSketch
    from eval_harness.research_adapter import ResearchAdapter

    comp = ResearchAdapter._build_kv_compressor(build_research_config(cfg, compressed=True))
    if comp is None:
        return None
    if isinstance(comp, DecodingSketch) or not comp.fires_on_prefill:
        raise ValueError(
            f"kv_compressor={cfg.kv_compression.kv_compressor!r} does not compress at prefill; "
            "training aligns the post-prefill continuation only (schedules: post_prefill / streaming)."
        )
    return comp


def resolve_segment_mode(model: nn.Module, scfg: StudentCfg) -> str:
    """``block`` unless the model has Mamba layers (same rule as ``generate_answer``)."""
    if scfg.segment_mode in ("block", "token_by_token"):
        return scfg.segment_mode
    from eval_harness.research_pipeline import _model_has_mamba_layers

    return "token_by_token" if _model_has_mamba_layers(model) else "block"


def model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


# ---------------------------------------------------------------------------
# cache helpers
# ---------------------------------------------------------------------------
def layer_cache_lengths(cache, layer_indices: Sequence[int]) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for i in layer_indices:
        layer = cache.layers[i]
        keys = getattr(layer, "keys", None)
        if keys is not None and hasattr(keys, "shape") and len(keys.shape) >= 3:
            out[int(i)] = int(keys.shape[2])
    return out


def assert_budget(cache, spec: ModelSpec, context_len: int, compression_ratio: float, *,
                  suffix_appended: int = 0) -> Dict[int, int]:
    """Every hooked layer holds exactly ``int(T*(1-r)) + suffix_appended`` entries."""
    lengths = layer_cache_lengths(cache, spec.full_attention_layers)
    expect = expected_budget(context_len, compression_ratio) + int(suffix_appended)
    bad = {i: n for i, n in lengths.items() if n != expect}
    if bad or set(lengths) != set(spec.full_attention_layers):
        raise AssertionError(
            f"cache budget mismatch: expected {expect} on layers {list(spec.full_attention_layers)}, got {lengths}"
        )
    return lengths


def assert_no_hooks(model: nn.Module) -> None:
    for i, layer in enumerate(decoder_layers(model)):
        attn = attention_module_of(layer)
        if attn is not None and (attn._forward_hooks or attn._forward_pre_hooks):
            raise AssertionError(f"layer {i}: attention module carries forward hooks (compressor leaked onto this model)")


def prefill_context(adapter, ctx_ids: torch.Tensor, compressor, *, prefill_chunk_size: Optional[int] = None,
                    grad: bool = False):
    """Context prefill exactly as ``ResearchGenerationPipeline._forward`` performs it."""
    model = adapter._model
    pipe = adapter._pipe
    cache_adapter = adapter._cache_adapter
    cache = cache_adapter.initialize_cache(None)
    ctx = ctx_ids.to(model_device(model))
    with torch.set_grad_enabled(bool(grad)):
        if compressor is not None:
            with compressor(model):
                compressor.set_phase("prefill")
                pipe._run_prefill(context_ids=ctx, cache=cache, prefill_chunk_size=prefill_chunk_size,
                                  kv_compressor=compressor)
                cache_adapter.maybe_slice_prefill(cache)
            compressor.set_phase("decode")
        else:
            pipe._run_prefill(context_ids=ctx, cache=cache, prefill_chunk_size=prefill_chunk_size,
                              kv_compressor=None)
    return cache


def segment_forward(model: nn.Module, cache, seg_ids: torch.Tensor, context_len: int, *,
                    logits_to_keep: int = 1, mode: str = "block") -> torch.Tensor:
    """The continuation segment against ``cache`` — the ``generate_answer`` call (block) or its
    Mamba token-by-token branch. Returns logits ``[1, k, V]`` (``k`` = ``logits_to_keep``, or all
    ``L`` positions when ``logits_to_keep`` is 0 / >= L)."""
    device = model_device(model)
    seg = seg_ids.to(device)
    L = int(seg.shape[1])
    position_ids = torch.arange(context_len, context_len + L, device=device).unsqueeze(0)
    if mode == "block":
        out = model(input_ids=seg, past_key_values=cache, position_ids=position_ids,
                    logits_to_keep=int(logits_to_keep))
        return out.logits
    if mode == "token_by_token":
        want_all = logits_to_keep == 0 or logits_to_keep >= L
        keep_from = 0 if want_all else L - int(logits_to_keep)
        chunks: List[torch.Tensor] = []
        for j in range(L):
            out = model(input_ids=seg[:, j:j + 1], past_key_values=cache,
                        position_ids=position_ids[:, j:j + 1], logits_to_keep=1)
            if j >= keep_from:
                chunks.append(out.logits)
        return torch.cat(chunks, dim=1)
    raise ValueError(f"unknown segment mode {mode!r}")


# ---------------------------------------------------------------------------
# teacher / student passes
# ---------------------------------------------------------------------------
def run_segment(adapter, ex: Example, compressor, layer_keys: Sequence[int], *, include_final_norm: bool,
                grad: bool, want_logits: bool, prefill_grad: bool = False,
                prefill_chunk_size: Optional[int] = None, mode: str = "block",
                spec: Optional[ModelSpec] = None, compression_ratio: float = 0.0,
                check_budget: bool = True) -> SegmentOutput:
    model = adapter._model
    cache_adapter = adapter._cache_adapter
    cache = prefill_context(adapter, ex.ctx_ids, compressor, prefill_chunk_size=prefill_chunk_size,
                            grad=bool(grad and prefill_grad))
    spec = spec or inspect_model(model)
    len_after_prefill = int(cache_adapter.get_seq_length(cache))
    if check_budget and prefill_chunk_size is None:
        ratio = compression_ratio if compressor is not None else 0.0
        assert_budget(cache, spec, ex.context_len, ratio)
    keep = ex.suffix_len if want_logits else 1
    with torch.set_grad_enabled(bool(grad)):
        with capture_layer_outputs(model, layer_keys, detach=not grad, include_final_norm=include_final_norm) as cap:
            logits = segment_forward(model, cache, ex.suffix_ids, ex.context_len, logits_to_keep=keep, mode=mode)
    states = cap.states()
    per_layer = layer_cache_lengths(cache, spec.full_attention_layers)
    len_after = int(cache_adapter.get_seq_length(cache))
    del cache
    return SegmentOutput(states=states, logits=logits if want_logits else None,
                         cache_len_after_prefill=len_after_prefill, cache_len_after_segment=len_after,
                         per_layer_cache_len=per_layer)


@torch.no_grad()
def run_teacher(adapter, ex: Example, layer_keys: Sequence[int], *, include_final_norm: bool,
                want_logits: bool, mode: str = "block", spec: Optional[ModelSpec] = None) -> SegmentOutput:
    assert_no_hooks(adapter._model)
    return run_segment(adapter, ex, None, layer_keys, include_final_norm=include_final_norm, grad=False,
                       want_logits=want_logits, mode=mode, spec=spec, compression_ratio=0.0)


def run_student(adapter, ex: Example, compressor, layer_keys: Sequence[int], *, include_final_norm: bool,
                grad: bool, want_logits: bool, prefill_grad: bool = False,
                prefill_chunk_size: Optional[int] = None, mode: str = "block",
                spec: Optional[ModelSpec] = None, compression_ratio: float = 0.0) -> SegmentOutput:
    return run_segment(adapter, ex, compressor, layer_keys, include_final_norm=include_final_norm, grad=grad,
                       want_logits=want_logits, prefill_grad=prefill_grad, prefill_chunk_size=prefill_chunk_size,
                       mode=mode, spec=spec, compression_ratio=compression_ratio)


# ---------------------------------------------------------------------------
# continuation probe (segment forward == full-sequence forward without compression)
# ---------------------------------------------------------------------------
@torch.no_grad()
def probe_block_continuation(model: nn.Module, cache_adapter, *, T: int = 64, L: int = 16, seed: int = 0,
                             rtol: float = 1e-4, layers: Optional[Sequence[int]] = None) -> Dict[str, Any]:
    """Compare per-layer suffix states of ``forward(ctx+seg)`` against ``prefill(ctx)`` followed by
    the block segment forward and the token-by-token segment forward. ``*_ok`` holds when the
    max abs difference is within ``rtol * max(1, max|ref|)`` on every layer."""
    device = model_device(model)
    tcfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    vocab = int(getattr(tcfg, "vocab_size", 256))
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, vocab, (1, T + L), generator=g).to(device)
    layers = list(layers) if layers is not None else list(range(len(decoder_layers(model))))
    lm = language_model(model)

    with capture_layer_outputs(model, layers, detach=True, include_final_norm=False) as cap:
        model(input_ids=ids, logits_to_keep=1)
    ref = {k: v[:, T:] for k, v in cap.states().items()}

    results: Dict[str, Dict[str, float]] = {}
    oks = {}
    for mode in ("block", "token_by_token"):
        cache = cache_adapter.initialize_cache(None)
        lm(input_ids=ids[:, :T], past_key_values=cache)
        with capture_layer_outputs(model, layers, detach=True, include_final_norm=False) as cap:
            segment_forward(model, cache, ids[:, T:], T, logits_to_keep=1, mode=mode)
        got = cap.states()
        ok = True
        for k in layers:
            r, s = ref[k].float(), got[k].float()
            diff = float((r - s).abs().max())
            scale = max(1.0, float(r.abs().max()))
            cos = float(torch.nn.functional.cosine_similarity(r.reshape(-1, r.shape[-1]), s.reshape(-1, s.shape[-1]), dim=-1).min())
            results.setdefault(mode, {})[str(k)] = {"max_abs_diff": diff, "ref_max_abs": scale, "min_cos": cos}
            ok = ok and diff <= rtol * scale
        oks[mode] = ok
        del cache
    return {"T": T, "L": L, "rtol": rtol, "layers": results, "block_ok": oks["block"],
            "token_by_token_ok": oks["token_by_token"]}
