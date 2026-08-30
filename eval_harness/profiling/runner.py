"""The measurement core: time the REAL research path, instrumented from outside.

Design rule: **do not re-implement any of the generation path.**  Everything is
measured by instrumenting the shipped code, so the benchmark cannot drift away
from what the eval harness actually runs:

* ``pipe._run_prefill`` is wrapped at the *instance* level  -> prefill time + the
  post-prefill cache, captured before the question is ever fed.
* forward hooks on the top-level ``model`` fire ONLY for the question block and
  the decode steps -- prefill calls ``model.model`` (the decoder stack) directly,
  so there is no ambiguity to disentangle.
* ``compressor.forward_hook`` is wrapped at the instance level -> per-layer
  compression cost (figure F6), with no subclassing and no class monkeypatch.
* ``generation_config.eos_token_id = [-1]`` disables the EOS ``break`` at
  ``research_pipeline.py:477`` so every cell runs an identical step count.  No
  token id equals -1, so the loop always runs ``max_new_tokens - 1`` iterations.

Two facts about this harness that the reported numbers depend on, both verified
against transformers 5.9.0:

* ``Pipeline.__call__`` -- not ``_forward`` -- supplies ``torch.no_grad``
  (``pipelines/base.py:1157-1163``).  We call ``_forward`` directly, so we must
  supply it ourselves or a 128K prefill builds an autograd graph.
* ``DynamicCache`` grows by ``torch.cat`` every decode step
  (``cache_utils.py:143-144``); there is no pre-allocation.  Per step the cache is
  read for the concat, written for the concat, and read again by attention -- so
  KV traffic is ~3x the naive model, and compression looks *better* here than it
  would in a paged engine.  Reported, not hidden.
"""

from __future__ import annotations

import contextlib
import gc
import statistics
import time as _time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from eval_harness.kv_compression.cache_adapter import create_cache_adapter
from eval_harness.profiling import stats as pstats
from eval_harness.profiling.cell import PerfCell
from eval_harness.profiling.kvsize import kv_cache_accounting
from eval_harness.profiling.prompts import build_exact_prompt
from eval_harness.profiling.timers import make_recorder, synchronize

DEFAULT_QUESTION = "\n\nQuestion: summarize the passage above.\nAnswer:"


@dataclass
class BenchRuntime:
    model: Any
    tokenizer: Any
    pipe: Any
    adapter: Any
    device: str
    hf_model: str
    attn_impl_actual: str
    dtype: str
    weights_bytes: int
    max_position_embeddings: int
    n_layers: int


def load_runtime(
    hf_model: str,
    *,
    dtype: str = "bfloat16",
    attn_impl: str = "sdpa",
    trust_remote_code: bool = True,
    max_model_len: Optional[int] = None,
    dequantize_fp8: bool = False,
) -> BenchRuntime:
    """Load through ResearchAdapter so the production loader path is used."""
    from eval_harness.research_adapter import ResearchAdapter, ResearchConfig

    kw: Dict[str, Any] = {"attn_implementation": attn_impl}
    if dequantize_fp8:
        kw["dequantize_fp8"] = True
    adapter = ResearchAdapter(
        model=hf_model,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        research_config=ResearchConfig(max_context_length=max_model_len),
        **kw,
    )
    model = adapter._model
    device = str(next(model.parameters()).device)
    weights = sum(p.numel() * p.element_size() for p in model.parameters())
    weights += sum(b.numel() * b.element_size() for b in model.buffers())

    actual = getattr(model.config, "_attn_implementation", None) or "unknown"
    if actual != attn_impl:
        # hf_adapter._load_model:376-386 silently falls back FA2 -> sdpa -> bare.
        # A node where flash_attn fails to import would otherwise produce a
        # quietly incomparable cell.
        raise RuntimeError(
            f"attn_implementation fell back: requested {attn_impl!r}, got {actual!r}. "
            "Refusing to produce an incomparable timing cell."
        )

    text_cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    return BenchRuntime(
        model=model,
        tokenizer=adapter._tokenizer,
        pipe=adapter._pipe,
        adapter=adapter,
        device=device,
        hf_model=hf_model,
        attn_impl_actual=actual,
        dtype=dtype,
        weights_bytes=weights,
        max_position_embeddings=int(getattr(text_cfg, "max_position_embeddings", 0) or 0),
        n_layers=int(getattr(text_cfg, "num_hidden_layers", 0) or 0),
    )


@contextlib.contextmanager
def _eos_disabled(model):
    """Force a fixed decode step count without touching research_pipeline."""
    gcfg = model.generation_config
    saved = gcfg.eos_token_id
    gcfg.eos_token_id = [-1]
    try:
        yield
    finally:
        gcfg.eos_token_id = saved


class _StepHooks:
    """Times each top-level ``model(...)`` call: [0] = question block, [1:] = decode.

    Two series are recorded, because they answer different questions:

    * ``device_ms[i]`` -- the CUDA-event span of forward *i*.  This is the GPU
      work (including any idle bubbles while the host enqueues the next kernel).
    * ``period_ms[i]`` -- the wall interval between the START of forward *i* and
      the START of forward *i+1*.  This is the true **token-to-token latency**:
      it also covers the ``logits.argmax()`` and the blocking ``new_id.item()``
      at ``research_pipeline.py:475-477`` that sit between two forwards.  The
      ``.item()`` is a device sync, so by the time forward *i+1* starts the GPU
      has genuinely finished forward *i* -- which is what makes this interval a
      complete, non-overlapping period.

    ``period_ms`` is the headline latency; reporting only the forward span would
    understate inter-token latency by the whole host-side gap.
    """

    def __init__(self, model, capacity: int, device: str):
        self.rec = make_recorder(capacity, device)
        self.starts: List[float] = []
        self._h1 = model.register_forward_pre_hook(self._pre, with_kwargs=True)
        self._h2 = model.register_forward_hook(self._post, with_kwargs=True)

    def _pre(self, module, args, kwargs):
        self.starts.append(_time.perf_counter())
        self.rec.start()

    def _post(self, module, args, kwargs, output):
        self.rec.stop()

    def periods_ms(self) -> List[float]:
        return [(b - a) * 1000.0 for a, b in zip(self.starts, self.starts[1:])]

    def remove(self):
        self._h1.remove()
        self._h2.remove()


def _build_compressor(method: str, ratio: float, kwargs: dict, schedule=None):
    from eval_harness.research_adapter import ResearchAdapter, ResearchConfig

    cfg = ResearchConfig(
        kv_compressor=method,
        kv_compressor_kwargs=dict(kwargs or {}),
        compression_ratio=ratio,
        compression_schedule=schedule,
    )
    # Reuse the production builder: registry lookup, compression_ratio setdefault,
    # and the decoding_knorm / prefill_decoding_knorm special cases.
    shell = object.__new__(ResearchAdapter)
    return shell._build_kv_compressor(cfg)


def _run_once(runtime: BenchRuntime, cell: PerfCell, ctx_ids, q_ids, *, measure_compression: bool):
    """One full repeat: prefill + question block + fixed-length decode."""
    model, pipe, device = runtime.model, runtime.pipe, runtime.device
    cache_adapter = create_cache_adapter(model)
    cache = cache_adapter.initialize_cache(None)
    compressor = None if cell.is_anchor else _build_compressor(
        cell.method, cell.compression_ratio, cell.kv_compressor_kwargs, cell.compression_schedule
    )

    out: Dict[str, Any] = {}
    prefill_rec = make_recorder(1, device)
    comp_rec = None
    if measure_compression and compressor is not None:
        comp_rec = make_recorder(max(4 * runtime.n_layers, 8), device)
        inner_hook = compressor.forward_hook

        def timed_hook(module, inputs, kwargs, output, _inner=inner_hook, _rec=comp_rec):
            _rec.start()
            try:
                return _inner(module, inputs, kwargs, output)
            finally:
                _rec.stop()

        compressor.forward_hook = timed_hook   # instance attribute; __call__ registers self.forward_hook

    inner_prefill = pipe._run_prefill

    def timed_prefill(*args, **kw):
        synchronize(device)
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        # Wall must ENCLOSE the device span (start before, stop after the trailing
        # sync) or the async launch makes wall < device and the host gap negative.
        t0 = _time.perf_counter()
        prefill_rec.start()
        try:
            return inner_prefill(*args, **kw)
        finally:
            prefill_rec.stop()
            synchronize(device)
            out["prefill_wall_enclosing_ms"] = (_time.perf_counter() - t0) * 1000.0
            c = kw.get("cache") or (args[1] if len(args) > 1 else None)
            out["kv"] = kv_cache_accounting(c).to_dict()
            if device.startswith("cuda"):
                out["peak_alloc_prefill"] = int(torch.cuda.max_memory_allocated())
                torch.cuda.reset_peak_memory_stats()

    pipe._run_prefill = timed_prefill
    hooks = _StepHooks(model, cell.decode_steps + 4, device)
    try:
        with torch.no_grad(), _eos_disabled(model):
            answers = pipe._forward(
                {"context_ids": ctx_ids, "questions_ids": [q_ids]},
                max_new_tokens=cell.decode_steps + 1,   # question block yields token 0
                kv_compressor=compressor,
                cache=cache,
                cache_adapter=cache_adapter,
            )
        synchronize(device)
    finally:
        hooks.remove()
        pipe._run_prefill = inner_prefill

    calls_dev = hooks.rec.read_device_ms() or hooks.rec.read_wall_ms()
    periods = hooks.periods_ms()
    out["prefill_ms"] = (prefill_rec.read_device_ms() or prefill_rec.read_wall_ms())[0]
    out["prefill_wall_ms"] = out.get("prefill_wall_enclosing_ms", out["prefill_ms"])
    out["question_block_ms"] = calls_dev[0] if calls_dev else None
    # periods[0] is question-block -> first decode step; periods[1:] are the
    # steady-state token-to-token intervals. The final step has no successor, so
    # it contributes no period -- n_decode_calls - 1 samples per repeat.
    out["step_ms"] = list(periods[1:])
    # Aligned with step_ms element-for-element: periods[1+i] is the period of
    # decode forward 1+i, so the device series must stop at the same forward.
    # (The final forward has no successor and therefore no period.)
    out["step_device_ms"] = list(calls_dev[1:len(periods)])
    out["n_calls"] = len(calls_dev)
    out["answer_head"] = (answers[0] or "")[:60]
    if comp_rec is not None:
        comp_ms = comp_rec.read_device_ms() or comp_rec.read_wall_ms()
        out["compress_ms"] = comp_ms
    if device.startswith("cuda"):
        out["peak_alloc_decode"] = int(torch.cuda.max_memory_allocated())
        out["peak_reserved"] = int(torch.cuda.max_memory_reserved())
        ms = torch.cuda.memory_stats()
        out["num_alloc_retries"] = int(ms.get("num_alloc_retries", 0))
        out["num_ooms"] = int(ms.get("num_ooms", 0))

    del cache, compressor
    return out


def time_cell(
    runtime: BenchRuntime,
    cell: PerfCell,
    *,
    question: str = DEFAULT_QUESTION,
    prompt_seed: int = 42,
    measure_compression: bool = True,
) -> Dict[str, Any]:
    """Warm up, then measure ``cell.repeats`` full repeats. Returns the payload."""
    tok = runtime.tokenizer
    q_ids = tok(question, return_tensors="pt", add_special_tokens=False)["input_ids"]
    reserve = int(q_ids.shape[1]) + cell.decode_steps + 8
    _, ctx_ids = build_exact_prompt(
        tok, cell.context_tokens, seed=prompt_seed,
        max_model_len=runtime.max_position_embeddings or None, reserve=reserve,
    )

    alloc_before = int(torch.cuda.memory_allocated()) if runtime.device.startswith("cuda") else 0
    gc.collect()
    if runtime.device.startswith("cuda"):
        torch.cuda.empty_cache()      # once, before warmup -- never between repeats
    gc.freeze()

    # Warm at the EXACT context length: a shorter warmup leaves the caching
    # allocator's pool too small, so repeat 1 still pays cudaMalloc for tens of GB.
    for _ in range(cell.warmup_repeats):
        _run_once(runtime, cell, ctx_ids, q_ids, measure_compression=False)

    gc.collect()
    gc.disable()
    reps: List[Dict[str, Any]] = []
    try:
        for i in range(cell.repeats):
            reps.append(_run_once(runtime, cell, ctx_ids, q_ids,
                                  measure_compression=measure_compression and i == 0))
    finally:
        gc.enable()
        gc.unfreeze()

    return _aggregate(runtime, cell, ctx_ids, q_ids, reps, alloc_before)


def _aggregate(runtime, cell, ctx_ids, q_ids, reps, alloc_before) -> Dict[str, Any]:
    all_steps = [ms for r in reps for ms in r["step_ms"]]
    prefills = [r["prefill_ms"] for r in reps]
    prefill_walls = [r["prefill_wall_ms"] for r in reps]
    qblocks = [r["question_block_ms"] for r in reps if r["question_block_ms"] is not None]
    per_repeat_tps = [pstats.decode_throughput_total(r["step_ms"]) for r in reps]
    per_repeat_tps = [t for t in per_repeat_tps if t]
    spread = None
    if len(per_repeat_tps) > 1:
        med = statistics.median(per_repeat_tps)
        spread = (max(per_repeat_tps) - min(per_repeat_tps)) / med if med else None

    kv = reps[-1].get("kv", {})
    s0 = kv.get("seq_len_max", 0)
    q_len = int(q_ids.shape[1])
    s_end = s0 + q_len + cell.decode_steps
    ttfts = [p + (qb or 0.0) for p, qb in zip(prefills, qblocks)] if qblocks else []

    payload: Dict[str, Any] = {
        "prefill": {
            "tokens": int(ctx_ids.shape[1]),
            "raw_ms": prefills,
            "summary": (pstats.summarize(prefills).to_dict() if prefills else None),
            "wall_summary": (pstats.summarize(prefill_walls).to_dict() if prefill_walls else None),
            # wall - device: host work off the CUDA stream. Expect ~0 for knorm and
            # a real gap for snapkv (scores.max().item() once per layer).
            "host_gap_ms": (statistics.fmean(prefill_walls) - statistics.fmean(prefills))
            if prefills else None,
            "throughput_tok_s": pstats.prefill_throughput(
                int(ctx_ids.shape[1]), statistics.median(prefills)) if prefills else None,
        },
        "ttft": {
            "question_block_ms": (pstats.summarize(qblocks).to_dict() if qblocks else None),
            "ttft_ms": (pstats.summarize(ttfts).to_dict() if ttfts else None),
            "definition": "prefill + question-block forward; the first token comes from "
                          "generate_answer:454-466, not from the decode loop",
        },
        "decode": {
            "steps_per_repeat": cell.decode_steps,
            "n_samples": len(all_steps),
            "raw_ms": [r["step_ms"] for r in reps],
            "per_step": (pstats.summarize(all_steps).to_dict() if all_steps else None),
            "throughput_tok_s": pstats.decode_throughput_total(all_steps),
            "throughput_tok_s_median_based": pstats.decode_throughput_median(all_steps),
            "throughput_by_repeat": per_repeat_tps,
            "forward_device_ms": (pstats.summarize(
                [ms for r in reps for ms in r.get("step_device_ms", [])]).to_dict()
                if any(r.get("step_device_ms") for r in reps) else None),
            "repeat_spread": spread,
            "cache_len_start": s0,
            "cache_len_end": s_end,
            "eos_disabled": True,
            "latency_definition": "token-to-token period: interval between the start of "
                                  "consecutive decode forwards, so it includes the argmax and "
                                  "the blocking .item() at research_pipeline.py:475-477. "
                                  "'forward_device_ms' is the CUDA-event span of the forward "
                                  "alone; the gap between them is host-side overhead.",
            "answer_head": reps[-1].get("answer_head"),
        },
        "kv_cache": kv,
        "memory": {
            "weights_bytes": runtime.weights_bytes,
            "allocated_before_cell_bytes": alloc_before,
            "peak_alloc_prefill_bytes": reps[-1].get("peak_alloc_prefill"),
            "peak_alloc_decode_bytes": reps[-1].get("peak_alloc_decode"),
            "peak_reserved_bytes": reps[-1].get("peak_reserved"),
            "num_alloc_retries": reps[-1].get("num_alloc_retries", 0),
            "num_ooms": reps[-1].get("num_ooms", 0),
        },
    }
    comp = next((r.get("compress_ms") for r in reps if r.get("compress_ms")), None)
    if comp:
        payload["compression_stage"] = {
            "n_calls": len(comp),
            "total_ms": sum(comp),
            "per_call_ms": comp,
            "frac_of_prefill_pct": 100.0 * sum(comp) / statistics.median(prefills)
            if prefills else None,
        }
    return payload
