"""Door 4 (reserved) — MLP / FFN methods.

This package is a **reservation, not an implementation**.  The three-door
research interface (Positional / Attention / KV-compression) widens the
framework from "long-context only" toward general inference efficiency; the
natural fourth lever is the **MLP/FFN block** — MoE routing, activation
sparsity, expert pruning.

Nothing is built here yet.  When Door 4 lands it should mirror the other doors:
a ``MLPMethod`` base + an auto-discovery registry (see
``eval_harness/kv_compression/`` for the pattern), installed by the
``ResearchGenerationPipeline`` as the innermost nested context manager so the
door stack becomes::

    positional_method(model)
      → attention_method(model)
        → kv_compressor(model)
          → mlp_method(model)        # door 4  ← reserved here
"""
