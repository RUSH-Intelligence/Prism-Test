"""Hidden-state KV recovery: minimal fine-tuning that aligns a KV-compressed
*student* with a full-cache *teacher* that shares the same pretrained weights.

Layout
------
config.py        ``RecoveryConfig`` (YAML + dotted CLI overrides) and the ONE shared
                 ``research_config`` block used by the student's compressor and by
                 the ``compressed`` / ``compressed_recovered`` evaluation arms.
model_spec.py    per-family isolation (decoder layers, full-attention layers,
                 parameter-name prefixes) built on ``kv_compression.base`` helpers.
trainable.py     trainable-subset selection, freezing, parameter accounting.
sensitivity.py   compression-sensitivity layer selection (``trainable.layers: sensitivity``):
                 E_l = ||H_dense - H_comp||_F / (||H_dense||_F + eps) per layer on held-out
                 calibration windows -> top-k eligible layers.
hidden_states.py forward-hook capture of decoder-layer outputs (+ final norm).
alignment.py     layer / position resolution and the alignment losses.
student.py       teacher / student execution through the production
                 ``ResearchAdapter`` + ``ResearchGenerationPipeline`` code path.
data.py          JSONL long-text windows -> ``[context | suffix]`` examples.
trainer.py       FP32-master AdamW loop, sanity checks, logs.
checkpoint.py    lightweight weight-delta checkpoints and their loader.
provenance.py    reproducibility metadata, seeding, determinism.
metrics.py       recovery metrics, paired bootstrap, representation metrics.
eval_configs.py  three-way (dense / compressed / compressed_recovered) EvalConfig
                 builder with drift guards.

The eval path imports nothing from here except ``checkpoint.apply_delta``
(lazily, from ``hf_adapter`` when ``llm_kwargs.weight_delta`` is set).
"""
from __future__ import annotations

__all__ = [
    "RecoveryConfig",
    "load_config",
    "research_config_dict",
    "build_research_config",
    "apply_delta",
    "write_delta",
]


def __getattr__(name: str):
    if name in ("RecoveryConfig", "load_config", "research_config_dict", "build_research_config"):
        from . import config as _config
        return getattr(_config, name)
    if name in ("apply_delta", "write_delta"):
        from . import checkpoint as _checkpoint
        return getattr(_checkpoint, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
