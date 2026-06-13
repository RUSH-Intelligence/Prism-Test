"""vLLM-based evaluation harness compatible with KVPress benchmark scoring."""

from .config import EvalConfig

__all__ = [
    "EvalConfig",
    "EvalRunner",
    "HFAdapter",
    "HFGenerateConfig",
    "ResearchAdapter",
    "ResearchConfig",
    "SketchTextGenerationPipeline",
    "KVCompressor",
    "ScorerKVCompressor",
    "KnormSketch",
    "RandomSketch",
    "DecodingSketch",
    "PrefillDecodingSketch",
]


def __getattr__(name: str):
    if name == "EvalRunner":
        from .runner import EvalRunner
        return EvalRunner
    if name == "HFAdapter":
        from .hf_adapter import HFAdapter
        return HFAdapter
    if name == "HFGenerateConfig":
        from .hf_adapter import HFGenerateConfig
        return HFGenerateConfig
    if name in ("ResearchAdapter", "ResearchConfig"):
        from .research_adapter import ResearchAdapter, ResearchConfig
        return ResearchAdapter if name == "ResearchAdapter" else ResearchConfig
    if name in (
        "SketchTextGenerationPipeline",
        "KVCompressor",
        "ScorerKVCompressor",
        "KnormSketch",
        "RandomSketch",
        "DecodingSketch",
        "PrefillDecodingSketch",
    ):
        from .kv_compression import (
            KVCompressor,
            DecodingSketch,
            KnormSketch,
            PrefillDecodingSketch,
            RandomSketch,
            ScorerKVCompressor,
        )
        from .research_pipeline import SketchTextGenerationPipeline

        mapping = {
            "SketchTextGenerationPipeline": SketchTextGenerationPipeline,
            "KVCompressor": KVCompressor,
            "ScorerKVCompressor": ScorerKVCompressor,
            "KnormSketch": KnormSketch,
            "RandomSketch": RandomSketch,
            "DecodingSketch": DecodingSketch,
            "PrefillDecodingSketch": PrefillDecodingSketch,
        }
        return mapping[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
