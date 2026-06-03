"""vLLM-based evaluation harness compatible with KVPress benchmark scoring."""

from .config import EvalConfig

__all__ = [
    "EvalConfig",
    "EvalRunner",
    "HFAdapter",
    "HFGenerateConfig",
    "ResearchAdapter",
    "CacheConfig",
    "SketchTextGenerationPipeline",
    "BaseSketch",
    "ScorerSketch",
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
    if name in ("ResearchAdapter", "CacheConfig"):
        from .research_adapter import ResearchAdapter, CacheConfig
        return ResearchAdapter if name == "ResearchAdapter" else CacheConfig
    if name in (
        "SketchTextGenerationPipeline",
        "BaseSketch",
        "ScorerSketch",
        "KnormSketch",
        "RandomSketch",
        "DecodingSketch",
        "PrefillDecodingSketch",
    ):
        from .sketch import (
            BaseSketch,
            DecodingSketch,
            KnormSketch,
            PrefillDecodingSketch,
            RandomSketch,
            ScorerSketch,
            SketchTextGenerationPipeline,
        )

        mapping = {
            "SketchTextGenerationPipeline": SketchTextGenerationPipeline,
            "BaseSketch": BaseSketch,
            "ScorerSketch": ScorerSketch,
            "KnormSketch": KnormSketch,
            "RandomSketch": RandomSketch,
            "DecodingSketch": DecodingSketch,
            "PrefillDecodingSketch": PrefillDecodingSketch,
        }
        return mapping[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
