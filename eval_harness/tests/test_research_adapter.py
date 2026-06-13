from __future__ import annotations

import unittest
from unittest.mock import patch

from eval_harness.hf_adapter import HFGenerateConfig
from eval_harness.prefill_methods.base import PrefillMethod
from eval_harness.research_adapter import CacheConfig, ResearchAdapter
from eval_harness.kv_compression import (
    DecodingSketch,
    KnormSketch,
    PrefillDecodingSketch,
    RandomSketch,
)


class _FakePipe:
    def __init__(self):
        self.calls = []

    def __call__(self, context, **kwargs):
        self.calls.append((context, kwargs))
        return {"answer": f"ok:{len(context)}"}


class _FakeCacheAdapter:
    def initialize_cache(self, cache):
        return cache

    def get_seq_length(self, cache):
        return 5


class _HybridLikeCache:
    def __len__(self):
        return 3

    def get_seq_length(self, layer_idx=0):
        if layer_idx == 1:
            raise ValueError("linear layer")
        return 5


class TestResearchAdapterSketchSelection(unittest.TestCase):
    def _shell(self, cfg: CacheConfig):
        adapter = object.__new__(ResearchAdapter)
        adapter._cache_cfg = cfg
        adapter._max_context_length = cfg.max_context_length
        adapter._pipe = _FakePipe()
        adapter._cache_adapter = _FakeCacheAdapter()
        return adapter

    def test_build_none_sketch(self):
        adapter = self._shell(CacheConfig(sketch_name="none"))
        self.assertIsNone(adapter._build_sketch(adapter._cache_cfg))

    def test_build_knorm_sketch(self):
        adapter = self._shell(CacheConfig(sketch_name="knorm", compression_ratio=0.3))
        sketch = adapter._build_sketch(adapter._cache_cfg)
        self.assertIsInstance(sketch, KnormSketch)
        self.assertAlmostEqual(sketch.compression_ratio, 0.3)

    def test_build_random_sketch(self):
        adapter = self._shell(CacheConfig(sketch_name="random", compression_ratio=0.2))
        self.assertIsInstance(adapter._build_sketch(adapter._cache_cfg), RandomSketch)

    def test_build_decoding_sketch(self):
        cfg = CacheConfig(sketch_name="decoding_knorm", compression_interval=7, target_size=123)
        adapter = self._shell(cfg)
        sketch = adapter._build_sketch(cfg)
        self.assertIsInstance(sketch, DecodingSketch)
        self.assertEqual(sketch.compression_interval, 7)
        self.assertEqual(sketch.target_size, 123)

    def test_build_prefill_decoding_sketch(self):
        cfg = CacheConfig(sketch_name="prefill_decoding_knorm")
        adapter = self._shell(cfg)
        self.assertIsInstance(adapter._build_sketch(cfg), PrefillDecodingSketch)

    def test_unknown_sketch_raises(self):
        adapter = self._shell(CacheConfig(sketch_name="unknown_x"))
        with self.assertRaises(ValueError):
            adapter._build_sketch(adapter._cache_cfg)


class TestResearchAdapterGenerate(unittest.TestCase):
    def test_generate_uses_pipeline_and_returns_answers(self):
        adapter = object.__new__(ResearchAdapter)
        adapter._cache_cfg = CacheConfig(log_cache_seq_len=False)
        adapter._max_context_length = 4096
        adapter._sketch = None
        adapter._prefill_method = PrefillMethod()
        adapter._pipe = _FakePipe()
        adapter._cache_adapter = _FakeCacheAdapter()

        cfg = HFGenerateConfig(max_tokens=5)
        outs = adapter.generate(["hello", "world"], cfg)

        self.assertEqual(outs, ["ok:5", "ok:5"])
        self.assertEqual(len(adapter._pipe.calls), 2)
        self.assertEqual(adapter._pipe.calls[0][1]["max_new_tokens"], 5)
        self.assertEqual(adapter._pipe.calls[0][1]["max_context_length"], 4096)

    def test_log_cache_seq_lengths_tolerates_linear_layers(self):
        # Should not raise when some layers do not expose sequence lengths.
        ResearchAdapter._log_cache_seq_lengths(_HybridLikeCache(), _FakeCacheAdapter())


class TestResearchAdapterInitRopeScaling(unittest.TestCase):
    @patch("eval_harness.research_adapter.SketchTextGenerationPipeline")
    @patch("eval_harness.research_adapter.HFAdapter.__init__", autospec=True)
    @patch("eval_harness.research_adapter.create_cache_adapter")
    def test_init_builds_cache_adapter_and_pipeline(
        self, mock_create_cache_adapter, mock_hf_init, mock_pipe
    ):
        mock_create_cache_adapter.return_value = _FakeCacheAdapter()

        def _hf_init(inst, **kwargs):
            inst._model = object()
            inst._tokenizer = object()

        mock_hf_init.side_effect = _hf_init

        cfg = CacheConfig(max_context_length=65536)
        adapter = ResearchAdapter(model="dummy/model", cache_config=cfg)

        self.assertTrue(mock_create_cache_adapter.called)
        self.assertIsNotNone(adapter._cache_adapter)


if __name__ == "__main__":
    unittest.main()
