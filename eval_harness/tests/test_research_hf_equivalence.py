from __future__ import annotations

import unittest
from unittest.mock import patch

from eval_harness.hf_adapter import HFGenerateConfig
from eval_harness.research_adapter import CacheConfig, ResearchAdapter


class _FakePipe:
    def __call__(self, context, **kwargs):
        return {"answer": context[:3] + str(kwargs["max_new_tokens"])}


class TestResearchAdapterContract(unittest.TestCase):
    @patch("eval_harness.research_adapter.SketchTextGenerationPipeline")
    @patch("eval_harness.research_adapter.HFAdapter.__init__", autospec=True)
    @patch("eval_harness.research_adapter.AutoConfig.from_pretrained")
    def test_init_builds_pipe_and_sketch(self, mock_cfg, mock_hf_init, mock_pipe):
        class _Cfg:
            max_position_embeddings = 4096

        mock_cfg.return_value = _Cfg()

        def _hf_init(inst, **kwargs):
            inst._model = object()
            inst._tokenizer = object()

        mock_hf_init.side_effect = _hf_init

        cfg = CacheConfig(sketch_name="knorm", compression_ratio=0.5, max_context_length=8192)
        adapter = ResearchAdapter(model="dummy/model", cache_config=cfg)

        self.assertIsNotNone(adapter._sketch)
        self.assertEqual(adapter._max_context_length, 8192)
        self.assertTrue(mock_pipe.called)

    def test_generate_returns_one_answer_per_prompt(self):
        adapter = object.__new__(ResearchAdapter)
        adapter._cache_cfg = CacheConfig()
        adapter._sketch = None
        adapter._max_context_length = 2048
        adapter._pipe = _FakePipe()

        out = adapter.generate(["abcdef", "xyz"], HFGenerateConfig(max_tokens=7))
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0], "abc7")
        self.assertEqual(out[1], "xyz7")


if __name__ == "__main__":
    unittest.main()
