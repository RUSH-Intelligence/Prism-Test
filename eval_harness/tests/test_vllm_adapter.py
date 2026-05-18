import importlib.util
import pathlib
import sys
import types
import unittest


class _FakeTokenizer:
    def encode(self, text):
        return [ord(c) for c in text]


class _FakeGeneratedText:
    def __init__(self, text):
        self.text = text


class _FakeRequestOutput:
    def __init__(self, text):
        self.outputs = [_FakeGeneratedText(text)]


class _FakeLLM:
    last_instance = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        _FakeLLM.last_instance = self

    def get_tokenizer(self):
        return _FakeTokenizer()

    def generate(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return [_FakeRequestOutput("ok")]


class _FakeSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _load_adapter_module_with_fake_vllm():
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = _FakeLLM
    fake_vllm.SamplingParams = _FakeSamplingParams
    sys.modules["vllm"] = fake_vllm

    adapter_path = pathlib.Path(__file__).resolve().parents[1] / "vllm_adapter.py"
    module_name = "_test_vllm_adapter_module"
    spec = importlib.util.spec_from_file_location(module_name, adapter_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class VLLMAdapterGenerateTests(unittest.TestCase):
    def test_generate_passes_token_ids_positionally(self):
        adapter_module = _load_adapter_module_with_fake_vllm()
        adapter = adapter_module.VLLMAdapter(model="dummy/model")
        cfg = adapter_module.VLLMGenerateConfig(max_tokens=8, temperature=0.0, top_p=1.0)

        outputs = adapter.generate(["ab"], cfg)

        self.assertEqual(outputs, ["ok"])
        llm = _FakeLLM.last_instance
        self.assertIsNotNone(llm)
        self.assertEqual(len(llm.calls), 1)

        args, kwargs = llm.calls[0]
        self.assertEqual(args, ([[97, 98]],))
        self.assertIn("sampling_params", kwargs)
        self.assertEqual(kwargs.get("use_tqdm"), False)
        self.assertNotIn("prompt_token_ids", kwargs)
        self.assertNotIn("prompts", kwargs)


if __name__ == "__main__":
    unittest.main()
