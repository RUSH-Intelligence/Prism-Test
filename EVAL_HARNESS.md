# Prism-Test vLLM Eval Harness

This is a fully standalone evaluation harness for Prism-Test.

It runs vLLM inference and computes benchmark metrics inside Prism-Test itself.

## Supported benchmarks

- `aime`
- `aime2024`
- `aime2025`
- `infinite_bench`
- `mock_benchmark`
- `ruler`
- `ruler16k`
- `ruler32k`
- `ruler64k`
- `ruler128k`
- `longbench`
- `longbenchv2`
- `loft`
- `loft_rag`
- `loogle`
- `prism1m`
- `zero_scrolls`

## User flow

1. Select a HuggingFace model and a benchmark in `evaluate_config.yaml`.
2. Run `python -m eval_harness.cli run --config_file ./evaluate_config.yaml`; the vLLM adapter executes generation over each context/question pair.
3. Read `predictions.csv` and `metrics.json` in the run directory to inspect model answers and benchmark scores.

## Layout

- `eval_harness/config.py`: run configuration and output naming
- `eval_harness/benchmarks/`: standalone benchmark loaders and scorers
- `eval_harness/vllm_adapter.py`: vLLM model wrapper
- `eval_harness/runner.py`: dataset load, generation, scoring, save
- `eval_harness/cli.py`: command-line entrypoint
- `evaluate_config.yaml`: default configuration

## Setup

1. Install dependencies in your environment:

```bash
pip install -r requirements-eval.txt
```

2. Configure your benchmark in `evaluate_config.yaml`.

## Run

From the Prism-Test root:

```bash
python -m eval_harness.cli run --config_file ./evaluate_config.yaml
```

Or override values on CLI:

```bash
python -m eval_harness.cli run \
  --config_file ./evaluate_config.yaml \
  --benchmark longbench \
  --subsets narrativeqa,hotpotqa \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --max_new_tokens 128
```

## Outputs

Each run writes to:

- `results/<benchmark>__<model>__vllm__.../predictions.csv`
- `results/<benchmark>__<model>__vllm__.../metrics.json`
- `results/<benchmark>__<model>__vllm__.../config.yaml`

## Notes

- The harness batches by shared context for efficient long-context evaluation.
- Prefix caching is enabled by default for better throughput on repeated contexts.
- If your model needs custom vLLM settings, add them under `llm_kwargs` in `evaluate_config.yaml`.
