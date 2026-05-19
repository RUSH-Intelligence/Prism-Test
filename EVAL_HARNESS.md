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
- `eval_harness/rag/base.py`: `RAGSystem` ABC and `PredictionResult` dataclass
- `eval_harness/rag/one_pass_rag.py`: OnePassRAG implementation
- `eval_harness/rag_adapter.py`: RAG inference backend wrapper
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

- `results/<benchmark>__<model>__<backend>__.../predictions.csv`
- `results/<benchmark>__<model>__<backend>__.../metrics.json`
- `results/<benchmark>__<model>__<backend>__.../config.yaml`

## Notes

- The harness batches by shared context for efficient long-context evaluation.
- Prefix caching is enabled by default for better throughput on repeated contexts.
- If your model needs custom vLLM settings, add them under `llm_kwargs` in `evaluate_config.yaml`.

---

## RAG Backend

The harness supports a RAG inference backend (`backend: rag` in `evaluate_config.yaml`) that uses the OnePassRAG system: LanceDB vector store, `BAAI/llm-embedder` embeddings, and `llama3.1` served through Ollama.

The OnePassRAG system indexes each document once then answers all questions against that index, matching the single-pass evaluation pattern used in the Prism benchmark paper.

### 1. Install Ollama and pull the model

Ollama only needs to be installed and the model pulled once per machine.

```bash
mkdir -p ~/.ollama
curl -fsSL https://ollama.com/download/ollama-linux-amd64.tar.zst | tar --zstd -x -C ~/.ollama
~/.ollama/bin/ollama pull llama3.1
```

### 2. Start the Ollama server

Run this before launching an eval with `backend: rag`. The server must be running for the duration of the benchmark.

```bash
CUDA_VISIBLE_DEVICES=0 nohup ~/.ollama/bin/ollama serve > ollama.log 2>&1 &
```

> **GPU wake lock (optional):** If you hit Ollama timeouts due to GPU sleep between long eval runs, keep the GPU awake with:
> ```bash
> nohup python -c "import torch, time; torch.zeros(1).cuda(); time.sleep(86400)" > wake_lock.log 2>&1 &
> ```

### 3. Configure and run

Set `backend: rag` in `evaluate_config.yaml`:

```yaml
benchmark: longbench
subsets: qasper
backend: rag
```

Then run as normal:

```bash
python -m eval_harness.cli run --config_file ./evaluate_config.yaml
```

### 4. Stop background processes when done

```bash
pkill ollama
pkill -f wake_lock   # only if you started the wake lock
```
