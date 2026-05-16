# RAG Benchmark Quickstart

This folder runs a simple retrieval-augmented generation benchmark using:
- `LanceDB` as the vector store
- `BAAI/bge-small-en-v1.5` as the embedding model
- `llama3` served through `Ollama`

The benchmark reads JSONL cases from:
- `../datasets/Prism-Data/1M/*.jsonl`

## 1) Go To This Folder

From the project root directory (`Prism-Test`), run:

```bash
cd rag
```

## 2) Create And Activate Python Env

```bash
conda create -n prism_rag python=3.10 -y
conda activate prism_rag
pip install -r requirements.txt
```

## 3) Start Ollama And Pull Model

```bash
mkdir -p ~/.ollama
curl -fsSL https://ollama.com/download/ollama-linux-amd64.tar.zst | tar --zstd -x -C ~/.ollama
~/.ollama/bin/ollama pull llama3
CUDA_VISIBLE_DEVICES=0 nohup ~/.ollama/bin/ollama serve > ollama.log 2>&1 &
```

Optional GPU wake lock (use only if you hit Ollama timeout / GPU sleep issues):

```bash
nohup python -c "import torch, time; torch.zeros(1).cuda(); time.sleep(86400)" > wake_lock.log 2>&1 &
```

Stop processes when needed:

```bash
pkill ollama
pkill -f wake_lock
```

## 4) Run The Benchmark

```bash
python benchmark.py
```

## 5) Sequence Of Events During Testing

For each dataset file in `../datasets/Prism-Data/1M`:
1. `rag.setup(context)` creates a fresh local `./lancedb` index for that case.
2. `rag.predict(question)` queries the index via Ollama and records latency.
3. `niah.evaluate(...)` checks whether the expected answer appears in the model output.
4. `rag.teardown()` deletes `./lancedb` so the next case starts clean.

Console output is printed per case:
- `rag_answer: ...`
- `eval_result: {'is_correct': <bool>, 'time': <seconds>}`

## Notes

- Run from `rag/` so relative dataset paths resolve correctly.
- `llama3` only needs to be pulled once per machine.
