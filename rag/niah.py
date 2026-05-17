from ABCs import BenchmarkDataset, PredictionResult

import json
from pathlib import Path
from typing import Iterable, Union


class NIAH(BenchmarkDataset):

    def __init__(self, dataset_name: str = "250K"):
        # Resolve from repo root so running location is less fragile.
        dataset_root = Path(__file__).resolve().parents[1] / "datasets" / "Prism-Data"
        self.directory_path = dataset_root / dataset_name

        if not self.directory_path.exists():
            raise ValueError(f"Dataset directory does not exist: {self.directory_path}")

        # Deterministic ordering for reproducible benchmark output.
        self.file_paths = sorted(self.directory_path.glob("*.jsonl"))
        if not self.file_paths:
            raise ValueError(f"No JSONL files found in dataset directory: {self.directory_path}")

        self.dataset_len = len(self.file_paths)

    def __iter__(self):
        for file_path in self.file_paths:
            with open(file_path, "r", encoding="utf-8") as f:
                # Read the first (and only) line of the file
                line = f.readline()
                data = json.loads(line)

                if "answer" in data:
                    yield file_path, data["context"], data["question"], data["answer"]
                elif "accepted_answers" in data:
                    yield file_path, data["context"], data["question"], data["accepted_answers"]
                else:
                    raise ValueError(f"Datapoint {file_path} doesn't have any answer field!")

    def __len__(self) -> int:
        return self.dataset_len

    def evaluate(self, query: str, expected_answer: Union[str, Iterable[str]], actual_result: PredictionResult) -> dict:
        predicted_answer = (actual_result.answer or "").lower()

        if isinstance(expected_answer, list):
            is_correct = any(str(candidate).lower() in predicted_answer for candidate in expected_answer)
        else:
            is_correct = str(expected_answer).lower() in predicted_answer

        return {"is_correct": is_correct, "time": actual_result.execution_time_seconds}
