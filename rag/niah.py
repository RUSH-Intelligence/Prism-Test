from ABCs import BenchmarkDataset, PredictionResult


import json
from pathlib import Path
from typing import Iterator, Tuple

class NIAH(BenchmarkDataset):

    def __init__(self):
        # Point this to the folder containing qa_1.jsonl, qa_2.jsonl, etc.
        self.directory_path = Path("../datasets/Prism-Data/1M")
        
        # Grab all .jsonl files in directory
        self.file_paths = list(self.directory_path.glob("*.jsonl"))

        self.dataset_len = len(self.file_paths)

    def __iter__(self):
        for file_path in self.file_paths:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Read the first (and only) line of the file
                line = f.readline()
                data = json.loads(line)
                
                yield data["context"], data["question"], data["answer"]
        
    def __len__(self) -> int:
        return self.dataset_len

    def evaluate(self, query: str, expected_answer: str, actual_result: PredictionResult) -> dict:
        if expected_answer.lower() in actual_result.answer.lower():
            return {"is_correct": True, "time": actual_result.execution_time_seconds}
        else:
            return {"is_correct": False, "time": actual_result.execution_time_seconds}