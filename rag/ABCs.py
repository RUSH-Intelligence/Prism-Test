from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

@dataclass
class PredictionResult:
    answer: str
    execution_time_seconds: float
    retrieved_context: Optional[List[Tuple[str, float]]] = None
    metadata: Optional[dict] = None # For custom attention timing or memory stats

class BaselineSystem(ABC):
    
    @abstractmethod
    def setup(self, document_text: str) -> None:
        """
        Prepares the system for the specific document.
        - RAG: Spins up LanceDB, chunks, embeds, and indexes the document.
        - Full-Context LLM: Might just load the document into memory.
        """
        pass

    @abstractmethod
    def predict(self, query: str) -> PredictionResult:
        """
        Executes the query against the setup environment.
        """
        pass

    @abstractmethod
    def teardown(self) -> None:
        """
        Destroys the environment.
        - RAG: Deletes the LanceDB directory.
        - Full-Context LLM: Clears the KV cache.
        """
        pass


class BenchmarkDataset(ABC):
    
    @abstractmethod
    def __iter__(self):
        """
        Yields a dictionary or tuple containing:
        (document_text, query, expected_answer)
        """
        pass
        
    @abstractmethod
    def __len__(self) -> int:
        """Returns the total number of test cases."""
        pass

    @abstractmethod
    def evaluate(self, query: str, expected_answer: str, actual_result: PredictionResult) -> dict:
        """
        Compares the expected answer against the actual result.
        Returns a dictionary of metrics.
        
        Example Return:
        {"is_correct": True, "score": 1.0, "hallucinated": False}
        """
        # might start with exact string matching, 
        # but as benchmarks get more sophisticated, 
        # will likely want to swap to an LLM-as-a-judge
        pass