from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class PredictionResult:
    answer: str
    execution_time_seconds: float
    retrieved_context: Optional[List[Tuple[str, float]]] = None
    metadata: Optional[Dict] = None


class RAGSystem(ABC):
    @abstractmethod
    def setup(self, document_text: str) -> None:
        """Prepares the system for the specific document."""
        pass

    @abstractmethod
    def predict(self, query: str) -> PredictionResult:
        """Executes the query against the setup environment."""
        pass

    @abstractmethod
    def teardown(self) -> None:
        """Destroys the environment."""
        pass
