from __future__ import annotations

from typing import List

from .rag.base import RAGSystem
from .rag.one_pass_rag import OnePassRAG


class RAGAdapter:
    """Wraps RAGSystem as an eval harness inference backend."""

    def __init__(self) -> None:
        self._rag: RAGSystem = OnePassRAG()

    def generate_for_context(self, context: str, questions: List[str]) -> List[str]:
        """Index `context` once via RAG, then answer each question."""
        self._rag.setup(context)
        answers = [self._rag.predict(q).answer for q in questions]
        self._rag.teardown()
        return answers
