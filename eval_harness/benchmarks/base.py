from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import pandas as pd


@dataclass(frozen=True)
class BenchmarkInfo:
    name: str
    description: str
    default_subsets: List[str]


class Benchmark(ABC):
    @property
    @abstractmethod
    def info(self) -> BenchmarkInfo:
        raise NotImplementedError

    @abstractmethod
    def load(self, subsets: Optional[List[str]] = None) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def score(self, df: pd.DataFrame) -> Dict[str, object]:
        raise NotImplementedError

    def resolve_subsets(self, subsets: Optional[Iterable[str]]) -> List[str]:
        if subsets is None:
            return list(self.info.default_subsets)
        cleaned = [s.strip() for s in subsets if s and s.strip()]
        return cleaned or list(self.info.default_subsets)
