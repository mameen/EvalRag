"""Core data types for EvalRag."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Chunk:
    text: str
    index: int
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Document:
    path: Path
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.path.name


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    chunks: list[Chunk]
    scores: list[float]


@dataclass(frozen=True)
class GenerationResult:
    query: str
    answer: str
    context: list[Chunk]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationScore:
    metric: str
    value: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QueryResult:
    query: str
    ground_truth: str
    answer: str
    retrieval: RetrievalResult
    scores: list[EvaluationScore]


@dataclass(frozen=True)
class RankingResult:
    metric: str
    value: float
    per_query: dict[str, float] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    name: str
    timestamp: datetime = field(default_factory=datetime.now)
    query_results: list[QueryResult] = field(default_factory=list)
    ranking_results: list[RankingResult] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def mean_scores(self) -> dict[str, float]:
        totals: dict[str, list[float]] = {}
        for qr in self.query_results:
            for s in qr.scores:
                totals.setdefault(s.metric, []).append(s.value)
        return {k: sum(v) / len(v) for k, v in totals.items()}
