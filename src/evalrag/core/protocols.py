"""Protocol definitions for every pipeline stage.

Each protocol is a structural interface. Implement the methods and pass it in --
no inheritance required. This is the contract between the core and all adapters.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from evalrag.core.types import Chunk, Document, EvaluationScore, RankingResult, RetrievalResult


@runtime_checkable
class Extractor(Protocol):
    """Extracts raw text from a file."""

    def extract(self, path: str) -> Document: ...


@runtime_checkable
class Chunker(Protocol):
    """Splits a document into retrievable chunks."""

    def chunk(self, document: Document) -> list[Chunk]: ...


@runtime_checkable
class Embedder(Protocol):
    """Produces vector embeddings for text."""

    def embed(self, texts: list[str]) -> list[list[float]]: ...


@runtime_checkable
class Store(Protocol):
    """Persists and queries vector embeddings."""

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None: ...

    def query(self, embedding: list[float], k: int) -> RetrievalResult: ...

    def reset(self) -> None: ...


@runtime_checkable
class Retriever(Protocol):
    """Finds relevant chunks for a query."""

    def retrieve(self, query: str, k: int = 10) -> RetrievalResult: ...


@runtime_checkable
class Generator(Protocol):
    """Generates an answer given a question and context chunks."""

    def generate(self, question: str, context: list[Chunk]) -> str: ...


@runtime_checkable
class Evaluator(Protocol):
    """Scores an answer against ground truth."""

    def evaluate(
        self,
        question: str,
        answer: str,
        ground_truth: str,
        context: list[Chunk],
    ) -> list[EvaluationScore]: ...


@runtime_checkable
class Ranker(Protocol):
    """Computes ranking quality metrics for retrieval results."""

    def rank(
        self,
        queries: list[str],
        retrievals: list[RetrievalResult],
        relevance: dict[str, list[str]],
    ) -> list[RankingResult]: ...
