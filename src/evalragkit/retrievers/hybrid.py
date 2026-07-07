"""Hybrid retriever combining vector and keyword search."""

from __future__ import annotations

from evalragkit.core.protocols import Retriever
from evalragkit.core.types import Chunk, RetrievalResult


class HybridRetriever:
    """Combines two retrievers using reciprocal rank fusion."""

    def __init__(self, retrievers: list[Retriever], weights: list[float] | None = None, rrf_k: int = 60):
        self._retrievers = retrievers
        self._weights = weights or [1.0] * len(retrievers)
        self._rrf_k = rrf_k

    def retrieve(self, query: str, k: int = 10) -> RetrievalResult:
        chunk_scores: dict[str, float] = {}
        chunk_map: dict[str, Chunk] = {}

        for retriever, weight in zip(self._retrievers, self._weights):
            result = retriever.retrieve(query, k=k * 2)
            for rank, chunk in enumerate(result.chunks):
                key = f"{chunk.source}_{chunk.index}"
                rrf_score = weight / (self._rrf_k + rank + 1)
                chunk_scores[key] = chunk_scores.get(key, 0.0) + rrf_score
                chunk_map[key] = chunk

        ranked = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return RetrievalResult(
            query=query,
            chunks=[chunk_map[key] for key, _ in ranked],
            scores=[score for _, score in ranked],
        )
