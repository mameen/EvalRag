"""Vector-based retriever."""

from __future__ import annotations

from evalragkit.core.protocols import Embedder, Store
from evalragkit.core.types import RetrievalResult


class VectorRetriever:
    """Retrieves chunks by embedding the query and searching the vector store."""

    def __init__(self, embedder: Embedder, store: Store):
        self._embedder = embedder
        self._store = store

    def retrieve(self, query: str, k: int = 10) -> RetrievalResult:
        embedding = self._embedder.embed([query])[0]
        result = self._store.query(embedding, k)
        return RetrievalResult(query=query, chunks=result.chunks, scores=result.scores)
