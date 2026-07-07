"""Sentence transformer embeddings via ChromaDB's default embedding function."""

from __future__ import annotations


class SentenceTransformerEmbedder:
    """Embeds text using sentence-transformers (via chromadb's built-in default)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
            self._model = SentenceTransformerEmbeddingFunction(model_name=self._model_name)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(v) for v in e] for e in self.model(texts)]
