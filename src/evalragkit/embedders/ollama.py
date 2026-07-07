"""Ollama local embeddings via OpenAI-compatible API."""

from __future__ import annotations


class OllamaEmbedder:
    """Embeds text using a local Ollama instance."""

    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434/v1"):
        self._model = model
        self._base_url = base_url
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key="ollama", base_url=self._base_url)
        return self._client

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(input=texts, model=self._model)
        return [item.embedding for item in response.data]
