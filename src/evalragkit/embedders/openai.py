"""OpenAI embeddings."""

from __future__ import annotations

import os


class OpenAIEmbedder:
    """Embeds text using OpenAI's embedding API."""

    def __init__(self, model: str = "text-embedding-3-small", api_key: str | None = None):
        self._model = model
        self._api_key = api_key or os.environ["OPENAI_API_KEY"]
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(input=texts, model=self._model)
        return [item.embedding for item in response.data]
