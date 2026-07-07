"""Ollama local generator via OpenAI-compatible API."""

from __future__ import annotations

from evalragkit.core.types import Chunk


class OllamaGenerator:
    """Generates answers using a local Ollama instance."""

    def __init__(
        self,
        model: str = "llama3",
        base_url: str = "http://localhost:11434/v1",
        system_prompt: str | None = None,
    ):
        self._model = model
        self._base_url = base_url
        self._system_prompt = system_prompt or (
            "Answer the question using only the provided context. "
            "If the context doesn't contain the answer, say so."
        )
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key="ollama", base_url=self._base_url)
        return self._client

    def generate(self, question: str, context: list[Chunk]) -> str:
        context_text = "\n\n---\n\n".join(c.text for c in context)
        response = self.client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {question}"},
            ],
        )
        return response.choices[0].message.content or ""
