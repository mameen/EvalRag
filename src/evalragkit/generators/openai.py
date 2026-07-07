"""OpenAI-based answer generator."""

from __future__ import annotations

import os

from evalragkit.core.types import Chunk


class OpenAIGenerator:
    """Generates answers using OpenAI chat completions."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        system_prompt: str | None = None,
    ):
        self._model = model
        self._api_key = api_key or os.environ["OPENAI_API_KEY"]
        self._system_prompt = system_prompt or (
            "Answer the question using only the provided context. "
            "If the context doesn't contain the answer, say so."
        )
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self._api_key)
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
