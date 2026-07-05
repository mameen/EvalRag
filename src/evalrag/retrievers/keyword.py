"""BM25 keyword-based retriever."""

from __future__ import annotations

import math
import re
from collections import Counter

from evalrag.core.types import Chunk, RetrievalResult


class BM25Retriever:
    """Retrieves chunks using BM25 scoring over an in-memory corpus."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self._k1 = k1
        self._b = b
        self._chunks: list[Chunk] = []
        self._doc_freqs: Counter = Counter()
        self._doc_lens: list[int] = []
        self._avg_dl: float = 0.0
        self._token_lists: list[list[str]] = []

    def add(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            tokens = self._tokenize(chunk.text)
            self._chunks.append(chunk)
            self._token_lists.append(tokens)
            self._doc_lens.append(len(tokens))
            self._doc_freqs.update(set(tokens))
        total = sum(self._doc_lens)
        self._avg_dl = total / len(self._doc_lens) if self._doc_lens else 0.0

    def retrieve(self, query: str, k: int = 10) -> RetrievalResult:
        if not self._chunks:
            return RetrievalResult(query=query, chunks=[], scores=[])

        query_tokens = self._tokenize(query)
        n = len(self._chunks)
        scores = []

        for i, doc_tokens in enumerate(self._token_lists):
            tf = Counter(doc_tokens)
            dl = self._doc_lens[i]
            score = 0.0
            for qt in query_tokens:
                if qt not in tf:
                    continue
                df = self._doc_freqs[qt]
                idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
                tf_norm = (tf[qt] * (self._k1 + 1)) / (
                    tf[qt] + self._k1 * (1 - self._b + self._b * dl / self._avg_dl)
                )
                score += idf * tf_norm
            scores.append(score)

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
        return RetrievalResult(
            query=query,
            chunks=[self._chunks[i] for i, _ in ranked],
            scores=[s for _, s in ranked],
        )

    def reset(self) -> None:
        self._chunks.clear()
        self._doc_freqs.clear()
        self._doc_lens.clear()
        self._token_lists.clear()
        self._avg_dl = 0.0

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())
