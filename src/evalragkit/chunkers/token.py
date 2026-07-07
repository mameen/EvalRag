"""Token-based text chunking."""

from __future__ import annotations

from evalragkit.core.types import Chunk, Document


class TokenChunker:
    """Splits text into chunks by character count with overlap.

    For token-precise splitting, use LangChainTokenChunker which requires tiktoken.
    This implementation uses character-based splitting as a zero-dependency default.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.text
        chunks = []
        start = 0
        index = 0

        while start < len(text):
            end = start + self._chunk_size
            chunk_text = text[start:end]
            if chunk_text.strip():
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        index=index,
                        source=document.name,
                        metadata={"start": start, "end": min(end, len(text))},
                    )
                )
                index += 1
            start += self._chunk_size - self._chunk_overlap

        return chunks
