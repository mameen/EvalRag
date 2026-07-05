from pathlib import Path

from evalrag.chunkers.token import TokenChunker
from evalrag.core.types import Document


def test_token_chunker_splits():
    doc = Document(path=Path("test.txt"), text="a" * 2500, metadata={})
    chunker = TokenChunker(chunk_size=1000, chunk_overlap=200)
    chunks = chunker.chunk(doc)
    assert len(chunks) >= 3
    assert all(c.source == "test.txt" for c in chunks)


def test_token_chunker_overlap():
    doc = Document(path=Path("test.txt"), text="word " * 500, metadata={})
    chunker = TokenChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk(doc)
    assert len(chunks) > 1
    for i in range(1, len(chunks)):
        assert chunks[i].metadata["start"] < chunks[i - 1].metadata["end"]
