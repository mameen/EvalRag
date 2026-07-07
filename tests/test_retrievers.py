from evalragkit.core.types import Chunk
from evalragkit.retrievers.keyword import BM25Retriever


def test_bm25_retriever():
    chunks = [
        Chunk(text="RAG combines retrieval and generation", index=0, source="a"),
        Chunk(text="Python is a programming language", index=1, source="b"),
        Chunk(text="Vector databases store embeddings for retrieval", index=2, source="c"),
    ]
    r = BM25Retriever()
    r.add(chunks)
    result = r.retrieve("retrieval augmented generation", k=2)
    assert len(result.chunks) == 2
    assert result.chunks[0].text == chunks[0].text


def test_bm25_empty():
    r = BM25Retriever()
    result = r.retrieve("anything")
    assert result.chunks == []


def test_bm25_reset():
    r = BM25Retriever()
    r.add([Chunk(text="hello world", index=0, source="x")])
    r.reset()
    result = r.retrieve("hello")
    assert result.chunks == []
