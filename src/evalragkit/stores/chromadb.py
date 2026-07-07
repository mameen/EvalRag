"""ChromaDB vector store."""

from __future__ import annotations

from evalragkit.core.types import Chunk, RetrievalResult


class ChromaDBStore:
    """Persistent vector store backed by ChromaDB."""

    def __init__(self, collection_name: str = "evalragkit", persist_dir: str | None = None):
        self._collection_name = collection_name
        self._persist_dir = persist_dir
        self._collection = None

    @property
    def collection(self):
        if self._collection is None:
            import chromadb

            if self._persist_dir:
                client = chromadb.PersistentClient(path=self._persist_dir)
            else:
                client = chromadb.EphemeralClient()
            self._collection = client.get_or_create_collection(name=self._collection_name)
        return self._collection

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        metadatas = []
        for c in chunks:
            meta = dict(c.metadata)
            meta["_source"] = c.source
            meta["_index"] = c.index
            metadatas.append(meta)
        self.collection.add(
            ids=[f"{c.source}_{c.index}" for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def query(self, embedding: list[float], k: int) -> RetrievalResult:
        results = self.collection.query(query_embeddings=[embedding], n_results=k)
        chunks = [
            Chunk(
                text=doc,
                index=meta.get("_index", i),
                source=meta.get("_source", "chromadb"),
                metadata={k: v for k, v in meta.items() if not k.startswith("_")},
            )
            for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0]))
        ]
        distances = results["distances"][0] if results["distances"] else [0.0] * len(chunks)
        scores = [1.0 / (1.0 + d) for d in distances]
        return RetrievalResult(query="", chunks=chunks, scores=scores)

    def reset(self) -> None:
        if self._collection is not None:
            import chromadb

            if self._persist_dir:
                client = chromadb.PersistentClient(path=self._persist_dir)
            else:
                client = chromadb.EphemeralClient()
            client.delete_collection(self._collection_name)
            self._collection = None
