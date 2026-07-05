"""Text extraction using the unstructured library."""

from __future__ import annotations

from pathlib import Path

from evalrag.core.types import Document


class UnstructuredExtractor:
    """Extracts text from PDF, DOCX, TXT, HTML using the unstructured library."""

    def extract(self, path: str) -> Document:
        from unstructured.partition.auto import partition

        elements = partition(filename=path)
        text = "\n\n".join(str(el) for el in elements)
        return Document(path=Path(path), text=text, metadata={"extractor": "unstructured", "elements": len(elements)})


class PlainTextExtractor:
    """Extracts text from plain text files. No dependencies."""

    def extract(self, path: str) -> Document:
        p = Path(path)
        return Document(path=p, text=p.read_text(encoding="utf-8"), metadata={"extractor": "plaintext"})
