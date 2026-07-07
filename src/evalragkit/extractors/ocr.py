"""Text extraction from images using EasyOCR."""

from __future__ import annotations

from pathlib import Path

from evalragkit.core.types import Document


class OCRExtractor:
    """Extracts text from images (PNG, JPG) using EasyOCR."""

    def __init__(self, languages: list[str] | None = None):
        self._languages = languages or ["en"]
        self._reader = None

    @property
    def reader(self):
        if self._reader is None:
            import easyocr
            self._reader = easyocr.Reader(self._languages)
        return self._reader

    def extract(self, path: str) -> Document:
        results = self.reader.readtext(path)
        text = "\n".join(entry[1] for entry in results)
        return Document(
            path=Path(path),
            text=text,
            metadata={"extractor": "ocr", "languages": self._languages, "detections": len(results)},
        )
