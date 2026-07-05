from evalrag.extractors.unstructured import PlainTextExtractor


def test_plaintext_extractor(sample_text_path):
    ext = PlainTextExtractor()
    doc = ext.extract(sample_text_path)
    assert doc.name == "sample.txt"
    assert "RAG" in doc.text
    assert len(doc.text) > 100
