from evalrag.core.types import Chunk, EvaluationScore, ExperimentResult, QueryResult, RetrievalResult
from evalrag.exploration.inspector import Inspector
from evalrag.exploration.reporter import Reporter


def _make_result(name: str, score: float) -> ExperimentResult:
    r = ExperimentResult(name=name)
    r.query_results.append(
        QueryResult(
            query="q",
            ground_truth="gt",
            answer="a",
            retrieval=RetrievalResult(query="q", chunks=[], scores=[]),
            scores=[EvaluationScore(metric="faithfulness", value=score)],
        )
    )
    return r


def test_chunk_stats():
    chunks = [Chunk(text="hello world", index=0, source="a"), Chunk(text="hi", index=1, source="b")]
    stats = Inspector.chunk_stats(chunks)
    assert stats["count"] == 2
    assert stats["min_length"] == 2
    assert stats["max_length"] == 11


def test_compare():
    r1 = _make_result("exp1", 0.8)
    r2 = _make_result("exp2", 0.6)
    rows = Inspector.compare([r1, r2])
    assert len(rows) == 2
    assert rows[0]["name"] == "exp1"


def test_reporter_table():
    r = _make_result("test", 0.9)
    table = Reporter.to_table([r])
    assert "test" in table
    assert "faithfulness" in table
