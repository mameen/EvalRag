from evalrag.ranking.metrics import (
    RankingEvaluator,
    average_precision,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_precision_at_k():
    assert precision_at_k({"a", "b"}, ["a", "c", "b", "d"], 2) == 0.5
    assert precision_at_k({"a", "b"}, ["a", "b", "c", "d"], 2) == 1.0


def test_recall_at_k():
    assert recall_at_k({"a", "b"}, ["a", "c", "b", "d"], 2) == 0.5
    assert recall_at_k({"a", "b"}, ["a", "b", "c", "d"], 4) == 1.0


def test_reciprocal_rank():
    assert reciprocal_rank({"b"}, ["a", "b", "c"]) == 0.5
    assert reciprocal_rank({"a"}, ["a", "b", "c"]) == 1.0
    assert reciprocal_rank({"d"}, ["a", "b", "c"]) == 0.0


def test_average_precision():
    ap = average_precision({"a", "c"}, ["a", "b", "c", "d"])
    assert 0.8 < ap < 0.9


def test_ndcg_at_k():
    assert ndcg_at_k({"a"}, ["a", "b", "c"], 3) == 1.0
    assert ndcg_at_k({"b"}, ["a", "b", "c"], 3) < 1.0


def test_ranking_evaluator():
    evaluator = RankingEvaluator(k_values=[1, 3])
    results = evaluator.rank(
        queries=["q1"],
        retrievals=[["a", "b", "c"]],
        relevance=[{"a", "c"}],
    )
    metrics = {r.metric for r in results}
    assert "mrr" in metrics
    assert "precision@1" in metrics
    assert "ndcg@3" in metrics
    mrr = next(r for r in results if r.metric == "mrr")
    assert mrr.value == 1.0
    assert "q1" in mrr.per_query
