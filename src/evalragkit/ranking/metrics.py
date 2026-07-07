"""Ranking evaluation metrics: MRR, NDCG, MAP, Precision@k, Recall@k."""

from __future__ import annotations

import math

from evalragkit.core.types import RankingResult


def f1_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    p = precision_at_k(relevant, retrieved, k)
    r = recall_at_k(relevant, retrieved, k)
    if p + r == 0:
        return 0.0
    return 2 * (p * r) / (p + r)


def precision_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    retrieved_k = retrieved[:k]
    if not retrieved_k:
        return 0.0
    return len(relevant & set(retrieved_k)) / k


def recall_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    if not relevant:
        return 0.0
    retrieved_k = retrieved[:k]
    return len(relevant & set(retrieved_k)) / len(relevant)


def average_precision(relevant: set[str], retrieved: list[str]) -> float:
    if not relevant:
        return 0.0
    hits = 0
    sum_precision = 0.0
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            hits += 1
            sum_precision += hits / (i + 1)
    return sum_precision / len(relevant)


def reciprocal_rank(relevant: set[str], retrieved: list[str]) -> float:
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    retrieved_k = retrieved[:k]
    dcg = sum(
        1.0 / math.log2(i + 2) for i, doc_id in enumerate(retrieved_k) if doc_id in relevant
    )
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else 0.0


class RankingEvaluator:
    """Evaluates retrieval quality using standard ranking metrics."""

    def __init__(self, k_values: list[int] | None = None):
        self._k_values = k_values or [1, 3, 5, 10]

    def rank(
        self,
        queries: list[str],
        retrievals: list[list[str]],
        relevance: list[set[str]],
    ) -> list[RankingResult]:
        per_query_scores: dict[str, dict[str, float]] = {}
        for query, retrieved, relevant in zip(queries, retrievals, relevance):
            scores = {}
            scores["mrr"] = reciprocal_rank(relevant, retrieved)
            scores["map"] = average_precision(relevant, retrieved)
            for k in self._k_values:
                scores[f"precision@{k}"] = precision_at_k(relevant, retrieved, k)
                scores[f"recall@{k}"] = recall_at_k(relevant, retrieved, k)
                scores[f"ndcg@{k}"] = ndcg_at_k(relevant, retrieved, k)
                scores[f"f1@{k}"] = f1_at_k(relevant, retrieved, k)
            per_query_scores[query] = scores

        all_metrics = list(next(iter(per_query_scores.values())).keys()) if per_query_scores else []
        results = []
        for metric in all_metrics:
            per_q = {q: s[metric] for q, s in per_query_scores.items()}
            mean_val = sum(per_q.values()) / len(per_q) if per_q else 0.0
            results.append(RankingResult(metric=metric, value=mean_val, per_query=per_q))
        return results
