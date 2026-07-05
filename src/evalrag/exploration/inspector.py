"""Inspect pipeline components and experiment results."""

from __future__ import annotations

from evalrag.core.types import Chunk, ExperimentResult


class Inspector:
    """Provides utilities to explore and inspect experiment artifacts."""

    @staticmethod
    def chunk_stats(chunks: list[Chunk]) -> dict:
        if not chunks:
            return {"count": 0, "avg_length": 0, "min_length": 0, "max_length": 0}
        lengths = [len(c.text) for c in chunks]
        return {
            "count": len(chunks),
            "avg_length": sum(lengths) / len(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "sources": list({c.source for c in chunks}),
        }

    @staticmethod
    def score_summary(result: ExperimentResult) -> dict:
        means = result.mean_scores
        per_query = []
        for qr in result.query_results:
            scores = {s.metric: s.value for s in qr.scores}
            per_query.append({"question": qr.question, "scores": scores})
        return {"mean_scores": means, "per_query": per_query, "num_queries": len(result.query_results)}

    @staticmethod
    def compare(results: list[ExperimentResult]) -> list[dict]:
        rows = []
        for r in results:
            rows.append({"name": r.name, "mean_scores": r.mean_scores, "num_queries": len(r.query_results)})
        return rows
