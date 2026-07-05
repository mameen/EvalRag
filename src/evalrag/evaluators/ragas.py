"""RAGAS-based evaluation metrics."""

from __future__ import annotations

from evalrag.core.types import Chunk, EvaluationScore


class RagasEvaluator:
    """Evaluates RAG output using RAGAS metrics."""

    def __init__(self, metrics: list[str] | None = None):
        self._metrics = metrics or ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    def evaluate(
        self,
        question: str,
        answer: str,
        ground_truth: str,
        context: list[Chunk],
    ) -> list[EvaluationScore]:
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
        from datasets import Dataset

        metric_map = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
        }
        selected = [metric_map[m] for m in self._metrics if m in metric_map]

        dataset = Dataset.from_dict({
            "question": [question],
            "answer": [answer],
            "ground_truth": [ground_truth],
            "contexts": [[c.text for c in context]],
        })

        result = ragas_evaluate(dataset, metrics=selected)
        scores = []
        for metric_name in self._metrics:
            if metric_name in result:
                scores.append(EvaluationScore(metric=metric_name, value=result[metric_name]))
        return scores
