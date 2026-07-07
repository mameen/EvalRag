"""Experiment runner -- the unit of composition in EvalRag.

An Experiment holds references to all pipeline components and a Q&A dataset.
Call run() to execute the full pipeline and collect results.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evalragkit.core.protocols import Chunker, Embedder, Evaluator, Extractor, Generator, Ranker, Retriever, Store
from evalragkit.core.types import (
    Chunk,
    EvaluationScore,
    ExperimentResult,
    QueryResult,
    RetrievalResult,
)

logger = logging.getLogger(__name__)


@dataclass
class QAPair:
    question: str
    ground_truth: str
    relevant_chunk_ids: list[str] = field(default_factory=list)


@dataclass
class Experiment:
    name: str
    extractor: Extractor
    chunker: Chunker
    embedder: Embedder
    store: Store
    retriever: Retriever
    generator: Generator
    evaluator: Evaluator
    ranker: Ranker | None = None
    config: dict[str, Any] = field(default_factory=dict)

    def ingest(self, path: str) -> list[Chunk]:
        logger.info("Extracting: %s", path)
        document = self.extractor.extract(path)

        logger.info("Chunking: %d chars", len(document.text))
        chunks = self.chunker.chunk(document)

        logger.info("Embedding: %d chunks", len(chunks))
        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed(texts)

        logger.info("Storing: %d vectors", len(embeddings))
        self.store.add(chunks, embeddings)

        return chunks

    def run(self, dataset: list[QAPair], k: int = 10) -> ExperimentResult:
        result = ExperimentResult(name=self.name, config=self.config)

        queries = []
        retrievals = []

        for qa in dataset:
            logger.info("Query: %s", qa.question[:80])

            retrieval = self.retriever.retrieve(qa.question, k=k)
            queries.append(qa.question)
            retrievals.append(retrieval)

            answer = self.generator.generate(qa.question, retrieval.chunks)

            scores = self.evaluator.evaluate(
                question=qa.question,
                answer=answer,
                ground_truth=qa.ground_truth,
                context=retrieval.chunks,
            )

            result.query_results.append(
                QueryResult(
                    query=qa.question,
                    ground_truth=qa.ground_truth,
                    answer=answer,
                    retrieval=retrieval,
                    scores=scores,
                )
            )

        if self.ranker:
            relevance_map = {
                qa.question: qa.relevant_chunk_ids
                for qa in dataset
                if qa.relevant_chunk_ids
            }
            if relevance_map:
                retrieved_ids = [
                    [f"{c.source}_{c.index}" for c in r.chunks]
                    for r in retrievals
                ]
                relevance_sets = [set(relevance_map.get(q, [])) for q in queries]
                result.ranking_results = self.ranker.rank(queries, retrieved_ids, relevance_sets)

        logger.info("Experiment '%s' complete: %d queries, mean scores: %s", self.name, len(dataset), result.mean_scores)
        return result

    @staticmethod
    def save_result(result: ExperimentResult, path: str) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "name": result.name,
            "timestamp": result.timestamp.isoformat(),
            "config": result.config,
            "mean_scores": result.mean_scores,
            "ranking_results": [
                {"metric": r.metric, "value": r.value, "per_query": r.per_query}
                for r in result.ranking_results
            ],
            "query_results": [
                {
                    "query": qr.query,
                    "ground_truth": qr.ground_truth,
                    "answer": qr.answer,
                    "retrieval": {
                        "chunks": [c.text[:200] for c in qr.retrieval.chunks],
                        "scores": qr.retrieval.scores,
                    },
                    "scores": [{"metric": s.metric, "value": s.value} for s in qr.scores],
                }
                for qr in result.query_results
            ],
        }

        out.write_text(json.dumps(data, indent=2, default=str))
        logger.info("Results saved to %s", out)

    @staticmethod
    def load_result(path: str) -> ExperimentResult:
        data = json.loads(Path(path).read_text())
        result = ExperimentResult(name=data["name"], config=data.get("config", {}))

        for qr_data in data.get("query_results", []):
            chunks = [Chunk(text=t, index=i, source="loaded") for i, t in enumerate(qr_data["retrieval"]["chunks"])]
            result.query_results.append(
                QueryResult(
                    query=qr_data["query"],
                    ground_truth=qr_data["ground_truth"],
                    answer=qr_data["answer"],
                    retrieval=RetrievalResult(
                        query=qr_data["query"],
                        chunks=chunks,
                        scores=qr_data["retrieval"]["scores"],
                    ),
                    scores=[EvaluationScore(metric=s["metric"], value=s["value"]) for s in qr_data["scores"]],
                )
            )
        return result
