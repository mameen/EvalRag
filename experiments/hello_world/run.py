"""EvalRag hello world — BM25 vs Vector vs Hybrid showdown.

Proves that hybrid search outperforms either approach alone,
using a 10-chapter AI knowledge base with keyword AND semantic queries.
Runs fully offline — SentenceTransformer embeddings are local, no API keys.
"""

from __future__ import annotations

import json
from pathlib import Path

from evalragkit.chunkers.token import TokenChunker
from evalragkit.core.types import EvaluationScore, Chunk
from evalragkit.exploration.reporter import Reporter
from evalragkit.extractors.unstructured import PlainTextExtractor
from evalragkit.ranking.metrics import RankingEvaluator
from evalragkit.retrievers.hybrid import HybridRetriever
from evalragkit.retrievers.keyword import BM25Retriever
from evalragkit.retrievers.vector import VectorRetriever
from evalragkit.embedders.sentence_transformer import SentenceTransformerEmbedder
from evalragkit.stores.chromadb import ChromaDBStore

HERE = Path(__file__).parent


class OverlapEvaluator:
    """Scores based on significant keyword overlap between answer and ground truth."""

    def evaluate(self, question: str, answer: str, ground_truth: str, context: list[Chunk]) -> list[EvaluationScore]:
        import re
        gt_tokens = set(re.findall(r"\w+", ground_truth.lower()))
        ans_tokens = set(re.findall(r"\w+", answer.lower()))
        significant = {t for t in gt_tokens if len(t) > 3}
        if not significant:
            return [EvaluationScore(metric="answer_overlap", value=0.0)]
        overlap = len(significant & ans_tokens) / len(significant)
        return [EvaluationScore(metric="answer_overlap", value=round(overlap, 4))]


class ContextReturnGenerator:
    """Returns concatenated context as the answer (measures retrieval quality directly)."""

    def generate(self, question: str, context: list[Chunk]) -> str:
        return "\n".join(c.text for c in context)


def main():
    print("=" * 65)
    print("  EvalRag Hello World — BM25 vs Vector vs Hybrid")
    print("=" * 65)

    # Load data
    kb_path = str(HERE / "data" / "knowledge_base.txt")
    eval_path = HERE / "data" / "eval_dataset.json"
    eval_data = json.loads(eval_path.read_text())

    kw_count = sum(1 for q in eval_data if q["type"] == "keyword")
    sem_count = sum(1 for q in eval_data if q["type"] == "semantic")
    print(f"\nKnowledge base: {kb_path}")
    print(f"Eval queries:   {len(eval_data)} ({kw_count} keyword, {sem_count} semantic)")

    # Extract and chunk
    extractor = PlainTextExtractor()
    chunker = TokenChunker(chunk_size=500, chunk_overlap=80)
    doc = extractor.extract(kb_path)
    chunks = chunker.chunk(doc)
    print(f"Chunks:         {len(chunks)} (avg {sum(len(c.text) for c in chunks) // len(chunks)} chars)")

    # Build real embeddings (local, no API key)
    print("\nLoading SentenceTransformer model (first run downloads ~90MB)...")
    embedder = SentenceTransformerEmbedder()
    embeddings = embedder.embed([c.text for c in chunks])
    print(f"Embedded {len(embeddings)} chunks ({len(embeddings[0])} dimensions)")

    # Build stores
    store = ChromaDBStore(collection_name="hello_world")
    store.reset()
    store.add(chunks, embeddings)

    bm25 = BM25Retriever()
    bm25.add(chunks)

    # Build retrievers
    vector_retriever = VectorRetriever(embedder=embedder, store=store)
    hybrid_retriever = HybridRetriever(
        retrievers=[vector_retriever, bm25],
        weights=[0.7, 0.3],
        rrf_k=10,
    )

    generator = ContextReturnGenerator()
    evaluator = OverlapEvaluator()
    ranker = RankingEvaluator(k_values=[1, 3, 5])

    configs = [
        ("BM25 Keyword", bm25),
        ("Vector Semantic", vector_retriever),
        ("Hybrid (BM25+Vector)", hybrid_retriever),
    ]

    from evalragkit.core.types import ExperimentResult, QueryResult

    all_results: list[ExperimentResult] = []
    k = 5

    for exp_name, retriever in configs:
        print(f"\n--- Running: {exp_name} ---")
        result = ExperimentResult(name=exp_name)
        queries = []
        retrieved_ids_list = []
        relevance_sets = []

        for qa in eval_data:
            retrieval = retriever.retrieve(qa["question"], k=k)
            answer = generator.generate(qa["question"], retrieval.chunks)
            scores = evaluator.evaluate(qa["question"], answer, qa["ground_truth"], retrieval.chunks)

            result.query_results.append(QueryResult(
                query=qa["question"],
                ground_truth=qa["ground_truth"],
                answer=answer[:500],
                retrieval=retrieval,
                scores=scores,
            ))

            queries.append(qa["question"])
            retrieved_ids = [f"{c.source}_{c.index}" for c in retrieval.chunks]
            retrieved_ids_list.append(retrieved_ids)

            relevant_ids = set()
            for chunk in chunks:
                chunk_id = f"{chunk.source}_{chunk.index}"
                for term in qa.get("relevant_chunks", []):
                    if term.lower() in chunk.text.lower():
                        relevant_ids.add(chunk_id)
                        break
            relevance_sets.append(relevant_ids)

        result.ranking_results = ranker.rank(queries, retrieved_ids_list, relevance_sets)

        mean = result.mean_scores
        f1 = next((rr.value for rr in result.ranking_results if rr.metric == "f1@5"), 0)
        mrr = next((rr.value for rr in result.ranking_results if rr.metric == "mrr"), 0)
        mapv = next((rr.value for rr in result.ranking_results if rr.metric == "map"), 0)
        print(f"  F1@5={f1:.3f}  MRR={mrr:.3f}  MAP={mapv:.3f}  overlap={mean.get('answer_overlap', 0):.3f}")

        all_results.append(result)

    # Compute embedding space data for visualization
    print("\nComputing embedding projections...")
    query_texts = [q["question"] for q in eval_data]
    query_embeddings = embedder.embed(query_texts)

    # PCA projection to 2D (all chunks + all queries together)
    import math
    all_vecs = embeddings + query_embeddings
    n = len(all_vecs)
    dim = len(all_vecs[0])

    # Center
    mean_vec = [sum(all_vecs[j][d] for j in range(n)) / n for d in range(dim)]
    centered = [[all_vecs[j][d] - mean_vec[d] for d in range(dim)] for j in range(n)]

    # Covariance top-2 eigenvectors via power iteration
    def mat_vec(mat, v):
        return [sum(mat[i][d] * v[d] for d in range(len(v))) for i in range(len(mat))]

    def dot(a, b):
        return sum(x * y for x, y in zip(a, b))

    def normalize(v):
        norm = math.sqrt(dot(v, v)) or 1.0
        return [x / norm for x in v]

    # Project centered data: X^T is dim x n, X is n x dim
    # We compute X^T @ X @ v iteratively (power iteration on covariance)
    def power_iter(centered_data, exclude=None, iters=50):
        v = normalize([1.0 + 0.01 * i for i in range(dim)])
        for _ in range(iters):
            proj = [dot(row, v) for row in centered_data]
            new_v = [sum(proj[j] * centered_data[j][d] for j in range(len(centered_data))) for d in range(dim)]
            if exclude:
                c = dot(new_v, exclude)
                new_v = [new_v[d] - c * exclude[d] for d in range(dim)]
            v = normalize(new_v)
        return v

    pc1 = power_iter(centered)
    pc2 = power_iter(centered, exclude=pc1)

    # Project all points
    projections_2d = [[dot(centered[j], pc1), dot(centered[j], pc2)] for j in range(n)]
    chunk_proj = projections_2d[:len(chunks)]
    query_proj = projections_2d[len(chunks):]

    # Cosine similarity matrix (queries × chunks)
    def cosine_sim(a, b):
        d = dot(a, b)
        na = math.sqrt(dot(a, a)) or 1.0
        nb = math.sqrt(dot(b, b)) or 1.0
        return d / (na * nb)

    sim_matrix = []
    for qi, qe in enumerate(query_embeddings):
        row = []
        for ci, ce in enumerate(embeddings):
            row.append(round(cosine_sim(qe, ce), 4))
        sim_matrix.append(row)

    # Retrieval edges per experiment (query_idx -> [chunk_idx, ...])
    retrieval_edges = {}
    for result in all_results:
        edges = []
        for qi, qr in enumerate(result.query_results):
            chunk_indices = []
            for rc in qr.retrieval.chunks:
                chunk_indices.append(rc.index)
            edges.append(chunk_indices[:3])  # top 3 for readability
        retrieval_edges[result.name] = edges

    embedding_viz = {
        "chunk_points": [{"x": p[0], "y": p[1], "index": i, "preview": chunks[i].text[:80]} for i, p in enumerate(chunk_proj)],
        "query_points": [{"x": p[0], "y": p[1], "index": i, "question": eval_data[i]["question"], "type": eval_data[i]["type"]} for i, p in enumerate(query_proj)],
        "similarity_matrix": sim_matrix,
        "query_labels": [q["question"][:50] for q in eval_data],
        "chunk_labels": [f"C{i}" for i in range(len(chunks))],
        "retrieval_edges": retrieval_edges,
    }

    # Build report context
    report_context = {
        "documents": [{
            "name": Path(kb_path).name,
            "path": kb_path,
            "chunks": len(chunks),
            "chars": len(doc.text),
            "summary": "10-chapter knowledge base covering the history of computing, AI/ML fundamentals, "
                       "neural networks, transformers, RAG pipelines, evaluation metrics, and vector databases.",
        }],
        "dataset": {
            "total": len(eval_data),
            "keyword_count": kw_count,
            "semantic_count": sem_count,
            "path": str(eval_path),
            "summary": "Balanced mix of keyword queries (exact names, dates, numbers) and semantic queries "
                       "(paraphrased concepts using different vocabulary than the source).",
        },
        "questions": [
            {"question": q["question"], "ground_truth": q["ground_truth"], "type": q["type"]}
            for q in eval_data
        ],
        "embedding_viz": embedding_viz,
    }

    # Generate reports
    report_dir = str(HERE / "reports")
    json_path = Reporter.to_json(all_results, output_dir=report_dir, context=report_context)
    html_path = Reporter.to_html(all_results, output_dir=report_dir, context=report_context)

    print(f"\n{'=' * 65}")
    print("  RESULTS")
    print(f"{'=' * 65}")
    print(f"{'Experiment':35s} {'F1@5':>8s} {'MRR':>8s} {'MAP':>8s} {'Overlap':>8s}")
    print("-" * 65)
    for r in all_results:
        f1 = next((rr.value for rr in r.ranking_results if rr.metric == "f1@5"), 0)
        mrr = next((rr.value for rr in r.ranking_results if rr.metric == "mrr"), 0)
        mapv = next((rr.value for rr in r.ranking_results if rr.metric == "map"), 0)
        ov = r.mean_scores.get("answer_overlap", 0)
        print(f"  {r.name:33s} {f1:7.1%} {mrr:7.1%} {mapv:7.1%} {ov:7.1%}")

    # Lift calculations
    f1s = {r.name: next((rr.value for rr in r.ranking_results if rr.metric == "f1@5"), 0) for r in all_results}
    hybrid_f1 = f1s.get("Hybrid (BM25+Vector)", 0)
    bm25_f1 = f1s.get("BM25 Keyword", 0)
    vector_f1 = f1s.get("Vector Semantic", 0)
    if bm25_f1 > 0:
        print(f"\n  Hybrid vs BM25 (traditional):  {(hybrid_f1 - bm25_f1) / bm25_f1 * 100:+.1f}% F1 lift")
    if vector_f1 > 0:
        print(f"  Hybrid vs Vector (semantic):   {(hybrid_f1 - vector_f1) / vector_f1 * 100:+.1f}% F1 lift")

    print(f"\n  JSON: {json_path}")
    print(f"  HTML: {html_path}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
