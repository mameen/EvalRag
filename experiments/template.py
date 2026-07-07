"""EvalRag experiment template — copy and customize for your own evaluations.

Steps:
  1. Replace the data paths with your own knowledge base and eval dataset
  2. Configure your retrievers (add/remove/modify configs list)
  3. Adjust chunking params, embedder, weights, k values
  4. Run: PYTHONPATH=src python experiments/my_experiment/run.py

Reports (HTML + JSON) are written to your experiment's reports/ folder.
See experiments/hello_world/ for a complete working example.
See experiments/hello_world/reports/demo_hybrid_vs_bm25_vs_vector.html for sample output.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from evalragkit.chunkers.token import TokenChunker
from evalragkit.core.types import EvaluationScore, Chunk, ExperimentResult, QueryResult
from evalragkit.exploration.reporter import Reporter
from evalragkit.extractors.unstructured import PlainTextExtractor
from evalragkit.ranking.metrics import RankingEvaluator
from evalragkit.retrievers.hybrid import HybridRetriever
from evalragkit.retrievers.keyword import BM25Retriever
from evalragkit.retrievers.vector import VectorRetriever
from evalragkit.embedders.sentence_transformer import SentenceTransformerEmbedder
from evalragkit.stores.chromadb import ChromaDBStore

HERE = Path(__file__).parent

# ---------------------------------------------------------------------------
# 1. CONFIGURE YOUR DATA
# ---------------------------------------------------------------------------
KB_PATH = str(HERE / "data" / "knowledge_base.txt")       # your knowledge base
EVAL_PATH = HERE / "data" / "eval_dataset.json"            # your eval questions

# ---------------------------------------------------------------------------
# 2. CONFIGURE CHUNKING & EMBEDDING
# ---------------------------------------------------------------------------
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
EMBEDDER_MODEL = "all-MiniLM-L6-v2"  # local, no API key

# ---------------------------------------------------------------------------
# 3. CONFIGURE RETRIEVAL EXPERIMENTS
# ---------------------------------------------------------------------------
HYBRID_WEIGHTS = [0.7, 0.3]  # [vector_weight, bm25_weight]
RRF_K = 10
K = 5  # top-k retrieval


class OverlapEvaluator:
    """Scores keyword overlap between answer and ground truth."""

    def evaluate(self, question: str, answer: str, ground_truth: str, context: list[Chunk]) -> list[EvaluationScore]:
        import re
        gt_tokens = {t for t in re.findall(r"\w+", ground_truth.lower()) if len(t) > 3}
        ans_tokens = set(re.findall(r"\w+", answer.lower()))
        if not gt_tokens:
            return [EvaluationScore(metric="answer_overlap", value=0.0)]
        return [EvaluationScore(metric="answer_overlap", value=round(len(gt_tokens & ans_tokens) / len(gt_tokens), 4))]


class ContextReturnGenerator:
    """Returns concatenated context as the 'answer' — measures retrieval quality directly."""

    def generate(self, question: str, context: list[Chunk]) -> str:
        return "\n".join(c.text for c in context)


def compute_embedding_viz(chunks, embeddings, eval_data, query_embeddings, all_results):
    """Compute PCA projections and similarity matrix for report visualizations."""
    all_vecs = embeddings + query_embeddings
    n, dim = len(all_vecs), len(all_vecs[0])

    mean_vec = [sum(all_vecs[j][d] for j in range(n)) / n for d in range(dim)]
    centered = [[all_vecs[j][d] - mean_vec[d] for d in range(dim)] for j in range(n)]

    def dot(a, b): return sum(x * y for x, y in zip(a, b))
    def normalize(v):
        norm = math.sqrt(dot(v, v)) or 1.0
        return [x / norm for x in v]

    def power_iter(data, exclude=None, iters=50):
        v = normalize([1.0 + 0.01 * i for i in range(dim)])
        for _ in range(iters):
            proj = [dot(row, v) for row in data]
            new_v = [sum(proj[j] * data[j][d] for j in range(len(data))) for d in range(dim)]
            if exclude:
                c = dot(new_v, exclude)
                new_v = [new_v[d] - c * exclude[d] for d in range(dim)]
            v = normalize(new_v)
        return v

    pc1 = power_iter(centered)
    pc2 = power_iter(centered, exclude=pc1)
    projections = [[dot(centered[j], pc1), dot(centered[j], pc2)] for j in range(n)]
    chunk_proj = projections[:len(chunks)]
    query_proj = projections[len(chunks):]

    def cosine_sim(a, b):
        d = dot(a, b)
        return d / ((math.sqrt(dot(a, a)) or 1.0) * (math.sqrt(dot(b, b)) or 1.0))

    sim_matrix = [[round(cosine_sim(qe, ce), 4) for ce in embeddings] for qe in query_embeddings]

    retrieval_edges = {}
    for result in all_results:
        edges = []
        for qr in result.query_results:
            edges.append([rc.index for rc in qr.retrieval.chunks][:3])
        retrieval_edges[result.name] = edges

    return {
        "chunk_points": [{"x": p[0], "y": p[1], "index": i, "preview": chunks[i].text[:80]} for i, p in enumerate(chunk_proj)],
        "query_points": [{"x": p[0], "y": p[1], "index": i, "question": eval_data[i]["question"], "type": eval_data[i].get("type", "unknown")} for i, p in enumerate(query_proj)],
        "similarity_matrix": sim_matrix,
        "query_labels": [q["question"][:50] for q in eval_data],
        "chunk_labels": [f"C{i}" for i in range(len(chunks))],
        "retrieval_edges": retrieval_edges,
    }


def main():
    # Load data
    eval_data = json.loads(EVAL_PATH.read_text())
    kw_count = sum(1 for q in eval_data if q.get("type") == "keyword")
    sem_count = sum(1 for q in eval_data if q.get("type") == "semantic")

    # Extract and chunk
    extractor = PlainTextExtractor()
    chunker = TokenChunker(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    doc = extractor.extract(KB_PATH)
    chunks = chunker.chunk(doc)
    print(f"Chunks: {len(chunks)} | Queries: {len(eval_data)} ({kw_count} kw, {sem_count} sem)")

    # Embed
    embedder = SentenceTransformerEmbedder(model_name=EMBEDDER_MODEL)
    embeddings = embedder.embed([c.text for c in chunks])

    # Build stores
    store = ChromaDBStore(collection_name="experiment")
    store.reset()
    store.add(chunks, embeddings)

    bm25 = BM25Retriever()
    bm25.add(chunks)
    vector_retriever = VectorRetriever(embedder=embedder, store=store)
    hybrid_retriever = HybridRetriever(
        retrievers=[vector_retriever, bm25],
        weights=HYBRID_WEIGHTS,
        rrf_k=RRF_K,
    )

    # ---------------------------------------------------------------------------
    # 4. DEFINE EXPERIMENTS — add or remove entries here
    # ---------------------------------------------------------------------------
    configs = [
        ("BM25 Keyword", bm25),
        ("Vector Semantic", vector_retriever),
        ("Hybrid (BM25+Vector)", hybrid_retriever),
    ]

    generator = ContextReturnGenerator()
    evaluator = OverlapEvaluator()
    ranker = RankingEvaluator(k_values=[1, 3, 5])

    all_results: list[ExperimentResult] = []
    for exp_name, retriever in configs:
        print(f"Running: {exp_name}")
        result = ExperimentResult(name=exp_name)
        retrieved_ids_list, relevance_sets = [], []

        for qa in eval_data:
            retrieval = retriever.retrieve(qa["question"], k=K)
            answer = generator.generate(qa["question"], retrieval.chunks)
            scores = evaluator.evaluate(qa["question"], answer, qa["ground_truth"], retrieval.chunks)
            result.query_results.append(QueryResult(
                query=qa["question"], ground_truth=qa["ground_truth"],
                answer=answer[:500], retrieval=retrieval, scores=scores,
            ))
            retrieved_ids_list.append([f"{c.source}_{c.index}" for c in retrieval.chunks])
            relevant_ids = set()
            for chunk in chunks:
                cid = f"{chunk.source}_{chunk.index}"
                for term in qa.get("relevant_chunks", []):
                    if term.lower() in chunk.text.lower():
                        relevant_ids.add(cid)
                        break
            relevance_sets.append(relevant_ids)

        result.ranking_results = ranker.rank(
            [qa["question"] for qa in eval_data], retrieved_ids_list, relevance_sets,
        )
        all_results.append(result)

    # Embedding visualizations
    query_embeddings = embedder.embed([q["question"] for q in eval_data])
    embedding_viz = compute_embedding_viz(chunks, embeddings, eval_data, query_embeddings, all_results)

    # Report context
    report_context = {
        "documents": [{"name": Path(KB_PATH).name, "path": KB_PATH, "chunks": len(chunks),
                        "chars": len(doc.text), "summary": "YOUR DOCUMENT SUMMARY HERE"}],
        "dataset": {"total": len(eval_data), "keyword_count": kw_count, "semantic_count": sem_count,
                     "path": str(EVAL_PATH), "summary": "YOUR DATASET SUMMARY HERE"},
        "questions": [{"question": q["question"], "ground_truth": q["ground_truth"],
                        "type": q.get("type", "")} for q in eval_data],
        "embedding_viz": embedding_viz,
    }

    # Generate reports
    report_dir = str(HERE / "reports")
    json_path = Reporter.to_json(all_results, output_dir=report_dir, context=report_context)
    html_path = Reporter.to_html(all_results, output_dir=report_dir, context=report_context)

    # Print summary
    print(f"\n{'Experiment':35s} {'F1@5':>8s} {'MRR':>8s} {'MAP':>8s}")
    print("-" * 55)
    for r in all_results:
        f1 = next((rr.value for rr in r.ranking_results if rr.metric == "f1@5"), 0)
        mrr = next((rr.value for rr in r.ranking_results if rr.metric == "mrr"), 0)
        mapv = next((rr.value for rr in r.ranking_results if rr.metric == "map"), 0)
        print(f"  {r.name:33s} {f1:7.1%} {mrr:7.1%} {mapv:7.1%}")

    print(f"\nJSON: {json_path}\nHTML: {html_path}")


if __name__ == "__main__":
    main()
