# EvalRAG — LLM Agent Guide

How to use EvalRAG for evaluating RAG pipelines. Aimed at LLM coding agents (Claude Code, Copilot, Cursor, etc.) assisting developers.

## What EvalRAG Does

EvalRAG is a composable RAG evaluation library. You plug together components (extractors, chunkers, embedders, retrievers, generators, evaluators) and compare retrieval strategies side-by-side with interactive reports.

**Key insight it proves**: Hybrid search (BM25 + Vector with Reciprocal Rank Fusion) outperforms either keyword or semantic search alone.

## Quick Start

```bash
PYTHONPATH=src python examples/hello_world.py
```

This runs a complete experiment comparing BM25 vs Vector vs Hybrid retrieval on a 10-chapter AI knowledge base with 20 evaluation queries (10 keyword, 10 semantic). Opens an interactive HTML report.

## Project Structure

```
src/evalrag/
  core/types.py          # Chunk, Document, RetrievalResult, ExperimentResult, etc.
  core/experiment.py     # Experiment runner
  extractors/            # PlainTextExtractor
  chunkers/              # TokenChunker
  embedders/             # SentenceTransformerEmbedder (local), OllamaEmbedder, OpenAIEmbedder
  stores/                # ChromaDBStore
  retrievers/            # BM25Retriever, VectorRetriever, HybridRetriever
  generators/            # OllamaGenerator, OpenAIGenerator
  evaluators/            # RagasEvaluator
  ranking/metrics.py     # RankingEvaluator (F1, MRR, MAP, Precision, Recall at k)
  exploration/
    reporter.py          # HTML + JSON report generation (D3 charts, embedding viz)
    inspector.py         # Chunk stats, score summary, compare utilities
  registry.py            # Plugin registry, config-driven experiment builder
  cli.py                 # Typer CLI (run, compare, datasets, download)

examples/
  hello_world.py                    # Complete working demo
  experiment_template.py            # Copy-and-customize template
  data/knowledge_base.txt           # 10-chapter AI knowledge base (40 chunks)
  data/eval_dataset.json            # 20 queries with ground truth
  reports/
    demo_hybrid_vs_bm25_vs_vector.html  # Sample report output
    demo_hybrid_vs_bm25_vs_vector.json  # Sample JSON output
```

## Creating a New Experiment

1. Copy `examples/experiment_template.py`
2. Replace data paths (`KB_PATH`, `EVAL_PATH`) with your own
3. Adjust configs: chunk size, embedder, retriever combos, hybrid weights
4. Run with `PYTHONPATH=src python examples/your_experiment.py`

## Report Features

Reports are self-contained HTML files with:

- **Executive summary** — green banner with the key finding
- **F1 scorecards** — one per experiment, winner badge on the best
- **Evaluation context** — source documents, dataset summary, Q&A table
- **F1 bar chart** — horizontal bars comparing experiments
- **All metrics chart** — grouped bar chart (F1, Precision, Recall, MRR, MAP)
- **Ranking table** — full metric breakdown
- **Per-query breakdown** — expandable ground truth and score details
- **PCA embedding scatter** — 2D projection of chunks and queries with retrieval edges, experiment tabs
- **Similarity heatmap** — query × chunk cosine similarity matrix

All charts are D3.js, interactive with hover tooltips. Light/dark mode toggle. Reports use timestamped filenames (`YYYYMMDDHHmmSS.html` + `.json`).

## Eval Dataset Format

```json
[
  {
    "question": "Who coined the term AI?",
    "ground_truth": "John McCarthy in 1956.",
    "type": "keyword",
    "relevant_chunks": ["John McCarthy", "1956"]
  },
  {
    "question": "How do computers understand word meaning?",
    "ground_truth": "Embeddings map text to dense vectors capturing semantics.",
    "type": "semantic",
    "relevant_chunks": ["embedding", "vector", "semantic"]
  }
]
```

- `type`: `"keyword"` (exact matches) or `"semantic"` (paraphrased, different vocabulary)
- `relevant_chunks`: terms that identify which chunks are relevant (for ranking metrics)

## Key Metrics

| Metric | What It Measures |
|--------|-----------------|
| **F1@k** | Harmonic mean of Precision and Recall at k — the executive-level metric |
| Precision@k | Fraction of retrieved chunks that are relevant |
| Recall@k | Fraction of relevant chunks that are retrieved |
| MRR | Mean Reciprocal Rank — how high is the first relevant result |
| MAP | Mean Average Precision — overall ranking quality |

## Component Composition

```python
# Local-first (no API keys)
embedder = SentenceTransformerEmbedder()       # all-MiniLM-L6-v2, 384 dims
store = ChromaDBStore(collection_name="my_exp")

# Ollama (local LLM)
from evalrag.generators.ollama import OllamaGenerator
generator = OllamaGenerator(model="llama3")

# Hybrid retrieval
hybrid = HybridRetriever(
    retrievers=[VectorRetriever(embedder, store), BM25Retriever()],
    weights=[0.7, 0.3],  # vector-heavy
    rrf_k=10,            # RRF smoothing (lower = more aggressive fusion)
)
```

## Embedding Visualizations

The report includes two scientific visualizations when `embedding_viz` data is provided in the report context:

1. **PCA Scatter Plot** — Projects all chunk and query embeddings to 2D via power iteration PCA. Shows retrieval edges connecting each query to its top-3 retrieved chunks. Experiment tabs let you compare retrieval patterns.

2. **Cosine Similarity Heatmap** — Full query × chunk similarity matrix. Reveals which chunks are semantically close to which queries, even if not retrieved.

See `compute_embedding_viz()` in `experiment_template.py` for how to generate this data.

## Versioning

Format: `major.minor.build(YYYYMMDDHHmmSS)`. Build auto-increments. Minor and above require maintainer approval.

## Testing

All tests use real implementations with test data — no mocking.

```bash
PYTHONPATH=src python -m pytest tests/ -v
```
