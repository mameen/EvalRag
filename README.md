
![](https://raw.githubusercontent.com/mameen/EvalRag/main/docs/evalrag_banner.png)

# EvalRag

[![CI](https://github.com/mameen/EvalRag/actions/workflows/ci.yml/badge.svg)](https://github.com/mameen/EvalRag/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/evalragkit.svg)](https://pypi.org/project/evalragkit/)
[![Python versions](https://img.shields.io/pypi/pyversions/evalragkit.svg)](https://pypi.org/project/evalragkit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/mameen/EvalRag/blob/main/LICENSE)

Composable RAG evaluation library. Build experiments by plugging together extractors, chunkers, embedders, retrievers, generators, and evaluators — then compare results across configurations.

- **Composable pipeline** — swap extractors, chunkers, embedders, retrievers, generators, evaluators like LEGO
- **Hybrid search proving ground** — built-in demo shows BM25+Vector outperforms either alone (+21% F1 lift)
- **Interactive HTML reports** — F1 scorecards, D3 charts, light/dark mode, timestamped exports
- **Embedding space visualizations** — PCA scatter plot with retrieval edges + cosine similarity heatmap
- **Local or cloud models** — run fully local with SentenceTransformers + Ollama (no API keys), or plug in cloud providers like OpenAI
- **Zero-config hello world** — one command, full experiment, publication-ready report
- **Pure Python BM25** — no native dependencies, no numpy
- **20 metrics out of the box** — F1, Precision, Recall, MRR, MAP at configurable k values

![Embedding Space Projection (PCA) — queries and chunks projected to 2D with retrieval edges](https://raw.githubusercontent.com/mameen/EvalRag/main/docs/PCA.png)

![Query × Chunk Cosine Similarity Heatmap](https://raw.githubusercontent.com/mameen/EvalRag/main/docs/Heatmap.png)

## Install

```bash
pip install evalragkit                  # core only
pip install evalragkit[chromadb,openai] # with ChromaDB + OpenAI
pip install evalragkit[all]             # everything
```

## Quickstart

```python
from evalragkit.extractors.unstructured import PlainTextExtractor
from evalragkit.chunkers.token import TokenChunker
from evalragkit.embedders.openai import OpenAIEmbedder
from evalragkit.stores.chromadb import ChromaDBStore
from evalragkit.retrievers.vector import VectorRetriever
from evalragkit.generators.openai import OpenAIGenerator
from evalragkit.evaluators.ragas import RagasEvaluator
from evalragkit.core.experiment import Experiment, QAPair

# wire up the pipeline
extractor = PlainTextExtractor()
chunker = TokenChunker(chunk_size=500, chunk_overlap=50)
embedder = OpenAIEmbedder()
store = ChromaDBStore()
retriever = VectorRetriever(embedder=embedder, store=store)
generator = OpenAIGenerator()
evaluator = RagasEvaluator()

exp = Experiment(
    name="baseline",
    extractor=extractor,
    chunker=chunker,
    embedder=embedder,
    store=store,
    retriever=retriever,
    generator=generator,
    evaluator=evaluator,
)

# ingest documents
exp.ingest("docs/my_knowledge_base.txt")

# evaluate
dataset = [
    QAPair(question="What is RAG?", ground_truth="RAG combines retrieval with generation."),
]
result = exp.run(dataset)

print(result.mean_scores)
# {'faithfulness': 0.92, 'answer_relevancy': 0.88, ...}

Experiment.save_result(result, "results/baseline.json")
```

## Swap components

```python
from evalragkit.retrievers.keyword import BM25Retriever
from evalragkit.retrievers.hybrid import HybridRetriever

keyword = BM25Retriever()
keyword.add(chunks)  # chunks from ingest

hybrid = HybridRetriever(retrievers=[retriever, keyword], weights=[0.7, 0.3])
```

## Ranking evaluation

```python
from evalragkit.ranking.metrics import RankingEvaluator

ranker = RankingEvaluator(k_values=[1, 3, 5, 10])
results = ranker.rank(
    queries=["What is RAG?"],
    retrievals=[["doc1", "doc2", "doc3"]],
    relevance=[{"doc1", "doc3"}],
)
for r in results:
    print(f"{r.metric}: {r.value:.3f}")
```

## Hello World — Hybrid vs BM25 vs Vector

Run the built-in demo that proves hybrid search outperforms either approach alone:

```bash
PYTHONPATH=src python experiments/hello_world/run.py
```

Generates a timestamped interactive HTML report with F1 scorecards, D3 charts, PCA embedding scatter plot, and cosine similarity heatmap. See the [demo report](https://github.com/mameen/EvalRag/blob/main/experiments/hello_world/reports/demo_hybrid_vs_bm25_vs_vector.html) for sample output.

To create your own experiment, duplicate the `experiments/hello_world/` folder or start from [`experiments/template.py`](https://github.com/mameen/EvalRag/blob/main/experiments/template.py).

## Docs

- [LLM Agent Guide](https://github.com/mameen/EvalRag/blob/main/docs/guide_llm_agents.md) — comprehensive guide for AI coding agents
- [Functional Requirements](https://github.com/mameen/EvalRag/blob/main/docs/functional_requirements.md) — FR01–FR13
- [Architecture ADR](https://github.com/mameen/EvalRag/blob/main/docs/adrs/001-architecture-pattern.md)

## Compare experiments

```python
from evalragkit.exploration.reporter import Reporter

table = Reporter.to_table([result_a, result_b])
print(table)
```

## CLI

```bash
evalragkit run experiment.json --output results.json
evalragkit compare results_a.json results_b.json
evalragkit datasets
evalragkit download sample
```

## Architecture

Strategy + Composition pattern. Every pipeline stage is a Python Protocol — implement the interface and plug it in. No base classes, no registration required.

| Stage      | Protocol    | Built-in implementations                    |
|------------|-------------|---------------------------------------------|
| Extract    | `Extractor` | PlainTextExtractor, UnstructuredExtractor, OCRExtractor |
| Chunk      | `Chunker`   | TokenChunker                                |
| Embed      | `Embedder`  | SentenceTransformerEmbedder, OpenAIEmbedder, OllamaEmbedder |
| Store      | `Store`     | ChromaDBStore                               |
| Retrieve   | `Retriever` | VectorRetriever, BM25Retriever, HybridRetriever |
| Generate   | `Generator` | OpenAIGenerator, OllamaGenerator            |
| Evaluate   | `Evaluator` | RagasEvaluator                              |
| Rank       | `Ranker`    | RankingEvaluator                            |

## License

MIT
