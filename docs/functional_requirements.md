# EvalRAG - Functional Requirements

## Purpose

A composable RAG evaluation library that tests different retrieval, generation, and scoring approaches in an experiment-based fashion. Users define experiments as combinations of components, run them against Q&A datasets, and compare results.

## Core Concepts

| Concept | Definition |
|---------|------------|
| **Document** | Source file (PDF, DOCX, TXT, image) to build a knowledge base from |
| **Chunker** | Splits extracted text into retrievable units |
| **Embedder** | Converts chunks to vectors |
| **Store** | Persists and queries vectors (ChromaDB, Pinecone, Weaviate, etc.) |
| **Retriever** | Finds relevant chunks for a query (vector, keyword, hybrid) |
| **Generator** | LLM that produces answers given context + question |
| **Evaluator** | Scores answers against ground truth (RAGAS, custom metrics) |
| **Experiment** | A named combination of the above components + a dataset |

## Functional Requirements

### FR1 - Document Ingestion

- FR1.1: Extract text from PDF, DOCX, TXT
- FR1.2: Extract text from images via OCR (PNG, JPG)
- FR1.3: Pluggable extractors (unstructured, textract, custom)

### FR2 - Chunking

- FR2.1: Split text by token count with configurable size and overlap
- FR2.2: Pluggable chunking strategies (token, sentence, semantic)

### FR3 - Embedding and Storage

- FR3.1: Embed chunks and persist to a vector store
- FR3.2: Support multiple vector stores (ChromaDB, Pinecone, Weaviate)
- FR3.3: Configurable embedding models

### FR4 - Retrieval

- FR4.1: Vector similarity search
- FR4.2: Keyword search (BM25)
- FR4.3: Hybrid search (union or reciprocal rank fusion)
- FR4.4: Configurable top-k results

### FR5 - Generation

- FR5.1: Query LLM with retrieved context + question
- FR5.2: Support multiple LLM providers (OpenAI, Anthropic, local)
- FR5.3: Configurable system prompt and temperature

### FR6 - Evaluation

- FR6.1: Score answers using RAGAS metrics (relevancy, faithfulness, context precision)
- FR6.2: Pluggable evaluation frameworks
- FR6.3: Custom metric functions

### FR7 - Experiments

- FR7.1: Define an experiment as a composition of components + dataset
- FR7.2: Run multiple experiments in batch
- FR7.3: Compare results across experiments (same dataset, different pipelines)
- FR7.4: Persist experiment results to JSON/CSV
- FR7.5: Reproducible via config (YAML or dict)

### FR8 - Ranking Evaluation

- FR8.1: Evaluate retrieval quality with ranking metrics (MRR, NDCG, MAP, Precision@k, Recall@k)
- FR8.2: Compare ranking quality across retriever configurations (vector vs keyword vs hybrid, different k values, different embedding models)
- FR8.3: Per-query ranking breakdown (which queries retrieve well, which fail)
- FR8.4: Pluggable ranking metrics (custom scoring functions)

### FR9 - Exploration

- FR9.1: Inspect retrieved chunks for a given query (show what the retriever actually returns)
- FR9.2: Side-by-side comparison of retrieval results across experiments
- FR9.3: Visualize score distributions (retrieval scores, evaluation metrics) across experiments
- FR9.4: Drill into failure cases (low-scoring queries, irrelevant retrievals, unfaithful answers)
- FR9.5: Export exploration artifacts (HTML report, plots, dataframes)

### FR10 - CLI

- FR10.1: `evalrag embed <file>` - ingest a document
- FR10.2: `evalrag run <config>` - run an experiment from config
- FR10.3: `evalrag compare <result1> <result2>` - compare experiment results
- FR10.4: `evalrag explore <result>` - inspect retrieval and ranking details

### FR11 - Library API

- FR11.1: Usable as `import evalrag` with fluent or builder API
- FR11.2: Each component independently instantiable and testable
- FR11.3: No global state; all config explicit per experiment

### FR12 - Reports & UI/UX

- FR12.1: Reports are timestamped with `YYYYMMDDHHmmSS` filenames (both `.html` and `.json`)
- FR12.2: HTML reports are interactive, self-contained, single-file (inline D3.js charts)
- FR12.3: Reports support light and dark mode with a toggle, respecting `prefers-color-scheme`
- FR12.4: Bar charts comparing mean scores across experiments with hover tooltips
- FR12.5: Radar chart per experiment for multi-metric overview
- FR12.6: Per-query breakdown table with inline score bars
- FR12.7: Ranking metrics table when ranking evaluation is present
- FR12.8: Charts are responsive, animated, and render pleasantly on any screen size
- FR12.9: JSON report contains the same data for programmatic consumption

### FR13 - Local-First LLM Support

- FR13.1: Support Ollama as embedder and generator via OpenAI-compatible API
- FR13.2: No API keys required for local Ollama usage
- FR13.3: Cloud providers (OpenAI, Anthropic) available as opt-in alternatives

## Non-Functional Requirements

- NFR1: Python 3.10+
- NFR2: All external services (LLMs, vector DBs) behind abstractions
- NFR3: No hardcoded credentials; env vars or explicit config
- NFR4: Typed interfaces with Protocol or ABC
- NFR5: Minimal core dependencies; extras for specific providers
