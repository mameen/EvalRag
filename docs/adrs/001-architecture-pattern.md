# ADR-001: Architecture Pattern Selection

**Status:** Proposed
**Date:** 2026-07-05
**Context:** Choosing the extensibility pattern for EvalRAG's composable pipeline

## Problem

EvalRAG must let users swap any pipeline component (chunker, embedder, store, retriever, generator, evaluator) independently. The original `evalrag_task` attempted a Visitor pattern that was never fully wired. We need a pattern that is:

1. Easy to extend (add a new vector DB without touching core)
2. Easy to compose (mix-and-match components per experiment)
3. Easy to understand (contributors grok it in minutes)

## Options Considered

### Option A: Visitor Pattern (original)

**How it works:** Components are "visitors" that operate on an "acceptor" (the pipeline). The acceptor dispatches to registered visitors by type.

```python
class ChromaDBVisitor(IVisitor):
    def visit(self, acceptor: IAcceptor):
        acceptor.collection = chromadb.Client().get_or_create_collection(...)

pipeline = EvalRag()
pipeline.accept(ChromaDBVisitor())
pipeline.accept(UnstructuredVisitor())
pipeline.embed(file)
```

| Pros | Cons |
|------|------|
| Separates operations from data structure | Visitor semantics are awkward for pipelines |
| Well-known GoF pattern | Double dispatch is confusing for new contributors |
| | Visitor "visits" an acceptor -- but here components don't inspect the pipeline, they ARE the pipeline |
| | Original impl proved this: visitors were defined but bypassed |

**Verdict:** Poor fit. Visitor is for adding operations to fixed data structures (e.g., AST traversal). RAG pipelines need swappable components, not swappable operations.

### Option B: Ports and Adapters (Hexagonal)

**How it works:** Core domain defines abstract "ports" (interfaces). Each external system implements an "adapter." The core only depends on ports.

```python
# Port
class VectorStore(Protocol):
    def add(self, chunks: list[Chunk]) -> None: ...
    def query(self, embedding: list[float], k: int) -> list[Chunk]: ...

# Adapter
class ChromaDBStore(VectorStore):
    def add(self, chunks): ...
    def query(self, embedding, k): ...

# Core composes via ports
class Pipeline:
    def __init__(self, store: VectorStore, generator: Generator, ...): ...
```

| Pros | Cons |
|------|------|
| Clean dependency inversion | More files and interfaces upfront |
| Each adapter is independently testable | Can feel over-engineered for small projects |
| Core never imports external libraries | |
| Natural fit for "swap ChromaDB for Pinecone" | |

**Verdict:** Good fit, but the full hexagonal ceremony (application services, driving/driven ports) is heavier than needed.

### Option C: Strategy Pattern with Composition (recommended)

**How it works:** Each pipeline stage is a Protocol (interface). Concrete implementations are strategies. An Experiment composes them. No framework, no dispatch -- just dependency injection via constructor.

```python
# Protocols (lightweight ports)
class Retriever(Protocol):
    def retrieve(self, query: str, k: int) -> list[Chunk]: ...

class Generator(Protocol):
    def generate(self, question: str, context: list[Chunk]) -> str: ...

# Concrete strategies
class ChromaDBRetriever:
    def retrieve(self, query, k): ...

class HybridRetriever:
    def __init__(self, vector: Retriever, keyword: Retriever): ...
    def retrieve(self, query, k): ...  # merge results

# Composition
class Experiment:
    def __init__(self, name, retriever, generator, evaluator, dataset): ...
    def run(self) -> ExperimentResult: ...
```

| Pros | Cons |
|------|------|
| Simplest mental model: interfaces + implementations | Less formal than hexagonal (no explicit port/adapter naming) |
| Python Protocol = structural typing, no inheritance needed | |
| Composition over configuration | |
| Each component is a standalone, testable unit | |
| `Experiment` is just a dataclass holding strategies | |
| Easy to add: implement the Protocol, pass it in | |

**Verdict:** Best fit. It's hexagonal in spirit (dependency inversion via protocols) without the ceremony. The `Experiment` as compositor makes the experiment-based workflow first-class.

### Option D: Plugin Registry

**How it works:** Components register themselves in a global registry by name. Experiments reference components by string key.

```python
@register("retriever", "chromadb")
class ChromaDBRetriever: ...

experiment = Experiment.from_config({"retriever": "chromadb", ...})
```

| Pros | Cons |
|------|------|
| Config-driven experiments | Global mutable state |
| Easy CLI integration | Magic strings lose type safety |
| | Import side effects for registration |
| | Harder to test in isolation |

**Verdict:** Nice for CLI/config layer on top of Option C, but not as the core pattern.

## Decision

**Option C: Strategy Pattern with Composition**, with Option D as an optional registry layer for CLI/YAML config.

### Key design rules

1. **Every pipeline stage is a Protocol** -- no ABC inheritance required
2. **Experiment is the unit of composition** -- holds references to all strategies
3. **No global state** -- everything explicit via constructors
4. **Extras pattern for dependencies** -- `pip install evalrag[chromadb]`, `evalrag[pinecone]`
5. **Registry is opt-in** -- used by CLI and YAML config, not by the library API
6. **Ranking is first-class** -- retrieval quality is evaluated independently from generation quality
7. **Exploration is built-in** -- every experiment result is inspectable and comparable

### Versioning

`major.minor.build` where build is a timestamp (`YYYYMMDDHHmmSS`).

- **build**: auto-incremented on each commit/release
- **minor**: new features, new protocols, new component types
- **major**: breaking changes to protocols or experiment config format

Minor and major bumps require maintainer approval.

## Package Structure

```
evalrag/
  __init__.py
  core/
    protocols.py      # All Protocol definitions (Retriever, Generator, Evaluator, Ranker, etc.)
    experiment.py     # Experiment runner and result types
    types.py          # Chunk, Document, ExperimentResult, RankingResult dataclasses
  extractors/
    unstructured.py
    ocr.py
  chunkers/
    token.py
    sentence.py
  embedders/
    sentence_transformer.py
    openai.py
  stores/
    chromadb.py
    pinecone.py
  retrievers/
    vector.py
    keyword.py
    hybrid.py
  generators/
    openai.py
    anthropic.py
  evaluators/
    ragas.py
    custom.py
  ranking/
    metrics.py        # MRR, NDCG, MAP, Precision@k, Recall@k
    comparator.py     # Cross-experiment ranking comparison
  exploration/
    inspector.py      # Per-query chunk inspection, failure drill-down
    reporter.py       # HTML/CSV/dataframe export
    visualization.py  # Score distributions, side-by-side plots
  cli.py
  registry.py         # Optional plugin registry for config-driven usage
```
