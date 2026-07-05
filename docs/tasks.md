# EvalRAG - Tasks to v0.1.x

Target: first usable release with core pipeline, one implementation per stage, tests, and docs.

---

2026-07-05 - Project init

- [x] **T01** Project scaffold: pyproject.toml, src layout, LICENSE (MIT), .gitignore
- [x] **T02** Core protocols: all Protocol definitions in `core/protocols.py`
- [x] **T03** Core types: Chunk, Document, ExperimentResult, RankingResult in `core/types.py`
- [x] **T04** Experiment runner: `core/experiment.py` - compose and run a pipeline
- [x] **T05** Extractor: unstructured text extraction (`extractors/unstructured.py`)
- [x] **T06** Chunker: token-based splitting (`chunkers/token.py`)
- [x] **T07** Embedder: sentence-transformer (`embedders/sentence_transformer.py`)
- [x] **T08** Store: ChromaDB (`stores/chromadb.py`)
- [x] **T09** Retriever: vector search (`retrievers/vector.py`)
- [x] **T10** Retriever: BM25 keyword search (`retrievers/keyword.py`)
- [x] **T11** Retriever: hybrid (`retrievers/hybrid.py`)
- [x] **T12** Generator: OpenAI (`generators/openai.py`)
- [x] **T13** Evaluator: RAGAS (`evaluators/ragas.py`)
- [x] **T14** Ranking metrics: MRR, NDCG, MAP, P@k, R@k (`ranking/metrics.py`)
- [x] **T15** Exploration: inspector + reporter (`exploration/inspector.py`, `exploration/reporter.py`)
- [x] **T16** CLI: embed, run, compare (`cli.py`)
- [x] **T17** Registry: optional plugin registry for config-driven usage
- [x] **T18** Test fixtures: small PDF, TXT, pre-built Q&A dataset
- [x] **T19** Integration tests: real ChromaDB, real extraction, LLM tests gated by API key
- [x] **T20** README: install, quickstart, architecture overview
- [ ] **T21** Version 0.1.0 - tag and release

---

2026-07-05 - Reports, demos, guides

- [x] **T22** D3 embedding scatter plot (PCA projection with retrieval edges, experiment tabs)
- [x] **T23** D3 cosine similarity heatmap (query × chunk matrix)
- [x] **T24** Demo report: `examples/reports/demo_hybrid_vs_bm25_vs_vector.html` + `.json`
- [x] **T25** Experiment template: `examples/experiment_template.py`
- [x] **T26** LLM agent guide: `docs/guide_llm_agents.md`
- [x] **T27** CLAUDE.md for agent onboarding

---
