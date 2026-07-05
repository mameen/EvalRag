"""Plugin registry for config-driven experiment construction."""

from __future__ import annotations

from typing import Any, Callable

from evalrag.core.experiment import Experiment, QAPair

_REGISTRY: dict[str, dict[str, Callable]] = {
    "extractor": {},
    "chunker": {},
    "embedder": {},
    "store": {},
    "retriever": {},
    "generator": {},
    "evaluator": {},
    "ranker": {},
}


def register(stage: str, name: str, factory: Callable) -> None:
    _REGISTRY[stage][name] = factory


def get(stage: str, name: str) -> Callable:
    if name not in _REGISTRY[stage]:
        raise KeyError(f"Unknown {stage}: {name}. Available: {list(_REGISTRY[stage])}")
    return _REGISTRY[stage][name]


def _register_builtins() -> None:
    from evalrag.chunkers.token import TokenChunker
    from evalrag.embedders.ollama import OllamaEmbedder
    from evalrag.embedders.openai import OpenAIEmbedder
    from evalrag.embedders.sentence_transformer import SentenceTransformerEmbedder
    from evalrag.evaluators.ragas import RagasEvaluator
    from evalrag.extractors.unstructured import PlainTextExtractor, UnstructuredExtractor
    from evalrag.generators.ollama import OllamaGenerator
    from evalrag.generators.openai import OpenAIGenerator
    from evalrag.ranking.metrics import RankingEvaluator
    from evalrag.retrievers.hybrid import HybridRetriever
    from evalrag.retrievers.keyword import BM25Retriever
    from evalrag.retrievers.vector import VectorRetriever
    from evalrag.stores.chromadb import ChromaDBStore

    register("extractor", "unstructured", UnstructuredExtractor)
    register("extractor", "plaintext", PlainTextExtractor)
    register("chunker", "token", TokenChunker)
    register("embedder", "sentence-transformer", SentenceTransformerEmbedder)
    register("embedder", "openai", OpenAIEmbedder)
    register("embedder", "ollama", OllamaEmbedder)
    register("store", "chromadb", ChromaDBStore)
    register("retriever", "vector", VectorRetriever)
    register("retriever", "keyword", BM25Retriever)
    register("retriever", "hybrid", HybridRetriever)
    register("generator", "openai", OpenAIGenerator)
    register("generator", "ollama", OllamaGenerator)
    register("evaluator", "ragas", RagasEvaluator)
    register("ranker", "ranking", RankingEvaluator)


_register_builtins()


def _build_component(stage: str, spec: dict[str, Any]) -> Any:
    name = spec.pop("type")
    return get(stage, name)(**spec)


def build_experiment(config: dict[str, Any]) -> tuple[Experiment, list[QAPair]]:
    """Build an Experiment and dataset from a config dict."""
    components = {}
    for stage in ["extractor", "chunker", "embedder", "store", "retriever", "generator", "evaluator"]:
        if stage in config:
            components[stage] = _build_component(stage, dict(config[stage]))

    ranker = None
    if "ranker" in config:
        ranker = _build_component("ranker", dict(config["ranker"]))

    dataset = []
    if "dataset" in config:
        ds_config = config["dataset"]
        if "file" in ds_config:
            import json
            from pathlib import Path
            items = json.loads(Path(ds_config["file"]).read_text())
        elif "name" in ds_config:
            from evalrag.datasets.registry import load
            items = load(ds_config["name"])
        else:
            items = ds_config.get("items", [])
        dataset = [QAPair(question=i["question"], ground_truth=i["ground_truth"]) for i in items]

    experiment = Experiment(
        name=config.get("name", "experiment"),
        ranker=ranker,
        **components,
    )
    return experiment, dataset
