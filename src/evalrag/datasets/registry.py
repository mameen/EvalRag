"""Dataset registry and download manager.

Datasets are downloaded to .data/ (gitignored). Each dataset entry defines
a name, source URL, description, and a loader that returns Q&A pairs.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from evalrag.core.experiment import QAPair

logger = logging.getLogger(__name__)

DATA_DIR = Path(".data")


@dataclass
class DatasetEntry:
    name: str
    description: str
    url: str
    loader: Callable[[Path], list[QAPair]]


def _load_json_qa(path: Path) -> list[QAPair]:
    """Load a simple JSON Q&A file: [{"question": "...", "ground_truth": "...", ...}, ...]"""
    data = json.loads(path.read_text())
    return [
        QAPair(
            question=item["question"],
            ground_truth=item["ground_truth"],
            relevant_chunk_ids=item.get("relevant_chunk_ids", []),
        )
        for item in data
    ]


def _load_hotpotqa(path: Path) -> list[QAPair]:
    """Load HotpotQA dataset (JSONL format)."""
    pairs = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        pairs.append(
            QAPair(
                question=item.get("question", ""),
                ground_truth=item.get("answer", ""),
            )
        )
    return pairs


REGISTRY: dict[str, DatasetEntry] = {
    "hotpotqa-dev": DatasetEntry(
        name="hotpotqa-dev",
        description="HotpotQA distractor dev set (7,405 multi-hop questions)",
        url="http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
        loader=_load_hotpotqa,
    ),
    "evalrag-sample": DatasetEntry(
        name="evalrag-sample",
        description="Built-in sample Q&A pairs for quick testing (ships with the repo)",
        url="",
        loader=_load_json_qa,
    ),
}


def list_datasets() -> list[DatasetEntry]:
    return list(REGISTRY.values())


def download(name: str, force: bool = False) -> Path:
    if name not in REGISTRY:
        available = ", ".join(REGISTRY.keys())
        raise ValueError(f"Unknown dataset '{name}'. Available: {available}")

    entry = REGISTRY[name]
    if not entry.url:
        sample_path = Path(__file__).parent / "sample.json"
        if sample_path.exists():
            return sample_path
        raise FileNotFoundError(f"Dataset '{name}' has no URL and no local file found.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    filename = entry.url.split("/")[-1]
    dest = DATA_DIR / name / filename

    if dest.exists() and not force:
        logger.info("Already downloaded: %s", dest)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s -> %s", entry.url, dest)
    urllib.request.urlretrieve(entry.url, dest)
    logger.info("Done: %s (%.1f MB)", dest, dest.stat().st_size / 1e6)
    return dest


def load(name: str) -> list[QAPair]:
    path = download(name)
    entry = REGISTRY[name]
    return entry.loader(path)


def register(name: str, description: str, url: str, loader: Callable[[Path], list[QAPair]]) -> None:
    REGISTRY[name] = DatasetEntry(name=name, description=description, url=url, loader=loader)
