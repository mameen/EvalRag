import os
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_text_path():
    return str(FIXTURES / "sample.txt")


@pytest.fixture
def qa_dataset_path():
    return str(FIXTURES / "qa_dataset.json")


requires_openai = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
