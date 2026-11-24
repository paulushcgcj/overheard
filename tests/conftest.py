"""Shared test wiring: model cache under a stable dir, tiny ASR model."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO = Path(__file__).parent
FIXTURES = REPO / "fixtures"

# keep model downloads in one cache reused across runs / CI
os.environ.setdefault(
    "OVERHEARD_MODEL_DIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "overheard-test-models")
)
os.environ.setdefault(
    "OVERHEARD_DATA_DIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "overheard-test-jobs")
)


def pytest_configure(config):
    config.addinivalue_line("markers", "model: downloads real models and runs inference")


@pytest.fixture(scope="session")
def settings():
    from overheard.config import Settings

    return Settings()


@pytest.fixture(scope="session")
def jfk_bytes():
    return (FIXTURES / "fixture_jfk.wav").read_bytes()


@pytest.fixture(scope="session")
def conversation_bytes():
    return (FIXTURES / "conversation_2spk.wav").read_bytes()
