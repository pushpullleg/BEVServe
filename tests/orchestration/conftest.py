"""Fixtures for the fake-Docker orchestration tests."""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest
from harness import ROOT, kill_fake, make_env


@pytest.fixture
def env(tmp_path: Path) -> Iterator[dict[str, str]]:
    environment = make_env(tmp_path)
    yield environment
    kill_fake(environment)


@pytest.fixture(autouse=True)
def _remove_outputs() -> Iterator[None]:
    """Remove every output directory a test run creates."""
    patterns = ("m5-2*", "m5-fault-2*")
    before = {p for pat in patterns for p in (ROOT / "outputs").glob(pat)}
    yield
    for d in {p for pat in patterns for p in (ROOT / "outputs").glob(pat)} - before:
        shutil.rmtree(d)
