"""Shared test fixtures for rbtr."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolate_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    """Redirect all config paths to temp dirs."""
    home = tmp_path / "rbtr"
    home.mkdir()

    monkeypatch.setenv("RBTR_HOME", str(home))

    from rbtr.config import config

    config.reload()
    yield
    config.reload()


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    """Return the temp user config path (already isolated by `isolate_config`)."""
    return tmp_path / "rbtr" / "config.toml"
