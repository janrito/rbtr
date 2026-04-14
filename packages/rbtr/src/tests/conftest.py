"""Shared test fixtures for rbtr."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    """Redirect all config paths to temp dirs."""
    user_dir = tmp_path / "rbtr"
    user_dir.mkdir()
    ws_dir = tmp_path / "ws"
    ws_dir.mkdir()
    mock_ws = lambda: ws_dir  # noqa: E731

    monkeypatch.setenv("RBTR_USER_DIR", str(user_dir))
    monkeypatch.setattr("rbtr.workspace.workspace_dir", mock_ws)

    from rbtr.config import config

    config.reload()
    yield
    config.reload()


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    """Return the temp user config path (already isolated by `_isolate_config`)."""
    return tmp_path / "rbtr" / "config.toml"
