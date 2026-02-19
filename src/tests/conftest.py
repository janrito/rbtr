"""Shared test fixtures."""

from pathlib import Path

import pytest

from rbtr.config import config
from rbtr.creds import Creds, creds


@pytest.fixture
def creds_path(tmp_path: Path, monkeypatch):
    """Point credential storage at a temp file for test isolation."""
    path = tmp_path / "creds.toml"
    monkeypatch.setattr("rbtr.creds.CREDS_PATH", path)
    monkeypatch.setitem(Creds.model_config, "toml_file", str(path))
    creds.__init__()  # type: ignore[misc]  # reload in place via pydantic re-init
    return path


@pytest.fixture
def config_path(tmp_path: Path, monkeypatch):
    """Point config storage at a temp file for test isolation."""
    path = tmp_path / "config.toml"
    monkeypatch.setattr("rbtr.config.CONFIG_PATH", path)
    monkeypatch.setattr("rbtr.config.WORKSPACE_PATH", tmp_path / "ws" / "config.toml")
    config.__init__()  # type: ignore[misc]  # reload in place via pydantic re-init
    return path
