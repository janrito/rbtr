"""Shared test fixtures."""

import socket
from collections.abc import Generator
from pathlib import Path
from typing import no_type_check

import pytest

from rbtr.config import config
from rbtr.creds import Creds, creds

# ── Network safety net ───────────────────────────────────────────────

_real_socket = socket.socket


@pytest.fixture(autouse=True)
def _block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any test that opens a real network connection.

    Applied globally so accidental HTTP calls from unmocked code paths
    surface as loud failures instead of silently hitting real services.
    Unix-domain and other non-inet sockets are allowed (e.g. DuckDB).
    """

    @no_type_check  # limited advantage for typing this monkeypatch
    def _guarded(*args, **kwargs):
        family = args[0] if args else kwargs.get("family", socket.AF_INET)
        if family in (socket.AF_INET, socket.AF_INET6):
            raise RuntimeError(
                "Test tried to open a real network connection — "
                "mock the HTTP call or add creds_path/config_path fixtures"
            )
        return _real_socket(*args, **kwargs)

    monkeypatch.setattr(socket, "socket", _guarded)


@pytest.fixture
def creds_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point credential storage at a temp file for test isolation."""
    path = tmp_path / "creds.toml"
    monkeypatch.setattr("rbtr.creds.CREDS_PATH", path)
    monkeypatch.setitem(Creds.model_config, "toml_file", str(path))
    creds.__init__()  # type: ignore[misc]  # reload in place via pydantic re-init
    return path


@pytest.fixture
def config_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[Path]:
    """Point config storage at a temp file for test isolation."""
    path = tmp_path / "config.toml"
    monkeypatch.setattr("rbtr.config.CONFIG_PATH", path)
    monkeypatch.setattr("rbtr.config.WORKSPACE_PATH", tmp_path / "ws" / "config.toml")
    config.__init__()  # type: ignore[misc]  # reload in place via pydantic re-init
    yield path
    config.__init__()  # type: ignore[misc]  # restore defaults after monkeypatch unwinds
