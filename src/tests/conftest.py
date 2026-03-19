"""Shared test fixtures and helpers."""

import queue
import socket
from collections.abc import Generator
from pathlib import Path
from typing import no_type_check

import pytest

from rbtr.config import SkillsConfig, config
from rbtr.creds import creds
from rbtr.engine import Engine
from rbtr.events import Event, MarkdownOutput, Output
from rbtr.llm.agent import register_tools
from rbtr.llm.context import LLMContext
from rbtr.sessions.store import SessionStore
from rbtr.state import EngineState

# Register all tool submodules in the correct order before any test
# runs.  Tests that directly import a single tool module would
# otherwise register tools out of order, breaking ordering assertions
# in test_toolsets.
register_tools()

# ── Event helpers ────────────────────────────────────────────────────


def drain(events: queue.Queue[Event]) -> list[Event]:
    """Drain all events from the queue into a list."""
    result: list[Event] = []
    while True:
        try:
            result.append(events.get_nowait())
        except queue.Empty:
            break
    return result


def output_texts(events: list[Event]) -> list[str]:
    """Extract text from Output and MarkdownOutput events."""
    texts: list[str] = []
    for e in events:
        if isinstance(e, (Output, MarkdownOutput)):
            texts.append(e.text)
    return texts


def has_event_type(events: list[Event], event_type: type) -> bool:
    """Check whether any event matches the given type."""
    return any(isinstance(e, event_type) for e in events)


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


@pytest.fixture(autouse=True)
def _isolate_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    """Redirect all config paths to temp dirs.

    Prevents tests from touching real user data (`~/.config/rbtr/`),
    the workspace (`.rbtr/`), or the developer's skill directories.

    `config` is a module-level singleton — every module holds a
    reference to the same object. `reload()` reloads it in place
    so all references stay valid.
    """
    user_dir = tmp_path / "rbtr"
    user_dir.mkdir()
    ws_dir = tmp_path / "ws"
    ws_dir.mkdir()
    mock_ws = lambda: ws_dir  # noqa: E731

    monkeypatch.setenv("RBTR_USER_DIR", str(user_dir))
    monkeypatch.setattr("rbtr.workspace.workspace_dir", mock_ws)

    config.reload()
    monkeypatch.setattr(config, "skills", SkillsConfig(project_dirs=[], user_dirs=[]))
    creds.reload()
    yield
    config.reload()
    creds.reload()


@pytest.fixture
def creds_path(tmp_path: Path) -> Path:
    """Return the temp creds path (already isolated by `_isolate_config`)."""
    return tmp_path / "rbtr" / "creds.toml"


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    """Return the temp user config path (already isolated by `_isolate_config`)."""
    return tmp_path / "rbtr" / "config.toml"


# ── Engine fixtures ──────────────────────────────────────────────────


@pytest.fixture
def engine() -> Generator[Engine]:
    """Default engine with auto-cleanup."""
    state = EngineState(owner="testowner", repo_name="testrepo")
    with Engine(state, queue.Queue(), store=SessionStore()) as eng:
        yield eng


@pytest.fixture
def llm_ctx(engine: Engine) -> LLMContext:
    """LLMContext backed by the default engine fixture."""
    return engine._llm_context()


@pytest.fixture
def llm_engine(creds_path: Path) -> Generator[Engine]:
    """Engine pre-wired for LLM tests (OpenAI connected, model set).

    Sets credentials, connects the OpenAI provider, assigns a model,
    and syncs the store context — ready for `handle_llm` calls.
    """
    from rbtr.providers import BuiltinProvider

    creds.update(openai_api_key="sk-test")
    state = EngineState(owner="testowner", repo_name="testrepo")
    state.connected_providers.add(BuiltinProvider.OPENAI)
    state.model_name = "openai/gpt-4o"
    with Engine(state, queue.Queue(), store=SessionStore()) as eng:
        eng._sync_store_context()
        yield eng
