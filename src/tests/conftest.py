"""Shared test fixtures and helpers."""

import queue
import socket
from collections.abc import Generator
from pathlib import Path
from typing import no_type_check

import pytest

from rbtr.config import config
from rbtr.creds import Creds, creds
from rbtr.engine import Engine
from rbtr.events import Event, MarkdownOutput, Output
from rbtr.llm.context import LLMContext
from rbtr.sessions.store import SessionStore
from rbtr.state import EngineState

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
def _isolate_session_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect SESSIONS_DB_PATH to a temp dir so no test touches the
    real user database at ``~/.config/rbtr/sessions.db``.

    Tests that create ``SessionStore()`` (no args) use ``:memory:``.
    This guard catches ``Engine(state, q)`` without an explicit
    ``store=`` kwarg — the fallback path will land in ``tmp_path``
    instead of the user's home directory.

    Both the canonical definition and every re-export must be patched
    because ``from X import Y`` creates a separate binding.
    """
    safe_path = tmp_path / "sessions.db"
    monkeypatch.setattr("rbtr.sessions.store.SESSIONS_DB_PATH", safe_path)
    monkeypatch.setattr("rbtr.engine.core.SESSIONS_DB_PATH", safe_path)


@pytest.fixture(autouse=True)
def _isolate_creds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    """Redirect credential storage to a temp file so no test leaks
    API keys into the singleton or touches the real creds file.

    Applied globally — every test starts with a clean ``creds``
    singleton and ends by restoring defaults.
    """
    path = tmp_path / "creds.toml"
    monkeypatch.setattr("rbtr.creds.CREDS_PATH", path)
    monkeypatch.setitem(Creds.model_config, "toml_file", str(path))
    creds.__init__()  # type: ignore[misc]  # reload in place via pydantic re-init
    yield
    creds.__init__()  # type: ignore[misc]  # restore defaults after monkeypatch unwinds


@pytest.fixture
def creds_path(tmp_path: Path) -> Path:
    """Return the temp creds path (already isolated by ``_isolate_creds``)."""
    return tmp_path / "creds.toml"


@pytest.fixture
def config_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[Path]:
    """Point config storage at a temp file for test isolation."""
    path = tmp_path / "config.toml"
    monkeypatch.setattr("rbtr.config.CONFIG_PATH", path)
    monkeypatch.setattr("rbtr.config.WORKSPACE_PATH", tmp_path / "ws" / "config.toml")
    config.__init__()  # type: ignore[misc]  # reload in place via pydantic re-init
    yield path
    config.__init__()  # type: ignore[misc]  # restore defaults after monkeypatch unwinds


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
    and syncs the store context — ready for ``handle_llm`` calls.
    """
    from rbtr.providers import BuiltinProvider

    creds.update(openai_api_key="sk-test")
    state = EngineState(owner="testowner", repo_name="testrepo")
    state.connected_providers.add(BuiltinProvider.OPENAI)
    state.model_name = "openai/gpt-4o"
    with Engine(state, queue.Queue(), store=SessionStore()) as eng:
        eng._sync_store_context()
        yield eng
