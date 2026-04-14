"""Shared test fixtures."""

import queue
import socket
from collections.abc import Generator
from pathlib import Path
from typing import no_type_check

import pytest

from rbtr.config import SkillsConfig, config
from rbtr.creds import creds
from rbtr.engine.core import Engine
from rbtr.llm.agent import register_tools
from rbtr.llm.context import LLMContext
from rbtr.providers import endpoint as endpoint_mod
from rbtr.providers.types import Provider
from rbtr.sessions.store import SessionStore
from rbtr.state import EngineState
from rbtr.tui.input import InputReader, InputState
from rbtr.tui.ui import UI
from tests.helpers import HeadlessUI, StubProvider

# Register all tool submodules in the correct order before any test
# runs.  Tests that directly import a single tool module would
# otherwise register tools out of order, breaking ordering assertions
# in test_toolsets.
register_tools()


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
def input_state() -> InputState:
    """Fresh `InputState` for each test."""
    return InputState()


@pytest.fixture
def input_reader(input_state: InputState) -> InputReader:
    """Headless `InputReader` wired to `input_state`."""
    return InputReader(input_state)


@pytest.fixture
def headless_ui(input_state: InputState) -> UI:
    """UI with only `inp` set — for completion tests."""
    return HeadlessUI(input_state)


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

    # Register the test provider so `build_model("test/...")` works
    # without credentials or network access.  Hooks into the endpoint
    # resolution path — `_resolve` checks `BuiltinProvider` first,
    # then `endpoint.resolve`, which we patch to recognise "test".
    _stub_provider = StubProvider()
    _original_resolve = endpoint_mod.resolve

    def _resolve_with_test(name: str) -> Provider | None:
        if name == "test":
            return _stub_provider
        return _original_resolve(name)

    monkeypatch.setattr(endpoint_mod, "resolve", _resolve_with_test)

    yield
    _stub_provider.reset()
    config.reload()
    creds.reload()


@pytest.fixture
def stub_provider() -> StubProvider:
    """Access the test provider to configure model responses per-test.

    Example::

        def test_compact(stub_provider, engine, llm_ctx):
            stub_provider.set_model(TestModel(custom_output_text="Summary."))
            engine.state.model_name = "test/default"
            compact_history(llm_ctx)
    """
    prov = endpoint_mod.resolve("test")
    assert isinstance(prov, StubProvider)
    return prov


@pytest.fixture
def creds_path(tmp_path: Path) -> Path:
    """Return the temp creds path (already isolated by `_isolate_config`)."""
    return tmp_path / "rbtr" / "creds.toml"


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    """Return the temp user config path (already isolated by `_isolate_config`)."""
    return tmp_path / "rbtr" / "config.toml"


# ── Session store ────────────────────────────────────────────────────


@pytest.fixture
def store() -> Generator[SessionStore]:
    """Empty in-memory session store."""
    with SessionStore() as s:
        yield s


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
def llm_engine() -> Generator[Engine]:
    """Engine pre-wired for LLM tests (test provider connected, model set).

    Uses the test provider so no credentials or network access needed.
    Ready for `handle_llm` calls.
    """
    state = EngineState(owner="testowner", repo_name="testrepo")
    state.connected_providers.add("test")
    state.model_name = "test/default"
    with Engine(state, queue.Queue(), store=SessionStore()) as eng:
        eng._sync_store_context()
        yield eng
