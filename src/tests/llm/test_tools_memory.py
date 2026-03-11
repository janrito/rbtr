"""Tests for the ``remember`` tool."""

from __future__ import annotations

import pytest
from pydantic_ai.tools import ToolDefinition

from rbtr.config import config
from rbtr.llm.tools.memory import _require_memory, remember
from rbtr.sessions.kinds import GLOBAL_SCOPE
from rbtr.sessions.store import SessionStore
from rbtr.state import EngineState

from .conftest import FakeCtx

SESSION = "test-session"


@pytest.fixture
def store() -> SessionStore:
    s = SessionStore(":memory:")
    s.set_context(session_id=SESSION)
    return s


@pytest.fixture
def ctx(store: SessionStore) -> FakeCtx:
    """FakeCtx with a repo connected (owner/repo)."""
    state = EngineState(owner="acme", repo_name="widgets")
    state.session_id = SESSION
    return FakeCtx(state, store)


@pytest.fixture
def ctx_no_repo(store: SessionStore) -> FakeCtx:
    """FakeCtx without a repository."""
    state = EngineState()
    state.session_id = SESSION
    return FakeCtx(state, store)


# ── prepare: visibility ──────────────────────────────────────────────


@pytest.mark.anyio
async def test_hidden_when_disabled(ctx: FakeCtx) -> None:
    """Tool is hidden when memory is disabled."""
    original = config.memory.enabled
    try:
        config.memory.enabled = False
        tool_def = ToolDefinition(name="remember")
        result = await _require_memory(ctx, tool_def)  # type: ignore[arg-type]
        assert result is None
    finally:
        config.memory.enabled = original


@pytest.mark.anyio
async def test_visible_when_enabled(ctx: FakeCtx) -> None:
    """Tool is visible when memory is enabled."""
    original = config.memory.enabled
    try:
        config.memory.enabled = True
        tool_def = ToolDefinition(name="remember")
        result = await _require_memory(ctx, tool_def)  # type: ignore[arg-type]
        assert result is tool_def
    finally:
        config.memory.enabled = original


# ── insert: global scope ─────────────────────────────────────────────


def test_insert_global(ctx: FakeCtx, store: SessionStore) -> None:
    """A global fact is inserted with scope `'global'`."""
    result = remember(ctx, "Always use type hints.")  # type: ignore[arg-type]
    assert "Saved" in result
    assert "global" in result

    facts = store.load_active_facts(GLOBAL_SCOPE)
    assert len(facts) == 1
    assert facts[0].content == "Always use type hints."
    assert facts[0].scope == GLOBAL_SCOPE


def test_insert_global_explicit(ctx: FakeCtx, store: SessionStore) -> None:
    """Passing scope='global' explicitly works."""
    result = remember(ctx, "Use ruff.", scope="global")  # type: ignore[arg-type]
    assert "Saved" in result

    facts = store.load_active_facts(GLOBAL_SCOPE)
    assert len(facts) == 1


# ── insert: repo scope ──────────────────────────────────────────────


def test_insert_repo(ctx: FakeCtx, store: SessionStore) -> None:
    """A repo-scoped fact resolves to `owner/repo`."""
    result = remember(ctx, "Uses Django 5.1.", scope="repo")  # type: ignore[arg-type]
    assert "Saved" in result
    assert "acme/widgets" in result

    facts = store.load_active_facts("acme/widgets")
    assert len(facts) == 1
    assert facts[0].content == "Uses Django 5.1."
    assert facts[0].scope == "acme/widgets"


def test_repo_scope_without_repo(ctx_no_repo: FakeCtx) -> None:
    """Repo scope without a connected repo returns an error."""
    result = remember(ctx_no_repo, "Some fact.", scope="repo")  # type: ignore[arg-type]
    assert "no repository" in result.lower()


def test_invalid_scope(ctx: FakeCtx) -> None:
    """An unrecognised scope returns an error."""
    result = remember(ctx, "Some fact.", scope="team")  # type: ignore[arg-type]
    assert "Invalid scope" in result


# ── supersede by content ─────────────────────────────────────────────


def test_supersede_by_content(ctx: FakeCtx, store: SessionStore) -> None:
    """Superseding by exact content marks old fact and inserts new."""
    store.insert_fact(GLOBAL_SCOPE, "Python 3.12+", SESSION)

    result = remember(  # type: ignore[arg-type]
        ctx, "Python 3.13+", scope="global", supersedes="Python 3.12+"
    )
    assert "Superseded" in result

    active = store.load_active_facts(GLOBAL_SCOPE)
    assert len(active) == 1
    assert active[0].content == "Python 3.13+"


def test_supersede_content_not_found(ctx: FakeCtx) -> None:
    """Superseding with non-matching content returns an error."""
    result = remember(  # type: ignore[arg-type]
        ctx, "New fact.", scope="global", supersedes="This does not exist."
    )
    assert "No active fact" in result


def test_supersede_content_wrong_scope(ctx: FakeCtx, store: SessionStore) -> None:
    """Superseding content from a different scope returns an error."""
    store.insert_fact("acme/widgets", "Repo-specific fact.", SESSION)

    result = remember(  # type: ignore[arg-type]
        ctx, "New global fact.", scope="global", supersedes="Repo-specific fact."
    )
    assert "No active fact" in result
    assert "global" in result


def test_supersede_strips_whitespace(ctx: FakeCtx, store: SessionStore) -> None:
    """Leading/trailing whitespace in supersedes is stripped."""
    store.insert_fact(GLOBAL_SCOPE, "Use ruff.", SESSION)

    result = remember(  # type: ignore[arg-type]
        ctx, "Use ruff with --fix.", scope="global", supersedes="  Use ruff.  "
    )
    assert "Superseded" in result
