"""Tests for tool prepare functions (tool hiding)."""

from __future__ import annotations

import tempfile

import pygit2

from rbtr.index.store import IndexStore
from rbtr.models import BranchTarget
from rbtr.state import EngineState

from .conftest import FakeCtx

# ── Prepare functions (tool hiding) ──────────────────────────────────


def testrequire_index_hides_when_no_index() -> None:
    """require_index returns None when no index is loaded."""
    import asyncio

    state = EngineState()
    state.review_target = BranchTarget(base_branch="main", head_branch="f", updated_at=0)
    assert state.index is None
    ctx = FakeCtx(state)

    from rbtr.llm.tools.common import require_index

    tool_def = object()  # stand-in
    result = asyncio.run(require_index(ctx, tool_def))  # type: ignore[arg-type]
    assert result is None


def testrequire_index_hides_when_no_target() -> None:
    """require_index returns None when no review target is set."""
    import asyncio

    state = EngineState()
    state.index = IndexStore()
    assert state.review_target is None
    ctx = FakeCtx(state)

    from rbtr.llm.tools.common import require_index

    tool_def = object()
    result = asyncio.run(require_index(ctx, tool_def))  # type: ignore[arg-type]
    assert result is None
    state.index.close()


def testrequire_index_returns_tool_when_ready(index_ctx: FakeCtx) -> None:
    """require_index returns the tool definition when both index and target exist."""
    import asyncio

    from rbtr.llm.tools.common import require_index

    tool_def = object()
    result = asyncio.run(require_index(index_ctx, tool_def))  # type: ignore[arg-type]
    assert result is tool_def


def testrequire_repo_hides_when_no_repo() -> None:
    """require_repo returns None when no repo is loaded."""
    import asyncio

    state = EngineState()
    state.review_target = BranchTarget(base_branch="main", head_branch="f", updated_at=0)
    ctx = FakeCtx(state)

    from rbtr.llm.tools.common import require_repo

    tool_def = object()
    result = asyncio.run(require_repo(ctx, tool_def))  # type: ignore[arg-type]
    assert result is None


def testrequire_repo_hides_when_no_target() -> None:
    """require_repo returns None when no review target is set."""
    import asyncio

    with tempfile.TemporaryDirectory() as tmp:
        repo = pygit2.init_repository(tmp)
        state = EngineState(repo=repo)
        ctx = FakeCtx(state)

        from rbtr.llm.tools.common import require_repo

        tool_def = object()
        result = asyncio.run(require_repo(ctx, tool_def))  # type: ignore[arg-type]
        assert result is None
