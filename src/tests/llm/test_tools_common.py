"""Tests for tool prepare functions (tool visibility guards)."""

from __future__ import annotations

import tempfile
from unittest.mock import MagicMock

import pygit2
import pytest

from rbtr.index.store import IndexStore
from rbtr.llm.tools.common import require_index, require_pr, require_pr_target, require_repo
from rbtr.models import BranchTarget, PRTarget
from rbtr.state import EngineState

from .conftest import FakeCtx

# ── Shared test data ─────────────────────────────────────────────────

_BRANCH_TARGET = BranchTarget(
    base_branch="main",
    head_branch="feature",
    base_commit="main",
    head_commit="feature",
    updated_at=0,
)

_PR_TARGET = PRTarget(
    number=42,
    title="Add feature",
    author="alice",
    base_branch="main",
    head_branch="feature",
    base_commit="abc",
    head_commit="def",
    updated_at=0,
)


# ── require_index ────────────────────────────────────────────────────


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("has_index", "has_target", "expected_visible"),
    [
        (False, True, False),
        (True, False, False),
        (False, False, False),
        (True, True, True),
    ],
    ids=["no_index", "no_target", "neither", "both"],
)
async def test_require_index(
    has_index: bool,
    has_target: bool,
    expected_visible: bool,
) -> None:
    """Tool is visible only when both index and review target exist."""
    state = EngineState()
    store: IndexStore | None = None
    if has_index:
        store = IndexStore()
        state.index = store
    if has_target:
        state.review_target = _BRANCH_TARGET

    tool_def = object()
    result = await require_index(FakeCtx(state), tool_def)  # type: ignore[arg-type]

    if expected_visible:
        assert result is tool_def
    else:
        assert result is None

    if store is not None:
        store.close()


# ── require_repo ─────────────────────────────────────────────────────


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("has_repo", "has_target", "expected_visible"),
    [
        (False, True, False),
        (True, False, False),
        (False, False, False),
        (True, True, True),
    ],
    ids=["no_repo", "no_target", "neither", "both"],
)
async def test_require_repo(
    has_repo: bool,
    has_target: bool,
    expected_visible: bool,
) -> None:
    """Tool is visible only when both repo and review target exist."""
    with tempfile.TemporaryDirectory() as tmp:
        state = EngineState()
        if has_repo:
            state.repo = pygit2.init_repository(tmp)
        if has_target:
            state.review_target = _BRANCH_TARGET

        tool_def = object()
        result = await require_repo(FakeCtx(state), tool_def)  # type: ignore[arg-type]

        if expected_visible:
            assert result is tool_def
        else:
            assert result is None


# ── require_pr ───────────────────────────────────────────────────────


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("has_gh", "target", "expected_visible"),
    [
        (False, _PR_TARGET, False),
        (True, _BRANCH_TARGET, False),
        (True, None, False),
        (True, _PR_TARGET, True),
    ],
    ids=["no_gh", "branch_target", "no_target", "pr_with_gh"],
)
async def test_require_pr(
    has_gh: bool,
    target: object,
    expected_visible: bool,
) -> None:
    """Tool is visible only when both GitHub auth and a PR target exist."""
    state = EngineState()
    if has_gh:
        state.gh = MagicMock()
        state.gh_username = "reviewer"
    state.review_target = target  # type: ignore[assignment]

    tool_def = object()
    result = await require_pr(FakeCtx(state), tool_def)  # type: ignore[arg-type]

    if expected_visible:
        assert result is tool_def
    else:
        assert result is None


# ── require_pr_target ────────────────────────────────────────────────


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("target", "expected_visible"),
    [
        (None, False),
        (_BRANCH_TARGET, False),
        (_PR_TARGET, True),
    ],
    ids=["no_target", "branch_target", "pr_target"],
)
async def test_require_pr_target(
    target: object,
    expected_visible: bool,
) -> None:
    """Tool is visible when a PR target is selected (no GitHub auth needed)."""
    state = EngineState()
    state.review_target = target  # type: ignore[assignment]

    tool_def = object()
    result = await require_pr_target(FakeCtx(state), tool_def)  # type: ignore[arg-type]

    if expected_visible:
        assert result is tool_def
    else:
        assert result is None
