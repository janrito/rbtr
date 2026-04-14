"""Tests for toolset filter and per-tool prepare functions."""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime

import pygit2
import pytest
from github import Github
from pydantic_ai.tools import ToolDefinition
from pytest_mock import MockerFixture

from rbtr_legacy.index.store import IndexStore
from rbtr_legacy.llm.tools.common import (
    has_index,
    has_pr_target,
    has_repo,
    matches_pathspec,
    require_pr,
)
from rbtr_legacy.models import BranchTarget, PRTarget, Target
from rbtr_legacy.sessions.store import SessionStore
from rbtr_legacy.state import EngineState

from .ctx import build_tool_ctx

# ── Shared test data ─────────────────────────────────────────────────

_BRANCH_TARGET = BranchTarget(
    base_branch="main",
    head_branch="feature",
    base_commit="main",
    head_commit="feature",
    updated_at=datetime.min.replace(tzinfo=UTC),
)

_PR_TARGET = PRTarget(
    number=42,
    title="Add feature",
    author="alice",
    base_branch="main",
    head_branch="feature",
    base_commit="abc",
    head_commit="def",
    updated_at=datetime.min.replace(tzinfo=UTC),
)

_TOOL_DEF = ToolDefinition(name="test_tool")


# ── has_index ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("has_idx", "has_target", "expected"),
    [
        (False, True, False),
        (True, False, False),
        (False, False, False),
        (True, True, True),
    ],
    ids=["no_index", "no_target", "neither", "both"],
)
def test_has_index(has_idx: bool, has_target: bool, expected: bool, store: SessionStore) -> None:
    """Filter returns True only when both index and review target exist."""
    state = EngineState()
    idx: IndexStore | None = None
    if has_idx:
        idx = IndexStore()
        state.index = idx
    if has_target:
        state.review_target = _BRANCH_TARGET

    result = has_index(build_tool_ctx(state, store), _TOOL_DEF)
    assert result is expected

    if idx is not None:
        idx.close()


# ── has_repo ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("has_rp", "has_target", "expected"),
    [
        (False, True, False),
        (True, False, False),
        (False, False, False),
        (True, True, True),
    ],
    ids=["no_repo", "no_target", "neither", "both"],
)
def test_has_repo(has_rp: bool, has_target: bool, expected: bool, store: SessionStore) -> None:
    """Filter returns True only when both repo and review target exist."""
    with tempfile.TemporaryDirectory() as tmp:
        state = EngineState()
        if has_rp:
            state.repo = pygit2.init_repository(tmp)
        if has_target:
            state.review_target = _BRANCH_TARGET

        result = has_repo(build_tool_ctx(state, store), _TOOL_DEF)
        assert result is expected


# ── require_pr (per-tool prepare — stricter than group filter) ───────


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
    target: Target | None,
    expected_visible: bool,
    mocker: MockerFixture,
    store: SessionStore,
) -> None:
    """Tool is visible only when both GitHub auth and a PR target exist."""
    state = EngineState()
    if has_gh:
        state.gh = mocker.create_autospec(Github, instance=True)
        state.gh_username = "reviewer"
    state.review_target = target

    result = await require_pr(build_tool_ctx(state, store), _TOOL_DEF)

    if expected_visible:
        assert result is _TOOL_DEF
    else:
        assert result is None


# ── has_pr_target ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        (None, False),
        (_BRANCH_TARGET, False),
        (_PR_TARGET, True),
    ],
    ids=["no_target", "branch_target", "pr_target"],
)
def test_has_pr_target(target: Target | None, expected: bool, store: SessionStore) -> None:
    """Filter returns True when a PR target is selected."""
    state = EngineState()
    state.review_target = target

    result = has_pr_target(build_tool_ctx(state, store), _TOOL_DEF)
    assert result is expected


# ── matches_pathspec ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("path", "pattern", "expected"),
    [
        # Empty pattern matches everything.
        ("src/api/handler.py", "", True),
        ("anything", "", True),
        # Plain prefix — directory scope.
        ("src/api/handler.py", "src/api", True),
        ("src/api", "src/api", True),
        ("src/api/nested/deep.py", "src/api", True),
        ("src/apix/handler.py", "src/api", False),
        ("other/file.py", "src/api", False),
        # Trailing slash stripped.
        ("src/api/handler.py", "src/api/", True),
        # Exact file as prefix (no metachar, no trailing slash).
        ("src/api/handler.py", "src/api/handler.py", True),
        ("src/api/other.py", "src/api/handler.py", False),
        # Glob — star.
        ("src/api/handler.py", "*.py", True),
        ("src/api/handler.js", "*.py", False),
        # Glob — globstar.
        ("src/api/handler.py", "src/**/*.py", True),
        ("src/api/nested/deep.py", "src/**/*.py", True),
        ("lib/util.py", "src/**/*.py", False),
        # Glob — single dir level.
        ("src/handler.py", "src/*.py", True),
        ("src/api/handler.py", "src/*.py", False),
        # Glob — question mark.
        ("src/foo.py", "src/???.py", True),
        ("src/fo.py", "src/???.py", False),
        # Glob — bracket.
        ("src/a.py", "src/[ab].py", True),
        ("src/b.py", "src/[ab].py", True),
        ("src/c.py", "src/[ab].py", False),
    ],
    ids=[
        "empty_matches_path",
        "empty_matches_any",
        "prefix_subdir",
        "prefix_exact",
        "prefix_nested",
        "prefix_no_partial",
        "prefix_different_dir",
        "prefix_trailing_slash",
        "exact_file_match",
        "exact_file_no_match",
        "star_py",
        "star_js_no_match",
        "globstar_direct",
        "globstar_nested",
        "globstar_wrong_root",
        "single_level_match",
        "single_level_no_nested",
        "question_match",
        "question_no_match",
        "bracket_a",
        "bracket_b",
        "bracket_no_match",
    ],
)
def test_matches_pathspec(path: str, pattern: str, expected: bool) -> None:
    assert matches_pathspec(path, pattern) is expected
