"""Tests for snapshot review mode — SnapshotTarget behaviour across tools and prompts."""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime

import pygit2
import pytest
from pydantic_ai.tools import ToolDefinition

from rbtr.llm.tools.common import (
    has_diff_target,
    has_index,
    has_pr_target,
    has_repo,
    require_diff_target,
    resolve_tool_ref,
)
from rbtr.llm.tools.git import changed_files, commit_log, diff
from rbtr.models import BranchTarget, PRTarget, SnapshotTarget
from rbtr.prompts import render_review
from rbtr.sessions.store import SessionStore
from rbtr.state import EngineState

from .ctx import tool_ctx

# ── Shared test data ─────────────────────────────────────────────────

_SNAPSHOT = SnapshotTarget(
    head_commit="abc123def456",
    ref_label="v2.1.0",
    updated_at=datetime(2025, 6, 1, tzinfo=UTC),
)

_BRANCH = BranchTarget(
    base_branch="main",
    head_branch="feature",
    base_commit="main",
    head_commit="feature",
    updated_at=0,
)

_PR = PRTarget(
    number=42,
    title="Fix",
    author="alice",
    base_branch="main",
    head_branch="fix",
    base_commit="abc",
    head_commit="def",
    updated_at=0,
)

_TOOL_DEF = ToolDefinition(name="test_tool")


def _snapshot_state(*, with_repo: bool = False) -> EngineState:
    """Build an EngineState with a SnapshotTarget."""
    state = EngineState(owner="acme", repo_name="app")
    state.review_target = _SNAPSHOT
    if with_repo:
        tmp = tempfile.mkdtemp()
        state.repo = pygit2.init_repository(tmp)
    return state


# ── SnapshotTarget model ────────────────────────────────────────────


def test_snapshot_target_fields() -> None:
    """SnapshotTarget stores head_commit, ref_label, updated_at."""
    assert _SNAPSHOT.head_commit == "abc123def456"
    assert _SNAPSHOT.ref_label == "v2.1.0"
    assert _SNAPSHOT.updated_at == datetime(2025, 6, 1, tzinfo=UTC)


def test_snapshot_target_serialise_roundtrip(store: SessionStore) -> None:
    """SnapshotTarget survives JSON round-trip."""
    data = _SNAPSHOT.model_dump()
    restored = SnapshotTarget.model_validate(data)
    assert restored == _SNAPSHOT


def test_snapshot_has_no_base_fields(store: SessionStore) -> None:
    """SnapshotTarget does not have base_branch or base_commit."""
    assert not hasattr(_SNAPSHOT, "base_branch")
    assert not hasattr(_SNAPSHOT, "base_commit")


# ── resolve_tool_ref ─────────────────────────────────────────────────


def test_resolve_tool_ref_head(store: SessionStore) -> None:
    """'head' resolves to the snapshot's commit."""
    state = _snapshot_state()
    result = resolve_tool_ref(tool_ctx(state, store), "head")
    assert result == "abc123def456"


def test_resolve_tool_ref_base_raises(store: SessionStore) -> None:
    """'base' raises RuntimeError for snapshots."""
    state = _snapshot_state()
    with pytest.raises(RuntimeError, match="no base commit"):
        resolve_tool_ref(tool_ctx(state, store), "base")


# ── Filter functions ─────────────────────────────────────────────────


def test_has_repo_true_for_snapshot(store: SessionStore) -> None:
    """has_repo returns True when repo + SnapshotTarget exist."""
    state = _snapshot_state(with_repo=True)
    assert has_repo(tool_ctx(state, store), _TOOL_DEF) is True


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        (_SNAPSHOT, False),
        (_BRANCH, True),
        (_PR, True),
        (None, False),
    ],
    ids=["snapshot", "branch", "pr", "none"],
)
def test_has_diff_target(target: object, expected: bool, store: SessionStore) -> None:
    """has_diff_target excludes SnapshotTarget and None."""
    with tempfile.TemporaryDirectory() as tmp:
        state = EngineState()
        state.repo = pygit2.init_repository(tmp)
        state.review_target = target  # type: ignore[assignment]
        result = has_diff_target(tool_ctx(state, store), _TOOL_DEF)
        assert result is expected


def test_has_diff_target_no_repo(store: SessionStore) -> None:
    """has_diff_target returns False when no repo."""
    state = EngineState()
    state.review_target = _BRANCH
    assert has_diff_target(tool_ctx(state, store), _TOOL_DEF) is False


def test_has_index_true_for_snapshot(store: SessionStore) -> None:
    """has_index works with SnapshotTarget."""
    from rbtr.index.store import IndexStore

    state = _snapshot_state()
    store = IndexStore()
    state.index = store
    assert has_index(tool_ctx(state, store), _TOOL_DEF) is True
    store.close()


def test_has_pr_target_false_for_snapshot(store: SessionStore) -> None:
    """has_pr_target rejects SnapshotTarget."""
    state = _snapshot_state()
    assert has_pr_target(tool_ctx(state, store), _TOOL_DEF) is False


# ── require_diff_target (per-tool prepare) ───────────────────────────


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("target", "visible"),
    [
        (_SNAPSHOT, False),
        (_BRANCH, True),
        (_PR, True),
        (None, False),
    ],
    ids=["snapshot", "branch", "pr", "none"],
)
async def test_require_diff_target(target: object, visible: bool, store: SessionStore) -> None:
    """require_diff_target hides tool for SnapshotTarget and None."""
    state = EngineState()
    state.review_target = target  # type: ignore[assignment]
    result = await require_diff_target(tool_ctx(state, store), _TOOL_DEF)
    if visible:
        assert result is _TOOL_DEF
    else:
        assert result is None


# ── Git tools return "No diff target" for snapshot ───────────────────


def test_diff_no_diff_target_for_snapshot(store: SessionStore) -> None:
    """diff returns error when target is a SnapshotTarget."""
    with tempfile.TemporaryDirectory() as tmp:
        state = _snapshot_state()
        state.repo = pygit2.init_repository(tmp)
        result = diff(tool_ctx(state, store))
        assert "No diff target" in result


def test_changed_files_no_diff_target_for_snapshot(store: SessionStore) -> None:
    """changed_files returns error when target is a SnapshotTarget."""
    with tempfile.TemporaryDirectory() as tmp:
        state = _snapshot_state()
        state.repo = pygit2.init_repository(tmp)
        result = changed_files(tool_ctx(state, store))
        assert "No diff target" in result


def test_commit_log_no_diff_target_for_snapshot(store: SessionStore) -> None:
    """commit_log returns error when target is a SnapshotTarget."""
    with tempfile.TemporaryDirectory() as tmp:
        state = _snapshot_state()
        state.repo = pygit2.init_repository(tmp)
        result = commit_log(tool_ctx(state, store))
        assert "No diff target" in result


# ── Prompt rendering ─────────────────────────────────────────────────


def test_review_prompt_snapshot_context() -> None:
    """Snapshot prompt includes ref label and commit."""
    state = _snapshot_state()
    text = render_review(state)
    assert "snapshot at `v2.1.0`" in text
    assert "abc123def456" in text


def test_review_prompt_snapshot_flow() -> None:
    """Snapshot prompt has Brief/Deepen/Evaluate but no Draft step."""
    state = _snapshot_state()
    text = render_review(state)
    assert "Brief" in text
    assert "Deepen" in text
    assert "Evaluate" in text
    assert "Draft" not in text


def test_review_prompt_snapshot_framing() -> None:
    """Snapshot prompt says 'codebase' not 'change' in the framing."""
    state = _snapshot_state()
    text = render_review(state)
    assert "mental model of the\ncodebase" in text


def test_review_prompt_snapshot_notes_example() -> None:
    """Snapshot prompt uses ref_label in notes example."""
    state = _snapshot_state()
    text = render_review(state)
    assert "snapshot-v2.1.0-notes.md" in text
