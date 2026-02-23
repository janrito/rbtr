"""Tests for draft management LLM tools — add, edit, remove, set_summary."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic_ai import RunContext

from rbtr.engine.agent import AgentDeps
from rbtr.engine.state import EngineState
from rbtr.engine.tools import (
    add_review_comment,
    edit_review_comment,
    remove_review_comment,
    set_review_summary,
)
from rbtr.github.draft import load_draft, save_draft
from rbtr.models import InlineComment, PRTarget, ReviewDraft

# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point WORKSPACE_DIR at a temp directory."""
    monkeypatch.setattr("rbtr.github.draft.WORKSPACE_DIR", tmp_path)
    return tmp_path


@pytest.fixture
def pr_target() -> PRTarget:
    from datetime import UTC, datetime

    return PRTarget(
        number=42,
        title="Test PR",
        author="alice",
        base_branch="main",
        head_branch="feature",
        updated_at=datetime(2025, 1, 1, tzinfo=UTC),
    )


@pytest.fixture
def ctx(pr_target: PRTarget) -> RunContext[AgentDeps]:
    """Build a minimal RunContext with a PR target in session."""
    session = EngineState()
    session.review_target = pr_target
    deps = AgentDeps(session=session)
    mock_ctx = MagicMock(spec=RunContext)
    mock_ctx.deps = deps
    return mock_ctx


# ── add_review_comment ───────────────────────────────────────────────


def test_add_comment(workspace: Path, ctx: RunContext[AgentDeps]) -> None:
    result = add_review_comment(ctx, "src/a.py", 10, "**blocker:** Bug here.")
    assert "Comment added" in result
    assert "1 comment" in result

    draft = load_draft(42)
    assert draft is not None
    assert len(draft.comments) == 1
    assert draft.comments[0].path == "src/a.py"
    assert draft.comments[0].body == "**blocker:** Bug here."


def test_add_comment_with_suggestion(workspace: Path, ctx: RunContext[AgentDeps]) -> None:
    result = add_review_comment(ctx, "src/a.py", 10, "Use this.", "fixed()")
    assert "Comment added" in result

    draft = load_draft(42)
    assert draft is not None
    assert draft.comments[0].suggestion == "fixed()"


def test_add_multiple_comments(workspace: Path, ctx: RunContext[AgentDeps]) -> None:
    add_review_comment(ctx, "src/a.py", 10, "First.")
    add_review_comment(ctx, "src/b.py", 20, "Second.")

    draft = load_draft(42)
    assert draft is not None
    assert len(draft.comments) == 2


# ── edit_review_comment ──────────────────────────────────────────────


def _seed_draft(pr_number: int) -> None:
    draft = ReviewDraft(
        summary="Review.",
        comments=[
            InlineComment(path="a.py", line=10, body="Original."),
            InlineComment(path="b.py", line=20, body="Also original."),
        ],
    )
    save_draft(pr_number, draft)


def test_edit_comment_body(workspace: Path, ctx: RunContext[AgentDeps]) -> None:
    _seed_draft(42)
    result = edit_review_comment(ctx, 1, body="Updated body.")
    assert "updated" in result

    draft = load_draft(42)
    assert draft is not None
    assert draft.comments[0].body == "Updated body."
    assert draft.comments[0].path == "a.py"


def test_edit_comment_clear_suggestion(workspace: Path, ctx: RunContext[AgentDeps]) -> None:
    draft = ReviewDraft(
        summary="",
        comments=[
            InlineComment(path="a.py", line=10, body="Fix.", suggestion="code"),
        ],
    )
    save_draft(42, draft)

    edit_review_comment(ctx, 1, suggestion="")
    loaded = load_draft(42)
    assert loaded is not None
    assert loaded.comments[0].suggestion == ""


def test_edit_comment_invalid_index(workspace: Path, ctx: RunContext[AgentDeps]) -> None:
    _seed_draft(42)
    result = edit_review_comment(ctx, 5, body="Nope.")
    assert "Invalid index" in result


def test_edit_comment_empty_draft(workspace: Path, ctx: RunContext[AgentDeps]) -> None:
    result = edit_review_comment(ctx, 1, body="Nope.")
    assert "no comments" in result


# ── remove_review_comment ────────────────────────────────────────────


def test_remove_comment(workspace: Path, ctx: RunContext[AgentDeps]) -> None:
    _seed_draft(42)
    result = remove_review_comment(ctx, 1)
    assert "Removed" in result

    draft = load_draft(42)
    assert draft is not None
    assert len(draft.comments) == 1
    assert draft.comments[0].path == "b.py"


def test_remove_last_comment(workspace: Path, ctx: RunContext[AgentDeps]) -> None:
    draft = ReviewDraft(
        summary="Summary.",
        comments=[InlineComment(path="a.py", line=1, body="Only.")],
    )
    save_draft(42, draft)

    result = remove_review_comment(ctx, 1)
    assert "0 comment" in result

    loaded = load_draft(42)
    assert loaded is not None
    assert loaded.comments == []
    assert loaded.summary == "Summary."


def test_remove_invalid_index(workspace: Path, ctx: RunContext[AgentDeps]) -> None:
    _seed_draft(42)
    result = remove_review_comment(ctx, 0)
    assert "Invalid index" in result


def test_remove_empty_draft(workspace: Path, ctx: RunContext[AgentDeps]) -> None:
    result = remove_review_comment(ctx, 1)
    assert "no comments" in result


# ── set_review_summary ───────────────────────────────────────────────


def test_set_summary(workspace: Path, ctx: RunContext[AgentDeps]) -> None:
    result = set_review_summary(ctx, "Great PR overall.")
    assert "updated" in result

    draft = load_draft(42)
    assert draft is not None
    assert draft.summary == "Great PR overall."


def test_set_summary_preserves_comments(workspace: Path, ctx: RunContext[AgentDeps]) -> None:
    _seed_draft(42)
    set_review_summary(ctx, "New summary.")

    draft = load_draft(42)
    assert draft is not None
    assert draft.summary == "New summary."
    assert len(draft.comments) == 2


def test_set_summary_overwrites(workspace: Path, ctx: RunContext[AgentDeps]) -> None:
    set_review_summary(ctx, "First.")
    set_review_summary(ctx, "Second.")

    draft = load_draft(42)
    assert draft is not None
    assert draft.summary == "Second."
