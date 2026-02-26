"""Tests for draft sync — get_pending_review and sync flow."""

from __future__ import annotations

import queue
from pathlib import Path
from typing import Any

import pytest

from rbtr.engine.review import sync_review_draft
from rbtr.engine.state import EngineState
from rbtr.events import Event, FlushPanel, Output
from rbtr.exceptions import RbtrError
from rbtr.github.client import get_pending_review, parse_comment_body
from rbtr.github.draft import _comment_hash, load_draft, save_draft
from rbtr.models import InlineComment, PRTarget, ReviewDraft

from .conftest import (
    FakeGithub,
    FakeInlineComment,
    FakePR,
    FakeRepo,
    FakeReview,
    FakeUser,
    fake_ctx,
)

# ── parse_comment_body ───────────────────────────────────────────────


def test_parse_body_plain() -> None:
    body, suggestion = parse_comment_body("Fix this bug.")
    assert body == "Fix this bug."
    assert suggestion == ""


def test_parse_body_with_suggestion() -> None:
    raw = "Use this instead.\n\n```suggestion\nbetter()\n```"
    body, suggestion = parse_comment_body(raw)
    assert body == "Use this instead."
    assert suggestion == "better()"


def test_parse_body_multiline_suggestion() -> None:
    raw = "Fix.\n\n```suggestion\nline1\nline2\n```"
    body, suggestion = parse_comment_body(raw)
    assert body == "Fix."
    assert suggestion == "line1\nline2"


def test_parse_body_no_closing_fence() -> None:
    raw = "Fix.\n\n```suggestion\norphan code"
    body, suggestion = parse_comment_body(raw)
    assert body == "Fix."
    assert suggestion == "orphan code"


# ── get_pending_review ───────────────────────────────────────────────


def test_pending_review_found() -> None:
    comments = [FakeInlineComment(comment_id=10, path="a.py", line=10, body="**blocker:** Bug.")]
    pr = FakePR(
        reviews=[FakeReview(review_id=100, state="PENDING", user=FakeUser("reviewer"))],
        review_comments_by_id={100: comments},
    )
    gh = FakeGithub(FakeRepo(pr))

    result = get_pending_review(fake_ctx(gh), 1, "reviewer")
    assert result is not None
    assert result.github_review_id == 100
    assert len(result.comments) == 1
    assert result.comments[0].path == "a.py"
    assert result.comments[0].github_id == 10


def test_pending_review_not_found() -> None:
    pr = FakePR(reviews=[FakeReview(state="APPROVED")])
    gh = FakeGithub(FakeRepo(pr))

    result = get_pending_review(fake_ctx(gh), 1, "reviewer")
    assert result is None


def test_pending_review_wrong_user() -> None:
    pr = FakePR(reviews=[FakeReview(state="PENDING", user=FakeUser("other"))])
    gh = FakeGithub(FakeRepo(pr))

    result = get_pending_review(fake_ctx(gh), 1, "reviewer")
    assert result is None


def test_pending_review_picks_latest() -> None:
    """When multiple PENDING reviews exist, the last one wins."""
    pr = FakePR(
        reviews=[
            FakeReview(review_id=1, state="PENDING", user=FakeUser("reviewer")),
            FakeReview(review_id=2, state="PENDING", user=FakeUser("reviewer")),
        ],
        review_comments_by_id={
            1: [FakeInlineComment(comment_id=10, body="old")],
            2: [FakeInlineComment(comment_id=20, body="new")],
        },
    )
    gh = FakeGithub(FakeRepo(pr))

    result = get_pending_review(fake_ctx(gh), 1, "reviewer")
    assert result is not None
    assert result.github_review_id == 2
    assert result.comments[0].body == "new"


def test_pending_review_preserves_body() -> None:
    pr = FakePR(
        reviews=[
            FakeReview(
                review_id=1,
                state="PENDING",
                body="Overall good.",
                user=FakeUser("reviewer"),
            )
        ],
        review_comments_by_id={1: []},
    )
    gh = FakeGithub(FakeRepo(pr))

    result = get_pending_review(fake_ctx(gh), 1, "reviewer")
    assert result is not None
    assert result.summary == "Overall good."


def test_pending_review_parses_suggestion() -> None:
    """Suggestion blocks in GitHub body are parsed into separate field."""
    raw_body = "Use this.\n\n```suggestion\nbetter()\n```"
    comments = [FakeInlineComment(comment_id=10, body=raw_body)]
    pr = FakePR(
        reviews=[FakeReview(review_id=1, state="PENDING", user=FakeUser("reviewer"))],
        review_comments_by_id={1: comments},
    )
    gh = FakeGithub(FakeRepo(pr))

    result = get_pending_review(fake_ctx(gh), 1, "reviewer")
    assert result is not None
    assert result.comments[0].body == "Use this."
    assert result.comments[0].suggestion == "better()"


# ── sync_review_draft orchestration ──────────────────────────────────


class _FakeEngine:
    """Minimal engine stub for sync_review_draft."""

    def __init__(self, *, gh: Any = None, gh_username: str = "") -> None:
        from datetime import UTC, datetime

        self.state = EngineState()
        self.state.review_target = PRTarget(
            number=42,
            title="Test PR",
            author="alice",
            base_branch="main",
            head_branch="feature",
            updated_at=datetime(2025, 1, 1, tzinfo=UTC),
        )
        self.state.gh = gh
        self.state.gh_username = gh_username
        self.state.owner = "owner"
        self.state.repo_name = "repo"
        self._events: queue.Queue[Event] = queue.Queue()

    def _emit(self, event: Event) -> None:
        self._events.put(event)

    def _out(self, text: str, style: str = "") -> None:
        self._emit(Output(text=text, style=style))

    def _warn(self, text: str) -> None:
        self._emit(Output(text=text, style="rbtr.out.warning"))

    def _clear(self) -> None:
        self._emit(FlushPanel(discard=True))

    def _check_cancel(self) -> None:
        pass

    def collected_text(self) -> str:
        lines: list[str] = []
        while not self._events.empty():
            ev = self._events.get_nowait()
            if isinstance(ev, Output):
                lines.append(ev.text)
        return "\n".join(lines)


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr("rbtr.github.draft.WORKSPACE_DIR", tmp_path)
    return tmp_path


def test_sync_orchestration_pulls_and_pushes(workspace: Path) -> None:
    """Full bidirectional sync: remote comments are merged then pushed."""
    # Local draft has one comment.
    local_comment = InlineComment(path="a.py", line=10, body="Local finding.")
    save_draft(42, ReviewDraft(summary="Local summary.", comments=[local_comment]))

    # Remote pending review has a different comment.
    remote_comment = FakeInlineComment(comment_id=50, path="b.py", line=20, body="Remote finding.")
    pr = FakePR(
        reviews=[FakeReview(review_id=99, state="PENDING", user=FakeUser("reviewer"))],
        review_comments_by_id={99: [remote_comment]},
    )
    gh = FakeGithub(FakeRepo(pr))
    engine = _FakeEngine(gh=gh, gh_username="reviewer")

    sync_review_draft(engine, 42)  # type: ignore[arg-type]  # FakeEngine stub

    # Local draft should have both comments.
    draft = load_draft(42)
    assert draft is not None
    assert len(draft.comments) == 2

    # Old pending review should have been deleted.
    review_99 = pr.get_review(99)
    assert review_99.deleted is True

    # New pending review should have been pushed with 2 comments.
    assert len(pr.created_reviews) == 1
    assert len(pr.created_reviews[0]["comments"]) == 2


def test_sync_orchestration_no_draft_no_remote(workspace: Path) -> None:
    """When there's no local draft and no remote, sync is a no-op."""
    pr = FakePR(reviews=[])
    gh = FakeGithub(FakeRepo(pr))
    engine = _FakeEngine(gh=gh, gh_username="reviewer")

    sync_review_draft(engine, 42)  # type: ignore[arg-type]  # FakeEngine stub

    assert load_draft(42) is None
    assert len(pr.created_reviews) == 0
    assert "Nothing to sync" in engine.collected_text()


def test_sync_orchestration_not_authenticated(workspace: Path) -> None:
    """sync_review_draft raises when not authenticated."""
    engine = _FakeEngine()

    with pytest.raises(RbtrError, match="Not authenticated"):
        sync_review_draft(engine, 42)  # type: ignore[arg-type]  # FakeEngine stub


def test_sync_orchestration_local_only_pushes(workspace: Path) -> None:
    """When there's a local draft but no remote, push without delete."""
    save_draft(
        42,
        ReviewDraft(
            summary="My review.",
            comments=[InlineComment(path="a.py", line=5, body="Issue.")],
        ),
    )

    pr = FakePR(reviews=[])
    gh = FakeGithub(FakeRepo(pr))
    engine = _FakeEngine(gh=gh, gh_username="reviewer")

    sync_review_draft(engine, 42)  # type: ignore[arg-type]  # FakeEngine stub

    # Pushed, no delete.
    assert len(pr.created_reviews) == 1
    assert len(pr.created_reviews[0]["comments"]) == 1
    assert "Draft synced" in engine.collected_text()


def test_sync_orchestration_remote_summary_used(workspace: Path) -> None:
    """Remote review body becomes local summary when local has none."""
    remote_comment = FakeInlineComment(comment_id=10, path="a.py", line=10, body="Fix.")
    pr = FakePR(
        reviews=[
            FakeReview(
                review_id=50,
                state="PENDING",
                body="Overall assessment.",
                user=FakeUser("reviewer"),
            )
        ],
        review_comments_by_id={50: [remote_comment]},
    )
    gh = FakeGithub(FakeRepo(pr))
    engine = _FakeEngine(gh=gh, gh_username="reviewer")

    sync_review_draft(engine, 42)  # type: ignore[arg-type]  # FakeEngine stub

    draft = load_draft(42)
    assert draft is not None
    assert draft.summary == "Overall assessment."


def test_sync_saves_sync_fields(workspace: Path) -> None:
    """After sync, draft has github_review_id and comments have comment_hash."""
    save_draft(
        42,
        ReviewDraft(
            summary="Review.",
            comments=[InlineComment(path="a.py", line=5, body="Finding.")],
        ),
    )

    # The re-fetch after push returns the new review's comments.
    pushed_comment = FakeInlineComment(comment_id=500, path="a.py", line=5, body="Finding.")
    pr = FakePR(
        reviews=[
            FakeReview(review_id=200, state="PENDING", user=FakeUser("reviewer")),
        ],
        review_comments_by_id={200: [pushed_comment]},
    )
    gh = FakeGithub(FakeRepo(pr))
    engine = _FakeEngine(gh=gh, gh_username="reviewer")

    sync_review_draft(engine, 42)  # type: ignore[arg-type]  # FakeEngine stub

    draft = load_draft(42)
    assert draft is not None
    assert draft.github_review_id == 200
    assert draft.summary_hash != ""
    assert draft.comments[0].github_id == 500
    assert draft.comments[0].comment_hash == _comment_hash(draft.comments[0])


def test_sync_detects_remote_edit(workspace: Path) -> None:
    """Sync pulls a remotely-edited comment and accepts the edit."""
    # Local has a synced comment (comment_hash matches "Original.").
    original = InlineComment(path="a.py", line=10, body="Original.", github_id=100)
    synced = original.model_copy(update={"comment_hash": _comment_hash(original)})
    save_draft(
        42,
        ReviewDraft(
            summary="Summary.",
            comments=[synced],
            github_review_id=99,
        ),
    )

    # Remote has the same comment, but edited.
    remote_comment = FakeInlineComment(comment_id=100, path="a.py", line=10, body="Edited on GH.")
    pr = FakePR(
        reviews=[FakeReview(review_id=99, state="PENDING", user=FakeUser("reviewer"))],
        review_comments_by_id={99: [remote_comment]},
    )
    gh = FakeGithub(FakeRepo(pr))
    engine = _FakeEngine(gh=gh, gh_username="reviewer")

    sync_review_draft(engine, 42)  # type: ignore[arg-type]  # FakeEngine stub

    draft = load_draft(42)
    assert draft is not None
    # The remote edit should have been accepted since local was clean.
    assert draft.comments[0].body == "Edited on GH."
