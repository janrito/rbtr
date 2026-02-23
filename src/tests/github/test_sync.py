"""Tests for draft sync — get_pending_review and sync flow."""

from __future__ import annotations

import queue
from pathlib import Path
from typing import Any

import pytest

from rbtr.engine.review import sync_review_draft
from rbtr.engine.session import Session
from rbtr.events import Event, FlushPanel, Output
from rbtr.github.client import get_pending_review
from rbtr.github.draft import load_draft, merge_remote, save_draft
from rbtr.models import InlineComment, PendingReview, PRTarget, ReviewDraft

from .conftest import (
    FakeGithub,
    FakeInlineComment,
    FakePR,
    FakeRepo,
    FakeReview,
    FakeUser,
    fake_ctx,
)

# ── get_pending_review ───────────────────────────────────────────────


def test_pending_review_found() -> None:
    comments = [FakeInlineComment(path="a.py", line=10, body="**blocker:** Bug.")]
    pr = FakePR(
        reviews=[FakeReview(review_id=100, state="PENDING", user=FakeUser("reviewer"))],
        review_comments_by_id={100: comments},
    )
    gh = FakeGithub(FakeRepo(pr))

    result = get_pending_review(fake_ctx(gh), 1, "reviewer")
    assert result is not None
    assert result.review_id == 100
    assert len(result.comments) == 1
    assert result.comments[0].path == "a.py"


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
            1: [FakeInlineComment(body="old")],
            2: [FakeInlineComment(body="new")],
        },
    )
    gh = FakeGithub(FakeRepo(pr))

    result = get_pending_review(fake_ctx(gh), 1, "reviewer")
    assert result is not None
    assert result.review_id == 2
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
    assert result.body == "Overall good."


# ── Sync flow (merge_remote + persistence) ───────────────────────────


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr("rbtr.github.draft.WORKSPACE_DIR", tmp_path)
    return tmp_path


def test_sync_seeds_empty_local(workspace: Path) -> None:
    """When no local draft exists, remote comments become the draft."""
    remote = [InlineComment(path="a.py", line=10, body="Fix.")]
    merged = merge_remote(None, remote)
    save_draft(42, merged)

    loaded = load_draft(42)
    assert loaded is not None
    assert len(loaded.comments) == 1
    assert loaded.comments[0].path == "a.py"


def test_sync_merges_without_duplicates(workspace: Path) -> None:
    """Remote comments already in local are not duplicated."""
    existing = InlineComment(path="a.py", line=10, body="Local.")
    local = ReviewDraft(summary="My review.", comments=[existing])
    save_draft(42, local)

    remote_dup = InlineComment(path="a.py", line=10, body="Remote.")
    remote_new = InlineComment(path="b.py", line=20, body="New.")

    loaded = load_draft(42)
    merged = merge_remote(loaded, [remote_dup, remote_new])
    save_draft(42, merged)

    final = load_draft(42)
    assert final is not None
    assert len(final.comments) == 2
    assert final.comments[0].body == "Local."
    assert final.comments[1].path == "b.py"
    assert final.summary == "My review."


def test_sync_preserves_summary_from_remote() -> None:
    """When local has no summary, remote review body is used."""
    pending = PendingReview(
        review_id=100,
        body="Overall looks good.",
        comments=[],
    )
    local = merge_remote(None, pending.comments)
    if not local.summary and pending.body:
        local = local.model_copy(update={"summary": pending.body})

    assert local.summary == "Overall looks good."


# ── sync_review_draft orchestration ──────────────────────────────────
# These tests go through the full pull → merge → delete → push cycle
# in review.py, with real disk persistence and fake GitHub objects.


class _FakeEngine:
    """Minimal engine stub for sync_review_draft."""

    def __init__(self, *, gh: Any = None, gh_username: str = "") -> None:
        from datetime import UTC, datetime

        self.session = Session()
        self.session.review_target = PRTarget(
            number=42,
            title="Test PR",
            author="alice",
            base_branch="main",
            head_branch="feature",
            updated_at=datetime(2025, 1, 1, tzinfo=UTC),
        )
        self.session.gh = gh
        self.session.gh_username = gh_username
        self.session.owner = "owner"
        self.session.repo_name = "repo"
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


def test_sync_orchestration_pulls_and_pushes(workspace: Path) -> None:
    """Full bidirectional sync: remote comments are merged then pushed."""
    # Local draft has one comment.
    local_comment = InlineComment(path="a.py", line=10, body="Local finding.")
    save_draft(42, ReviewDraft(summary="Local summary.", comments=[local_comment]))

    # Remote pending review has a different comment.
    remote_comment = FakeInlineComment(path="b.py", line=20, body="Remote finding.")
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
    assert draft.comments[0].body == "Local finding."
    assert draft.comments[1].body == "Remote finding."
    assert draft.summary == "Local summary."

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
    """sync_review_draft warns when not authenticated."""
    engine = _FakeEngine()

    sync_review_draft(engine, 42)  # type: ignore[arg-type]  # FakeEngine stub
    assert "Not authenticated" in engine.collected_text()


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
    remote_comment = FakeInlineComment(path="a.py", line=10, body="Fix.")
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
