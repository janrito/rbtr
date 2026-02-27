"""Tests for posting and pushing reviews to GitHub."""

from __future__ import annotations

from pathlib import Path

import pytest

from rbtr.github.client import (
    delete_pending_review,
    format_comment_body,
    post_review,
    push_pending_review,
)
from rbtr.github.draft import save_draft
from rbtr.models import InlineComment, ReviewDraft, ReviewEvent

from .conftest import FakeGithub, FakePR, FakeRepo, FakeReview, fake_ctx

# ── format_comment_body ──────────────────────────────────────────────


def test_format_body_plain() -> None:
    c = InlineComment(path="a.py", line=1, body="Fix this.")
    assert format_comment_body(c) == "Fix this."


def test_format_body_with_suggestion() -> None:
    c = InlineComment(path="a.py", line=1, body="Use this instead.", suggestion="better()")
    result = format_comment_body(c)
    assert "Use this instead." in result
    assert "```suggestion\nbetter()\n```" in result


def test_format_body_no_suggestion_when_empty() -> None:
    c = InlineComment(path="a.py", line=1, body="Comment.", suggestion="")
    assert format_comment_body(c) == "Comment."


# ── post_review ──────────────────────────────────────────────────────


def test_post_review_sends_correct_data() -> None:
    pr = FakePR()
    gh = FakeGithub(FakeRepo(pr))
    draft = ReviewDraft(
        summary="Looks good overall.",
        comments=[
            InlineComment(path="src/a.py", line=10, body="**blocker:** Bug."),
            InlineComment(path="src/b.py", line=20, body="Nit.", suggestion="fixed()"),
        ],
    )

    url = post_review(fake_ctx(gh), 1, draft, ReviewEvent.COMMENT)
    assert url == "https://github.com/pr/1#review"
    assert len(pr.created_reviews) == 1

    review = pr.created_reviews[0]
    assert review["body"] == "Looks good overall."
    assert review["event"] == "COMMENT"
    assert len(review["comments"]) == 2
    assert review["comments"][0]["path"] == "src/a.py"
    assert review["comments"][0]["line"] == 10
    assert "suggestion\nfixed()\n" in review["comments"][1]["body"]


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        (ReviewEvent.APPROVE, "APPROVE"),
        (ReviewEvent.REQUEST_CHANGES, "REQUEST_CHANGES"),
    ],
)
def test_post_review_event_types(event: ReviewEvent, expected: str) -> None:
    pr = FakePR()
    gh = FakeGithub(FakeRepo(pr))
    draft = ReviewDraft(summary=".")

    post_review(fake_ctx(gh), 1, draft, event)
    assert pr.created_reviews[0]["event"] == expected


# ── push_pending_review ───────────────────────────────────────────────


def test_push_pending_creates_without_event() -> None:
    """push_pending_review creates a review with no event (PENDING state)."""
    pr = FakePR()
    gh = FakeGithub(FakeRepo(pr))
    draft = ReviewDraft(
        summary="WIP review.",
        comments=[InlineComment(path="a.py", line=5, body="Needs work.")],
    )

    review_id = push_pending_review(fake_ctx(gh), 1, draft)
    assert review_id == 200
    assert len(pr.created_reviews) == 1

    review = pr.created_reviews[0]
    assert review["body"] == "WIP review."
    assert review["event"] == ""  # no event = PENDING
    assert len(review["comments"]) == 1
    assert review["comments"][0]["path"] == "a.py"


def test_push_pending_formats_suggestions() -> None:
    pr = FakePR()
    gh = FakeGithub(FakeRepo(pr))
    draft = ReviewDraft(
        comments=[InlineComment(path="a.py", line=1, body="Fix.", suggestion="better()")],
    )

    push_pending_review(fake_ctx(gh), 1, draft)
    assert "```suggestion" in pr.created_reviews[0]["comments"][0]["body"]


# ── delete_pending_review ─────────────────────────────────────────────


def test_delete_pending_review() -> None:
    tracked = FakeReview(review_id=42)
    pr = FakePR(reviews=[tracked])
    gh = FakeGithub(FakeRepo(pr))

    delete_pending_review(fake_ctx(gh), 1, 42)
    assert tracked.deleted is True


# ── Draft persistence after post ─────────────────────────────────────


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr("rbtr.config.config.tools.drafts_dir", str(tmp_path / "drafts"))
    return tmp_path


def test_draft_file_used_for_post(workspace: Path) -> None:
    """Verify a draft saved to disk can be loaded and posted."""
    draft = ReviewDraft(
        summary="Summary.",
        comments=[InlineComment(path="x.py", line=5, body="Comment.")],
    )
    save_draft(99, draft)

    from rbtr.github.draft import load_draft

    loaded = load_draft(99)
    assert loaded is not None

    pr = FakePR()
    gh = FakeGithub(FakeRepo(pr))
    post_review(fake_ctx(gh), 99, loaded, ReviewEvent.COMMENT)
    assert len(pr.created_reviews) == 1
    assert pr.created_reviews[0]["comments"][0]["path"] == "x.py"
