"""Tests for get_pr_discussion and discussion formatting."""

from __future__ import annotations

from datetime import UTC, datetime

from rbtr.github.client import (
    _aggregate_reactions,
    _inline_to_entry,
    _issue_comment_to_entry,
    _review_to_entry,
    format_discussion_entry,
    get_pr_discussion,
)
from rbtr.models import DiscussionEntryKind

from .conftest import (
    FakeGithub,
    FakeInlineComment,
    FakeIssueComment,
    FakePR,
    FakeReaction,
    FakeRepo,
    FakeReview,
    FakeUser,
)

# ── aggregate_reactions ──────────────────────────────────────────────


def test_aggregate_reactions_empty() -> None:
    assert _aggregate_reactions([]) == {}


def test_aggregate_reactions_counts() -> None:
    reactions = [FakeReaction("+1"), FakeReaction("+1"), FakeReaction("-1")]
    result = _aggregate_reactions(reactions)  # type: ignore[arg-type]  # fake stubs
    assert result == {"+1": 2, "-1": 1}


# ── Converter functions ──────────────────────────────────────────────


def test_review_to_entry() -> None:
    review = FakeReview(review_id=1, body="LGTM", state="APPROVED")
    entry = _review_to_entry(review)  # type: ignore[arg-type]  # fake stub
    assert entry.kind == DiscussionEntryKind.REVIEW
    assert entry.comment_id == 1
    assert entry.author == "alice"
    assert entry.review_state == "APPROVED"
    assert entry.body == "LGTM"


def test_review_to_entry_bot() -> None:
    review = FakeReview(user=FakeUser("codecov", "Bot"))
    entry = _review_to_entry(review)  # type: ignore[arg-type]  # fake stub
    assert entry.is_bot is True


def test_inline_to_entry() -> None:
    comment = FakeInlineComment(
        path="src/api.py",
        line=10,
        diff_hunk="@@ context",
        reactions=[FakeReaction("eyes")],
    )
    entry = _inline_to_entry(comment)  # type: ignore[arg-type]  # fake stub
    assert entry.kind == DiscussionEntryKind.INLINE
    assert entry.path == "src/api.py"
    assert entry.line == 10
    assert entry.diff_hunk == "@@ context"
    assert entry.reactions == {"eyes": 1}


def test_inline_to_entry_reply() -> None:
    comment = FakeInlineComment(in_reply_to_id=5)
    entry = _inline_to_entry(comment)  # type: ignore[arg-type]  # fake stub
    assert entry.in_reply_to_id == 5


def test_issue_comment_to_entry() -> None:
    comment = FakeIssueComment(body="Nice work!")
    entry = _issue_comment_to_entry(comment)  # type: ignore[arg-type]  # fake stub
    assert entry.kind == DiscussionEntryKind.COMMENT
    assert entry.body == "Nice work!"


def test_issue_comment_bot() -> None:
    comment = FakeIssueComment(user=FakeUser("dependabot", "Bot"))
    entry = _issue_comment_to_entry(comment)  # type: ignore[arg-type]  # fake stub
    assert entry.is_bot is True


# ── get_pr_discussion integration ────────────────────────────────────


def test_discussion_sorted_chronologically() -> None:
    """Entries are returned oldest first regardless of type."""
    review = FakeReview(
        review_id=1,
        submitted_at=datetime(2025, 1, 15, 12, 0, tzinfo=UTC),
    )
    inline = FakeInlineComment(
        comment_id=2,
        created_at=datetime(2025, 1, 15, 10, 0, tzinfo=UTC),
    )
    issue = FakeIssueComment(
        comment_id=3,
        created_at=datetime(2025, 1, 15, 11, 0, tzinfo=UTC),
    )
    pr = FakePR(reviews=[review], inline_comments=[inline], issue_comments=[issue])
    gh = FakeGithub(FakeRepo(pr))

    entries = get_pr_discussion(gh, "owner", "repo", 1)  # type: ignore[arg-type]  # fake stub
    assert len(entries) == 3
    assert entries[0].comment_id == 2  # inline (10:00)
    assert entries[1].comment_id == 3  # issue (11:00)
    assert entries[2].comment_id == 1  # review (12:00)


def test_discussion_skips_empty_review_bodies() -> None:
    """Reviews without a body (e.g. approval clicks) are excluded."""
    review_with_body = FakeReview(review_id=1, body="Good stuff.")
    review_empty = FakeReview(review_id=2, body="")
    pr = FakePR(reviews=[review_with_body, review_empty])
    gh = FakeGithub(FakeRepo(pr))

    entries = get_pr_discussion(gh, "owner", "repo", 1)  # type: ignore[arg-type]  # fake stub
    assert len(entries) == 1
    assert entries[0].comment_id == 1


def test_discussion_empty_pr() -> None:
    gh = FakeGithub()
    entries = get_pr_discussion(gh, "owner", "repo", 1)  # type: ignore[arg-type]  # fake stub
    assert entries == []


# ── Formatting ───────────────────────────────────────────────────────


def test_format_review_entry() -> None:
    review = FakeReview(state="CHANGES_REQUESTED", body="Needs work.")
    entry = _review_to_entry(review)  # type: ignore[arg-type]  # fake stub
    text = format_discussion_entry(entry)
    assert "CHANGES_REQUESTED" in text
    assert "@alice" in text
    assert "Needs work." in text


def test_format_inline_entry_with_diff_hunk() -> None:
    comment = FakeInlineComment(
        path="src/handler.py",
        line=42,
        body="Off by one.",
        diff_hunk="@@ -40,5 +40,5 @@\n def process():",
    )
    entry = _inline_to_entry(comment)  # type: ignore[arg-type]  # fake stub
    text = format_discussion_entry(entry)
    assert "src/handler.py:42" in text
    assert "Off by one." in text
    assert "@@ -40,5 +40,5 @@" in text


def test_format_inline_reply() -> None:
    comment = FakeInlineComment(in_reply_to_id=5, body="Agreed.")
    entry = _inline_to_entry(comment)  # type: ignore[arg-type]  # fake stub
    text = format_discussion_entry(entry)
    assert "(reply)" in text


def test_format_with_reactions() -> None:
    comment = FakeIssueComment(
        reactions=[FakeReaction("+1"), FakeReaction("+1"), FakeReaction("heart")],
    )
    entry = _issue_comment_to_entry(comment)  # type: ignore[arg-type]  # fake stub
    text = format_discussion_entry(entry)
    assert "Reactions:" in text
    assert "+1 2" in text
    assert "heart 1" in text


def test_format_bot_tag() -> None:
    comment = FakeIssueComment(user=FakeUser("ci-bot", "Bot"))
    entry = _issue_comment_to_entry(comment)  # type: ignore[arg-type]  # fake stub
    text = format_discussion_entry(entry)
    assert "[bot]" in text
