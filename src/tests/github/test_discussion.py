"""Tests for get_pr_discussion and discussion formatting."""

from __future__ import annotations

from datetime import UTC, datetime

from github import Github
from github.PullRequest import PullRequest
from pytest_mock import MockerFixture

from rbtr.github.client import (
    GitHubCtx,
    _aggregate_reactions,
    _inline_to_entry,
    _issue_comment_to_entry,
    _review_to_entry,
    get_pr_discussion,
)
from rbtr.llm.tools.discussion import format_discussion_entry
from rbtr.models import DiscussionEntryKind

from .conftest import (
    make_comment,
    make_issue_comment,
    make_review,
)

# ── aggregate_reactions ──────────────────────────────────────────────


def test_aggregate_reactions_empty() -> None:
    assert _aggregate_reactions([]) == {}


def test_aggregate_reactions_counts(mocker: MockerFixture) -> None:
    r1, r2, r3 = (
        mocker.MagicMock(content="+1"),
        mocker.MagicMock(content="+1"),
        mocker.MagicMock(content="-1"),
    )
    assert _aggregate_reactions([r1, r2, r3]) == {"+1": 2, "-1": 1}


# ── Converter functions ──────────────────────────────────────────────


def test_review_to_entry(mocker: MockerFixture) -> None:
    entry = _review_to_entry(
        make_review(
            mocker,
        )
    )
    assert entry.kind == DiscussionEntryKind.REVIEW
    assert entry.comment_id == 1
    assert entry.author == "alice"
    assert entry.review_state == "APPROVED"
    assert entry.body == "LGTM"


def test_review_to_entry_bot(mocker: MockerFixture) -> None:
    entry = _review_to_entry(make_review(mocker, user="codecov", user_type="Bot"))
    assert entry.is_bot is True


def test_inline_to_entry(mocker: MockerFixture) -> None:
    entry = _inline_to_entry(
        make_comment(
            mocker,
            path="src/api.py",
            line=10,
            diff_hunk="@@ ctx",
            reactions=[mocker.MagicMock(content="eyes")],
        )
    )
    assert entry.kind == DiscussionEntryKind.INLINE
    assert entry.path == "src/api.py"
    assert entry.line == 10
    assert entry.diff_hunk == "@@ ctx"
    assert entry.reactions == {"eyes": 1}


def test_inline_to_entry_reply(mocker: MockerFixture) -> None:
    entry = _inline_to_entry(make_comment(mocker, in_reply_to_id=5))
    assert entry.in_reply_to_id == 5


def test_issue_comment_to_entry(mocker: MockerFixture) -> None:
    entry = _issue_comment_to_entry(make_issue_comment(mocker, body="Nice work!"))
    assert entry.kind == DiscussionEntryKind.COMMENT
    assert entry.body == "Nice work!"


def test_issue_comment_bot(mocker: MockerFixture) -> None:
    entry = _issue_comment_to_entry(make_issue_comment(mocker, user="dependabot", user_type="Bot"))
    assert entry.is_bot is True


# ── get_pr_discussion integration ────────────────────────────────────


def test_discussion_sorted_chronologically(
    gh: Github,
    mock_pr: PullRequest,
    mocker: MockerFixture,
) -> None:
    """Entries are returned oldest first regardless of type."""
    mock_pr.get_reviews.return_value = [  # type: ignore[attr-defined]  # mock attrs on autospecced PyGithub type
        make_review(mocker, review_id=1, submitted_at=datetime(2025, 1, 15, 12, 0, tzinfo=UTC)),
    ]
    mock_pr.get_review_comments.return_value = [  # type: ignore[attr-defined]  # mock attrs on autospecced PyGithub type
        make_comment(mocker, comment_id=2, created_at=datetime(2025, 1, 15, 10, 0, tzinfo=UTC)),
    ]
    mock_pr.get_issue_comments.return_value = [  # type: ignore[attr-defined]  # mock attrs on autospecced PyGithub type
        make_issue_comment(
            mocker, comment_id=3, created_at=datetime(2025, 1, 15, 11, 0, tzinfo=UTC)
        ),
    ]

    entries = get_pr_discussion(GitHubCtx(gh=gh, owner="o", repo_name="r"), 1)
    assert len(entries) == 3
    assert entries[0].comment_id == 2  # inline (10:00)
    assert entries[1].comment_id == 3  # issue (11:00)
    assert entries[2].comment_id == 1  # review (12:00)


def test_discussion_skips_empty_review_bodies(
    gh: Github,
    mock_pr: PullRequest,
    mocker: MockerFixture,
) -> None:
    mock_pr.get_reviews.return_value = [  # type: ignore[attr-defined]  # mock attrs on autospecced PyGithub type
        make_review(mocker, review_id=1, body="Good stuff."),
        make_review(mocker, review_id=2, body=""),
    ]

    entries = get_pr_discussion(GitHubCtx(gh=gh, owner="o", repo_name="r"), 1)
    assert len(entries) == 1
    assert entries[0].comment_id == 1


def test_discussion_empty_pr(gh: Github, mock_pr: PullRequest) -> None:
    entries = get_pr_discussion(GitHubCtx(gh=gh, owner="o", repo_name="r"), 1)
    assert entries == []


# ── Formatting ───────────────────────────────────────────────────────


def test_format_review_entry(mocker: MockerFixture) -> None:
    entry = _review_to_entry(make_review(mocker, state="CHANGES_REQUESTED", body="Needs work."))
    text = format_discussion_entry(entry)
    assert "CHANGES_REQUESTED" in text
    assert "@alice" in text
    assert "Needs work." in text


def test_format_inline_entry_with_diff_hunk(mocker: MockerFixture) -> None:
    entry = _inline_to_entry(
        make_comment(
            mocker,
            path="src/handler.py",
            line=42,
            body="Off by one.",
            diff_hunk="@@ -40,5 +40,5 @@\n def process():",
        )
    )
    text = format_discussion_entry(entry)
    assert "src/handler.py:42" in text
    assert "Off by one." in text
    assert "@@ -40,5 +40,5 @@" in text


def test_format_inline_reply(mocker: MockerFixture) -> None:
    entry = _inline_to_entry(make_comment(mocker, in_reply_to_id=5, body="Agreed."))
    text = format_discussion_entry(entry)
    assert "(reply)" in text


def test_format_with_reactions(mocker: MockerFixture) -> None:
    entry = _issue_comment_to_entry(
        make_issue_comment(
            mocker,
            reactions=[
                mocker.MagicMock(content="+1"),
                mocker.MagicMock(content="+1"),
                mocker.MagicMock(content="heart"),
            ],
        )
    )
    text = format_discussion_entry(entry)
    assert "Reactions:" in text
    assert "+1 2" in text
    assert "heart 1" in text


def test_format_bot_tag(mocker: MockerFixture) -> None:
    entry = _issue_comment_to_entry(make_issue_comment(mocker, user="ci-bot", user_type="Bot"))
    text = format_discussion_entry(entry)
    assert "[bot]" in text
