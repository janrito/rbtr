"""Shared fake PyGithub objects for GitHub integration tests.

All test files in this package use these stubs instead of the
real PyGithub classes.  Each fake implements only the methods
needed by the production code under test.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from rbtr.github.client import GitHubCtx


class FakeUser:
    def __init__(self, login: str = "alice", user_type: str = "User") -> None:
        self.login = login
        self.type = user_type


class FakeReaction:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeReview:
    """Stub for PullRequestReview."""

    def __init__(
        self,
        *,
        review_id: int = 100,
        body: str = "Looks good.",
        state: str = "APPROVED",
        user: FakeUser | None = None,
        submitted_at: datetime | None = None,
        html_url: str = "https://github.com/pr/1#review",
    ) -> None:
        self.id = review_id
        self.body = body
        self.state = state
        self.user = user or FakeUser()
        self.submitted_at = submitted_at or datetime(2025, 1, 15, 10, 0, tzinfo=UTC)
        self.html_url = html_url
        self.deleted = False

    def delete(self) -> None:
        self.deleted = True


class FakeInlineComment:
    """Stub for PullRequestComment (inline review comment)."""

    def __init__(
        self,
        *,
        comment_id: int = 10,
        body: str = "Fix this.",
        path: str = "src/handler.py",
        line: int | None = 42,
        diff_hunk: str = "@@ -40,5 +40,5 @@",
        user: FakeUser | None = None,
        created_at: datetime | None = None,
        in_reply_to_id: int | None = None,
        reactions: list[FakeReaction] | None = None,
    ) -> None:
        self.id = comment_id
        self.body = body
        self.path = path
        self.line = line
        self.diff_hunk = diff_hunk
        self.user = user or FakeUser()
        self.created_at = created_at or datetime(2025, 1, 15, 11, 0, tzinfo=UTC)
        self.in_reply_to_id = in_reply_to_id
        self._reactions = reactions or []
        # Matches GithubObject._rawData so get_pending_review can
        # read fields without triggering lazy-load completion.
        self._rawData = {
            "id": comment_id,
            "path": path,
            "line": line,
            "body": body,
        }

    def get_reactions(self) -> list[FakeReaction]:
        return self._reactions


class FakeIssueComment:
    """Stub for IssueComment."""

    def __init__(
        self,
        *,
        comment_id: int = 20,
        body: str = "General comment.",
        user: FakeUser | None = None,
        created_at: datetime | None = None,
        reactions: list[FakeReaction] | None = None,
    ) -> None:
        self.id = comment_id
        self.body = body
        self.user = user or FakeUser()
        self.created_at = created_at or datetime(2025, 1, 15, 12, 0, tzinfo=UTC)
        self._reactions = reactions or []

    def get_reactions(self) -> list[FakeReaction]:
        return self._reactions


class FakePR:
    """Stub for PullRequest — supports all API surfaces used in tests."""

    def __init__(
        self,
        *,
        reviews: list[FakeReview] | None = None,
        inline_comments: list[FakeInlineComment] | None = None,
        issue_comments: list[FakeIssueComment] | None = None,
        review_comments_by_id: dict[int, list[FakeInlineComment]] | None = None,
    ) -> None:
        self._reviews = reviews or []
        self._inline_comments = inline_comments or []
        self._issue_comments = issue_comments or []
        self._review_comments_by_id = review_comments_by_id or {}
        self.created_reviews: list[dict[str, Any]] = []

    def get_reviews(self) -> list[FakeReview]:
        return self._reviews

    def get_review_comments(self) -> list[FakeInlineComment]:
        return self._inline_comments

    def get_issue_comments(self) -> list[FakeIssueComment]:
        return self._issue_comments

    def get_single_review_comments(self, review_id: int) -> list[FakeInlineComment]:
        return self._review_comments_by_id.get(review_id, [])

    def get_review(self, review_id: int) -> FakeReview:
        for r in self._reviews:
            if r.id == review_id:
                return r
        return FakeReview(review_id=review_id)

    def create_review(
        self,
        body: str = "",
        event: str = "",
        comments: list[Any] | None = None,
    ) -> FakeReview:
        review_id = 200 + len(self.created_reviews)
        self.created_reviews.append(
            {
                "body": body,
                "event": event,
                "comments": comments or [],
            }
        )
        return FakeReview(review_id=review_id)


class FakeRepo:
    def __init__(self, pr: FakePR | None = None) -> None:
        self._pr = pr or FakePR()

    def get_pull(self, number: int) -> FakePR:
        return self._pr


class FakeGithub:
    def __init__(self, repo: FakeRepo | None = None) -> None:
        self._repo = repo or FakeRepo()

    def get_repo(self, full_name: str) -> FakeRepo:
        return self._repo


def fake_ctx(
    gh: FakeGithub | None = None,
    owner: str = "owner",
    repo_name: str = "repo",
) -> GitHubCtx:
    """Build a ``GitHubCtx`` backed by fake objects for tests."""
    return GitHubCtx(
        gh=gh or FakeGithub(),  # type: ignore[arg-type]  # fake stub
        owner=owner,
        repo_name=repo_name,
    )
