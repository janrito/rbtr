"""GitHub API operations for rbtr."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pygit2
from github import Github, UnknownObjectException
from github.GithubObject import GithubObject
from github.IssueComment import IssueComment
from github.PullRequest import PullRequest, ReviewComment
from github.PullRequestComment import PullRequestComment
from github.PullRequestReview import PullRequestReview
from github.Reaction import Reaction

from rbtr.config import config
from rbtr.exceptions import RbtrError
from rbtr.models import (
    BranchSummary,
    DiscussionEntry,
    DiscussionEntryKind,
    InlineComment,
    PendingReview,
    PRSummary,
    ReviewDraft,
)

# ── Remote URL parsing ───────────────────────────────────────────────

_GITHUB_PATTERNS = [
    # SSH: git@github.com:owner/repo.git
    re.compile(r"git@github\.com:(?P<owner>[^/]+)/(?P<repo>[^/.]+?)(?:\.git)?$"),
    # HTTPS: https://github.com/owner/repo.git
    re.compile(r"https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/.]+?)(?:\.git)?$"),
]


def _parse_github_url(url: str) -> tuple[str, str] | None:
    for pattern in _GITHUB_PATTERNS:
        m = pattern.match(url)
        if m:
            return m.group("owner"), m.group("repo")
    return None


def parse_github_remote(repo: pygit2.Repository) -> tuple[str, str]:
    """Extract (owner, repo_name) from the GitHub remote URL.

    Tries 'origin' first, then falls back to the first remote that looks like GitHub.
    """
    remotes: list[pygit2.Remote] = list(repo.remotes)

    # Try origin first
    for remote in remotes:
        if remote.name == "origin" and remote.url is not None:
            result = _parse_github_url(remote.url)
            if result is not None:
                return result

    # Fall back to any GitHub remote
    for remote in remotes:
        if remote.url is not None:
            result = _parse_github_url(remote.url)
            if result is not None:
                return result

    raise RbtrError("No GitHub remote found. rbtr requires a repository with a GitHub remote.")


# ── Context ───────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class GitHubCtx:
    """Lightweight bundle for the three values every API call needs."""

    gh: Github
    owner: str
    repo_name: str

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.repo_name}"


def _pull(ctx: GitHubCtx, pr_number: int) -> PullRequest:
    """Fetch a pull request or raise ``RbtrError``."""
    try:
        repo = ctx.gh.get_repo(ctx.full_name)
        return repo.get_pull(pr_number)
    except UnknownObjectException as err:
        raise RbtrError(f"PR #{pr_number} not found in {ctx.full_name}.") from err


# ── PR / branch listing ──────────────────────────────────────────────


def list_open_prs(ctx: GitHubCtx) -> list[PRSummary]:
    """List open pull requests, most recently updated first.

    Fetches only the first page (up to 30 PRs) to avoid
    multiple paginated API calls on large repositories.
    """
    repo = ctx.gh.get_repo(ctx.full_name)
    pulls = repo.get_pulls(state="open", sort="updated", direction="desc")

    results: list[PRSummary] = []
    for pr in pulls.get_page(0):
        results.append(
            PRSummary(
                number=pr.number,
                title=pr.title,
                author=pr.user.login if pr.user else "unknown",
                body=pr.body or "",
                base_branch=pr.base.ref,
                head_branch=pr.head.ref,
                head_sha=pr.head.sha or "",
                updated_at=pr.updated_at or datetime.now(tz=UTC),
            )
        )
    return results


def list_unmerged_branches(ctx: GitHubCtx, open_pr_branches: set[str]) -> list[BranchSummary]:
    """List remote branches that have no open PR, excluding the default branch.

    Returns at most ``config.github.max_branches`` results, sorted by
    most recently updated first.  Fetches branch pages incrementally
    and stops as soon as the limit is reached to avoid exhausting the
    full branch list on large repositories.
    """
    repo = ctx.gh.get_repo(ctx.full_name)
    default = repo.default_branch
    limit = config.github.max_branches
    skip = {default} | open_pr_branches

    branches = repo.get_branches()
    results: list[BranchSummary] = []
    page = 0
    while len(results) < limit:
        batch = branches.get_page(page)
        if not batch:
            break
        for branch in batch:
            if branch.name in skip:
                continue
            commit = branch.commit
            results.append(
                BranchSummary(
                    name=branch.name,
                    last_commit_sha=commit.sha,
                    last_commit_message=(
                        commit.commit.message.split("\n", 1)[0] if commit.commit else ""
                    ),
                    updated_at=commit.commit.committer.date
                    if commit.commit and commit.commit.committer
                    else datetime.now(tz=UTC),
                )
            )
            if len(results) >= limit:
                break
        page += 1

    results.sort(key=lambda b: b.updated_at, reverse=True)
    return results


def validate_pr_number(ctx: GitHubCtx, pr_number: int) -> PRSummary:
    """Fetch a specific PR by number. Raises RbtrError if not found."""
    pr = _pull(ctx, pr_number)
    return PRSummary(
        number=pr.number,
        title=pr.title,
        author=pr.user.login if pr.user else "unknown",
        body=pr.body or "",
        base_branch=pr.base.ref,
        head_branch=pr.head.ref,
        head_sha=pr.head.sha or "",
        updated_at=pr.updated_at or datetime.now(tz=UTC),
    )


# ── Pending review ────────────────────────────────────────────────────


def get_pending_review(ctx: GitHubCtx, pr_number: int, username: str) -> PendingReview | None:
    """Return the user's PENDING review on a PR, or None if there isn't one.

    Fetches all reviews, finds the most recent one in PENDING state
    belonging to *username*, and returns it with its inline comments
    converted to ``InlineComment`` models.
    """
    pr = _pull(ctx, pr_number)

    # Find the user's most recent PENDING review.
    pending = None
    for review in pr.get_reviews():
        if review.state == "PENDING" and review.user is not None and review.user.login == username:
            pending = review

    if pending is None:
        return None

    # Fetch inline comments for this specific review.
    #
    # Read from the list-response data directly via the base-class
    # raw_data getter.  PullRequestComment properties trigger lazy-load
    # completion against GET /repos/{owner}/{repo}/pulls/comments/{id},
    # but pending review comments are NOT accessible at that endpoint
    # (404).  The data we need is already present from the list call.
    comments: list[InlineComment] = []
    raw_data = GithubObject.raw_data.fget  # type: ignore[attr-defined]  # @property always has fget
    for comment in pr.get_single_review_comments(pending.id):
        data: dict[str, Any] = raw_data(comment)
        comments.append(
            InlineComment(
                path=data.get("path") or "",
                line=data.get("line") or 0,
                body=data.get("body") or "",
            )
        )

    return PendingReview(
        review_id=pending.id,
        body=pending.body or "",
        comments=comments,
    )


# ── Post / delete reviews ────────────────────────────────────────────


def format_comment_body(comment: InlineComment) -> str:
    """Format an InlineComment body for GitHub, including suggestion blocks."""
    body = comment.body
    if comment.suggestion:
        body += f"\n\n```suggestion\n{comment.suggestion}\n```"
    return body


def _build_review_comments(draft: ReviewDraft) -> list[ReviewComment]:
    """Build the list of ReviewComment dicts for a GitHub API call."""
    return [
        {
            "path": c.path,
            "body": format_comment_body(c),
            "line": c.line,
            "side": "RIGHT",
        }
        for c in draft.comments
    ]


def post_review(ctx: GitHubCtx, pr_number: int, draft: ReviewDraft) -> str:
    """Post a review to GitHub.  Returns the review's HTML URL.

    If ``draft.event`` is ``COMMENT``, ``APPROVE``, or
    ``REQUEST_CHANGES``, the review is submitted immediately.
    """
    pr = _pull(ctx, pr_number)
    review = pr.create_review(
        body=draft.summary,
        event=draft.event.value,
        comments=_build_review_comments(draft),
    )
    return review.html_url


def push_pending_review(ctx: GitHubCtx, pr_number: int, draft: ReviewDraft) -> int:
    """Push a draft as a PENDING review (not submitted).

    Comments appear in the GitHub UI but the review stays in
    draft state.  Returns the review ID for future operations.
    """
    pr = _pull(ctx, pr_number)
    review = pr.create_review(
        body=draft.summary,
        comments=_build_review_comments(draft),
    )
    return review.id


def delete_pending_review(ctx: GitHubCtx, pr_number: int, review_id: int) -> None:
    """Delete a pending (unsubmitted) review."""
    pr = _pull(ctx, pr_number)
    review = pr.get_review(review_id)
    review.delete()


# ── PR discussion ────────────────────────────────────────────────────


def get_pr_discussion(ctx: GitHubCtx, pr_number: int) -> list[DiscussionEntry]:
    """Fetch all discussion on a PR — reviews, inline comments, and issue comments.

    Returns a flat list sorted chronologically (oldest first).
    """
    pr = _pull(ctx, pr_number)

    entries: list[DiscussionEntry] = []

    # Top-level review summaries.
    for review in pr.get_reviews():
        # Skip reviews with empty bodies (approval clicks, etc.).
        if not review.body:
            continue
        entries.append(_review_to_entry(review))

    # Inline review comments.
    for comment in pr.get_review_comments():
        entries.append(_inline_to_entry(comment))

    # General PR comments (issue-style, not attached to code).
    for issue_comment in pr.get_issue_comments():
        entries.append(_issue_comment_to_entry(issue_comment))

    entries.sort(key=lambda e: e.created_at)
    return entries


def _aggregate_reactions(reactions: list[Reaction]) -> dict[str, int]:
    """Aggregate a list of Reaction objects into emoji → count."""
    return dict(Counter(r.content for r in reactions))


def _review_to_entry(review: PullRequestReview) -> DiscussionEntry:
    """Convert a PyGithub PullRequestReview to a DiscussionEntry."""
    return DiscussionEntry(
        kind=DiscussionEntryKind.REVIEW,
        comment_id=review.id,
        author=review.user.login if review.user else "unknown",
        body=review.body or "",
        created_at=review.submitted_at or datetime.now(tz=UTC),
        is_bot=review.user.type == "Bot" if review.user else False,
        review_state=review.state or "",
    )


def _inline_to_entry(comment: PullRequestComment) -> DiscussionEntry:
    """Convert a PyGithub PullRequestComment to a DiscussionEntry."""
    reactions = _aggregate_reactions(list(comment.get_reactions()))
    return DiscussionEntry(
        kind=DiscussionEntryKind.INLINE,
        comment_id=comment.id,
        author=comment.user.login if comment.user else "unknown",
        body=comment.body or "",
        created_at=comment.created_at or datetime.now(tz=UTC),
        is_bot=comment.user.type == "Bot" if comment.user else False,
        path=comment.path or "",
        line=comment.line,
        diff_hunk=comment.diff_hunk or "",
        in_reply_to_id=comment.in_reply_to_id,
        reactions=reactions,
    )


def _issue_comment_to_entry(comment: IssueComment) -> DiscussionEntry:
    """Convert a PyGithub IssueComment to a DiscussionEntry."""
    reactions = _aggregate_reactions(list(comment.get_reactions()))
    return DiscussionEntry(
        kind=DiscussionEntryKind.COMMENT,
        comment_id=comment.id,
        author=comment.user.login if comment.user else "unknown",
        body=comment.body or "",
        created_at=comment.created_at or datetime.now(tz=UTC),
        is_bot=comment.user.type == "Bot" if comment.user else False,
        reactions=reactions,
    )
