"""GitHub API operations for rbtr."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime

from github import Github, GithubException
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


def list_open_prs(gh: Github, owner: str, repo_name: str) -> list[PRSummary]:
    """List open pull requests, most recently updated first."""
    repo = gh.get_repo(f"{owner}/{repo_name}")
    pulls = repo.get_pulls(state="open", sort="updated", direction="desc")

    results: list[PRSummary] = []
    for pr in pulls:
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


def list_unmerged_branches(
    gh: Github, owner: str, repo_name: str, open_pr_branches: set[str]
) -> list[BranchSummary]:
    """List remote branches that have no open PR, excluding the default branch.

    Returns at most config.github.max_branches results, sorted by most recently updated first.
    """
    repo = gh.get_repo(f"{owner}/{repo_name}")
    default_branch = repo.default_branch

    results: list[BranchSummary] = []
    for branch in repo.get_branches():
        if len(results) >= config.github.max_branches:
            break
        if branch.name == default_branch:
            continue
        if branch.name in open_pr_branches:
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

    results.sort(key=lambda b: b.updated_at, reverse=True)
    return results


def _get_pull(gh: Github, owner: str, repo_name: str, pr_number: int) -> PullRequest:
    """Fetch a pull request or raise ``RbtrError``."""
    repo = gh.get_repo(f"{owner}/{repo_name}")
    try:
        return repo.get_pull(pr_number)
    except GithubException as err:
        raise RbtrError(f"PR #{pr_number} not found in {owner}/{repo_name}.") from err


def validate_pr_number(gh: Github, owner: str, repo_name: str, pr_number: int) -> PRSummary:
    """Fetch a specific PR by number. Raises RbtrError if not found."""
    pr = _get_pull(gh, owner, repo_name, pr_number)
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


def get_pending_review(
    gh: Github, owner: str, repo_name: str, pr_number: int, username: str
) -> PendingReview | None:
    """Return the user's PENDING review on a PR, or None if there isn't one.

    Fetches all reviews, finds the most recent one in PENDING state
    belonging to *username*, and returns it with its inline comments
    converted to ``InlineComment`` models.
    """
    pr = _get_pull(gh, owner, repo_name, pr_number)

    # Find the user's most recent PENDING review.
    pending = None
    for review in pr.get_reviews():
        if (
            review.state == "PENDING"
            and review.user is not None
            and review.user.login == username
        ):
            pending = review

    if pending is None:
        return None

    # Fetch inline comments for this specific review.
    comments: list[InlineComment] = []
    for comment in pr.get_single_review_comments(pending.id):
        comments.append(
            InlineComment(
                path=comment.path or "",
                line=comment.line or 0,
                body=comment.body or "",
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


def post_review(
    gh: Github,
    owner: str,
    repo_name: str,
    pr_number: int,
    draft: ReviewDraft,
) -> str:
    """Post a review to GitHub.  Returns the review's HTML URL.

    If ``draft.event`` is ``COMMENT``, ``APPROVE``, or
    ``REQUEST_CHANGES``, the review is submitted immediately.
    """
    pr = _get_pull(gh, owner, repo_name, pr_number)
    review = pr.create_review(
        body=draft.summary,
        event=draft.event.value,
        comments=_build_review_comments(draft),
    )
    return review.html_url


def push_pending_review(
    gh: Github,
    owner: str,
    repo_name: str,
    pr_number: int,
    draft: ReviewDraft,
) -> int:
    """Push a draft as a PENDING review (not submitted).

    Comments appear in the GitHub UI but the review stays in
    draft state.  Returns the review ID for future operations.
    """
    pr = _get_pull(gh, owner, repo_name, pr_number)
    review = pr.create_review(
        body=draft.summary,
        comments=_build_review_comments(draft),
    )
    return review.id


def delete_pending_review(
    gh: Github,
    owner: str,
    repo_name: str,
    pr_number: int,
    review_id: int,
) -> None:
    """Delete a pending (unsubmitted) review."""
    pr = _get_pull(gh, owner, repo_name, pr_number)
    review = pr.get_review(review_id)
    review.delete()


# ── PR discussion ────────────────────────────────────────────────────


def get_pr_discussion(
    gh: Github, owner: str, repo_name: str, pr_number: int
) -> list[DiscussionEntry]:
    """Fetch all discussion on a PR — reviews, inline comments, and issue comments.

    Returns a flat list sorted chronologically (oldest first).
    """
    pr = _get_pull(gh, owner, repo_name, pr_number)

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


# ── Formatting ───────────────────────────────────────────────────────


def format_discussion_entry(entry: DiscussionEntry) -> str:
    """Format a single discussion entry as text for LLM consumption.

    Produces a self-contained block with header (timestamp, author,
    kind-specific metadata), optional diff hunk, body, and reactions.
    """
    ts = entry.created_at.strftime("%Y-%m-%d %H:%M")
    bot_tag = " [bot]" if entry.is_bot else ""
    header_parts: list[str] = [f"[{ts}] @{entry.author}{bot_tag}"]

    match entry.kind:
        case DiscussionEntryKind.REVIEW:
            header_parts.append(f"({entry.review_state})")
        case DiscussionEntryKind.INLINE:
            location = entry.path
            if entry.line is not None:
                location += f":{entry.line}"
            header_parts.append(f"on {location}")
            if entry.in_reply_to_id is not None:
                header_parts.append("(reply)")
        case DiscussionEntryKind.COMMENT:
            pass

    header = " ".join(header_parts)
    parts: list[str] = [header]

    if entry.diff_hunk:
        parts.append(f"```\n{entry.diff_hunk}\n```")

    if entry.body:
        parts.append(entry.body)

    if entry.reactions:
        reaction_str = "  ".join(f"{emoji} {count}" for emoji, count in entry.reactions.items())
        parts.append(f"Reactions: {reaction_str}")

    return "\n".join(parts)
