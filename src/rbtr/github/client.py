"""GitHub API operations for rbtr."""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pygit2
from github import Github, UnknownObjectException
from github.GithubObject import GithubObject
from github.IssueComment import IssueComment
from github.PullRequest import PullRequest
from github.PullRequestComment import PullRequestComment
from github.PullRequestReview import PullRequestReview
from github.Reaction import Reaction

from rbtr.config import config
from rbtr.exceptions import RbtrError
from rbtr.models import (
    BranchSummary,
    DiffSide,
    DiscussionEntry,
    DiscussionEntryKind,
    InlineComment,
    PRSummary,
    ReviewDraft,
    ReviewEvent,
)

log = logging.getLogger(__name__)

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
#
# GITHUB API WORKAROUND — position ↔ line conversion
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# As of 2026-02, the per-review comments endpoint
# ``GET /repos/{owner}/{repo}/pulls/{pr}/reviews/{id}/comments``
# does NOT return the modern ``line``, ``original_line``, or ``side``
# fields.  These are always ``null`` — even for submitted reviews,
# even with ``X-GitHub-Api-Version: 2022-11-28``.  The per-PR
# endpoint (``GET /pulls/{pr}/comments``) does return them, but
# excludes pending review comments entirely.
#
# The only line-related data available for pending comments is the
# deprecated ``position`` (1-based offset into the diff hunk) and
# ``diff_hunk`` (hunk header through the commented line).
#
# ``_walk_hunk`` / ``_position_to_line`` / ``_line_to_position``
# exist solely to work around this gap.  If GitHub updates the
# per-review endpoint to return ``line`` and ``side``, this
# machinery can be removed and ``get_pending_review`` can read
# those fields directly (the ``data.get("line")`` path already
# takes priority — see the read logic below).
#
# Verified 2026-02-27 with direct httpx calls against both
# endpoints using ``Accept: application/vnd.github+json`` and
# ``X-GitHub-Api-Version: 2022-11-28``.  Both return
# ``line: null, side: null`` for review comments fetched via
# the per-review endpoint.
#
# Sync hashing
# ~~~~~~~~~~~~
# ``_comment_hash`` (in ``draft.py``) covers ``path``, ``line``,
# ``body``, and ``suggestion``.  ``side`` and ``commit_id`` are
# excluded (resolution metadata not visible in the GitHub UI).
# ``line`` IS included because the ``position`` ↔ ``line``
# conversion is deterministic — ``_walk_hunk`` is the single
# source of truth for both directions.

_HUNK_RE = re.compile(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def _walk_hunk(diff_hunk: str) -> Iterator[tuple[int, int, DiffSide]]:
    """Yield ``(position, line, side)`` for each diff line in *diff_hunk*.

    Skips ``\\ No newline at end of file`` markers.  ``position`` is
    1-based.  This is the single source of truth for mapping between
    GitHub's deprecated ``position`` and ``(line, side)``.
    """
    lines = diff_hunk.split("\n")

    # Find the last @@ header (multi-hunk hunks are possible).
    header_idx = -1
    for i, ln in enumerate(lines):
        if ln.startswith("@@"):
            header_idx = i
    if header_idx == -1:
        return

    m = _HUNK_RE.match(lines[header_idx])
    if not m:
        return

    old_line = int(m.group(1))
    new_line = int(m.group(2))

    pos = 0
    for ln in lines[header_idx + 1 :]:
        if ln.startswith("\\"):
            continue
        pos += 1
        if ln.startswith("+"):
            yield pos, new_line, DiffSide.RIGHT
            new_line += 1
        elif ln.startswith("-"):
            yield pos, old_line, DiffSide.LEFT
            old_line += 1
        else:
            # Context line (starts with " " or is empty).
            yield pos, new_line, DiffSide.RIGHT
            old_line += 1
            new_line += 1


def _position_to_line(diff_hunk: str) -> tuple[int, DiffSide]:
    """Recover ``(line, side)`` from a GitHub ``diff_hunk``.

    Pending review comments lack the ``line`` / ``side`` fields and
    only provide the deprecated ``position`` plus the ``diff_hunk``
    that runs from the hunk header through the commented line.  The
    last diff line determines the file line number and diff side.
    """
    result_line = 0
    result_side: DiffSide = DiffSide.RIGHT
    for _pos, line, side in _walk_hunk(diff_hunk):
        result_line = line
        result_side = side
    return result_line, result_side


def _line_to_position(diff_hunk: str, line: int, side: DiffSide) -> int | None:
    """Convert ``(line, side)`` to a diff position within *diff_hunk*.

    Returns the 1-based position, or ``None`` if the line/side pair
    is not present in the hunk.  This is the inverse of
    ``_position_to_line``.
    """
    for pos, hunk_line, hunk_side in _walk_hunk(diff_hunk):
        if hunk_line == line and hunk_side == side:
            return pos
    return None


def get_pending_review(ctx: GitHubCtx, pr_number: int, username: str) -> ReviewDraft | None:
    """Return the user's PENDING review on a PR, or ``None``.

    Fetches all reviews, finds the most recent one in PENDING state
    belonging to *username*, and returns it as a ``ReviewDraft`` with
    ``github_review_id`` set and hashes empty (not yet synced locally).
    """
    pr = _pull(ctx, pr_number)

    # Find the user's most recent PENDING review.
    pending = None
    reviews_seen = 0
    for review in pr.get_reviews():
        reviews_seen += 1
        user_login = review.user.login if review.user else "<none>"
        log.info(
            "PR #%d review id=%d state=%s user=%s",
            pr_number,
            review.id,
            review.state,
            user_login,
        )
        if review.state == "PENDING" and review.user is not None and review.user.login == username:
            pending = review

    if pending is None:
        log.info(
            "No PENDING review for '%s' on PR #%d (%d reviews seen)",
            username,
            pr_number,
            reviews_seen,
        )
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
        body_raw = data.get("body") or ""
        body, suggestion = parse_comment_body(body_raw)

        # Submitted reviews return ``line`` / ``original_line`` /
        # ``side``.  Pending reviews omit those and only provide the
        # deprecated ``position`` + ``diff_hunk``.  Recover the real
        # file line number (and side) from the hunk when necessary.
        line = data.get("line") or data.get("original_line") or 0
        side_raw = data.get("side") or ""
        side: DiffSide
        if not line and data.get("diff_hunk"):
            line, side = _position_to_line(data["diff_hunk"])
        elif side_raw in ("LEFT", "RIGHT"):
            side = DiffSide(side_raw)
        else:
            side = DiffSide.RIGHT

        comments.append(
            InlineComment(
                path=data.get("path") or "",
                line=line,
                side=side,
                commit_id=data.get("commit_id") or "",
                body=body,
                suggestion=suggestion,
                github_id=data.get("id"),
            )
        )

    return ReviewDraft(
        summary=pending.body or "",
        comments=comments,
        github_review_id=pending.id,
    )


# ── Post / delete reviews ────────────────────────────────────────────


_SUGGESTION_FENCE = "```suggestion"


def parse_comment_body(raw: str) -> tuple[str, str]:
    """Split a GitHub comment body into ``(body, suggestion)``.

    GitHub stores suggestion blocks inline::

        Review text.

        ```suggestion
        replacement_code()
        ```

    Returns ``(body_text, suggestion_code)`` with the fence removed.
    If there is no suggestion block, *suggestion* is ``""``.
    """
    idx = raw.find(_SUGGESTION_FENCE)
    if idx == -1:
        return raw, ""
    body = raw[:idx].rstrip()
    rest = raw[idx + len(_SUGGESTION_FENCE) :]
    # Strip optional newline after the fence opening.
    if rest.startswith("\n"):
        rest = rest[1:]
    # Find closing fence.
    end = rest.find("```")
    if end == -1:
        # Malformed — treat everything after as suggestion.
        return body, rest.rstrip()
    suggestion = rest[:end].rstrip("\n")
    return body, suggestion


def format_comment_body(comment: InlineComment) -> str:
    """Format an InlineComment body for GitHub, including suggestion blocks."""
    body = comment.body
    if comment.suggestion:
        body += f"\n\n```suggestion\n{comment.suggestion}\n```"
    return body


def _build_review_comments(draft: ReviewDraft) -> list[dict[str, Any]]:
    """Build the list of comment dicts for a GitHub API call.

    File-level comments (``line == 0``) omit ``line`` and ``side``
    so GitHub creates them as file-scoped rather than rejecting them.
    """
    result: list[dict[str, Any]] = []
    for c in draft.comments:
        entry: dict[str, Any] = {
            "path": c.path,
            "body": format_comment_body(c),
        }
        if c.line > 0:
            entry["line"] = c.line
            entry["side"] = c.side
        result.append(entry)
    return result


def _resolve_commit(ctx: GitHubCtx, commit_id: str) -> Any:
    """Fetch a ``Commit`` object by SHA for ``create_review``.

    PyGithub's ``create_review`` asserts ``isinstance(commit,
    Commit)`` so we cannot pass a lightweight stub.
    """
    repo = ctx.gh.get_repo(ctx.full_name)
    return repo.get_commit(commit_id)


def post_review(
    ctx: GitHubCtx,
    pr_number: int,
    draft: ReviewDraft,
    event: ReviewEvent,
    commit_id: str = "",
) -> str:
    """Post a review to GitHub.  Returns the review's HTML URL.

    When *commit_id* is provided the review is pinned to that
    commit, ensuring line numbers match the diff GitHub computes.
    """
    pr = _pull(ctx, pr_number)
    kwargs: dict[str, Any] = {
        "body": draft.summary,
        "event": event.value,
        "comments": _build_review_comments(draft),
    }
    if commit_id:
        kwargs["commit"] = _resolve_commit(ctx, commit_id)
    review = pr.create_review(**kwargs)
    return review.html_url


def push_pending_review(
    ctx: GitHubCtx,
    pr_number: int,
    draft: ReviewDraft,
    commit_id: str = "",
) -> int:
    """Push a draft as a PENDING review (not submitted).

    Comments appear in the GitHub UI but the review stays in
    draft state.  Returns the review ID for future operations.
    """
    pr = _pull(ctx, pr_number)
    kwargs: dict[str, Any] = {
        "body": draft.summary,
        "comments": _build_review_comments(draft),
    }
    if commit_id:
        kwargs["commit"] = _resolve_commit(ctx, commit_id)
    review = pr.create_review(**kwargs)
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
