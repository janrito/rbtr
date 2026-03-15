"""Data models for rbtr."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel

# ── Review targets ───────────────────────────────────────────────────
# A review target is either a GitHub PR, a local branch diff, or a
# single-commit snapshot.  Each has the fields it actually needs —
# no optional sentinels.


class ReviewTarget(BaseModel):
    """Common fields for anything rbtr can review."""

    base_branch: str
    """Human-readable branch name (for display)."""

    head_branch: str
    """Human-readable branch name (for display)."""

    base_commit: str
    """Git-resolvable ref for the base commit.

    For PRs this is the exact SHA from the GitHub API (immune
    to stale local branches).  For local branch reviews it
    equals `base_branch`.
    """

    head_commit: str
    """Git-resolvable ref for the head commit.

    For PRs this is the exact SHA from the GitHub API.
    For local branch reviews it equals `head_branch`.
    """

    updated_at: datetime


class PRTarget(ReviewTarget):
    """A GitHub pull request selected for review."""

    number: int
    title: str
    author: str
    body: str = ""
    head_sha: str = ""
    """Commit SHA of the PR head (from the GitHub API).

    Kept separately from `head_commit` because the GitHub
    review API requires the exact SHA as `commit_id` when
    posting reviews and inline comments.
    """


class BranchTarget(ReviewTarget):
    """A local branch diff selected for review (no PR)."""


class SnapshotTarget(BaseModel):
    """A single commit selected for exploration (no diff)."""

    head_commit: str
    """Git-resolvable ref for the snapshot commit."""

    ref_label: str
    """Human-readable label: branch name, tag, or short SHA."""

    updated_at: datetime


# Union type used on EngineState.review_target.
type Target = PRTarget | BranchTarget | SnapshotTarget


# ── Listing models ───────────────────────────────────────────────────


class PRSummary(BaseModel):
    """Lightweight summary of a GitHub pull request (for listing)."""

    number: int
    title: str
    author: str
    body: str = ""
    base_branch: str
    head_branch: str
    base_sha: str = ""
    head_sha: str = ""
    updated_at: datetime


class BranchSummary(BaseModel):
    """Lightweight summary of a git branch (for listing)."""

    name: str
    last_commit_sha: str
    last_commit_message: str
    updated_at: datetime


# ── Review draft ─────────────────────────────────────────────────────
# Structured representation of a review that can be posted to GitHub.
# The LLM produces these via tool calls; the user approves before posting.


class ReviewEvent(StrEnum):
    """GitHub pull request review event type.

    Chosen by the user, not the LLM.
    """

    COMMENT = "COMMENT"
    APPROVE = "APPROVE"
    REQUEST_CHANGES = "REQUEST_CHANGES"


class DiffSide(StrEnum):
    """Which side of a diff a comment targets.

    GitHub uses this with `line` to locate the comment.
    """

    RIGHT = "RIGHT"
    """Head (new) version of the file."""

    LEFT = "LEFT"
    """Base (old) version of the file."""


class InlineComment(BaseModel):
    """A single inline comment on a specific line in the diff."""

    path: str
    """File path relative to repo root (e.g. `src/api/handler.py`)."""

    line: int
    """Line number in the file (1-indexed).  For `RIGHT`
    this is the line in the head version; for `LEFT`
    the line in the base version."""

    side: DiffSide = DiffSide.RIGHT
    """Which side of the diff this comment targets."""

    commit_id: str = ""
    """Commit SHA that `line` was resolved against.  Set from
    `PRTarget.head_sha` for locally-created comments, from
    the GitHub response for remote-imported comments.  Empty
    means unknown (legacy drafts) — falls back to line-number
    validation at push time."""

    body: str
    """Markdown body of the comment.  Severity labels (blocker,
    suggestion, nit, question) are part of the body text — the
    LLM writes them naturally based on the review prompt."""

    suggestion: str = ""
    """Optional suggested replacement code (GitHub suggestion block).
    Empty string means no suggestion."""

    github_id: int | None = None
    """GitHub's comment ID — set when pulled from or pushed to GitHub.
    `None` for locally-created comments that haven't been synced."""

    comment_hash: str = ""
    """Content hash frozen at last sync — only updated by
    `stamp_synced()`, never by local edits.  Empty means
    never synced.  Comparing this against the live content
    hash detects local modifications."""


# ── PR discussion ────────────────────────────────────────────────────
# Unified representation of any comment on a pull request — review
# summaries, inline comments, and top-level issue comments.


class DiscussionEntryKind(StrEnum):
    """Type of discussion entry on a pull request."""

    REVIEW = "review"
    """Top-level review summary (APPROVED, CHANGES_REQUESTED, etc.)."""

    INLINE = "inline"
    """Inline comment on a specific file and line."""

    COMMENT = "comment"
    """General top-level comment on the PR (not attached to code)."""


class DiscussionEntry(BaseModel):
    """A single comment or review in a PR discussion.

    Fields that don't apply to a particular kind are left at their
    defaults (empty string, None, etc.).
    """

    kind: DiscussionEntryKind
    """What type of entry this is."""

    comment_id: int
    """GitHub's unique ID for this entry."""

    author: str
    """GitHub login of the commenter."""

    body: str
    """Markdown body of the comment."""

    created_at: datetime
    """When the comment was created."""

    is_bot: bool = False
    """Whether the author is a bot account."""

    # ── Inline-only fields ───────────────────────────────────────────

    path: str = ""
    """File path (inline comments only)."""

    line: int | None = None
    """Line number in the file (inline comments only)."""

    diff_hunk: str = ""
    """Diff context surrounding the comment (inline comments only)."""

    in_reply_to_id: int | None = None
    """Parent comment ID for threaded replies (inline comments only)."""

    # ── Review-only fields ───────────────────────────────────────────

    review_state: str = ""
    """Review verdict: APPROVED, CHANGES_REQUESTED, COMMENTED,
    DISMISSED, or PENDING (review entries only)."""

    # ── Reactions ────────────────────────────────────────────────────

    reactions: dict[str, int] = {}
    """Emoji → count mapping (e.g. `{"+1": 3, "-1": 1}`)."""


class ReviewDraft(BaseModel):
    """A complete review ready to post to GitHub.

    Contains a top-level summary and zero or more inline comments.
    The `event` field is set by the user at post time, not by
    the LLM.

    Sync tracking is flat: `github_review_id` and `summary_hash`
    at the draft level, `comment_hash` on each comment.  No nested
    sync section — the TOML stays simple and hand-editable.
    """

    summary: str = ""
    """Top-level review body (markdown)."""

    comments: list[InlineComment] = []
    """Inline comments attached to specific lines in the diff."""

    github_review_id: int | None = None
    """The PENDING review ID on GitHub.  `None` = never pushed."""

    summary_hash: str = ""
    """Hash of the review summary frozen at last sync — only
    updated by `stamp_synced()`, never by local edits.
    Empty means never synced."""
