"""Data models for rbtr."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel

# ── Review targets ───────────────────────────────────────────────────
# A review target is either a GitHub PR or a local branch.  Each has
# the fields it actually needs — no optional sentinels.


class ReviewTarget(BaseModel):
    """Common fields for anything rbtr can review."""

    base_branch: str
    head_branch: str
    updated_at: datetime

    @property
    def head_ref(self) -> str:
        """Git ref to use for resolving the head commit.

        Subclasses may override to return a more precise ref
        (e.g. the exact SHA from the GitHub API).
        """
        return self.head_branch


class PRTarget(ReviewTarget):
    """A GitHub pull request selected for review."""

    number: int
    title: str
    author: str
    body: str = ""
    head_sha: str = ""
    """Commit SHA of the PR head (from the GitHub API).
    Used to resolve the head commit directly when the branch
    name isn't available as a local or remote ref."""

    @property
    def head_ref(self) -> str:
        """Prefer the exact SHA from the GitHub API.

        After ``fetch_pr_head`` has fetched ``refs/pull/<n>/head``,
        the SHA is resolvable in the local object store.
        """
        return self.head_sha if self.head_sha else self.head_branch


class BranchTarget(ReviewTarget):
    """A local branch selected for review (no PR)."""


# Union type used on Session.review_target.
type Target = PRTarget | BranchTarget


# ── Listing models ───────────────────────────────────────────────────


class PRSummary(BaseModel):
    """Lightweight summary of a GitHub pull request (for listing)."""

    number: int
    title: str
    author: str
    body: str = ""
    base_branch: str
    head_branch: str
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


class InlineComment(BaseModel):
    """A single inline comment on a specific line in the diff."""

    path: str
    """File path relative to repo root (e.g. ``src/api/handler.py``)."""

    line: int
    """Line number in the head version of the file (1-indexed)."""

    body: str
    """Markdown body of the comment.  Severity labels (blocker,
    suggestion, nit, question) are part of the body text — the
    LLM writes them naturally based on the review prompt."""

    suggestion: str = ""
    """Optional suggested replacement code (GitHub suggestion block).
    Empty string means no suggestion."""


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
    """Emoji → count mapping (e.g. ``{"+1": 3, "-1": 1}``)."""


class PendingReview(BaseModel):
    """A user's pending (unsubmitted) review on a PR.

    Returned by ``get_pending_review`` for sync purposes.
    """

    review_id: int
    """GitHub's ID for this review (needed for delete/replace)."""

    body: str = ""
    """Top-level review summary, if any."""

    comments: list[InlineComment]
    """Inline comments attached to this pending review."""


class ReviewDraft(BaseModel):
    """A complete review ready to post to GitHub.

    Contains a top-level summary and zero or more inline comments.
    The ``event`` field is set by the user at post time, not by
    the LLM.
    """

    summary: str = ""
    """Top-level review body (markdown)."""

    comments: list[InlineComment] = []
    """Inline comments attached to specific lines in the diff."""

    event: ReviewEvent = ReviewEvent.COMMENT
    """Review event type — set by the user before posting."""
