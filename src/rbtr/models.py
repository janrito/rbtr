"""Data models for rbtr."""

from datetime import datetime

from pydantic import BaseModel

# ── Review targets ───────────────────────────────────────────────────
# A review target is either a GitHub PR or a local branch.  Each has
# the fields it actually needs — no optional sentinels.


class ReviewTarget(BaseModel):
    """Common fields for anything rbtr can review."""

    head_branch: str
    updated_at: datetime


class PRTarget(ReviewTarget):
    """A GitHub pull request selected for review."""

    number: int
    title: str
    author: str


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
    head_branch: str
    updated_at: datetime


class BranchSummary(BaseModel):
    """Lightweight summary of a git branch (for listing)."""

    name: str
    last_commit_sha: str
    last_commit_message: str
    updated_at: datetime
