"""Data models for rbtr."""

from datetime import datetime

from pydantic import BaseModel


class PRSummary(BaseModel):
    """Lightweight summary of a GitHub pull request, or a local branch."""

    number: int | None = None
    title: str
    author: str
    head_branch: str
    updated_at: datetime


class BranchSummary(BaseModel):
    """Lightweight summary of a git branch."""

    name: str
    last_commit_sha: str
    last_commit_message: str
    updated_at: datetime
