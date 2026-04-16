"""CLI-specific output schemas.

Most CLI output uses the index models directly (`Chunk`, `Edge`,
`ScoredResult`, `IndexStats`). This module defines only the models
that are genuinely CLI-specific — composites or views that don't
exist in the index layer.
"""

from __future__ import annotations

from pydantic import BaseModel

from rbtr.index.models import IndexStats


class BuildIndexResult(BaseModel):
    """Output of `rbtr index`."""

    refs: list[str]
    stats: IndexStats
    errors: list[str]


class IndexStatus(BaseModel):
    """Output of `rbtr status`."""

    exists: bool
    db_path: str | None = None
    total_chunks: int | None = None
