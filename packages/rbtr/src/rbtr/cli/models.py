"""CLI-specific output schemas.

Most CLI output uses the index models directly (`Chunk`, `Edge`,
`IndexStats`). This module defines only the models that are
genuinely CLI-specific — composites or views that don't exist in
the index layer.
"""

from __future__ import annotations

from pydantic import BaseModel

from rbtr.index.models import Chunk, IndexStats


class BuildResult(BaseModel):
    """Output of `rbtr build`."""

    ref: str
    stats: IndexStats
    errors: list[str]


class SearchHit(BaseModel):
    """Flattened search result (ScoredResult is a dataclass, not serialisable)."""

    score: float
    lexical: float
    semantic: float
    name_score: float
    kind_boost: float
    file_penalty: float
    importance: float
    proximity: float
    chunk: Chunk


class IndexStatus(BaseModel):
    """Output of `rbtr status`."""

    exists: bool
    db_path: str | None = None
    total_chunks: int | None = None
