"""Output schemas for the rbtr CLI.

These models are the strict interface between the index layer
and the CLI. All output flows through them — in both JSON and
TTY mode. No ad-hoc dicts, no data outside the schema.
"""

from __future__ import annotations

from pydantic import BaseModel


class BuildResult(BaseModel):
    """Output of `rbtr build`."""

    ref: str
    total_files: int
    parsed_files: int
    skipped_files: int
    total_chunks: int
    total_edges: int
    elapsed_seconds: float
    errors: list[str]


class SearchHit(BaseModel):
    """One result from `rbtr search`."""

    id: str
    file_path: str
    name: str
    kind: str
    score: float
    line_start: int
    line_end: int
    content: str


class SymbolInfo(BaseModel):
    """A symbol with its full source (`rbtr read-symbol`)."""

    file_path: str
    name: str
    kind: str
    line_start: int
    line_end: int
    content: str


class SymbolSummary(BaseModel):
    """A symbol without content (`rbtr list-symbols`, `rbtr changed-symbols`)."""

    name: str
    kind: str
    line_start: int
    line_end: int
    file_path: str | None = None


class EdgeInfo(BaseModel):
    """A dependency edge (`rbtr find-refs`)."""

    source_id: str
    target_id: str
    kind: str


class IndexStatus(BaseModel):
    """Output of `rbtr status`."""

    exists: bool
    db_path: str | None = None
    total_chunks: int | None = None
