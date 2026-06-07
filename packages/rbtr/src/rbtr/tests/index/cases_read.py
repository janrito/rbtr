"""Cases for read-side store behaviours.

Scenarios for `get_chunks`, `get_edges`, `has_blob`,
upsert, and multi-repo isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pytest_cases import case

from rbtr.index.models import ChunkKind, Snapshot, TokenisedChunk

from .conftest import make_chunk, make_snap

# ── Scenario dataclasses ────────────────────────────────────────────


@dataclass(frozen=True)
class ChunkQueryScenario:
    """Seed data + filter args + expected chunk IDs."""

    chunks: list[TokenisedChunk]
    snapshots: list[Snapshot]
    commit_sha: str = "head"
    file_path: str | None = None
    kind: ChunkKind | None = None
    name: str | None = None
    expected_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class HasBlobScenario:
    """Seed data + has_blob query + expected result."""

    chunks: list[TokenisedChunk]
    snapshots: list[Snapshot]
    query_blob: str
    query_language: str
    query_language_plugin_version: int
    expected: bool


# ── get_chunks filter cases ─────────────────────────────────────────


@case(tags=["get_chunks"])
def case_get_chunks_unfiltered() -> ChunkQueryScenario:
    """All chunks visible at the commit."""
    c1 = make_chunk("fn1", path="a.py")
    c2 = make_chunk("fn2", path="b.py", kind=ChunkKind.CLASS)
    return ChunkQueryScenario(
        chunks=[c1, c2],
        snapshots=[make_snap("head", "a.py", c1.blob_sha), make_snap("head", "b.py", c2.blob_sha)],
        expected_ids=["fn1", "fn2"],
    )


@case(tags=["get_chunks"])
def case_get_chunks_by_file_path() -> ChunkQueryScenario:
    """Filter by file_path returns only chunks in that file."""
    c1 = make_chunk("fn1", path="a.py")
    c2 = make_chunk("fn2", path="b.py")
    return ChunkQueryScenario(
        chunks=[c1, c2],
        snapshots=[make_snap("head", "a.py", c1.blob_sha), make_snap("head", "b.py", c2.blob_sha)],
        file_path="a.py",
        expected_ids=["fn1"],
    )


@case(tags=["get_chunks"])
def case_get_chunks_by_kind() -> ChunkQueryScenario:
    """Filter by kind returns only chunks of that kind."""
    c1 = make_chunk("fn1", kind=ChunkKind.FUNCTION)
    c2 = make_chunk("cls1", kind=ChunkKind.CLASS, path="b.py")
    return ChunkQueryScenario(
        chunks=[c1, c2],
        snapshots=[make_snap("head", "f.py", c1.blob_sha), make_snap("head", "b.py", c2.blob_sha)],
        kind=ChunkKind.CLASS,
        expected_ids=["cls1"],
    )


@case(tags=["get_chunks"])
def case_get_chunks_by_name() -> ChunkQueryScenario:
    """Filter by name returns only matching chunk."""
    c1 = make_chunk("fn1", name="helper")
    c2 = make_chunk("fn2", name="runner", path="b.py")
    return ChunkQueryScenario(
        chunks=[c1, c2],
        snapshots=[make_snap("head", "f.py", c1.blob_sha), make_snap("head", "b.py", c2.blob_sha)],
        name="helper",
        expected_ids=["fn1"],
    )


@case(tags=["get_chunks"])
def case_get_chunks_nonexistent_file() -> ChunkQueryScenario:
    """Filter by nonexistent file returns empty."""
    c1 = make_chunk("fn1")
    return ChunkQueryScenario(
        chunks=[c1],
        snapshots=[make_snap("head", "f.py", c1.blob_sha)],
        file_path="nope.py",
        expected_ids=[],
    )


# ── has_blob cases ──────────────────────────────────────────────────


@case(tags=["has_blob"])
def case_has_blob_same_language() -> HasBlobScenario:
    """Blob exists with matching language."""
    c = make_chunk("fn1")
    return HasBlobScenario(
        chunks=[c],
        snapshots=[make_snap("head", "f.py", c.blob_sha)],
        query_blob=c.blob_sha,
        query_language="",
        query_language_plugin_version=1,
        expected=True,
    )


@case(tags=["has_blob"])
def case_has_blob_different_language() -> HasBlobScenario:
    """Blob exists but with different language → False."""
    c = make_chunk("fn1")
    return HasBlobScenario(
        chunks=[c],
        snapshots=[make_snap("head", "f.py", c.blob_sha)],
        query_blob=c.blob_sha,
        query_language="swift",
        query_language_plugin_version=1,
        expected=False,
    )


@case(tags=["has_blob"])
def case_has_blob_same_language_and_version() -> HasBlobScenario:
    """Blob stored as 'markdown' v1, queried with same → True."""
    c = make_chunk("doc1", kind=ChunkKind.DOC_SECTION)
    c = TokenisedChunk(**{**c.model_dump(), "language": "markdown"})
    return HasBlobScenario(
        chunks=[c],
        snapshots=[make_snap("head", "f.py", c.blob_sha)],
        query_blob=c.blob_sha,
        query_language="markdown",
        query_language_plugin_version=1,
        expected=True,
    )


@case(tags=["has_blob"])
def case_has_blob_different_version() -> HasBlobScenario:
    """Blob stored at version 1, queried with version 2 → False."""
    c = make_chunk("doc1", kind=ChunkKind.DOC_SECTION)
    c = TokenisedChunk(**{**c.model_dump(), "language": "markdown"})
    return HasBlobScenario(
        chunks=[c],
        snapshots=[make_snap("head", "f.py", c.blob_sha)],
        query_blob=c.blob_sha,
        query_language="markdown",
        query_language_plugin_version=2,
        expected=False,
    )


@case(tags=["has_blob"])
def case_has_blob_nonexistent() -> HasBlobScenario:
    """Blob doesn't exist at all → False."""
    c = make_chunk("fn1")
    return HasBlobScenario(
        chunks=[c],
        snapshots=[make_snap("head", "f.py", c.blob_sha)],
        query_blob="no_such_blob",
        query_language="",
        query_language_plugin_version=1,
        expected=False,
    )
