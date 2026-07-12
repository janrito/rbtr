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
    """Seed data + has_blob(serial map) query + expected result."""

    chunks: list[TokenisedChunk]
    snapshots: list[Snapshot]
    query_blob: str
    query_language: str
    serial_map: dict[str, int]
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


@case(tags=["get_chunks"])
def case_get_chunks_same_blob_two_paths() -> ChunkQueryScenario:
    """Identical content at two paths does not cross-contaminate.

    Both chunks share one `blob_sha` but sit at different paths, so the
    `(blob_sha, file_path)` join must return only the queried path's
    chunk — never the same-blob chunk at the other path.
    """
    at_a = make_chunk("at_a", path="a.py", blob="b")
    at_b = make_chunk("at_b", path="b.py", blob="b")
    return ChunkQueryScenario(
        chunks=[at_a, at_b],
        snapshots=[make_snap("head", "a.py", "b"), make_snap("head", "b.py", "b")],
        file_path="a.py",
        expected_ids=["at_a"],
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
        serial_map={"": 1},
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
        serial_map={"swift": 1},
        expected=False,
    )


@case(tags=["has_blob"])
def case_has_blob_same_language_and_serial() -> HasBlobScenario:
    """Blob stored as 'markdown' v1, queried with same → True."""
    c = make_chunk("doc1", kind=ChunkKind.DOC_SECTION)
    c = c.model_copy(update={"language": "markdown"})
    return HasBlobScenario(
        chunks=[c],
        snapshots=[make_snap("head", "f.py", c.blob_sha)],
        query_blob=c.blob_sha,
        query_language="markdown",
        serial_map={"markdown": 1},
        expected=True,
    )


@case(tags=["has_blob"])
def case_has_blob_different_serial() -> HasBlobScenario:
    """Blob stored at serial 1, queried with serial 2 → False."""
    c = make_chunk("doc1", kind=ChunkKind.DOC_SECTION)
    c = c.model_copy(update={"language": "markdown"})
    return HasBlobScenario(
        chunks=[c],
        snapshots=[make_snap("head", "f.py", c.blob_sha)],
        query_blob=c.blob_sha,
        query_language="markdown",
        serial_map={"markdown": 2},
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
        serial_map={"": 1},
        expected=False,
    )


@case(tags=["has_blob"])
def case_has_blob_detected_language_changed() -> HasBlobScenario:
    """Plaintext blob, now detected as python (plugin registered) → False.

    The stored chunk is at a current serial, but there is no chunk in the
    newly detected language, so the file must re-extract.
    """
    c = make_chunk("raw1")  # language "", serial 1
    return HasBlobScenario(
        chunks=[c],
        snapshots=[make_snap("head", "mod.py", c.blob_sha)],
        query_blob=c.blob_sha,
        query_language="python",
        serial_map={"python": 4, "": 1},
        expected=False,
    )


@case(tags=["has_blob"])
def case_has_blob_multilanguage_all_current() -> HasBlobScenario:
    """SFC blob: host + embedded chunks all at current serials → True."""
    host = make_chunk("tpl", blob="sfc").model_copy(
        update={"language": "svelte", "extraction_serial": 2}
    )
    ts = make_chunk("fn", blob="sfc").model_copy(
        update={"language": "typescript", "extraction_serial": 7}
    )
    return HasBlobScenario(
        chunks=[host, ts],
        snapshots=[make_snap("head", "C.svelte", "sfc")],
        query_blob="sfc",
        query_language="svelte",
        serial_map={"svelte": 2, "typescript": 7, "": 1},
        expected=True,
    )


@case(tags=["has_blob"])
def case_has_blob_multilanguage_embedded_bump() -> HasBlobScenario:
    """SFC blob: a delegated chunk stale vs the current embedded serial → False."""
    host = make_chunk("tpl", blob="sfc").model_copy(
        update={"language": "svelte", "extraction_serial": 2}
    )
    ts = make_chunk("fn", blob="sfc").model_copy(
        update={"language": "typescript", "extraction_serial": 7}
    )
    return HasBlobScenario(
        chunks=[host, ts],
        snapshots=[make_snap("head", "C.svelte", "sfc")],
        query_blob="sfc",
        query_language="svelte",
        serial_map={"svelte": 2, "typescript": 8, "": 1},
        expected=False,
    )


# ── Scenario dataclasses for the GC chunk split ─────────────────────


@dataclass(frozen=True)
class SnapshotGroup:
    """Snapshots for one commit of one repo, plus the repo to seed under."""

    repo_id: int
    commit_sha: str
    snapshots: list[Snapshot]


@dataclass(frozen=True)
class GcCountScenario:
    """Seed data + a drop set + expected (dropped, kept) chunk counts."""

    chunks: list[TokenisedChunk]
    groups: list[SnapshotGroup]
    drop_repo_id: int
    drop_shas: list[str]
    expected_dropped: int
    expected_kept: int


# ── gc_counts cases ─────────────────────────────────────────────────


@case(tags=["gc_counts"])
def case_gc_split_last_reference() -> GcCountScenario:
    """The drop removes the chunk's only reference → dropped."""
    c = make_chunk("only", path="a.py", blob="b1")
    return GcCountScenario(
        chunks=[c],
        groups=[SnapshotGroup(1, "c1", [make_snap("c1", "a.py", "b1")])],
        drop_repo_id=1,
        drop_shas=["c1"],
        expected_dropped=1,
        expected_kept=0,
    )


@case(tags=["gc_counts"])
def case_gc_split_shared_cross_repo() -> GcCountScenario:
    """Another repo references the same blob+path → kept."""
    c = make_chunk("shared", path="x.py", blob="b")
    return GcCountScenario(
        chunks=[c],
        groups=[
            SnapshotGroup(1, "c1", [make_snap("c1", "x.py", "b")]),
            SnapshotGroup(2, "c2", [make_snap("c2", "x.py", "b")]),
        ],
        drop_repo_id=1,
        drop_shas=["c1"],
        expected_dropped=0,
        expected_kept=1,
    )


@case(tags=["gc_counts"])
def case_gc_split_shared_same_repo_other_ref() -> GcCountScenario:
    """A surviving ref of the same repo references the chunk → kept."""
    c = make_chunk("multi", path="a.py", blob="b")
    return GcCountScenario(
        chunks=[c],
        groups=[
            SnapshotGroup(1, "c1", [make_snap("c1", "a.py", "b")]),
            SnapshotGroup(1, "c2", [make_snap("c2", "a.py", "b")]),
        ],
        drop_repo_id=1,
        drop_shas=["c1"],
        expected_dropped=0,
        expected_kept=1,
    )


@case(tags=["gc_counts"])
def case_gc_split_mixed() -> GcCountScenario:
    """One candidate loses its last reference; another survives via c2."""
    gone = make_chunk("gone", path="a.py", blob="ba")
    stays = make_chunk("stays", path="b.py", blob="bb")
    return GcCountScenario(
        chunks=[gone, stays],
        groups=[
            SnapshotGroup(1, "c1", [make_snap("c1", "a.py", "ba"), make_snap("c1", "b.py", "bb")]),
            SnapshotGroup(1, "c2", [make_snap("c2", "b.py", "bb")]),
        ],
        drop_repo_id=1,
        drop_shas=["c1"],
        expected_dropped=1,
        expected_kept=1,
    )
