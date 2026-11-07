"""Cases for read-side store behaviours.

Scenarios for `chunks`, `edges`, `blob_is_current`,
upsert, and multi-repo isolation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from pytest_cases import case

from rbtr.domain.models import ChunkKind, FileSnapshot, SnapshotRef
from rbtr.index.index import Index
from rbtr.index.search import search
from rbtr.index.staging import TokenisedChunk

from .conftest import make_chunk, make_snap


@dataclass(frozen=True)
class ChunkQueryScenario:
    """Files to build + filter args + the names of the chunks expected back."""

    files: dict[str, bytes]
    file_path: str | None = None
    kind: ChunkKind | None = None
    name: str | None = None
    expected_names: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EmbedOrderScenario:
    """Snapshots to mark indexed, grouped by transaction, and the order
    the embed and counts queries should return them in."""

    transactions: list[list[str]]
    expected: list[str]


@dataclass(frozen=True)
class BlobCurrentScenario:
    """Seed data + blob_is_current(serial map) query + expected result."""

    chunks: list[TokenisedChunk]
    snapshots: list[FileSnapshot]
    query_blob: str
    query_language: str
    serial_map: dict[str, int]
    expected: bool


@case(tags=["chunks"])
def case_get_chunks_unfiltered() -> ChunkQueryScenario:
    """All chunks visible at the commit."""
    a = b"""\
def fn1():
    pass
"""
    b = b"""\
class fn2:
    pass
"""
    return ChunkQueryScenario(files={"a.py": a, "b.py": b}, expected_names=["fn1", "fn2"])


@case(tags=["chunks"])
def case_get_chunks_by_file_path() -> ChunkQueryScenario:
    """Filter by file_path returns only chunks in that file."""
    a = b"""\
def fn1():
    pass
"""
    b = b"""\
def fn2():
    pass
"""
    return ChunkQueryScenario(
        files={"a.py": a, "b.py": b}, file_path="a.py", expected_names=["fn1"]
    )


@case(tags=["chunks"])
def case_get_chunks_by_kind() -> ChunkQueryScenario:
    """Filter by kind returns only chunks of that kind."""
    f = b"""\
def fn1():
    pass
"""
    b = b"""\
class cls1:
    pass
"""
    return ChunkQueryScenario(
        files={"f.py": f, "b.py": b}, kind=ChunkKind.CLASS, expected_names=["cls1"]
    )


@case(tags=["chunks"])
def case_get_chunks_by_name() -> ChunkQueryScenario:
    """Filter by name returns only matching chunk."""
    f = b"""\
def helper():
    pass
"""
    b = b"""\
def runner():
    pass
"""
    return ChunkQueryScenario(
        files={"f.py": f, "b.py": b}, name="helper", expected_names=["helper"]
    )


@case(tags=["chunks"])
def case_get_chunks_nonexistent_file() -> ChunkQueryScenario:
    """Filter by nonexistent file returns empty."""
    f = b"""\
def fn1():
    pass
"""
    return ChunkQueryScenario(files={"f.py": f}, file_path="nope.py", expected_names=[])


@case(tags=["chunks"])
def case_get_chunks_same_blob_two_paths() -> ChunkQueryScenario:
    """Identical content at two paths is one chunk, returned once.

    The blob backs `a.py` and `b.py`, so there is a single chunk reached
    through either. Asking for one path returns it once — not twice for
    the two snapshot rows that share the blob.
    """
    shared = b"""\
def shared():
    pass
"""
    return ChunkQueryScenario(
        files={"a.py": shared, "b.py": shared}, file_path="a.py", expected_names=["shared"]
    )


@case(tags=["blob_is_current"])
def case_blob_is_current_same_language() -> BlobCurrentScenario:
    """Blob exists with matching language."""
    c = make_chunk("fn1")
    return BlobCurrentScenario(
        chunks=[c],
        snapshots=[make_snap("head", "f.py", c.blob_sha)],
        query_blob=c.blob_sha,
        query_language="",
        serial_map={"": 1},
        expected=True,
    )


@case(tags=["blob_is_current"])
def case_blob_is_current_different_language() -> BlobCurrentScenario:
    """Blob exists but with different language → False."""
    c = make_chunk("fn1")
    return BlobCurrentScenario(
        chunks=[c],
        snapshots=[make_snap("head", "f.py", c.blob_sha)],
        query_blob=c.blob_sha,
        query_language="swift",
        serial_map={"swift": 1},
        expected=False,
    )


@case(tags=["blob_is_current"])
def case_blob_is_current_same_language_and_serial() -> BlobCurrentScenario:
    """Blob stored as 'markdown' v1, queried with same → True."""
    c = make_chunk("doc1", kind=ChunkKind.DOC_SECTION, language="markdown")
    return BlobCurrentScenario(
        chunks=[c],
        snapshots=[make_snap("head", "f.py", c.blob_sha, "markdown")],
        query_blob=c.blob_sha,
        query_language="markdown",
        serial_map={"markdown": 1},
        expected=True,
    )


@case(tags=["blob_is_current"])
def case_blob_is_current_different_serial() -> BlobCurrentScenario:
    """Blob stored at serial 1, queried with serial 2 → False."""
    c = make_chunk("doc1", kind=ChunkKind.DOC_SECTION, language="markdown")
    return BlobCurrentScenario(
        chunks=[c],
        snapshots=[make_snap("head", "f.py", c.blob_sha, "markdown")],
        query_blob=c.blob_sha,
        query_language="markdown",
        serial_map={"markdown": 2},
        expected=False,
    )


@case(tags=["blob_is_current"])
def case_blob_is_current_nonexistent() -> BlobCurrentScenario:
    """Blob doesn't exist at all → False."""
    c = make_chunk("fn1")
    return BlobCurrentScenario(
        chunks=[c],
        snapshots=[make_snap("head", "f.py", c.blob_sha)],
        query_blob="no_such_blob",
        query_language="",
        serial_map={"": 1},
        expected=False,
    )


@case(tags=["blob_is_current"])
def case_blob_is_current_detected_language_changed() -> BlobCurrentScenario:
    """Plaintext blob, now detected as python (plugin registered) → False.

    The stored chunk is at a current serial, but there is no chunk in the
    newly detected language, so the file must re-extract.
    """
    c = make_chunk("raw1")  # language "", serial 1
    return BlobCurrentScenario(
        chunks=[c],
        snapshots=[make_snap("head", "mod.py", c.blob_sha)],
        query_blob=c.blob_sha,
        query_language="python",
        serial_map={"python": 4, "": 1},
        expected=False,
    )


@case(tags=["blob_is_current"])
def case_blob_is_current_multilanguage_all_current() -> BlobCurrentScenario:
    """SFC blob: host + embedded chunks all at current serials → True."""
    host = make_chunk("tpl", blob="sfc", language="svelte").model_copy(
        update={"extraction_serial": 2}
    )
    ts = make_chunk("fn", blob="sfc", language="typescript", file_language="svelte").model_copy(
        update={"extraction_serial": 7}
    )
    return BlobCurrentScenario(
        chunks=[host, ts],
        snapshots=[make_snap("head", "C.svelte", "sfc", "svelte")],
        query_blob="sfc",
        query_language="svelte",
        serial_map={"svelte": 2, "typescript": 7, "": 1},
        expected=True,
    )


@case(tags=["blob_is_current"])
def case_blob_is_current_multilanguage_embedded_bump() -> BlobCurrentScenario:
    """SFC blob: a delegated chunk stale vs the current embedded serial → False."""
    host = make_chunk("tpl", blob="sfc", language="svelte").model_copy(
        update={"extraction_serial": 2}
    )
    ts = make_chunk("fn", blob="sfc", language="typescript", file_language="svelte").model_copy(
        update={"extraction_serial": 7}
    )
    return BlobCurrentScenario(
        chunks=[host, ts],
        snapshots=[make_snap("head", "C.svelte", "sfc", "svelte")],
        query_blob="sfc",
        query_language="svelte",
        serial_map={"svelte": 2, "typescript": 8, "": 1},
        expected=False,
    )


@dataclass(frozen=True)
class CheckoutCommit:
    """A commit to build in one checkout of `project_index`.

    *checkout* is the checkout's position in `project_index.repos()`: 0
    is the main checkout, 1 its linked worktree.  *label* names the
    commit, so a drop set can refer to it.
    """

    checkout: int
    label: str
    files: dict[str, bytes]


@dataclass(frozen=True)
class GcCountScenario:
    """Commits to build + a drop set (by label) + expected (dropped, kept) chunk counts."""

    commits: list[CheckoutCommit]
    drop_checkout: int
    drop: list[str]
    expected_dropped: int
    expected_kept: int


@case(tags=["gc_counts"])
def case_gc_split_last_reference() -> GcCountScenario:
    """The drop removes the chunk's only reference → dropped."""
    only = b"""\
def only():
    pass
"""
    return GcCountScenario(
        commits=[CheckoutCommit(0, "c1", {"a.py": only})],
        drop_checkout=0,
        drop=["c1"],
        expected_dropped=1,
        expected_kept=0,
    )


@case(tags=["gc_counts"])
def case_gc_split_shared_cross_repo() -> GcCountScenario:
    """Another repo references the same blob+path → kept."""
    shared = b"""\
def shared():
    pass
"""
    return GcCountScenario(
        commits=[
            CheckoutCommit(0, "c1", {"x.py": shared}),
            CheckoutCommit(1, "c2", {"x.py": shared}),
        ],
        drop_checkout=0,
        drop=["c1"],
        expected_dropped=0,
        expected_kept=1,
    )


@case(tags=["gc_counts"])
def case_gc_split_shared_same_repo_other_ref() -> GcCountScenario:
    """A surviving ref of the same repo references the chunk → kept."""
    multi = b"""\
def multi():
    pass
"""
    return GcCountScenario(
        commits=[
            CheckoutCommit(0, "c1", {"a.py": multi}),
            CheckoutCommit(0, "c2", {"a.py": multi}),
        ],
        drop_checkout=0,
        drop=["c1"],
        expected_dropped=0,
        expected_kept=1,
    )


@case(tags=["gc_counts"])
def case_gc_split_mixed() -> GcCountScenario:
    """One candidate loses its last reference; another survives via c2."""
    gone = b"""\
def gone():
    pass
"""
    stays = b"""\
def stays():
    pass
"""
    return GcCountScenario(
        commits=[
            CheckoutCommit(0, "c1", {"a.py": gone, "b.py": stays}),
            CheckoutCommit(0, "c2", {"b.py": stays}),
        ],
        drop_checkout=0,
        drop=["c1"],
        expected_dropped=1,
        expected_kept=1,
    )


@case(tags=["embed_order"])
def case_newest_transaction_is_embedded_first() -> EmbedOrderScenario:
    """Separate transactions get distinct `indexed_at`, so the newer wins."""
    return EmbedOrderScenario(transactions=[["c1"], ["c2"]], expected=["c2", "c1"])


@case(tags=["embed_order"])
def case_one_transaction_falls_to_the_sha() -> EmbedOrderScenario:
    """One transaction stamps one `indexed_at`, so the tie-break decides.

    Without it the pick would be arbitrary and this test flaky.
    """
    return EmbedOrderScenario(transactions=[["c3", "c1", "c2"]], expected=["c1", "c2", "c3"])


type PathsRead = Callable[[Index, SnapshotRef], set[str]]
"""A read at one snapshot, reduced to the file paths it returned."""


@case(tags=["isolation"])
def case_isolated_chunks() -> PathsRead:
    return lambda index, at: set(index.chunks(at=at)["file_path"])


@case(tags=["isolation"])
def case_isolated_edges() -> PathsRead:
    return lambda index, at: set(index.edges(within=[at])["source_path"])


@case(tags=["isolation"])
def case_isolated_fulltext() -> PathsRead:
    return lambda index, at: set(index.fulltext_matches("shared_term", within=[at])["file_path"])


@case(tags=["isolation"])
def case_isolated_names() -> PathsRead:
    return lambda index, at: set(index.name_matches("fn", within=[at])["file_path"])


@case(tags=["isolation"])
def case_isolated_similar() -> PathsRead:
    return lambda index, at: set(
        index.similar_matches([[1.0, 0.0, 0.0, 0.0]], within=[at])["file_path"]
    )


@case(tags=["isolation"])
def case_isolated_search() -> PathsRead:
    return lambda index, at: {
        path for hit in search(index, "shared_term", within=[at]) for path in hit.file_paths
    }
