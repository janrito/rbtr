"""Read-side behavioural tests for Index.

Covers: chunks filters, edges filters, blob_is_current
language matching, chunk upsert, delete_chunks_for_blobs,
multi-repo data isolation, and cross-repo content sharing
(content-addressed dedup, shared embeddings, reference-counted
cleanup).
"""

from __future__ import annotations

import pytest
from pytest_cases import fixture, parametrize_with_cases

from rbtr.domain.models import ChunkKind, Edge, EdgeKind, SnapshotRef
from rbtr.index.index import Index
from rbtr.index.staging import TokenisedChunk

from .cases_read import (
    BlobCurrentScenario,
    ChunkQueryScenario,
    EmbedOrderScenario,
    GcCountScenario,
    PathsRead,
)
from .conftest import make_chunk, make_snap, seed_index


@fixture
@parametrize_with_cases("scenario", has_tag="chunks")
def chunk_query(scenario: ChunkQueryScenario, index: Index) -> tuple[Index, ChunkQueryScenario]:
    repo = index.register("/repo")
    with index.session(repo) as ws:
        for c in scenario.chunks:
            ws.add_chunk(c)
        ws.insert_snapshots(list(scenario.snapshots))
    return index, scenario


def test_chunks_returns_expected(
    chunk_query: tuple[Index, ChunkQueryScenario],
) -> None:
    index, s = chunk_query
    [repo] = index.repos()
    chunks = index.chunks(
        at=repo.at(s.snapshot_sha),
        file_path=s.file_path,
        kind=s.kind,
        name=s.name,
    )
    assert sorted(c.name for c in chunks) == sorted(s.expected_names)


@fixture
@parametrize_with_cases("scenario", has_tag="blob_is_current")
def blob_query(scenario: BlobCurrentScenario, index: Index) -> tuple[Index, BlobCurrentScenario]:
    repo = index.register("/repo")
    with index.session(repo) as ws:
        for c in scenario.chunks:
            ws.add_chunk(c)
        ws.insert_snapshots(list(scenario.snapshots))
    return index, scenario


def test_blob_is_current_matches(
    blob_query: tuple[Index, BlobCurrentScenario],
) -> None:
    index, s = blob_query
    with index.session(index.repo("/repo")) as ws:
        current = ws.blob_is_current(s.query_blob, s.query_language, s.serial_map)
    assert current == s.expected


def test_upsert_replaces_content(index: Index) -> None:
    """Adding a chunk with the same ID replaces the content."""
    c1 = make_chunk("fn1", content="def fn1(): return 1")
    c2 = make_chunk("fn1", content="def fn1(): return 2")

    repo = index.register("/repo")
    with index.session(repo) as ws:
        ws.add_chunk(c1)
        ws.insert_snapshots([make_snap("head", "f.py", c1.blob_sha)])

    with index.session(repo) as ws:
        ws.add_chunk(c2)

    chunks = index.chunks(at=repo.at("head"))
    assert len(chunks) == 1
    assert "return 2" in chunks[0].content


def test_delete_chunks_for_blobs_removes_target(index: Index) -> None:
    """Deleting a blob's chunks removes them, leaves others."""
    c1 = make_chunk("fn1", blob="b1", path="a.py")
    c2 = make_chunk("fn2", blob="b2", path="b.py")

    repo = index.register("/repo")
    with index.session(repo) as ws:
        ws.add_chunk(c1)
        ws.add_chunk(c2)
        ws.insert_snapshots(
            [make_snap("head", "a.py", "b1"), make_snap("head", "b.py", "b2")],
        )

    with index.session(repo) as ws:
        ws.delete_chunks_for_blobs({"b1"}, file_language="")

    with index.session(repo) as ws:
        assert ws.blob_is_current("b1", "", {"": 1}) is False
        assert ws.blob_is_current("b2", "", {"": 1}) is True


def test_edges_returns_all(index: Index) -> None:
    """Edges inserted for a commit are all returned."""
    e1 = Edge(
        source_id="a",
        target_id="b",
        kind=EdgeKind.IMPORTS,
        source_path="src/a.py",
        target_path="src/b.py",
    )
    e2 = Edge(
        source_id="c",
        target_id="d",
        kind=EdgeKind.DOCUMENTS,
        source_path="src/c.py",
        target_path="src/d.py",
    )

    repo = index.register("/repo")
    with index.session(repo) as ws:
        ws.insert_edges([e1, e2], snapshot_sha="head")

    edges = index.edges(within=[repo.at("head")])
    assert len(edges) == 2


def test_edges_filter_by_kind(index: Index) -> None:
    """Filtering edges by kind returns only matching."""
    e1 = Edge(
        source_id="a",
        target_id="b",
        kind=EdgeKind.IMPORTS,
        source_path="src/a.py",
        target_path="src/b.py",
    )
    e2 = Edge(
        source_id="c",
        target_id="d",
        kind=EdgeKind.DOCUMENTS,
        source_path="src/c.py",
        target_path="src/d.py",
    )

    repo = index.register("/repo")
    with index.session(repo) as ws:
        ws.insert_edges([e1, e2], snapshot_sha="head")

    edges = index.edges(within=[repo.at("head")], kind=EdgeKind.IMPORTS)
    assert edges["kind"].to_list() == [EdgeKind.IMPORTS.value]


def test_inbound_refs_resolves_source(index: Index) -> None:
    """Each inbound edge resolves to its source chunk plus the edge kind."""
    src = make_chunk(
        "imp", name="from m import fn", path="app.py", blob="b_app", kind=ChunkKind.IMPORT
    )
    target = make_chunk("fn", path="m.py", blob="b_m")
    repo = index.register("/repo")
    with index.session(repo) as ws:
        ws.add_chunk(src)
        ws.add_chunk(target)
        ws.insert_snapshots(
            [make_snap("head", "app.py", "b_app"), make_snap("head", "m.py", "b_m")]
        )
        ws.insert_edges(
            [
                Edge(
                    source_id=src.id,
                    target_id=target.id,
                    kind=EdgeKind.IMPORTS,
                    source_path=src.file_path,
                    target_path=target.file_path,
                )
            ],
            snapshot_sha="head",
        )

    frame = index.inbound_refs([target.id], at=repo.at("head"))
    assert frame.height == 1
    row = frame.to_dicts()[0]
    assert row["name"] == "from m import fn"
    assert row["kind"] == "import"
    assert row["file_path"] == "app.py"
    assert row["edge"] == "imports"


def test_inbound_refs_empty_targets(index: Index, head_ref: SnapshotRef) -> None:
    """No target IDs returns an empty frame without touching the store."""
    assert index.inbound_refs([], at=head_ref).is_empty()


@pytest.fixture
def checkouts(project_index: Index) -> Index:
    """`project_index` with different content in each checkout at `head`.

    The main checkout holds `alpha.py` and the linked worktree `beta.py`,
    each with an embedded chunk and an edge from it.  Both checkouts
    write one project file, so only a read's repo filter keeps them
    apart.
    """
    for repo, handle in zip(project_index.repos(), ("alpha", "beta"), strict=True):
        chunk = make_chunk(
            handle,
            name=f"{handle}_fn",
            content=f"def {handle}_fn(): shared_term",
            path=f"{handle}.py",
        )
        seed_index(project_index, [chunk], repo.at("head"))
        edge = Edge(
            source_id=chunk.id,
            target_id=chunk.id,
            kind=EdgeKind.IMPORTS,
            source_path=chunk.file_path,
            target_path=chunk.file_path,
        )
        with project_index.session(repo) as ws:
            ws.insert_edges([edge], snapshot_sha="head")
            ws.update_embeddings([chunk.id], [[1.0, 0.0, 0.0, 0.0]])
    return project_index


@parametrize_with_cases("read", cases=".cases_read", has_tag="isolation")
def test_a_read_sees_only_its_own_checkout(read: PathsRead, checkouts: Index) -> None:
    """Two checkouts share a project file; a read at one sees none of the other's rows."""
    main, worktree = checkouts.repos()
    assert read(checkouts, main.at("head")) == {"alpha.py"}
    assert read(checkouts, worktree.at("head")) == {"beta.py"}


def test_shared_content_is_one_row_visible_to_both_repos(
    shared_chunk_index: Index,
    shared_chunk: TokenisedChunk,
) -> None:
    """Shared content is a single physical row, visible to every repo."""
    index = shared_chunk_index
    repo1, repo2 = index.repos()
    assert [c.id for c in index.chunks(at=repo1.at("head"))] == [shared_chunk.id]
    assert [c.id for c in index.chunks(at=repo2.at("head"))] == [shared_chunk.id]
    with index.reader(repo1) as cur:
        rows = cur.sql("SELECT count(*) FROM chunks").fetchall()
    assert rows == [(1,)]


def test_shared_chunk_embedded_once_across_repos(
    shared_chunk_index: Index,
    shared_chunk: TokenisedChunk,
) -> None:
    """Embedding the shared chunk once leaves no repo with work to do."""
    index = shared_chunk_index
    repo1, repo2 = index.repos()
    in_repo1 = repo1.at("head")
    in_repo2 = repo2.at("head")
    assert index.chunk_counts_for_snapshot(at=in_repo1).unembedded == 1
    assert index.chunk_counts_for_snapshot(at=in_repo2).unembedded == 1

    with index.session(repo1) as ws:
        ws.update_embeddings([shared_chunk.id], [[0.1, 0.2, 0.3]])

    assert index.chunk_counts_for_snapshot(at=in_repo1).is_fully_embedded
    assert index.chunk_counts_for_snapshot(at=in_repo2).is_fully_embedded


def test_cleanup_keeps_chunk_referenced_by_another_repo(
    shared_chunk_index: Index,
    shared_chunk: TokenisedChunk,
) -> None:
    """A repo's build-finalise `cleanup()` must not prune a co-tenant's chunk.

    `cleanup()` runs the global `prune_chunks`. While repo 2 still
    references the shared blob, repo 1's cleanup must remove nothing and
    leave the chunk visible to both — the realistic co-tenant case, since
    `cleanup` runs at the end of every build.
    """
    index = shared_chunk_index
    repo1, repo2 = index.repos()
    with index.session(repo1) as ws:
        cleaned = ws.cleanup()
    assert cleaned.chunks == 0
    assert index.count_orphan_chunks() == 0
    assert [c.id for c in index.chunks(at=repo1.at("head"))] == [shared_chunk.id]
    assert [c.id for c in index.chunks(at=repo2.at("head"))] == [shared_chunk.id]


def test_shared_chunk_swept_only_after_last_repo_drops_it(
    shared_chunk_index: Index,
    shared_chunk: TokenisedChunk,
) -> None:
    """Cross-repo refcount: the chunk survives until the last reference goes."""
    index = shared_chunk_index
    repo1, repo2 = index.repos()
    # Repo 1 drops its commit: shared chunk survives (repo 2 references it).
    with index.session(repo1) as ws:
        first_drop = ws.drop_snapshot(snapshot_sha="head")
    assert first_drop.chunks == 0
    assert [c.id for c in index.chunks(at=repo2.at("head"))] == [shared_chunk.id]

    # Repo 2 drops its commit: last reference gone, chunk is swept.
    with index.session(repo2) as ws:
        second_drop = ws.drop_snapshot(snapshot_sha="head")
    assert second_drop.chunks == 1
    assert index.count_orphan_chunks() == 0
    assert index.chunks(at=repo2.at("head")) == []


def test_rechunk_of_shared_blob_propagates_to_all_repos(
    shared_chunk_index: Index,
) -> None:
    """Re-chunking a shared blob updates the content for every repo.

    A plugin upgrade re-chunks a blob globally (delete its chunks, then
    re-insert at the new serial). Because the chunk is content-addressed
    and shared, every repo referencing the blob — not just the one that
    triggered the re-chunk — sees the new chunks, without re-indexing.
    """
    index = shared_chunk_index  # repos 1 and 2 reference blob "b_shared"
    repo1, repo2 = index.repos()
    rechunked = make_chunk("shared_fn_v2", path="x.py", blob="b_shared").model_copy(
        update={"extraction_serial": 2}
    )
    # A different name, so a different id: the upgrade replaces the row.
    with index.session(repo1) as ws:
        ws.delete_chunks_for_blobs({"b_shared"}, file_language="")
        ws.add_chunk(rechunked)

    assert [c.id for c in index.chunks(at=repo1.at("head"))] == [rechunked.id]
    assert [c.id for c in index.chunks(at=repo2.at("head"))] == [rechunked.id]


def test_drop_snapshot_keeps_a_chunk_still_reachable_at_another_path(
    index: Index,
) -> None:
    """`drop_snapshot` keeps a chunk whose blob is referenced elsewhere.

    Identical content at `a.py` and `b.py` is one chunk reached through
    two paths, so dropping the commit that referenced `b.py` collects
    nothing: the chunk is still reachable at `a.py`, and deleting it
    would empty a file that has not changed.
    """
    repo = index.register("/repo")
    with index.session(repo) as ws:
        shared = make_chunk("shared", path="a.py", blob="b")
        ws.add_chunk(shared)
        # c1 references the blob at both paths; c2 keeps only a.py.
        ws.insert_snapshots([make_snap("c1", "a.py", "b"), make_snap("c1", "b.py", "b")])
        ws.insert_snapshots([make_snap("c2", "a.py", "b")])
        ws.mark_indexed(snapshot_sha="c1")
        ws.mark_indexed(snapshot_sha="c2")

    with index.session(repo) as ws:
        dropped = ws.drop_snapshot(snapshot_sha="c1")

    assert dropped.chunks == 0
    assert index.count_orphan_chunks() == 0
    assert [c.id for c in index.chunks(at=repo.at("c2"))] == [shared.id]


def test_inbound_refs_to_shared_chunk_resolve_per_repo(
    shared_chunk_index: Index,
    shared_chunk: TokenisedChunk,
) -> None:
    """inbound_refs to a shared chunk returns only the querying repo's referrers.

    Edges are per-repo: each repo infers its own inbound edge to the
    shared chunk from its own importer. Querying the shared target must
    resolve the querying repo's referrer, never the other repo's.
    """
    index = shared_chunk_index
    repo1, repo2 = index.repos()
    with index.session(repo1) as ws:
        imp1 = make_chunk(
            "imp1",
            name="from m import shared_fn",
            path="a.py",
            blob="b_imp1",
            kind=ChunkKind.IMPORT,
        )
        ws.add_chunk(imp1)
        ws.insert_snapshots([make_snap("head", "a.py", "b_imp1")])
        ws.insert_edges(
            [
                Edge(
                    source_id=imp1.id,
                    target_id=shared_chunk.id,
                    kind=EdgeKind.IMPORTS,
                    source_path=imp1.file_path,
                    target_path=shared_chunk.file_path,
                )
            ],
            snapshot_sha="head",
        )
    with index.session(repo2) as ws:
        imp2 = make_chunk(
            "imp2",
            name="from m import shared_fn",
            path="b.py",
            blob="b_imp2",
            kind=ChunkKind.IMPORT,
        )
        ws.add_chunk(imp2)
        ws.insert_snapshots([make_snap("head", "b.py", "b_imp2")])
        ws.insert_edges(
            [
                Edge(
                    source_id=imp2.id,
                    target_id=shared_chunk.id,
                    kind=EdgeKind.IMPORTS,
                    source_path=imp2.file_path,
                    target_path=shared_chunk.file_path,
                )
            ],
            snapshot_sha="head",
        )

    r1 = index.inbound_refs([shared_chunk.id], at=repo1.at("head")).to_dicts()
    r2 = index.inbound_refs([shared_chunk.id], at=repo2.at("head")).to_dicts()
    assert [row["file_path"] for row in r1] == ["a.py"]
    assert [row["file_path"] for row in r2] == ["b.py"]


@fixture
@parametrize_with_cases("scenario", cases=".cases_read", has_tag="gc_counts")
def gc_count_query(
    scenario: GcCountScenario, project_index: Index
) -> tuple[Index, GcCountScenario]:
    index = project_index
    # Claims first, one session per repo; then the chunks, which a
    # session may add only for blobs some snapshot of the project claims.
    repos = index.repos()
    for g in scenario.groups:
        with index.session(repos[g.checkout]) as ws:
            ws.insert_snapshots(g.snapshots)
            ws.mark_indexed(snapshot_sha=g.snapshot_sha)
    with index.session(repos[0]) as ws:
        for c in scenario.chunks:
            ws.add_chunk(c)
    return index, scenario


def test_gc_chunk_split_counts(
    gc_count_query: tuple[Index, GcCountScenario],
) -> None:
    """`count_gc_chunk_split` splits candidate chunks into dropped vs kept.

    A candidate (referenced by the drop set) is kept when a snapshot
    outside the drop set still references its `(blob_sha, file_path)`,
    else it is dropped.
    """
    index, s = gc_count_query
    split = index.count_gc_chunk_split(index.repos()[s.drop_checkout], s.drop_shas)
    assert (split.dropped, split.kept) == (s.expected_dropped, s.expected_kept)


def test_gc_chunk_split_agrees_with_real_deletion(
    gc_count_query: tuple[Index, GcCountScenario],
) -> None:
    """The predicted drop count matches a real deletion, leaving no orphans.

    Guards against drift between `count_gc_chunk_split` (the prediction)
    and the `drop_snapshot` sweep (the deletion), and between the sweep and
    `count_orphan_chunks`: dropping the drop set's commits must remove
    exactly the predicted number of chunks and leave the store
    orphan-free. Both sides are read off the same reference graph, so
    they must never diverge.
    """
    index, s = gc_count_query
    drop = index.repos()[s.drop_checkout]
    predicted_dropped = index.count_gc_chunk_split(drop, s.drop_shas).dropped

    with index.reader(drop) as cur:
        ((before,),) = cur.sql("SELECT count(*) FROM chunks").fetchall()
    with index.session(drop) as ws:
        for sha in s.drop_shas:
            ws.drop_snapshot(snapshot_sha=sha)
    with index.reader(drop) as cur:
        ((after,),) = cur.sql("SELECT count(*) FROM chunks").fetchall()

    assert before - after == predicted_dropped
    assert index.count_orphan_chunks() == 0


def test_forget_removes_references_keeps_shared_chunk(
    shared_chunk_index: Index,
) -> None:
    """`forget` purges a checkout's references and `repos` row but leaves a
    chunk the project's other checkout still references; reclamation is
    gc's job."""
    index = shared_chunk_index
    repo1, repo2 = index.repos()
    with index.session(repo1) as ws:
        ws.add_watched_refs(["main"])

    index.forget(repo1)

    # Repo 1 is gone entirely, its rows with it.
    assert index.repos() == [repo2]
    gone = {"repo_id": repo1.repo_id}
    with index.reader(repo2) as cur:
        watched = cur.sql("SELECT count(*) FROM watched_refs WHERE repo_id = $repo_id", params=gone)
        assert watched.fetchall() == [(0,)]
        indexed = cur.sql(
            "SELECT count(*) FROM indexed_snapshots WHERE repo_id = $repo_id", params=gone
        )
        assert indexed.fetchall() == [(0,)]
    # Repo 2 is untouched and the shared chunk is still visible to it.
    assert index.has_indexed(at=repo2.at("head")) is True
    assert len(index.chunks(at=repo2.at("head"))) > 0


def test_forget_leaves_orphan_chunks_for_gc(
    shared_chunk_index: Index,
) -> None:
    """`forget` does not sweep: a chunk only the forgotten repo
    referenced becomes an orphan that a later GC reclaims."""
    index = shared_chunk_index
    repo1, _repo2 = index.repos()
    with index.session(repo1) as ws:
        ws.add_chunk(make_chunk("only1", path="y.py", blob="b_only1"))
        ws.insert_snapshots([make_snap("head", "y.py", "b_only1")])
    assert index.count_orphan_chunks() == 0

    index.forget(repo1)

    # only1 is now unreferenced (repo 1 gone) but still present — no sweep.
    # (A sweep would have removed it, leaving zero orphans.)
    assert index.count_orphan_chunks() == 1


def test_an_indexed_snapshot_with_no_chunks_counts_zero(index: Index) -> None:
    """A commit that yielded nothing to index is still reported, at zero.

    `mark_indexed` records completion for any tree, including one that
    held nothing extractable, so the query answers for it with a row of
    zeroes and the caller reads a figure rather than supplying one.
    """
    repo = index.register("/r")
    with index.session(repo) as ws:
        ws.mark_indexed(snapshot_sha="c1")

    counts = index.chunk_counts_for_snapshot(at=repo.at("c1"))
    assert counts.total == 0
    assert counts.is_fully_embedded, "nothing outstanding means no embed work"
    assert [ref.snapshot_sha for ref in index.chunk_counts_by_snapshot(repo=repo)] == ["c1"]


@parametrize_with_cases("scenario", cases=".cases_read", has_tag="embed_order")
def test_counts_come_back_in_embed_order(scenario: EmbedOrderScenario, index: Index) -> None:
    """Order decides which snapshot is embedded next, so the query fixes it.

    Asserted through `chunk_counts_by_snapshot`, because that is what
    both the status handler and the job picker actually read.
    """
    for shas in scenario.transactions:
        repo = index.register("/r")
        with index.session(repo) as ws:
            for sha in shas:
                ws.mark_indexed(snapshot_sha=sha)

    counted = index.chunk_counts_by_snapshot(repo=repo)
    assert [ref.snapshot_sha for ref in counted] == scenario.expected
