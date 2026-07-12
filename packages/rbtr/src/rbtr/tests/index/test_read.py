"""Read-side behavioural tests for IndexStore.

Covers: get_chunks filters, get_edges filters, has_blob
language matching, chunk upsert, delete_chunks_for_blobs,
multi-repo data isolation, and cross-repo content sharing
(content-addressed dedup, shared embeddings, reference-counted
cleanup).
"""

from __future__ import annotations

from pytest_cases import fixture, parametrize_with_cases

from rbtr.index.models import ChunkKind, Edge, EdgeKind
from rbtr.index.store import IndexStore

from .cases_read import ChunkQueryScenario, GcCountScenario, HasBlobScenario
from .conftest import make_chunk, make_snap

# ── get_chunks ──────────────────────────────────────────────────────


@fixture
@parametrize_with_cases("scenario", has_tag="get_chunks")
def chunk_query(
    scenario: ChunkQueryScenario, store: IndexStore
) -> tuple[IndexStore, ChunkQueryScenario]:
    with store.session() as ws:
        for c in scenario.chunks:
            ws.add_chunk(c)
        ws.insert_snapshots(list(scenario.snapshots), repo_id=1)
    return store, scenario


def test_get_chunks_returns_expected(
    chunk_query: tuple[IndexStore, ChunkQueryScenario],
) -> None:
    store, s = chunk_query
    chunks = store.get_chunks(
        s.commit_sha,
        file_path=s.file_path,
        kind=s.kind,
        name=s.name,
        repo_id=1,
    )
    assert sorted(c.id for c in chunks) == sorted(s.expected_ids)


# ── has_blob ────────────────────────────────────────────────────────


@fixture
@parametrize_with_cases("scenario", has_tag="has_blob")
def blob_query(scenario: HasBlobScenario, store: IndexStore) -> tuple[IndexStore, HasBlobScenario]:
    with store.session() as ws:
        for c in scenario.chunks:
            ws.add_chunk(c)
        ws.insert_snapshots(list(scenario.snapshots), repo_id=1)
    return store, scenario


def test_has_blob_matches(
    blob_query: tuple[IndexStore, HasBlobScenario],
) -> None:
    store, s = blob_query
    assert store.has_blob(s.query_blob, s.query_language, s.version_map) == s.expected


# ── Chunk upsert ────────────────────────────────────────────────────


def test_upsert_replaces_content(store: IndexStore) -> None:
    """Adding a chunk with the same ID replaces the content."""
    c1 = make_chunk("fn1", content="def fn1(): return 1")
    c2 = make_chunk("fn1", content="def fn1(): return 2")

    with store.session() as ws:
        ws.add_chunk(c1)
        ws.insert_snapshots([make_snap("head", "f.py", c1.blob_sha)], repo_id=1)

    with store.session() as ws:
        ws.add_chunk(c2)

    chunks = store.get_chunks("head", repo_id=1)
    assert len(chunks) == 1
    assert "return 2" in chunks[0].content


# ── delete_chunks_for_blobs ─────────────────────────────────────────


def test_delete_chunks_for_blobs_removes_target(store: IndexStore) -> None:
    """Deleting a blob's chunks removes them, leaves others."""
    c1 = make_chunk("fn1", blob="b1", path="a.py")
    c2 = make_chunk("fn2", blob="b2", path="b.py")

    with store.session() as ws:
        ws.add_chunk(c1)
        ws.add_chunk(c2)
        ws.insert_snapshots(
            [make_snap("head", "a.py", "b1"), make_snap("head", "b.py", "b2")],
            repo_id=1,
        )

    with store.session() as ws:
        ws.delete_chunks_for_blobs({"b1"})

    assert store.has_blob("b1", "", {"": 1}) is False
    assert store.has_blob("b2", "", {"": 1}) is True


# ── get_edges ───────────────────────────────────────────────────────


def test_get_edges_returns_all(store: IndexStore) -> None:
    """Edges inserted for a commit are all returned."""
    e1 = Edge(source_id="a", target_id="b", kind=EdgeKind.IMPORTS)
    e2 = Edge(source_id="c", target_id="d", kind=EdgeKind.TESTS)

    with store.session() as ws:
        ws.insert_edges([e1, e2], "head", repo_id=1)

    edges = store.get_edges("head", repo_id=1)
    assert len(edges) == 2


def test_get_edges_filter_by_kind(store: IndexStore) -> None:
    """Filtering edges by kind returns only matching."""
    e1 = Edge(source_id="a", target_id="b", kind=EdgeKind.IMPORTS)
    e2 = Edge(source_id="c", target_id="d", kind=EdgeKind.TESTS)

    with store.session() as ws:
        ws.insert_edges([e1, e2], "head", repo_id=1)

    edges = store.get_edges("head", kind=EdgeKind.IMPORTS, repo_id=1)
    assert len(edges) == 1
    assert edges[0].kind == EdgeKind.IMPORTS


# ── inbound_refs ─────────────────────────────────────


def test_inbound_refs_resolves_source(store: IndexStore) -> None:
    """Each inbound edge resolves to its source chunk plus the edge kind."""
    src = make_chunk(
        "imp", name="from m import fn", path="app.py", blob="b_app", kind=ChunkKind.IMPORT
    )
    with store.session() as ws:
        ws.register_repo("/repo")
        ws.add_chunk(src)
        ws.insert_snapshots([make_snap("head", "app.py", "b_app")], repo_id=1)
        ws.insert_edges(
            [Edge(source_id="imp", target_id="fn", kind=EdgeKind.IMPORTS)], "head", repo_id=1
        )

    frame = store.inbound_refs("head", ["fn"], repo_id=1)
    assert frame.height == 1
    row = frame.to_dicts()[0]
    assert row["name"] == "from m import fn"
    assert row["kind"] == "import"
    assert row["file_path"] == "app.py"
    assert row["edge"] == "imports"


def test_inbound_refs_empty_targets(store: IndexStore) -> None:
    """No target IDs returns an empty frame without touching the store."""
    assert store.inbound_refs("head", [], repo_id=1).is_empty()


# ── Multi-repo isolation ────────────────────────────────────────────


def test_get_chunks_isolated_per_repo(store: IndexStore) -> None:
    """Chunks in repo 1 are invisible to repo 2."""
    with store.session() as ws:
        ws.register_repo("/repo1")
        ws.register_repo("/repo2")

    c1 = make_chunk("r1_fn", path="a.py", blob="b_r1")
    c2 = make_chunk("r2_fn", path="a.py", blob="b_r2")

    with store.session() as ws:
        ws.add_chunk(c1)
        ws.insert_snapshots([make_snap("head", "a.py", "b_r1")], repo_id=1)

    with store.session() as ws:
        ws.add_chunk(c2)
        ws.insert_snapshots([make_snap("head", "a.py", "b_r2")], repo_id=2)

    r1_chunks = store.get_chunks("head", repo_id=1)
    r2_chunks = store.get_chunks("head", repo_id=2)

    assert [c.id for c in r1_chunks] == ["r1_fn"]
    assert [c.id for c in r2_chunks] == ["r2_fn"]


def test_get_edges_isolated_per_repo(store: IndexStore) -> None:
    """Edges in repo 1 are invisible to repo 2."""
    with store.session() as ws:
        ws.register_repo("/repo1")
        ws.register_repo("/repo2")

    e1 = Edge(source_id="a", target_id="b", kind=EdgeKind.IMPORTS)
    e2 = Edge(source_id="x", target_id="y", kind=EdgeKind.IMPORTS)

    with store.session() as ws:
        ws.insert_edges([e1], "head", repo_id=1)

    with store.session() as ws:
        ws.insert_edges([e2], "head", repo_id=2)

    assert len(store.get_edges("head", repo_id=1)) == 1
    assert len(store.get_edges("head", repo_id=2)) == 1
    assert store.get_edges("head", repo_id=1)[0].source_id == "a"
    assert store.get_edges("head", repo_id=2)[0].source_id == "x"


# ── Cross-repo content sharing ───────────────────────────────────────


def test_shared_content_is_one_row_visible_to_both_repos(
    shared_chunk_store: IndexStore,
) -> None:
    """Shared content is a single physical row, visible to every repo."""
    store = shared_chunk_store
    assert [c.id for c in store.get_chunks("head", repo_id=1)] == ["shared_fn"]
    assert [c.id for c in store.get_chunks("head", repo_id=2)] == ["shared_fn"]
    row = store._cursor.execute("SELECT count(*) FROM chunks").fetchone()
    assert row is not None
    assert row[0] == 1


def test_shared_chunk_embedded_once_across_repos(
    shared_chunk_store: IndexStore,
) -> None:
    """Embedding the shared chunk once leaves no repo with work to do."""
    store = shared_chunk_store
    assert store.count_unembedded(repo_id=1, commit_sha="head") == 1
    assert store.count_unembedded(repo_id=2, commit_sha="head") == 1

    with store.session() as ws:
        ws.update_embeddings(["shared_fn"], [[0.1, 0.2, 0.3]])

    assert store.count_unembedded(repo_id=1, commit_sha="head") == 0
    assert store.count_unembedded(repo_id=2, commit_sha="head") == 0


def test_cleanup_keeps_chunk_referenced_by_another_repo(
    shared_chunk_store: IndexStore,
) -> None:
    """A repo's build-finalise `cleanup()` must not prune a co-tenant's chunk.

    `cleanup()` runs the global `prune_chunks`. While repo 2 still
    references the shared blob, repo 1's cleanup must remove nothing and
    leave the chunk visible to both — the realistic co-tenant case, since
    `cleanup` runs at the end of every build.
    """
    store = shared_chunk_store
    with store.session() as ws:
        cleaned = ws.cleanup(1)
    assert cleaned.chunks == 0
    assert store.count_orphan_chunks() == 0
    assert [c.id for c in store.get_chunks("head", repo_id=1)] == ["shared_fn"]
    assert [c.id for c in store.get_chunks("head", repo_id=2)] == ["shared_fn"]


def test_shared_chunk_swept_only_after_last_repo_drops_it(
    shared_chunk_store: IndexStore,
) -> None:
    """Cross-repo refcount: the chunk survives until the last reference goes."""
    store = shared_chunk_store
    # Repo 1 drops its commit: shared chunk survives (repo 2 references it).
    with store.session() as ws:
        first_drop = ws.drop_commit(1, "head")
    assert first_drop.chunks == 0
    assert [c.id for c in store.get_chunks("head", repo_id=2)] == ["shared_fn"]

    # Repo 2 drops its commit: last reference gone, chunk is swept.
    with store.session() as ws:
        second_drop = ws.drop_commit(2, "head")
    assert second_drop.chunks == 1
    assert store.count_orphan_chunks() == 0
    assert store.get_chunks("head", repo_id=2) == []


def test_rechunk_of_shared_blob_propagates_to_all_repos(
    shared_chunk_store: IndexStore,
) -> None:
    """Re-chunking a shared blob updates the content for every repo.

    A plugin upgrade re-chunks a blob globally (delete its chunks, then
    re-insert at the new version). Because the chunk is content-addressed
    and shared, every repo referencing the blob — not just the one that
    triggered the re-chunk — sees the new chunks, without re-indexing.
    """
    store = shared_chunk_store  # repos 1 and 2 reference blob "b_shared"
    rechunked = make_chunk("shared_fn_v2", path="x.py", blob="b_shared").model_copy(
        update={"language_plugin_version": 2}
    )
    with store.session() as ws:
        ws.delete_chunks_for_blobs({"b_shared"})
        ws.add_chunk(rechunked)

    assert [c.id for c in store.get_chunks("head", repo_id=1)] == ["shared_fn_v2"]
    assert [c.id for c in store.get_chunks("head", repo_id=2)] == ["shared_fn_v2"]


def test_drop_commit_sweeps_chunk_orphaned_at_one_path_of_a_shared_blob(
    store: IndexStore,
) -> None:
    """`drop_commit` collects a chunk orphaned at its path even when the
    same blob stays referenced at another path.

    A blob backing identical content at `a.py` and `b.py` yields two
    distinct chunks sharing one `blob_sha` (the path is part of the id).
    Dropping the commit that referenced `b.py` orphans that path's chunk;
    the sweep must collect it though the blob is still referenced via
    `a.py` — i.e. it must key on `(blob_sha, file_path)`, like prune,
    not on `blob_sha` alone.
    """
    with store.session() as ws:
        ws.add_chunk(make_chunk("at_a", path="a.py", blob="b"))
        ws.add_chunk(make_chunk("at_b", path="b.py", blob="b"))
        # c1 references the blob at both paths; c2 keeps only a.py.
        ws.insert_snapshots([make_snap("c1", "a.py", "b"), make_snap("c1", "b.py", "b")], repo_id=1)
        ws.insert_snapshots([make_snap("c2", "a.py", "b")], repo_id=1)
        ws.mark_indexed(1, "c1")
        ws.mark_indexed(1, "c2")

    with store.session() as ws:
        dropped = ws.drop_commit(1, "c1")

    assert dropped.chunks == 1
    assert store.count_orphan_chunks() == 0
    assert [c.id for c in store.get_chunks("c2", repo_id=1)] == ["at_a"]


def test_inbound_refs_to_shared_chunk_resolve_per_repo(
    shared_chunk_store: IndexStore,
) -> None:
    """inbound_refs to a shared chunk returns only the querying repo's referrers.

    Edges are per-repo: each repo infers its own inbound edge to the
    shared chunk from its own importer. Querying the shared target must
    resolve the querying repo's referrer, never the other repo's.
    """
    store = shared_chunk_store
    with store.session() as ws:
        ws.add_chunk(
            make_chunk(
                "imp1",
                name="from m import shared_fn",
                path="a.py",
                blob="b_imp1",
                kind=ChunkKind.IMPORT,
            )
        )
        ws.insert_snapshots([make_snap("head", "a.py", "b_imp1")], repo_id=1)
        ws.insert_edges(
            [Edge(source_id="imp1", target_id="shared_fn", kind=EdgeKind.IMPORTS)],
            "head",
            repo_id=1,
        )
        ws.add_chunk(
            make_chunk(
                "imp2",
                name="from m import shared_fn",
                path="b.py",
                blob="b_imp2",
                kind=ChunkKind.IMPORT,
            )
        )
        ws.insert_snapshots([make_snap("head", "b.py", "b_imp2")], repo_id=2)
        ws.insert_edges(
            [Edge(source_id="imp2", target_id="shared_fn", kind=EdgeKind.IMPORTS)],
            "head",
            repo_id=2,
        )

    r1 = store.inbound_refs("head", ["shared_fn"], repo_id=1).to_dicts()
    r2 = store.inbound_refs("head", ["shared_fn"], repo_id=2).to_dicts()
    assert [row["file_path"] for row in r1] == ["a.py"]
    assert [row["file_path"] for row in r2] == ["b.py"]


# ── gc chunk split (dropped vs kept-because-shared) ──────────────────


@fixture
@parametrize_with_cases("scenario", cases=".cases_read", has_tag="gc_counts")
def gc_count_query(
    scenario: GcCountScenario, store: IndexStore
) -> tuple[IndexStore, GcCountScenario]:
    with store.session() as ws:
        for c in scenario.chunks:
            ws.add_chunk(c)
        for g in scenario.groups:
            ws.insert_snapshots(g.snapshots, repo_id=g.repo_id)
            ws.mark_indexed(g.repo_id, g.commit_sha)
    return store, scenario


def test_gc_chunk_split_counts(
    gc_count_query: tuple[IndexStore, GcCountScenario],
) -> None:
    """`count_gc_chunk_split` splits candidate chunks into dropped vs kept.

    A candidate (referenced by the drop set) is kept when a snapshot
    outside the drop set still references its `(blob_sha, file_path)`,
    else it is dropped.
    """
    store, s = gc_count_query
    assert store.count_gc_chunk_split(s.drop_repo_id, s.drop_shas) == (
        s.expected_dropped,
        s.expected_kept,
    )


def test_gc_chunk_split_agrees_with_real_deletion(
    gc_count_query: tuple[IndexStore, GcCountScenario],
) -> None:
    """The predicted drop count matches a real deletion, leaving no orphans.

    Guards against drift between `count_gc_chunk_split` (the prediction)
    and the `drop_commit` sweep (the deletion), and between the sweep and
    `count_orphan_chunks`: dropping the drop set's commits must remove
    exactly the predicted number of chunks and leave the store
    orphan-free. Both sides are read off the same reference graph, so
    they must never diverge.
    """
    store, s = gc_count_query
    predicted_dropped, _kept = store.count_gc_chunk_split(s.drop_repo_id, s.drop_shas)

    before = store._cursor.execute("SELECT count(*) FROM chunks").fetchone()
    with store.session() as ws:
        for sha in s.drop_shas:
            ws.drop_commit(s.drop_repo_id, sha)
    after = store._cursor.execute("SELECT count(*) FROM chunks").fetchone()

    assert before is not None
    assert after is not None
    assert before[0] - after[0] == predicted_dropped
    assert store.count_orphan_chunks() == 0


# ── forget_repo: metadata-only purge of a whole repo ─────────────────


def test_forget_repo_removes_references_keeps_shared_chunk(
    shared_chunk_store: IndexStore,
) -> None:
    """forget_repo purges a repo's references and `repos` row but leaves a
    chunk another repo still references; reclamation is gc's job."""
    store = shared_chunk_store
    with store.session() as ws:
        ws.add_watched_refs(1, ["main"])

    with store.session() as ws:
        ws.forget_repo(1)

    # Repo 1 is gone entirely.
    assert store.get_repo_id("/repo1") is None
    assert store.list_watched_refs(1) == []
    assert store.has_indexed(1, "head") is False
    # Repo 2 is untouched and the shared chunk is still visible to it.
    assert store.get_repo_id("/repo2") == 2
    assert store.has_indexed(2, "head") is True
    assert len(store.get_chunks("head", repo_id=2)) > 0


def test_forget_repo_leaves_orphan_chunks_for_gc(
    shared_chunk_store: IndexStore,
) -> None:
    """forget_repo does not sweep: a chunk only the forgotten repo
    referenced becomes an orphan that a later GC reclaims."""
    store = shared_chunk_store
    with store.session() as ws:
        ws.add_chunk(make_chunk("only1", path="y.py", blob="b_only1"))
        ws.insert_snapshots([make_snap("head", "y.py", "b_only1")], repo_id=1)
    assert store.count_orphan_chunks() == 0

    with store.session() as ws:
        ws.forget_repo(1)

    # only1 is now unreferenced (repo 1 gone) but still present — no sweep.
    # (A sweep would have removed it, leaving zero orphans.)
    assert store.count_orphan_chunks() == 1


def test_forget_repo_purges_indexed_commits_incl_worktree_sha(
    store: IndexStore,
) -> None:
    """Every indexed commit for the repo — HEAD and any worktree tree SHA —
    is purged along with the `repos` row."""
    with store.session() as ws:
        ws.register_repo("/r")
        ws.add_chunk(make_chunk("c", path="a.py", blob="b"))
        ws.insert_snapshots([make_snap("headsha", "a.py", "b")], repo_id=1)
        ws.mark_indexed(1, "headsha")
        ws.mark_indexed(1, "treesha")  # a working-tree tree SHA
    assert store.has_indexed(1, "headsha") is True
    assert store.has_indexed(1, "treesha") is True

    with store.session() as ws:
        ws.forget_repo(1)

    assert store.get_repo_id("/r") is None
    assert store.has_indexed(1, "headsha") is False
    assert store.has_indexed(1, "treesha") is False
