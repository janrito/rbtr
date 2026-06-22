"""Read-side behavioural tests for IndexStore.

Covers: get_chunks filters, get_edges filters, has_blob
language matching, chunk upsert, delete_chunks_for_blobs,
and multi-repo data isolation.
"""

from __future__ import annotations

from pytest_cases import fixture, parametrize_with_cases

from rbtr.index.models import ChunkKind, Edge, EdgeKind
from rbtr.index.store import IndexStore

from .cases_read import ChunkQueryScenario, HasBlobScenario
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
    assert (
        store.has_blob(
            s.query_blob,
            repo_id=1,
            language=s.query_language,
            language_plugin_version=s.query_language_plugin_version,
        )
        == s.expected
    )


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
        ws.delete_chunks_for_blobs({"b1"}, repo_id=1)

    assert store.has_blob("b1", repo_id=1) is False
    assert store.has_blob("b2", repo_id=1) is True


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
    c2 = make_chunk("r2_fn", path="a.py", blob="b_r2", repo_id=2)

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
