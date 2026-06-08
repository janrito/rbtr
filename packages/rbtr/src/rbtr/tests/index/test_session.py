"""Behavioural tests for WriteSession mechanics.

Verifies the transactional contract: commit-on-exit,
rollback-on-exception, guard rails, sweep behaviour,
FTS rebuild, chunk buffering, and write method round-trips.
"""

from __future__ import annotations

import pytest
from dataframely.exc import ValidationError
from pytest_cases import parametrize_with_cases

from rbtr.config import config
from rbtr.errors import RbtrError
from rbtr.index.models import Edge, EdgeKind, Snapshot, TokenisedChunk
from rbtr.index.store import IndexStore

from .cases_store_repos import RepoSequence
from .conftest import make_chunk, make_snap

# ── Commit / rollback ───────────────────────────────────────────────


def test_clean_exit_commits(store: IndexStore) -> None:
    """Data written in a session is visible after clean exit."""
    with store.session() as ws:
        ws.add_chunk(make_chunk("a"))
        ws.insert_snapshots([make_snap("c1", "f.py", "blob_a")], repo_id=1)

    chunks = store.get_chunks("c1", repo_id=1)
    assert any(c.id == "a" for c in chunks)


def test_exception_rolls_back(store: IndexStore) -> None:
    """Data written in a session is invisible after an exception."""

    def _failing_session() -> None:
        with store.session() as ws:
            ws.add_chunk(make_chunk("a"))
            ws.insert_snapshots([make_snap("c1", "f.py", "blob_a")], repo_id=1)
            msg = "boom"
            raise RuntimeError(msg)

    with pytest.raises(RuntimeError):
        _failing_session()

    chunks = store.get_chunks("c1", repo_id=1)
    assert chunks == []


# ── Guard rails ──────────────────────────────────────────────────────


def test_write_outside_session_raises(store: IndexStore) -> None:
    """Calling a write method without an active session raises."""
    ws = store.session()
    # ws.__enter__ not called — no active transaction.
    with pytest.raises(RuntimeError, match="No active transaction"):
        ws.add_chunk(make_chunk("a"))


def test_read_only_store_rejects_session() -> None:
    """A store created without writable=True rejects session()."""
    store = IndexStore(writable=False)
    try:
        with pytest.raises(RuntimeError, match="writable"):
            store.session()
    finally:
        store.close()


def test_add_chunk_attributes_per_repo_in_one_session(store: IndexStore) -> None:
    """Chunks for different repos in one session keep their repo_id.

    Regression guard for the buffer-attribution bug: the buffer
    once tracked a single scalar repo_id and stamped every chunk
    with the last one seen.  Here a single session interleaves
    chunks for two repos; each must surface under its own repo.
    """
    with store.session() as ws:
        ws.add_chunk(make_chunk("r1_a", path="a.py", blob="b1a"))
        ws.add_chunk(make_chunk("r2_a", path="a.py", blob="b2a", repo_id=2))
        ws.add_chunk(make_chunk("r1_b", path="b.py", blob="b1b"))
        ws.insert_snapshots(
            [make_snap("head", "a.py", "b1a"), make_snap("head", "b.py", "b1b")],
            repo_id=1,
        )
        ws.insert_snapshots([make_snap("head", "a.py", "b2a")], repo_id=2)

    r1_ids = {c.id for c in store.get_chunks("head", repo_id=1)}
    r2_ids = {c.id for c in store.get_chunks("head", repo_id=2)}
    assert r1_ids == {"r1_a", "r1_b"}
    assert r2_ids == {"r2_a"}


# ── Sweep behaviour ─────────────────────────────────────────────────


def test_sweep_cleans_orphans(store: IndexStore) -> None:
    """Explicit sweep() removes crash residue."""
    # Simulate a crashed build: snapshot without mark_indexed,
    # but with a completed build already present so the repo
    # doesn't look like a first-build-in-progress.
    with store.session() as ws:
        ws.add_chunk(make_chunk("good"))
        ws.insert_snapshots([make_snap("good_sha", "f.py", "blob_good")], repo_id=1)
        ws.mark_indexed(1, "good_sha")

    with store.session() as ws:
        ws.add_chunk(make_chunk("orphan"))
        ws.insert_snapshots([make_snap("crashed_sha", "f.py", "blob_orphan")], repo_id=1)
        # No mark_indexed — simulates crash.

    with store.session() as ws:
        ws.sweep()

    assert store.count_snapshots_for_commit(1, "crashed_sha") == 0
    assert store.count_snapshots_for_commit(1, "good_sha") > 0


def test_sweep_skips_first_build(store: IndexStore) -> None:
    """sweep() does NOT delete data for a repo with no completed builds.

    A repo mid-first-build has snapshots but no mark_indexed.
    That's not crash residue — it's an in-progress build.
    """
    with store.session() as ws:
        ws.add_chunk(make_chunk("wip"))
        ws.insert_snapshots([make_snap("wip_sha", "f.py", "blob_wip")], repo_id=1)
        # No mark_indexed — first build in progress.

    with store.session() as ws:
        ws.sweep()

    assert store.count_snapshots_for_commit(1, "wip_sha") > 0


# ── FTS rebuild ──────────────────────────────────────────────────────


def test_fts_rebuilt_after_chunk_insert(store: IndexStore) -> None:
    """Session that inserts chunks rebuilds FTS on exit."""
    with store.session() as ws:
        ws.add_chunk(make_chunk("searchable"))
        ws.insert_snapshots([make_snap("c1", "f.py", "blob_searchable")], repo_id=1)
        ws.mark_indexed(1, "c1")

    results = store.match_fulltext("c1", "searchable", repo_id=1)
    assert len(results) > 0


# ── Chunk buffering ──────────────────────────────────────────────────


def test_buffer_flushes_at_batch_size(store: IndexStore, monkeypatch: pytest.MonkeyPatch) -> None:
    """Chunks are flushed when the buffer reaches batch size."""
    monkeypatch.setattr(config, "insert_batch_size", 5)

    with store.session() as ws:
        for i in range(12):
            ws.add_chunk(make_chunk(f"c{i}", path=f"f{i}.py"))
        ws.insert_snapshots(
            [make_snap("head", f"f{i}.py", f"blob_c{i}") for i in range(12)],
            repo_id=1,
        )

    chunks = store.get_chunks("head", repo_id=1)
    assert len(chunks) == 12


def test_buffer_flushes_before_dependent_ops(store: IndexStore) -> None:
    """Buffer is flushed before replace_snapshots (which depends on chunks)."""
    with store.session() as ws:
        ws.add_chunk(make_chunk("a", blob="b1", path="a.py"))
        # replace_snapshots triggers _flush_chunks internally.
        ws.replace_snapshots("c1", [make_snap("c1", "a.py", "b1")], repo_id=1)

    chunks = store.get_chunks("c1", repo_id=1)
    assert len(chunks) == 1


# ── Empty operations ─────────────────────────────────────────────────


def test_empty_operations_are_noops(store: IndexStore) -> None:
    """Empty lists don't cause errors."""
    with store.session() as ws:
        ws.insert_snapshots([], repo_id=1)
        ws.insert_edges([], "c1", repo_id=1)
        ws.update_embeddings([], [], repo_id=1)
        ws.delete_chunks_for_blobs(set(), repo_id=1)


# ── register_repo ────────────────────────────────────────────────────


@parametrize_with_cases("scenario", cases=".cases_store_repos")
def test_register_repo(scenario: RepoSequence, store: IndexStore) -> None:
    with store.session() as ws:
        for path, expected_id in zip(scenario.calls, scenario.expected_ids, strict=True):
            assert ws.register_repo(path) == expected_id
    assert store.list_repos() == scenario.expected_listing


# ── resolve_repo ──────────────────────────────────────────────────


def test_resolve_repo_returns_id(store: IndexStore) -> None:
    """resolve_repo returns the id for a registered repo."""
    with store.session() as ws:
        ws.register_repo("/r")
    assert store.resolve_repo("/r") == 1


def test_resolve_repo_raises_for_unknown(store: IndexStore) -> None:
    """resolve_repo raises RbtrError for an unregistered repo."""
    with pytest.raises(RbtrError, match="not registered"):
        store.resolve_repo("nonexistent")


# ── replace_snapshots / replace_edges ────────────────────────────────


def test_replace_snapshots_scoped_to_commit(store: IndexStore) -> None:
    """Replacing snapshots for one commit doesn't touch another."""
    with store.session() as ws:
        ws.add_chunk(make_chunk("a", blob="b1", path="a.py"))
        ws.add_chunk(make_chunk("b", blob="b2", path="b.py"))
        ws.insert_snapshots([make_snap("c1", "a.py", "b1")], repo_id=1)
        ws.insert_snapshots([make_snap("c2", "b.py", "b2")], repo_id=1)

    with store.session() as ws:
        ws.replace_snapshots("c1", [make_snap("c1", "a.py", "b1")], repo_id=1)

    assert store.count_snapshots_for_commit(1, "c2") == 1


def test_replace_edges_scoped_to_commit(store: IndexStore) -> None:
    """Replacing edges for one commit doesn't touch another."""
    e1 = Edge(source_id="a", target_id="b", kind=EdgeKind.IMPORTS)
    e2 = Edge(source_id="c", target_id="d", kind=EdgeKind.IMPORTS)

    with store.session() as ws:
        ws.insert_edges([e1], "c1", repo_id=1)
        ws.insert_edges([e2], "c2", repo_id=1)

    with store.session() as ws:
        ws.replace_edges("c1", [], repo_id=1)

    assert store.count_edges_for_commit(1, "c1") == 0
    assert store.count_edges_for_commit(1, "c2") == 1


# ── update_embeddings ────────────────────────────────────────────────


def test_update_embeddings_round_trip(store: IndexStore) -> None:
    """Embeddings and truncation flags are persisted correctly.

    Seeds two chunks, embeds one as truncated and one as not,
    then verifies both the embedding flag and the
    `embedding_truncated` column in the DB.
    """
    with store.session() as ws:
        ws.add_chunk(make_chunk("a"))
        ws.add_chunk(make_chunk("b", blob="blob_b", path="g.py"))
        ws.insert_snapshots(
            [
                make_snap("c1", "f.py", "blob_a"),
                make_snap("c1", "g.py", "blob_b"),
            ],
            repo_id=1,
        )

    # Before embedding — chunks have no embedding.
    chunks = store.get_chunks("c1", repo_id=1)
    assert all(not c.embedding for c in chunks)

    vec = [0.1, 0.2, 0.3]
    with store.session() as ws:
        ws.update_embeddings(["a", "b"], [vec, vec], repo_id=1, truncated=[True, False])

    # After embedding — both flagged as embedded.
    chunks = store.get_chunks("c1", repo_id=1)
    assert len(chunks) == 2
    assert all(c.embedding for c in chunks)

    # Verify truncation flags in the DB.
    rows = store._cursor.execute(
        "SELECT id, embedding_truncated FROM chunks ORDER BY id"
    ).fetchall()
    by_id = {row[0]: row[1] for row in rows}
    assert by_id["a"] is True
    assert by_id["b"] is False


def test_update_embeddings_rejects_mixed_dims(store: IndexStore) -> None:
    """A batch with vectors of differing lengths is rejected.

    All embeddings come from one model, so they must share a
    length; the staging frame enforces this.
    """
    with store.session() as ws:
        ws.add_chunk(make_chunk("a"))
        ws.add_chunk(make_chunk("b", blob="blob_b", path="g.py"))
        with pytest.raises(ValidationError):
            ws.update_embeddings(["a", "b"], [[0.1, 0.2, 0.3], [0.4, 0.5]], repo_id=1)


# ── delete_snapshots ─────────────────────────────────────────────────


def test_delete_snapshots_hides_chunks(
    store: IndexStore, math_func: TokenisedChunk, http_func: TokenisedChunk
) -> None:
    """Deleting snapshots hides chunks at a ref without touching chunks."""
    with store.session() as ws:
        ws.add_chunk(math_func)
        ws.add_chunk(http_func)
        ws.insert_snapshots(
            [
                Snapshot(
                    commit_sha="head",
                    file_path=math_func.file_path,
                    blob_sha=math_func.blob_sha,
                ),
                Snapshot(
                    commit_sha="head",
                    file_path=http_func.file_path,
                    blob_sha=http_func.blob_sha,
                ),
            ],
            repo_id=1,
        )
    assert len(store.get_chunks("head", repo_id=1)) == 2

    with store.session() as ws:
        ws.delete_snapshots("head", repo_id=1)

    assert store.get_chunks("head", repo_id=1) == []
    assert store.has_blob(math_func.blob_sha, repo_id=1) is True
    assert store.has_blob(http_func.blob_sha, repo_id=1) is True
