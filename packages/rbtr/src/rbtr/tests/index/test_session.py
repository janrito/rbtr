"""Behavioural tests for WriteSession mechanics.

Verifies the transactional contract: commit-on-exit,
rollback-on-exception, guard rails, sweep behaviour,
FTS rebuild, chunk buffering, and write method round-trips.
"""

from __future__ import annotations

import duckdb
import pytest
from dataframely.exc import ValidationError
from pytest_cases import parametrize_with_cases

from rbtr.config import config
from rbtr.domain.models import Edge, EdgeKind, FileSnapshot, Repo, SnapshotRef
from rbtr.errors import RepoNotFoundError
from rbtr.index.index import Index
from rbtr.index.staging import TokenisedChunk

from .cases_index_repos import RepoSequence
from .cases_index_watched_refs import WatchedRefScenario
from .conftest import make_chunk, make_snap, seed_index


def test_clean_exit_commits(index: Index) -> None:
    """Data written in a session is visible after clean exit."""
    with index.session() as ws:
        repo = ws.register_repo("/repo")
        ws.add_chunk(make_chunk("a"))
        ws.insert_snapshots([make_snap("c1", "f.py", "blob_a")], repo=repo)

    chunks = index.chunks(at=repo.at("c1"))
    assert any(c.name == "a" for c in chunks)


def test_exception_rolls_back(index: Index) -> None:
    """Data written in a session is invisible after an exception."""

    with index.session() as ws:
        repo = ws.register_repo("/repo")

    def _failing_session() -> None:
        with index.session() as ws:
            ws.add_chunk(make_chunk("a"))
            ws.insert_snapshots([make_snap("c1", "f.py", "blob_a")], repo=repo)
            msg = "boom"
            raise RuntimeError(msg)

    with pytest.raises(RuntimeError):
        _failing_session()

    chunks = index.chunks(at=repo.at("c1"))
    assert chunks == []


def test_write_outside_session_raises(index: Index) -> None:
    """Calling a write method without an active session raises."""
    ws = index.session()
    # ws.__enter__ not called — no active transaction.
    with pytest.raises(RuntimeError, match="No active transaction"):
        ws.add_chunk(make_chunk("a"))


def test_chunks_cannot_be_committed_without_a_claim(index: Index) -> None:
    """A session inserting chunks and no `file_snapshots` rows is refused.

    The sweeps delete every chunk no repo claims, so a commit holding
    an unclaimed chunk leaves it for the next sweep to take.
    """
    with index.session() as ws:
        ws.register_repo("/repo")

    with pytest.raises(RuntimeError, match="file_snapshots"), index.session() as ws:
        ws.add_chunk(make_chunk("unclaimed"))

    assert index.count_orphan_chunks() == 0


def test_read_only_index_rejects_session() -> None:
    """A store created without writable=True rejects session()."""
    index = Index(writable=False)
    try:
        with pytest.raises(RuntimeError, match="writable"):
            index.session()
    finally:
        index.close()


def test_add_chunk_attributes_per_repo_in_one_session(index: Index) -> None:
    """Chunks for different repos in one session keep their repo_id.

    Regression guard for the buffer-attribution bug: the buffer
    once tracked a single scalar repo_id and stamped every chunk
    with the last one seen.  Here a single session interleaves
    chunks for two repos; each must surface under its own repo.
    """
    with index.session() as ws:
        repo = ws.register_repo("/repo")
        repo2 = ws.register_repo("/repo2")
        ws.add_chunk(make_chunk("r1_a", path="a.py", blob="b1a"))
        ws.add_chunk(make_chunk("r2_a", path="a.py", blob="b2a"))
        ws.add_chunk(make_chunk("r1_b", path="b.py", blob="b1b"))
        ws.insert_snapshots(
            [make_snap("head", "a.py", "b1a"), make_snap("head", "b.py", "b1b")],
            repo=repo,
        )
        ws.insert_snapshots([make_snap("head", "a.py", "b2a")], repo=repo2)

    r1_names = {c.name for c in index.chunks(at=repo.at("head"))}
    r2_names = {c.name for c in index.chunks(at=repo2.at("head"))}
    assert r1_names == {"r1_a", "r1_b"}
    assert r2_names == {"r2_a"}


def test_sweep_cleans_orphans(index: Index) -> None:
    """Explicit sweep() removes crash residue."""
    # Simulate a crashed build: snapshot without mark_indexed,
    # but with a completed build already present so the repo
    # doesn't look like a first-build-in-progress.
    with index.session() as ws:
        repo = ws.register_repo("/repo")
        ws.add_chunk(make_chunk("good"))
        ws.insert_snapshots([make_snap("good_sha", "f.py", "blob_good")], repo=repo)
        ws.mark_indexed(at=repo.at("good_sha"))

    with index.session() as ws:
        ws.add_chunk(make_chunk("orphan"))
        ws.insert_snapshots([make_snap("crashed_sha", "f.py", "blob_orphan")], repo=repo)
        # No mark_indexed — simulates crash.

    with index.session() as ws:
        ws.sweep()

    assert index.count_file_snapshots(at=repo.at("crashed_sha")) == 0
    assert index.count_file_snapshots(at=repo.at("good_sha")) > 0


def test_sweep_skips_first_build(index: Index) -> None:
    """sweep() does NOT delete data for a repo with no completed builds.

    A repo mid-first-build has snapshots but no mark_indexed.
    That's not crash residue — it's an in-progress build.
    """
    with index.session() as ws:
        repo = ws.register_repo("/repo")
        ws.add_chunk(make_chunk("wip"))
        ws.insert_snapshots([make_snap("wip_sha", "f.py", "blob_wip")], repo=repo)
        # No mark_indexed — first build in progress.

    with index.session() as ws:
        ws.sweep()

    assert index.count_file_snapshots(at=repo.at("wip_sha")) > 0


def test_fts_rebuilt_after_chunk_insert(index: Index) -> None:
    """Session that inserts chunks rebuilds FTS on exit."""
    with index.session() as ws:
        repo = ws.register_repo("/repo")
        ws.add_chunk(make_chunk("searchable"))
        ws.insert_snapshots([make_snap("c1", "f.py", "blob_searchable")], repo=repo)
        ws.mark_indexed(at=repo.at("c1"))

    results = index.fulltext_matches("searchable", within=[repo.at("c1")])
    assert len(results) > 0


def test_buffer_flushes_at_batch_size(index: Index, monkeypatch: pytest.MonkeyPatch) -> None:
    """Chunks are flushed when the buffer reaches batch size."""
    monkeypatch.setattr(config, "insert_batch_size", 5)

    with index.session() as ws:
        repo = ws.register_repo("/repo")
        for i in range(12):
            ws.add_chunk(make_chunk(f"c{i}", path=f"f{i}.py"))
        ws.insert_snapshots(
            [make_snap("head", f"f{i}.py", f"blob_c{i}") for i in range(12)],
            repo=repo,
        )

    chunks = index.chunks(at=repo.at("head"))
    assert len(chunks) == 12


def test_buffer_flushes_before_dependent_ops(index: Index) -> None:
    """Buffer is flushed before replace_snapshots (which depends on chunks)."""
    with index.session() as ws:
        repo = ws.register_repo("/repo")
        ws.add_chunk(make_chunk("a", blob="b1", path="a.py"))
        # replace_snapshots triggers _flush_chunks internally.
        ws.replace_snapshots([make_snap("c1", "a.py", "b1")], at=repo.at("c1"))

    chunks = index.chunks(at=repo.at("c1"))
    assert len(chunks) == 1


def test_empty_operations_are_noops(index: Index) -> None:
    """Empty lists don't cause errors."""
    with index.session() as ws:
        repo = ws.register_repo("/repo")
        ws.insert_snapshots([], repo=repo)
        ws.insert_edges([], at=repo.at("c1"))
        ws.update_embeddings([], [])
        ws.delete_chunks_for_blobs(set(), file_language="")


@parametrize_with_cases("scenario", cases=".cases_index_repos")
def test_register_repo(scenario: RepoSequence, index: Index) -> None:
    with index.session() as ws:
        for path, expected_id in zip(scenario.calls, scenario.expected_ids, strict=True):
            assert ws.register_repo(path).repo_id == expected_id
    listing = [(r.repo_id, r.repo_path) for r in index.repos()]
    assert listing == scenario.expected_listing


@parametrize_with_cases("scenario", cases=".cases_index_watched_refs")
def test_watched_refs(scenario: WatchedRefScenario, index: Index) -> None:
    repos: dict[str, Repo] = {}
    with index.session() as ws:
        for key, path in scenario.repos.items():
            repos[key] = ws.register_repo(path)
        for key, op, refs in scenario.ops:
            if op == "add":
                ws.add_watched_refs(repos[key], refs)
            else:
                ws.remove_watched_refs(repos[key], refs)
    for key, expected in scenario.expected.items():
        assert index.watched_refs(repos[key]) == expected


def test_repo_returns_the_registered_repo(index: Index) -> None:
    """`repo` returns the repo registered at a path."""
    with index.session() as ws:
        registered = ws.register_repo("/r")
    assert index.repo("/r") == registered


def test_repo_raises_for_unknown(index: Index) -> None:
    """`repo` raises `RepoNotFoundError`, coded `repo_not_found`."""
    with pytest.raises(RepoNotFoundError, match="not registered") as raised:
        index.repo("nonexistent")
    assert raised.value.error_code == "repo_not_found"


def test_replace_snapshots_scoped_to_commit(index: Index) -> None:
    """Replacing snapshots for one commit doesn't touch another."""
    with index.session() as ws:
        repo = ws.register_repo("/repo")
        ws.add_chunk(make_chunk("a", blob="b1", path="a.py"))
        ws.add_chunk(make_chunk("b", blob="b2", path="b.py"))
        ws.insert_snapshots([make_snap("c1", "a.py", "b1")], repo=repo)
        ws.insert_snapshots([make_snap("c2", "b.py", "b2")], repo=repo)

    with index.session() as ws:
        ws.replace_snapshots([make_snap("c1", "a.py", "b1")], at=repo.at("c1"))

    assert index.count_file_snapshots(at=repo.at("c2")) == 1


def test_replace_edges_scoped_to_commit(index: Index) -> None:
    """Replacing edges for one commit doesn't touch another."""
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
        kind=EdgeKind.IMPORTS,
        source_path="src/c.py",
        target_path="src/d.py",
    )

    with index.session() as ws:
        repo = ws.register_repo("/repo")
        ws.insert_edges([e1], at=repo.at("c1"))
        ws.insert_edges([e2], at=repo.at("c2"))

    with index.session() as ws:
        ws.replace_edges([], at=repo.at("c1"))

    assert index.count_edges(at=repo.at("c1")) == 0
    assert index.count_edges(at=repo.at("c2")) == 1


def test_update_embeddings_round_trip(index: Index) -> None:
    """Embeddings and truncation flags are persisted correctly.

    Seeds two chunks, embeds one as truncated and one as not,
    then verifies both the embedding flag and the
    `embedding_truncated` column in the DB.
    """
    with index.session() as ws:
        repo = ws.register_repo("/repo")
        chunk_a = make_chunk("a")
        chunk_b = make_chunk("b", blob="blob_b", path="g.py")
        ws.add_chunk(chunk_a)
        ws.add_chunk(chunk_b)
        ws.insert_snapshots(
            [
                make_snap("c1", "f.py", "blob_a"),
                make_snap("c1", "g.py", "blob_b"),
            ],
            repo=repo,
        )

    # Before embedding — chunks have no embedding.
    chunks = index.chunks(at=repo.at("c1"))
    assert all(not c.has_embedding for c in chunks)

    vec = [0.1, 0.2, 0.3]
    with index.session() as ws:
        ws.update_embeddings([chunk_a.id, chunk_b.id], [vec, vec], truncated=[True, False])

    # After embedding — both flagged as embedded.
    chunks = index.chunks(at=repo.at("c1"))
    assert len(chunks) == 2
    assert all(c.has_embedding for c in chunks)

    # Verify truncation flags in the DB.
    rows = index._cursor.execute(
        "SELECT id, embedding_truncated FROM chunks ORDER BY id"
    ).fetchall()
    by_id = {row[0]: row[1] for row in rows}
    assert by_id[chunk_a.id] is True
    assert by_id[chunk_b.id] is False


def test_update_embeddings_rejects_mixed_dims(index: Index) -> None:
    """A batch with vectors of differing lengths is rejected.

    All embeddings come from one model, so they must share a
    length; the staging frame enforces this.
    """
    with index.session() as ws:
        chunk_a = make_chunk("a")
        chunk_b = make_chunk("b", blob="blob_b", path="g.py")
        repo = ws.register_repo("/repo")
        ws.add_chunk(chunk_a)
        ws.add_chunk(chunk_b)
        ws.insert_snapshots(
            [make_snap("head", c.file_path, c.blob_sha) for c in (chunk_a, chunk_b)], repo=repo
        )
        with pytest.raises(ValidationError):
            ws.update_embeddings(["a", "b"], [[0.1, 0.2, 0.3], [0.4, 0.5]])


def test_delete_snapshots_hides_chunks(
    index: Index, math_func: TokenisedChunk, http_func: TokenisedChunk
) -> None:
    """Deleting snapshots hides chunks at a ref without touching chunks."""
    with index.session() as ws:
        repo = ws.register_repo("/repo")
        ws.add_chunk(math_func)
        ws.add_chunk(http_func)
        ws.insert_snapshots(
            [
                FileSnapshot(
                    snapshot_sha="head",
                    file_path=math_func.file_path,
                    blob_sha=math_func.blob_sha,
                ),
                FileSnapshot(
                    snapshot_sha="head",
                    file_path=http_func.file_path,
                    blob_sha=http_func.blob_sha,
                ),
            ],
            repo=repo,
        )
    assert len(index.chunks(at=repo.at("head"))) == 2

    with index.session() as ws:
        ws.delete_snapshots(at=repo.at("head"))

    assert index.chunks(at=repo.at("head")) == []
    assert index.blob_is_current(math_func.blob_sha, "", {"": 1}) is True
    assert index.blob_is_current(http_func.blob_sha, "", {"": 1}) is True


def test_chunk_with_inverted_span_is_rejected(index: Index, head_ref: SnapshotRef) -> None:
    """A span that ends before it starts does not reach the table."""
    inverted = make_chunk("a").model_copy(update={"line_start": 7, "line_end": 6})
    with pytest.raises(duckdb.ConstraintException):
        seed_index(index, [inverted], head_ref)
