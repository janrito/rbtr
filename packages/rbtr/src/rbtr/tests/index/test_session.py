"""Behavioural tests for WriteSession mechanics.

Verifies the transactional contract: commit-on-exit,
rollback-on-exception, guard rails, sweep behaviour,
FTS rebuild, chunk buffering, and write method round-trips.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import duckdb
import pytest
from dataframely.exc import ValidationError
from pytest_cases import parametrize_with_cases
from pytest_mock import MockerFixture

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
    repo = index.register("/repo")
    with index.session(repo) as ws:
        ws.add_chunk(make_chunk("a"))
        ws.insert_snapshots([make_snap("c1", "f.py", "blob_a")])

    chunks = index.chunks(at=repo.at("c1"))
    assert any(c.name == "a" for c in chunks)


def test_exception_rolls_back(index: Index) -> None:
    """Data written in a session is invisible after an exception."""

    repo = index.register("/repo")

    def _failing_session() -> None:
        with index.session(repo) as ws:
            ws.add_chunk(make_chunk("a"))
            ws.insert_snapshots([make_snap("c1", "f.py", "blob_a")])
            msg = "boom"
            raise RuntimeError(msg)

    with pytest.raises(RuntimeError):
        _failing_session()

    chunks = index.chunks(at=repo.at("c1"))
    assert chunks == []


def test_write_outside_session_raises(index: Index) -> None:
    """Calling a write method without an active session raises."""
    ws = index.session(index.register("/repo"))
    # ws.__enter__ not called — no active transaction.
    with pytest.raises(RuntimeError, match="No active transaction"):
        ws.add_chunk(make_chunk("a"))


def test_chunks_cannot_be_committed_without_a_claim(index: Index) -> None:
    """A session inserting chunks and no `file_snapshots` rows is refused.

    The sweeps delete every chunk no repo claims, so a commit holding
    an unclaimed chunk leaves it for the next sweep to take.
    """
    repo = index.register("/repo")

    with pytest.raises(RuntimeError, match="file_snapshots"), index.session(repo) as ws:
        ws.add_chunk(make_chunk("unclaimed"))

    assert index.count_orphan_chunks() == 0


@pytest.mark.parametrize(
    "write",
    [
        lambda index, repo: index.session(repo),
        lambda index, repo: index.register("/other"),
        lambda index, repo: index.forget(repo),
        lambda index, repo: index.compact(repo),
        lambda index, repo: index.compact_all(),
        lambda index, repo: index.compact_grown(1.0),
        lambda index, repo: index.set_missing_since(repo, None),
    ],
    ids=["session", "register", "forget", "compact", "compact_all", "compact_grown", "missing"],
)
def test_a_read_only_index_refuses_every_write(
    tmp_path: Path, write: Callable[[Index, Repo], object]
) -> None:
    """An index opened without writable=True raises before it writes anything."""
    db = tmp_path / "index.duckdb"
    seed = Index(db, writable=True)
    repo = seed.register("/repo")
    seed.close()
    index = Index(db, writable=False)
    try:
        with pytest.raises(RuntimeError, match="writable"):
            write(index, repo)
    finally:
        index.close()


def test_cleanup_removes_the_repos_unfinished_snapshots(index: Index) -> None:
    """`cleanup` removes the residue of a build that stopped part-way."""
    # Simulate a crashed build: a snapshot never marked indexed.
    repo = index.register("/repo")
    with index.session(repo) as ws:
        ws.add_chunk(make_chunk("good"))
        ws.insert_snapshots([make_snap("good_sha", "f.py", "blob_good")])
        ws.mark_indexed(snapshot_sha="good_sha")

    with index.session(repo) as ws:
        ws.add_chunk(make_chunk("orphan"))
        ws.insert_snapshots([make_snap("crashed_sha", "f.py", "blob_orphan")])
        # No mark_indexed — simulates crash.

    with index.session(repo) as ws:
        ws.cleanup()

    assert index.count_file_snapshots(at=repo.at("crashed_sha")) == 0
    assert index.count_file_snapshots(at=repo.at("good_sha")) > 0


def test_cleanup_leaves_a_sibling_checkouts_residue(project_index: Index) -> None:
    """`cleanup` clears its own repo's residue, not another checkout's.

    The sibling's unfinished snapshot still claims its chunk, so the
    chunk is not an orphan; the sibling's own clean-up removes both.
    """
    index = project_index
    main, worktree = index.repos()
    with index.session(main) as ws:
        ws.add_chunk(make_chunk("wip"))
        ws.insert_snapshots([make_snap("wip_sha", "f.py", "blob_wip")])
        # No mark_indexed: the build stopped here.

    with index.session(worktree) as ws:
        ws.cleanup()

    assert index.count_file_snapshots(at=main.at("wip_sha")) > 0
    assert index.count_orphan_chunks() == 0


def test_fts_rebuilt_after_chunk_insert(index: Index) -> None:
    """Session that inserts chunks rebuilds FTS on exit."""
    repo = index.register("/repo")
    with index.session(repo) as ws:
        ws.add_chunk(make_chunk("searchable"))
        ws.insert_snapshots([make_snap("c1", "f.py", "blob_searchable")])
        ws.mark_indexed(snapshot_sha="c1")

    results = index.fulltext_matches("searchable", within=[repo.at("c1")])
    assert len(results) > 0


def test_a_failed_fts_rebuild_keeps_the_previous_index(index: Index, mocker: MockerFixture) -> None:
    """A rebuild that fails part-way leaves the previous FTS index whole."""
    ref = index.register("/repo").at("c1")
    seed_index(index, [make_chunk("before", name="before_check")], ref)
    mocker.patch("rbtr.index.db._NEUTRALISE_FTS_IDF_SQL", "SELECT * FROM no_such_table")
    after = make_chunk("after", name="after_check", path="g.py")
    with pytest.raises(duckdb.CatalogException):
        seed_index(index, [after], ref)

    assert len(index.fulltext_matches("before", within=[ref])) == 1
    assert len(index.fulltext_matches("after", within=[ref])) == 0


def test_buffer_flushes_at_batch_size(index: Index, monkeypatch: pytest.MonkeyPatch) -> None:
    """Chunks are flushed when the buffer reaches batch size."""
    monkeypatch.setattr(config, "insert_batch_size", 5)

    repo = index.register("/repo")
    with index.session(repo) as ws:
        for i in range(12):
            ws.add_chunk(make_chunk(f"c{i}", path=f"f{i}.py"))
        ws.insert_snapshots(
            [make_snap("head", f"f{i}.py", f"blob_c{i}") for i in range(12)],
        )

    chunks = index.chunks(at=repo.at("head"))
    assert len(chunks) == 12


def test_buffer_flushes_before_dependent_ops(index: Index) -> None:
    """Buffer is flushed before replace_snapshots (which depends on chunks)."""
    repo = index.register("/repo")
    with index.session(repo) as ws:
        ws.add_chunk(make_chunk("a", blob="b1", path="a.py"))
        # replace_snapshots triggers _flush_chunks internally.
        ws.replace_snapshots([make_snap("c1", "a.py", "b1")], snapshot_sha="c1")

    chunks = index.chunks(at=repo.at("c1"))
    assert len(chunks) == 1


def test_empty_operations_are_noops(index: Index) -> None:
    """Empty lists don't cause errors."""
    repo = index.register("/repo")
    with index.session(repo) as ws:
        ws.insert_snapshots([])
        ws.insert_edges([], snapshot_sha="c1")
        ws.update_embeddings([], [])
        ws.delete_chunks_for_blobs(set(), file_language="")


@parametrize_with_cases("scenario", cases=".cases_index_repos")
def test_register(scenario: RepoSequence, index: Index) -> None:
    """A path is one repo, found again by `repo`; `repos` lists them as registered."""
    registered = [index.register(path) for path in scenario.calls]
    for path, repo in zip(scenario.calls, registered, strict=True):
        assert repo.repo_path == path
        assert index.repo(path) == repo
    assert len(set(registered)) == len(set(scenario.calls))
    assert [r.repo_path for r in index.repos()] == scenario.expected_paths


@parametrize_with_cases("scenario", cases=".cases_index_watched_refs")
def test_watched_refs(scenario: WatchedRefScenario, index: Index) -> None:
    repos = {key: index.register(path) for key, path in scenario.repos.items()}
    for key, op, refs in scenario.ops:
        with index.session(repos[key]) as ws:
            if op == "add":
                ws.add_watched_refs(refs)
            else:
                ws.remove_watched_refs(refs)
    for key, expected in scenario.expected.items():
        assert index.watched_refs(repos[key]) == expected


def test_repo_returns_the_registered_repo(index: Index) -> None:
    """`repo` returns the repo registered at a path."""
    registered = index.register("/r")
    assert index.repo("/r") == registered


def test_repo_follows_a_forgotten_path_registered_again(index: Index) -> None:
    """`repo` returns the new registration, not the forgotten one."""
    forgotten = index.repo(index.register("/r").repo_path)
    index.forget(forgotten)
    registered = index.register("/r")
    assert registered != forgotten
    assert index.repo("/r") == registered


def test_repo_raises_for_unknown(index: Index) -> None:
    """`repo` raises `RepoNotFoundError`, coded `repo_not_found`."""
    with pytest.raises(RepoNotFoundError, match="not registered") as raised:
        index.repo("nonexistent")
    assert raised.value.error_code == "repo_not_found"


def test_replace_snapshots_scoped_to_commit(index: Index) -> None:
    """Replacing snapshots for one commit doesn't touch another."""
    repo = index.register("/repo")
    with index.session(repo) as ws:
        ws.add_chunk(make_chunk("a", blob="b1", path="a.py"))
        ws.add_chunk(make_chunk("b", blob="b2", path="b.py"))
        ws.insert_snapshots([make_snap("c1", "a.py", "b1")])
        ws.insert_snapshots([make_snap("c2", "b.py", "b2")])

    with index.session(repo) as ws:
        ws.replace_snapshots([make_snap("c1", "a.py", "b1")], snapshot_sha="c1")

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

    repo = index.register("/repo")
    with index.session(repo) as ws:
        ws.insert_edges([e1], snapshot_sha="c1")
        ws.insert_edges([e2], snapshot_sha="c2")

    with index.session(repo) as ws:
        ws.replace_edges([], snapshot_sha="c1")

    assert index.count_edges(at=repo.at("c1")) == 0
    assert index.count_edges(at=repo.at("c2")) == 1


def test_update_embeddings_round_trip(index: Index) -> None:
    """Embeddings and truncation flags are persisted correctly.

    Seeds two chunks, embeds one as truncated and one as not,
    then verifies both the embedding flag and the
    `embedding_truncated` column in the DB.
    """
    repo = index.register("/repo")
    with index.session(repo) as ws:
        chunk_a = make_chunk("a")
        chunk_b = make_chunk("b", blob="blob_b", path="g.py")
        ws.add_chunk(chunk_a)
        ws.add_chunk(chunk_b)
        ws.insert_snapshots(
            [
                make_snap("c1", "f.py", "blob_a"),
                make_snap("c1", "g.py", "blob_b"),
            ],
        )

    # Before embedding — chunks have no embedding.
    chunks = index.chunks(at=repo.at("c1"))
    assert all(not c.has_embedding for c in chunks)

    vec = [0.1, 0.2, 0.3]
    with index.session(repo) as ws:
        ws.update_embeddings([chunk_a.id, chunk_b.id], [vec, vec], truncated=[True, False])

    # After embedding — both flagged as embedded.
    chunks = index.chunks(at=repo.at("c1"))
    assert len(chunks) == 2
    assert all(c.has_embedding for c in chunks)

    # Verify truncation flags in the DB.
    with index.reader(repo) as cur:
        rows = cur.sql("SELECT id, embedding_truncated FROM chunks ORDER BY id").fetchall()
    by_id = {row[0]: row[1] for row in rows}
    assert by_id[chunk_a.id] is True
    assert by_id[chunk_b.id] is False


def test_update_embeddings_rejects_mixed_dims(index: Index) -> None:
    """A batch with vectors of differing lengths is rejected.

    All embeddings come from one model, so they must share a
    length; the staging frame enforces this.
    """
    repo = index.register("/repo")
    with index.session(repo) as ws:
        chunk_a = make_chunk("a")
        chunk_b = make_chunk("b", blob="blob_b", path="g.py")
        ws.add_chunk(chunk_a)
        ws.add_chunk(chunk_b)
        ws.insert_snapshots(
            [make_snap("head", c.file_path, c.blob_sha) for c in (chunk_a, chunk_b)]
        )
        with pytest.raises(ValidationError):
            ws.update_embeddings(["a", "b"], [[0.1, 0.2, 0.3], [0.4, 0.5]])


def test_replacing_a_snapshot_with_nothing_hides_its_chunks(
    index: Index, math_func: TokenisedChunk, http_func: TokenisedChunk
) -> None:
    """An emptied snapshot hides its chunks without deleting them."""
    repo = index.register("/repo")
    with index.session(repo) as ws:
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
        )
    assert len(index.chunks(at=repo.at("head"))) == 2

    with index.session(repo) as ws:
        ws.replace_snapshots([], snapshot_sha="head")

    assert index.chunks(at=repo.at("head")) == []
    with index.session(repo) as ws:
        assert ws.blob_is_current(math_func.blob_sha, "", {"": 1}) is True
        assert ws.blob_is_current(http_func.blob_sha, "", {"": 1}) is True


def test_chunk_with_inverted_span_is_rejected(index: Index, head_ref: SnapshotRef) -> None:
    """A span that ends before it starts does not reach the table."""
    inverted = make_chunk("a").model_copy(update={"line_start": 7, "line_end": 6})
    with pytest.raises(duckdb.ConstraintException):
        seed_index(index, [inverted], head_ref)
