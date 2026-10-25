"""Schema DDL, index files, version handling, and DB lifecycle tests.

Covers:
- The registry and the project files it names
- Schema version mismatch recovery
- Embedding version/model change
- DB reopen, dual-open
- FTS persistence across reopen
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import duckdb
import pygit2
import pytest
from pytest_cases import fixture, parametrize_with_cases
from pytest_mock import MockerFixture

from rbtr.config import config, index_dir_of
from rbtr.domain.models import FileSnapshot
from rbtr.errors import (
    IndexLockedError,
    IndexNotBuiltError,
    IndexSchemaTooNewError,
    RepoNotFoundError,
)
from rbtr.git import normalise_repo_path
from rbtr.index.constants import EMBEDDING_FORMAT_VERSION
from rbtr.index.index import Index
from rbtr.index.staging import TokenisedChunk

from .cases_index_versioning import VersioningScenario
from .conftest import make_chunk, make_snap


def test_registry_and_project_file_hold_their_tables(tmp_path: Path) -> None:
    """The registry holds the repos; a registered repo's file, in `index_dir_of` the registry, holds its data."""
    index = Index(tmp_path / "index.duckdb", writable=True)
    index.register("/repo")
    index.close()
    query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
    with duckdb.connect(str(tmp_path / "index.duckdb"), read_only=True) as con:
        registry = {row[0] for row in con.sql(query).fetchall()}
    (project_file,) = index_dir_of(tmp_path / "index.duckdb").glob("*.duckdb")
    with duckdb.connect(str(project_file), read_only=True) as con:
        data = {row[0] for row in con.sql(query).fetchall()}
    assert {"meta", "projects", "repos"} <= registry
    assert {"meta", "file_snapshots", "chunks", "edges", "indexed_snapshots"} <= data
    assert "repos" not in data


@fixture
@parametrize_with_cases("scenario", cases=".cases_index_versioning")
def reopened(
    scenario: VersioningScenario,
    tmp_path: Path,
    mocker: MockerFixture,
) -> Iterator[tuple[Index, VersioningScenario]]:
    path = tmp_path / "index.duckdb"
    before = scenario.before

    if before.create_bare_file:
        con = duckdb.connect(str(path))
        con.execute("CREATE TABLE dummy (x INT)")
        con.close()
    else:
        seed = Index(path, writable=True)
        registered = seed.register("/repo")
        (project_file,) = (path.parent / "index").glob("*.duckdb")
        with seed.session(registered) as ws:
            for chunk in before.seeded_chunks:
                tc = (
                    chunk
                    if isinstance(chunk, TokenisedChunk)
                    else TokenisedChunk(**chunk.model_dump())
                )
                ws.add_chunk(tc)
            if before.seeded_chunks:
                ws.insert_snapshots(
                    [
                        FileSnapshot(
                            snapshot_sha="head",
                            file_path=c.file_path,
                            blob_sha=c.blob_sha,
                        )
                        for c in before.seeded_chunks
                    ],
                )
            if before.seeded_embeddings:
                ids = list(before.seeded_embeddings.keys())
                vecs = list(before.seeded_embeddings.values())
                ws.update_embeddings(ids, vecs)
        seed.close()

        # The schema version is the registry's; the embedding meta, the file's.
        updates: list[tuple[Path, str, str]] = []
        if before.schema_version != "":
            updates.append((path, "schema_version", str(before.schema_version)))
        if before.embedding_model is not None:
            updates.append((project_file, "embedding_model", before.embedding_model))
        if before.embedding_version is not None:
            updates.append((project_file, "embedding_version", before.embedding_version))
        for file, key, value in updates:
            con = duckdb.connect(str(file))
            con.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", [key, value])
            con.close()

    if scenario.config_embedding_model is not None:
        mocker.patch(
            "rbtr.index.writer.config.embedding_model",
            scenario.config_embedding_model,
        )

    index = Index(path, writable=True)
    try:
        yield index, scenario
    finally:
        index.close()


def test_chunks_survive_matches_scenario(
    reopened: tuple[Index, VersioningScenario],
) -> None:
    index, scenario = reopened
    if not scenario.before.seeded_chunks:
        return
    if scenario.expected_chunks_survive:
        [repo] = index.repos()
        assert index.chunks(at=repo.at("head")), "expected chunks to survive"
    else:
        assert index.repos() == [], "expected the index to be replaced"
        projects = Path(str(index.db_path)).parent / "index"
        assert list(projects.iterdir()) == [], "expected project files deleted"


def test_embeddings_survive_matches_scenario(
    reopened: tuple[Index, VersioningScenario],
) -> None:
    index, scenario = reopened
    if not scenario.before.seeded_embeddings:
        return
    if not scenario.expected_chunks_survive:
        return
    [repo] = index.repos()
    chunks = index.chunks(at=repo.at("head"))
    any_embedded = any(c.has_embedding for c in chunks)
    if scenario.expected_embeddings_survive:
        assert any_embedded, "expected some embedding to survive"
    else:
        assert not any_embedded, "expected embeddings to be cleared"


def test_embedding_meta_stored_after_change(
    reopened: tuple[Index, VersioningScenario],
) -> None:
    index, scenario = reopened
    if not scenario.expected_meta_stored:
        return
    with index.reader(index.repo("/repo")) as cur:
        rows = cur.sql(
            "SELECT key, value FROM meta WHERE key IN ('embedding_model', 'embedding_version')"
        ).fetchall()
    stored = {str(r[0]): str(r[1]) for r in rows}
    assert stored["embedding_model"] == config.embedding_model
    assert stored["embedding_version"] == str(EMBEDDING_FORMAT_VERSION)


def _seed_one_chunk(path: Path) -> None:
    """Seed *path* with a single indexed chunk at the current schema."""
    seed = Index(path, writable=True)
    repo = seed.register("/repo")
    with seed.session(repo) as ws:
        ws.add_chunk(make_chunk("c1"))
        ws.insert_snapshots(
            [FileSnapshot(snapshot_sha="head", file_path="f.py", blob_sha="blob_c1")],
        )
    seed.close()


def test_reading_a_db_rbtr_never_built_says_it_is_not_built(tmp_path: Path) -> None:
    """A read-only open of a DB with no index in it is `IndexNotBuiltError`.

    Only a writable store builds the schema, so a read that went on would
    fail on the first missing table.
    """
    with pytest.raises(IndexNotBuiltError, match="rbtr watch"):
        Index(tmp_path / "index.duckdb")


def test_older_binary_refuses_and_preserves_index(tmp_path: Path, mocker: MockerFixture) -> None:
    """An rbtr older than the on-disk schema refuses to open and never wipes.

    Guards the cross-install flip-flop: an older install must not
    destroy an index a newer install wrote.
    """
    path = tmp_path / "index.duckdb"
    _seed_one_chunk(path)
    inode_before = path.stat().st_ino

    mocker.patch("rbtr.index.store.SCHEMA_VERSION", "2000.1.1")
    with pytest.raises(IndexSchemaTooNewError):
        Index(path, writable=True)

    assert path.exists()
    assert path.stat().st_ino == inode_before  # never unlinked/recreated


def test_a_stale_index_is_replaced_by_an_empty_one(tmp_path: Path) -> None:
    """A newer rbtr replaces a stale index whole, and its files with it.

    Deleting rows does not shrink a DuckDB file, so the registry is a new
    file and the old project files are deleted.  A leftover too large to
    stand for an empty index would mean the old file was kept.
    """
    path = tmp_path / "index.duckdb"
    _seed_one_chunk(path)
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE padding AS SELECT md5(range::VARCHAR) AS x FROM range(200000)")
    con.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', '2000.1.1')")
    con.close()
    size_before = path.stat().st_size

    index = Index(path, writable=True)  # newer code -> replace
    try:
        assert index.repos() == []
        assert list((path.parent / "index").iterdir()) == []
    finally:
        index.close()
    assert path.stat().st_size < size_before / 4


def test_second_open_in_process_is_locked_out(tmp_path: Path) -> None:
    """A second index on an open one is refused and the first keeps working.

    DuckDB attaches a file to one instance per process, so the second index
    meets the index held, as one in another process would.
    """
    db_path = tmp_path / "index.duckdb"
    _seed_one_chunk(db_path)
    index = Index(db_path, writable=True)
    [repo] = index.repos()
    try:
        with pytest.raises(IndexLockedError):
            Index(db_path)
        assert index.chunks(at=repo.at("head"))
    finally:
        index.close()


def test_fts_persists_across_reopen(tmp_path: Path) -> None:
    """FTS index survives close + reopen without rebuild."""
    db_path = tmp_path / "index.duckdb"
    chunk = make_chunk("persist_1", name="persist_check")

    index1 = Index(db_path, writable=True)
    repo = index1.register("/repo")
    with index1.session(repo) as ws:
        ws.add_chunk(chunk)
        ws.insert_snapshots(
            [FileSnapshot(snapshot_sha="head", file_path=chunk.file_path, blob_sha=chunk.blob_sha)],
        )
    ref = repo.at("head")
    assert len(index1.fulltext_matches("persist", within=[ref], top_k=5)) == 1
    index1.close()

    index2 = Index(db_path, writable=True)
    assert len(index2.fulltext_matches("persist", within=[ref], top_k=5)) == 1
    index2.close()


def test_checkouts_of_one_repository_share_a_project_file(
    git_repo: pygit2.Repository,
    linked_worktree: pygit2.Repository,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """A linked worktree joins its main checkout's project; other paths do not.

    Keyed by git's common directory, so a new worktree reuses the file its
    main checkout filled.  A path git does not know is a project of its own.
    """
    index = Index(tmp_path_factory.mktemp("data") / "index.duckdb", writable=True)
    main = index.register(normalise_repo_path(git_repo.workdir))
    worktree = index.register(normalise_repo_path(linked_worktree.workdir))
    elsewhere = index.register("/not/a/repository")
    try:
        files = {}
        for repo in (main, worktree, elsewhere):
            with index.reader(repo) as cur:
                files[repo] = cur.sql("SELECT current_database()").fetchall()
        assert files[main] == files[worktree]
        assert files[elsewhere] != files[main]
        projects = Path(str(index.db_path)).parent / "index"
        assert len(list(projects.glob("*.duckdb"))) == 2
    finally:
        index.close()


def test_opening_deletes_files_the_registry_does_not_name(tmp_path: Path) -> None:
    """A build or retirement a crash interrupted leaves files nothing names.

    Opening the index deletes them, and the copies the compaction that once
    renamed `index.duckdb.compact-*` over the index left beside it.
    """
    path = tmp_path / "index.duckdb"
    _seed_one_chunk(path)
    projects = tmp_path / "index"
    (named,) = projects.iterdir()
    strays = [projects / "stray.duckdb", projects / "stray.duckdb.wal"]
    for stray in strays:
        stray.write_bytes(b"left by a crash")
    old_copy = tmp_path / "index.duckdb.compact-abc"
    old_copy.write_bytes(b"left by the old compaction")
    old_spill = tmp_path / "index.duckdb.compact-abc.tmp"
    old_spill.mkdir()

    index = Index(path, writable=True)
    [repo] = index.repos()
    try:
        assert list(projects.iterdir()) == [named]
        assert not old_copy.exists()
        assert not old_spill.exists()
        assert index.chunks(at=repo.at("head"))
    finally:
        index.close()


def test_forgetting_a_projects_last_checkout_deletes_its_file(tmp_path: Path) -> None:
    """The project's file is deleted with its last checkout, and the repo is unknown after.

    Other projects stay as they were.
    """
    index = Index(tmp_path / "index.duckdb", writable=True)
    kept = index.register("/kept")
    forgotten = index.register("/forgotten")
    with index.session(forgotten) as ws:
        ws.add_chunk(make_chunk("c", path="a.py", blob="b"))
        ws.insert_snapshots([make_snap("head", "a.py", "b")])
        ws.mark_indexed(snapshot_sha="head")
    query = "SELECT path FROM duckdb_databases() WHERE database_name = current_database()"
    with index.reader(kept) as cur:
        kept_file = Path(cur.sql(query).fetchall()[0][0]).name
    with index.reader(forgotten) as cur:
        forgotten_file = Path(cur.sql(query).fetchall()[0][0]).name

    index.forget(forgotten)

    assert index.repos() == [kept]
    with pytest.raises(RepoNotFoundError):
        index.repo("/forgotten")
    with pytest.raises(RepoNotFoundError):
        index.has_indexed(at=forgotten.at("head"))
    index.close()
    files = {p.name for p in (tmp_path / "index").iterdir()}
    assert kept_file in files
    assert forgotten_file not in files


def test_a_writable_open_adds_tables_an_index_lacks(tmp_path: Path) -> None:
    """A table the schema gained reaches an existing project file on open.

    An additive change needs no version bump, so a file written before it
    must get the table when it is next opened for writing.
    """
    path = tmp_path / "index.duckdb"
    _seed_one_chunk(path)
    (project_file,) = (tmp_path / "index").glob("*.duckdb")
    con = duckdb.connect(str(project_file))
    con.execute("DROP TABLE watched_refs")
    con.close()

    index = Index(path, writable=True)
    try:
        assert index.watched_refs(index.repo("/repo")) == []
    finally:
        index.close()
