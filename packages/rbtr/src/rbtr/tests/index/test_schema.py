"""Schema DDL, index files, version handling, and DB lifecycle tests.

Covers:
- The registry and the project files it names
- Migrating each file from an older schema version, or refusing to
- Replacing an index too old to migrate
- Embedding version/model change
- DB reopen, dual-open
- FTS persistence across reopen
"""

from __future__ import annotations

import importlib.resources
import itertools
from collections.abc import Iterator
from pathlib import Path

import duckdb
import pygit2
import pytest
import structlog
from pytest_cases import filters as ft, fixture, parametrize_with_cases
from pytest_mock import MockerFixture

from rbtr.config import config, index_dir_of
from rbtr.domain.models import FileSnapshot
from rbtr.errors import (
    IndexLockedError,
    IndexMigratingError,
    IndexNotBuiltError,
    RepoNotFoundError,
)
from rbtr.git import normalise_repo_path
from rbtr.index import load_sql
from rbtr.index.constants import EMBEDDING_FORMAT_VERSION, SCHEMA_VERSION
from rbtr.index.index import Index
from rbtr.index.migrations import BringForward, FileKind, Migrate, chain, parse_calver, run
from rbtr.index.staging import TokenisedChunk

from .cases_index_versioning import FileAfter, MigrationScenario, VersioningScenario
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
@parametrize_with_cases(
    "scenario", cases=".cases_index_versioning", filter=~ft.has_tag("migration")
)
def reopened(
    scenario: VersioningScenario,
    tmp_path: Path,
    chains: Path,
    mocker: MockerFixture,
) -> Iterator[tuple[Index, VersioningScenario]]:
    path = tmp_path / "index.duckdb"
    before = scenario.before

    if before.create_bare_file:
        with duckdb.connect(str(path)) as con:
            con.execute("CREATE TABLE dummy (x INT)")
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
            with duckdb.connect(str(file)) as con:
                con.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", [key, value])

    for name, sql in scenario.migrations.items():
        (chains / name).parent.mkdir(exist_ok=True)
        (chains / name).write_text(sql)
    if scenario.current is not None:
        mocker.patch("rbtr.index.store.SCHEMA_VERSION", scenario.current)
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
        assert not index.chunks(at=repo.at("head")).is_empty(), "expected chunks to survive"
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
    any_embedded = chunks["has_embedding"].any()
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


def test_an_index_under_a_path_with_a_quote_opens_and_reopens(tmp_path: Path) -> None:
    """A single quote in the index's path reaches DuckDB intact."""
    path = tmp_path / "o'brien" / "index.duckdb"
    _seed_one_chunk(path)

    index = Index(path, writable=True)
    try:
        [repo] = index.repos()
        assert len(index.chunks(at=repo.at("head"))) == 1
    finally:
        index.close()


def test_reading_a_db_rbtr_never_built_says_it_is_not_built(tmp_path: Path) -> None:
    """A read-only open of a DB with no index in it is `IndexNotBuiltError`.

    Only a writable store builds the schema, so a read that went on would
    fail on the first missing table.
    """
    with pytest.raises(IndexNotBuiltError, match="rbtr watch"):
        Index(tmp_path / "index.duckdb")


def test_an_index_older_than_migrations_is_replaced_by_an_empty_one(
    tmp_path: Path, log_output: structlog.testing.LogCapture
) -> None:
    """An index older than this rbtr can migrate is replaced whole, its files with it, and the open warns.

    Deleting rows does not shrink a DuckDB file, so the registry is a new
    file and the old project files are deleted.  A leftover too large to
    stand for an empty index would mean the old file was kept.  The index
    says what it replaced, for the daemon to record in discovery.
    """
    path = tmp_path / "index.duckdb"
    _seed_one_chunk(path)
    with duckdb.connect(str(path)) as con:
        con.execute("CREATE TABLE padding AS SELECT md5(range::VARCHAR) AS x FROM range(1000000)")
        con.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', '2000.1.1')"
        )
    size_before = path.stat().st_size

    index = Index(path, writable=True)  # too old to migrate -> replaced
    try:
        assert index.repos() == []
        assert list((path.parent / "index").iterdir()) == []
        assert index.replaced is not None
        assert index.replaced.stored == "2000.1.1"
    finally:
        index.close()
    assert path.stat().st_size < size_before / 4
    [warning] = [e for e in log_output.entries if e["event"] == "index_replaced"]
    assert warning["log_level"] == "warning"
    assert warning["stored"] == "2000.1.1"


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
        assert not index.chunks(at=repo.at("head")).is_empty()
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
        assert not index.chunks(at=repo.at("head")).is_empty()
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


@fixture
@parametrize_with_cases("scenario", cases=".cases_index_versioning", has_tag="migration")
def migration(
    scenario: MigrationScenario, tmp_path: Path, chains: Path, mocker: MockerFixture
) -> MigrationScenario:
    """The index at `tmp_path / "index.duckdb"`, its files at *scenario*'s stored versions.

    Its migrations are written under `chains`, and this rbtr's schema
    version is *scenario*'s `current`.
    """
    path = tmp_path / "index.duckdb"
    _seed_one_chunk(path)
    (project_file,) = (tmp_path / "index").glob("*.duckdb")
    with duckdb.connect(str(path)) as con:
        con.execute(
            "UPDATE meta SET value = ? WHERE key = 'schema_version'", [scenario.registry_stored]
        )
    with duckdb.connect(str(project_file)) as con:
        con.execute("DELETE FROM meta WHERE key = 'schema_version'")
        if scenario.project_stored is not None:
            con.execute("INSERT INTO meta VALUES ('schema_version', ?)", [scenario.project_stored])
    for name, sql in scenario.migrations.items():
        (chains / name).parent.mkdir(exist_ok=True)
        (chains / name).write_text(sql)
    mocker.patch("rbtr.index.store.SCHEMA_VERSION", scenario.current)
    return scenario


def test_opening_brings_each_file_forward_or_leaves_it(
    migration: MigrationScenario, tmp_path: Path
) -> None:
    """Opening runs each file's migrations, or refuses and leaves the files where they were.

    Either way each file records the version its schema has, and an
    index that opens still holds its rows.
    """
    path = tmp_path / "index.duckdb"
    (old_file,) = (tmp_path / "index").glob("*.duckdb")
    if migration.raises is not None:
        with pytest.raises(migration.raises):
            Index(path, writable=migration.writable)
    else:
        index = Index(path, writable=migration.writable)
        try:
            [repo] = index.repos()
            assert not index.chunks(at=repo.at("head")).is_empty()
            assert not index.fulltext_matches("c1", within=[repo.at("head")], top_k=5).is_empty()
            assert index.replaced is None
        finally:
            index.close()
    (project_file,) = (tmp_path / "index").glob("*.duckdb")
    assert (project_file != old_file) == migration.copied
    for file, expected in [
        (path, migration.registry_after),
        (project_file, migration.project_after),
    ]:
        with duckdb.connect(str(file), read_only=True) as con:
            version = con.sql("SELECT value FROM meta WHERE key = 'schema_version'").fetchall()
            added = con.sql(
                "SELECT table_name FROM information_schema.tables WHERE table_name LIKE 'added_%'"
            ).fetchall()
        assert (
            FileAfter(version[0][0] if version else None, frozenset(r[0] for r in added))
            == expected
        )


def test_a_deferred_migration_refuses_its_file_until_it_runs(
    tmp_path: Path, chains: Path, mocker: MockerFixture
) -> None:
    """`Migrate.LATER` leaves a project file that migrates on a copy unreadable until `migrate`."""
    path = tmp_path / "index.duckdb"
    _seed_one_chunk(path)
    (chains / "project").mkdir()
    (chains / "project" / f"{SCHEMA_VERSION}__2099.1.0.sql").write_text("SELECT 1;")
    mocker.patch("rbtr.index.store.SCHEMA_VERSION", "2099.1.0")

    index = Index(path, writable=True, migrate=Migrate.LATER)
    try:
        [repo] = index.repos()
        assert index.migrating
        with pytest.raises(IndexMigratingError):
            index.chunks(at=repo.at("head"))
        index.migrate()
        assert not index.migrating
        assert not index.chunks(at=repo.at("head")).is_empty()
    finally:
        index.close()


@pytest.mark.parametrize(("ratio", "compacted"), [(1000.0, False), (1.0, True)])
def test_a_copy_a_migration_grew_is_copied_again(
    ratio: float, compacted: bool, tmp_path: Path, chains: Path, mocker: MockerFixture
) -> None:
    """A migration that rebuilds a table leaves the old one's space; past the growth ratio the copy is compacted.

    Either way maintenance measures the file's growth from the data, the
    copy before the migration ran or the compacted one, not from the space
    the rebuild left: the uncompacted file is compacted by maintenance.
    """
    path = tmp_path / "index.duckdb"
    _seed_one_chunk(path)
    (old_file,) = (tmp_path / "index").glob("*.duckdb")
    with duckdb.connect(str(old_file)) as con:
        con.execute("CREATE TABLE padding AS SELECT md5(range::VARCHAR) AS x FROM range(200000)")
    old_bytes = old_file.stat().st_size
    (chains / "project").mkdir()
    (chains / "project" / f"{SCHEMA_VERSION}__2099.1.0.sql").write_text("""\
CREATE TABLE rebuilt AS SELECT * FROM padding;
DROP TABLE padding;
ALTER TABLE rebuilt RENAME TO padding;
""")
    mocker.patch("rbtr.index.store.SCHEMA_VERSION", "2099.1.0")
    mocker.patch("rbtr.index.store.config.compact_growth_ratio", ratio)

    Index(path, writable=True).close()

    (new_file,) = (tmp_path / "index").glob("*.duckdb")
    assert (new_file.stat().st_size < 1.5 * old_bytes) == compacted
    index = Index(path, writable=True)
    try:
        assert index.compact_grown(1.5) == (0 if compacted else 1)
    finally:
        index.close()


@pytest.mark.parametrize(
    ("kind", "schema"),
    [(FileKind.REGISTRY, "registry.sql"), (FileKind.PROJECT, "schema.sql")],
)
def test_migrating_a_kind_from_its_base_gives_the_schema_of_a_new_file(
    kind: FileKind, schema: str
) -> None:
    """A file migrated through every shipped migration has the schema a new file gets.

    The base is the schema as it stood at the oldest migration's start,
    kept beside the chain as `<start>.base.sql`.  With no migrations, the
    base is the kind's schema file itself.
    """
    migrations = chain(kind)
    directory = importlib.resources.files("rbtr.index") / "sql" / "migrations" / kind
    if migrations:
        base_file = directory / f"{migrations[0].start}.base.sql"
        assert base_file.is_file(), (
            f"the {kind} chain starts at {migrations[0].start} \
but ships no {base_file.name}: add it beside the chain, holding {schema} as it stood \
at {migrations[0].start}"
        )
        base = base_file.read_text(encoding="utf-8")
    else:
        base = load_sql(schema)
    with duckdb.connect() as migrated, duckdb.connect() as fresh:
        migrated.execute(base)
        start = migrations[0].start if migrations else SCHEMA_VERSION
        run(migrated, BringForward(start, tuple(migrations)), current=SCHEMA_VERSION)
        fresh.execute(load_sql(schema))
        # DuckDB writes each object's `CREATE` statement from its catalog entry.
        migrated_ddl, fresh_ddl = (
            sorted(
                row
                for catalog in ("duckdb_tables", "duckdb_indexes", "duckdb_sequences")
                for row in con.table_function(catalog).select("sql").fetchall()
            )
            for con in (migrated, fresh)
        )
        assert migrated_ddl == fresh_ddl


@pytest.mark.parametrize("kind", list(FileKind))
def test_a_shipped_chain_runs_unbroken_to_this_version(kind: FileKind) -> None:
    """Each migration starts where the one before it ends, and none ends past this version."""
    migrations = chain(kind)
    for earlier, later in itertools.pairwise(migrations):
        assert later.start == earlier.end
    for migration in migrations:
        assert parse_calver(migration.start) < parse_calver(migration.end)
        assert parse_calver(migration.end) <= parse_calver(SCHEMA_VERSION)
