"""Schema DDL, version migration, and DB lifecycle tests.

Covers:
- Multi-statement schema loading
- Schema version mismatch recovery
- Embedding version/model migration
- DB reopen, dual-open
- FTS persistence across reopen
- Search before FTS built
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import duckdb
import pytest
from pytest_cases import fixture, parametrize_with_cases
from pytest_mock import MockerFixture

from rbtr.config import config
from rbtr.errors import IndexSchemaTooNewError
from rbtr.index.constants import EMBEDDING_FORMAT_VERSION
from rbtr.index.models import Snapshot, TokenisedChunk
from rbtr.index.store import IndexStore

from .cases_store_versioning import VersioningScenario
from .conftest import make_chunk

# ── Multi-statement schema ──────────────────────────────────────────


def test_all_schema_tables_exist_after_open(tmp_path: Path) -> None:
    """Every table the full schema declares is present on open."""
    store = IndexStore(tmp_path / "index.duckdb", writable=True)
    expected = {
        "meta",
        "repos",
        "file_snapshots",
        "chunks",
        "edges",
        "indexed_commits",
    }
    tables = {
        row[0]
        for row in store._cursor.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()
    }
    assert expected <= tables


# ── Version migration ───────────────────────────────────────────────


@fixture
@parametrize_with_cases("scenario", cases=".cases_store_versioning")
def reopened(
    scenario: VersioningScenario,
    tmp_path: Path,
    mocker: MockerFixture,
) -> Iterator[tuple[IndexStore, VersioningScenario]]:
    path = tmp_path / "index.duckdb"
    before = scenario.before

    if before.create_bare_file:
        con = duckdb.connect(str(path))
        con.execute("CREATE TABLE dummy (x INT)")
        con.close()
    else:
        seed = IndexStore(path, writable=True)
        with seed.session() as ws:
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
                        Snapshot(
                            commit_sha="head",
                            file_path=c.file_path,
                            blob_sha=c.blob_sha,
                        )
                        for c in before.seeded_chunks
                    ],
                    repo_id=1,
                )
            if before.seeded_embeddings:
                ids = list(before.seeded_embeddings.keys())
                vecs = list(before.seeded_embeddings.values())
                ws.update_embeddings(ids, vecs)
        seed.close()

        updates: list[tuple[str, str]] = []
        if before.schema_version != "":
            updates.append(("schema_version", str(before.schema_version)))
        if before.embedding_model is not None:
            updates.append(("embedding_model", before.embedding_model))
        if before.embedding_version is not None:
            updates.append(("embedding_version", before.embedding_version))
        if updates:
            con = duckdb.connect(str(path))
            for key, value in updates:
                con.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", [key, value])
            con.close()

    if scenario.config_embedding_model is not None:
        mocker.patch(
            "rbtr.index.writer.config.embedding_model",
            scenario.config_embedding_model,
        )

    store = IndexStore(path, writable=True)
    try:
        yield store, scenario
    finally:
        store.close()


def test_chunks_survive_matches_scenario(
    reopened: tuple[IndexStore, VersioningScenario],
) -> None:
    store, scenario = reopened
    if not scenario.before.seeded_chunks:
        return
    chunks = store.get_chunks("head", repo_id=1)
    if scenario.expected_chunks_survive:
        assert chunks, "expected chunks to survive"
    else:
        assert not chunks, "expected chunks to be wiped"


def test_embeddings_survive_matches_scenario(
    reopened: tuple[IndexStore, VersioningScenario],
) -> None:
    store, scenario = reopened
    if not scenario.before.seeded_embeddings:
        return
    chunks = store.get_chunks("head", repo_id=1)
    if not chunks:
        assert not scenario.expected_chunks_survive
        return
    any_embedded = any(c.embedding for c in chunks)
    if scenario.expected_embeddings_survive:
        assert any_embedded, "expected some embedding to survive"
    else:
        assert not any_embedded, "expected embeddings to be cleared"


def test_embedding_meta_stored_after_change(
    reopened: tuple[IndexStore, VersioningScenario],
) -> None:
    store, scenario = reopened
    if not scenario.expected_meta_stored:
        return
    rows = store._cursor.execute(
        "SELECT key, value FROM meta WHERE key IN ('embedding_model', 'embedding_version')"
    ).fetchall()
    stored = {str(r[0]): str(r[1]) for r in rows}
    assert stored["embedding_model"] == config.embedding_model
    assert stored["embedding_version"] == str(EMBEDDING_FORMAT_VERSION)


# ── DB lifecycle ────────────────────────────────────────────────────


def _seed_one_chunk(path: Path) -> None:
    """Seed *path* with a single indexed chunk at the current schema."""
    seed = IndexStore(path, writable=True)
    with seed.session() as ws:
        ws.add_chunk(make_chunk("c1"))
        ws.insert_snapshots(
            [Snapshot(commit_sha="head", file_path="f.py", blob_sha="blob_c1")],
            repo_id=1,
        )
    seed.close()


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
        IndexStore(path, writable=True)

    assert path.exists()
    assert path.stat().st_ino == inode_before  # never unlinked/recreated


def test_schema_wipe_is_in_place_not_unlink(tmp_path: Path) -> None:
    """A newer rbtr wipes a stale index in place, never unlinking the file.

    An unlink-based wipe defeats DuckDB's exclusive lock (a second
    process opens a fresh DB at the same path); an in-place wipe keeps
    the same inode, so the lock keeps serialising writers.
    """
    path = tmp_path / "index.duckdb"
    _seed_one_chunk(path)
    con = duckdb.connect(str(path))
    con.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', '2000.1.1')")
    con.close()
    inode_before = path.stat().st_ino

    store = IndexStore(path, writable=True)  # newer code -> wipe in place
    try:
        assert path.stat().st_ino == inode_before, "wipe must not unlink/recreate the DB file"
        assert not store.get_chunks("head", repo_id=1), "stale data must be wiped"
    finally:
        store.close()


def test_second_open_against_same_db_succeeds(tmp_path: Path) -> None:
    """Opening the same DB twice in-process does not destroy it."""
    db_path = tmp_path / "index.duckdb"
    store = IndexStore(db_path)
    store2 = IndexStore(db_path)
    store2.close()
    store.close()


def test_fts_persists_across_reopen(tmp_path: Path) -> None:
    """FTS index survives close + reopen without rebuild."""
    db_path = tmp_path / "index.duckdb"
    chunk = make_chunk("persist_1", name="persist_check")

    store1 = IndexStore(db_path, writable=True)
    with store1.session() as ws:
        ws.add_chunk(chunk)
        ws.insert_snapshots(
            [Snapshot(commit_sha="head", file_path=chunk.file_path, blob_sha=chunk.blob_sha)],
            repo_id=1,
        )
    results1 = store1.match_fulltext("head", "persist", top_k=5, repo_id=1)
    assert len(results1) == 1
    store1.close()

    store2 = IndexStore(db_path, writable=True)
    results2 = store2.match_fulltext("head", "persist", top_k=5, repo_id=1)
    assert len(results2) == 1
    store2.close()
