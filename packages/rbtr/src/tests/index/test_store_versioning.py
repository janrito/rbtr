"""Behavioural tests for DB schema and embedding-version recovery.

The fixture materialises a ``VersioningScenario`` to disk \u2014 real
file, real ``IndexStore(path)`` calls \u2014 because the recovery
code only runs when a non-``:memory:`` backend is used and a
file already exists in a specific state.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import duckdb
from pytest_cases import fixture, parametrize_with_cases
from pytest_mock import MockerFixture

from rbtr.index.store import IndexStore
from tests.index.case_store_versioning import VersioningScenario


@fixture
@parametrize_with_cases("scenario", cases="tests.index.case_store_versioning")
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
        # Seed through IndexStore so every internal invariant is set up
        # (meta rows, schema_version, embedding_model, ...) before we
        # mutate what we want to mutate.
        seed = IndexStore(path)
        if before.seeded_chunks:
            seed.insert_chunks(list(before.seeded_chunks))
            for chunk in before.seeded_chunks:
                seed.insert_snapshot("head", chunk.file_path, chunk.blob_sha)
        for chunk_id, vec in before.seeded_embeddings.items():
            seed.update_embedding(chunk_id, vec)
        seed.close()

        updates: list[tuple[str, str]] = []
        if before.schema_version != "":
            updates.append(("schema_version", str(before.schema_version)))
        if before.embedding_version is not None:
            updates.append(("embedding_version", str(before.embedding_version)))
        if before.embedding_model is not None:
            updates.append(("embedding_model", before.embedding_model))
        if updates:
            con = duckdb.connect(str(path))
            for key, value in updates:
                con.execute("UPDATE meta SET value = ? WHERE key = ?", [value, key])
            con.close()

    if scenario.config_embedding_model is not None:
        mocker.patch(
            "rbtr.index.store.config.embedding_model",
            scenario.config_embedding_model,
        )

    store = IndexStore(path)
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
    chunks = store.get_chunks("head")
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
    chunks = store.get_chunks("head")
    if not chunks:
        # Chunks were wiped; there's nothing to check.
        assert not scenario.expected_chunks_survive
        return
    any_embedded = any(c.embedding for c in chunks)
    if scenario.expected_embeddings_survive:
        assert any_embedded, "expected some embedding to survive"
    else:
        assert not any_embedded, "expected embeddings to be cleared"


def test_expected_model_stamp_matches_scenario(
    reopened: tuple[IndexStore, VersioningScenario],
) -> None:
    store, scenario = reopened
    if scenario.expected_model_stamp is None:
        return
    rows = store._cur().execute("SELECT value FROM meta WHERE key = 'embedding_model'").fetchall()
    assert rows
    assert rows[0][0] == scenario.expected_model_stamp
