"""Behavioural tests for chunk storage and retrieval.

Every test runs against the same seeded-store fixture, which is
parametrised by every case in ``case_store_chunks.py``.  Each test
function asserts a different *dimension* of the scenario's
expectations (commit scoping, path filter, kind filter, etc.), so
adding a new scenario exercises all dimensions at once.

A scenario whose ``expected_*`` map is empty for a given dimension
is skipped for the test function covering that dimension — this
keeps cases pure data without forcing every case to declare
every expectation.
"""

from __future__ import annotations

from pytest_cases import fixture, parametrize_with_cases

from rbtr.index.models import Chunk
from rbtr.index.store import IndexStore
from tests.index.case_store_chunks import ChunkScenario


@fixture
@parametrize_with_cases("scenario", cases="tests.index.case_store_chunks")
def seeded(scenario: ChunkScenario) -> tuple[IndexStore, ChunkScenario]:
    """Build an ``IndexStore`` populated per *scenario*."""
    store = IndexStore()
    for i, path in enumerate(scenario.repo_paths):
        repo_id = store.register_repo(path)
        assert repo_id == i + 1, "cases rely on sequential repo_ids"
        data = scenario.per_repo[i] if i < len(scenario.per_repo) else None
        if data is None:
            continue
        for batch in data.inserts:
            store.insert_chunks(list(batch), repo_id=repo_id)
        for commit, file_path, blob_sha in data.snapshots:
            store.insert_snapshot(commit, file_path, blob_sha, repo_id=repo_id)
    return store, scenario


def test_get_chunks_returns_expected_ids_per_commit(
    seeded: tuple[IndexStore, ChunkScenario],
) -> None:
    store, scenario = seeded
    for (repo_id, commit), expected_ids in scenario.expected_chunk_ids.items():
        actual = sorted(c.id for c in store.get_chunks(commit, repo_id=repo_id))
        assert actual == expected_ids, f"repo_id={repo_id}, commit={commit!r}"


def test_get_chunks_filter_by_file_path(
    seeded: tuple[IndexStore, ChunkScenario],
) -> None:
    store, scenario = seeded
    for (repo_id, commit, path), expected_ids in scenario.expected_by_path.items():
        actual = sorted(c.id for c in store.get_chunks(commit, file_path=path, repo_id=repo_id))
        assert actual == expected_ids, f"repo_id={repo_id}, commit={commit!r}, file_path={path!r}"


def test_get_chunks_filter_by_kind(
    seeded: tuple[IndexStore, ChunkScenario],
) -> None:
    store, scenario = seeded
    for (repo_id, commit, kind), expected_ids in scenario.expected_by_kind.items():
        actual = sorted(c.id for c in store.get_chunks(commit, kind=kind, repo_id=repo_id))
        assert actual == expected_ids, f"repo_id={repo_id}, commit={commit!r}, kind={kind!r}"


def test_get_chunks_filter_by_name(
    seeded: tuple[IndexStore, ChunkScenario],
) -> None:
    store, scenario = seeded
    for (repo_id, commit, name), expected_ids in scenario.expected_by_name.items():
        actual = sorted(c.id for c in store.get_chunks(commit, name=name, repo_id=repo_id))
        assert actual == expected_ids, f"repo_id={repo_id}, commit={commit!r}, name={name!r}"


def test_has_blob_matches_scenario(
    seeded: tuple[IndexStore, ChunkScenario],
) -> None:
    store, scenario = seeded
    for (repo_id, blob_sha), expected in scenario.expected_has_blob.items():
        assert store.has_blob(blob_sha, repo_id=repo_id) is expected, (
            f"repo_id={repo_id}, blob_sha={blob_sha!r}"
        )


def test_count_chunks_matches_scenario(
    seeded: tuple[IndexStore, ChunkScenario],
) -> None:
    store, scenario = seeded
    for (repo_id, commit), expected in scenario.expected_counts.items():
        assert store.count_chunks(commit, repo_id=repo_id) == expected, (
            f"repo_id={repo_id}, commit={commit!r}"
        )


def test_roundtrip_preserves_every_chunk_field(
    seeded: tuple[IndexStore, ChunkScenario],
) -> None:
    """Every field of every inserted chunk survives ``get_chunks``.

    Skips repos/commits the scenario didn't record expectations for
    — there is nothing to compare against.
    """
    store, scenario = seeded
    for (repo_id, commit), expected_ids in scenario.expected_chunk_ids.items():
        if not expected_ids:
            continue
        # Build the expected final chunk-by-id from the insert batches.
        data = scenario.per_repo[repo_id - 1]
        final: dict[str, Chunk] = {}
        for batch in data.inserts:
            for chunk in batch:
                final[chunk.id] = chunk

        retrieved = {c.id: c for c in store.get_chunks(commit, repo_id=repo_id)}
        for chunk_id in expected_ids:
            want = final[chunk_id]
            got = retrieved[chunk_id]
            assert got.name == want.name
            assert got.kind == want.kind
            assert got.file_path == want.file_path
            assert got.blob_sha == want.blob_sha
            assert got.line_start == want.line_start
            assert got.line_end == want.line_end
            assert got.scope == want.scope
            assert got.content == want.content


def test_upsert_replaces_chunk_content(
    seeded: tuple[IndexStore, ChunkScenario],
) -> None:
    """Scenarios with multiple insert batches against one id.

    The last batch's version of each id must be what the store
    returns.  Scenarios with a single batch skip this check.
    """
    store, scenario = seeded
    if not scenario.per_repo:
        return
    inserts = scenario.per_repo[0].inserts
    if len(inserts) < 2:
        return

    final_by_id: dict[str, Chunk] = {}
    for batch in inserts:
        for chunk in batch:
            final_by_id[chunk.id] = chunk

    for chunk_id, final in final_by_id.items():
        if chunk_id not in (cid for cids in scenario.expected_chunk_ids.values() for cid in cids):
            continue
        actual = store.get_chunks("head", name=final.name, repo_id=1)
        assert len(actual) == 1
        assert actual[0].content == final.content
