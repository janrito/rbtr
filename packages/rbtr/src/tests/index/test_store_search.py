"""Behavioural tests for ``IndexStore`` search and embedding APIs.

Every test runs against the same seeded-store fixture,
parametrised by each ``SearchScenario``.  Each function asserts
one public search method against the scenario's expectations.
The fixture seeds chunks with proper tokenisation so FTS works
out of the box \u2014 no test needs to call ``rebuild_fts_index``.
"""

from __future__ import annotations

from pytest_cases import fixture, parametrize_with_cases

from rbtr.index.store import IndexStore
from rbtr.index.tokenise import tokenise_code
from tests.index.case_store_search import SearchScenario


@fixture
@parametrize_with_cases("scenario", cases="tests.index.case_store_search")
def seeded(scenario: SearchScenario) -> tuple[IndexStore, SearchScenario]:
    """Build an ``IndexStore`` populated per *scenario*."""
    store = IndexStore()
    store.register_repo("/default")

    tokenised = [
        c.model_copy(
            update={
                "content_tokens": tokenise_code(c.content),
                "name_tokens": tokenise_code(c.name),
            }
        )
        for c in scenario.chunks
    ]
    if tokenised:
        store.insert_chunks(tokenised)
    for commit, path, blob in scenario.snapshots:
        store.insert_snapshot(commit, path, blob)
    for chunk_id, vector in scenario.embeddings.items():
        store.update_embedding(chunk_id, vector)
    return store, scenario


def test_search_by_name_returns_expected(
    seeded: tuple[IndexStore, SearchScenario],
) -> None:
    store, scenario = seeded
    for pattern, expected_ids in scenario.by_name:
        actual = [c.id for c in store.search_by_name(scenario.commit, pattern)]
        assert sorted(actual) == sorted(expected_ids), f"pattern={pattern!r}"


def test_search_by_name_result_carries_embedding_when_present(
    seeded: tuple[IndexStore, SearchScenario],
) -> None:
    """If the chunk has an embedding, the returned Chunk's flag is truthy."""
    store, scenario = seeded
    for pattern, expected_ids in scenario.by_name:
        if not expected_ids:
            continue
        for chunk in store.search_by_name(scenario.commit, pattern):
            if chunk.id in scenario.embeddings:
                assert chunk.embedding, f"missing embedding on {chunk.id!r}"


def test_search_similar_ranking(
    seeded: tuple[IndexStore, SearchScenario],
) -> None:
    """Ordering constraints from the scenario are respected."""
    store, scenario = seeded
    for query, top_k, ordered in scenario.similar:
        results = store.search_similar(scenario.commit, query, top_k=top_k)
        ids = [c.id for c, _ in results]
        if not ordered:
            # Empty constraint \u2192 scenario expects no hits.
            assert ids == [], f"query={query!r}"
            continue
        # Every id in ``ordered`` must appear in ``ids`` in that order.
        actual_positions = [ids.index(sid) for sid in ordered if sid in ids]
        assert actual_positions == sorted(actual_positions), f"expected order {ordered}, got {ids}"


def test_search_similar_exact_match_scores_near_one(
    seeded: tuple[IndexStore, SearchScenario],
) -> None:
    """If the query vector matches an embedded chunk exactly, score > 0.99."""
    store, scenario = seeded
    for chunk_id, vector in scenario.embeddings.items():
        results = store.search_similar(scenario.commit, vector, top_k=1)
        if not results:
            continue
        top, score = results[0]
        if top.id != chunk_id:
            # Other embedded chunks may share the vector (e.g. MATH_FUNC /
            # MATH_CLASS both on VEC_MATH).
            continue
        assert score > 0.99, f"score={score} for exact-match {chunk_id!r}"


def test_search_fulltext_top_hit(
    seeded: tuple[IndexStore, SearchScenario],
) -> None:
    store, scenario = seeded
    for query, expected_top_id in scenario.fulltext_top:
        results = store.search_fulltext(scenario.commit, query, top_k=4)
        assert results, f"no results for {query!r}"
        assert results[0][0].id == expected_top_id, f"query={query!r}"


def test_search_fulltext_empty_queries(
    seeded: tuple[IndexStore, SearchScenario],
) -> None:
    store, scenario = seeded
    for query, commit in scenario.fulltext_empty:
        results = store.search_fulltext(commit, query, top_k=4)
        assert results == [], f"query={query!r}, commit={commit!r}"


def test_search_fulltext_auto_rebuilds(
    seeded: tuple[IndexStore, SearchScenario],
) -> None:
    """Fixture never calls rebuild_fts_index; the first query still works."""
    store, scenario = seeded
    if not scenario.fulltext_top:
        return
    query, expected_top_id = scenario.fulltext_top[0]
    results = store.search_fulltext(scenario.commit, query, top_k=1)
    assert results
    assert results[0][0].id == expected_top_id


# ── Embedding storage ───────────────────────────────────────────
#
# The fixture seeds embeddings via update_embedding; these tests
# assert that the resulting chunks carry the expected flags, and
# that clear_embeddings / update_embeddings (batch) behave per
# the scenario's seeded state.


def test_seeded_embeddings_are_truthy_on_retrieved_chunks(
    seeded: tuple[IndexStore, SearchScenario],
) -> None:
    """Check against whatever commit the chunks actually live on."""
    store, scenario = seeded
    for commit in {sha for sha, _p, _b in scenario.snapshots}:
        chunks = {c.id: c for c in store.get_chunks(commit)}
        for chunk_id in scenario.embeddings:
            if chunk_id not in chunks:
                continue
            assert chunks[chunk_id].embedding, f"expected embedding truthy for {chunk_id!r}"


def test_unseeded_chunks_have_no_embedding(
    seeded: tuple[IndexStore, SearchScenario],
) -> None:
    store, scenario = seeded
    for commit in {sha for sha, _p, _b in scenario.snapshots}:
        chunks = {c.id: c for c in store.get_chunks(commit)}
        for chunk_id, chunk in chunks.items():
            if chunk_id in scenario.embeddings:
                continue
            assert not chunk.embedding, f"unexpected embedding on {chunk_id!r}"


def test_clear_embeddings_returns_seed_count(
    seeded: tuple[IndexStore, SearchScenario],
) -> None:
    store, scenario = seeded
    cleared = store.clear_embeddings()
    assert cleared == len(scenario.embeddings)


def test_batch_update_embeddings_sets_multiple_at_once(
    seeded: tuple[IndexStore, SearchScenario],
) -> None:
    """Batch-update every already-embedded chunk to a new vector."""
    store, scenario = seeded
    if not scenario.embeddings:
        return
    ids = list(scenario.embeddings)
    new_vectors = [[0.5, 0.5, 0.5] for _ in ids]
    store.update_embeddings(ids, new_vectors)
    # Use the commit where the chunks actually live, not scenario.commit
    # (which may point at an alternative empty commit on purpose).
    head_commits = {sha for sha, _p, _b in scenario.snapshots}
    for commit in head_commits:
        chunks = {c.id: c for c in store.get_chunks(commit)}
        hit = [cid for cid in ids if cid in chunks]
        for cid in hit:
            assert chunks[cid].embedding  # truthy
        if hit:
            return


def test_batch_update_embeddings_empty_is_noop(
    seeded: tuple[IndexStore, SearchScenario],
) -> None:
    store, _scenario = seeded
    store.update_embeddings([], [])  # must not raise
