"""Behavioural tests for store-level search — FTS, name, and semantic.

All data seeded through sessions. Case-driven for FTS and
name search; standalone tests for multi-repo, semantic,
IDF, persistence, and unified search.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pytest_cases import fixture, parametrize_with_cases

from rbtr.index.models import ChunkKind, Edge, EdgeKind, RepoRef
from rbtr.index.store import IndexStore

from .cases_search import SearchScenario
from .conftest import make_chunk, seed_store

# ── Shared seeding fixture ──────────────────────────────────────────


# ── FTS hits ────────────────────────────────────────────────────────


@fixture
@parametrize_with_cases("scenario", cases=".cases_search", has_tag="fts_hit")
def fts_hit(scenario: SearchScenario, store: IndexStore) -> tuple[IndexStore, SearchScenario]:
    seed_store(store, scenario.chunks)
    return store, scenario


def test_fts_finds_hit(fts_hit: tuple[IndexStore, SearchScenario]) -> None:
    store, s = fts_hit
    results = store.match_fulltext("head", s.query, repo_id=1)
    assert len(results) > 0
    assert results[0][0].id == s.expected_hit_ids[0]


# ── FTS empty ───────────────────────────────────────────────────────


@fixture
@parametrize_with_cases("scenario", cases=".cases_search", has_tag="fts_empty")
def fts_empty(scenario: SearchScenario, store: IndexStore) -> tuple[IndexStore, SearchScenario]:
    seed_store(store, scenario.chunks)
    return store, scenario


def test_fts_returns_empty(fts_empty: tuple[IndexStore, SearchScenario]) -> None:
    store, s = fts_empty
    results = store.match_fulltext("head", s.query, repo_id=1)
    assert results == []


# ── Name search ─────────────────────────────────────────────────────


@fixture
@parametrize_with_cases("scenario", cases=".cases_search", has_tag="name_hit")
def name_hit(scenario: SearchScenario, store: IndexStore) -> tuple[IndexStore, SearchScenario]:
    seed_store(store, scenario.chunks)
    return store, scenario


def test_name_search_finds_hit(name_hit: tuple[IndexStore, SearchScenario]) -> None:
    store, s = name_hit
    results = store.match_by_name("head", s.query, repo_id=1)
    assert len(results) > 0
    assert results[0].id == s.expected_hit_ids[0]


@fixture
@parametrize_with_cases("scenario", cases=".cases_search", has_tag="name_empty")
def name_empty(scenario: SearchScenario, store: IndexStore) -> tuple[IndexStore, SearchScenario]:
    seed_store(store, scenario.chunks)
    return store, scenario


def test_name_search_returns_empty(name_empty: tuple[IndexStore, SearchScenario]) -> None:
    store, s = name_empty
    results = store.match_by_name("head", s.query, repo_id=1)
    assert results == []


# ── Standalone tests (single-scenario behaviours) ───────────────────


@pytest.fixture
def multi_repo_store(store: IndexStore) -> tuple[IndexStore, int, int]:
    """Store with two repos seeded under their real registered ids.

    Returns `(store, repo_one_id, repo_two_id)`.  Each repo holds
    one chunk whose name shares the token `func` (so a single query
    reaches both) but is otherwise distinct.
    """
    with store.session() as ws:
        repo_one = ws.register_repo("/repo_one")
        repo_two = ws.register_repo("/repo_two")
    seed_store(
        store,
        [
            make_chunk("r1_a", name="alpha_func", path="a.py", blob="b_r1", repo_id=repo_one),
            make_chunk("r2_b", name="beta_func", path="b.py", blob="b_r2", repo_id=repo_two),
        ],
    )
    return store, repo_one, repo_two


def test_fts_scoped_to_repo(multi_repo_store: tuple[IndexStore, int, int]) -> None:
    """FTS results are scoped to the queried repo."""
    store, repo_one, repo_two = multi_repo_store
    r1 = store.match_fulltext("head", "alpha", repo_id=repo_one)
    assert len(r1) > 0
    assert all(c.id.startswith("r1") for c, _score in r1)

    r2 = store.match_fulltext("head", "alpha", repo_id=repo_two)
    assert r2 == []


def test_cross_repo_search_merges_both_repos(multi_repo_store: tuple[IndexStore, int, int]) -> None:
    """Two refs return hits from both repos."""
    store, repo_one, repo_two = multi_repo_store
    refs = [
        RepoRef(repo_id=repo_one, commit_sha="head"),
        RepoRef(repo_id=repo_two, commit_sha="head"),
    ]
    results = store.search(refs, "func", top_k=10)
    ids = {r.id for r in results}
    assert "r1_a" in ids
    assert "r2_b" in ids


def test_single_ref_search_scopes_to_one_repo(
    multi_repo_store: tuple[IndexStore, int, int],
) -> None:
    """One ref excludes the other repo's chunks."""
    store, repo_one, _repo_two = multi_repo_store
    results = store.search([RepoRef(repo_id=repo_one, commit_sha="head")], "func", top_k=10)
    ids = {r.id for r in results}
    assert "r1_a" in ids
    assert "r2_b" not in ids


@pytest.fixture
def semantic_store(store: IndexStore) -> IndexStore:
    """Store with two embedded chunks at different distances."""
    s = SearchScenario(
        chunks=[
            make_chunk("close", name="close_match", path="close.py"),
            make_chunk("far", name="far_match", path="far.py"),
        ],
        query="",
    )
    seed_store(store, s.chunks)
    vec_close = [0.9, 0.1, 0.1, 0.1]
    vec_far = [0.3, 0.7, 0.7, 0.7]
    with store.session() as ws:
        ws.update_embeddings(["close", "far"], [vec_close, vec_far], repo_id=1)
    return store


def test_match_similar_ranks_by_cosine(semantic_store: IndexStore) -> None:
    """Closest embedding ranks first."""
    query_vec = [1.0, 0.0, 0.0, 0.0]
    results = semantic_store.match_similar("head", query_vec, top_k=2, repo_id=1)
    assert len(results) >= 2
    assert results[0][0].id == "close"


def test_match_similar_single_vector(semantic_store: IndexStore) -> None:
    """Single vector returns closest chunk first."""
    query_vec = [1.0, 0.0, 0.0, 0.0]
    result = semantic_store._match_similar(
        [RepoRef(repo_id=1, commit_sha="head")], [query_vec], top_k=2
    )
    assert len(result) >= 2
    assert result["id"].to_list()[0] == "close"


def test_match_similar_picks_best_score(semantic_store: IndexStore) -> None:
    """Two query vectors — each chunk keeps its best similarity."""
    # vec_a is close to "close" chunk ([0.9, 0.1, ...]).
    vec_a = [1.0, 0.0, 0.0, 0.0]
    # vec_b is close to "far" chunk ([0.3, 0.7, ...]).
    vec_b = [0.0, 1.0, 0.0, 0.0]
    result = semantic_store._match_similar(
        [RepoRef(repo_id=1, commit_sha="head")], [vec_a, vec_b], top_k=2
    )
    ids = result["id"].to_list()
    assert "close" in ids
    assert "far" in ids
    # Each chunk should score better with the multi-vector query
    # than with only the *other* vector (the one it's far from).
    scores = dict(zip(result["id"].to_list(), result["score"].to_list(), strict=True))
    assert scores["close"] > scores["far"]  # vec_a boosts "close" more


def test_match_similar_empty(store: IndexStore) -> None:
    """No embeddings in store — returns empty frame."""
    s = SearchScenario(chunks=[make_chunk("x")], query="")
    seed_store(store, s.chunks)
    vec = [1.0, 0.0, 0.0, 0.0]
    result = store._match_similar([RepoRef(repo_id=1, commit_sha="head")], [vec], top_k=5)
    assert len(result) == 0


def test_unseeded_chunks_have_no_embedding(store: IndexStore) -> None:
    s = SearchScenario(chunks=[make_chunk("a")], query="")
    seed_store(store, s.chunks)
    chunks = store.get_chunks("head", repo_id=1)
    assert not chunks[0].embedding


def test_seeded_chunks_have_embedding_flag(store: IndexStore) -> None:
    s = SearchScenario(chunks=[make_chunk("a")], query="")
    seed_store(store, s.chunks)
    vec = [0.5, 0.5, 0.5, 0.5]
    with store.session() as ws:
        ws.update_embeddings(["a"], [vec], repo_id=1)
    chunks = store.get_chunks("head", repo_id=1)
    assert chunks[0].embedding


@pytest.fixture
def idf_store(store: IndexStore) -> IndexStore:
    """Store with many chunks sharing a common term."""
    s = SearchScenario(
        chunks=[
            make_chunk(f"c{i}", name=f"config_{i}", content=f"config = load_{i}()")
            for i in range(10)
        ],
        query="config",
    )
    seed_store(store, s.chunks)
    return store


def test_idf_neutralised_common_term(idf_store: IndexStore) -> None:
    """A term appearing in many chunks is still findable."""
    results = idf_store.match_fulltext("head", "config", repo_id=1)
    assert len(results) > 0


def test_fts_persists_across_reopen(tmp_path: Path) -> None:
    """FTS index survives close + reopen on a file-backed store."""
    db_path = tmp_path / "index.duckdb"
    s = SearchScenario(
        chunks=[make_chunk("a", name="persistent_func")],
        query="persistent",
        expected_hit_ids=["a"],
    )
    store = IndexStore(db_path, writable=True)
    seed_store(store, s.chunks)
    store.close()

    store2 = IndexStore(db_path, writable=True)
    try:
        results = store2.match_fulltext("head", "persistent", repo_id=1)
        assert len(results) > 0
    finally:
        store2.close()


@pytest.fixture
def unified_store(store: IndexStore) -> IndexStore:
    """Store with chunks + edges for unified search."""
    s = SearchScenario(
        chunks=[
            make_chunk("a", name="AppConfig", kind=ChunkKind.CLASS),
            make_chunk("b", name="load_config", kind=ChunkKind.FUNCTION),
        ],
        query="config",
    )
    seed_store(store, s.chunks)
    with store.session() as ws:
        ws.insert_edges(
            [Edge(source_id="b", target_id="a", kind=EdgeKind.IMPORTS)],
            "head",
            repo_id=1,
        )
    return store


def test_unified_search_returns_results_with_breakdown(unified_store: IndexStore) -> None:
    """store.search() returns ScoredChunks with score breakdown."""
    results = unified_store.search([RepoRef(repo_id=1, commit_sha="head")], "config")
    assert len(results) > 0
    top = results[0]
    assert top.score >= 0.0
    assert top.kind_boost > 0.0
    assert top.file_penalty > 0.0


# ── Unified search without embeddings ────────────────────────────────


@fixture
@parametrize_with_cases("scenario", cases=".cases_search", has_tag="unified_no_embed")
def unified_no_embed(
    scenario: SearchScenario, store: IndexStore
) -> tuple[IndexStore, SearchScenario]:
    seed_store(store, scenario.chunks)
    return store, scenario


def test_unified_search_without_embeddings(
    unified_no_embed: tuple[IndexStore, SearchScenario],
) -> None:
    """search() works when no embeddings exist (semantic weight redistributed)."""
    store, s = unified_no_embed
    results = store.search([RepoRef(repo_id=1, commit_sha="head")], s.query)
    assert len(results) > 0
    assert all(r.score >= 0.0 for r in results)
