"""Behavioural tests for store-level search — FTS, name, and semantic.

All data seeded through sessions. Case-driven for FTS and
name search; standalone tests for multi-repo, semantic,
IDF, persistence, and unified search.
"""

from __future__ import annotations

import pytest
from pytest_cases import fixture, parametrize_with_cases

from rbtr.domain.models import ChunkKind, Edge, EdgeKind, SnapshotRef
from rbtr.index.index import Index
from rbtr.index.search import search
from rbtr.index.staging import TokenisedChunk

from .cases_search import SearchScenario
from .conftest import make_chunk, seed_index


@fixture
@parametrize_with_cases("scenario", cases=".cases_search", has_tag="fts_hit")
def fts_hit(scenario: SearchScenario, index: Index, head_ref: SnapshotRef) -> SearchScenario:
    seed_index(index, scenario.chunks, head_ref)
    return scenario


def test_fts_finds_hit(fts_hit: SearchScenario, index: Index, head_ref: SnapshotRef) -> None:
    matched = index.fulltext_matches(fts_hit.query, within=[head_ref])
    assert len(matched) > 0
    assert matched["name"].to_list()[0] == fts_hit.expected_hit_names[0]


@fixture
@parametrize_with_cases("scenario", cases=".cases_search", has_tag="fts_empty")
def fts_empty(scenario: SearchScenario, index: Index, head_ref: SnapshotRef) -> SearchScenario:
    seed_index(index, scenario.chunks, head_ref)
    return scenario


def test_fts_returns_empty(fts_empty: SearchScenario, index: Index, head_ref: SnapshotRef) -> None:
    assert len(index.fulltext_matches(fts_empty.query, within=[head_ref])) == 0


@fixture
@parametrize_with_cases("scenario", cases=".cases_search", has_tag="name_hit")
def name_hit(scenario: SearchScenario, index: Index, head_ref: SnapshotRef) -> SearchScenario:
    seed_index(index, scenario.chunks, head_ref)
    return scenario


def test_name_search_finds_hit(
    name_hit: SearchScenario, index: Index, head_ref: SnapshotRef
) -> None:
    names = index.name_matches(name_hit.query, within=[head_ref])["name"].to_list()
    assert names[:1] == name_hit.expected_hit_names[:1]


@fixture
@parametrize_with_cases("scenario", cases=".cases_search", has_tag="name_empty")
def name_empty(scenario: SearchScenario, index: Index, head_ref: SnapshotRef) -> SearchScenario:
    seed_index(index, scenario.chunks, head_ref)
    return scenario


def test_name_search_returns_empty(
    name_empty: SearchScenario, index: Index, head_ref: SnapshotRef
) -> None:
    assert index.name_matches(name_empty.query, within=[head_ref]).is_empty()


@pytest.fixture
def repo_one_ref(index: Index) -> SnapshotRef:
    """`/repo_one` at `head`, holding `alpha_func`.

    Its name shares the token `func` with `repo_two_ref`'s chunk, so one
    query reaches both repos; the rest of the name is distinct, so a
    result says which repo it came from.
    """
    repo = index.register("/repo_one")
    ref = repo.at("head")
    seed_index(index, [make_chunk("r1_a", name="alpha_func", path="a.py", blob="b_r1")], ref)
    return ref


@pytest.fixture
def repo_two_ref(index: Index) -> SnapshotRef:
    """`/repo_two` at `head`, holding `beta_func` — see `repo_one_ref`."""
    repo = index.register("/repo_two")
    ref = repo.at("head")
    seed_index(index, [make_chunk("r2_b", name="beta_func", path="b.py", blob="b_r2")], ref)
    return ref


def test_shared_chunk_found_via_fts_from_both_repos(shared_chunk_index: Index) -> None:
    """A chunk shared by two repos is found by FTS from either repo.

    One physical row, two snapshot references: the id-keyed FTS index
    must surface it for whichever repo scopes the query.
    """
    index = shared_chunk_index
    repo1, repo2 = index.repos()
    r1 = index.fulltext_matches("shared", within=[repo1.at("head")])
    r2 = index.fulltext_matches("shared", within=[repo2.at("head")])
    assert r1["name"].to_list() == ["shared_fn"]
    assert r2["name"].to_list() == ["shared_fn"]


def test_shared_chunk_found_via_semantic_from_both_repos(
    shared_chunk_index: Index, shared_chunk: TokenisedChunk
) -> None:
    """A shared chunk's single embedding is reachable by semantic search from either repo."""
    index = shared_chunk_index
    repo1, repo2 = index.repos()
    with index.session(repo1) as ws:
        ws.update_embeddings([shared_chunk.id], [[1.0, 0.0, 0.0, 0.0]])
    query_vec = [1.0, 0.0, 0.0, 0.0]
    r1 = index.similar_matches([query_vec], within=[repo1.at("head")], top_k=5)
    r2 = index.similar_matches([query_vec], within=[repo2.at("head")], top_k=5)
    assert r1["name"].to_list() == ["shared_fn"]
    assert r2["name"].to_list() == ["shared_fn"]


def test_cross_repo_search_attributes_shared_chunk_to_each_repo(
    shared_chunk_index: Index,
) -> None:
    """Cross-repo search keeps a shared chunk as one row per repo, each attributed."""
    index = shared_chunk_index
    repo1, repo2 = index.repos()
    refs = [
        repo1.at("head"),
        repo2.at("head"),
    ]
    results = search(index, "shared", within=refs, top_k=10, attribute_repos=True)
    shared = [r for r in results if r.name == "shared_fn"]
    assert {r.repo_path for r in shared} == {repo1.repo_path, repo2.repo_path}


def test_cross_repo_search_merges_both_repos(
    index: Index, repo_one_ref: SnapshotRef, repo_two_ref: SnapshotRef
) -> None:
    """Two refs return hits from both repos."""
    results = search(index, "func", within=[repo_one_ref, repo_two_ref], top_k=10)
    names = {r.name for r in results}
    assert "alpha_func" in names
    assert "beta_func" in names


@pytest.fixture
def semantic_ref(index: Index, head_ref: SnapshotRef) -> SnapshotRef:
    """`head_ref`, seeded with two embedded chunks at different distances."""
    close = make_chunk("close", name="close_match", path="close.py")
    far = make_chunk("far", name="far_match", path="far.py")
    seed_index(index, [close, far], head_ref)
    vec_close = [0.9, 0.1, 0.1, 0.1]
    vec_far = [0.3, 0.7, 0.7, 0.7]
    with index.session(head_ref.repo) as ws:
        ws.update_embeddings([close.id, far.id], [vec_close, vec_far])
    return head_ref


def test_similar_matches_single_vector(index: Index, semantic_ref: SnapshotRef) -> None:
    """Single vector returns closest chunk first."""
    query_vec = [1.0, 0.0, 0.0, 0.0]
    result = index.similar_matches([query_vec], within=[semantic_ref], top_k=2)
    assert len(result) >= 2
    assert result["name"].to_list()[0] == "close_match"


def test_similar_matches_picks_best_score(index: Index, semantic_ref: SnapshotRef) -> None:
    """Two query vectors — each chunk keeps its best similarity."""
    # vec_a is close to "close" chunk ([0.9, 0.1, ...]).
    vec_a = [1.0, 0.0, 0.0, 0.0]
    # vec_b is close to "far" chunk ([0.3, 0.7, ...]).
    vec_b = [0.0, 1.0, 0.0, 0.0]
    result = index.similar_matches([vec_a, vec_b], within=[semantic_ref], top_k=2)
    names = result["name"].to_list()
    assert "close_match" in names
    assert "far_match" in names
    # Each chunk should score better with the multi-vector query
    # than with only the *other* vector (the one it's far from).
    scores = dict(zip(names, result["score"].to_list(), strict=True))
    assert scores["close_match"] > scores["far_match"]  # vec_a boosts "close" more


def test_similar_matches_empty(index: Index, head_ref: SnapshotRef) -> None:
    """No embeddings in store — returns empty frame."""
    s = SearchScenario(chunks=[make_chunk("x")], query="")
    seed_index(index, s.chunks, head_ref)
    vec = [1.0, 0.0, 0.0, 0.0]
    assert len(index.similar_matches([vec], within=[head_ref], top_k=5)) == 0


def test_unseeded_chunks_have_no_embedding(index: Index, head_ref: SnapshotRef) -> None:
    s = SearchScenario(chunks=[make_chunk("a")], query="")
    seed_index(index, s.chunks, head_ref)
    chunks = index.chunks(at=head_ref)
    assert not chunks[0].has_embedding


def test_seeded_chunks_have_embedding_flag(index: Index, head_ref: SnapshotRef) -> None:
    chunk = make_chunk("a")
    seed_index(index, [chunk], head_ref)
    vec = [0.5, 0.5, 0.5, 0.5]
    with index.session(head_ref.repo) as ws:
        ws.update_embeddings([chunk.id], [vec])
    chunks = index.chunks(at=head_ref)
    assert chunks[0].has_embedding


@pytest.fixture
def idf_ref(index: Index, head_ref: SnapshotRef) -> SnapshotRef:
    """`head_ref`, seeded with many chunks sharing a common term."""
    s = SearchScenario(
        chunks=[
            make_chunk(f"c{i}", name=f"config_{i}", content=f"config = load_{i}()", path=f"c{i}.py")
            for i in range(10)
        ],
        query="config",
    )
    seed_index(index, s.chunks, head_ref)
    return head_ref


def test_idf_neutralised_common_term(index: Index, idf_ref: SnapshotRef) -> None:
    """A term appearing in many chunks is still findable."""
    assert len(index.fulltext_matches("config", within=[idf_ref])) > 0


@pytest.fixture
def unified_ref(index: Index, head_ref: SnapshotRef) -> SnapshotRef:
    """`head_ref`, seeded with chunks and one edge between them."""
    s = SearchScenario(
        chunks=[
            make_chunk("a", name="AppConfig", kind=ChunkKind.CLASS, path="a.py"),
            make_chunk("b", name="load_config", kind=ChunkKind.FUNCTION, path="b.py"),
        ],
        query="config",
    )
    seed_index(index, s.chunks, head_ref)
    with index.session(head_ref.repo) as ws:
        ws.insert_edges(
            [
                Edge(
                    source_id="b",
                    target_id="a",
                    kind=EdgeKind.IMPORTS,
                    source_path="src/b.py",
                    target_path="src/a.py",
                )
            ],
            snapshot_sha=head_ref.snapshot_sha,
        )
    return head_ref


def test_unified_search_returns_results_with_breakdown(
    index: Index, unified_ref: SnapshotRef
) -> None:
    """search(
    store,
    ) returns ScoredChunks with score breakdown."""
    results = search(index, "config", within=[unified_ref])
    assert len(results) > 0
    top = results[0]
    assert top.score >= 0.0
    assert top.kind_boost > 0.0
    assert top.file_penalty > 0.0


@fixture
@parametrize_with_cases("scenario", cases=".cases_search", has_tag="unified_no_embed")
def unified_no_embed(
    scenario: SearchScenario, index: Index, head_ref: SnapshotRef
) -> SearchScenario:
    seed_index(index, scenario.chunks, head_ref)
    return scenario


def test_unified_search_without_embeddings(
    unified_no_embed: SearchScenario, index: Index, head_ref: SnapshotRef
) -> None:
    """search() works when no embeddings exist (semantic weight redistributed)."""
    results = search(index, unified_no_embed.query, within=[head_ref])
    assert len(results) > 0
    assert all(r.score >= 0.0 for r in results)
