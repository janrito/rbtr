"""Behavioural tests for store-level search — FTS, name, and semantic.

All data seeded through sessions. Case-driven for FTS and
name search; standalone tests for multi-repo, semantic,
IDF, persistence, and unified search.
"""

from __future__ import annotations

import pygit2
import pytest
from pytest_cases import fixture, parametrize_with_cases

from rbtr.domain.models import EmbeddingState, SnapshotRef
from rbtr.git import normalise_repo_path
from rbtr.index.build import build_index
from rbtr.index.embed import embed_index
from rbtr.index.embeddings import Embedder
from rbtr.index.index import Index
from rbtr.index.search import search

from ..conftest import make_commit
from .cases_search import SearchScenario


@fixture
@parametrize_with_cases("scenario", cases=".cases_search", has_tag="fts_hit")
def fts_hit(scenario: SearchScenario) -> SearchScenario:
    return scenario


@pytest.fixture
def fts_hit_ref(
    fts_hit: SearchScenario, index: Index, git_repo: pygit2.Repository, repo_path: str
) -> SnapshotRef:
    sha = str(make_commit(git_repo, fts_hit.files, ref="refs/heads/case"))
    build_index(repo_path, sha, index)
    return index.register(repo_path).at(sha)


def test_fts_finds_hit(fts_hit: SearchScenario, fts_hit_ref: SnapshotRef, index: Index) -> None:
    matched = index.fulltext_matches(fts_hit.query, within=[fts_hit_ref])
    assert len(matched) > 0
    assert matched["name"].to_list()[0] == fts_hit.expected_hit_names[0]


@fixture
@parametrize_with_cases("scenario", cases=".cases_search", has_tag="fts_empty")
def fts_empty(scenario: SearchScenario) -> SearchScenario:
    return scenario


@pytest.fixture
def fts_empty_ref(
    fts_empty: SearchScenario, index: Index, git_repo: pygit2.Repository, repo_path: str
) -> SnapshotRef:
    sha = str(make_commit(git_repo, fts_empty.files, ref="refs/heads/case"))
    build_index(repo_path, sha, index)
    return index.register(repo_path).at(sha)


def test_fts_returns_empty(
    fts_empty: SearchScenario, fts_empty_ref: SnapshotRef, index: Index
) -> None:
    assert len(index.fulltext_matches(fts_empty.query, within=[fts_empty_ref])) == 0


@fixture
@parametrize_with_cases("scenario", cases=".cases_search", has_tag="name_hit")
def name_hit(scenario: SearchScenario) -> SearchScenario:
    return scenario


@pytest.fixture
def name_hit_ref(
    name_hit: SearchScenario, index: Index, git_repo: pygit2.Repository, repo_path: str
) -> SnapshotRef:
    sha = str(make_commit(git_repo, name_hit.files, ref="refs/heads/case"))
    build_index(repo_path, sha, index)
    return index.register(repo_path).at(sha)


def test_name_search_finds_hit(
    name_hit: SearchScenario, name_hit_ref: SnapshotRef, index: Index
) -> None:
    names = index.name_matches(name_hit.query, within=[name_hit_ref])["name"].to_list()
    assert names[:1] == name_hit.expected_hit_names[:1]


@fixture
@parametrize_with_cases("scenario", cases=".cases_search", has_tag="name_empty")
def name_empty(scenario: SearchScenario) -> SearchScenario:
    return scenario


@pytest.fixture
def name_empty_ref(
    name_empty: SearchScenario, index: Index, git_repo: pygit2.Repository, repo_path: str
) -> SnapshotRef:
    sha = str(make_commit(git_repo, name_empty.files, ref="refs/heads/case"))
    build_index(repo_path, sha, index)
    return index.register(repo_path).at(sha)


def test_name_search_returns_empty(
    name_empty: SearchScenario, name_empty_ref: SnapshotRef, index: Index
) -> None:
    assert index.name_matches(name_empty.query, within=[name_empty_ref]).is_empty()


@pytest.fixture
def repo_one_ref(index: Index, git_repo: pygit2.Repository, repo_path: str) -> SnapshotRef:
    """`git_repo` built holding `alpha_func`.

    Its name shares the token `func` with `repo_two_ref`'s function, so
    one query reaches both repos; the rest of the name is distinct, so a
    result says which repo it came from.
    """
    a = b"""\
def alpha_func():
    pass
"""
    sha = str(make_commit(git_repo, {"a.py": a}, ref="refs/heads/alpha"))
    build_index(repo_path, sha, index)
    return index.register(repo_path).at(sha)


@pytest.fixture
def repo_two_ref(index: Index, second_repo: str) -> SnapshotRef:
    """`second_repo` built holding `beta_func` — see `repo_one_ref`."""
    b = b"""\
def beta_func():
    pass
"""
    sha = str(make_commit(pygit2.Repository(second_repo), {"b.py": b}, ref="refs/heads/beta"))
    build_index(second_repo, sha, index)
    return index.register(second_repo).at(sha)


def test_shared_chunk_found_via_fts_from_both_repos(
    shared_chunk_index: Index, shared_sha: str
) -> None:
    """A chunk shared by two repos is found by FTS from either repo.

    One physical row, two snapshot references: the id-keyed FTS index
    must surface it for whichever repo scopes the query.
    """
    index = shared_chunk_index
    repo1, repo2 = index.repos()
    r1 = index.fulltext_matches("shared", within=[repo1.at(shared_sha)])
    r2 = index.fulltext_matches("shared", within=[repo2.at(shared_sha)])
    assert r1["name"].to_list() == ["shared_fn"]
    assert r2["name"].to_list() == ["shared_fn"]


def test_shared_chunk_found_via_semantic_from_both_repos(
    shared_chunk_index: Index, shared_sha: str, stubbed_embedder: Embedder
) -> None:
    """A shared chunk's single embedding is reachable by semantic search from either repo."""
    index = shared_chunk_index
    repo1, repo2 = index.repos()
    embed_index(index, at=repo1.at(shared_sha), embedder=stubbed_embedder)
    query_vec = [1.0, 0.0, 0.0, 0.0]
    r1 = index.similar_matches([query_vec], within=[repo1.at(shared_sha)], top_k=5)
    r2 = index.similar_matches([query_vec], within=[repo2.at(shared_sha)], top_k=5)
    assert r1["name"].to_list() == ["shared_fn"]
    assert r2["name"].to_list() == ["shared_fn"]


def test_cross_repo_search_attributes_shared_chunk_to_each_repo(
    shared_chunk_index: Index,
    shared_sha: str,
) -> None:
    """Cross-repo search keeps a shared chunk as one row per repo, each attributed."""
    index = shared_chunk_index
    repo1, repo2 = index.repos()
    refs = [
        repo1.at(shared_sha),
        repo2.at(shared_sha),
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
def stub_vectors() -> dict[str, list[float]]:
    """Chosen vectors: `close_match` near `[1, 0, 0, 0]`, `far_match` away from it."""
    return {
        "close_match": [0.9, 0.1, 0.1, 0.1],
        "far_match": [0.3, 0.7, 0.7, 0.7],
        "shared_fn": [1.0, 0.0, 0.0, 0.0],
    }


@pytest.fixture
def semantic_ref(
    index: Index, git_repo: pygit2.Repository, repo_path: str, stubbed_embedder: Embedder
) -> SnapshotRef:
    """A built snapshot of two functions, embedded at different distances."""
    close = b"""\
def close_match():
    pass
"""
    far = b"""\
def far_match():
    pass
"""
    sha = str(make_commit(git_repo, {"close.py": close, "far.py": far}, ref="refs/heads/semantic"))
    build_index(repo_path, sha, index)
    ref = index.register(repo_path).at(sha)
    embed_index(index, at=ref, embedder=stubbed_embedder)
    return ref


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


@pytest.fixture(scope="module")
def unembedded_ref(module_index: Index, module_repo: pygit2.Repository) -> SnapshotRef:
    """A snapshot of one function, built into `module_index` and never embedded."""
    a = b"""\
def a():
    pass
"""
    sha = str(make_commit(module_repo, {"a.py": a}, ref="refs/heads/unembedded"))
    path = normalise_repo_path(module_repo.workdir)
    build_index(path, sha, module_index)
    return module_index.register(path).at(sha)


def test_similar_matches_empty(module_index: Index, unembedded_ref: SnapshotRef) -> None:
    """No embeddings in store — returns empty frame."""
    vec = [1.0, 0.0, 0.0, 0.0]
    assert len(module_index.similar_matches([vec], within=[unembedded_ref], top_k=5)) == 0


def test_unseeded_chunks_have_no_embedding(
    module_index: Index, unembedded_ref: SnapshotRef
) -> None:
    assert module_index.chunks(at=unembedded_ref)["embedding_state"].to_list() == [
        EmbeddingState.NONE
    ]


def test_embedded_chunks_have_embedding_flag(index: Index, semantic_ref: SnapshotRef) -> None:
    assert index.chunks(at=semantic_ref)["embedding_state"].to_list() == [EmbeddingState.FULL] * 2


@pytest.fixture
def idf_ref(index: Index, git_repo: pygit2.Repository, repo_path: str) -> SnapshotRef:
    """A built snapshot of many files sharing a common term."""
    template = """\
def config_{i}():
    config = load_{i}()
"""
    files = {f"c{i}.py": template.format(i=i).encode() for i in range(10)}
    sha = str(make_commit(git_repo, files, ref="refs/heads/idf"))
    build_index(repo_path, sha, index)
    return index.register(repo_path).at(sha)


def test_idf_neutralised_common_term(index: Index, idf_ref: SnapshotRef) -> None:
    """A term appearing in many chunks is still findable."""
    assert len(index.fulltext_matches("config", within=[idf_ref])) > 0


@pytest.fixture
def unified_ref(index: Index, git_repo: pygit2.Repository, repo_path: str) -> SnapshotRef:
    """A built snapshot where `b.py` imports the class `a.py` defines."""
    a = b"""\
class AppConfig:
    pass
"""
    b = b"""\
from a import AppConfig


def load_config():
    return AppConfig()
"""
    sha = str(make_commit(git_repo, {"a.py": a, "b.py": b}, ref="refs/heads/unified"))
    build_index(repo_path, sha, index)
    return index.register(repo_path).at(sha)


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
def unified_no_embed(scenario: SearchScenario) -> SearchScenario:
    return scenario


@pytest.fixture
def unified_no_embed_ref(
    unified_no_embed: SearchScenario, index: Index, git_repo: pygit2.Repository, repo_path: str
) -> SnapshotRef:
    sha = str(make_commit(git_repo, unified_no_embed.files, ref="refs/heads/case"))
    build_index(repo_path, sha, index)
    return index.register(repo_path).at(sha)


def test_unified_search_without_embeddings(
    unified_no_embed: SearchScenario, unified_no_embed_ref: SnapshotRef, index: Index
) -> None:
    """search() works when no embeddings exist (semantic weight redistributed)."""
    results = search(index, unified_no_embed.query, within=[unified_no_embed_ref])
    assert len(results) > 0
    assert all(r.score >= 0.0 for r in results)
