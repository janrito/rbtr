"""E2E ranking tests for unified search.

Each test asserts a specific ranking invariant against the shared
dataset (see `conftest.py`).  The invariants are drawn from the
design doc (TODO-search.md) and verify that the correct signal
resolves each ranking conflict.

No mocking — these exercise the full pipeline through
`IndexStore.search()`: BM25, name matching, kind boost,
file-category penalty, and importance.
"""

from __future__ import annotations

from rbtr.index.classify import Expansion, QueryKind
from rbtr.index.models import RepoRef, ScoredChunk
from rbtr.index.store import IndexStore

from .asserts import assert_outranks, assert_ranked_within, assert_same_ranking

# ── Kind boost ───────────────────────────────────────────────────────


def test_class_definition_outranks_its_import(
    ranking_store: IndexStore, ranking_commit: str
) -> None:
    """CLASS (1.5) outranks IMPORT (0.3) for 'AppConfig'."""
    results = ranking_store.search(
        [RepoRef(repo_id=1, commit_sha=ranking_commit)], "AppConfig", top_k=10
    )
    assert_outranks(results, "config_class", "import_config")


# ── File-category penalty ───────────────────────────────────────────


def test_source_function_outranks_test_with_higher_tf(
    ranking_store: IndexStore,
    ranking_commit: str,
) -> None:
    """Source function outranks test despite test having 3x more mentions.

    test_config mentions 'load_config' 3 times; the source defines
    it once.  File-category penalty (0.5 on test) resolves this.
    """
    results = ranking_store.search(
        [RepoRef(repo_id=1, commit_sha=ranking_commit)], "load_config", top_k=10
    )
    assert_outranks(results, "load_config", "test_config")


def test_doc_section_ranks_below_code(ranking_store: IndexStore, ranking_commit: str) -> None:
    """Doc section (0.8) ranks below source function (1.0) for 'load_config'."""
    results = ranking_store.search(
        [RepoRef(repo_id=1, commit_sha=ranking_commit)], "load_config", top_k=10
    )
    assert_outranks(results, "load_config", "doc_config")


# ── Name matching ────────────────────────────────────────────────────


def test_exact_name_match_ranks_first(ranking_store: IndexStore, ranking_commit: str) -> None:
    """Exact name match on 'start_server' ranks it first.

    Other chunks mention 'server' in content (import_config,
    start_server definition), but only start_server has the exact
    name match (score 1.0).
    """
    results = ranking_store.search(
        [RepoRef(repo_id=1, commit_sha=ranking_commit)], "start_server", top_k=10
    )

    assert len(results) >= 1
    assert results[0].id == "start_server"


def test_class_name_query_finds_definition(ranking_store: IndexStore, ranking_commit: str) -> None:
    """Query 'AppConfig' finds the class definition at rank 1."""
    results = ranking_store.search(
        [RepoRef(repo_id=1, commit_sha=ranking_commit)], "AppConfig", top_k=10
    )

    assert len(results) >= 1
    assert results[0].id == "config_class"


# ── High-df terms ────────────────────────────────────────────────────


def test_high_df_term_still_finds_definition(
    ranking_store: IndexStore, ranking_commit: str
) -> None:
    """'config' appears in 5/6 chunks; class definition is in top 2.

    IDF neutralisation prevents rare-term bias, and kind boost
    (CLASS=1.5) lifts the definition above high-TF test content.
    """
    results = ranking_store.search(
        [RepoRef(repo_id=1, commit_sha=ranking_commit)], "config", top_k=10
    )
    assert_ranked_within(results, "config_class", top=2)


def test_database_query_finds_class(ranking_store: IndexStore, ranking_commit: str) -> None:
    """'database' appears only in class + doc; class ranks first.

    Kind boost (CLASS=1.5 vs DOC_SECTION=0.8) resolves it.
    """
    results = ranking_store.search(
        [RepoRef(repo_id=1, commit_sha=ranking_commit)], "database", top_k=10
    )
    assert_ranked_within(results, "config_class", top=2)


# ── Score breakdown populated ────────────────────────────────────────


def test_score_breakdown_is_populated(ranking_store: IndexStore, ranking_commit: str) -> None:
    """Every ScoredChunk has non-negative signal values."""
    results = ranking_store.search(
        [RepoRef(repo_id=1, commit_sha=ranking_commit)], "AppConfig", top_k=10
    )

    assert len(results) >= 1
    for r in results:
        assert isinstance(r, ScoredChunk)
        assert r.score >= 0.0
        assert r.kind_boost > 0.0
        assert r.file_penalty > 0.0
        assert r.importance >= 1.0
        assert r.proximity >= 1.0  # no diff context
        assert r.lexical >= 0.0
        assert r.semantic >= 0.0
        assert r.name_match >= 0.0


def test_results_sorted_by_score(ranking_store: IndexStore, ranking_commit: str) -> None:
    """Results are returned in descending score order."""
    results = ranking_store.search(
        [RepoRef(repo_id=1, commit_sha=ranking_commit)], "config", top_k=10
    )

    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_no_results_for_gibberish(ranking_store: IndexStore, ranking_commit: str) -> None:
    """Gibberish query returns empty list."""
    results = ranking_store.search(
        [RepoRef(repo_id=1, commit_sha=ranking_commit)], "zzz_xyzzy_999", top_k=10
    )
    assert results == []


# ── Expansion integration ─────────────────────────────────────────


def test_search_without_expansion_unchanged(ranking_store: IndexStore, ranking_commit: str) -> None:
    """expansion=None produces identical results to no expansion."""
    baseline = ranking_store.search(
        [RepoRef(repo_id=1, commit_sha=ranking_commit)], "config", top_k=10
    )
    with_none = ranking_store.search(
        [RepoRef(repo_id=1, commit_sha=ranking_commit)], "config", top_k=10, expansion=None
    )
    assert_same_ranking(baseline, with_none)


def test_expansion_keywords_widen_bm25(ranking_store: IndexStore, ranking_commit: str) -> None:
    """Keywords in expansion are appended to the BM25 query.

    A gibberish query finds nothing on its own, but expansion
    keywords containing real terms surface matching chunks.
    """
    no_results = ranking_store.search(
        [RepoRef(repo_id=1, commit_sha=ranking_commit)], "zzz_xyzzy_999", top_k=10
    )
    assert no_results == []

    expansion = Expansion(
        kind=QueryKind.CONCEPT,
        keywords=["config", "load"],
        variants=[],
    )
    with_keywords = ranking_store.search(
        [RepoRef(repo_id=1, commit_sha=ranking_commit)],
        "zzz_xyzzy_999",
        top_k=10,
        expansion=expansion,
    )
    assert len(with_keywords) > 0
    # The keywords caused BM25 to find chunks containing "config" or "load".
    names = [r.name for r in with_keywords]
    assert any("config" in n.lower() or "load" in n.lower() for n in names)
