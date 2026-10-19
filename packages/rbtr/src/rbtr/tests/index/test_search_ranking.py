"""E2E ranking tests for unified search.

Each test asserts a specific ranking invariant against the shared
dataset (see `conftest.py`).  The invariants are drawn from the
design doc (TODO-search.md) and verify that the correct signal
resolves each ranking conflict.

No mocking — these exercise the full pipeline through
`Index.search()`: BM25, name matching, kind boost,
file-category penalty, and importance.
"""

from __future__ import annotations

from rbtr.domain.models import ScoredChunk
from rbtr.index.index import Index
from rbtr.index.search import search

from .asserts import assert_outranks, assert_ranked_within


def test_class_definition_outranks_its_import(ranking_index: Index, ranking_commit: str) -> None:
    """CLASS (1.5) outranks IMPORT (0.3) for 'AppConfig'."""
    [repo] = ranking_index.repos()
    results = search(
        ranking_index,
        "AppConfig",
        within=[repo.at(ranking_commit)],
        top_k=10,
    )
    assert_outranks(results, "AppConfig", "from config import AppConfig")


def test_source_function_outranks_test_with_higher_tf(
    ranking_index: Index,
    ranking_commit: str,
) -> None:
    """Source function outranks test despite test having 3x more mentions.

    test_config mentions 'load_config' 3 times; the source defines
    it once.  File-category penalty (0.5 on test) resolves this.
    """
    [repo] = ranking_index.repos()
    results = search(
        ranking_index,
        "load_config",
        within=[repo.at(ranking_commit)],
        top_k=10,
    )
    assert_outranks(results, "load_config", "test_load_config")


def test_doc_section_ranks_below_code(ranking_index: Index, ranking_commit: str) -> None:
    """Doc section (0.8) ranks below source function (1.0) for 'load_config'."""
    [repo] = ranking_index.repos()
    results = search(
        ranking_index,
        "load_config",
        within=[repo.at(ranking_commit)],
        top_k=10,
    )
    assert_outranks(results, "load_config", "Configuration")


def test_exact_name_match_ranks_first(ranking_index: Index, ranking_commit: str) -> None:
    """Exact name match on 'start_server' ranks it first.

    Other chunks mention 'server' in content (import_config,
    start_server definition), but only start_server has the exact
    name match (score 1.0).
    """
    [repo] = ranking_index.repos()
    results = search(
        ranking_index,
        "start_server",
        within=[repo.at(ranking_commit)],
        top_k=10,
    )

    assert len(results) >= 1
    assert results[0].name == "start_server"


def test_class_name_query_finds_definition(ranking_index: Index, ranking_commit: str) -> None:
    """Query 'AppConfig' finds the class definition at rank 1."""
    [repo] = ranking_index.repos()
    results = search(
        ranking_index,
        "AppConfig",
        within=[repo.at(ranking_commit)],
        top_k=10,
    )

    assert len(results) >= 1
    assert results[0].name == "AppConfig"


def test_high_df_term_still_finds_definition(ranking_index: Index, ranking_commit: str) -> None:
    """'config' appears in 5/6 chunks; class definition is in top 2.

    IDF neutralisation prevents rare-term bias, and kind boost
    (CLASS=1.5) lifts the definition above high-TF test content.
    """
    [repo] = ranking_index.repos()
    results = search(
        ranking_index,
        "config",
        within=[repo.at(ranking_commit)],
        top_k=10,
    )
    assert_ranked_within(results, "AppConfig", top=2)


def test_database_query_finds_class(ranking_index: Index, ranking_commit: str) -> None:
    """'database' appears only in class + doc; class ranks first.

    Kind boost (CLASS=1.5 vs DOC_SECTION=0.8) resolves it.
    """
    [repo] = ranking_index.repos()
    results = search(
        ranking_index,
        "database",
        within=[repo.at(ranking_commit)],
        top_k=10,
    )
    assert_ranked_within(results, "AppConfig", top=2)


def test_score_breakdown_is_populated(ranking_index: Index, ranking_commit: str) -> None:
    """Every ScoredChunk has non-negative signal values."""
    [repo] = ranking_index.repos()
    results = search(
        ranking_index,
        "AppConfig",
        within=[repo.at(ranking_commit)],
        top_k=10,
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


def test_results_sorted_by_score(ranking_index: Index, ranking_commit: str) -> None:
    """Results are returned in descending score order."""
    [repo] = ranking_index.repos()
    results = search(
        ranking_index,
        "config",
        within=[repo.at(ranking_commit)],
        top_k=10,
    )

    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_no_results_for_gibberish(ranking_index: Index, ranking_commit: str) -> None:
    """Gibberish query returns empty list."""
    [repo] = ranking_index.repos()
    results = search(
        ranking_index,
        "zzz_xyzzy_999",
        within=[repo.at(ranking_commit)],
        top_k=10,
    )
    assert results == []


def test_expansion_keywords_widen_bm25(ranking_index: Index, ranking_commit: str) -> None:
    """Keywords in expansion are appended to the BM25 query.

    A gibberish query finds nothing on its own, but expansion
    keywords containing real terms surface matching chunks.
    """
    [repo] = ranking_index.repos()
    no_results = search(
        ranking_index,
        "zzz_xyzzy_999",
        within=[repo.at(ranking_commit)],
        top_k=10,
    )
    assert no_results == []

    with_keywords = search(
        ranking_index,
        "zzz_xyzzy_999",
        within=[repo.at(ranking_commit)],
        top_k=10,
        keywords=["config", "load"],
    )
    assert len(with_keywords) > 0
    # The keywords caused BM25 to find chunks containing "config" or "load".
    names = [r.name for r in with_keywords]
    assert any("config" in n.lower() or "load" in n.lower() for n in names)


def test_match_preview_populated_for_lexical_hit(ranking_index: Index, ranking_commit: str) -> None:
    """A literal hit carries a preview anchor through to `ScoredChunk`.

    Guards the `search` → `materialise_scored` (lex_query) wiring:
    the `load_config` chunk's signature line contains the query
    tokens, so it anchors there with the terms populated.
    """
    [repo] = ranking_index.repos()
    results = search(
        ranking_index,
        "load_config",
        within=[repo.at(ranking_commit)],
        top_k=10,
    )
    hit = next(r for r in results if r.name == "load_config")
    assert hit.match_line_offset is not None
    assert "load" in hit.matched_terms
    assert "config" in hit.matched_terms
