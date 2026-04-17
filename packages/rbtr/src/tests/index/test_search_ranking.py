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

from rbtr.index.search import ScoredResult
from rbtr.index.store import IndexStore

# ── Helpers ──────────────────────────────────────────────────────────


def _ids(results: list[ScoredResult]) -> list[str]:
    """Extract chunk IDs in rank order."""
    return [r.chunk.id for r in results]


def _rank(results: list[ScoredResult], chunk_id: str) -> int | None:
    """Return 1-indexed rank of *chunk_id*, or None if absent."""
    for i, r in enumerate(results, 1):
        if r.chunk.id == chunk_id:
            return i
    return None


# ── Kind boost ───────────────────────────────────────────────────────


def test_class_definition_outranks_its_import(ranking_store: IndexStore, ranking_commit: str) -> None:
    """CLASS (1.5) outranks IMPORT (0.3) for 'AppConfig'."""
    results = ranking_store.search(ranking_commit, "AppConfig", top_k=10)

    r_class = _rank(results, "config_class")
    r_import = _rank(results, "import_config")
    assert r_class is not None
    assert r_import is not None
    assert r_class < r_import


def test_import_chunk_ranks_below_class_definition(
    ranking_store: IndexStore, ranking_commit: str,
) -> None:
    """Import chunk ranks below class definition for 'AppConfig'.

    The import contains 'AppConfig' in its name (substring match
    → name_score=0.5), but its kind boost (0.3) keeps it below
    the CLASS definition (kind_boost=1.5).

    Note: the import may outrank chunks that don't mention
    'AppConfig' in their name at all (e.g. load_config), because
    identifier queries heavily weight the name channel.
    """
    results = ranking_store.search(ranking_commit, "AppConfig", top_k=10)

    r_class = _rank(results, "config_class")
    r_import = _rank(results, "import_config")
    assert r_class is not None
    assert r_import is not None
    assert r_class < r_import


# ── File-category penalty ───────────────────────────────────────────


def test_source_function_outranks_test_with_higher_tf(
    ranking_store: IndexStore, ranking_commit: str,
) -> None:
    """Source function outranks test despite test having 3x more mentions.

    test_config mentions 'load_config' 3 times; the source defines
    it once.  File-category penalty (0.5 on test) resolves this.
    """
    results = ranking_store.search(ranking_commit, "load_config", top_k=10)

    r_source = _rank(results, "load_config")
    r_test = _rank(results, "test_config")
    assert r_source is not None
    assert r_test is not None
    assert r_source < r_test


def test_doc_section_ranks_below_code(ranking_store: IndexStore, ranking_commit: str) -> None:
    """Doc section (0.8) ranks below source function (1.0) for 'load_config'."""
    results = ranking_store.search(ranking_commit, "load_config", top_k=10)

    r_fn = _rank(results, "load_config")
    r_doc = _rank(results, "doc_config")
    assert r_fn is not None
    assert r_doc is not None
    assert r_fn < r_doc


# ── Name matching ────────────────────────────────────────────────────


def test_exact_name_match_ranks_first(ranking_store: IndexStore, ranking_commit: str) -> None:
    """Exact name match on 'start_server' ranks it first.

    Other chunks mention 'server' in content (import_config,
    start_server definition), but only start_server has the exact
    name match (score 1.0).
    """
    results = ranking_store.search(ranking_commit, "start_server", top_k=10)

    assert len(results) >= 1
    assert results[0].chunk.id == "start_server"


def test_class_name_query_finds_definition(ranking_store: IndexStore, ranking_commit: str) -> None:
    """Query 'AppConfig' finds the class definition at rank 1."""
    results = ranking_store.search(ranking_commit, "AppConfig", top_k=10)

    assert len(results) >= 1
    assert results[0].chunk.id == "config_class"


# ── High-df terms ────────────────────────────────────────────────────


def test_high_df_term_still_finds_definition(ranking_store: IndexStore, ranking_commit: str) -> None:
    """'config' appears in 5/6 chunks; class definition is in top 2.

    IDF neutralisation prevents rare-term bias, and kind boost
    (CLASS=1.5) lifts the definition above high-TF test content.
    """
    results = ranking_store.search(ranking_commit, "config", top_k=10)

    r_class = _rank(results, "config_class")
    assert r_class is not None
    assert r_class <= 2


def test_database_query_finds_class(ranking_store: IndexStore, ranking_commit: str) -> None:
    """'database' appears only in class + doc; class ranks first.

    Kind boost (CLASS=1.5 vs DOC_SECTION=0.8) resolves it.
    """
    results = ranking_store.search(ranking_commit, "database", top_k=10)

    r_class = _rank(results, "config_class")
    r_doc = _rank(results, "doc_config")
    assert r_class is not None
    # Class should rank above doc, if doc appears at all.
    if r_doc is not None:
        assert r_class < r_doc


# ── Score breakdown populated ────────────────────────────────────────


def test_score_breakdown_is_populated(ranking_store: IndexStore, ranking_commit: str) -> None:
    """Every ScoredResult has non-negative signal values."""
    results = ranking_store.search(ranking_commit, "AppConfig", top_k=10)

    assert len(results) >= 1
    for r in results:
        assert isinstance(r, ScoredResult)
        assert r.score >= 0.0
        assert r.kind_boost > 0.0
        assert r.file_penalty > 0.0
        assert r.importance >= 1.0
        assert r.proximity >= 1.0  # no diff context
        assert r.lexical >= 0.0
        assert r.semantic >= 0.0
        assert r.name >= 0.0


def test_results_sorted_by_score(ranking_store: IndexStore, ranking_commit: str) -> None:
    """Results are returned in descending score order."""
    results = ranking_store.search(ranking_commit, "config", top_k=10)

    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_no_results_for_gibberish(ranking_store: IndexStore, ranking_commit: str) -> None:
    """Gibberish query returns empty list."""
    results = ranking_store.search(ranking_commit, "zzz_xyzzy_999", top_k=10)
    assert results == []
