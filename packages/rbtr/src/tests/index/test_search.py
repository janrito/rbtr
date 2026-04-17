"""Tests for unified search scoring and fusion.

Pure function tests first, then integration tests that verify
the full pipeline produces correct rankings.
"""

from __future__ import annotations

import pytest

from rbtr.index.models import Chunk, ChunkKind, Edge, EdgeKind
from rbtr.index.search import (
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    DEFAULT_GAMMA,
    FileCategory,
    QueryKind,
    ScoredResult,
    classify_query,
    file_category,
    file_category_penalty,
    fuse_scores,
    importance_score,
    kind_boost,
    name_score,
    normalise_scores,
    proximity_score,
    weights_for_query,
)
from rbtr.index.store import IndexStore
from rbtr.index.tokenise import tokenise_code

# ── normalise_scores ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        ([], []),
        ([5.0], [0.0]),
        ([1.0, 3.0], [0.0, 1.0]),
        ([3.0, 1.0], [1.0, 0.0]),
        ([1.0, 2.0, 3.0], [0.0, 0.5, 1.0]),
        # All same → no signal.
        ([7.0, 7.0, 7.0], [0.0, 0.0, 0.0]),
        # Negative values work.
        ([-1.0, 0.0, 1.0], [0.0, 0.5, 1.0]),
    ],
    ids=lambda v: str(v)[:40],
)
def test_normalise_scores(scores: list[float], expected: list[float]) -> None:
    result = normalise_scores(scores)
    assert len(result) == len(expected)
    for r, e in zip(result, expected, strict=True):
        assert r == pytest.approx(e)


# ── name_score ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("query", "name", "expected"),
    [
        # Exact match (case-insensitive).
        ("IndexStore", "IndexStore", 1.0),
        ("indexstore", "IndexStore", 1.0),
        ("INDEXSTORE", "IndexStore", 1.0),
        # Prefix match.
        ("Index", "IndexStore", 0.8),
        ("index", "IndexStore", 0.8),
        # Substring match.
        ("Store", "IndexStore", 0.5),
        ("store", "IndexStore", 0.5),
        # No match.
        ("xyz", "IndexStore", 0.0),
        ("store", "insert_chunks", 0.0),
        # Single character.
        ("I", "IndexStore", 0.8),
        ("e", "IndexStore", 0.5),  # 'e' is a substring of 'indexstore'
        # Token-level match: every query token is a substring of name.
        ("import edge", "infer_import_edges", 0.4),
        ("import edges", "infer_import_edges", 0.4),
        ("deep merge", "_deep_merge", 0.4),
        ("build model", "build_model", 0.4),  # token-level (space ≠ underscore)
        # Token-level: partial token match → 0.0.
        ("store chunk", "insert_chunks", 0.0),  # 'store' not in 'insert_chunks'
        ("import xyz", "infer_import_edges", 0.0),
        ("foo edge", "infer_import_edges", 0.0),
        # Single-word queries: substring check, not token-level.
        ("import", "infer_import_edges", 0.5),  # 'import' is a substring
    ],
    ids=lambda v: str(v)[:30],
)
def test_name_score(query: str, name: str, expected: float) -> None:
    assert name_score(query, name) == pytest.approx(expected)


# ── classify_query ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        # Identifiers: single token, may contain camelCase/snake_case/dots.
        ("IndexStore", QueryKind.IDENTIFIER),
        ("build_model", QueryKind.IDENTIFIER),
        ("getUserById", QueryKind.IDENTIFIER),
        ("_deep_merge", QueryKind.IDENTIFIER),
        ("Config", QueryKind.IDENTIFIER),
        ("a.b.c", QueryKind.IDENTIFIER),
        # Concepts: multi-word, no regex metacharacters.
        ("how does auth work", QueryKind.CONCEPT),
        ("database storage for session", QueryKind.CONCEPT),
        ("import edge", QueryKind.CONCEPT),
        ("store chunk", QueryKind.CONCEPT),
        ("class config", QueryKind.CONCEPT),
        # Patterns: regex metacharacters present.
        ("def test_", QueryKind.CONCEPT),  # no regex chars, just concept
        ("def test_.*", QueryKind.PATTERN),
        ("raise.*Error", QueryKind.PATTERN),
        ("TODO:", QueryKind.CONCEPT),  # colon is not a regex metachar
        ("foo(bar)", QueryKind.PATTERN),
        ("^import", QueryKind.PATTERN),
        ("config\\.toml", QueryKind.PATTERN),
    ],
    ids=lambda v: str(v)[:30],
)
def test_classify_query(query: str, expected: QueryKind) -> None:
    assert classify_query(query) == expected


def test_weights_for_query_sum_to_one() -> None:
    """All weight tuples are valid convex combinations."""
    for q in ["IndexStore", "how does auth work", "raise.*Error"]:
        a, b, g = weights_for_query(q)
        assert pytest.approx(1.0) == a + b + g


def test_weights_identifier_favours_name() -> None:
    """Identifier queries assign highest weight to name match."""
    a, b, g = weights_for_query("IndexStore")
    assert g > a
    assert g > b


def test_weights_concept_favours_name() -> None:
    """Concept queries assign highest weight to name match."""
    a, b, g = weights_for_query("how does auth work")
    assert g >= a
    assert g >= b


# ── kind_boost ───────────────────────────────────────────────────────


def test_kind_boost_covers_all_kinds() -> None:
    """Every ChunkKind has an explicit boost value — no missing kinds."""
    for kind in ChunkKind:
        boost = kind_boost(kind)
        assert isinstance(boost, float)
        assert boost > 0.0, f"{kind} has zero or negative boost"


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        (ChunkKind.CLASS, 1.5),
        (ChunkKind.FUNCTION, 1.3),
        (ChunkKind.METHOD, 1.3),
        (ChunkKind.TEST_FUNCTION, 0.7),
        (ChunkKind.IMPORT, 0.3),
        (ChunkKind.RAW_CHUNK, 0.5),
        (ChunkKind.DOC_SECTION, 0.8),
        (ChunkKind.CONFIG_KEY, 0.6),
        (ChunkKind.VARIABLE, 1.0),
        (ChunkKind.MIGRATION, 0.4),
        (ChunkKind.API_ENDPOINT, 1.0),
    ],
)
def test_kind_boost_values(kind: ChunkKind, expected: float) -> None:
    assert kind_boost(kind) == pytest.approx(expected)


# ── file_category ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        # Source files.
        ("src/rbtr/index/store.py", FileCategory.SOURCE),
        ("lib/main.go", FileCategory.SOURCE),
        ("app/models/user.rb", FileCategory.SOURCE),
        # Test files.
        ("src/tests/index/test_store.py", FileCategory.TEST),
        ("tests/test_handler.py", FileCategory.TEST),
        ("src/rbtr/store_test.go", FileCategory.TEST),
        ("src/tests/conftest.py", FileCategory.TEST),
        # Vendor.
        ("vendor/lib/foo.py", FileCategory.VENDOR),
        ("node_modules/react/index.js", FileCategory.VENDOR),
        ("third_party/proto/api.proto", FileCategory.VENDOR),
        # Generated.
        ("api/models_pb2.py", FileCategory.GENERATED),
        ("proto/service.pb.go", FileCategory.GENERATED),
        # Docs.
        ("docs/guide.md", FileCategory.DOC),
        ("README.md", FileCategory.DOC),
        ("CHANGELOG.rst", FileCategory.DOC),
        # Config.
        ("pyproject.toml", FileCategory.CONFIG),
        (".github/workflows/ci.yaml", FileCategory.CONFIG),
        ("package.json", FileCategory.CONFIG),
        ("config/settings.yml", FileCategory.CONFIG),
    ],
    ids=lambda v: str(v)[:40],
)
def test_file_category(path: str, expected: FileCategory) -> None:
    assert file_category(path) == expected


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("src/rbtr/index/store.py", 1.0),
        ("src/tests/index/test_store.py", 0.5),
        ("vendor/lib/foo.py", 0.3),
        ("docs/guide.md", 0.8),
        ("pyproject.toml", 0.7),
    ],
)
def test_file_category_penalty(path: str, expected: float) -> None:
    assert file_category_penalty(path) == pytest.approx(expected)


# ── importance_score ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("degree", "expected"),
    [
        (0, 1.0),
        (1, pytest.approx(1.25, abs=0.01)),
        (3, pytest.approx(1.50, abs=0.01)),
        (7, pytest.approx(1.75, abs=0.01)),
        (15, pytest.approx(2.00, abs=0.01)),
        (1000, 3.0),  # capped
    ],
    ids=lambda v: str(v),
)
def test_importance_score(degree: int, expected: float) -> None:
    assert importance_score(degree) == expected


def test_importance_score_monotonic() -> None:
    """Higher degree always gives higher or equal score."""
    prev = importance_score(0)
    for d in range(1, 50):
        curr = importance_score(d)
        assert curr >= prev
        prev = curr


# ── proximity_score ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("chunk_file", "changed", "has_edge", "expected"),
    [
        # In changed file.
        ("src/store.py", {"src/store.py"}, False, 1.5),
        # Has edge to changed file.
        ("src/orchestrator.py", {"src/store.py"}, True, 1.2),
        # Same directory, no edge.
        ("src/models.py", {"src/store.py"}, False, 1.1),
        # Different directory, no edge.
        ("cli/main.py", {"src/store.py"}, False, 1.0),
        # Empty changed set.
        ("src/store.py", set(), False, 1.0),
        # Root-level files — not considered "same directory".
        ("README.md", {"setup.py"}, False, 1.0),
    ],
    ids=lambda v: str(v)[:30],
)
def test_proximity_score(
    chunk_file: str,
    changed: set[str],
    has_edge: bool,
    expected: float,
) -> None:
    assert proximity_score(chunk_file, changed, has_edge) == expected
