"""Tests for unified search scoring and fusion.

Pure function tests first, then integration tests that verify
the full pipeline produces correct rankings.
"""

from __future__ import annotations

import polars as pl
import pytest

from rbtr.index.models import ChunkKind
from rbtr.index.search import (
    _file_category_penalty_expr,
    _importance_expr,
    _kind_boost_expr,
    _name_score_expr,
)

# ── _name_score_expr ─────────────────────────────────────────────────


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
def test_name_score_expr(query: str, name: str, expected: float) -> None:
    result = pl.DataFrame({"name": [name]}).select(_name_score_expr(query).alias("score")).item()
    assert result == pytest.approx(expected)


# ── _kind_boost_expr ─────────────────────────────────────────────────


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
def test_kind_boost_expr(kind: ChunkKind, expected: float) -> None:
    result = (
        pl.DataFrame(
            {"kind": [k.value for k in ChunkKind]},
            schema={"kind": pl.Categorical},
        )
        .select(_kind_boost_expr().alias("boost"))
        .filter(pl.Series([k == kind for k in ChunkKind]))
        .item()
    )
    assert result > 0.0
    assert result == pytest.approx(expected)


# ── _file_category_penalty_expr ──────────────────────────────────────


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        # Source.
        ("src/rbtr/index/store.py", 1.0),
        ("lib/main.go", 1.0),
        ("app/models/user.rb", 1.0),
        # Test.
        ("src/tests/index/test_store.py", 0.5),
        ("tests/test_handler.py", 0.5),
        ("src/rbtr/store_test.go", 0.5),
        ("src/tests/conftest.py", 0.5),
        # Vendor.
        ("vendor/lib/foo.py", 0.3),
        ("node_modules/react/index.js", 0.3),
        ("third_party/proto/api.proto", 0.3),
        # Generated.
        ("api/models_pb2.py", 0.3),
        ("proto/service.pb.go", 0.3),
        # Doc.
        ("docs/guide.md", 0.8),
        ("README.md", 0.8),
        ("CHANGELOG.rst", 0.8),
        # Config.
        ("pyproject.toml", 0.7),
        (".github/workflows/ci.yaml", 0.7),
        ("package.json", 0.7),
        ("config/settings.yml", 0.7),
    ],
    ids=lambda v: str(v)[:40],
)
def test_file_category_penalty_expr(path: str, expected: float) -> None:
    result = (
        pl.DataFrame({"file_path": [path]})
        .select(_file_category_penalty_expr().alias("penalty"))
        .item()
    )
    assert result == pytest.approx(expected)


# ── _importance_expr ─────────────────────────────────────────────────


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
def test_importance_expr(degree: int, expected: float) -> None:
    result = pl.DataFrame({"degree": [degree]}).select(_importance_expr().alias("imp")).item()
    assert result == expected


def test_importance_expr_monotonic() -> None:
    """Higher degree always gives higher or equal score."""
    result = (
        pl.DataFrame({"degree": list(range(50))})
        .select(_importance_expr().alias("imp"))
        .to_series()
    )
    for i in range(1, len(result)):
        assert result[i] >= result[i - 1]
