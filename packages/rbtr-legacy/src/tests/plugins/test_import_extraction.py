"""Tests for tree-sitter import metadata extraction across all languages.

Thin behavioral assertions — all test data lives in
``case_extraction.py``.
"""

from __future__ import annotations

from pytest_cases import parametrize_with_cases

from rbtr_legacy.index.models import ChunkKind
from tests.plugins.conftest import extract_chunks

# ── Single-import metadata ───────────────────────────────────────────


@parametrize_with_cases(
    "lang, source, expected",
    cases="tests.plugins.case_extraction",
    has_tag="import",
)
def test_extracts_import_metadata(lang: str, source: str, expected: dict) -> None:
    """First import chunk has the expected metadata dict."""
    chunks = extract_chunks(lang, source)
    imports = [c for c in chunks if c.kind == ChunkKind.IMPORT]
    assert len(imports) >= 1, f"no import chunks extracted from {source!r}"
    assert imports[0].metadata == expected


# ── Multi-import ─────────────────────────────────────────────────────


@parametrize_with_cases(
    "lang, source, count, metadata_list",
    cases="tests.plugins.case_extraction",
    has_tag="multi_import",
)
def test_extracts_multi_import(
    lang: str, source: str, count: int, metadata_list: list[dict]
) -> None:
    """Multiple imports have correct count and per-import metadata."""
    chunks = extract_chunks(lang, source)
    imports = [c for c in chunks if c.kind == ChunkKind.IMPORT]
    assert len(imports) == count
    for imp, expected in zip(imports, metadata_list, strict=True):
        assert imp.metadata == expected
