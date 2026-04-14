"""Cross-language property tests for tree-sitter extraction.

Each test verifies a universal property of `extract_symbols` across
all languages with grammars.
"""

from __future__ import annotations

import pytest

from rbtr_legacy.index.models import ChunkKind
from tests.plugins.conftest import extract_chunks, skip_unless_grammar


@pytest.fixture(
    params=[
        pytest.param(
            ("python", "def f():\n    pass\n"),
            id="python",
        ),
        pytest.param(
            ("javascript", "function f() {}\n"),
            id="javascript",
            marks=skip_unless_grammar("javascript"),
        ),
        pytest.param(
            ("typescript", "function f(): void {}\n"),
            id="typescript",
            marks=skip_unless_grammar("typescript"),
        ),
        pytest.param(
            ("go", "package main\nfunc f() {}\n"),
            id="go",
            marks=skip_unless_grammar("go"),
        ),
        pytest.param(
            ("rust", "fn f() {}\n"),
            id="rust",
            marks=skip_unless_grammar("rust"),
        ),
        pytest.param(
            ("java", "class C { void f() {} }\n"),
            id="java",
            marks=skip_unless_grammar("java"),
        ),
        pytest.param(
            ("bash", "f() { :; }\n"),
            id="bash",
        ),
        pytest.param(
            ("c", "void f(void) { }\n"),
            id="c",
            marks=skip_unless_grammar("c"),
        ),
        pytest.param(
            ("cpp", "void f() { }\n"),
            id="cpp",
            marks=skip_unless_grammar("cpp"),
        ),
        pytest.param(
            ("ruby", "def f\n  1\nend\n"),
            id="ruby",
            marks=skip_unless_grammar("ruby"),
        ),
    ]
)
def lang_and_source(request: pytest.FixtureRequest) -> tuple[str, str]:
    """Language ID paired with a minimal source snippet.

    The source is the smallest valid snippet that produces at
    least one chunk for the language.
    """
    return request.param


@pytest.fixture
def lang(lang_and_source: tuple[str, str]) -> str:
    """Language ID extracted from the ``lang_and_source`` pair."""
    return lang_and_source[0]


@pytest.fixture
def minimal_source(lang_and_source: tuple[str, str]) -> str:
    """Minimal source extracted from the ``lang_and_source`` pair."""
    return lang_and_source[1]


# ── Universal properties ─────────────────────────────────────────────


def test_empty_source_returns_empty(lang: str) -> None:
    """Empty source produces no chunks, regardless of language."""
    assert extract_chunks(lang, "") == []


def test_deterministic_chunk_ids(lang: str, minimal_source: str) -> None:
    """Extracting the same source twice produces identical chunk IDs."""
    c1 = extract_chunks(lang, minimal_source)
    c2 = extract_chunks(lang, minimal_source)
    assert [c.id for c in c1] == [c.id for c in c2]


def test_blob_sha_propagated(lang: str, minimal_source: str) -> None:
    """All chunks carry the blob_sha passed to extract_symbols."""
    chunks = extract_chunks(lang, minimal_source)
    assert len(chunks) >= 1
    # extract_chunks in conftest uses "sha1" as the blob_sha.
    assert all(c.blob_sha == "sha1" for c in chunks)


def test_non_import_chunks_have_empty_metadata(lang: str, minimal_source: str) -> None:
    """Non-import chunks always have metadata == {}."""
    chunks = extract_chunks(lang, minimal_source)
    for c in chunks:
        if c.kind != ChunkKind.IMPORT:
            assert c.metadata == {}, f"{c.kind} chunk {c.name!r} has metadata {c.metadata}"


def test_line_numbers_are_positive(lang: str, minimal_source: str) -> None:
    """All chunks have positive, 1-indexed line numbers."""
    chunks = extract_chunks(lang, minimal_source)
    for c in chunks:
        assert c.line_start >= 1, f"{c.name} has line_start={c.line_start}"
        assert c.line_end >= c.line_start, f"{c.name} has line_end < line_start"


def test_content_is_nonempty(lang: str, minimal_source: str) -> None:
    """All chunks have non-empty content."""
    chunks = extract_chunks(lang, minimal_source)
    for c in chunks:
        assert c.content, f"{c.kind} chunk {c.name!r} has empty content"


def test_syntax_error_still_extracts_valid_parts(lang: str, minimal_source: str) -> None:
    """Tree-sitter error recovery: valid symbols extracted even with trailing garbage."""
    broken = minimal_source + "\n\x00\x00INVALID{{{[[\n"
    chunks = extract_chunks(lang, broken)
    # The valid part of the source should still produce at least one chunk.
    assert len(chunks) >= 1
