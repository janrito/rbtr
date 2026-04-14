"""Tests for tree-sitter symbol extraction across all languages.

Thin behavioral assertions — all test data lives in
``case_extraction.py``.
"""

from __future__ import annotations

from pytest_cases import parametrize_with_cases

from rbtr.index.models import ChunkKind
from rbtr.index.treesitter import extract_symbols
from rbtr.languages.manager import get_manager
from tests.plugins.conftest import extract_chunks, skip_unless_grammar

# ── Symbol extraction ────────────────────────────────────────────────


@parametrize_with_cases(
    "lang, source, expected",
    cases="tests.plugins.case_extraction",
    has_tag="symbol",
)
def test_extracts_expected_symbols(lang: str, source: str, expected: list) -> None:
    """Each expected (kind, name, scope) tuple appears in the output."""
    chunks = extract_chunks(lang, source)
    symbols = [(c.kind, c.name, c.scope) for c in chunks]
    for exp in expected:
        assert exp in symbols, f"expected {exp} not found in {symbols}"


# ── Mixed extraction ─────────────────────────────────────────────────


@parametrize_with_cases(
    "lang, source, expected_kinds, expected_methods",
    cases="tests.plugins.case_extraction",
    has_tag="mixed",
)
def test_extracts_all_expected_kinds(
    lang: str,
    source: str,
    expected_kinds: set[str],
    expected_methods: list[tuple[str, str]],
) -> None:
    """Realistic source produces all expected chunk kinds and method scoping."""
    chunks = extract_chunks(lang, source)
    kinds = {c.kind for c in chunks}
    for kind in expected_kinds:
        assert kind in kinds, f"expected kind {kind!r} not in {kinds}"
    methods = [(c.name, c.scope) for c in chunks if c.kind == ChunkKind.METHOD]
    for name, scope in expected_methods:
        assert (name, scope) in methods, f"expected method ({name}, {scope}) not in {methods}"


# ── Language-specific edge cases ─────────────────────────────────────


@skip_unless_grammar("java")
def test_java_constructor_not_captured() -> None:
    """Java constructors use constructor_declaration, not method_declaration."""
    src = """\
class Foo {
    Foo() {}
}
"""
    chunks = extract_chunks("java", src)
    methods = [c for c in chunks if c.kind == ChunkKind.METHOD and c.name == "Foo"]
    assert methods == []


@skip_unless_grammar("rust")
def test_rust_impl_captures_struct_and_impl() -> None:
    """Both struct and impl produce class chunks for the same type."""
    src = """\
struct Svc {}
impl Svc {
    fn new() -> Self { Svc {} }
}
"""
    chunks = extract_chunks("rust", src)
    svc_classes = [c for c in chunks if c.kind == ChunkKind.CLASS and c.name == "Svc"]
    assert len(svc_classes) == 2  # struct + impl


@skip_unless_grammar("bash")
def test_bash_no_imports_extracted() -> None:
    """source/. commands are not captured as imports."""
    src = """\
source ./env.sh
. /etc/profile
"""
    chunks = extract_chunks("bash", src)
    imports = [c for c in chunks if c.kind == ChunkKind.IMPORT]
    assert imports == []


# ── Coverage gap tests ───────────────────────────────────────────────


def test_anonymous_chunk_when_name_capture_missing() -> None:
    """Chunks get name='<anonymous>' when the query omits the name capture.

    Covers `treesitter.py` line 151.
    """
    mgr = get_manager()
    grammar = mgr.load_grammar("python")
    assert grammar is not None
    # Query captures function_definition as @function but has no @_fn_name.
    query_no_name = "(function_definition) @function\n"
    src = b"""\
def hello():
    pass
"""
    chunks = extract_symbols("test.py", "sha1", src, grammar, query_no_name)
    assert len(chunks) >= 1
    assert chunks[0].name == "<anonymous>"


def test_unknown_capture_name_ignored() -> None:
    """Captures not in _CAPTURE_KIND are silently skipped.

    Covers `treesitter.py` line 132.
    """
    mgr = get_manager()
    grammar = mgr.load_grammar("python")
    assert grammar is not None
    # @unknown_thing is not in _CAPTURE_KIND — should be skipped.
    query_unknown = """\
(function_definition
  name: (identifier) @_fn_name) @function
(class_definition) @unknown_thing
"""
    src = b"""\
def f():
    pass

class C:
    pass
"""
    chunks = extract_symbols("test.py", "sha1", src, grammar, query_unknown)
    # Only the function should be extracted, not the class
    # (class was captured as @unknown_thing, not @class).
    kinds = {c.kind for c in chunks}
    assert "function" in kinds
    assert "class" not in kinds
