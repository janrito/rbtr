"""Python extraction tests.

Symbol, import, and mixed cases (`cases_extraction.py`) drive the shared
checks; the functions below pin Python's module-variable, function-local,
class-attribute, and tuple-unpacking edge behaviour.
"""

from __future__ import annotations

from pytest_cases import parametrize_with_cases

from rbtr.git import FileEntry
from rbtr.index.models import ChunkKind, ImportMeta
from rbtr.languages.extract import extract_file


@parametrize_with_cases("lang, source, expected", cases=".cases_extraction", has_tag="symbol")
def test_extracts_expected_symbols(lang: str, source: str, expected: list) -> None:
    """Each expected (kind, name, scope) tuple appears in the output."""
    chunks = extract_file(FileEntry("input", "sha1", source.encode()), lang)
    symbols = [(c.kind, c.name, c.scope) for c in chunks]
    for exp in expected:
        assert exp in symbols, f"expected {exp} not found in {symbols}"


@parametrize_with_cases(
    "lang, source, expected_kinds, expected_methods", cases=".cases_extraction", has_tag="mixed"
)
def test_extracts_all_expected_kinds(
    lang: str, source: str, expected_kinds: set[str], expected_methods: list[tuple[str, str]]
) -> None:
    """Realistic source produces all expected chunk kinds and method scoping."""
    chunks = extract_file(FileEntry("input", "sha1", source.encode()), lang)
    kinds = {c.kind for c in chunks}
    for kind in expected_kinds:
        assert kind in kinds, f"expected kind {kind!r} not in {kinds}"
    methods = [(c.name, c.scope) for c in chunks if c.kind == ChunkKind.METHOD]
    for name, scope in expected_methods:
        assert (name, scope) in methods, f"expected method ({name}, {scope}) not in {methods}"


@parametrize_with_cases("lang, source, expected", cases=".cases_extraction", has_tag="import")
def test_extracts_import_metadata(lang: str, source: str, expected: dict) -> None:
    """First import chunk has the expected metadata."""
    chunks = extract_file(FileEntry("input", "sha1", source.encode()), lang)
    imports = [c for c in chunks if c.kind == ChunkKind.IMPORT]
    assert len(imports) >= 1, f"no import chunks extracted from {source!r}"
    assert imports[0].metadata == ImportMeta(**expected)


@parametrize_with_cases(
    "lang, source, count, metadata_list", cases=".cases_extraction", has_tag="multi_import"
)
def test_extracts_multi_import(
    lang: str, source: str, count: int, metadata_list: list[dict]
) -> None:
    """Multiple imports have correct count and per-import metadata."""
    chunks = extract_file(FileEntry("input", "sha1", source.encode()), lang)
    imports = [c for c in chunks if c.kind == ChunkKind.IMPORT]
    assert len(imports) == count
    for imp, expected in zip(imports, metadata_list, strict=True):
        assert imp.metadata == ImportMeta(**expected)


def test_py_module_variable_content_is_whole_statement() -> None:
    """A module-level VARIABLE chunk spans the whole statement, named by LHS."""
    src = """\
MAX_SIZE = 100
"""
    chunks = extract_file(FileEntry("input", "sha1", src.encode()), "python")
    variables = [c for c in chunks if c.kind == ChunkKind.VARIABLE]
    assert len(variables) == 1
    assert variables[0].name == "MAX_SIZE"
    assert variables[0].content.strip() == "MAX_SIZE = 100"


def test_py_function_local_not_captured_as_variable() -> None:
    """Assignments inside a function stay part of the function chunk."""
    src = """\
def f():
    tmp = 1
    return tmp
"""
    chunks = extract_file(FileEntry("input", "sha1", src.encode()), "python")
    assert [c for c in chunks if c.kind == ChunkKind.VARIABLE] == []


def test_py_class_attribute_not_captured_as_variable() -> None:
    """Class-body attributes stay part of the class chunk, not VARIABLE chunks."""
    src = """\
class Config:
    DEFAULT = 30
"""
    chunks = extract_file(FileEntry("input", "sha1", src.encode()), "python")
    assert [c for c in chunks if c.kind == ChunkKind.VARIABLE] == []


def test_py_tuple_unpacking_captured_as_variables() -> None:
    """Flat tuple-unpacking binds each target as its own VARIABLE chunk.

    Both names come from one statement (tree-sitter fans the destructuring
    into a match per identifier), and each chunk spans the whole statement.
    """
    src = """\
a, b = compute()
"""
    chunks = extract_file(FileEntry("input", "sha1", src.encode()), "python")
    variables = [c for c in chunks if c.kind == ChunkKind.VARIABLE]
    assert {c.name for c in variables} == {"a", "b"}
    assert all(c.content.strip() == "a, b = compute()" for c in variables)
