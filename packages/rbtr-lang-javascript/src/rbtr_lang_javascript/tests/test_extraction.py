"""JavaScript / TypeScript / TSX extraction tests.

Symbol, import, and mixed cases (`cases_extraction.py`) drive the shared
checks via pytest-cases.
"""

from __future__ import annotations

from pytest_cases import parametrize_with_cases

from rbtr.domain.models import (
    ChunkKind,
    EdgeKind,
    ImportMeta,
)
from rbtr.testing import Indexed
from rbtr.testing.assertions import Referrer, Target, assert_reached
from rbtr.testing.reads import outline


@parametrize_with_cases("files, expected", cases=".cases_extraction", has_tag="symbol")
def test_extracts_expected_symbols(indexed: Indexed, files: dict[str, str], expected: list) -> None:
    """Each expected (kind, name, scope) tuple appears in the outline."""
    with indexed(files) as build:
        reply = outline(build)
    symbols = [(s.kind, s.name, s.scope) for s in reply]
    for exp in expected:
        assert exp in symbols, f"expected {exp} not found in {symbols}"


@parametrize_with_cases(
    "files, expected_kinds, expected_methods", cases=".cases_extraction", has_tag="mixed"
)
def test_extracts_all_expected_kinds(
    indexed: Indexed,
    files: dict[str, str],
    expected_kinds: set[str],
    expected_methods: list[tuple[str, str]],
) -> None:
    """Realistic source produces all expected chunk kinds and method scoping."""
    with indexed(files) as build:
        reply = outline(build)
    kinds = {s.kind for s in reply}
    for kind in expected_kinds:
        assert kind in kinds, f"expected kind {kind!r} not in {kinds}"
    methods = [(s.name, s.scope) for s in reply if s.kind == ChunkKind.METHOD]
    for name, scope in expected_methods:
        assert (name, scope) in methods, f"expected method ({name}, {scope}) not in {methods}"


@parametrize_with_cases("files, expected", cases=".cases_extraction", has_tag="import")
def test_extracts_import_metadata(indexed: Indexed, files: dict[str, str], expected: dict) -> None:
    """The first import has the expected metadata."""
    with indexed(files) as build:
        reply = outline(build)
    imports = [meta for s in reply for meta in s.imports]
    assert imports, f"no import extracted from {files!r}"
    assert imports[0] == ImportMeta(**expected)


@parametrize_with_cases(
    "files, count, metadata_list", cases=".cases_extraction", has_tag="multi_import"
)
def test_extracts_multi_import(
    indexed: Indexed,
    files: dict[str, str],
    count: int,
    metadata_list: list[dict],
) -> None:
    """Multiple imports have correct count and per-import metadata."""
    with indexed(files) as build:
        reply = outline(build)
    imports = [meta for s in reply for meta in s.imports]
    assert len(imports) == count
    assert imports == [ImportMeta(**expected) for expected in metadata_list]


def test_re_export_is_an_import_carrying_its_module(indexed: Indexed) -> None:
    """A re-export reads from another module, so it resolves like an import.

    It is also how a symbol reaches a consumer that never names the file
    defining it, so the edge matters as much as the text.
    """
    src = """\
export * from "./c";
export { helper } from "./b";
"""
    with indexed({"input.ts": src}) as build:
        chunks = outline(build)
    imports = [(meta.module, c.name) for c in chunks for meta in c.imports]
    assert ("./c", 'export * from "./c";') in imports
    assert ("./b", 'export { helper } from "./b";') in imports


@parametrize_with_cases("files, reached", cases=".cases_extraction", has_tag="references")
def test_a_call_or_base_reaches_what_it_names(
    indexed: Indexed,
    files: dict[str, str],
    reached: dict[Target, set[Referrer]],
) -> None:
    """Each target `(path, symbol)` is reached by exactly the callers and subclasses named."""
    with indexed(files) as build:
        assert_reached(build, reached, edges={EdgeKind.CALLS, EdgeKind.INHERITS})


def test_tests_and_statements_are_chunks_holding_what_is_captured_in_them(indexed: Indexed) -> None:
    """A `test` is one test, its callback's comment inside it; a `describe` holds a helper and an `it`; JSDoc above a method folds into it; a top-level `if` is a statement."""
    src = """\
test("hello", () => {
  // check it
  expect(1).toBe(1);
});

describe("group", () => {
  function helper() {}
  it("works", () => {});
});

class A {
  /** Doc for m. */
  m() {}
}

if (ready) {
  go();
}
"""
    with indexed({"a.ts": src}) as build:
        reply = outline(build)
    assert sorted((s.line_start, s.line_end, s.kind.value, s.name) for s in reply) == [
        (1, 4, "test_function", "hello"),
        (6, 9, "test_function", "group"),
        (7, 7, "function", "helper"),
        (8, 8, "test_function", "works"),
        (11, 14, "class", "A"),
        (12, 13, "method", "m"),
        (16, 18, "statement", ""),
    ]
