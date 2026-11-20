"""Rust extraction tests (cases in `cases_extraction.py`)."""

from __future__ import annotations

from pytest_cases import parametrize_with_cases

from rbtr.domain.models import (
    ChunkKind,
    EdgeKind,
    ImportMeta,
)
from rbtr.testing import Indexed
from rbtr.testing.assertions import Referrer, Target, assert_reached
from rbtr.testing.reads import outline, read


@parametrize_with_cases("files, expected", cases=".cases_extraction", has_tag="symbol")
def test_extracts_expected_symbols(indexed: Indexed, files: dict[str, str], expected: list) -> None:
    """Each expected (kind, name, scope) tuple appears in the outline."""
    with indexed(files) as index:
        reply = outline(index)
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
    with indexed(files) as index:
        reply = outline(index)
    kinds = {s.kind for s in reply}
    for kind in expected_kinds:
        assert kind in kinds, f"expected kind {kind!r} not in {kinds}"
    methods = [(s.name, s.scope) for s in reply if s.kind == ChunkKind.METHOD]
    for name, scope in expected_methods:
        assert (name, scope) in methods, f"expected method ({name}, {scope}) not in {methods}"


@parametrize_with_cases("files, expected", cases=".cases_extraction", has_tag="import")
def test_extracts_import_metadata(indexed: Indexed, files: dict[str, str], expected: dict) -> None:
    """The first import has the expected metadata."""
    with indexed(files) as index:
        reply = outline(index)
    imports = [s.metadata for s in reply if s.kind == ChunkKind.IMPORT]
    assert imports, f"no import extracted from {files!r}"
    assert imports[0] == ImportMeta(**expected)


def test_rust_impl_captures_struct_and_impl(indexed: Indexed) -> None:
    """Both struct and impl produce class chunks for the same type."""
    src = """\
struct Svc {}
impl Svc {
    fn new() -> Self { Svc {} }
}
"""
    with indexed({"input.rs": src}) as index:
        chunks = outline(index)
    svc_classes = [c for c in chunks if c.kind == ChunkKind.CLASS and c.name == "Svc"]
    assert len(svc_classes) == 2  # struct + impl


def test_rust_attribute_belongs_to_the_item_below_it(indexed: Indexed) -> None:
    """A derive list is searchable, as part of the type it describes.

    An attribute is a sibling of its item in the tree, so without this it
    sits in no chunk and `#[derive(Serialize)]` cannot be found at all.
    """
    src = """\
#![allow(dead_code)]

#[derive(Serialize, Debug)]
pub struct Config {
    name: String,
}
"""
    with indexed({"input.rs": src}) as index:
        chunks = outline(index)
        [struct] = read(index, "Config")
    assert struct.line_start == 3
    assert "#[derive(Serialize, Debug)]" in struct.content
    assert [(c.line_start, c.line_end) for c in chunks if c.kind == ChunkKind.COMMENT] == [(1, 1)]


@parametrize_with_cases("files, reached", cases=".cases_extraction", has_tag="references")
def test_a_call_or_base_reaches_what_it_names(
    indexed: Indexed,
    files: dict[str, str],
    reached: dict[Target, set[Referrer]],
) -> None:
    """Each target `(path, symbol)` is reached by exactly the callers and subclasses named."""
    with indexed(files) as index:
        assert_reached(index, reached, edges={EdgeKind.CALLS, EdgeKind.INHERITS})
