"""Rust extraction tests (cases in `cases_extraction.py`)."""

from __future__ import annotations

from pytest_cases import parametrize_with_cases

from rbtr.domain.models import ChunkKind, ImportMeta
from rbtr.git import FileEntry
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


def test_rust_impl_captures_struct_and_impl() -> None:
    """Both struct and impl produce class chunks for the same type."""
    src = """\
struct Svc {}
impl Svc {
    fn new() -> Self { Svc {} }
}
"""
    chunks = extract_file(FileEntry("input", "sha1", src.encode()), "rust")
    svc_classes = [c for c in chunks if c.kind == ChunkKind.CLASS and c.name == "Svc"]
    assert len(svc_classes) == 2  # struct + impl


def test_rust_attribute_belongs_to_the_item_below_it() -> None:
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
    chunks = list(extract_file(FileEntry("input", "sha1", src.encode()), "rust"))
    struct = next(c for c in chunks if c.name == "Config")
    assert struct.line_start == 3
    assert "#[derive(Serialize, Debug)]" in struct.content
    module_attribute = next(c for c in chunks if c.kind == ChunkKind.COMMENT)
    assert module_attribute.content == "#![allow(dead_code)]"
