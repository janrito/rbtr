"""Tests for the Rust language plugin.

Covers functions, structs, enums, impl blocks, and all
`use` declaration forms including `super` and `crate`.
Skipped when tree-sitter-rust is not installed.
"""

from __future__ import annotations

import pytest

from rbtr.index.models import ChunkKind
from rbtr.index.treesitter import extract_symbols
from rbtr.plugins.manager import get_manager

# ── Grammar availability ─────────────────────────────────────────────

_mgr = get_manager()
_has_rust = _mgr.load_grammar("rust") is not None
skip_no_rust = pytest.mark.skipif(not _has_rust, reason="tree-sitter-rust not installed")
pytestmark = skip_no_rust

# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def grammar():
    g = _mgr.load_grammar("rust")
    assert g is not None
    return g


@pytest.fixture
def registration():
    reg = _mgr.get_registration("rust")
    assert reg is not None
    return reg


@pytest.fixture
def query(registration):
    assert registration.query is not None
    return registration.query


@pytest.fixture
def extractor(registration):
    assert registration.import_extractor is not None
    return registration.import_extractor


@pytest.fixture
def scope_types(registration):
    return registration.scope_types


# ── Helpers ──────────────────────────────────────────────────────────


def _extract(source, grammar, query, extractor, scope_types, file_path="src/lib.rs"):
    return extract_symbols(
        file_path,
        "sha1",
        source.encode(),
        grammar,
        query,
        import_extractor=extractor,
        scope_types=scope_types,
    )


def _symbols(source, grammar, query, extractor, scope_types, file_path="src/lib.rs"):
    return [
        (c.kind, c.name, c.scope)
        for c in _extract(source, grammar, query, extractor, scope_types, file_path)
    ]


def _imports(source, grammar, query, extractor, scope_types, file_path="src/lib.rs"):
    return [
        c
        for c in _extract(source, grammar, query, extractor, scope_types, file_path)
        if c.kind == ChunkKind.IMPORT
    ]


# ── Registration ─────────────────────────────────────────────────────


def test_registration_exists(registration) -> None:
    assert registration.id == "rust"


def test_extensions(registration) -> None:
    assert ".rs" in registration.extensions


def test_grammar_module(registration) -> None:
    assert registration.grammar_module == "tree_sitter_rust"


def test_scope_types(registration) -> None:
    assert "impl_item" in registration.scope_types
    assert "struct_item" in registration.scope_types


# ── Function extraction ──────────────────────────────────────────────


def test_extract_function(grammar, query, extractor, scope_types) -> None:
    syms = _symbols("fn hello() {}\n", grammar, query, extractor, scope_types)
    assert ("function", "hello", "") in syms


def test_extract_function_with_params(grammar, query, extractor, scope_types) -> None:
    src = "fn add(a: i32, b: i32) -> i32 { a + b }\n"
    syms = _symbols(src, grammar, query, extractor, scope_types)
    assert ("function", "add", "") in syms


def test_extract_pub_function(grammar, query, extractor, scope_types) -> None:
    syms = _symbols("pub fn visible() {}\n", grammar, query, extractor, scope_types)
    assert ("function", "visible", "") in syms


def test_extract_async_function(grammar, query, extractor, scope_types) -> None:
    syms = _symbols("async fn fetch() {}\n", grammar, query, extractor, scope_types)
    assert ("function", "fetch", "") in syms


def test_extract_multiple_functions(grammar, query, extractor, scope_types) -> None:
    src = """\
fn a() {}
fn b() {}
fn c() {}
"""
    names = [
        s[1] for s in _symbols(src, grammar, query, extractor, scope_types) if s[0] == "function"
    ]
    assert names == ["a", "b", "c"]


# ── Struct extraction ────────────────────────────────────────────────


def test_extract_struct(grammar, query, extractor, scope_types) -> None:
    src = """\
struct User {
    name: String,
}
"""
    syms = _symbols(src, grammar, query, extractor, scope_types)
    assert ("class", "User", "") in syms


def test_extract_unit_struct(grammar, query, extractor, scope_types) -> None:
    syms = _symbols("struct Marker;\n", grammar, query, extractor, scope_types)
    assert ("class", "Marker", "") in syms


def test_extract_tuple_struct(grammar, query, extractor, scope_types) -> None:
    syms = _symbols("struct Point(f64, f64);\n", grammar, query, extractor, scope_types)
    assert ("class", "Point", "") in syms


# ── Enum extraction ──────────────────────────────────────────────────


def test_extract_enum(grammar, query, extractor, scope_types) -> None:
    src = """\
enum Color {
    Red,
    Green,
    Blue,
}
"""
    syms = _symbols(src, grammar, query, extractor, scope_types)
    assert ("class", "Color", "") in syms


# ── Impl block extraction ───────────────────────────────────────────


def test_extract_impl(grammar, query, extractor, scope_types) -> None:
    src = """\
struct Svc {}
impl Svc {
    fn new() -> Self { Svc {} }
}
"""
    syms = _symbols(src, grammar, query, extractor, scope_types)
    assert ("class", "Svc", "") in syms
    # impl Svc is also captured as a class-like scope.
    impl_names = [s[1] for s in syms if s[0] == "class"]
    assert impl_names.count("Svc") == 2  # struct + impl


def test_method_in_impl(grammar, query, extractor, scope_types) -> None:
    src = """\
struct Svc {}
impl Svc {
    fn start(&self) {}
    fn stop(&self) {}
}
"""
    syms = _symbols(src, grammar, query, extractor, scope_types)
    methods = [(s[1], s[2]) for s in syms if s[0] == "method"]
    assert ("start", "Svc") in methods
    assert ("stop", "Svc") in methods


# ── Import: scoped identifier ────────────────────────────────────────


def test_import_scoped(grammar, query, extractor, scope_types) -> None:
    imp = _imports("use std::collections::HashMap;\n", grammar, query, extractor, scope_types)[0]
    assert imp.metadata == {"module": "std/collections", "names": "HashMap"}


def test_import_deeply_nested(grammar, query, extractor, scope_types) -> None:
    imp = _imports("use a::b::c::d::Item;\n", grammar, query, extractor, scope_types)[0]
    assert imp.metadata == {"module": "a/b/c/d", "names": "Item"}


# ── Import: crate-relative ───────────────────────────────────────────


def test_import_crate_simple(grammar, query, extractor, scope_types) -> None:
    imp = _imports("use crate::models::Chunk;\n", grammar, query, extractor, scope_types)[0]
    assert imp.metadata == {"module": "crate/models", "names": "Chunk"}


def test_import_crate_with_braces(grammar, query, extractor, scope_types) -> None:
    imp = _imports("use crate::models::{Chunk, Edge};\n", grammar, query, extractor, scope_types)[0]
    assert imp.metadata["module"] == "crate/models"
    assert set(imp.metadata["names"].split(",")) == {"Chunk", "Edge"}


# ── Import: super-relative ───────────────────────────────────────────


def test_import_super_single(grammar, query, extractor, scope_types) -> None:
    imp = _imports("use super::utils;\n", grammar, query, extractor, scope_types)[0]
    assert imp.metadata == {"names": "utils", "dots": "2"}


def test_import_super_nested(grammar, query, extractor, scope_types) -> None:
    imp = _imports("use super::helpers::run;\n", grammar, query, extractor, scope_types)[0]
    assert imp.metadata == {"module": "helpers", "names": "run", "dots": "2"}


def test_import_super_super(grammar, query, extractor, scope_types) -> None:
    imp = _imports("use super::super::common::Config;\n", grammar, query, extractor, scope_types)[0]
    assert imp.metadata["dots"] == "3"
    assert imp.metadata["module"] == "common"
    assert imp.metadata["names"] == "Config"


# ── Import: scoped use list ──────────────────────────────────────────


def test_import_use_list(grammar, query, extractor, scope_types) -> None:
    imp = _imports("use std::io::{Read, Write};\n", grammar, query, extractor, scope_types)[0]
    assert imp.metadata["module"] == "std/io"
    assert set(imp.metadata["names"].split(",")) == {"Read", "Write"}


def test_import_use_list_with_self(grammar, query, extractor, scope_types) -> None:
    imp = _imports("use std::io::{self, Read};\n", grammar, query, extractor, scope_types)[0]
    assert imp.metadata["module"] == "std/io"
    assert "self" in imp.metadata["names"]
    assert "Read" in imp.metadata["names"]


# ── Import: bare identifier ──────────────────────────────────────────


def test_import_bare(grammar, query, extractor, scope_types) -> None:
    imp = _imports("use serde;\n", grammar, query, extractor, scope_types)[0]
    assert imp.metadata == {"module": "serde"}


# ── Mixed extraction ─────────────────────────────────────────────────


def test_full_module(grammar, query, extractor, scope_types) -> None:
    src = """\
use std::collections::HashMap;
use crate::models::Config;

struct App {
    config: Config,
}

enum Status {
    Running,
    Stopped,
}

impl App {
    fn new(config: Config) -> Self {
        App { config }
    }
    fn run(&self) {}
}

fn main() {}
"""
    chunks = _extract(src, grammar, query, extractor, scope_types)
    kinds = {c.kind for c in chunks}
    assert ChunkKind.IMPORT in kinds
    assert ChunkKind.CLASS in kinds
    assert ChunkKind.METHOD in kinds
    assert ChunkKind.FUNCTION in kinds

    # Verify method scoping.
    methods = [(c.name, c.scope) for c in chunks if c.kind == ChunkKind.METHOD]
    assert ("new", "App") in methods
    assert ("run", "App") in methods


def test_empty_source(grammar, query, extractor, scope_types) -> None:
    chunks = _extract("", grammar, query, extractor, scope_types)
    assert chunks == []
