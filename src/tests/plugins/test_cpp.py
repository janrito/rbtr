"""Tests for the C++ language plugin."""

from __future__ import annotations

import pytest
from tree_sitter import Language, Parser

from rbtr.index.models import ChunkKind
from rbtr.index.treesitter import extract_symbols
from rbtr.plugins.cpp import CppPlugin, extract_import_meta

# ── Helpers ──────────────────────────────────────────────────────────

_HAS_GRAMMAR = True
try:
    import tree_sitter_cpp

    _LANG = Language(tree_sitter_cpp.language())
except ImportError:
    _HAS_GRAMMAR = False
    _LANG = None  # type: ignore[assignment]

needs_grammar = pytest.mark.skipif(not _HAS_GRAMMAR, reason="tree-sitter-cpp not installed")


def _extract(code: str):
    reg = CppPlugin().rbtr_register_languages()[0]
    return extract_symbols(
        "test.cpp",
        "abc123",
        code.encode(),
        _LANG,
        reg.query,
        import_extractor=reg.import_extractor,
        scope_types=reg.scope_types,
    )


def _parse_import(code: str):
    """Parse code and extract import metadata from the first include."""
    parser = Parser(_LANG)
    tree = parser.parse(code.encode())
    for child in tree.root_node.children:
        if child.type == "preproc_include":
            return extract_import_meta(child)
    return {}


# ── Registration ─────────────────────────────────────────────────────


def test_registration():
    regs = CppPlugin().rbtr_register_languages()
    assert len(regs) == 1
    reg = regs[0]
    assert reg.id == "cpp"
    assert ".cpp" in reg.extensions
    assert ".cc" in reg.extensions
    assert ".hpp" in reg.extensions
    assert reg.grammar_module == "tree_sitter_cpp"
    assert reg.query is not None
    assert reg.import_extractor is not None
    assert "class_specifier" in reg.scope_types
    assert "struct_specifier" in reg.scope_types


# ── Include extraction ───────────────────────────────────────────────


@needs_grammar
@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("#include <iostream>\n", {"module": "iostream"}),
        ('#include "myheader.h"\n', {"module": "myheader.h"}),
        ("#include <boost/optional.hpp>\n", {"module": "boost/optional.hpp"}),
    ],
    ids=["system", "local", "nested-path"],
)
def test_include_meta(code: str, expected: dict):
    assert _parse_import(code) == expected


# ── Free function extraction ─────────────────────────────────────────


@needs_grammar
@pytest.mark.parametrize(
    ("code", "expected_name"),
    [
        ("void greet() { }\n", "greet"),
        ("int compute(int x) { return x; }\n", "compute"),
    ],
    ids=["void", "returning"],
)
def test_free_function(code: str, expected_name: str):
    fns = [c for c in _extract(code) if c.kind == ChunkKind.FUNCTION]
    assert len(fns) == 1
    assert fns[0].name == expected_name
    assert fns[0].scope == ""


@needs_grammar
def test_multiple_functions():
    chunks = _extract("""\
int foo() { return 0; }
void bar() { }
""")
    names = {c.name for c in chunks if c.kind == ChunkKind.FUNCTION}
    assert names == {"foo", "bar"}


# ── Class-like extraction ────────────────────────────────────────────


@needs_grammar
@pytest.mark.parametrize(
    ("code", "expected_name"),
    [
        ("class Shape { };\n", "Shape"),
        ("struct Point { double x; double y; };\n", "Point"),
        ("enum class Color { Red, Green, Blue };\n", "Color"),
    ],
    ids=["class", "struct", "enum-class"],
)
def test_class_like(code: str, expected_name: str):
    classes = [c for c in _extract(code) if c.kind == ChunkKind.CLASS]
    assert len(classes) == 1
    assert classes[0].name == expected_name


@needs_grammar
def test_class_with_inheritance():
    chunks = _extract("""\
class Base { };
class Derived : public Base { };
""")
    names = {c.name for c in chunks if c.kind == ChunkKind.CLASS}
    assert names == {"Base", "Derived"}


# ── Method extraction (scoped) ───────────────────────────────────────


@needs_grammar
@pytest.mark.parametrize(
    ("code", "method_name", "scope"),
    [
        (
            """\
class Foo {
public:
    void bar() { }
};
""",
            "bar",
            "Foo",
        ),
        (
            "struct Vec { void push(int v) { } };\n",
            "push",
            "Vec",
        ),
    ],
    ids=["class-method", "struct-method"],
)
def test_method_scoped(code: str, method_name: str, scope: str):
    methods = [c for c in _extract(code) if c.kind == ChunkKind.METHOD]
    assert len(methods) == 1
    assert methods[0].name == method_name
    assert methods[0].scope == scope


@needs_grammar
def test_multiple_methods():
    chunks = _extract("""\
class Calculator {
public:
    int add(int a, int b) { return a + b; }
    int sub(int a, int b) { return a - b; }
};
""")
    methods = [c for c in chunks if c.kind == ChunkKind.METHOD]
    names = {m.name for m in methods}
    assert names == {"add", "sub"}
    assert all(m.scope == "Calculator" for m in methods)


@needs_grammar
def test_free_function_not_scoped():
    """Functions outside classes should not be methods."""
    chunks = _extract("""\
class C { };
void standalone() { }
""")
    fns = [c for c in chunks if c.kind == ChunkKind.FUNCTION]
    assert len(fns) == 1
    assert fns[0].name == "standalone"
    assert fns[0].scope == ""


# ── Full file ────────────────────────────────────────────────────────


@needs_grammar
def test_full_cpp_file():
    chunks = _extract("""\
#include <vector>
#include "config.h"

class Engine {
public:
    void start() { }
    void stop() { }
};

struct Options {
    int timeout;
};

enum class Mode { Fast, Safe };

void run(Engine& e) {
    e.start();
}
""")
    imports = [c for c in chunks if c.kind == ChunkKind.IMPORT]
    classes = [c for c in chunks if c.kind == ChunkKind.CLASS]
    methods = [c for c in chunks if c.kind == ChunkKind.METHOD]
    fns = [c for c in chunks if c.kind == ChunkKind.FUNCTION]

    assert len(imports) == 2
    assert {c.metadata.get("module") for c in imports} == {"vector", "config.h"}
    assert {c.name for c in classes} == {"Engine", "Options", "Mode"}
    assert {c.name for c in methods} == {"start", "stop"}
    assert all(m.scope == "Engine" for m in methods)
    assert len(fns) == 1
    assert fns[0].name == "run"
