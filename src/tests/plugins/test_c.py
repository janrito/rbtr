"""Tests for the C language plugin."""

from __future__ import annotations

import pytest
from tree_sitter import Language, Parser

from rbtr.index.models import ChunkKind
from rbtr.index.treesitter import extract_symbols
from rbtr.plugins.c import CPlugin, extract_import_meta

# ── Helpers ──────────────────────────────────────────────────────────

_HAS_GRAMMAR = True
try:
    import tree_sitter_c

    _LANG = Language(tree_sitter_c.language())
except ImportError:
    _HAS_GRAMMAR = False
    _LANG = None  # type: ignore[assignment]

needs_grammar = pytest.mark.skipif(not _HAS_GRAMMAR, reason="tree-sitter-c not installed")


def _extract(code: str):
    reg = CPlugin().rbtr_register_languages()[0]
    return extract_symbols(
        "test.c",
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
    regs = CPlugin().rbtr_register_languages()
    assert len(regs) == 1
    reg = regs[0]
    assert reg.id == "c"
    assert reg.extensions == frozenset({".c", ".h"})
    assert reg.grammar_module == "tree_sitter_c"
    assert reg.query is not None
    assert reg.import_extractor is not None
    assert reg.scope_types == frozenset()


# ── Include extraction ───────────────────────────────────────────────


@needs_grammar
@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("#include <stdio.h>\n", {"module": "stdio.h"}),
        ('#include "mylib.h"\n', {"module": "mylib.h"}),
        ('#include "utils/helpers.h"\n', {"module": "utils/helpers.h"}),
        ("#include <sys/types.h>\n", {"module": "sys/types.h"}),
    ],
    ids=["system", "local", "nested-path", "system-nested"],
)
def test_include_meta(code: str, expected: dict):
    assert _parse_import(code) == expected


@needs_grammar
def test_multiple_includes():
    chunks = _extract("""\
#include <stdlib.h>
#include "local.h"
""")
    imports = [c for c in chunks if c.kind == ChunkKind.IMPORT]
    assert len(imports) == 2
    assert imports[0].metadata == {"module": "stdlib.h"}
    assert imports[1].metadata == {"module": "local.h"}


# ── Function extraction ─────────────────────────────────────────────


@needs_grammar
@pytest.mark.parametrize(
    ("code", "expected_name"),
    [
        ("int add(int a, int b) { return a + b; }\n", "add"),
        ("void do_stuff(void) { }\n", "do_stuff"),
        ("static int helper(void) { return 1; }\n", "helper"),
    ],
    ids=["basic", "void", "static"],
)
def test_function(code: str, expected_name: str):
    fns = [c for c in _extract(code) if c.kind == ChunkKind.FUNCTION]
    assert len(fns) == 1
    assert fns[0].name == expected_name


@needs_grammar
def test_multiple_functions():
    chunks = _extract("""\
int foo(void) { return 0; }
void bar(void) { }
""")
    names = {c.name for c in chunks if c.kind == ChunkKind.FUNCTION}
    assert names == {"foo", "bar"}


# ── Struct and enum extraction ───────────────────────────────────────


@needs_grammar
@pytest.mark.parametrize(
    ("code", "expected_name"),
    [
        ("struct Node { int value; };\n", "Node"),
        ("enum Color { RED, GREEN, BLUE };\n", "Color"),
        ("typedef struct { int x; int y; } Point;\n", "Point"),
    ],
    ids=["struct", "enum", "typedef-struct"],
)
def test_class_like(code: str, expected_name: str):
    classes = [c for c in _extract(code) if c.kind == ChunkKind.CLASS]
    assert len(classes) == 1
    assert classes[0].name == expected_name


# ── No scope in C ───────────────────────────────────────────────────


@needs_grammar
def test_no_scope_for_c_functions():
    """C has no classes — functions should never be scoped."""
    chunks = _extract("""\
struct S { int x; };
int func(void) { return 0; }
""")
    fns = [c for c in chunks if c.kind == ChunkKind.FUNCTION]
    assert len(fns) == 1
    assert fns[0].scope == ""


# ── Full file ────────────────────────────────────────────────────────


@needs_grammar
def test_full_c_file():
    chunks = _extract("""\
#include <stdio.h>
#include "utils.h"

struct Config {
    int timeout;
    int retries;
};

enum Status { OK, ERR };

int parse_config(const char *path) {
    return 0;
}

static void cleanup(void) {
}
""")
    imports = [c for c in chunks if c.kind == ChunkKind.IMPORT]
    classes = [c for c in chunks if c.kind == ChunkKind.CLASS]
    fns = [c for c in chunks if c.kind == ChunkKind.FUNCTION]

    assert len(imports) == 2
    assert {c.metadata.get("module") for c in imports} == {"stdio.h", "utils.h"}
    assert {c.name for c in classes} == {"Config", "Status"}
    assert {c.name for c in fns} == {"parse_config", "cleanup"}
