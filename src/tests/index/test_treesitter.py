"""Tests for Tree-sitter symbol extraction.

Tests for base grammars (Python, Bash) run unconditionally.
Tests for optional grammars (JS, TS, Go, Rust, Java) are skipped
when the grammar package is not installed.
"""

from __future__ import annotations

import pytest

from rbtr.index.models import Chunk, ChunkKind
from rbtr.index.treesitter import extract_symbols
from rbtr.plugins.manager import get_manager

# ── Helpers ──────────────────────────────────────────────────────────

_EXT: dict[str, str] = {
    "python": "py",
    "javascript": "js",
    "typescript": "ts",
    "go": "go",
    "rust": "rs",
    "java": "java",
    "bash": "sh",
}

_manager = get_manager()


def _extract_chunks(
    source: str | bytes,
    lang: str = "python",
    file_path: str | None = None,
) -> list[Chunk]:
    """Extract chunks using the plugin system."""
    grammar = _manager.load_grammar(lang)
    assert grammar is not None, f"grammar for {lang} not installed"
    reg = _manager.get_registration(lang)
    assert reg is not None
    assert reg.query is not None, f"no query for {lang}"
    ext = _EXT.get(lang, "txt")
    path = file_path or f"test.{ext}"
    src = source.encode() if isinstance(source, str) else source
    return extract_symbols(
        path,
        "sha1",
        src,
        grammar,
        reg.query,
        import_extractor=reg.import_extractor,
        scope_types=reg.scope_types,
    )


def _tuples(source: str, lang: str = "python") -> list[tuple[str, str, str]]:
    """Extract symbols and return (kind, name, scope) tuples."""
    return [(c.kind, c.name, c.scope) for c in _extract_chunks(source, lang)]


def _imports(
    source: str | bytes,
    lang: str = "python",
    file_path: str | None = None,
) -> list[Chunk]:
    """Return only import chunks."""
    return [c for c in _extract_chunks(source, lang, file_path) if c.kind == ChunkKind.IMPORT]


# ── Skip markers for optional grammars ───────────────────────────────

_has_js = _manager.load_grammar("javascript") is not None
_has_ts = _manager.load_grammar("typescript") is not None
_has_go = _manager.load_grammar("go") is not None
_has_rust = _manager.load_grammar("rust") is not None
_has_java = _manager.load_grammar("java") is not None

skip_no_js = pytest.mark.skipif(not _has_js, reason="tree-sitter-javascript not installed")
skip_no_ts = pytest.mark.skipif(not _has_ts, reason="tree-sitter-typescript not installed")
skip_no_go = pytest.mark.skipif(not _has_go, reason="tree-sitter-go not installed")
skip_no_rust = pytest.mark.skipif(not _has_rust, reason="tree-sitter-rust not installed")
skip_no_java = pytest.mark.skipif(not _has_java, reason="tree-sitter-java not installed")

# ── Python: symbol extraction ────────────────────────────────────────


def test_extract_function() -> None:
    results = _tuples("def hello():\n    pass\n")
    assert ("function", "hello", "") in results


def test_extract_class() -> None:
    results = _tuples("class Foo:\n    pass\n")
    assert ("class", "Foo", "") in results


def test_extract_method_in_class() -> None:
    results = _tuples("class Foo:\n    def bar(self):\n        pass\n")
    assert ("method", "bar", "Foo") in results


def test_extract_import() -> None:
    kinds = [r[0] for r in _tuples("import os\n")]
    assert "import" in kinds


def test_extract_from_import() -> None:
    kinds = [r[0] for r in _tuples("from os.path import join\n")]
    assert "import" in kinds


def test_extract_multiple_functions() -> None:
    names = [
        r[1]
        for r in _tuples("def foo():\n    pass\n\ndef bar():\n    pass\n")
        if r[0] == "function"
    ]
    assert "foo" in names
    assert "bar" in names


def test_extract_nested_class_method() -> None:
    source = "class Outer:\n    class Inner:\n        def deep(self):\n            pass\n"
    methods = [(r[1], r[2]) for r in _tuples(source) if r[0] == "method"]
    assert ("deep", "Inner") in methods


def test_extract_sets_line_numbers() -> None:
    chunks = _extract_chunks("\n\ndef third_line():\n    pass\n")
    fn = next(c for c in chunks if c.kind == ChunkKind.FUNCTION)
    assert fn.line_start == 3


def test_extract_sets_blob_sha() -> None:
    chunks = _extract_chunks("def f():\n    pass\n")
    assert all(c.blob_sha == "sha1" for c in chunks)


def test_extract_empty_source() -> None:
    grammar = _manager.load_grammar("python")
    assert grammar is not None
    reg = _manager.get_registration("python")
    assert reg is not None
    assert reg.query is not None
    chunks = extract_symbols("t.py", "x", b"", grammar, reg.query)
    assert chunks == []


def test_non_import_chunks_have_empty_metadata() -> None:
    chunks = _extract_chunks("def foo():\n    pass\n")
    fn = next(c for c in chunks if c.kind == ChunkKind.FUNCTION)
    assert fn.metadata == {}


# ── Python: import metadata ──────────────────────────────────────────


def test_py_import_absolute() -> None:
    imp = _imports("import os.path\n")[0]
    assert imp.metadata == {"module": "os.path"}


def test_py_import_from() -> None:
    imp = _imports("from rbtr.index.models import Chunk, Edge\n")[0]
    assert imp.metadata["module"] == "rbtr.index.models"
    assert imp.metadata["names"] == "Chunk,Edge"


def test_py_import_relative() -> None:
    imp = _imports("from ..core import engine\n", file_path="src/pkg/sub/mod.py")[0]
    assert imp.metadata == {"dots": "2", "module": "core", "names": "engine"}


def test_py_import_relative_dot_only() -> None:
    imp = _imports("from . import utils\n", file_path="src/pkg/mod.py")[0]
    assert imp.metadata["dots"] == "1"
    assert "module" not in imp.metadata
    assert imp.metadata["names"] == "utils"


def test_py_import_aliased() -> None:
    imp = _imports("from .models import Chunk as C\n", file_path="src/pkg/mod.py")[0]
    assert imp.metadata["names"] == "Chunk"


# ── JavaScript: import metadata ──────────────────────────────────────


@skip_no_js
def test_js_import_named() -> None:
    imp = _imports("import { foo, bar } from './models'\n", "javascript")[0]
    assert imp.metadata == {"module": "models", "names": "foo,bar", "dots": "1"}


@skip_no_js
def test_js_import_default() -> None:
    imp = _imports("import React from 'react'\n", "javascript")[0]
    assert imp.metadata == {"module": "react", "names": "React"}


@skip_no_js
def test_js_import_namespace() -> None:
    imp = _imports("import * as utils from '../utils'\n", "javascript")[0]
    assert imp.metadata == {"module": "utils", "names": "utils", "dots": "2"}


@skip_no_js
def test_js_import_side_effect() -> None:
    imp = _imports("import './styles.css'\n", "javascript")[0]
    assert imp.metadata == {"module": "styles", "dots": "1"}


@skip_no_js
def test_js_extract_function_and_class() -> None:
    results = _tuples("function greet() {}\nclass User {}\n", "javascript")
    assert ("function", "greet", "") in results
    assert ("class", "User", "") in results


@skip_no_js
def test_js_arrow_function() -> None:
    results = _tuples("const greet = () => {};\n", "javascript")
    assert ("function", "greet", "") in results


# ── TypeScript: import metadata ──────────────────────────────────────


@skip_no_ts
def test_ts_import_named() -> None:
    imp = _imports("import { User } from './types'\n", "typescript")[0]
    assert imp.metadata == {"module": "types", "names": "User", "dots": "1"}


@skip_no_ts
def test_ts_import_type() -> None:
    imp = _imports("import type { Config } from './config'\n", "typescript")[0]
    assert imp.metadata == {"module": "config", "names": "Config", "dots": "1"}


@skip_no_ts
def test_ts_extract_class() -> None:
    results = _tuples("class Service {}\n", "typescript")
    assert ("class", "Service", "") in results


# ── Go: import metadata ─────────────────────────────────────────────


@skip_no_go
def test_go_import_single() -> None:
    imp = _imports('package main\nimport "fmt"\n', "go")[0]
    assert imp.metadata == {"module": "fmt"}


@skip_no_go
def test_go_import_grouped() -> None:
    src = 'package main\nimport (\n    "fmt"\n    "os/exec"\n)\n'
    imp = _imports(src, "go")[0]
    assert imp.metadata == {"module": "fmt,os/exec"}


@skip_no_go
def test_go_extract_function() -> None:
    results = _tuples("package main\nfunc hello() {}\n", "go")
    assert ("function", "hello", "") in results


@skip_no_go
def test_go_extract_type() -> None:
    results = _tuples("package main\ntype User struct {}\n", "go")
    assert ("class", "User", "") in results


# ── Rust: import metadata ───────────────────────────────────────────


@skip_no_rust
def test_rust_import_scoped() -> None:
    imp = _imports("use std::collections::HashMap;\n", "rust")[0]
    assert imp.metadata == {"module": "std/collections", "names": "HashMap"}


@skip_no_rust
def test_rust_import_crate() -> None:
    imp = _imports("use crate::models::{Chunk, Edge};\n", "rust")[0]
    assert imp.metadata["module"] == "crate/models"
    assert imp.metadata["names"] == "Chunk,Edge"


@skip_no_rust
def test_rust_import_super() -> None:
    imp = _imports("use super::utils;\n", "rust")[0]
    assert imp.metadata == {"names": "utils", "dots": "2"}


@skip_no_rust
def test_rust_extract_function_and_struct() -> None:
    src = "fn process() {}\nstruct Item {}\n"
    results = _tuples(src, "rust")
    assert ("function", "process", "") in results
    assert ("class", "Item", "") in results


# ── Java: import metadata ───────────────────────────────────────────


@skip_no_java
def test_java_import_class() -> None:
    imp = _imports("import java.util.HashMap;\n", "java")[0]
    assert imp.metadata == {"module": "java.util", "names": "HashMap"}


@skip_no_java
def test_java_import_static() -> None:
    src = "import static org.junit.Assert.assertEquals;\n"
    imp = _imports(src, "java")[0]
    assert imp.metadata["module"] == "org.junit.Assert"
    assert imp.metadata["names"] == "assertEquals"


@skip_no_java
def test_java_extract_class_and_method() -> None:
    src = "class Service { void process() {} }\n"
    results = _tuples(src, "java")
    assert ("class", "Service", "") in results
    assert ("method", "process", "Service") in results
