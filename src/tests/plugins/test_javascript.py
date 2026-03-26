"""Tests for the JavaScript and TypeScript language plugins.

Both languages share an import extractor but use different grammars
and queries.  Tests for each are skipped if the respective grammar
package is not installed.
"""

from __future__ import annotations

import pytest
from tree_sitter import Language

from rbtr.index.models import Chunk, ChunkKind
from rbtr.index.treesitter import extract_symbols
from rbtr.plugins.hookspec import ImportExtractor, LanguageRegistration
from rbtr.plugins.manager import get_manager

# ── Grammar availability ─────────────────────────────────────────────

_mgr = get_manager()

_has_js = _mgr.load_grammar("javascript") is not None
_has_ts = _mgr.load_grammar("typescript") is not None

skip_no_js = pytest.mark.skipif(not _has_js, reason="tree-sitter-javascript not installed")
skip_no_ts = pytest.mark.skipif(not _has_ts, reason="tree-sitter-typescript not installed")

# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def js_grammar() -> Language:
    g = _mgr.load_grammar("javascript")
    assert g is not None
    return g


@pytest.fixture
def js_reg() -> LanguageRegistration:
    reg = _mgr.get_registration("javascript")
    assert reg is not None
    return reg


@pytest.fixture
def js_query(js_reg: LanguageRegistration) -> str:
    assert js_reg.query is not None
    return js_reg.query


@pytest.fixture
def js_extractor(js_reg: LanguageRegistration) -> ImportExtractor:
    assert js_reg.import_extractor is not None
    return js_reg.import_extractor


@pytest.fixture
def js_scope_types(js_reg: LanguageRegistration) -> frozenset[str]:
    return js_reg.scope_types


@pytest.fixture
def ts_grammar() -> Language:
    g = _mgr.load_grammar("typescript")
    assert g is not None
    return g


@pytest.fixture
def ts_reg() -> LanguageRegistration:
    reg = _mgr.get_registration("typescript")
    assert reg is not None
    return reg


@pytest.fixture
def ts_query(ts_reg: LanguageRegistration) -> str:
    assert ts_reg.query is not None
    return ts_reg.query


@pytest.fixture
def ts_extractor(ts_reg: LanguageRegistration) -> ImportExtractor:
    assert ts_reg.import_extractor is not None
    return ts_reg.import_extractor


@pytest.fixture
def ts_scope_types(ts_reg: LanguageRegistration) -> frozenset[str]:
    return ts_reg.scope_types


# ── Helpers ──────────────────────────────────────────────────────────


def _extract(
    source: str,
    grammar: Language,
    query_str: str,
    extractor: ImportExtractor,
    scope_types: frozenset[str],
    file_path: str = "src/app.js",
) -> list[Chunk]:
    return extract_symbols(
        file_path,
        "sha1",
        source.encode(),
        grammar,
        query_str,
        import_extractor=extractor,
        scope_types=scope_types,
    )


def _symbols(
    source: str,
    grammar: Language,
    query_str: str,
    extractor: ImportExtractor,
    scope_types: frozenset[str],
    file_path: str = "src/app.js",
) -> list[tuple[str, str, str]]:
    return [
        (c.kind, c.name, c.scope)
        for c in _extract(source, grammar, query_str, extractor, scope_types, file_path)
    ]


def _imports(
    source: str,
    grammar: Language,
    query_str: str,
    extractor: ImportExtractor,
    scope_types: frozenset[str],
    file_path: str = "src/app.js",
) -> list[Chunk]:
    return [
        c
        for c in _extract(source, grammar, query_str, extractor, scope_types, file_path)
        if c.kind == ChunkKind.IMPORT
    ]


# ══════════════════════════════════════════════════════════════════════
# JavaScript
# ══════════════════════════════════════════════════════════════════════

# ── Registration ─────────────────────────────────────────────────────


@skip_no_js
def test_js_registration_exists(js_reg: LanguageRegistration) -> None:
    assert js_reg.id == "javascript"


@skip_no_js
def test_js_extensions(js_reg: LanguageRegistration) -> None:
    assert ".js" in js_reg.extensions
    assert ".jsx" in js_reg.extensions
    assert ".mjs" in js_reg.extensions


@skip_no_js
def test_js_grammar_module(js_reg: LanguageRegistration) -> None:
    assert js_reg.grammar_module == "tree_sitter_javascript"


@skip_no_js
def test_js_scope_types_contain_class_declaration(js_reg: LanguageRegistration) -> None:
    assert "class_declaration" in js_reg.scope_types


# ── Function extraction ──────────────────────────────────────────────


@skip_no_js
def test_js_function_declaration(
    js_grammar: Language,
    js_query: str,
    js_extractor: ImportExtractor,
    js_scope_types: frozenset[str],
) -> None:
    syms = _symbols("function greet() {}\n", js_grammar, js_query, js_extractor, js_scope_types)
    assert ("function", "greet", "") in syms


@skip_no_js
def test_js_arrow_function(
    js_grammar: Language,
    js_query: str,
    js_extractor: ImportExtractor,
    js_scope_types: frozenset[str],
) -> None:
    syms = _symbols(
        "const add = (a, b) => a + b;\n", js_grammar, js_query, js_extractor, js_scope_types
    )
    assert ("function", "add", "") in syms


@skip_no_js
def test_js_arrow_function_block_body(
    js_grammar: Language,
    js_query: str,
    js_extractor: ImportExtractor,
    js_scope_types: frozenset[str],
) -> None:
    syms = _symbols(
        """\
const fetch = () => {
  return data;
};
""",
        js_grammar,
        js_query,
        js_extractor,
        js_scope_types,
    )
    assert ("function", "fetch", "") in syms


@skip_no_js
def test_js_multiple_functions(
    js_grammar: Language,
    js_query: str,
    js_extractor: ImportExtractor,
    js_scope_types: frozenset[str],
) -> None:
    src = """\
function a() {}
function b() {}
const c = () => {};
"""
    names = [
        s[1]
        for s in _symbols(src, js_grammar, js_query, js_extractor, js_scope_types)
        if s[0] == "function"
    ]
    assert "a" in names
    assert "b" in names
    assert "c" in names


# ── Class extraction ─────────────────────────────────────────────────


@skip_no_js
def test_js_class(
    js_grammar: Language,
    js_query: str,
    js_extractor: ImportExtractor,
    js_scope_types: frozenset[str],
) -> None:
    syms = _symbols("class User {}\n", js_grammar, js_query, js_extractor, js_scope_types)
    assert ("class", "User", "") in syms


@skip_no_js
def test_js_class_with_extends(
    js_grammar: Language,
    js_query: str,
    js_extractor: ImportExtractor,
    js_scope_types: frozenset[str],
) -> None:
    syms = _symbols(
        "class Admin extends User {}\n", js_grammar, js_query, js_extractor, js_scope_types
    )
    assert ("class", "Admin", "") in syms


# ── Import: named imports ────────────────────────────────────────────


@skip_no_js
def test_js_import_named_single(
    js_grammar: Language,
    js_query: str,
    js_extractor: ImportExtractor,
    js_scope_types: frozenset[str],
) -> None:
    imp = _imports(
        "import { foo } from './models';\n", js_grammar, js_query, js_extractor, js_scope_types
    )[0]
    assert imp.metadata == {"module": "models", "names": "foo", "dots": "1"}


@skip_no_js
def test_js_import_named_multiple(
    js_grammar: Language,
    js_query: str,
    js_extractor: ImportExtractor,
    js_scope_types: frozenset[str],
) -> None:
    imp = _imports(
        "import { foo, bar } from './models';\n",
        js_grammar,
        js_query,
        js_extractor,
        js_scope_types,
    )[0]
    assert imp.metadata == {"module": "models", "names": "foo,bar", "dots": "1"}


@skip_no_js
def test_js_import_named_from_parent(
    js_grammar: Language,
    js_query: str,
    js_extractor: ImportExtractor,
    js_scope_types: frozenset[str],
) -> None:
    imp = _imports(
        "import { Config } from '../config';\n",
        js_grammar,
        js_query,
        js_extractor,
        js_scope_types,
    )[0]
    assert imp.metadata == {"module": "config", "names": "Config", "dots": "2"}


@skip_no_js
def test_js_import_named_from_grandparent(
    js_grammar: Language,
    js_query: str,
    js_extractor: ImportExtractor,
    js_scope_types: frozenset[str],
) -> None:
    imp = _imports(
        "import { util } from '../../shared/util';\n",
        js_grammar,
        js_query,
        js_extractor,
        js_scope_types,
    )[0]
    assert imp.metadata == {"module": "shared/util", "names": "util", "dots": "3"}


# ── Import: default import ───────────────────────────────────────────


@skip_no_js
def test_js_import_default(
    js_grammar: Language,
    js_query: str,
    js_extractor: ImportExtractor,
    js_scope_types: frozenset[str],
) -> None:
    imp = _imports(
        "import React from 'react';\n", js_grammar, js_query, js_extractor, js_scope_types
    )[0]
    assert imp.metadata == {"module": "react", "names": "React"}


@skip_no_js
def test_js_import_default_relative(
    js_grammar: Language,
    js_query: str,
    js_extractor: ImportExtractor,
    js_scope_types: frozenset[str],
) -> None:
    imp = _imports(
        "import App from './App';\n", js_grammar, js_query, js_extractor, js_scope_types
    )[0]
    assert imp.metadata == {"module": "App", "names": "App", "dots": "1"}


# ── Import: namespace import ─────────────────────────────────────────


@skip_no_js
def test_js_import_namespace(
    js_grammar: Language,
    js_query: str,
    js_extractor: ImportExtractor,
    js_scope_types: frozenset[str],
) -> None:
    imp = _imports(
        "import * as utils from '../utils';\n", js_grammar, js_query, js_extractor, js_scope_types
    )[0]
    assert imp.metadata == {"module": "utils", "names": "utils", "dots": "2"}


# ── Import: side-effect import ───────────────────────────────────────


@skip_no_js
def test_js_import_side_effect(
    js_grammar: Language,
    js_query: str,
    js_extractor: ImportExtractor,
    js_scope_types: frozenset[str],
) -> None:
    imp = _imports("import './styles.css';\n", js_grammar, js_query, js_extractor, js_scope_types)[
        0
    ]
    assert imp.metadata == {"module": "styles", "dots": "1"}


@skip_no_js
def test_js_import_side_effect_no_extension(
    js_grammar: Language,
    js_query: str,
    js_extractor: ImportExtractor,
    js_scope_types: frozenset[str],
) -> None:
    imp = _imports("import './polyfills';\n", js_grammar, js_query, js_extractor, js_scope_types)[0]
    assert imp.metadata == {"module": "polyfills", "dots": "1"}


# ── Import: absolute (package) imports ───────────────────────────────


@skip_no_js
def test_js_import_package(
    js_grammar: Language,
    js_query: str,
    js_extractor: ImportExtractor,
    js_scope_types: frozenset[str],
) -> None:
    imp = _imports(
        "import express from 'express';\n", js_grammar, js_query, js_extractor, js_scope_types
    )[0]
    assert imp.metadata == {"module": "express", "names": "express"}
    assert "dots" not in imp.metadata


@skip_no_js
def test_js_import_scoped_package(
    js_grammar: Language,
    js_query: str,
    js_extractor: ImportExtractor,
    js_scope_types: frozenset[str],
) -> None:
    imp = _imports(
        "import { render } from '@testing-library/react';\n",
        js_grammar,
        js_query,
        js_extractor,
        js_scope_types,
    )[0]
    assert imp.metadata["module"] == "@testing-library/react"
    assert imp.metadata["names"] == "render"
    assert "dots" not in imp.metadata


# ── Import: multiple statements ──────────────────────────────────────


@skip_no_js
def test_js_multiple_imports(
    js_grammar: Language,
    js_query: str,
    js_extractor: ImportExtractor,
    js_scope_types: frozenset[str],
) -> None:
    src = """\
import React from 'react';
import { useState } from 'react';
"""
    imps = _imports(src, js_grammar, js_query, js_extractor, js_scope_types)
    assert len(imps) == 2


# ── Mixed extraction ─────────────────────────────────────────────────


@skip_no_js
def test_js_full_module(
    js_grammar: Language,
    js_query: str,
    js_extractor: ImportExtractor,
    js_scope_types: frozenset[str],
) -> None:
    src = """\
import { Model } from './model';

class Service {
}

function create() {
}

const destroy = () => {};
"""
    chunks = _extract(src, js_grammar, js_query, js_extractor, js_scope_types)
    kinds = {c.kind for c in chunks}
    assert ChunkKind.IMPORT in kinds
    assert ChunkKind.CLASS in kinds
    assert ChunkKind.FUNCTION in kinds


@skip_no_js
def test_js_empty_source(
    js_grammar: Language,
    js_query: str,
    js_extractor: ImportExtractor,
    js_scope_types: frozenset[str],
) -> None:
    chunks = _extract("", js_grammar, js_query, js_extractor, js_scope_types)
    assert chunks == []


# ══════════════════════════════════════════════════════════════════════
# TypeScript
# ══════════════════════════════════════════════════════════════════════

# ── Registration ─────────────────────────────────────────────────────


@skip_no_ts
def test_ts_registration_exists(ts_reg: LanguageRegistration) -> None:
    assert ts_reg.id == "typescript"


@skip_no_ts
def test_ts_extensions(ts_reg: LanguageRegistration) -> None:
    assert ".ts" in ts_reg.extensions
    assert ".tsx" in ts_reg.extensions


@skip_no_ts
def test_ts_grammar_entry(ts_reg: LanguageRegistration) -> None:
    assert ts_reg.grammar_entry == "language_typescript"


# ── Function extraction ──────────────────────────────────────────────


@skip_no_ts
def test_ts_function(
    ts_grammar: Language,
    ts_query: str,
    ts_extractor: ImportExtractor,
    ts_scope_types: frozenset[str],
) -> None:
    syms = _symbols(
        "function greet(): void {}\n", ts_grammar, ts_query, ts_extractor, ts_scope_types, "app.ts"
    )
    assert ("function", "greet", "") in syms


@skip_no_ts
def test_ts_arrow_function(
    ts_grammar: Language,
    ts_query: str,
    ts_extractor: ImportExtractor,
    ts_scope_types: frozenset[str],
) -> None:
    src = "const add = (a: number, b: number): number => a + b;\n"
    syms = _symbols(src, ts_grammar, ts_query, ts_extractor, ts_scope_types, "app.ts")
    assert ("function", "add", "") in syms


# ── Class extraction ─────────────────────────────────────────────────


@skip_no_ts
def test_ts_class(
    ts_grammar: Language,
    ts_query: str,
    ts_extractor: ImportExtractor,
    ts_scope_types: frozenset[str],
) -> None:
    syms = _symbols(
        "class Service {}\n", ts_grammar, ts_query, ts_extractor, ts_scope_types, "app.ts"
    )
    assert ("class", "Service", "") in syms


@skip_no_ts
def test_ts_class_with_generics(
    ts_grammar: Language,
    ts_query: str,
    ts_extractor: ImportExtractor,
    ts_scope_types: frozenset[str],
) -> None:
    syms = _symbols(
        "class Container<T> {}\n", ts_grammar, ts_query, ts_extractor, ts_scope_types, "app.ts"
    )
    assert ("class", "Container", "") in syms


# ── Import: named imports ────────────────────────────────────────────


@skip_no_ts
def test_ts_import_named(
    ts_grammar: Language,
    ts_query: str,
    ts_extractor: ImportExtractor,
    ts_scope_types: frozenset[str],
) -> None:
    imp = _imports(
        "import { User } from './types';\n",
        ts_grammar,
        ts_query,
        ts_extractor,
        ts_scope_types,
        "app.ts",
    )[0]
    assert imp.metadata == {"module": "types", "names": "User", "dots": "1"}


@skip_no_ts
def test_ts_import_type(
    ts_grammar: Language,
    ts_query: str,
    ts_extractor: ImportExtractor,
    ts_scope_types: frozenset[str],
) -> None:
    """TypeScript type-only imports should extract the same metadata."""
    imp = _imports(
        "import type { Config } from './config';\n",
        ts_grammar,
        ts_query,
        ts_extractor,
        ts_scope_types,
        "app.ts",
    )[0]
    assert imp.metadata == {"module": "config", "names": "Config", "dots": "1"}


@skip_no_ts
def test_ts_import_default(
    ts_grammar: Language,
    ts_query: str,
    ts_extractor: ImportExtractor,
    ts_scope_types: frozenset[str],
) -> None:
    imp = _imports(
        "import Express from 'express';\n",
        ts_grammar,
        ts_query,
        ts_extractor,
        ts_scope_types,
        "app.ts",
    )[0]
    assert imp.metadata == {"module": "express", "names": "Express"}


@skip_no_ts
def test_ts_import_namespace(
    ts_grammar: Language,
    ts_query: str,
    ts_extractor: ImportExtractor,
    ts_scope_types: frozenset[str],
) -> None:
    imp = _imports(
        "import * as path from 'path';\n",
        ts_grammar,
        ts_query,
        ts_extractor,
        ts_scope_types,
        "app.ts",
    )[0]
    assert imp.metadata == {"module": "path", "names": "path"}


@skip_no_ts
def test_ts_import_side_effect(
    ts_grammar: Language,
    ts_query: str,
    ts_extractor: ImportExtractor,
    ts_scope_types: frozenset[str],
) -> None:
    imp = _imports(
        "import './setup';\n", ts_grammar, ts_query, ts_extractor, ts_scope_types, "app.ts"
    )[0]
    assert imp.metadata == {"module": "setup", "dots": "1"}


# ── Mixed extraction ─────────────────────────────────────────────────


@skip_no_ts
def test_ts_full_module(
    ts_grammar: Language,
    ts_query: str,
    ts_extractor: ImportExtractor,
    ts_scope_types: frozenset[str],
) -> None:
    src = """\
import { Model } from './model';

class Repository {
}

function query(): void {
}
"""
    chunks = _extract(src, ts_grammar, ts_query, ts_extractor, ts_scope_types, "app.ts")
    kinds = {c.kind for c in chunks}
    assert ChunkKind.IMPORT in kinds
    assert ChunkKind.CLASS in kinds
    assert ChunkKind.FUNCTION in kinds
