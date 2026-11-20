"""JavaScript and TypeScript language plugins.

Registers three languages (one per tree-sitter grammar) sharing a
common import extractor and a shared core query (classes, functions,
generators, variables, imports, and methods — class/object members
including get/set accessors).  TS additionally captures interfaces,
enums, type aliases, abstract classes, and interface/abstract method
signatures.

Extracted chunks::

    function greet() {}             → function "greet", scope ""
    const add = (a, b) => a + b;    → function "add", scope ""
    class Service {}                → class "Service", scope ""

    import { foo } from './models'
        → import, metadata {module: "./models", names: "foo"}
    import React from 'react'
        → import, metadata {module: "react", names: "React"}
    import './styles.css'
        → import, metadata {module: "./styles"}
"""

from __future__ import annotations

from rbtr.languages.registration import (
    LanguageRegistration,
    QueryExtraction,
    load_query,
)

_SHARED_QUERIES = (
    load_query(__package__, "shared")
    + load_query(__package__, "variables")
    + load_query(__package__, "references")
)
_JS_QUERY = load_query(__package__, "javascript") + _SHARED_QUERIES + load_query(__package__, "jsx")
_TS_QUERY = load_query(__package__, "typescript") + _SHARED_QUERIES
_TSX_QUERY = _TS_QUERY + load_query(__package__, "jsx")


# Shared by the JavaScript and TypeScript plugins.


javascript = LanguageRegistration(
    id="javascript",
    extensions=frozenset({".js", ".jsx", ".mjs"}),
    grammar_module="tree_sitter_javascript",
    extraction=QueryExtraction(
        query=_JS_QUERY,
        scope_types=frozenset({"class_declaration", "function_declaration"}),
        class_scope_types=frozenset({"class_declaration"}),
        # Both `/** */` JSDoc and `//` comments land in
        # the grammar as a single `comment` node type.
    ),
    index_files=frozenset({"index.js"}),
    import_targets=frozenset({"javascript", "css"}),
    source_roots=("", "src"),
    extraction_serial=11,
)
typescript = LanguageRegistration(
    id="typescript",
    extensions=frozenset({".ts"}),
    grammar_module="tree_sitter_typescript",
    grammar_entry="language_typescript",
    extraction=QueryExtraction(
        query=_TS_QUERY,
        scope_types=frozenset(
            {
                "class_declaration",
                "abstract_class_declaration",
                "interface_declaration",
                "function_declaration",
                "internal_module",
                "module",
                "enum_declaration",
            }
        ),
        class_scope_types=frozenset(
            {"class_declaration", "abstract_class_declaration", "interface_declaration"}
        ),
    ),
    index_files=frozenset({"index.ts", "index.js"}),
    import_targets=frozenset({"typescript", "tsx", "javascript", "css"}),
    source_roots=("", "src"),
    extraction_serial=11,
)
tsx = LanguageRegistration(
    id="tsx",
    extensions=frozenset({".tsx"}),
    grammar_module="tree_sitter_typescript",
    grammar_entry="language_tsx",
    extraction=QueryExtraction(
        query=_TSX_QUERY,
        scope_types=frozenset(
            {
                "class_declaration",
                "abstract_class_declaration",
                "interface_declaration",
                "function_declaration",
                "internal_module",
                "module",
                "enum_declaration",
            }
        ),
        class_scope_types=frozenset(
            {"class_declaration", "abstract_class_declaration", "interface_declaration"}
        ),
    ),
    index_files=frozenset({"index.tsx", "index.ts", "index.js"}),
    import_targets=frozenset({"tsx", "typescript", "javascript", "css"}),
    source_roots=("", "src"),
    extraction_serial=8,
)
