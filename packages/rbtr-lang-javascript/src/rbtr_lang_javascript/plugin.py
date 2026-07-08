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
        → import, metadata {module: "models", names: "foo", dots: "1"}
    import React from 'react'
        → import, metadata {module: "react", names: "React"}
    import './styles.css'
        → import, metadata {module: "styles", dots: "1"}
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from rbtr.index.models import ImportMeta
from rbtr.languages.registration import (
    ImportResolver,
    LanguageRegistration,
    QueryExtraction,
    load_query,
    parse_path_relative,
)

if TYPE_CHECKING:
    from tree_sitter import Node

_SHARED_QUERIES = load_query(__package__, "shared") + load_query(__package__, "variables")
_JS_QUERY = load_query(__package__, "javascript") + _SHARED_QUERIES
_TS_QUERY = load_query(__package__, "typescript") + _SHARED_QUERIES


# ── Import extractor (shared by JS and TS) ───────────────────────────


def extract_import_meta(
    resolver: ImportResolver, node: Node, captures: dict[str, list[Node]]
) -> ImportMeta:
    """Extract import data from a JS/TS `import_statement` node.

    Reads `@_import_module` from captures, then walks the node
    for import names (which the query can't capture because
    they're multi-valued).  Applies `parse_path_relative` and
    extension stripping to the module path.

    Examples:

        `import { foo } from './models'`:
            module="models", names="foo", dots="1"

        `import React from 'react'`:
            module="react", names="React"

        `import * as utils from '../utils'`:
            module="utils", names="utils", dots="2"

        `import './styles.css'`:
            module="styles", dots="1"
    """
    meta = resolver(node, captures)
    names: list[str] = []

    # Module comes from @_import_module capture (string_fragment).
    # Still need to parse relative paths and strip extensions.
    raw_module = meta.module

    # Import clause (no named fields — child iteration needed).
    for child in node.children:
        if child.type != "import_clause":
            continue
        for ic in child.children:
            match ic.type:
                case "identifier":
                    if ic.text:
                        names.append(ic.text.decode())
                case "named_imports":
                    for spec in ic.children:
                        if spec.type == "import_specifier":
                            sid = spec.child_by_field_name("name")
                            if sid and sid.text:
                                names.append(sid.text.decode())
                case "namespace_import":
                    for ns in ic.children:
                        if ns.type == "identifier" and ns.text:
                            names.append(ns.text.decode())
                            break

    if raw_module:
        dots, cleaned = parse_path_relative(raw_module)
        if dots:
            meta.dots = str(dots)
        # Strip file extensions so edges.py can match without guessing.
        cleaned = str(PurePosixPath(cleaned).with_suffix("")) or cleaned
        meta.module = cleaned

    if names:
        meta.names = ",".join(names)
    return meta


# ── Plugin ───────────────────────────────────────────────────────────


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
        doc_comment_node_types=frozenset({"comment"}),
    ),
    index_files=frozenset({"index.js"}),
    import_targets=frozenset({"javascript", "css"}),
    source_roots=("", "src"),
    language_plugin_version=4,
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
        doc_comment_node_types=frozenset({"comment"}),
    ),
    index_files=frozenset({"index.ts", "index.js"}),
    import_targets=frozenset({"typescript", "tsx", "javascript", "css"}),
    source_roots=("", "src"),
    language_plugin_version=4,
)
tsx = LanguageRegistration(
    id="tsx",
    extensions=frozenset({".tsx"}),
    grammar_module="tree_sitter_typescript",
    grammar_entry="language_tsx",
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
        doc_comment_node_types=frozenset({"comment"}),
    ),
    index_files=frozenset({"index.tsx", "index.ts", "index.js"}),
    import_targets=frozenset({"tsx", "typescript", "javascript", "css"}),
    source_roots=("", "src"),
    language_plugin_version=1,
)

javascript.import_extractor(extract_import_meta)
typescript.import_extractor(extract_import_meta)
tsx.import_extractor(extract_import_meta)
