"""JavaScript and TypeScript language plugins.

Registers two languages (one per tree-sitter grammar) sharing a
common import extractor.  The only query difference is the class
name node type: JS uses `identifier`, TS uses `type_identifier`.

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
from rbtr.languages.hookspec import (
    LanguageRegistration,
    build_import_from_captures,
    hookimpl,
    parse_path_relative,
)

if TYPE_CHECKING:
    from tree_sitter import Node

# ── Queries ──────────────────────────────────────────────────────────

# Shared patterns — identical across JS and TS grammars.
_SHARED = """\
(function_declaration
  name: (identifier) @_fn_name) @function

(import_statement
  source: (string (string_fragment) @_import_module)) @import

(lexical_declaration
  (variable_declarator
    name: (identifier) @_fn_name
    value: (arrow_function))) @function
"""

# Module-level const/let bindings with a non-function value become
# variables.  The value allowlist excludes arrow/function expressions,
# which `_SHARED` already captures as functions.
_VAR_VALUES = """\
[(number) (string) (template_string) (true) (false) (null)
 (object) (array) (identifier) (member_expression)
 (call_expression) (new_expression) (binary_expression) (unary_expression)]"""

# Flat destructuring targets (object/array patterns).  The binding names
# differ by shape: object shorthand, renamed `value:`, defaulted `left:`,
# and rest elements.  Nested patterns are not captured (no query-only
# recursion).  No value allowlist needed — a destructuring target is never a
# function.
_DESTRUCTURE_NAMES = """\
name: [
  (object_pattern [
    (shorthand_property_identifier_pattern) @_var_name
    (pair_pattern value: (identifier) @_var_name)
    (object_assignment_pattern left: (shorthand_property_identifier_pattern) @_var_name)
    (rest_pattern (identifier) @_var_name)
  ])
  (array_pattern [
    (identifier) @_var_name
    (rest_pattern (identifier) @_var_name)
  ])
]"""

_VARIABLES = f"""
(program
  (lexical_declaration
    (variable_declarator
      name: (identifier) @_var_name
      value: {_VAR_VALUES}) @variable))

(program
  (export_statement
    declaration: (lexical_declaration
      (variable_declarator
        name: (identifier) @_var_name
        value: {_VAR_VALUES}) @variable)))

(program
  (lexical_declaration
    (variable_declarator
      {_DESTRUCTURE_NAMES}) @variable))

(program
  (export_statement
    declaration: (lexical_declaration
      (variable_declarator
        {_DESTRUCTURE_NAMES}) @variable)))
"""

# JS uses `identifier` for class names, TS uses `type_identifier`.
_JS_QUERY = (
    """\
(class_declaration
  name: (identifier) @_cls_name) @class

"""
    + _SHARED
    + _VARIABLES
)

_TS_QUERY = (
    """\
(class_declaration
  name: (type_identifier) @_cls_name) @class

"""
    + _SHARED
    + _VARIABLES
)

# ── Import extractor (shared by JS and TS) ───────────────────────────


def extract_import_meta(node: Node, captures: dict[str, list[Node]]) -> ImportMeta:
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
    meta = build_import_from_captures(node, captures)
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


class JavaScriptPlugin:
    """JavaScript and TypeScript language support.

    Registers two languages with separate grammars but a shared
    import extractor.  TypeScript requires `grammar_entry=
    "language_typescript"` because the `tree_sitter_typescript`
    package exposes `language_typescript()` instead of the
    standard `language()`.

    Example registration output::

        [
            LanguageRegistration(id="javascript", ...),
            LanguageRegistration(id="typescript", grammar_entry="language_typescript", ...),
        ]
    """

    @hookimpl
    def rbtr_register_languages(self) -> list[LanguageRegistration]:
        return [
            LanguageRegistration(
                id="javascript",
                extensions=frozenset({".js", ".jsx", ".mjs"}),
                grammar_module="tree_sitter_javascript",
                query=_JS_QUERY,
                import_extractor=extract_import_meta,
                scope_types=frozenset({"class_declaration", "function_declaration"}),
                class_scope_types=frozenset({"class_declaration"}),
                # Both `/** */` JSDoc and `//` comments land in
                # the grammar as a single `comment` node type.
                doc_comment_node_types=frozenset({"comment"}),
                index_files=frozenset({"index.js"}),
                import_targets=frozenset({"javascript", "css"}),
                source_roots=("", "src"),
                test_suffix=".test",
                language_plugin_version=3,
            ),
            LanguageRegistration(
                id="typescript",
                extensions=frozenset({".ts", ".tsx"}),
                grammar_module="tree_sitter_typescript",
                grammar_entry="language_typescript",
                query=_TS_QUERY,
                import_extractor=extract_import_meta,
                scope_types=frozenset(
                    {"class_declaration", "function_declaration", "internal_module"}
                ),
                class_scope_types=frozenset({"class_declaration"}),
                doc_comment_node_types=frozenset({"comment"}),
                index_files=frozenset({"index.ts", "index.js"}),
                import_targets=frozenset({"typescript", "javascript", "css"}),
                source_roots=("", "src"),
                test_suffix=".test",
                language_plugin_version=3,
            ),
        ]
