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
# `method_definition` covers class and object-literal members, generators,
# and get/set accessors; inside a class it promotes to a method scoped to
# the class, elsewhere it is a scope-less method.
_SHARED = """\
(function_declaration
  name: (identifier) @_fn_name) @function

(generator_function_declaration
  name: (identifier) @_fn_name) @function

(import_statement
  source: (string (string_fragment) @_import_module)) @import

(lexical_declaration
  (variable_declarator
    name: (identifier) @_fn_name
    value: (arrow_function))) @function

(method_definition
  name: (property_identifier) @_method_name) @method
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

# TypeScript adds type-level declarations (interface, enum, type alias,
# abstract class) as classes, and class/interface members as methods.
# `method_definition` covers regular methods, constructors, and get/set
# accessors; `method_signature` covers interface methods;
# `abstract_method_signature` covers abstract declarations.
_TS_ONLY = """\
(class_declaration
  name: (type_identifier) @_cls_name) @class

(abstract_class_declaration
  name: (type_identifier) @_cls_name) @class

(interface_declaration
  name: (type_identifier) @_cls_name) @class

(enum_declaration
  name: (identifier) @_cls_name) @class

(enum_body
  (property_identifier) @_var_name @variable)

(enum_body
  (enum_assignment
    name: (property_identifier) @_var_name) @variable)

(type_alias_declaration
  name: (type_identifier) @_cls_name) @class

(internal_module
  name: (identifier) @_cls_name) @class

(module
  name: (identifier) @_cls_name) @class

(method_signature
  name: (property_identifier) @_method_name) @method

(abstract_method_signature
  name: (property_identifier) @_method_name) @method

"""

_TS_QUERY = _TS_ONLY + _SHARED + _VARIABLES

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

    Registers three languages with separate grammars but a shared
    import extractor.  The `tree_sitter_typescript` package ships two
    grammars: `language_typescript()` for `.ts` and `language_tsx()`
    for `.tsx`.  They are registered separately because the TypeScript
    grammar cannot parse JSX (a `.tsx` file parses with errors under
    `language_typescript`, cleanly under `language_tsx`).

    Example registration output::

        [
            LanguageRegistration(id="javascript", ...),
            LanguageRegistration(id="typescript", grammar_entry="language_typescript", ...),
            LanguageRegistration(id="tsx", grammar_entry="language_tsx", ...),
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
                language_plugin_version=4,
            ),
            LanguageRegistration(
                id="typescript",
                extensions=frozenset({".ts"}),
                grammar_module="tree_sitter_typescript",
                grammar_entry="language_typescript",
                query=_TS_QUERY,
                import_extractor=extract_import_meta,
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
                index_files=frozenset({"index.ts", "index.js"}),
                import_targets=frozenset({"typescript", "tsx", "javascript", "css"}),
                source_roots=("", "src"),
                test_suffix=".test",
                language_plugin_version=4,
            ),
            LanguageRegistration(
                id="tsx",
                extensions=frozenset({".tsx"}),
                grammar_module="tree_sitter_typescript",
                grammar_entry="language_tsx",
                query=_TS_QUERY,
                import_extractor=extract_import_meta,
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
                index_files=frozenset({"index.tsx", "index.ts", "index.js"}),
                import_targets=frozenset({"tsx", "typescript", "javascript", "css"}),
                source_roots=("", "src"),
                test_suffix=".test",
                language_plugin_version=1,
            ),
        ]
