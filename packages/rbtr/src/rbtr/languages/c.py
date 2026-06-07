"""C language plugin.

Provides symbol extraction (functions, structs, enums) and
include directive capture.

Extracted chunks::

    int add(int a, int b) { ... }   → function "add", scope ""
    struct Node { int value; };     → class "Node", scope ""
    enum Color { RED, GREEN };      → class "Color", scope ""
    typedef struct { ... } Point;   → class "Point", scope ""

    #include <stdio.h>
        → import, metadata {module: "stdio.h"}
    #include "mylib.h"
        → import, metadata {module: "mylib.h"}
"""

from __future__ import annotations

from rbtr.languages.hookspec import LanguageRegistration, hookimpl

# ── Query ────────────────────────────────────────────────────────────

_QUERY = """
(function_definition
  declarator: (function_declarator
    declarator: (identifier) @_fn_name)) @function

(preproc_include
  path: (system_lib_string) @_import_module) @import

(preproc_include
  path: (string_literal) @_import_module) @import

(struct_specifier
  name: (type_identifier) @_cls_name) @class

(enum_specifier
  name: (type_identifier) @_cls_name) @class

(type_definition
  declarator: (type_identifier) @_cls_name) @class
"""

# ── Plugin ───────────────────────────────────────────────────────────


class CPlugin:
    """C language support — functions, structs, enums, includes."""

    @hookimpl
    def rbtr_register_languages(self) -> list[LanguageRegistration]:
        return [
            LanguageRegistration(
                id="c",
                extensions=frozenset({".c", ".h"}),
                grammar_module="tree_sitter_c",
                query=_QUERY,
                # C grammar uses a single `comment` node for
                # both `//` and `/* */` (and `/** */`).  Attach
                # any leading run.
                doc_comment_node_types=frozenset({"comment"}),
                source_roots=("", "include", "src"),
                test_prefix="test_",
            ),
        ]
