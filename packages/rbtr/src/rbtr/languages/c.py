"""C language plugin.

Provides symbol extraction (functions, function prototypes, structs,
unions, enums, typedefs, global variables, and function/object-like
macros) and include directive capture.  Object-like macros and
pointer-declared globals are variables; function-like macros and
prototypes are functions.

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
  name: (type_identifier) @_cls_name
  body: (field_declaration_list)) @class

(union_specifier
  name: (type_identifier) @_cls_name
  body: (field_declaration_list)) @class

(enum_specifier
  name: (type_identifier) @_cls_name
  body: (enumerator_list)) @class

(enumerator
  name: (identifier) @_var_name) @variable

(type_definition
  declarator: (type_identifier) @_cls_name) @class

(type_definition
  declarator: (function_declarator
    declarator: (parenthesized_declarator
      (pointer_declarator
        declarator: (type_identifier) @_cls_name)))) @class

(preproc_function_def
  name: (identifier) @_fn_name) @function

(preproc_def
  name: (identifier) @_var_name) @variable

(translation_unit
  (declaration
    declarator: (function_declarator
      declarator: (identifier) @_fn_name)) @function)

(translation_unit
  (declaration
    declarator: (init_declarator
      declarator: (identifier) @_var_name)) @variable)

(translation_unit
  (declaration
    declarator: (pointer_declarator
      declarator: (identifier) @_var_name)) @variable)

(translation_unit
  (declaration
    declarator: (init_declarator
      declarator: (pointer_declarator
        declarator: (identifier) @_var_name))) @variable)
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
                language_plugin_version=3,
            ),
        ]
