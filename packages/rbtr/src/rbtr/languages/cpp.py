"""C++ language plugin.

Provides symbol extraction (functions, classes, structs, enums,
methods) and include directive capture.

Extracted chunks::

    void greet() {}                 → function "greet", scope ""
    class Shape {};                 → class "Shape", scope ""
    struct Point { ... };           → class "Point", scope ""
    class Foo {
      void bar() {}                 → method "bar", scope "Foo"
    };

    #include <iostream>
        → import, metadata {module: "iostream"}
"""

from __future__ import annotations

from rbtr.languages.hookspec import LanguageRegistration, hookimpl

# ── Query ────────────────────────────────────────────────────────────

_QUERY = """
(function_definition
  declarator: (function_declarator
    declarator: (identifier) @_fn_name)) @function

(function_definition
  declarator: (function_declarator
    declarator: (field_identifier) @_fn_name)) @function

(preproc_include
  path: (system_lib_string) @_import_module) @import

(preproc_include
  path: (string_literal) @_import_module) @import

(class_specifier
  name: (type_identifier) @_cls_name) @class

(struct_specifier
  name: (type_identifier) @_cls_name) @class

(enum_specifier
  name: (type_identifier) @_cls_name) @class
"""

# ── Plugin ───────────────────────────────────────────────────────────


class CppPlugin:
    """C++ language support — functions, classes, methods, includes."""

    @hookimpl
    def rbtr_register_languages(self) -> list[LanguageRegistration]:
        return [
            LanguageRegistration(
                id="cpp",
                extensions=frozenset({".cpp", ".cc", ".cxx", ".hpp", ".hxx"}),
                grammar_module="tree_sitter_cpp",
                query=_QUERY,
                scope_types=frozenset({"class_specifier", "struct_specifier"}),
                # Same grammar as C — single `comment` node.
                doc_comment_node_types=frozenset({"comment"}),
                source_roots=("", "include", "src"),
                test_prefix="test_",
            ),
        ]
