"""C++ language plugin.

Provides symbol extraction (functions, prototypes, classes, structs,
unions, enums, type aliases, namespaces, methods including operator
overloads, namespace/global variables, and function/object-like
macros) and include directive capture.  Namespaces are both a symbol
(class) and a naming scope.

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

(function_definition
  declarator: (function_declarator
    declarator: (operator_name) @_fn_name)) @function

(field_declaration
  declarator: (function_declarator
    declarator: (field_identifier) @_method_name)) @method

(field_declaration_list
  (declaration
    declarator: (function_declarator
      declarator: (identifier) @_method_name)) @method)

(preproc_include
  path: (system_lib_string) @_import_module) @import

(preproc_include
  path: (string_literal) @_import_module) @import

(preproc_function_def
  name: (identifier) @_fn_name) @function

(preproc_def
  name: (identifier) @_var_name) @variable

(class_specifier
  name: (type_identifier) @_cls_name
  body: (field_declaration_list)) @class

(struct_specifier
  name: (type_identifier) @_cls_name
  body: (field_declaration_list)) @class

(union_specifier
  name: (type_identifier) @_cls_name
  body: (field_declaration_list)) @class

(enum_specifier
  name: (type_identifier) @_cls_name
  body: (enumerator_list)) @class

(alias_declaration
  name: (type_identifier) @_cls_name) @class

(namespace_definition
  name: (namespace_identifier) @_cls_name) @class

(concept_definition
  name: (identifier) @_cls_name) @class

(translation_unit
  (declaration
    declarator: (function_declarator
      declarator: (identifier) @_fn_name)) @function)

(namespace_definition
  body: (declaration_list
    (declaration
      declarator: (function_declarator
        declarator: (identifier) @_fn_name)) @function))

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

(namespace_definition
  body: (declaration_list
    (declaration
      declarator: (init_declarator
        declarator: (identifier) @_var_name)) @variable))

(namespace_definition
  body: (declaration_list
    (declaration
      declarator: (pointer_declarator
        declarator: (identifier) @_var_name)) @variable))

(namespace_definition
  body: (declaration_list
    (declaration
      declarator: (init_declarator
        declarator: (pointer_declarator
          declarator: (identifier) @_var_name))) @variable))
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
                scope_types=frozenset(
                    {"class_specifier", "struct_specifier", "namespace_definition"}
                ),
                class_scope_types=frozenset({"class_specifier", "struct_specifier"}),
                # Same grammar as C — single `comment` node.
                doc_comment_node_types=frozenset({"comment"}),
                source_roots=("", "include", "src"),
                test_prefix="test_",
                language_plugin_version=5,
            ),
        ]
