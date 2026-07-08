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

from rbtr.languages.queries import load_query
from rbtr.languages.registration import LanguageRegistration

# ── Query ────────────────────────────────────────────────────────────


# ── Plugin ───────────────────────────────────────────────────────────


cpp = LanguageRegistration(
    id="cpp",
    extensions=frozenset({".cpp", ".cc", ".cxx", ".hpp", ".hxx"}),
    grammar_module="tree_sitter_cpp",
    query=load_query(__package__, "cpp"),
    scope_types=frozenset({"class_specifier", "struct_specifier", "namespace_definition"}),
    class_scope_types=frozenset({"class_specifier", "struct_specifier"}),
    # Same grammar as C — single `comment` node.
    doc_comment_node_types=frozenset({"comment"}),
    source_roots=("", "include", "src"),
    test_prefix="test_",
    language_plugin_version=5,
)
