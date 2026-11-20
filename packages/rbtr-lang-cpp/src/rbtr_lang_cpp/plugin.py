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
        → import, metadata {module: "iostream", names: "*"}
"""

from __future__ import annotations

from rbtr.languages.registration import LanguageRegistration, QueryExtraction, load_query

cpp = LanguageRegistration(
    id="cpp",
    extensions=frozenset({".cpp", ".cc", ".cxx", ".hpp", ".hxx"}),
    grammar_module="tree_sitter_cpp",
    extraction=QueryExtraction(
        query=load_query(__package__, "cpp") + load_query(__package__, "references"),
        scope_types=frozenset({"class_specifier", "struct_specifier", "namespace_definition"}),
        class_scope_types=frozenset({"class_specifier", "struct_specifier"}),
    ),
    import_targets=frozenset({"cpp", "c"}),
    source_roots=("", "include", "src"),
    members_in_scope=True,
    linked_functions=True,
    extraction_serial=9,
)
