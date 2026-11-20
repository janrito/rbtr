"""Java language plugin.

Provides class, method, and import extraction.

Extracted chunks::

    class User {}                   → class "User", scope ""
    class Service {
        void process() {}           → method "process", scope "Service"
    }

    import java.util.HashMap
        → import, metadata {module: "java.util.HashMap", names: "HashMap"}
    import static org.junit.Assert.assertEquals
        → import, metadata {module: "org.junit.Assert", names: "assertEquals"}
    import com.example.models.*
        → import, metadata {module: "com.example.models", names: "*"}

A method's unqualified names reach its class's members (`members_in_scope`),
and a package's classes see each other without imports (`directory_scope`).
"""

from __future__ import annotations

from rbtr.domain.edge_frames import ModuleStyle
from rbtr.languages.registration import LanguageRegistration, QueryExtraction, load_query

java = LanguageRegistration(
    id="java",
    extensions=frozenset({".java"}),
    grammar_module="tree_sitter_java",
    extraction=QueryExtraction(
        query=load_query(__package__, "java") + load_query(__package__, "references"),
        scope_types=frozenset(
            {
                "class_declaration",
                "interface_declaration",
                "enum_declaration",
                "record_declaration",
                "annotation_type_declaration",
            }
        ),
    ),
    source_roots=("", "src/main/java"),
    module_style=ModuleStyle.DOTTED,
    package_directory=True,
    members_in_scope=True,
    directory_scope=True,
    extraction_serial=7,
)
