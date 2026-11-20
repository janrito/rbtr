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
"""

from __future__ import annotations

from rbtr.domain.edge_frames import ModuleStyle
from rbtr.languages.registration import LanguageRegistration, QueryExtraction, load_query

java = LanguageRegistration(
    id="java",
    extensions=frozenset({".java"}),
    grammar_module="tree_sitter_java",
    extraction=QueryExtraction(
        query=load_query(__package__, "java"),
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
    extraction_serial=6,
)
