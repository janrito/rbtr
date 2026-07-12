"""Java language plugin.

Provides class, method, and import extraction.

Extracted chunks::

    class User {}                   → class "User", scope ""
    class Service {
        void process() {}           → method "process", scope "Service"
    }

    import java.util.HashMap
        → import, metadata {module: "java.util.HashMap"}
    import static org.junit.Assert.assertEquals
        → import, metadata {module: "org.junit.Assert.assertEquals"}
"""

from __future__ import annotations

from rbtr.languages.registration import (
    LanguageRegistration,
    ModuleStyle,
    QueryExtraction,
    load_query,
)

# ── Query ────────────────────────────────────────────────────────────


# ── Plugin ───────────────────────────────────────────────────────────


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
    extraction_serial=4,
)
