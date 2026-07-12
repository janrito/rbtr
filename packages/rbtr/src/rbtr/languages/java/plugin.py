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

from rbtr.languages._queries import load_query
from rbtr.languages.hookspec import LanguageRegistration, ModuleStyle, hookimpl

# ── Query ────────────────────────────────────────────────────────────

_QUERY = load_query(__package__, "java")

# ── Plugin ───────────────────────────────────────────────────────────


class JavaPlugin:
    """Java language support.

    Note: Java uses `method_declaration` (not
    `function_declaration`) for all methods, including static
    ones.  The query captures them as `@method`, and
    `extract_symbols` promotes to `ChunkKind.METHOD` when a
    scope is found.
    """

    @hookimpl
    def rbtr_register_languages(self) -> list[LanguageRegistration]:
        return [
            LanguageRegistration(
                id="java",
                extensions=frozenset({".java"}),
                grammar_module="tree_sitter_java",
                query=_QUERY,
                scope_types=frozenset(
                    {
                        "class_declaration",
                        "interface_declaration",
                        "enum_declaration",
                        "record_declaration",
                        "annotation_type_declaration",
                    }
                ),
                # Javadoc uses `block_comment`; `//` runs use
                # `line_comment`.  Attach either when they sit
                # directly above a method or class.
                doc_comment_node_types=frozenset({"block_comment", "line_comment"}),
                source_roots=("", "src/main/java"),
                test_suffix="Test",
                module_style=ModuleStyle.DOTTED,
                language_plugin_version=3,
            ),
        ]
