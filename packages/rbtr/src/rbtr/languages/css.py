"""CSS language plugin.

Splits CSS by rule sets using their selectors as the chunk
name, via a tree-sitter query. At-rules (`@media`,
`@charset`) are captured as doc sections. `@import`
statements are captured as imports for cross-language edges.

Extracted chunks::

    body { color: #333; }           → doc_section "body", scope ""
    .header { background: blue; }   → doc_section ".header", scope ""
    @media (max-width: 600px) {}    → doc_section "", scope ""
    @import url("reset.css");       → import, metadata {module: "reset.css"}
"""

from __future__ import annotations

from rbtr.languages.hookspec import LanguageRegistration, hookimpl

_QUERY = """\
(rule_set
  (selectors) @_section_name) @doc_section

(media_statement) @doc_section

(charset_statement) @doc_section

(import_statement
  (call_expression
    (arguments
      (string_value (string_content) @_import_module)))) @import

(import_statement
  (string_value (string_content) @_import_module)) @import
"""


class CssPlugin:
    """CSS language support — rule-set extraction + @import edges."""

    @hookimpl
    def rbtr_register_languages(self) -> list[LanguageRegistration]:
        return [
            LanguageRegistration(
                id="css",
                extensions=frozenset({".css"}),
                grammar_module="tree_sitter_css",
                query=_QUERY,
                import_targets=frozenset({"css"}),
            ),
        ]
