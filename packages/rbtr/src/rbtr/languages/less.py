"""Less language plugin.

Extends CSS extraction with the Less preprocessor constructs,
via a tree-sitter query. Rule sets and `@media`/`@charset`/
`@keyframes` are captured as doc sections; `@`-prefixed
declarations as variables; `.mixin()` definitions as functions;
`@import` as imports for cross-file edges. Mixin *calls*
(`mixin_statement`) are references, not definitions, and are
skipped.

Extracted chunks::

    @primary: #333;              → variable "@primary"
    .rounded(@r) { ... }         → function "rounded"
    .btn { ... }                 → doc_section ".btn"
    @keyframes slide { ... }     → doc_section "slide"
    @import "reset";             → import, metadata {module: "reset"}
"""

from __future__ import annotations

from rbtr.languages.css import css_nesting_scope
from rbtr.languages.hookspec import (
    LanguageRegistration,
    build_quoted_import,
    hookimpl,
)

_QUERY = r"""
(declaration
  (property_name) @_var_name
  (#match? @_var_name "^[@]")) @variable

(mixin_definition
  (class_name) @_fn_name) @function

(rule_set
  (selectors) @_section_name) @doc_section

(media_statement) @doc_section

(charset_statement) @doc_section

(keyframes_statement
  (keyframes_name) @_section_name) @doc_section

(import_statement
  (string_value) @_import_module) @import
"""


class LessPlugin:
    """Less language support — Less constructs + @import edges."""

    @hookimpl
    def rbtr_register_languages(self) -> list[LanguageRegistration]:
        return [
            LanguageRegistration(
                id="less",
                extensions=frozenset({".less"}),
                grammar_module="tree_sitter_less",
                query=_QUERY,
                scope_extractor=css_nesting_scope,
                import_extractor=build_quoted_import,
                import_targets=frozenset({"css", "scss", "less"}),
                language_plugin_version=1,
            ),
        ]
