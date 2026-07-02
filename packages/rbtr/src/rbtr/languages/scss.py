"""SCSS / Sass language plugin.

Extends CSS extraction with the Sass preprocessor constructs,
via a tree-sitter query. Rule sets (incl. `%placeholder`
selectors), `@media`/`@charset`/`@keyframes` are captured as
doc sections; `$`-prefixed declarations as variables; `@mixin`
and `@function` definitions as functions; `@use`/`@forward`/
`@import` as imports for cross-file edges.

Extracted chunks::

    $primary: #333;              → variable "$primary"
    @mixin flex($dir) { ... }    → function "flex"
    @function rem($px) { ... }   → function "rem"
    %card { ... }                → doc_section "%card"
    .btn { ... }                 → doc_section ".btn"
    @keyframes slide { ... }     → doc_section "slide"
    @use "config";               → import, metadata {module: "config"}
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
  (#match? @_var_name "^[$]")) @variable

(mixin_statement
  (identifier) @_fn_name) @function

(function_statement
  (identifier) @_fn_name) @function

(rule_set
  (selectors) @_section_name) @doc_section

(media_statement) @doc_section

(charset_statement) @doc_section

(keyframes_statement
  (keyframes_name) @_section_name) @doc_section

(use_statement
  (string_value) @_import_module) @import

(forward_statement
  (string_value) @_import_module) @import

(import_statement
  (string_value) @_import_module) @import
"""


class ScssPlugin:
    """SCSS language support — Sass constructs + @use/@forward edges."""

    @hookimpl
    def rbtr_register_languages(self) -> list[LanguageRegistration]:
        return [
            LanguageRegistration(
                id="scss",
                extensions=frozenset({".scss"}),
                grammar_module="tree_sitter_scss",
                query=_QUERY,
                scope_extractor=css_nesting_scope,
                import_extractor=build_quoted_import,
                import_targets=frozenset({"css", "scss", "less"}),
                language_plugin_version=1,
            ),
        ]
