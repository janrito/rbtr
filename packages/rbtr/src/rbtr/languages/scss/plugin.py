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

from rbtr.languages._queries import load_query
from rbtr.languages.css.plugin import css_nesting_scope
from rbtr.languages.hookspec import (
    LanguageRegistration,
    build_quoted_import,
    hookimpl,
)

_QUERY = load_query(__package__, "scss")


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
