"""SCSS / Sass language plugin.

Extends CSS extraction with the Sass preprocessor constructs,
via a tree-sitter query. Rule sets (incl. `%placeholder`
selectors) and `@media`/`@keyframes` blocks are captured as
classes; `@charset` as a config key; `$`-prefixed declarations
as variables; `@mixin` and `@function` definitions as functions;
`@use`/`@forward`/`@import` as imports for cross-file edges.

Extracted chunks::

    $primary: #333;              → variable "$primary"
    @mixin flex($dir) { ... }    → function "flex"
    @function rem($px) { ... }   → function "rem"
    %card { ... }                → class "%card"
    .btn { ... }                 → class ".btn"
    @keyframes slide { ... }     → class "slide"
    @use "config";               → import, metadata {module: "config"}
"""

from __future__ import annotations

from rbtr_lang_css.plugin import css_nesting_scope

from rbtr.languages.registration import (
    LanguageRegistration,
    QueryExtraction,
    build_quoted_import,
    load_query,
)

scss = LanguageRegistration(
    id="scss",
    extensions=frozenset({".scss"}),
    grammar_module="tree_sitter_scss",
    extraction=QueryExtraction(
        query=load_query(__package__, "scss"),
    ),
    import_targets=frozenset({"css", "scss", "less"}),
    language_plugin_version=3,
)

scss.scope_extractor(css_nesting_scope)
scss.import_extractor(build_quoted_import)
