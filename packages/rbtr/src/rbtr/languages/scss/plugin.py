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

from rbtr.languages.css.plugin import css_nesting_scope
from rbtr.languages.queries import load_query
from rbtr.languages.registration import (
    LanguageRegistration,
    build_quoted_import,
)

scss = LanguageRegistration(
    id="scss",
    extensions=frozenset({".scss"}),
    grammar_module="tree_sitter_scss",
    query=load_query(__package__, "scss"),
    import_targets=frozenset({"css", "scss", "less"}),
    language_plugin_version=1,
)

scss.scope_extractor(css_nesting_scope)
scss.import_extractor(build_quoted_import)
