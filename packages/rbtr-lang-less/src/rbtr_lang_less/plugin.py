"""Less language plugin.

Extends CSS extraction with the Less preprocessor constructs,
via a tree-sitter query. Rule sets and `@media`/`@keyframes`
blocks are captured as classes; `@charset` as a config key;
`@`-prefixed declarations as variables; `.mixin()` definitions
as functions;
`@import` as imports for cross-file edges. Mixin *calls*
(`mixin_statement`) are references, not definitions, and are
skipped.

Extracted chunks::

    @primary: #333;              → variable "@primary"
    .rounded(@r) { ... }         → function "rounded"
    .btn { ... }                 → class ".btn"
    @keyframes slide { ... }     → class "slide"
    @import "reset";             → import, metadata {module: "reset"}
"""

from __future__ import annotations

from rbtr_lang_css.plugin import css_nesting_scope

from rbtr.languages.registration import (
    LanguageRegistration,
    QueryExtraction,
    build_quoted_import,
    load_query,
)

less = LanguageRegistration(
    id="less",
    extensions=frozenset({".less"}),
    grammar_module="tree_sitter_less",
    extraction=QueryExtraction(
        query=load_query(__package__, "less"),
    ),
    import_targets=frozenset({"css", "scss", "less"}),
    language_plugin_version=2,
)

less.scope_extractor(css_nesting_scope)
less.import_extractor(build_quoted_import)
