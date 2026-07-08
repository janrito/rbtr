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

from rbtr.languages.css.plugin import css_nesting_scope
from rbtr.languages.queries import load_query
from rbtr.languages.registration import (
    LanguageRegistration,
    build_quoted_import,
)

less = LanguageRegistration(
    id="less",
    extensions=frozenset({".less"}),
    grammar_module="tree_sitter_less",
    query=load_query(__package__, "less"),
    import_targets=frozenset({"css", "scss", "less"}),
    language_plugin_version=1,
)

less.scope_extractor(css_nesting_scope)
less.import_extractor(build_quoted_import)
