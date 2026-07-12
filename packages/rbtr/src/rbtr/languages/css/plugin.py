"""CSS language plugin.

Splits CSS by rule sets using their selectors as the chunk
name, via a tree-sitter query. At-rules (`@media`,
`@charset`) are captured as doc sections. `@import`
statements are captured as imports for cross-language edges.

Extracted chunks::

    body { color: #333; }           → doc_section "body", scope ""
    .header { background: blue; }   → doc_section ".header", scope ""
    @media (max-width: 600px) {}    → doc_section "", scope ""
    @keyframes slide { ... }        → doc_section "slide", scope ""
    @import url("reset.css");       → import, metadata {module: "reset.css"}
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rbtr.languages.registration import (
    LanguageRegistration,
    QueryExtraction,
    ScopeResolver,
    enclosing_nodes_of_type,
    load_query,
)

if TYPE_CHECKING:
    from tree_sitter import Node


def css_nesting_scope(
    _resolver: ScopeResolver, capture_name: str, node: Node, captures: dict[str, list[Node]]
) -> list[str]:
    """Scope a chunk under its ancestor rule-set selectors.

    Shared by the CSS family (CSS/SCSS/Less), whose rule sets nest:
    `.card { .title { … } }` scopes `.title` under `.card`. A rule set
    inside an `@media` block has no rule-set ancestor, so it stays
    unscoped. Segments are outermost-first.
    """
    segments: list[str] = []
    for ancestor in enclosing_nodes_of_type(node, frozenset({"rule_set"})):
        for child in ancestor.children:
            if child.type == "selectors" and child.text:
                segments.append(child.text.decode("utf-8", errors="replace").strip())
                break
    return segments


css = LanguageRegistration(
    id="css",
    extensions=frozenset({".css"}),
    grammar_module="tree_sitter_css",
    extraction=QueryExtraction(
        query=load_query(__package__, "css"),
    ),
    import_targets=frozenset({"css"}),
    language_plugin_version=2,
)

css.scope_extractor(css_nesting_scope)
