"""HTML language plugin.

Extracts semantic and structural elements as doc sections and cross-language
import references, via a tree-sitter query.

Extracted chunks::

    <main id="content">…</main>   → doc_section "content" (named by id)
    <nav>…</nav>                   → doc_section "nav" (named by tag)
    <script src="app.js">          → import {module: "app.js", hint javascript}
    <link href="styles.css">       → import {module: "styles.css", hint css}

Element chunks cover `head`, `body`, sectioning content, landmarks, and
self-contained units; nested elements are each extracted (overlapping, like a
class and its methods). A file with none of these is left to the engine's
presence handling. Inline `<script>`/`<style>` delegate to JavaScript/CSS via
`injection_query`; external `<script src>`/`<link href>` become import edges.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rbtr.index.models import ImportMeta
from rbtr.languages.registration import (
    ImportResolver,
    LanguageRegistration,
    NameResolver,
    QueryExtraction,
    load_query,
)

if TYPE_CHECKING:
    from tree_sitter import Node

# Elements that hold meaning of their own: the document containers, HTML5
# sectioning content and landmarks, and self-contained units. Named by their
# `id` when present, else by tag.


# Inline <script>/<style> delegate to JavaScript/CSS. External <script src>/
# <link href> have no raw_text, so they are skipped here and become imports.


def _get_attr(start_tag: Node, attr_name: str) -> str | None:
    """Extract an attribute value from a start_tag node."""
    for child in start_tag.children:
        if child.type != "attribute":
            continue
        name_node = None
        value_node = None
        for gc in child.children:
            if gc.type == "attribute_name" and gc.text:
                name_node = gc
            elif gc.type == "quoted_attribute_value":
                for vg in gc.children:
                    if vg.type == "attribute_value" and vg.text:
                        value_node = vg
        if (
            name_node is not None
            and value_node is not None
            and name_node.text == attr_name.encode()
        ):
            text = value_node.text
            return text.decode("utf-8", errors="replace") if text else None
    return None


# ── Plugin ───────────────────────────────────────────────────────────


html = LanguageRegistration(
    id="html",
    extensions=frozenset({".html", ".htm"}),
    grammar_module="tree_sitter_html",
    extraction=QueryExtraction(
        query=load_query(__package__, "html"),
    ),
    injection_query=load_query(__package__, "injections"),
    import_targets=frozenset({"javascript", "typescript", "css"}),
    language_plugin_version=3,
)


@html.name_extractor
def _element_name(
    resolver: NameResolver, capture_name: str, node: Node, captures: dict[str, list[Node]]
) -> str:
    """Name a semantic element by its `id`, else its tag; others by default."""
    if capture_name != "doc_section":
        return resolver(capture_name, node, captures)
    start_tag = next((c for c in node.children if c.type == "start_tag"), None)
    if start_tag is not None:
        elem_id = _get_attr(start_tag, "id")
        if elem_id:
            return elem_id
    tag_nodes = captures.get("_tag")
    if tag_nodes and tag_nodes[0].text:
        return tag_nodes[0].text.decode()
    return "<anonymous>"


@html.import_extractor
def _import_meta(
    resolver: ImportResolver, node: Node, captures: dict[str, list[Node]]
) -> ImportMeta:
    """Import metadata with a language hint from the element kind."""
    meta = resolver(node, captures)
    meta.language_hint = "javascript" if node.type == "script_element" else "css"
    return meta
