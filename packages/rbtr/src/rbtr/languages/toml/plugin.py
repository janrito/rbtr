"""TOML language plugin.

Extracts each table and array-of-tables as a doc section, via a
tree-sitter query. A dotted table is named by its last key segment
and scoped under the preceding segments.

Extracted chunks::

    [project]                       → doc_section "project", scope ""
    [tool.ruff]                     → doc_section "ruff", scope "tool"
    [tool.ruff.lint]                → doc_section "lint", scope "tool::ruff"
    [[locales]]                     → doc_section "locales", scope ""

The dotted-key hierarchy lives in the key *string*, not tree
ancestry, so name and scope are derived by walking the key's
ordered segment nodes (`name_extractor` + `scope_extractor`) —
robust to quoted segments containing dots (`[a."b.c"]`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rbtr.languages.registration import (
    LanguageRegistration,
    NameResolver,
    QueryExtraction,
    ScopeResolver,
    load_query,
)

if TYPE_CHECKING:
    from tree_sitter import Node


def _key_text(node: Node) -> str:
    """Text of a single key node, stripping the quotes off a quoted key."""
    text = node.text.decode("utf-8", errors="replace") if node.text else ""
    if node.type == "quoted_key" and len(text) >= 2 and text[0] in "\"'" and text[-1] == text[0]:
        return text[1:-1]
    return text


def _key_segments(key_node: Node) -> list[str]:
    """Ordered segments of a table key.

    A `bare_key`/`quoted_key` is one segment; a `dotted_key` is its
    child segments in order (left-recursive, so an in-order walk yields
    `[tool, ruff, lint]` for `tool.ruff.lint`).
    """
    if key_node.type in ("bare_key", "quoted_key"):
        return [_key_text(key_node)]
    if key_node.type == "dotted_key":
        segments: list[str] = []
        for child in key_node.children:
            if child.type in ("bare_key", "quoted_key", "dotted_key"):
                segments.extend(_key_segments(child))
        return segments
    return []


toml = LanguageRegistration(
    id="toml",
    extensions=frozenset({".toml"}),
    grammar_module="tree_sitter_toml",
    extraction=QueryExtraction(
        query=load_query(__package__, "toml"),
    ),
    language_plugin_version=2,
)


@toml.name_extractor
def toml_table_name(
    resolver: NameResolver, capture_name: str, node: Node, captures: dict[str, list[Node]]
) -> str:
    """Name a table by its last key segment (`[tool.ruff]` → `"ruff"`)."""
    key_nodes = captures.get("_section_name", [])
    if key_nodes:
        segments = _key_segments(key_nodes[0])
        if segments:
            return segments[-1]
    return resolver(capture_name, node, captures)


@toml.scope_extractor
def toml_table_scope(
    _resolver: ScopeResolver, _capture_name: str, _node: Node, captures: dict[str, list[Node]]
) -> list[str]:
    """Scope a table under its leading key segments (`[tool.ruff]` → `["tool"]`)."""
    key_nodes = captures.get("_section_name", [])
    if not key_nodes:
        return []
    return _key_segments(key_nodes[0])[:-1]
