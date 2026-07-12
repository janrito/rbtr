"""HCL language plugin.

Extracts each top-level block as a doc section, via a tree-sitter
query. The block name combines its type and labels.

Extracted chunks::

    resource "aws_instance" "web" {}  → doc_section "resource aws_instance web"
    variable "region" {}              → doc_section "variable region"
    terraform {}                      → doc_section "terraform"
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rbtr.languages.registration import (
    LanguageRegistration,
    NameResolver,
    QueryExtraction,
    load_query,
)

if TYPE_CHECKING:
    from tree_sitter import Node

# Top-level blocks only: a `block` whose body is the file's own body,
# not one nested inside another block.


hcl = LanguageRegistration(
    id="hcl",
    extensions=frozenset({".hcl", ".tf"}),
    grammar_module="tree_sitter_hcl",
    extraction=QueryExtraction(
        query=load_query(__package__, "hcl"),
    ),
    language_plugin_version=2,
)


@hcl.name_extractor
def hcl_block_name(
    resolver: NameResolver, capture_name: str, node: Node, captures: dict[str, list[Node]]
) -> str:
    """Name a top-level HCL block by its type and labels.

    `resource "aws_instance" "web"` → `"resource aws_instance web"`;
    a bare `terraform {}` block → `"terraform"`. Non-block captures
    fall back to the default resolver.
    """
    if capture_name != "doc_section":
        return resolver(capture_name, node, captures)
    labels: list[str] = []
    for child in node.children:
        if child.type == "identifier" and child.text:
            labels.append(child.text.decode("utf-8", errors="replace"))
        elif child.type == "string_lit" and child.text:
            labels.append(child.text.decode("utf-8", errors="replace").strip('"'))
    return " ".join(labels)
