"""Rust language plugin.

Provides full support: functions, structs, enums, impl blocks,
and `use` declaration extraction.

Extracted import examples::

    "use std::collections::HashMap;"
        → {"module": "std/collections", "names": "HashMap"}

    "use crate::models::{Chunk, Edge};"
        → {"module": "crate/models", "names": "Chunk,Edge"}

    "use super::utils;"
        → {"names": "utils", "dots": "2"}

    "use super::super::helpers::run;"
        → {"module": "helpers", "names": "run", "dots": "3"}
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rbtr.index.models import ImportMeta
from rbtr.languages.hookspec import LanguageRegistration, collect_scoped_path, hookimpl

if TYPE_CHECKING:
    from tree_sitter import Node

# ── Query ────────────────────────────────────────────────────────────

_QUERY = """\
(function_item
  name: (identifier) @_fn_name) @function

(struct_item
  name: (type_identifier) @_cls_name) @class

(enum_item
  name: (type_identifier) @_cls_name) @class

(impl_item
  type: (type_identifier) @_cls_name) @class

(use_declaration) @import
"""

# ── Import extractor ─────────────────────────────────────────────────


def _path_to_meta(parts: list[str], meta: ImportMeta) -> None:
    """Convert collected Rust path segments into module/dots metadata.

    Leading `super` segments are counted as relative dots using
    the unified convention (each `super` = one extra level up,
    plus 1 for the file itself).  `crate` is kept as a literal
    path segment — it's root-relative, not parent-relative.

    Examples::

        >>> meta = {}
        >>> _path_to_meta(["std", "collections"], meta)
        >>> meta
        {"module": "std/collections"}

        >>> meta = {}
        >>> _path_to_meta(["super"], meta)
        >>> meta
        {"dots": "2"}

        >>> meta = {}
        >>> _path_to_meta(["super", "super", "helpers"], meta)
        >>> meta
        {"dots": "3", "module": "helpers"}

        >>> meta = {}
        >>> _path_to_meta(["crate", "models"], meta)
        >>> meta
        {"module": "crate/models"}
    """
    dots = 0
    while parts and parts[0] == "super":
        dots += 1
        parts = parts[1:]
    if dots:
        meta["dots"] = str(dots + 1)  # +1: super = parent dir = 2 levels from file

    if parts:
        meta["module"] = "/".join(parts)


def extract_import_meta(node: Node) -> ImportMeta:
    """Extract import data from a Rust `use_declaration` node.

    Uses `collect_scoped_path` to walk
    the nested `scoped_identifier` nodes.

    **Simple scoped** — `use std::collections::HashMap;`::

        >>> extract_import_meta(node)
        {"module": "std/collections", "names": "HashMap"}

    **Crate-relative with braces** —
    `use crate::models::{Chunk, Edge};`::

        >>> extract_import_meta(node)
        {"module": "crate/models", "names": "Chunk,Edge"}

    **Super-relative** — `use super::utils;`::

        >>> extract_import_meta(node)
        {"names": "utils", "dots": "2"}

    **Self re-export** — `use std::io::{self, Read};`::

        >>> extract_import_meta(node)
        {"module": "std/io", "names": "self,Read"}
    """
    meta: ImportMeta = {}

    for child in node.children:
        match child.type:
            case "scoped_identifier":
                parts = collect_scoped_path(child)
                if len(parts) > 1:
                    _path_to_meta(parts[:-1], meta)
                    meta["names"] = parts[-1]
                elif parts:
                    _path_to_meta(parts, meta)

            case "scoped_use_list":
                path_parts: list[str] = []
                names: list[str] = []
                for sc in child.children:
                    if sc.type == "scoped_identifier":
                        path_parts = collect_scoped_path(sc)
                    elif sc.type == "use_list":
                        for item in sc.children:
                            if item.type == "identifier" and item.text:
                                names.append(item.text.decode())
                            elif item.type == "self":
                                names.append("self")
                _path_to_meta(path_parts, meta)
                if names:
                    meta["names"] = ",".join(names)

            case "identifier":
                if child.text:
                    meta["module"] = child.text.decode()

    return meta


# ── Plugin ───────────────────────────────────────────────────────────


class RustPlugin:
    """Rust language support.

    Uses `impl_item` and `struct_item` for scope detection
    because Rust methods are defined inside `impl` blocks, not
    directly inside `struct` definitions.
    """

    @hookimpl
    def rbtr_register_languages(self) -> list[LanguageRegistration]:
        return [
            LanguageRegistration(
                id="rust",
                extensions=frozenset({".rs"}),
                grammar_module="tree_sitter_rust",
                query=_QUERY,
                import_extractor=extract_import_meta,
                scope_types=frozenset({"impl_item", "struct_item"}),
                # Rust splits doc comments (`///`, `//!`) from
                # regular line/block comments at parse time by
                # wrapping them in distinct grammar rules, but
                # the containing sibling type is always
                # `line_comment` or `block_comment`.  Include
                # both so any leading comment run attaches.
                doc_comment_node_types=frozenset({"line_comment", "block_comment"}),
            ),
        ]
