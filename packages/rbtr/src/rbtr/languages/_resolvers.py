"""Built-in resolvers — the engine defaults for name/scope/import extraction.

Each is a callable null-object (à la pydantic-ai's `NoOpTracer`): a
`LanguageRegistration` slot defaults to one, `resolve_*` calls it directly, and
an override composes over it. Private module — imported by `registration` (for
the slot defaults) and by `languages.treesitter` (for `NAME_CAPTURE_KEY`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rbtr.domain.models import ImportMeta

if TYPE_CHECKING:
    from tree_sitter import Node


NAME_CAPTURE_KEY: dict[str, str] = {
    "function": "_fn_name",
    "method": "_method_name",
    "class": "_cls_name",
    "variable": "_var_name",
    "doc_section": "_section_name",
    "config_key": "_section_name",
}
"""Maps a structural capture name to its paired `@_*_name` helper capture."""

# Node types whose captured text needs delimiter stripping, each with the
# (open, close) pairs the grammar admits for that type. A grammar that gives
# one node type to every string shape needs every pair it accepts: Ruby's
# `require 'b'` and `require "b"` are both a `string`.
DELIMITER_STRIP: dict[str, tuple[tuple[str, str], ...]] = {
    "system_lib_string": (("<", ">"),),
    "string_literal": (('"', '"'), ("'", "'")),
    "interpreted_string_literal": (('"', '"'),),
    "string": (('"', '"'), ("'", "'")),
}


class DefaultName:
    """Built-in display-name resolver: the paired `@_*_name` capture.

    For symbol captures (`@function`, `@class`, …) the name is the paired
    `@_fn_name` / `@_cls_name` helper capture; for `@import` it is the statement
    text; otherwise empty. A construct with no name carries none: `kind` says
    whether one was expected, and readers label it from its location. An
    override delegates to this.
    """

    def __call__(self, capture_name: str, node: Node, captures: dict[str, list[Node]]) -> str:
        name_key = NAME_CAPTURE_KEY.get(capture_name)
        name_nodes = captures.get(name_key, []) if name_key else []
        first_text = name_nodes[0].text if name_nodes else None
        if first_text:
            return first_text.decode()
        if capture_name == "import" and node.text:
            return node.text.decode().strip()[:120]
        return ""


class DefaultScope:
    """Built-in scope resolver: the `@_scope` capture as a scope segment.

    Returns the text of an `@_scope` node clipped to *node*'s span as a
    one-element list, else `[]`. `@_scope` supplies a non-lexical scope tree
    ancestry cannot reach (a Go method's receiver type). Most languages carry
    none, so this is usually a no-op `[]`; the engine appends it to the
    tree-ancestry scope. An ancestry-based override delegates to this.
    """

    def __call__(
        self, _capture_name: str, node: Node, captures: dict[str, list[Node]]
    ) -> list[str]:
        for s in captures.get("_scope", []):
            if node.start_byte <= s.start_byte and s.end_byte <= node.end_byte and s.text:
                return [s.text.decode()]
        return []


class DefaultImport:
    """Built-in import-metadata resolver: read captures, strip delimiters.

    For languages whose query captures all the import metadata they need. Strips
    delimiters by the capture node's type: `<>` from `system_lib_string`, `"`
    from `string_literal` / `interpreted_string_literal` / `string`. Richer
    imports (Python/JS multi-name, Rust scoped paths) override and delegate to
    this. Examples: C `#include <stdio.h>` → `stdio.h`; Go `import "fmt"` →
    `fmt`; Java `import java.util.List;` → `java.util.List`.
    """

    def __call__(self, _node: Node, captures: dict[str, list[Node]]) -> ImportMeta:
        meta = ImportMeta()
        if (module := self._read(captures, "_import_module")) is not None:
            meta.module = module
        if (names := self._read(captures, "_import_names")) is not None:
            meta.names = names
        return meta

    @staticmethod
    def _read(captures: dict[str, list[Node]], key: str) -> str | None:
        nodes = captures.get(key, [])
        if not (nodes and nodes[0].text):
            return None
        raw = nodes[0].text.decode()
        for opening, closing in DELIMITER_STRIP.get(nodes[0].type, ()):
            if raw.startswith(opening) and raw.endswith(closing):
                return raw[len(opening) : -len(closing)]
        return raw
