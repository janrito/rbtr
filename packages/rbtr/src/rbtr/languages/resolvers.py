"""Built-in resolvers — the engine defaults for name and scope extraction, and import reading.

The name and scope resolvers are callable null-objects (à la pydantic-ai's
`NoOpTracer`): a `LanguageRegistration` slot defaults to one, `resolve_*` calls
it directly, and an override composes over it. `read_import` reads an import's
module for every language. Private module — imported by `registration` (for
the slot defaults) and by `languages.treesitter`.
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
    "test_function": "_test_name",
    "doc_section": "_section_name",
    "config_key": "_section_name",
}
"""Maps a structural capture name to its paired `@_*_name` helper capture."""

# Delimiters a captured import string may be wrapped in, opening to closing.
# C writes `#include <stdio.h>`, and a language that quotes with either mark
# writes both `require "b"` and `require 'b'`.
STRING_DELIMITERS: dict[str, str] = {"<": ">", '"': '"', "'": "'"}


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
    """Built-in scope resolver: the `@_scope` captures as scope segments.

    Returns the text of each `@_scope` node within *node*'s span, in source
    order (`shop::Order::close` captured as `shop` and `Order` gives both),
    else `[]`. `@_scope` supplies a non-lexical scope tree ancestry cannot
    reach (a Go method's receiver type). Most languages carry none, so this
    is usually a no-op `[]`; the engine appends it to the tree-ancestry
    scope. An ancestry-based override delegates to this.
    """

    def __call__(
        self, _capture_name: str, node: Node, captures: dict[str, list[Node]]
    ) -> list[str]:
        return [
            s.text.decode()
            for s in sorted(captures.get("_scope", []), key=lambda s: s.start_byte)
            if node.start_byte <= s.start_byte and s.end_byte <= node.end_byte and s.text
        ]


def read_import(captures: dict[str, list[Node]]) -> ImportMeta:
    """The import a match's captures state: its `@_import_module`, delimiters stripped.

    A captured module wrapped in a matching pair from `STRING_DELIMITERS` is
    unwrapped; anything else is read as it stands: C `#include <stdio.h>` is
    `stdio.h`, Go `import "fmt"` is `fmt`. Names, bindings and settings come
    from the query's own captures and directives.
    """
    meta = ImportMeta()
    nodes = captures.get("_import_module", [])
    if nodes and nodes[0].text:
        raw = nodes[0].text.decode()
        for opening, closing in STRING_DELIMITERS.items():
            if len(raw) > 1 and raw.startswith(opening) and raw.endswith(closing):
                raw = raw[1:-1]
                break
        meta.module = raw
    return meta
