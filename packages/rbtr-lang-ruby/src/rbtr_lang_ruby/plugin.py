"""Ruby language plugin.

Provides symbol extraction (methods, classes, modules, constants,
and the RSpec `describe`/`context`/`it` DSL) and structured import
metadata from `require` / `require_relative`.  Constants scope to
their enclosing class/module.  RSpec groups (`describe`/`context`/
`feature`) are classes and examples (`it`/`specify`/`example`) are
functions, named by their description string.

Extracted chunks::

    def greet ... end               → function "greet", scope ""
    class Shape ... end             → class "Shape", scope ""
    module Utils ... end            → class "Utils", scope ""
    class Foo
      def bar ... end               → method "bar", scope "Foo"
    end

    require "json"
        → import, metadata {module: "json"}
    require_relative "helpers"
        → import, metadata {module: "helpers", dots: "1"}
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rbtr.index.models import ImportMeta
from rbtr.languages.registration import (
    ImportResolver,
    LanguageRegistration,
    QueryExtraction,
    load_query,
    parse_path_relative,
)

if TYPE_CHECKING:
    from tree_sitter import Node

# ── Query ────────────────────────────────────────────────────────────


# ── Import extractor ─────────────────────────────────────────────────


def extract_import_meta(
    resolver: ImportResolver, node: Node, captures: dict[str, list[Node]]
) -> ImportMeta:
    """Extract import data from a Ruby `require` / `require_relative` node.

    Reads `@_import_module` from captures (the query captures
    the string argument), then sets `dots` for
    `require_relative` (always relative to the current file).

    Examples:

        `require "json"`:
            module="json"

        `require_relative "helpers"`:
            module="helpers", dots="1"

        `require_relative "./config"`:
            module="config", dots="1"

        `require_relative "../lib/utils"`:
            module="lib/utils", dots="2"
    """
    meta = resolver(node, captures)
    method = node.child_by_field_name("method")
    if method and method.text == b"require_relative":
        # `require_relative` is always relative to the current file. Strip any
        # `./`/`../` prefix into `dots` (a bare path is the current dir, dots=1)
        # so the resolver doesn't see a leftover `./` it can't match.
        dots, cleaned = parse_path_relative(meta.module)
        meta.module = cleaned
        meta.dots = str(dots or 1)
    return meta


# ── Plugin ───────────────────────────────────────────────────────────


ruby = LanguageRegistration(
    id="ruby",
    extensions=frozenset({".rb"}),
    grammar_module="tree_sitter_ruby",
    extraction=QueryExtraction(
        query=load_query(__package__, "ruby"),
        scope_types=frozenset({"class", "module"}),
    ),
    source_roots=("", "lib"),
    extraction_serial=5,
)

ruby.import_extractor(extract_import_meta)
