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
        → import, metadata {module: "./helpers"}
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rbtr.domain.models import ImportMeta
from rbtr.languages.registration import (
    ImportResolver,
    LanguageRegistration,
    QueryExtraction,
    load_query,
)

if TYPE_CHECKING:
    from tree_sitter import Node


def extract_import_meta(
    resolver: ImportResolver, node: Node, captures: dict[str, list[Node]]
) -> ImportMeta:
    """Extract import data from a Ruby `require` / `require_relative` node.

    Reads `@_import_module` from captures (the query captures
    the string argument). `require_relative` is relative to the current
    file even without a `./`, so a bare path is written as one.

    Examples:

        `require "json"`:
            module="json"

        `require_relative "helpers"`:
            module="./helpers"

        `require_relative "./config"`:
            module="./config"

        `require_relative "../lib/utils"`:
            module="../lib/utils"
    """
    meta = resolver(node, captures)
    method = node.child_by_field_name("method")
    relative = method is not None and method.text == b"require_relative"
    if relative and meta.module and not meta.module.startswith(("./", "../")):
        meta.module = f"./{meta.module}"
    return meta


ruby = LanguageRegistration(
    id="ruby",
    extensions=frozenset({".rb"}),
    grammar_module="tree_sitter_ruby",
    extraction=QueryExtraction(
        query=load_query(__package__, "ruby"),
        scope_types=frozenset({"class", "module"}),
    ),
    source_roots=("", "lib"),
    extraction_serial=7,
)

ruby.import_extractor(extract_import_meta)
