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
        → import, metadata {module: "json", names: "*"}
    require_relative "helpers"
        → import, metadata {module: "./helpers", names: "*"}
"""

from __future__ import annotations

from rbtr.languages.registration import (
    LanguageRegistration,
    QueryExtraction,
    load_query,
)

ruby = LanguageRegistration(
    id="ruby",
    extensions=frozenset({".rb"}),
    grammar_module="tree_sitter_ruby",
    extraction=QueryExtraction(
        query=load_query(__package__, "ruby"),
        scope_types=frozenset({"class", "module"}),
    ),
    source_roots=("", "lib"),
    extraction_serial=8,
)
