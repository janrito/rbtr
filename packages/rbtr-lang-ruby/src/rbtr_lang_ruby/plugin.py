"""Ruby language plugin.

Provides symbol extraction (methods, classes, modules, constants,
and the RSpec `describe`/`context`/`it` DSL) and structured import
metadata from `require` / `require_relative`.  Constants scope to
their enclosing class/module.  RSpec groups (`describe`/`context`/
`feature`) and examples (`it`/`specify`/`example`) are tests, named by
their description string, which no call reaches.  A class's bases are
searched last written first (`MemberOrder.REVERSED`): a later `include`
before an earlier one, and both before the superclass.

Extracted chunks::

    def greet ... end               → function "greet", scope ""
    class Shape ... end             → class "Shape", scope ""
    module Utils ... end            → class "Utils", scope ""
    class Foo
      def bar ... end               → method "bar", scope "Foo"
    end

    require "json"
        → import, imports [{module: "json", names: "*"}]
    require_relative "helpers"
        → import, imports [{module: "./helpers", names: "*"}]
"""

from __future__ import annotations

from rbtr.domain.models import MemberOrder
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
        query=load_query(__package__, "ruby") + load_query(__package__, "references"),
        scope_types=frozenset({"class", "module"}),
    ),
    source_roots=("", "lib"),
    members_in_scope=True,
    member_order=MemberOrder.REVERSED,
    extraction_serial=13,
)
