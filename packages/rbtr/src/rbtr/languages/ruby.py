"""Ruby language plugin.

Provides symbol extraction (methods, classes, modules) and
structured import metadata from `require` / `require_relative`.

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
from rbtr.languages.hookspec import LanguageRegistration, build_import_from_captures, hookimpl

if TYPE_CHECKING:
    from tree_sitter import Node

# ── Query ────────────────────────────────────────────────────────────

_QUERY = """
(method
  name: (identifier) @_fn_name) @function

(singleton_method
  name: (identifier) @_fn_name) @function

(class
  name: (constant) @_cls_name) @class

(module
  name: (constant) @_cls_name) @class

(call
  method: (identifier) @_call_name
  arguments: (argument_list
    (string) @_import_module)
  (#eq? @_call_name "require"))  @import

(call
  method: (identifier) @_call_name
  arguments: (argument_list
    (string) @_import_module)
  (#eq? @_call_name "require_relative"))  @import

(program
  (assignment
    left: (constant) @_var_name) @variable)

(program
  (assignment
    left: (left_assignment_list (constant) @_var_name)) @variable)

(program
  (assignment
    left: (left_assignment_list (rest_assignment (constant) @_var_name))) @variable)
"""

# ── Import extractor ─────────────────────────────────────────────────


def extract_import_meta(node: Node, captures: dict[str, list[Node]]) -> ImportMeta:
    """Extract import data from a Ruby `require` / `require_relative` node.

    Reads `@_import_module` from captures (the query captures
    the string argument), then sets `dots` for
    `require_relative` (always relative to the current file).

    Examples:

        `require "json"`:
            module="json"

        `require_relative "helpers"`:
            module="helpers", dots="1"
    """
    meta = build_import_from_captures(node, captures)
    method = node.child_by_field_name("method")
    if method and method.text == b"require_relative":
        meta.dots = "1"
    return meta


# ── Plugin ───────────────────────────────────────────────────────────


class RubyPlugin:
    """Ruby language support — methods, classes, modules, requires."""

    @hookimpl
    def rbtr_register_languages(self) -> list[LanguageRegistration]:
        return [
            LanguageRegistration(
                id="ruby",
                extensions=frozenset({".rb"}),
                grammar_module="tree_sitter_ruby",
                query=_QUERY,
                import_extractor=extract_import_meta,
                scope_types=frozenset({"class", "module"}),
                # Ruby convention: `#` runs above a `def` or
                # `class` document it.  Single `comment` node
                # type.
                doc_comment_node_types=frozenset({"comment"}),
                source_roots=("", "lib"),
                test_prefix="test_",
                language_plugin_version=2,
            ),
        ]
