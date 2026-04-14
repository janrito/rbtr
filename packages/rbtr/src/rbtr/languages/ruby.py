"""Ruby language plugin.

Provides symbol extraction (methods, classes, modules) and
structured import metadata from `require` / `require_relative`.

Extracted require examples::

    `require "json"`              → `{"module": "json"}`
    `require "net/http"`          → `{"module": "net/http"}`
    `require_relative "helpers"`  → `{"module": "helpers", "dots": "1"}`
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rbtr.index.models import ImportMeta
from rbtr.languages.hookspec import LanguageRegistration, hookimpl

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
    (string) @_arg)
  (#match? @_call_name "^require"))  @import
"""

# ── Import extractor ─────────────────────────────────────────────────


def _string_content(node: Node) -> str:
    """Extract plain text from a Ruby string node."""
    for child in node.children:
        if child.type == "string_content" and child.text:
            return child.text.decode()
    return ""


def extract_import_meta(node: Node) -> ImportMeta:
    """Extract require/require_relative metadata from a `call` node."""
    method_name = ""
    arg_value = ""
    for child in node.children:
        if child.type == "identifier" and child.text:
            method_name = child.text.decode()
        if child.type == "argument_list":
            for arg in child.children:
                if arg.type == "string":
                    arg_value = _string_content(arg)
                    break

    if not arg_value:
        return {}

    meta = ImportMeta(module=arg_value)
    if method_name == "require_relative":
        meta["dots"] = "1"
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
            ),
        ]
