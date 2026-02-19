"""C++ language plugin.

Provides symbol extraction (functions, classes, structs, enums,
methods) and include directive capture.  Scope types include
``class_specifier`` and ``struct_specifier`` so methods inside
classes are correctly scoped.

Extracted include examples::

    ``#include <iostream>``  → ``{"module": "iostream"}``
    ``#include "mylib.h"``   → ``{"module": "mylib.h"}``
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rbtr.index.models import ImportMeta
from rbtr.plugins.hookspec import LanguageRegistration, hookimpl

if TYPE_CHECKING:
    from tree_sitter import Node

# ── Query ────────────────────────────────────────────────────────────

_QUERY = """
(function_definition
  declarator: (function_declarator
    declarator: (identifier) @_fn_name)) @function

(function_definition
  declarator: (function_declarator
    declarator: (field_identifier) @_fn_name)) @function

(preproc_include) @import

(class_specifier
  name: (type_identifier) @_cls_name) @class

(struct_specifier
  name: (type_identifier) @_cls_name) @class

(enum_specifier
  name: (type_identifier) @_cls_name) @class
"""

# ── Import extractor ─────────────────────────────────────────────────


def extract_import_meta(node: Node) -> ImportMeta:
    """Extract include path from a ``preproc_include`` node."""
    for child in node.children:
        if child.type == "system_lib_string" and child.text:
            raw = child.text.decode()
            return ImportMeta(module=raw.strip("<>"))
        if child.type == "string_literal" and child.text:
            raw = child.text.decode()
            return ImportMeta(module=raw.strip('"'))
    return {}


# ── Plugin ───────────────────────────────────────────────────────────


class CppPlugin:
    """C++ language support — functions, classes, methods, includes."""

    @hookimpl
    def rbtr_register_languages(self) -> list[LanguageRegistration]:
        return [
            LanguageRegistration(
                id="cpp",
                extensions=frozenset({".cpp", ".cc", ".cxx", ".hpp", ".hxx"}),
                grammar_module="tree_sitter_cpp",
                query=_QUERY,
                import_extractor=extract_import_meta,
                scope_types=frozenset({"class_specifier", "struct_specifier"}),
            ),
        ]
