"""C language plugin.

Provides symbol extraction (functions, structs, enums) and
include directive capture.  C has no module system — `#include`
directives are captured as import chunks with the header path
in the `module` metadata field.

Extracted include examples::

    `#include <stdio.h>`   → `{"module": "stdio.h"}`
    `#include "mylib.h"`   → `{"module": "mylib.h"}`
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rbtr.index.models import ImportMeta
from rbtr.languages.hookspec import LanguageRegistration, hookimpl

if TYPE_CHECKING:
    from tree_sitter import Node

# ── Query ────────────────────────────────────────────────────────────

_QUERY = """
(function_definition
  declarator: (function_declarator
    declarator: (identifier) @_fn_name)) @function

(preproc_include) @import

(struct_specifier
  name: (type_identifier) @_cls_name) @class

(enum_specifier
  name: (type_identifier) @_cls_name) @class

(type_definition
  declarator: (type_identifier) @_cls_name) @class
"""

# ── Import extractor ─────────────────────────────────────────────────


def extract_import_meta(node: Node) -> ImportMeta:
    """Extract include path from a `preproc_include` node.

    Handles both `<header.h>` (system) and `"header.h"` (local).
    """
    for child in node.children:
        if child.type == "system_lib_string" and child.text:
            # <stdio.h> → strip angle brackets.
            raw = child.text.decode()
            return ImportMeta(module=raw.strip("<>"))
        if child.type == "string_literal" and child.text:
            # "mylib.h" → strip quotes.
            raw = child.text.decode()
            return ImportMeta(module=raw.strip('"'))
    return {}


# ── Plugin ───────────────────────────────────────────────────────────


class CPlugin:
    """C language support — functions, structs, enums, includes."""

    @hookimpl
    def rbtr_register_languages(self) -> list[LanguageRegistration]:
        return [
            LanguageRegistration(
                id="c",
                extensions=frozenset({".c", ".h"}),
                grammar_module="tree_sitter_c",
                query=_QUERY,
                import_extractor=extract_import_meta,
                scope_types=frozenset(),
            ),
        ]
