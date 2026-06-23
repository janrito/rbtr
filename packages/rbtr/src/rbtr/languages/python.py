"""Python language plugin.

Provides full support: symbol extraction (functions, classes,
methods, imports) and structured import metadata.

Extracted chunks::

    def hello():            → function "hello", scope ""
        pass

    class Config:           → class "Config", scope ""
        def load(self):     → method "load", scope "Config"
            pass

    import os               → import, metadata {module: "os"}
    from pathlib import Path
                            → import, metadata {module: "pathlib", names: "Path"}
    from ..core import engine
                            → import, metadata {dots: "2", module: "core",
                                                names: "engine"}
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rbtr.index.models import ImportMeta
from rbtr.languages.hookspec import (
    LanguageRegistration,
    ModuleStyle,
    build_import_from_captures,
    hookimpl,
)

if TYPE_CHECKING:
    from tree_sitter import Node

# ── Query ────────────────────────────────────────────────────────────

# The `@_docstring` sub-capture marks the first-statement body
# docstring for doc-range detection (used by `extract_doc_spans`
# and the eval query sampler).  It is optional in the match
# (`?`), so functions/classes without a docstring still match.
_QUERY = """\
(function_definition
  name: (identifier) @_fn_name
  body: (block
    . (expression_statement (string) @_docstring)?)) @function

(class_definition
  name: (identifier) @_cls_name
  body: (block
    . (expression_statement (string) @_docstring)?)) @class

(module
  (expression_statement
    (assignment
      left: (identifier) @_var_name) @variable))

(module
  (expression_statement
    (assignment
      left: (pattern_list (identifier) @_var_name)) @variable))

(module
  (expression_statement
    (assignment
      left: (tuple_pattern (identifier) @_var_name)) @variable))

(module
  (expression_statement
    (assignment
      left: (list_pattern (identifier) @_var_name)) @variable))

(module
  (expression_statement
    (assignment
      left: (pattern_list (list_splat_pattern (identifier) @_var_name))) @variable))

(import_statement
  name: (dotted_name) @_import_module) @import

(import_from_statement
  module_name: (dotted_name) @_import_module) @import

(import_from_statement
  module_name: (relative_import
    (import_prefix) @_import_dots
    (dotted_name) @_import_module)) @import

(import_from_statement
  module_name: (relative_import
    (import_prefix) @_import_dots .)) @import
"""

# ── Import extractor ─────────────────────────────────────────────────


def extract_import_meta(node: Node, captures: dict[str, list[Node]]) -> ImportMeta:
    """Extract structured import data from a Python import node.

    Reads query captures (`@_import_module`, `@_import_dots`)
    and walks the node for multi-valued import names (which
    the query can't capture).

    Examples:

        `import os.path`:
            module="os.path"

        `from pathlib import Path`:
            module="pathlib", names="Path"

        `from ..core import engine`:
            dots="2", module="core", names="engine"

        `from .models import Chunk as C`:
            dots="1", module="models", names="Chunk"

        `from . import utils`:
            dots="1", names="utils"
    """
    meta = build_import_from_captures(node, captures)

    # Convert @_import_dots from raw import_prefix (e.g. "..")
    # to a count string.
    dots_nodes = captures.get("_import_dots", [])
    if dots_nodes and dots_nodes[0].text:
        meta.dots = str(dots_nodes[0].text.decode().count("."))

    if node.type == "import_from_statement":
        # Imported names.  The `name` field can be dotted_name
        # or aliased_import; there may be multiple `name` fields.
        names: list[str] = []
        for name_node in node.children_by_field_name("name"):
            match name_node.type:
                case "dotted_name":
                    if name_node.text:
                        names.append(name_node.text.decode())
                case "aliased_import":
                    original = name_node.child_by_field_name("name")
                    if original and original.text:
                        names.append(original.text.decode())
        if names:
            meta.names = ",".join(names)

    return meta


# ── Plugin ───────────────────────────────────────────────────────────


class PythonPlugin:
    """Python language support.

    Registers `.py` and `.pyi` files.  Uses
    `class_definition` for scope detection (Python's AST uses
    `class_definition`, not `class_declaration`).
    """

    @hookimpl
    def rbtr_register_languages(self) -> list[LanguageRegistration]:
        return [
            LanguageRegistration(
                id="python",
                extensions=frozenset({".py", ".pyi"}),
                grammar_module="tree_sitter_python",
                query=_QUERY,
                import_extractor=extract_import_meta,
                scope_types=frozenset({"class_definition"}),
                index_files=frozenset({"__init__.py"}),
                source_roots=("", "src"),
                test_prefix="test_",
                module_style=ModuleStyle.DOTTED,
                language_plugin_version=2,
            ),
        ]
