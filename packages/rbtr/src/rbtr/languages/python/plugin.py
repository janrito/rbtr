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
from rbtr.languages.queries import load_query
from rbtr.languages.registration import (
    ImportResolver,
    LanguageRegistration,
    ModuleStyle,
)

if TYPE_CHECKING:
    from tree_sitter import Node

# ── Query ────────────────────────────────────────────────────────────

# The `@_docstring` sub-capture marks the first-statement body
# docstring for doc-range detection (used by `extract_doc_spans`
# and the eval query sampler).  It is optional in the match
# (`?`), so functions/classes without a docstring still match.

# ── Import extractor ─────────────────────────────────────────────────


def extract_import_meta(
    resolver: ImportResolver, node: Node, captures: dict[str, list[Node]]
) -> ImportMeta:
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
    meta = resolver(node, captures)

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


python = LanguageRegistration(
    id="python",
    extensions=frozenset({".py", ".pyi"}),
    grammar_module="tree_sitter_python",
    query=load_query(__package__, "python"),
    scope_types=frozenset({"class_definition", "function_definition"}),
    class_scope_types=frozenset({"class_definition"}),
    index_files=frozenset({"__init__.py"}),
    source_roots=("", "src"),
    module_style=ModuleStyle.DOTTED,
    language_plugin_version=4,
)

python.import_extractor(extract_import_meta)
