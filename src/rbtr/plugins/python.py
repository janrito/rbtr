"""Python language plugin.

Provides full support: symbol extraction (functions, classes,
methods, imports) and structured import metadata.

Extracted import examples::

    "import os"                        → {"module": "os"}
    "import os.path"                   → {"module": "os.path"}
    "from pathlib import Path"         → {"module": "pathlib", "names": "Path"}
    "from . import utils"              → {"dots": "1", "names": "utils"}
    "from ..core import engine"        → {"dots": "2", "module": "core",
                                           "names": "engine"}
    "from .models import Chunk as C"   → {"dots": "1", "module": "models",
                                           "names": "Chunk"}
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rbtr.index.models import ImportMeta
from rbtr.plugins.hookspec import LanguageRegistration, hookimpl

if TYPE_CHECKING:
    from tree_sitter import Node

# ── Query ────────────────────────────────────────────────────────────

_QUERY = """\
(function_definition
  name: (identifier) @_fn_name) @function

(class_definition
  name: (identifier) @_cls_name) @class

(import_statement) @import
(import_from_statement) @import
"""

# ── Import extractor ─────────────────────────────────────────────────


def extract_import_meta(node: Node) -> ImportMeta:
    """Extract structured import data from a Python import node.

    Walks the tree-sitter AST for ``import_statement`` and
    ``import_from_statement`` nodes.

    Examples:

        ``import os.path`` — bare import::

            >>> extract_import_meta(node)
            {"module": "os.path"}

        ``from pathlib import Path`` — absolute from-import::

            >>> extract_import_meta(node)
            {"module": "pathlib", "names": "Path"}

        ``from ..core import engine`` — relative from-import::

            >>> extract_import_meta(node)
            {"dots": "2", "module": "core", "names": "engine"}

        ``from .models import Chunk as C`` — aliased import
        (extracts the original name, not the alias)::

            >>> extract_import_meta(node)
            {"dots": "1", "module": "models", "names": "Chunk"}

        ``from . import utils`` — bare relative (no module tail)::

            >>> extract_import_meta(node)
            {"dots": "1", "names": "utils"}
    """
    meta: ImportMeta = {}

    match node.type:
        case "import_statement":
            for child in node.children:
                if child.type == "dotted_name" and child.text:
                    meta["module"] = child.text.decode()
                    break

        case "import_from_statement":
            names: list[str] = []
            seen_source = False
            for child in node.children:
                match child.type:
                    case "dotted_name" if not seen_source:
                        meta["module"] = child.text.decode() if child.text else ""
                        seen_source = True
                    case "relative_import":
                        dots = 0
                        module_parts: list[str] = []
                        for rc in child.children:
                            if rc.type == "import_prefix" and rc.text:
                                dots = rc.text.decode().count(".")
                            elif rc.type == "dotted_name" and rc.text:
                                module_parts.append(rc.text.decode())
                        meta["dots"] = str(dots)
                        if module_parts:
                            meta["module"] = ".".join(module_parts)
                        seen_source = True
                    case "dotted_name":
                        if child.text:
                            names.append(child.text.decode())
                    case "aliased_import":
                        for ac in child.children:
                            if ac.type == "dotted_name" and ac.text:
                                names.append(ac.text.decode())
                                break
            if names:
                meta["names"] = ",".join(names)

    return meta


# ── Plugin ───────────────────────────────────────────────────────────


class PythonPlugin:
    """Python language support.

    Registers ``.py`` and ``.pyi`` files.  Uses
    ``class_definition`` for scope detection (Python's AST uses
    ``class_definition``, not ``class_declaration``).
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
            ),
        ]
