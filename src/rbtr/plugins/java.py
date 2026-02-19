"""Java language plugin.

Provides class, method, and import extraction.

Extracted import examples::

    "import java.util.HashMap;"
        → {"module": "java.util", "names": "HashMap"}

    "import com.example.models.*;"
        → {"module": "com.example", "names": "models"}

    "import static org.junit.Assert.assertEquals;"
        → {"module": "org.junit.Assert", "names": "assertEquals"}
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rbtr.index.models import ImportMeta
from rbtr.plugins.hookspec import LanguageRegistration, collect_scoped_path, hookimpl

if TYPE_CHECKING:
    from tree_sitter import Node

# ── Query ────────────────────────────────────────────────────────────

_QUERY = """\
(method_declaration
  name: (identifier) @_method_name) @method

(class_declaration
  name: (identifier) @_cls_name) @class

(import_declaration) @import
"""

# ── Import extractor ─────────────────────────────────────────────────


def extract_import_meta(node: Node) -> ImportMeta:
    """Extract import data from a Java ``import_declaration`` node.

    Uses :func:`~rbtr.plugins.hookspec.collect_scoped_path` to walk
    the nested ``scoped_identifier`` nodes.  The last segment is
    treated as the imported name, everything before it as the
    module path.

    **Class import** — ``import java.util.HashMap;``::

        >>> extract_import_meta(node)
        {"module": "java.util", "names": "HashMap"}

    **Wildcard import** — ``import com.example.models.*;``
    (the ``*`` is a separate AST node; the last scoped segment
    is ``"models"``)::

        >>> extract_import_meta(node)
        {"module": "com.example", "names": "models"}

    **Static import** — ``import static org.junit.Assert.assertEquals;``
    (the ``static`` keyword is skipped; scoped path is walked
    the same way)::

        >>> extract_import_meta(node)
        {"module": "org.junit.Assert", "names": "assertEquals"}
    """
    meta: ImportMeta = {}

    for child in node.children:
        if child.type == "scoped_identifier":
            parts = collect_scoped_path(child)
            if len(parts) > 1:
                meta["module"] = ".".join(parts[:-1])
                meta["names"] = parts[-1]
            elif parts:
                meta["module"] = parts[0]

    return meta


# ── Plugin ───────────────────────────────────────────────────────────


class JavaPlugin:
    """Java language support.

    Note: Java uses ``method_declaration`` (not
    ``function_declaration``) for all methods, including static
    ones.  The query captures them as ``@method``, and
    ``extract_symbols`` promotes to ``ChunkKind.METHOD`` when a
    scope is found.
    """

    @hookimpl
    def rbtr_register_languages(self) -> list[LanguageRegistration]:
        return [
            LanguageRegistration(
                id="java",
                extensions=frozenset({".java"}),
                grammar_module="tree_sitter_java",
                query=_QUERY,
                import_extractor=extract_import_meta,
                scope_types=frozenset({"class_declaration"}),
            ),
        ]
