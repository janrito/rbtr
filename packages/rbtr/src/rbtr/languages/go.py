"""Go language plugin.

Provides full support: functions, methods, type declarations,
and import extraction.

Extracted import examples::

    'import "fmt"'                → {"module": "fmt"}
    'import ("fmt" "os/exec")'   → {"module": "fmt,os/exec"}
    'import alias "github.com/foo/bar"'
                                  → {"module": "github.com/foo/bar"}
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rbtr.index.models import ImportMeta
from rbtr.languages.hookspec import LanguageRegistration, hookimpl

if TYPE_CHECKING:
    from tree_sitter import Node

# ── Query ────────────────────────────────────────────────────────────

_QUERY = """\
(function_declaration
  name: (identifier) @_fn_name) @function

(method_declaration
  name: (field_identifier) @_method_name) @method

(type_declaration
  (type_spec
    name: (type_identifier) @_cls_name)) @class

(import_declaration) @import
"""

# ── Import extractor ─────────────────────────────────────────────────


def _extract_string(node: Node) -> str | None:
    """Extract the content of an `interpreted_string_literal` node.

    Go string literals in the AST look like::

        interpreted_string_literal
          " (quote)
          interpreted_string_literal_content "fmt"
          " (quote)

    Returns the content text, or `None`.
    """
    for child in node.children:
        if child.type == "interpreted_string_literal_content" and child.text:
            return child.text.decode()
    return None


def extract_import_meta(node: Node) -> ImportMeta:
    """Extract import data from a Go `import_declaration` node.

    Handles both single and grouped import declarations.

    **Single import** — `import "fmt"`::

        >>> extract_import_meta(node)
        {"module": "fmt"}

    **Grouped import** — `import ("fmt"  "os/exec")`::

        >>> extract_import_meta(node)
        {"module": "fmt,os/exec"}

    **Aliased import** — `import alias "github.com/foo/bar"`::

        >>> extract_import_meta(node)
        {"module": "github.com/foo/bar"}

    Grouped imports are joined with `,` since a single Go
    `import_declaration` node may contain multiple specs.
    """
    meta: ImportMeta = {}
    modules: list[str] = []

    for child in node.children:
        match child.type:
            case "import_spec":
                for sc in child.children:
                    if sc.type == "interpreted_string_literal":
                        s = _extract_string(sc)
                        if s:
                            modules.append(s)
            case "import_spec_list":
                for spec in child.children:
                    if spec.type == "import_spec":
                        for sc in spec.children:
                            if sc.type == "interpreted_string_literal":
                                s = _extract_string(sc)
                                if s:
                                    modules.append(s)

    if len(modules) == 1:
        meta["module"] = modules[0]
    elif modules:
        meta["module"] = ",".join(modules)

    return meta


# ── Plugin ───────────────────────────────────────────────────────────


class GoPlugin:
    """Go language support.

    Uses `type_spec` for scope detection because Go's type
    declarations (`type User struct { ... }`) nest the name
    inside a `type_spec` node within the `type_declaration`.
    """

    @hookimpl
    def rbtr_register_languages(self) -> list[LanguageRegistration]:
        return [
            LanguageRegistration(
                id="go",
                extensions=frozenset({".go"}),
                grammar_module="tree_sitter_go",
                query=_QUERY,
                import_extractor=extract_import_meta,
                scope_types=frozenset({"type_spec"}),
                # Go convention: `//` runs directly above a
                # declaration document it (gofmt preserves this
                # link).  The grammar uses a single `comment`
                # type for both line and block forms.
                doc_comment_node_types=frozenset({"comment"}),
            ),
        ]
