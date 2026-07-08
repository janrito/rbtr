"""Go language plugin.

Provides full support: functions, methods, type declarations,
and import extraction.

Extracted chunks::

    func hello() {}                 → function "hello", scope ""
    func (u User) Name() string {}  → method "Name", scope ""
    type User struct { ... }        → class "User", scope ""
    type Reader interface { ... }   → class "Reader", scope ""

    import "fmt"
        → import, metadata {module: "fmt"}
    import ("fmt" "os/exec")
        → 2 import chunks: {module: "fmt"}, {module: "os/exec"}
"""

from __future__ import annotations

from rbtr.languages.queries import load_query
from rbtr.languages.registration import LanguageRegistration

# ── Query ────────────────────────────────────────────────────────────


# ── Plugin ───────────────────────────────────────────────────────────


go = LanguageRegistration(
    id="go",
    extensions=frozenset({".go"}),
    grammar_module="tree_sitter_go",
    query=load_query(__package__, "go"),
    scope_types=frozenset({"type_spec"}),
    # Go convention: `//` runs directly above a
    # declaration document it (gofmt preserves this
    # link).  The grammar uses a single `comment`
    # type for both line and block forms.
    doc_comment_node_types=frozenset({"comment"}),
    test_suffix="_test",
    language_plugin_version=3,
)
