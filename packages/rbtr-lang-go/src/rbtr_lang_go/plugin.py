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

from rbtr.languages.registration import LanguageRegistration, QueryExtraction, load_query

# ── Query ────────────────────────────────────────────────────────────


# ── Plugin ───────────────────────────────────────────────────────────


go = LanguageRegistration(
    id="go",
    extensions=frozenset({".go"}),
    grammar_module="tree_sitter_go",
    extraction=QueryExtraction(
        query=load_query(__package__, "go"),
        scope_types=frozenset({"type_spec"}),
    ),
    language_plugin_version=4,
)
