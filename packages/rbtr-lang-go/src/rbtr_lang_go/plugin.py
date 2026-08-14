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

import re

from rbtr.languages.registration import LanguageRegistration, QueryExtraction, load_query

# ── Manifest ─────────────────────────────────────────────────────────

_MODULE_LINE = re.compile(r"^\s*module\s+(\S+)", re.MULTILINE)


# ── Plugin ───────────────────────────────────────────────────────────


go = LanguageRegistration(
    id="go",
    extensions=frozenset({".go"}),
    grammar_module="tree_sitter_go",
    extraction=QueryExtraction(
        query=load_query(__package__, "go"),
        scope_types=frozenset({"type_spec"}),
    ),
    package_directory=True,
    extraction_serial=4,
    manifest="go.mod",
)


@go.manifest_reader
def go_module_prefix(text: str) -> tuple[tuple[str, str], ...]:
    """Strip the module path `go.mod` declares from every import.

    Imports inside a module are written from the module root, so `module
    example.com/m` makes `example.com/m/b` the directory `b`.
    """
    match = _MODULE_LINE.search(text)
    if match is None:
        return ()
    return ((f"{match.group(1)}/", ""),)
