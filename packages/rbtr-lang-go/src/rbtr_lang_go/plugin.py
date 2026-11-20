"""Go language plugin.

Provides full support: functions, methods, type declarations,
and import extraction.

Extracted chunks::

    func hello() {}                 → function "hello", scope ""
    func (u User) Name() string {}  → method "Name", scope ""
    type User struct { ... }        → class "User", scope ""
    type Reader interface { ... }   → class "Reader", scope ""

    import "fmt"
        → import, metadata {module: "fmt"}, binding `fmt`
    import ("fmt" "os/exec")
        → 2 import chunks: {module: "fmt"}, {module: "os/exec"}, binding `exec`

A package's files see each other's names without imports
(`directory_scope`), and an embedded struct field is a base; the
shallowest embedding's method wins (`MemberOrder.NEAREST`).
"""

from __future__ import annotations

import re

from rbtr.domain.edge_frames import MemberOrder
from rbtr.languages.registration import (
    LanguageRegistration,
    QueryExtraction,
    load_query,
)

_MODULE_LINE = re.compile(r"^\s*module\s+(\S+)", re.MULTILINE)


go = LanguageRegistration(
    id="go",
    extensions=frozenset({".go"}),
    grammar_module="tree_sitter_go",
    extraction=QueryExtraction(
        query=load_query(__package__, "go") + load_query(__package__, "references"),
        scope_types=frozenset({"type_spec"}),
    ),
    package_directory=True,
    directory_scope=True,
    member_order=MemberOrder.NEAREST,
    extraction_serial=8,
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
