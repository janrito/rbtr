"""C language plugin.

Provides symbol extraction (functions, function prototypes, structs,
unions, enums, typedefs, global variables, and function/object-like
macros) and include directive capture.  Object-like macros and
pointer-declared globals are variables; function-like macros and
prototypes are functions.

Extracted chunks::

    int add(int a, int b) { ... }   → function "add", scope ""
    struct Node { int value; };     → class "Node", scope ""
    enum Color { RED, GREEN };      → class "Color", scope ""
    typedef struct { ... } Point;   → class "Point", scope ""

    #include <stdio.h>
        → import, metadata {module: "stdio.h"}
    #include "mylib.h"
        → import, metadata {module: "mylib.h"}
"""

from __future__ import annotations

from rbtr.languages.queries import load_query
from rbtr.languages.registration import LanguageRegistration

# ── Query ────────────────────────────────────────────────────────────


# ── Plugin ───────────────────────────────────────────────────────────


c = LanguageRegistration(
    id="c",
    extensions=frozenset({".c", ".h"}),
    grammar_module="tree_sitter_c",
    query=load_query(__package__, "c"),
    # C grammar uses a single `comment` node for
    # both `//` and `/* */` (and `/** */`).  Attach
    # any leading run.
    doc_comment_node_types=frozenset({"comment"}),
    source_roots=("", "include", "src"),
    test_prefix="test_",
    language_plugin_version=3,
)
