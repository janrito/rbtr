"""Language plugin contract for the code index.

A language plugin provides `rbtr` with the ability to detect,
parse, and extract structural data from source files in a given
language.

Minimum implementation::

    from rbtr.languages.hookspec import LanguageRegistration, hookimpl

    class KotlinPlugin:
        @hookimpl
        def rbtr_register_languages(self):
            return [LanguageRegistration(
                id="kotlin",
                extensions=frozenset({".kt", ".kts"}),
            )]

This gives file detection, line-based chunking, and text-search
import resolution.  Each additional field on `LanguageRegistration`
unlocks more precise analysis — see the class docstring.

External plugins register via setuptools entry points::

    [project.entry-points."rbtr.languages"]
    kotlin = "rbtr_kotlin:KotlinPlugin"

Shared utilities for `import_extractor` implementations:

- `parse_path_relative` — parse `./`/`../` prefixes
- `collect_scoped_path` — collect nested `scoped_identifier`
  segments
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

import pluggy

from rbtr.index.models import Chunk, ImportMeta


class ModuleStyle(StrEnum):
    """How import module strings map to file paths.

    Languages declare this on `LanguageRegistration` to control
    how `edges.py` resolves module strings to repo files.

    `PATH`:
        Slash-separated or bare filenames.  The resolver tries
        the path as-is first (handles pre-extensioned filenames
        like `app.js` or `reset.css`), then appends extensions.
        Detects `./`/`../` prefixes and resolves relative to the
        importing file.  Used by JS, TS, CSS, HTML, C, Go, Rust,
        Ruby, and most other languages.

    `DOTTED`:
        Dot-separated module hierarchy.  The resolver converts
        `.` to `/` before searching.  Used by Python and Java.
    """

    PATH = "path"
    DOTTED = "dotted"


if TYPE_CHECKING:
    from tree_sitter import Language, Node

type ImportExtractor = Callable[[Node, dict[str, list[Node]]], ImportMeta] | None
"""Type alias for the import-metadata extractor callback.

Signature: `(node, captures) -> ImportMeta`.

*node* is the `@import` AST node. *captures* is the match
dict from the tree-sitter query, keyed by capture name.
Extractors read captures first (fast path from the query),
then walk the node for what the query couldn't capture
(e.g. multi-valued import names).

Languages that don't need custom extraction register
`build_import_from_captures` (exported from
`rbtr.languages.hookspec`), which reads `@_import_module`
and `@_import_names` from the captures and applies
delimiter stripping.
"""

type Chunker = Callable[[str, str, str, Language], Iterator[Chunk]] | None
"""Type alias for the optional custom-chunking callback.

Signature: `(file_path, blob_sha, content, grammar) -> Iterator[Chunk]`.
The grammar is loaded by the manager and passed by the orchestrator.
"""

PROJECT_NAME = "rbtr"

hookspec = pluggy.HookspecMarker(PROJECT_NAME)
hookimpl = pluggy.HookimplMarker(PROJECT_NAME)

# Re-export for plugin authors.
__all__ = [
    "LanguageHookspec",
    "LanguageRegistration",
    "ModuleStyle",
    "collect_scoped_path",
    "hookimpl",
    "parse_path_relative",
]


# ── Registration dataclass ───────────────────────────────────────────

# Language ID must be lowercase ascii, digits, or underscores.
_VALID_ID = re.compile(r"[a-z][a-z0-9_]*")


@dataclass(frozen=True, kw_only=True)
class LanguageRegistration:
    """Describes a language plugin's capabilities.

    Only `id` is required.  Every other field has a sensible
    default — provide only what your language needs.

    Examples:

        Detection only (line-based chunking, text-search imports)::

            LanguageRegistration(
                id="csv",
                extensions=frozenset({".csv"}),
            )

        Custom chunker without tree-sitter (prose, or languages
        without a grammar — regex-based extraction, indentation-
        aware splitting, etc.)::

            LanguageRegistration(
                id="markdown",
                extensions=frozenset({".md"}),
                chunker=chunk_markdown,
            )

        Grammar but no queries (tree-sitter parse tree available,
        but no symbol extraction yet)::

            LanguageRegistration(
                id="ruby",
                extensions=frozenset({".rb"}),
                grammar_module="tree_sitter_ruby",
            )

        Full support with queries and import extractor::

            LanguageRegistration(
                id="python",
                extensions=frozenset({".py", ".pyi"}),
                grammar_module="tree_sitter_python",
                query='(function_definition ...) @function',
                import_extractor=extract_python_import_meta,
                scope_types=frozenset({"class_definition"}),
            )

    Attributes:
        id:               Unique language identifier (e.g. `"python"`).
                          Must be lowercase, no spaces.
        extensions:       File extensions handled, including the leading
                          dot (e.g. `frozenset({".py", ".pyi"})`).
        filenames:        Exact filenames handled without extension
                          matching (e.g.
                          `frozenset({"Makefile", "Dockerfile"})`).
        grammar_module:   Python module exposing a grammar factory
                          function (e.g. `"tree_sitter_python"`).
                          `None` means no tree-sitter parsing — the
                          file will get line-based chunking only.
        grammar_entry:    Name of the grammar factory function on
                          *grammar_module*.  Defaults to `"language"`.
                          Override for packages with non-standard names,
                          e.g. `"language_typescript"` for
                          `tree_sitter_typescript`.
        query:            Tree-sitter S-expression query for symbol
                          extraction.  Must use these capture-name
                          conventions:

                          - `@function` / `@_fn_name` — functions
                          - `@class` / `@_cls_name` — classes/types
                          - `@method` / `@_method_name` — methods
                          - `@import` — import statements

                          `None` means no structural extraction even
                          if a grammar is available.
        import_extractor: Callable `(node, captures) -> ImportMeta`.
                          Receives the `@import` AST node and the
                          match dict from the tree-sitter query.
                          Read captures first (fast path), then
                          walk the node for what the query can't
                          capture.  Languages without custom
                          logic register `build_import_from_captures`
                          from `rbtr.languages.hookspec`.  `None` is
                          filled with that default at call time.
        scope_types:      Tree-sitter node types that create a naming
                          scope (used to detect methods inside classes).
                          Examples: `{"class_definition"}` for Python,
                          `{"class_declaration"}` for Java/JS,
                          `{"impl_item", "struct_item"}` for Rust.
                          Empty `frozenset()` for languages without
                          classes (e.g. Bash).
        chunker:          Custom chunking function with signature
                          `(file_path, blob_sha, content, grammar)
                          -> Iterator[Chunk]`.  The grammar is loaded by
                          the manager and passed by the orchestrator.
                          When set, the orchestrator calls this instead
                          of tree-sitter extraction or plaintext
                          fallback.  Used by formats whose structural
                          units can't be expressed as query captures
                          (heading hierarchy, composed block names,
                          etc.).  `None` uses the default strategy
                          (tree-sitter if grammar + query, plaintext
                          otherwise).

        doc_comment_node_types:
                          AST node types that count as comments
                          when attaching leading documentation to
                          a captured symbol.  Default empty means
                          *no leading-comment attachment* — the
                          chunk covers exactly the symbol node's
                          byte span.  When
                          non-empty, `extract_symbols` walks each
                          symbol's `prev_named_sibling` chain,
                          collecting consecutive nodes of these
                          types up to a blank-line boundary, and
                          extends the chunk's byte span to cover
                          them.  See `_collect_leading_doc_comments`
                          in `treesitter.py` for the walk algorithm.
    """

    id: str
    extensions: frozenset[str] = frozenset()
    filenames: frozenset[str] = frozenset()
    grammar_module: str | None = None
    grammar_entry: str = "language"
    query: str | None = None
    import_extractor: ImportExtractor = None
    scope_types: frozenset[str] = frozenset()
    chunker: Chunker = None
    doc_comment_node_types: frozenset[str] = frozenset()
    index_files: frozenset[str] = frozenset()
    """Directory entry-point files (e.g. `__init__.py`, `index.ts`, `mod.rs`)."""
    import_targets: frozenset[str] | None = None
    """Languages this language can import from. `None` = same language only."""
    source_roots: tuple[str, ...] = ("",)
    """Directories to prepend when resolving absolute imports."""
    path_substitutions: tuple[tuple[str, str], ...] = ()
    """Module path prefix replacements (e.g. `("crate/", "src/")` for Rust)."""
    test_prefix: str = ""
    """Test file name prefix (e.g. `test_` for Python)."""
    test_suffix: str = ""
    """Test file name suffix (e.g. `.test` for JS/TS, `_test` for Go)."""
    language_plugin_version: int = 1
    """Extractor version. Bump when the query, chunker, or any
    extraction logic changes — triggers re-extraction of all
    blobs stored at older versions."""
    module_style: ModuleStyle = ModuleStyle.PATH
    """How module strings map to file paths. See `ModuleStyle`."""

    def __post_init__(self) -> None:
        if not _VALID_ID.fullmatch(self.id):
            msg = (
                f"invalid language id {self.id!r} — must be lowercase ascii, digits, or underscores"
            )
            raise ValueError(msg)


# ── Hook specification ───────────────────────────────────────────────


class LanguageHookspec:
    """Plugin hook specifications for language support.

    A plugin class implements one or more of these methods and is
    registered with the plugin manager (built-in) or via the
    `rbtr.languages` entry-point group (external).

    Example plugin class::

        from rbtr.languages.hookspec import LanguageRegistration, hookimpl

        class RubyPlugin:
            @hookimpl
            def rbtr_register_languages(self):
                return [LanguageRegistration(
                    id="ruby",
                    extensions=frozenset({".rb"}),
                    grammar_module="tree_sitter_ruby",
                )]
    """

    @hookspec
    def rbtr_register_languages(self) -> list[LanguageRegistration]:
        """Register one or more languages.

        Called once at startup.  Return a list of
        `LanguageRegistration` instances — one per language.

        A single plugin may register multiple languages.  For
        example, the built-in `JavaScriptPlugin` registers both
        `"javascript"` and `"typescript"`.

        Example::

            @hookimpl
            def rbtr_register_languages(self):
                return [
                    LanguageRegistration(id="javascript", ...),
                    LanguageRegistration(id="typescript", ...),
                ]
        """
        return []  # hookspec — never called directly


# ── Shared utilities for plugin authors ──────────────────────────────

# Maps import-specific capture keys to `ImportMeta` field names.
# Used by `build_import_from_captures` to read captures into
# the metadata fields.  `_import_dots` is excluded because its
# semantics are language-specific (Python counts dots from
# `import_prefix`; other languages don't use it).
_IMPORT_CAPTURE_KEYS: dict[str, str] = {
    "_import_module": "module",
    "_import_names": "names",
}

# Node types whose captured text needs delimiter stripping.
# Maps node type → (open_delimiter, close_delimiter).
_DELIMITER_STRIP: dict[str, tuple[str, str]] = {
    "system_lib_string": ("<", ">"),
    "string_literal": ('"', '"'),
    "interpreted_string_literal": ('"', '"'),
    "string": ('"', '"'),
}


def build_import_from_captures(
    _node: Node,
    captures: dict[str, list[Node]],
) -> ImportMeta:
    """Build `ImportMeta` from query captures.

    Default extractor for languages whose queries capture all
    the import metadata they need.  Strips delimiters based on
    the capture node's type: `<>` from `system_lib_string`,
    `"` from `string_literal` / `interpreted_string_literal`
    / `string`.

    Languages with richer import structures (Python, JS/TS)
    provide their own extractor that reads captures first, then
    walks the node for what the query can't express.  Rust
    walks the node entirely because `scoped_identifier` paths
    need `collect_scoped_path` processing.

    Examples (C):

        `#include <stdio.h>`:
            captures={"_import_module": [node(type="system_lib_string", text="<stdio.h>")]}
            → module="stdio.h"

    Examples (Go):

        `import "fmt"`:
            captures={"_import_module": [node(type="interpreted_string_literal", text="\"fmt\"")]}
            → module="fmt"

    Examples (Java):

        `import java.util.List;`:
            captures={"_import_module": [node(type="scoped_identifier", text="java.util.List")]}
            → module="java.util.List"
    """
    meta = ImportMeta()
    for cap_key, field_name in _IMPORT_CAPTURE_KEYS.items():
        cap_nodes = captures.get(cap_key, [])
        if cap_nodes and cap_nodes[0].text:
            raw = cap_nodes[0].text.decode()
            node_type = cap_nodes[0].type
            if node_type in _DELIMITER_STRIP:
                open_d, close_d = _DELIMITER_STRIP[node_type]
                if raw.startswith(open_d) and raw.endswith(close_d):
                    raw = raw[len(open_d) : -len(close_d)]
            setattr(meta, field_name, raw)
    return meta


def parse_path_relative(specifier: str) -> tuple[int, str]:
    """Parse a path-relative module specifier into `(dots, cleaned)`.

    Strips `./` and `../` prefixes and returns a `dots` count
    using the unified depth convention (1 = current directory,
    2 = parent, etc.) plus the remaining module path.

    Use this in import extractors for languages with filesystem-
    relative import specifiers (JS, TS, CSS modules, etc.).

    Examples:

        `./models` → (1, "models")
        `../utils` → (2, "utils")
        `../../shared/helpers` → (3, "shared/helpers")
        `react` → (0, "react")
        `./styles.css` → (1, "styles.css")
        (1, "styles.css")
    """
    dots = 0
    rest = specifier
    if rest.startswith("./"):
        dots = 1
        rest = rest[2:]
    else:
        while rest.startswith("../"):
            dots += 1
            rest = rest[3:]
        if dots:
            dots += 1  # "../" = 2 (parent dir = 2 levels up from file)
    return dots, rest


def collect_scoped_path(node: Node) -> list[str]:
    """Recursively collect path segments from a `scoped_identifier`.

    Walks the tree-sitter `scoped_identifier` node and collects
    all identifier-like children into a flat list.  Handles Rust
    keywords (`crate`, `super`, `self`) and Java-style
    nested scoped identifiers.

    Use this in import extractors for languages with `::`-separated
    or `.`-separated scoped paths (Rust, Java, C#, Scala, etc.).

    Examples:

        `std::collections::HashMap` → `["std", "collections", "HashMap"]`

        `crate::models` → `["crate", "models"]`

        `super::utils` → `["super", "utils"]`
    """
    parts: list[str] = []
    for child in node.children:
        match child.type:
            case "scoped_identifier":
                parts.extend(collect_scoped_path(child))
            case "identifier" | "crate" | "super" | "self":
                if child.text:
                    parts.append(child.text.decode())
    return parts
