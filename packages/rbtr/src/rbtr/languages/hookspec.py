"""Language plugin contract for the code index.

A language plugin provides `rbtr` with the ability to detect,
parse, and extract structural data from source files in a given
language.

Minimum implementation
~~~~~~~~~~~~~~~~~~~~~~

A plugin class needs exactly **one method**:

.. code-block:: python

    from rbtr.languages.hookspec import LanguageRegistration, hookimpl

    class KotlinPlugin:
        @hookimpl
        def rbtr_register_languages(self):
            return [LanguageRegistration(
                id="kotlin",
                extensions=frozenset({".kt", ".kts"}),
            )]

This gives file detection, line-based chunking, and text-search
import resolution.

Progressive capability
~~~~~~~~~~~~~~~~~~~~~~

Each additional field on `LanguageRegistration` unlocks more
precise analysis:

============================================= ===================================
Field                                          Capability
============================================= ===================================
`id` + `extensions`                        File detection + line-based chunks
`chunker`                                    Custom chunking (no grammar needed)
`grammar_module`                             Tree-sitter grammar loading
`query`                                      Structural symbol extraction
`import_extractor`                           Structural import metadata
`scope_types`                                Correct method-in-class scoping
============================================= ===================================

Here is a complete example showing every field:

.. code-block:: python

    from rbtr.languages.hookspec import LanguageRegistration, hookimpl
    from rbtr.index.models import ImportMeta

    def extract_kotlin_import(node):
        # Walk node.children to extract module/names...
        return ImportMeta(module="com.example", names="Foo")

    class KotlinPlugin:
        @hookimpl
        def rbtr_register_languages(self):
            return [LanguageRegistration(
                id="kotlin",
                extensions=frozenset({".kt", ".kts"}),
                grammar_module="tree_sitter_kotlin",
                query=(
                    '(function_declaration'
                    '  name: (simple_identifier) @_fn_name) @function\\n'
                    '(class_declaration'
                    '  name: (type_identifier) @_cls_name) @class\\n'
                    '(import_header) @import'
                ),
                import_extractor=extract_kotlin_import,
                scope_types=frozenset({"class_declaration"}),
            )]

External plugins
~~~~~~~~~~~~~~~~

Third-party packages register via setuptools entry points::

    [project.entry-points."rbtr.languages"]
    kotlin = "rbtr_kotlin:KotlinPlugin"

Built-in plugins are registered directly and take precedence over
external ones by default.

Shared utilities
~~~~~~~~~~~~~~~~

This module also exports helper functions that plugin authors can
use in their `import_extractor` implementations:

- `parse_path_relative` — parse `./`/`../` prefixes
- `collect_scoped_path` — collect nested `scoped_identifier`
  segments
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pluggy

from rbtr.index.models import Chunk, ImportMeta

if TYPE_CHECKING:
    from tree_sitter import Node

type ImportExtractor = Callable[[Node], ImportMeta] | None
"""Type alias for the optional import-metadata extractor callback."""

PROJECT_NAME = "rbtr"

hookspec = pluggy.HookspecMarker(PROJECT_NAME)
hookimpl = pluggy.HookimplMarker(PROJECT_NAME)

# Re-export for plugin authors.
__all__ = [
    "LanguageHookspec",
    "LanguageRegistration",
    "collect_scoped_path",
    "hookimpl",
    "parse_path_relative",
]


# ── Registration dataclass ───────────────────────────────────────────

# Language ID must be lowercase ascii, digits, or underscores.
_VALID_ID = re.compile(r"[a-z][a-z0-9_]*")

# Default node types that create scopes (classes).  Covers Python
# (class_definition) and most C-family languages (class_declaration).
DEFAULT_SCOPE_TYPES: frozenset[str] = frozenset(
    {
        "class_definition",
        "class_declaration",
    }
)


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
        import_extractor: Callable that takes a tree-sitter `Node`
                          (the `@import` capture) and returns an
                          `ImportMeta` dict.  Populate `module` for
                          the import path, `names` for imported
                          symbols (comma-separated), and `dots` for
                          relative import depth.  Return `{}` for
                          unrecognised nodes.  `None` falls back to
                          text-search import resolution in `edges.py`.
        scope_types:      Tree-sitter node types that create a naming
                          scope (used to detect methods inside classes).
                          Examples: `{"class_definition"}` for Python,
                          `{"class_declaration"}` for Java/JS,
                          `{"impl_item", "struct_item"}` for Rust.
                          Empty `frozenset()` for languages without
                          classes (e.g. Bash).
        chunker:          Custom chunking function with signature
                          `(file_path, blob_sha, content) -> list[Chunk]`.
                          When set, the orchestrator calls this instead
                          of tree-sitter extraction or plaintext
                          fallback.  Used by prose formats like
                          Markdown and RST for heading-hierarchy
                          chunking.  `None` uses the default
                          strategy (tree-sitter if grammar + query,
                          plaintext otherwise).
        pygments_lexer:   Pygments lexer name for syntax highlighting.
                          Defaults to `id`, which works for most
                          languages.  Override when the language ID
                          doesn't match the Pygments name (e.g.
                          `c_sharp` → `"csharp"`).

    Docstring stripping
    ~~~~~~~~~~~~~~~~~~~

    Plugins opt into `--strip-docstrings` support by adding a
    `@_docstring` sub-capture to their query — same convention
    as the existing `@_fn_name` / `@_cls_name` helpers.  When
    stripping is enabled, `extract_symbols` blanks the byte
    ranges covered by `@_docstring` (preserving newlines so
    line numbers stay valid).  Plugins without a `@_docstring`
    capture are unaffected by the flag.
    """

    id: str
    extensions: frozenset[str] = frozenset()
    filenames: frozenset[str] = frozenset()
    grammar_module: str | None = None
    grammar_entry: str = "language"
    query: str | None = None
    import_extractor: ImportExtractor = None
    scope_types: frozenset[str] = field(default=DEFAULT_SCOPE_TYPES)
    chunker: Callable[[str, str, str], list[Chunk]] | None = None
    pygments_lexer: str | None = None

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


def parse_path_relative(specifier: str) -> tuple[int, str]:
    """Parse a path-relative module specifier into `(dots, cleaned)`.

    Strips `./` and `../` prefixes and returns a `dots` count
    using the unified depth convention (1 = current directory,
    2 = parent, etc.) plus the remaining module path.

    Use this in import extractors for languages with filesystem-
    relative import specifiers (JS, TS, CSS modules, etc.).

    Examples::

        >>> parse_path_relative("./models")
        (1, "models")
        >>> parse_path_relative("../utils")
        (2, "utils")
        >>> parse_path_relative("../../shared/helpers")
        (3, "shared/helpers")
        >>> parse_path_relative("react")
        (0, "react")
        >>> parse_path_relative("./styles.css")
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

        Given the Rust AST for `std::collections::HashMap`::

            scoped_identifier
              scoped_identifier
                identifier "std"
                identifier "collections"
              identifier "HashMap"

        Returns `["std", "collections", "HashMap"]`.

        Given `crate::models`::

            scoped_identifier
              crate "crate"
              identifier "models"

        Returns `["crate", "models"]`.

        Given `super::utils`::

            scoped_identifier
              super "super"
              identifier "utils"

        Returns `["super", "utils"]`.
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
