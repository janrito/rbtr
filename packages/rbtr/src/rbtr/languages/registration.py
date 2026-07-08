"""Language plugin contract for the code index.

A language plugin provides `rbtr` with the ability to detect,
parse, and extract structural data from source files in a given
language.

Minimum implementation — a module-level `LanguageRegistration` value::

    from rbtr.languages.registration import LanguageRegistration

    kotlin = LanguageRegistration(
        id="kotlin",
        extensions=frozenset({".kt", ".kts"}),
    )

This gives file detection, line-based chunking, and text-search
import resolution.  Each additional field on `LanguageRegistration`
unlocks more precise analysis — see the class docstring.

For a worked example per language, see each plugin package's
golden-tested sample under its `tests/samples/` (one source file per
language, showing the constructs that language's plugin extracts).

Plugins register via the `rbtr.languages` entry-point group; the value
is the `LanguageRegistration`, named by its language id::

    [project.entry-points."rbtr.languages"]
    kotlin = "rbtr_lang_kotlin.plugin:kotlin"

Shared utilities for `import_extractor` implementations:

- `parse_path_relative` — parse `./`/`../` prefixes
- `collect_scoped_path` — collect nested `scoped_identifier`
  segments
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

from rbtr.index.models import Chunk, ImportMeta

if TYPE_CHECKING:
    from tree_sitter import Language, Node, Range


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


type ImportResolver = Callable[[Node, dict[str, list[Node]]], ImportMeta]
type ImportExtractor = Callable[[ImportResolver, Node, dict[str, list[Node]]], ImportMeta]
"""Type alias for the import-metadata extractor override.

Signature: `(resolver, node, captures) -> ImportMeta`, where *resolver* is
the built-in `default_import` (call it to delegate, then refine) — a
wrap-style callback. *node* is the `@import` AST node; *captures* is the
match dict from the tree-sitter query. Read captures first (fast path),
then walk the node for what the query couldn't capture (e.g. multi-valued
import names). Unset → the engine uses `default_import`, which reads
`@_import_module` / `@_import_names` and strips delimiters.
"""

type NameResolver = Callable[[str, Node, dict[str, list[Node]]], str]
type NameExtractor = Callable[[NameResolver, str, Node, dict[str, list[Node]]], str]
"""Type alias for the display-name resolver override.

Signature: `(resolver, capture_name, node, captures) -> str`, where
*resolver* is the built-in `default_name` (call it to delegate) — a
wrap-style callback. Override only as a last resort, when a query cannot
express the name (e.g. Bash strips the `=` the grammar fuses onto an
alias; HTML names an element by its `id`); delegate to *resolver* for the
captures you do not special-case. Unset → the engine uses `default_name`.
"""

type ScopeResolver = Callable[[str, Node, dict[str, list[Node]]], list[str]]
type ScopeExtractor = Callable[[ScopeResolver, str, Node, dict[str, list[Node]]], list[str]]
"""Type alias for the scope-address resolver override.

Signature: `(resolver, capture_name, node, captures) -> list[str]`, where
*resolver* is the built-in `default_scope` — a wrap-style callback.
Returns scope segments outermost-first (`[]` for none), which the engine
*appends* to any tree-ancestry scope (`scope_types`). Override only as a
last resort, for hierarchies a query and ancestry cannot express — e.g.
SCSS/Less nested rules (walk ancestor `rule_set` selectors) or TOML dotted
tables. Segments compose into a `::` address via the `Chunk.scope`
validator. Unset → the engine uses `default_scope` (the `@_scope` capture).
"""

type Chunker = Callable[[str, str, str, Language, list[Range] | None], Iterator[Chunk]]
"""Type alias for the optional custom-chunking callback.

Signature:
`(file_path, blob_sha, content, grammar, ranges) -> Iterator[Chunk]`.
The grammar is loaded by the manager and passed by the orchestrator.
`ranges` restricts parsing to those byte/point spans (via
`parser.included_ranges`) so the chunker can serve as an injection target
for an embedded block; `None` parses the whole file. A chunker MUST honour
`ranges` to be usable as a delegation target. Delegating embedded code to
other languages is the engine's job, via a registration's `injection_query`
— not the chunker's.
"""


# ── Registration dataclass ───────────────────────────────────────────

# Language ID must be lowercase ascii, digits, or underscores.
_VALID_ID = re.compile(r"[a-z][a-z0-9_]*")


@dataclass(frozen=True, kw_only=True)
class LanguageRegistration:
    """Describes a language plugin's capabilities.

    Only `id` is required.  Every other field has a sensible default —
    provide only what your language needs.  Each field is documented
    inline below, grouped as:

    - **Detection** — `id`, `extensions`, `filenames`.
    - **Grammar** — `grammar_module`, `grammar_entry`, `grammar_factory`.
    - **Extraction** — `query`, `scope_types`, `class_scope_types`,
      `doc_comment_node_types`, `injection_query`.
    - **Import resolution** (feeds the edge graph) — `index_files`,
      `import_targets`, `source_roots`, `path_substitutions`,
      `module_style`.
    - **Housekeeping** — `language_plugin_version`.

    Extraction overrides — a custom name/scope/import resolver or a
    chunker — are *not* constructor arguments; attach them with the
    decorator methods `name_extractor`, `scope_extractor`,
    `import_extractor`, and `chunker` (each also works as a plain call for
    a shared function).  See those methods and the `*Extractor` type
    aliases for signatures.

    Examples:

        Detection only (line-based chunking, text-search imports)::

            csv = LanguageRegistration(
                id="csv",
                extensions=frozenset({".csv"}),
            )

        Custom chunker without tree-sitter (prose, or a language with no
        grammar)::

            markdown = LanguageRegistration(
                id="markdown",
                extensions=frozenset({".md"}),
            )

            @markdown.chunker
            def chunk_markdown(file_path, blob_sha, content, grammar, ranges):
                ...

        Grammar but no query (parse tree available, no symbol extraction
        yet)::

            ruby = LanguageRegistration(
                id="ruby",
                extensions=frozenset({".rb"}),
                grammar_module="tree_sitter_ruby",
            )

        Full support with a query and an import override::

            python = LanguageRegistration(
                id="python",
                extensions=frozenset({".py", ".pyi"}),
                grammar_module="tree_sitter_python",
                query=load_query(__package__, "python"),
                scope_types=frozenset({"class_definition"}),
                module_style=ModuleStyle.DOTTED,
            )

            @python.import_extractor
            def _py_import(resolver, node, captures):
                ...
    """

    id: str
    """Unique language id, e.g. `"python"`.  Lowercase ascii, digits, or
    underscores (validated in `__post_init__`).  Stamped on every chunk as
    its `language`, and the name the entry point resolves to."""
    extensions: frozenset[str] = frozenset()
    """File extensions this plugin claims, leading dot included, e.g.
    `frozenset({".py", ".pyi"})`.  Matched against a file's suffix."""
    filenames: frozenset[str] = frozenset()
    """Exact base filenames claimed regardless of extension, e.g.
    `frozenset({"Makefile", "Dockerfile"})`."""
    grammar_module: str | None = None
    """Python module exposing a tree-sitter grammar factory, e.g.
    `"tree_sitter_python"`.  `None` → no parsing; the file gets line-based
    chunking only.  Paired with `grammar_entry`, or bypassed by
    `grammar_factory`."""
    grammar_entry: str = "language"
    """Name of the factory function on `grammar_module`, called as
    `Language(getattr(module, grammar_entry)())`.  Defaults to `"language"`;
    override for non-standard names, e.g. `"language_typescript"` in
    `tree_sitter_typescript`."""
    grammar_factory: Callable[[], Language] | None = None
    """Alternative to `grammar_module`: a callable returning a ready
    `Language`.  For bindings that hand one back directly (so the
    `Language(entry())` path would double-wrap), e.g.
    `tree-sitter-language-pack`'s `get_language(name)`.  Takes precedence
    over `grammar_module` when set."""
    query: str | None = None
    """Tree-sitter S-expression query for symbol extraction, loaded from a
    co-located `.scm` via `load_query` (never inline — house rule).  Capture
    names drive the chunk kind:

    - `@function` / `@_fn_name` — functions
    - `@class` / `@_cls_name` — classes, structs, enums, traits, types
    - `@method` / `@_method_name` — methods (a `@function` whose nearest
      scope is class-like is also promoted)
    - `@variable` / `@_var_name` — top-level data
    - `@import` — import statements (metadata via the import override)
    - `@_scope` — optional: a node whose *text* becomes the innermost scope
      segment, for scopes lexical nesting can't reach (e.g. a Go method's
      receiver type)

    `None` → no structural extraction even if a grammar is present.  Refine
    a `@_*_name` display name with a `name_extractor` only as a last
    resort."""
    scope_types: frozenset[str] = frozenset()
    """Tree-sitter node types that open a naming scope, composed into a
    symbol's `::` address — e.g. `{"class_definition"}` (Python),
    `{"impl_item", "struct_item"}` (Rust).  Include every nesting container
    that should appear in an address (classes, namespaces, modules, and
    functions where nested defs matter).  Empty for languages without
    scopes (e.g. Bash)."""
    class_scope_types: frozenset[str] = frozenset()
    """Subset of `scope_types` whose nodes are class-like — a function
    directly inside one is promoted to a method.  Defaults to `scope_types`
    when unset, so a language whose every scope is a class needs only
    `scope_types`.  A language that also puts non-class scopes (nested
    functions, namespaces, modules) in `scope_types` for addressing must
    set this to just its class-like types so method promotion stays
    correct."""
    injection_query: str | None = None
    """Tree-sitter query (against this host grammar) that delegates embedded
    code to other languages. Capture the embedded code as `@injection.content`
    and set its language with `(#set! injection.language "<id>")` — to a
    canonical rbtr language id — plus an optional
    `(#set! injection.priority "<n>")` (default 0) to disambiguate overlapping
    matches. The engine runs it after the primary extraction and delegates
    each captured range via `extract_query`. `None` means no embedded
    delegation."""
    doc_comment_node_types: frozenset[str] = frozenset()
    """AST node types treated as leading documentation for a captured symbol.
    Empty (default) → the chunk covers exactly the symbol's byte span, no
    comment attachment.  When non-empty, `extract_symbols` walks each
    symbol's `prev_named_sibling` chain, gathering consecutive nodes of
    these types up to a blank-line boundary, and extends the chunk to cover
    them (see `_collect_leading_doc_comments` in `treesitter.py`).  Left
    empty by languages whose docs are interior rather than leading (Python,
    captured via `@_docstring`)."""
    index_files: frozenset[str] = frozenset()
    """Directory entry-point filenames: an import of a *directory* resolves
    to one of these inside it — e.g. `{"__init__.py"}` (Python),
    `{"index.ts", "index.js"}` (JS/TS), `{"mod.rs"}` (Rust).  Empty → a
    directory import does not resolve to a file."""
    import_targets: frozenset[str] | None = None
    """Language ids this language's imports may resolve to, for cross-language
    edges.  The resolver looks up each target's `extensions` and
    `index_files` to find the imported file, so e.g. HTML sets
    `{"javascript", "typescript", "css"}` to link `<script src>` / `<link>`,
    and CSS sets `{"css"}` for `@import`.  `None` (default) → same-language
    only.  Each target's plugin must be installed for those edges to
    materialise."""
    source_roots: tuple[str, ...] = ("",)
    """Directory prefixes tried, in order, when resolving an absolute import
    — each is prepended to the module path before the file search.  `("",)`
    (default) resolves from the repo root; add roots for a configured source
    dir, e.g. `("", "src")`."""
    path_substitutions: tuple[tuple[str, str], ...] = ()
    """Ordered `(prefix, replacement)` rewrites applied to a module path
    before resolution (first matching prefix wins), for alias prefixes that
    don't map 1:1 to directories — e.g. `(("crate/", "src/"),)` so Rust's
    `crate::` resolves under `src/`.  Empty → no rewriting."""
    language_plugin_version: int = 1
    """Extractor version.  Bump on any extraction change (query, chunker,
    override, or scope config) — it triggers re-extraction of every blob
    stored at an older version.  A pure package *move* is not an extraction
    change; do not bump it then."""
    module_style: ModuleStyle = ModuleStyle.PATH
    """How import module strings map to file paths — `PATH` (slash-separated
    or bare) or `DOTTED` (dot-separated, e.g. Python/Java).  See
    `ModuleStyle`."""

    # Extraction overrides — attached via the decorator methods below, not the
    # constructor. `None` means "use the engine default" (the orchestrator
    # substitutes `default_name` / `default_scope` / `default_import`).
    _name_extractor: NameExtractor | None = field(
        default=None, init=False, compare=False, repr=False
    )
    _scope_extractor: ScopeExtractor | None = field(
        default=None, init=False, compare=False, repr=False
    )
    _import_extractor: ImportExtractor | None = field(
        default=None, init=False, compare=False, repr=False
    )
    _chunker: Chunker | None = field(default=None, init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not _VALID_ID.fullmatch(self.id):
            msg = (
                f"invalid language id {self.id!r} — must be lowercase ascii, digits, or underscores"
            )
            raise ValueError(msg)
        if not self.class_scope_types:
            object.__setattr__(self, "class_scope_types", self.scope_types)

    # ── Extraction overrides ─────────────────────────────────────────
    # Each works as a decorator (`@reg.name_extractor`) on a fresh function or
    # as a call (`reg.name_extractor(fn)`) for a shared/imported one.

    def name_extractor(self, fn: NameExtractor) -> NameExtractor:
        """Register a custom display-name resolver (see `NameExtractor`)."""
        object.__setattr__(self, "_name_extractor", fn)
        return fn

    def scope_extractor(self, fn: ScopeExtractor) -> ScopeExtractor:
        """Register a custom scope-address resolver (see `ScopeExtractor`)."""
        object.__setattr__(self, "_scope_extractor", fn)
        return fn

    def import_extractor(self, fn: ImportExtractor) -> ImportExtractor:
        """Register a custom import-metadata extractor (see `ImportExtractor`)."""
        object.__setattr__(self, "_import_extractor", fn)
        return fn

    def chunker(self, fn: Chunker) -> Chunker:
        """Register a custom chunker (see `Chunker`)."""
        object.__setattr__(self, "_chunker", fn)
        return fn

    # ── Resolution (called by the engine) ────────────────────────
    # Apply the override if set (passing the built-in as its wrap `resolver`),
    # else the built-in directly.

    def resolve_name(self, capture_name: str, node: Node, captures: dict[str, list[Node]]) -> str:
        fn = self._name_extractor
        return (
            fn(default_name, capture_name, node, captures)
            if fn
            else default_name(capture_name, node, captures)
        )

    def resolve_scope(
        self, capture_name: str, node: Node, captures: dict[str, list[Node]]
    ) -> list[str]:
        fn = self._scope_extractor
        return (
            fn(default_scope, capture_name, node, captures)
            if fn
            else default_scope(capture_name, node, captures)
        )

    def resolve_import(self, node: Node, captures: dict[str, list[Node]]) -> ImportMeta:
        fn = self._import_extractor
        return fn(default_import, node, captures) if fn else default_import(node, captures)

    def chunk(
        self,
        file_path: str,
        blob_sha: str,
        content: str,
        grammar: Language,
        ranges: list[Range] | None,
    ) -> list[Chunk] | None:
        """Run the custom chunker, or `None` if this language has none."""
        fn = self._chunker
        return list(fn(file_path, blob_sha, content, grammar, ranges)) if fn is not None else None


# ── Shared utilities for plugin authors ──────────────────────────────

# Maps import-specific capture keys to `ImportMeta` field names.
# Used by `default_import` to read captures into
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


def default_import(
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


NAME_CAPTURE_KEY: dict[str, str] = {
    "function": "_fn_name",
    "method": "_method_name",
    "class": "_cls_name",
    "variable": "_var_name",
    "doc_section": "_section_name",
}
"""Maps a structural capture name to its paired `@_*_name` helper capture."""


def default_name(
    capture_name: str,
    node: Node,
    captures: dict[str, list[Node]],
) -> str:
    """Resolve a captured node's display name from the query captures.

    Default `name_extractor`: for symbol captures (`@function`, `@class`, …)
    the name is the paired `@_fn_name` / `@_cls_name` helper capture; for
    `@import` it is the statement text; otherwise `"<anonymous>"`. Plugins
    that override `name_extractor` typically call this first, then refine.
    """
    name_key = NAME_CAPTURE_KEY.get(capture_name)
    name_nodes = captures.get(name_key, []) if name_key else []
    first_text = name_nodes[0].text if name_nodes else None
    if first_text:
        return first_text.decode()
    if capture_name == "import" and node.text:
        return node.text.decode().strip()[:120]
    return "<anonymous>"


def default_scope(
    _capture_name: str,
    node: Node,
    captures: dict[str, list[Node]],
) -> list[str]:
    """Default `scope_extractor`: the `@_scope` capture as a scope segment.

    Returns the text of an `@_scope` node clipped to *node*'s span as a
    one-element list, else `[]`. `@_scope` supplies a non-lexical scope that
    tree ancestry cannot reach — e.g. a Go method's receiver type, a child of
    the `method_declaration` rather than an enclosing node. Most languages
    carry no `@_scope`, so this is a no-op returning `[]`; the engine appends
    the result to the tree-ancestry scope. Overrides return the full scope for
    hierarchies ancestry cannot express.
    """
    for s in captures.get("_scope", []):
        if node.start_byte <= s.start_byte and s.end_byte <= node.end_byte and s.text:
            return [s.text.decode()]
    return []


def enclosing_nodes_of_type(node: Node, types: frozenset[str]) -> list[Node]:
    """Ancestor nodes of *node* whose type is in *types*, outermost-first.

    A building block for ancestry-based `scope_extractor`s (e.g. SCSS/Less
    nested rules), so they need not re-implement the parent walk. *node*
    itself is not included.
    """
    found: list[Node] = []
    parent = node.parent
    while parent is not None:
        if parent.type in types:
            found.append(parent)
        parent = parent.parent
    found.reverse()
    return found


def build_quoted_import(
    _resolver: ImportResolver,
    _node: Node,
    captures: dict[str, list[Node]],
) -> ImportMeta:
    """Build `ImportMeta` from a directly-captured quoted specifier.

    For grammars whose import string is a single node carrying its
    own quotes with no inner content child (SCSS/Less `string_value`),
    so the `@_import_module` capture is e.g. `"config"` or `'config'`.
    Strips one matching surrounding single or double quote and stores
    the result as `module`.
    """
    meta = ImportMeta()
    cap_nodes = captures.get("_import_module", [])
    if cap_nodes and cap_nodes[0].text:
        raw = cap_nodes[0].text.decode()
        if len(raw) >= 2 and raw[0] in "\"'" and raw[-1] == raw[0]:
            raw = raw[1:-1]
        meta.module = raw
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
