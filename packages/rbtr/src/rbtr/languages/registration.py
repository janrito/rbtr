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

This module is the plugin-authoring surface: everything a plugin imports to
declare itself — `LanguageRegistration`, `ModuleStyle`, the resolver type
aliases, `load_query`, and the capture/import helpers (`parse_path_relative`,
`collect_scoped_path`, `enclosing_nodes_of_type`, `build_quoted_import`). The
engine's built-in resolvers live in the private `_resolvers` module.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from importlib.resources import files
from typing import TYPE_CHECKING

from rbtr.index.models import Chunk, ImportMeta
from rbtr.languages._resolvers import DefaultImport, DefaultName, DefaultScope

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
"""Import-metadata resolver and its wrap-style override.

`ImportResolver` — `(node, captures) -> ImportMeta` — is the resolver the
engine calls and the shape of the built-in default. `ImportExtractor` is what
`@reg.import_extractor` accepts: `(resolver, node, captures) -> ImportMeta`,
with *resolver* the current resolver handed in so the override can delegate
(read captures first, then walk the node — Python/JS multi-name imports, Rust
scoped paths). The decorator composes the override over the built-in.
"""

type NameResolver = Callable[[str, Node, dict[str, list[Node]]], str]
type NameExtractor = Callable[[NameResolver, str, Node, dict[str, list[Node]]], str]
"""Display-name resolver and its wrap-style override.

`NameResolver` — `(capture_name, node, captures) -> str` — is the resolver the
engine calls. `NameExtractor` is what `@reg.name_extractor` accepts:
`(resolver, capture_name, node, captures) -> str`, with *resolver* handed in to
delegate. Override only as a last resort, when a query cannot express the name
(Bash strips the `=` the grammar fuses onto an alias; HTML names an element by
its `id`).
"""

type ScopeResolver = Callable[[str, Node, dict[str, list[Node]]], list[str]]
type ScopeExtractor = Callable[[ScopeResolver, str, Node, dict[str, list[Node]]], list[str]]
"""Scope-address resolver and its wrap-style override.

`ScopeResolver` — `(capture_name, node, captures) -> list[str]` (segments
outermost-first, `[]` for none) — is the resolver the engine calls; its result
is *appended* to the tree-ancestry scope. `ScopeExtractor` is what
`@reg.scope_extractor` accepts: `(resolver, capture_name, node, captures) ->
list[str]`. Override for hierarchies a query and ancestry cannot express
(SCSS/Less nested rules, TOML dotted tables).
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


@dataclass(frozen=True, kw_only=True)
class QueryExtraction:
    """Query-based extraction: a tree-sitter query plus scope settings.

    Capture names in `query` drive the chunk kind (`@function` / `@_fn_name`,
    `@class` / `@_cls_name`, `@method` / `@_method_name`, `@variable` /
    `@_var_name`, `@import`, `@_scope`). Load `query` from a co-located `.scm`
    via `load_query` (never inline). For a name/scope/import a query cannot
    express, attach an override on the `LanguageRegistration`.
    """

    query: str
    """Tree-sitter S-expression query for symbol extraction."""
    scope_types: frozenset[str] = frozenset()
    """Node types that open a naming scope, composed into a symbol's `::`
    address — e.g. `{"class_definition"}` (Python), `{"impl_item",
    "struct_item"}` (Rust). Empty for languages without scopes."""
    class_scope_types: frozenset[str] = frozenset()
    """Subset of `scope_types` that is class-like — a function directly inside
    one is promoted to a method. Defaults to `scope_types` when unset."""

    def __post_init__(self) -> None:
        if not self.class_scope_types:
            object.__setattr__(self, "class_scope_types", self.scope_types)


@dataclass(frozen=True)
class ChunkExtraction:
    """Chunker-based extraction: a custom chunker replaces query extraction.

    The chunker MUST honour `ranges` (parse only those spans) to be usable as
    an injection-delegation target. Attach it via the `chunker` decorator/call
    on the `LanguageRegistration`, or pass this value object directly.
    """

    chunker: Chunker


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
    - **Extraction** — `extraction` (a `QueryExtraction` or `ChunkExtraction`,
      or `None`), `injection_query`.
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
                extraction=QueryExtraction(
                    query=load_query(__package__, "python"),
                    scope_types=frozenset({"class_definition"}),
                ),
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
    `frozenset({".bashrc", ".zshrc"})`."""
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
    extraction: QueryExtraction | ChunkExtraction | None = None
    """Primary extraction strategy: a `QueryExtraction` (tree-sitter query +
    scope config), a `ChunkExtraction` (custom chunker), or `None` (line-based
    chunking). Pass either value object directly, or set a chunker via the
    `chunker` decorator/call. Orthogonal to `injection_query`."""
    injection_query: str | None = None
    """Tree-sitter query (against this host grammar) that delegates embedded
    code to other languages. Capture the embedded code as `@injection.content`
    and set its language with `(#set! injection.language "<id>")` — to a
    canonical rbtr language id — plus an optional
    `(#set! injection.priority "<n>")` (default 0) to disambiguate overlapping
    matches. The engine runs it after the primary extraction and delegates
    each captured range via `extract_query`. `None` means no embedded
    delegation."""
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

    # Extraction overrides — attached via the decorator methods below. Each slot
    # holds the *effective* resolver: a built-in default (a `_Default*`
    # null-object) until an override is composed over it. `resolve_*` just calls
    # the slot. (The chunker is the `extraction` field, not a resolver slot.)
    _name_resolver: NameResolver = field(
        default_factory=DefaultName, init=False, compare=False, repr=False
    )
    _scope_resolver: ScopeResolver = field(
        default_factory=DefaultScope, init=False, compare=False, repr=False
    )
    _import_resolver: ImportResolver = field(
        default_factory=DefaultImport, init=False, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        if not _VALID_ID.fullmatch(self.id):
            msg = (
                f"invalid language id {self.id!r} — must be lowercase ascii, digits, or underscores"
            )
            raise ValueError(msg)

    # ── Extraction overrides ─────────────────────────────────────────
    # Wrap-style: an override receives the current resolver as its first
    # argument, so it can delegate. Composed over the slot via `partial`, so the
    # slot always holds a plain resolver. Works as a decorator
    # (`@reg.name_extractor`) or a call (`reg.name_extractor(fn)`).

    def name_extractor(self, fn: NameExtractor) -> NameExtractor:
        """Compose a custom display-name resolver over the current one."""
        object.__setattr__(self, "_name_resolver", functools.partial(fn, self._name_resolver))
        return fn

    def scope_extractor(self, fn: ScopeExtractor) -> ScopeExtractor:
        """Compose a custom scope-address resolver over the current one."""
        object.__setattr__(self, "_scope_resolver", functools.partial(fn, self._scope_resolver))
        return fn

    def import_extractor(self, fn: ImportExtractor) -> ImportExtractor:
        """Compose a custom import-metadata resolver over the current one."""
        object.__setattr__(self, "_import_resolver", functools.partial(fn, self._import_resolver))
        return fn

    def chunker(self, fn: Chunker) -> Chunker:
        """Set a custom chunker as the extraction strategy (`ChunkExtraction`).

        Works as a decorator (`@reg.chunker`) or a call (`reg.chunker(fn)`, for
        a chunker shared across registrations).
        """
        object.__setattr__(self, "extraction", ChunkExtraction(fn))
        return fn

    # ── Resolution (called by the engine) ────────────────────────
    # The slot holds the effective resolver (built-in default or a composed
    # override), so these just call it — no branching, no injection.

    def resolve_name(self, capture_name: str, node: Node, captures: dict[str, list[Node]]) -> str:
        return self._name_resolver(capture_name, node, captures)

    def resolve_scope(
        self, capture_name: str, node: Node, captures: dict[str, list[Node]]
    ) -> list[str]:
        return self._scope_resolver(capture_name, node, captures)

    def resolve_import(self, node: Node, captures: dict[str, list[Node]]) -> ImportMeta:
        return self._import_resolver(node, captures)


# ── Shared utilities for plugin authors ──────────────────────────────


@functools.cache
def load_query(package: str, name: str) -> str:
    """Return the text of a `<name>.scm` query file in *package*.

    Query text lives in `<name>.scm` beside the plugin, loaded at import time
    (never inlined as a Python string — house rule). *package* is normally the
    caller's own `__package__`; *name* omits the `.scm` extension (e.g.
    `"python"`, `"injections"`). See ARCHITECTURE "Where queries live".

    Cached on `(package, name)`: a query shared by two registrations (e.g. the
    SFC injection query used by svelte and vue) is read once, so authors can
    inline the call in each registration rather than hoisting a module
    constant.
    """
    return (files(package) / f"{name}.scm").read_text(encoding="utf-8")


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
