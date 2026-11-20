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

This module provides language registrations, resolver type aliases, query
loading and tree-derived capture helpers. Policy enums live in
`domain.edge_frames`; path-relative parsing lives in `domain.resolution`.
The engine's built-in resolvers live in the private `_resolvers` module.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from importlib.resources import files
from typing import TYPE_CHECKING

from rbtr.domain.edge_frames import ModuleStyle, SectionNames
from rbtr.domain.models import Chunk
from rbtr.languages._resolvers import DefaultName, DefaultScope

if TYPE_CHECKING:
    from tree_sitter import Language, Node, Range


type ManifestReader = Callable[[str], tuple[tuple[str, str], ...]]
"""Reads a manifest's text into `(prefix, replacement)` path substitutions.

A language whose imports carry a prefix the directory layout does not have
declares where that prefix is written down: Go's `go.mod` states `module
example.com/m`, so `example.com/m/b` is the directory `b`."""

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
    `@_var_name`, `@import`, `@_scope`), and `@_call` (a call's callee),
    `@_base` (a class's base), `@_self` and `@_super` (the receivers a path
    may start from) record what a chunk refers to. Load `query` from a co-located `.scm`
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
    fragments: bool = True
    """Whether code no capture covers becomes `fragment` chunks. `False` for
    a grammar whose text another language's chunks already hold: Markdown's
    inline half, whose text lies in Markdown's sections."""

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
      `module_style`, `prose`.
    - **Housekeeping** — `extraction_serial`.

    Extraction overrides — a custom name or scope resolver, or a chunker —
    are *not* constructor arguments; attach them with the decorator methods
    `name_extractor`, `scope_extractor` and `chunker` (each also works as a
    plain call for a shared function). An import is stated in the query
    alone.  See those methods and the `*Extractor` type
    aliases for signatures.

    Examples:

        Detection only (line-based chunking, text-search imports)::

            csv = LanguageRegistration(
                id="csv",
                extensions=frozenset({".csv"}),
            )

        Custom chunker without tree-sitter (prose, or a language with no
        grammar)::

            svelte = LanguageRegistration(
                id="svelte",
                extensions=frozenset({".svelte"}),
            )

            @svelte.chunker
            def chunk_sfc(file_path, blob_sha, content, grammar, ranges):
                ...

        Grammar but no query (parse tree available, no symbol extraction
        yet)::

            ruby = LanguageRegistration(
                id="ruby",
                extensions=frozenset({".rb"}),
                grammar_module="tree_sitter_ruby",
            )

        Full support with a query::

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
    content_signals: tuple[re.Pattern[str], ...] = ()
    """Patterns whose matches in a file's text suggest this language, for a
    file its path names no language for — Markdown's ATX headings, rST's
    directives.  The language whose signals match most often in the file's
    first `config.content_sniff_chars` characters is chosen
    (`LanguageManager.detect_language_by_content`).
    Empty → never chosen by content."""
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
    matches. The engine reads each captured range as that language, as
    another source of the same file. `None` means no embedded delegation."""
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
    parent_module: str | None = None
    """The word a module path starts with to climb one level from the
    importing module, as Rust's `super`: resolution reads each leading
    `super::` as one level up. An identifier. None for a language whose
    relative modules are written as `./` and `../`, or as a dotted
    module's leading dots."""
    current_module: str | None = None
    """The word a module path starts with to name the importing module
    itself, as Rust's `self`: `self::util` is the importing module's child
    `util`. An identifier. None for a language without one."""
    directory_module_files: frozenset[str] = frozenset()
    """The file names that are their directory's module, as Rust's
    `mod.rs`, `lib.rs` and `main.rs`: a module word read in one starts at
    its directory. Any other file is a module of its own inside its
    directory, its child modules in the directory named by its stem
    (`order.rs`'s in `order/`). Read only with `parent_module` or
    `current_module`."""
    package_directory: bool = False
    """Whether a directory is itself a unit of this language, so a reference
    naming one reaches every file inside it: a Go package is its directory and
    a Terraform module is a directory of `.tf` files, each file belonging to
    the whole.  Languages that name a directory's stand-in file instead say so
    with `index_files`; for the rest a directory is only a place, and a link to
    one reaches nothing."""
    manifest: str = ""
    """Repository file whose text supplies further `path_substitutions`, read
    per repository rather than declared here — Go's `go.mod` names the module
    prefix its own imports carry.  Empty → the language's substitutions are
    the same in every repository.  Paired with `manifest_reader`."""
    extraction_serial: int = 1
    """Extraction cache key: a manually-bumped stamp that invalidates a
    language's stored chunks.  Every chunk records its language's serial;
    the blob-dedup gate re-extracts a blob whenever a chunk's serial does
    not equal the registry's current one (an equality check — a lower
    value re-extracts too, not just a higher one).  Bump it on any change
    to *extraction output*: the query, chunker, scope config, or a
    name/scope/import override — an import-resolver change included, since
    a matching serial skips the blob and would otherwise leave its edges
    stale.  It is **independent of the package version**: bump it during
    development, before any release, when output changes; and do *not*
    bump it for a package-only release or a pure move that leaves output
    identical."""
    module_style: ModuleStyle = ModuleStyle.PATH
    """How import module strings map to file paths — `PATH` (slash-separated
    or bare) or `DOTTED` (dot-separated, e.g. Python/Java).  See
    `ModuleStyle`."""
    prose: bool = False
    """Whether the language is prose, which documents what it references
    rather than importing it: its references make `documents` edges, and one
    may name a symbol and no file (an rST `:func:` role), found by name
    anywhere."""
    section_names: SectionNames = SectionNames.EXACT
    """How a reference compares with the name of one of this language's
    sections. See `SectionNames`."""

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
    _manifest_reader: ManifestReader | None = field(
        default=None, init=False, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        if not _VALID_ID.fullmatch(self.id):
            msg = (
                f"invalid language id {self.id!r} — must be lowercase ascii, digits, or underscores"
            )
            raise ValueError(msg)

    # Extraction overrides
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

    def manifest_reader(self, fn: ManifestReader) -> ManifestReader:
        """Set the reader for this language's `manifest`."""
        object.__setattr__(self, "_manifest_reader", fn)
        return fn

    def substitutions_from_manifest(self, text: str) -> tuple[tuple[str, str], ...]:
        """Substitutions this repository's manifest adds, if any."""
        reader = self._manifest_reader
        return reader(text) if reader is not None else ()

    def chunker(self, fn: Chunker) -> Chunker:
        """Set a custom chunker as the extraction strategy (`ChunkExtraction`).

        Works as a decorator (`@reg.chunker`) or a call (`reg.chunker(fn)`, for
        a chunker shared across registrations).
        """
        object.__setattr__(self, "extraction", ChunkExtraction(fn))
        return fn

    # Resolution (called by the engine)
    # The slot holds the effective resolver (built-in default or a composed
    # override), so these just call it — no branching, no injection.

    def resolve_name(self, capture_name: str, node: Node, captures: dict[str, list[Node]]) -> str:
        return self._name_resolver(capture_name, node, captures)

    def resolve_scope(
        self, capture_name: str, node: Node, captures: dict[str, list[Node]]
    ) -> list[str]:
        return self._scope_resolver(capture_name, node, captures)


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
