"""Tree-sitter parsing and symbol extraction.

Extracts structural chunks (functions, classes, methods, imports) from
source files using tree-sitter queries.  Language-specific behaviour
(queries, import extractors, scope types) is provided by plugins via
the `LanguageManager`.

This module contains only language-agnostic extraction logic.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterator
from functools import lru_cache
from typing import TYPE_CHECKING

from tree_sitter import Language, Parser, Query, QueryCursor

from rbtr.index.identity import compose_scope
from rbtr.index.models import Chunk, ChunkKind, ImportMeta
from rbtr.languages.hookspec import build_import_from_captures

if TYPE_CHECKING:
    from tree_sitter import Node

# ── Constants ────────────────────────────────────────────────────────

# Capture name → ChunkKind mapping (excludes _ prefixed helper captures).
_CAPTURE_KIND: dict[str, ChunkKind] = {
    "function": ChunkKind.FUNCTION,
    "method": ChunkKind.METHOD,
    "class": ChunkKind.CLASS,
    "variable": ChunkKind.VARIABLE,
    "import": ChunkKind.IMPORT,
    "doc_section": ChunkKind.DOC_SECTION,
}

# Maps structural capture names to their paired name-capture key.
# Queries must use these capture names by convention.
_NAME_CAPTURE_KEY: dict[str, str] = {
    "function": "_fn_name",
    "method": "_method_name",
    "class": "_cls_name",
    "variable": "_var_name",
    "doc_section": "_section_name",
}

# Helper-capture key: byte ranges covering a symbol's documentation.
# Used by `_doc_ranges_for_symbol` and `extract_doc_spans` to locate
# interior docstrings.  Plugins opt in by adding a `@_docstring`
# sub-capture to their query.
_DOCSTRING_CAPTURE = "_docstring"


# ── Internal helpers ─────────────────────────────────────────────────


@lru_cache(maxsize=32)
def _get_query(grammar: Language, query_str: str) -> Query:
    """Compile and cache a tree-sitter query.

    `Query()` compilation is expensive (~0.6 ms each).  Caching
    avoids recompiling the same query for every file of the same
    language — saves ~0.9 s on a 1 500-file Python repo.
    """
    return Query(grammar, query_str)


def _resolve_name(
    capture_name: str,
    node: Node,
    capture_dict: dict[str, list[Node]],
) -> str:
    """Resolve the display name for a captured node.

    For symbol captures (`@function`, `@class`, etc.) the name
    comes from the paired `@_fn_name` / `@_cls_name` helper
    capture.  For `@import` captures the name is the full
    statement text (e.g. `import os`).  Falls back to
    `"<anonymous>"` when no name is found.
    """
    name_key = _NAME_CAPTURE_KEY.get(capture_name)
    name_nodes = capture_dict.get(name_key, []) if name_key else []
    first_text = name_nodes[0].text if name_nodes else None
    if first_text:
        return first_text.decode()

    if capture_name == "import" and node.text:
        return node.text.decode().strip()[:120]

    return "<anonymous>"


def _resolve_kind(
    kind: ChunkKind,
    *,
    nearest_scope_is_class: bool,
) -> ChunkKind:
    """Promote `FUNCTION` to `METHOD` inside a class-like scope.

    Tree-sitter grammars use `function_definition` /
    `function_declaration` for both free functions and methods —
    the distinction is structural.  A function is a method only
    when its *nearest* enclosing naming scope is class-like (a
    class/struct/impl, per the language's `class_scope_types`); a
    function nested in another function stays a function.
    """
    if kind == ChunkKind.FUNCTION and nearest_scope_is_class:
        return ChunkKind.METHOD
    return kind


def _chunk_line_start(
    node: Node,
    content: bytes,
    *,
    is_import: bool,
    doc_comment_node_types: frozenset[str],
) -> tuple[int, int]:
    """Return the `(start_byte, line_start)` for a chunk.

    For symbol captures (`@function`, `@class`, `@method`),
    extends the span backward over any attached leading doc
    comments.  For `@import` captures, uses the node's own
    span — imports are not a documented surface.
    """
    start_byte = node.start_byte
    line = node.start_point[0] + 1
    if not is_import and doc_comment_node_types:
        attached = _collect_leading_doc_comments(node, doc_comment_node_types, content)
        if attached:
            start_byte = attached[0].start_byte
            line = attached[0].start_point[0] + 1
    return start_byte, line


def _scope_name(node: Node) -> str:
    """Name of a scope-bearing node via its `name` (or Rust `type`) field."""
    name_node = node.child_by_field_name("name")
    if name_node is None:
        name_node = node.child_by_field_name("type")
    if name_node and name_node.text:
        return name_node.text.decode()
    return ""


def _enclosing_scopes(node: Node, scope_types: frozenset[str]) -> list[Node]:
    """Ancestor nodes that open a naming scope, nearest first."""
    scopes: list[Node] = []
    current = node.parent
    while current is not None:
        if current.type in scope_types:
            scopes.append(current)
        current = current.parent
    return scopes


def _enclosing_scope_names(
    node: Node,
    scope_types: frozenset[str],
    class_scope_types: frozenset[str],
) -> tuple[list[str], bool]:
    """Return *(scope_names, nearest_scope_is_class)* for a symbol node.

    *scope_names* are the enclosing naming scopes outermost-first
    (e.g. `["Outer", "Inner"]`, `["Svc", "start"]`) — discovery only,
    not composed; the `Chunk.scope` validator forms the `::` address
    (via `compose_scope`) and drops empty/anonymous segments.  The
    boolean reports whether the *nearest* enclosing scope is
    class-like, which drives function→method promotion in
    `_resolve_kind`.
    """
    scopes = _enclosing_scopes(node, scope_types)
    names = [_scope_name(s) for s in reversed(scopes)]
    nearest_is_class = bool(scopes) and scopes[0].type in class_scope_types
    return names, nearest_is_class


def _doc_ranges_for_symbol(
    node: Node,
    content: bytes,
    capture_dict: dict[str, list[Node]],
    doc_comment_node_types: frozenset[str],
) -> list[tuple[int, int]]:
    """Collect absolute byte ranges covering a symbol's documentation.

    Merges two sources:

    1. Interior `@_docstring` captures from the same match,
       clipped to *node*'s own byte span (a single match can in
       principle carry more than one `@function` capture; each
       symbol owns only the docstring captures that fall inside
       its own bytes).
    2. Leading-comment siblings attached by the sibling walk
       (exterior-comment languages — Rust, Go, JS, …).  Skipped
       when `doc_comment_node_types` is empty.

    Returns sorted, possibly-empty byte ranges.  Used by
    `extract_doc_spans` to recover raw docstring text.
    """
    ranges: list[tuple[int, int]] = []
    for d in capture_dict.get(_DOCSTRING_CAPTURE, []):
        if node.start_byte <= d.start_byte and d.end_byte <= node.end_byte:
            ranges.append((d.start_byte, d.end_byte))
    if doc_comment_node_types:
        for c in _collect_leading_doc_comments(node, doc_comment_node_types, content):
            ranges.append((c.start_byte, c.end_byte))
    ranges.sort()
    return ranges


def _collect_leading_doc_comments(
    node: Node,
    comment_types: frozenset[str],
    source: bytes,
) -> list[Node]:
    """Walk back over *node*'s previous named siblings, collecting comments.

    Tree-sitter queries can't cleanly express "leading doc comment
    block" with blank-line separation, so this post-extraction walk
    is how rbtr attaches doc comments to their symbol.  The walk
    stops at the first non-comment sibling, or when a blank line
    separates a comment from the next node — that boundary is what
    distinguishes a symbol's own doc block from an unrelated
    preceding comment such as a license header or a comment that
    belongs to the *previous* symbol.

    **Blank-line detection.**  Different grammars disagree on
    whether a line-comment node includes its trailing newline:
    tree-sitter-go does not (`// A` ends at byte 4, the `\n`
    follows at 4..5) but tree-sitter-rust does (`// A\n` is
    part of the `line_comment` span).  To treat both
    consistently we count newlines over
    `[content_end, next_start)` — where `content_end` is
    `prev.end_byte` with a trailing `\n` excluded if present.
    A blank line corresponds to exactly 2 newlines in that window
    (one ends `prev`, one is the empty line).

    Parameters:
        node:          The captured symbol node.
        comment_types: AST node types that count as comments for
                       this language (plugin-provided via
                       `LanguageRegistration.doc_comment_node_types`).
                       Empty → empty result.
        source:        The full file bytes, for blank-line detection
                       in the inter-node gap.

    Returns:
        Comment nodes in source order (earliest first).  Empty list
        when *comment_types* is empty or no attached comments exist.
    """
    if not comment_types:
        return []
    collected: list[Node] = []
    prev = node.prev_named_sibling
    next_start = node.start_byte
    while prev is not None and prev.type in comment_types:
        content_end = prev.end_byte
        if content_end > prev.start_byte and source[content_end - 1 : content_end] == b"\n":
            content_end -= 1
        separator = source[content_end:next_start]
        if separator.count(b"\n") >= 2:
            break
        collected.append(prev)
        next_start = prev.start_byte
        prev = prev.prev_named_sibling
    collected.reverse()
    return collected


# ── Public API ───────────────────────────────────────────────────────


def extract_symbols(
    file_path: str,
    blob_sha: str,
    content: bytes,
    grammar: Language,
    query_str: str,
    *,
    language: str = "",
    import_extractor: Callable[
        [Node, dict[str, list[Node]]], ImportMeta
    ] = build_import_from_captures,
    scope_types: frozenset[str] = frozenset(),
    class_scope_types: frozenset[str] = frozenset(),
    doc_comment_node_types: frozenset[str] = frozenset(),
) -> Iterator[Chunk]:
    """Parse *content* and yield structural chunks.

    Runs a tree-sitter query over the parsed AST and yields one
    `Chunk` per match.  Each chunk's kind, name, scope, and
    metadata are derived from the query's capture conventions:

    - `@function` / `@_fn_name` → `ChunkKind.FUNCTION`
    - `@class` / `@_cls_name` → `ChunkKind.CLASS`
    - `@method` / `@_method_name` → `ChunkKind.METHOD`
    - `@import` / `@_import_module` → `ChunkKind.IMPORT`
    - `@doc_section` / `@_section_name` → `ChunkKind.DOC_SECTION`

    Import metadata is built by the *import_extractor* callback,
    which receives the `@import` node and the capture dict from
    the match.  The default (`build_import_from_captures`) reads
    `@_import_module`, `@_import_names`, and `@_import_dots` from
    the captures and applies delimiter stripping.  Languages
    with richer import structures (Python, JS/TS, Rust) provide
    their own extractor that reads captures first, then walks
    the node for what the query can't express.

    Functions inside a class scope are promoted to methods
    (see `_resolve_kind`).

    Parameters:
        file_path:              Repo-relative path for chunk IDs.
        blob_sha:               Git blob SHA for dedup.
        content:                Raw file bytes.
        grammar:                Tree-sitter `Language`.
        query_str:              S-expression query string.
        import_extractor:       Callback that builds `ImportMeta`
                                from an `@import` node and its
                                capture dict.  Defaults to
                                `build_import_from_captures`.
        scope_types:            Node types that define naming
                                scopes, used to compose the full
                                enclosing-scope address (classes,
                                functions, namespaces, modules…).
        class_scope_types:      Subset of *scope_types* that is
                                class-like; a function whose nearest
                                enclosing scope is one of these is a
                                method (see `_resolve_kind`).
        doc_comment_node_types: Comment node types for leading-doc
                                attachment.  When non-empty, each
                                symbol's span extends backward over
                                any attached comments.
    """
    if not content:
        return

    parser = Parser(grammar)
    tree = parser.parse(content)
    query = _get_query(grammar, query_str)
    matches = QueryCursor(query).matches(tree.root_node)

    for _pattern_idx, capture_dict in matches:
        for capture_name, nodes in capture_dict.items():
            if capture_name.startswith("_"):
                continue
            kind = _CAPTURE_KIND.get(capture_name)
            if kind is None:
                continue

            for node in nodes:
                name = _resolve_name(capture_name, node, capture_dict)

                meta = ImportMeta()
                if capture_name == "import":
                    meta = import_extractor(node, capture_dict)

                scope_names, nearest_is_class = _enclosing_scope_names(
                    node, scope_types, class_scope_types
                )
                actual_kind = _resolve_kind(kind, nearest_scope_is_class=nearest_is_class)

                start_byte, line_start = _chunk_line_start(
                    node,
                    content,
                    is_import=capture_name == "import",
                    doc_comment_node_types=doc_comment_node_types,
                )

                text = content[start_byte : node.end_byte].decode("utf-8", errors="replace")

                # `scope` is given as segment names; the `Chunk` scope
                # validator composes them and the id is derived there.
                # model_validate keeps that input type-honest.
                yield Chunk.model_validate(
                    {
                        "blob_sha": blob_sha,
                        "file_path": file_path,
                        "kind": actual_kind,
                        "name": name,
                        "scope": scope_names,
                        "language": language,
                        "content": text,
                        "line_start": line_start,
                        "line_end": node.end_point[0] + 1,
                        "metadata": meta,
                    }
                )


@dataclasses.dataclass(frozen=True)
class DocSpan:
    """Absolute byte ranges of one symbol's documentation.

    Returned by `extract_doc_spans`.  Consumers that need the raw
    docstring text decode `source[start:end]` for each range in
    `ranges` and join them.

    `start_byte` / `end_byte` delimit the whole symbol (body +
    doc comments) so a caller can recover the body by subtracting
    the doc ranges from `[start_byte, end_byte]`.
    """

    name: str
    kind: ChunkKind
    scope: str
    line_start: int
    line_end: int
    ranges: list[tuple[int, int]]
    start_byte: int
    end_byte: int


def extract_doc_spans(
    content: bytes,
    grammar: Language,
    query_str: str,
    *,
    scope_types: frozenset[str] = frozenset(),
    class_scope_types: frozenset[str] = frozenset(),
    doc_comment_node_types: frozenset[str] = frozenset(),
) -> Iterator[DocSpan]:
    """Yield `DocSpan` records for every documented symbol.

    Parses *content* once and collects doc byte ranges via interior
    `@_docstring` captures (clipped to the enclosing symbol) plus
    any leading-comment siblings attached by the language plugin.
    Symbols without any doc bytes are omitted.

    Line numbers are aligned with `extract_symbols`: when doc
    comments are attached exterior to the symbol, `line_start`
    reflects the first attached comment, not the symbol node
    itself.

    Meant for tooling — the benchmark's query sampler decodes the
    returned ranges to recover each symbol's raw docstring.  Not
    used by the indexer itself.
    """
    if not content:
        return

    parser = Parser(grammar)
    tree = parser.parse(content)
    query = _get_query(grammar, query_str)
    matches = QueryCursor(query).matches(tree.root_node)

    for _pattern_idx, capture_dict in matches:
        for capture_name, nodes in capture_dict.items():
            if capture_name not in _NAME_CAPTURE_KEY:
                continue
            name_key = _NAME_CAPTURE_KEY[capture_name]

            for node in nodes:
                doc_ranges = _doc_ranges_for_symbol(
                    node,
                    content,
                    capture_dict,
                    doc_comment_node_types,
                )
                if not doc_ranges:
                    continue

                name_nodes = capture_dict.get(name_key, [])
                if not name_nodes or not name_nodes[0].text:
                    continue

                name = name_nodes[0].text.decode()
                scope_names, nearest_is_class = _enclosing_scope_names(
                    node, scope_types, class_scope_types
                )
                scope = compose_scope(scope_names)
                kind = _resolve_kind(
                    _CAPTURE_KIND[capture_name], nearest_scope_is_class=nearest_is_class
                )

                _, line_start = _chunk_line_start(
                    node,
                    content,
                    is_import=False,
                    doc_comment_node_types=doc_comment_node_types,
                )

                yield DocSpan(
                    name=name,
                    kind=kind,
                    scope=scope,
                    line_start=line_start,
                    line_end=node.end_point[0] + 1,
                    ranges=doc_ranges,
                    start_byte=node.start_byte,
                    end_byte=node.end_byte,
                )
