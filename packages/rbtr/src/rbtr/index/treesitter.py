"""Tree-sitter parsing and symbol extraction.

Extracts structural chunks (functions, classes, methods, imports) from
source files using tree-sitter queries.  Language-specific behaviour
(queries, import extractors, scope types) is provided by plugins via
the `LanguageManager`.

This module contains only language-agnostic extraction logic.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from functools import lru_cache
from typing import TYPE_CHECKING

from tree_sitter import Language, Parser, Query, QueryCursor

from rbtr.index.models import Chunk, ChunkKind, ImportMeta
from rbtr.languages.hookspec import DEFAULT_SCOPE_TYPES

if TYPE_CHECKING:
    from tree_sitter import Node

# ── Constants ────────────────────────────────────────────────────────

# Capture name → ChunkKind mapping (excludes _ prefixed helper captures).
_CAPTURE_KIND: dict[str, ChunkKind] = {
    "function": ChunkKind.FUNCTION,
    "method": ChunkKind.METHOD,
    "class": ChunkKind.CLASS,
    "import": ChunkKind.IMPORT,
}

# Maps structural capture names to their paired name-capture key.
# Queries must use these capture names by convention.
_NAME_CAPTURE_KEY: dict[str, str] = {
    "function": "_fn_name",
    "method": "_method_name",
    "class": "_cls_name",
}

# Helper-capture key: byte ranges to redact from chunk content when
# `strip_docstrings=True`.  Plugins opt in by adding a `@_docstring`
# sub-capture to their query.  Optional — languages without the
# capture are simply unaffected by the flag.
_DOCSTRING_CAPTURE = "_docstring"

# Node types whose child identifiers name a scope.
_SCOPE_NAME_TYPES: frozenset[str] = frozenset(
    {
        "constant",
        "identifier",
        "type_identifier",
    }
)


@lru_cache(maxsize=32)
def _get_query(grammar: Language, query_str: str) -> Query:
    """Compile and cache a tree-sitter query.

    `Query()` compilation is expensive (~0.6 ms each).  Caching
    avoids recompiling the same query for every file of the same
    language — saves ~0.9 s on a 1 500-file Python repo.
    """
    return Query(grammar, query_str)


def _chunk_id(file_path: str, name: str, line_start: int) -> str:
    """Deterministic chunk ID from file path, symbol name, and line."""
    raw = f"{file_path}:{name}:{line_start}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _redact_ranges(
    node_bytes: bytes,
    node_start: int,
    node_end: int,
    ranges: list[tuple[int, int]],
) -> str:
    """Blank out *ranges* inside *node_bytes*.

    *ranges* are absolute `(start_byte, end_byte)` pairs in the
    source file; each is clipped to `[node_start, node_end)` and
    applied to a mutable copy of *node_bytes*.  Newline bytes
    inside the redacted region are preserved so `line_start` and
    `line_end` on the resulting chunk stay valid; every other
    byte is overwritten with an ASCII space.
    """
    buf = bytearray(node_bytes)
    for start, stop in ranges:
        s = max(start, node_start) - node_start
        e = min(stop, node_end) - node_start
        if s >= e:
            continue
        for i in range(s, e):
            if buf[i] != 0x0A:  # newline
                buf[i] = 0x20  # space
    return buf.decode("utf-8", errors="replace")


def _find_scope(node: Node, scope_types: frozenset[str]) -> str:
    """Walk up the tree to find the enclosing scope name."""
    current = node.parent
    while current is not None:
        if current.type in scope_types:
            for child in current.children:
                if child.type in _SCOPE_NAME_TYPES and child.text:
                    return child.text.decode()
        current = current.parent
    return ""


def _collect_leading_doc_comments(
    node: Node,
    comment_types: frozenset[str],
    source: bytes,
) -> list[Node]:
    """Walk back over *node*'s previous named siblings, collecting comments.

    Tree-sitter queries can't cleanly express "leading doc comment
    block" with blank-line separation, so this post-extraction walk
    is how rbtr attaches doc comments to their symbol.  The walk
    stops at the first non-comment sibling, or when the gap between
    a comment and the next node spans a blank line (two or more
    newlines) — that boundary is what distinguishes a symbol's
    own doc block from an unrelated preceding comment such as a
    license header or a comment that belongs to the *previous*
    symbol.

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
        gap = source[prev.end_byte : next_start]
        if gap.count(b"\n") >= 2:
            # Blank line between this comment and the next node —
            # attachment stops here.
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
    import_extractor: Callable[[Node], ImportMeta] | None = None,
    scope_types: frozenset[str] = DEFAULT_SCOPE_TYPES,
    doc_comment_node_types: frozenset[str] = frozenset(),
    strip_docstrings: bool = False,
) -> list[Chunk]:
    """Parse *content* and extract structural chunks.

    Parameters:
        file_path:              Repo-relative path for chunk IDs.
        blob_sha:               Git blob SHA for dedup.
        content:                Raw file bytes.
        grammar:                Tree-sitter `Language`.
        query_str:              S-expression query.  Capture
                                conventions:
                                `@function` / `@_fn_name`,
                                `@class` / `@_cls_name`,
                                `@method` / `@_method_name`,
                                `@import`, `@_docstring`.
        import_extractor:       Optional callable for structured
                                `ImportMeta`.
        scope_types:            Node types that define naming
                                scopes (e.g. `class_definition`).
        doc_comment_node_types: Plugin-declared comment node
                                types for leading-doc attachment.
                                Non-empty means `extract_symbols`
                                walks each function/class/method
                                node's previous named siblings
                                (stopping at a blank line) and
                                extends the chunk span to cover
                                any attached comments.  The
                                attached comments' byte ranges are
                                implicitly added to the redaction
                                set used by *strip_docstrings*, so
                                no `@_docstring` capture is needed
                                for exterior languages.
        strip_docstrings:       When true, blank out bytes covered
                                by `@_docstring` captures **and**
                                by attached leading comments.
                                Newlines are preserved so line
                                numbers stay valid.

    Imports (`@import` captures) are always chunked at the
    statement's own bytes — leading-comment attachment applies
    only to symbol-kind captures (`@function`, `@class`,
    `@method`).
    """
    if not content:
        return []

    parser = Parser(grammar)
    tree = parser.parse(content)
    query = _get_query(grammar, query_str)
    cursor = QueryCursor(query)

    # matches() returns list[(pattern_idx, dict[capture_name, list[Node]])].
    matches = cursor.matches(tree.root_node)

    chunks: list[Chunk] = []
    for _pattern_idx, capture_dict in matches:
        for capture_name, nodes in capture_dict.items():
            if capture_name.startswith("_"):
                continue

            kind = _CAPTURE_KIND.get(capture_name)
            if kind is None:
                continue

            for node in nodes:
                # Get the paired name capture from the same match.
                name_key = _NAME_CAPTURE_KEY.get(capture_name)
                name_nodes = capture_dict.get(name_key, []) if name_key else []
                first_text = name_nodes[0].text if name_nodes else None
                name = first_text.decode() if first_text else ""

                # For imports, use the full statement text and
                # extract structured metadata via the plugin.
                metadata: ImportMeta = {}
                if capture_name == "import":
                    if not name and node.text:
                        name = node.text.decode().strip()[:120]
                    if import_extractor is not None:
                        metadata = import_extractor(node)

                if not name:
                    name = "<anonymous>"

                scope = _find_scope(node, scope_types)

                # Methods are functions inside a class.
                actual_kind = kind
                if kind == ChunkKind.FUNCTION and scope:
                    actual_kind = ChunkKind.METHOD

                # Determine the chunk's byte span and starting
                # line, extending backward over attached leading
                # comments for non-import symbols.  Imports keep
                # their own span — they're not a documented
                # surface.
                chunk_start = node.start_byte
                chunk_line = node.start_point[0] + 1
                attached_ranges: list[tuple[int, int]] = []
                if capture_name != "import" and doc_comment_node_types:
                    attached = _collect_leading_doc_comments(node, doc_comment_node_types, content)
                    if attached:
                        chunk_start = attached[0].start_byte
                        chunk_line = attached[0].start_point[0] + 1
                        attached_ranges = [(c.start_byte, c.end_byte) for c in attached]

                chunk_bytes = content[chunk_start : node.end_byte]
                text = chunk_bytes.decode("utf-8", errors="replace")

                if strip_docstrings and chunk_bytes:
                    # Merge interior `@_docstring` captures (from
                    # this match only) with attached leading
                    # comments into a single redaction set.
                    doc_nodes = capture_dict.get(_DOCSTRING_CAPTURE, [])
                    ranges = list(attached_ranges)
                    ranges.extend((d.start_byte, d.end_byte) for d in doc_nodes)
                    if ranges:
                        text = _redact_ranges(chunk_bytes, chunk_start, node.end_byte, ranges)

                chunks.append(
                    Chunk(
                        id=_chunk_id(file_path, name, chunk_line - 1),
                        blob_sha=blob_sha,
                        file_path=file_path,
                        kind=actual_kind,
                        name=name,
                        scope=scope,
                        content=text,
                        line_start=chunk_line,
                        line_end=node.end_point[0] + 1,
                        metadata=metadata,
                    )
                )

    return chunks
