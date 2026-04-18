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
    strip_docstrings: bool = False,
) -> list[Chunk]:
    """Parse *content* and extract structural chunks.

    Parameters:
        file_path:        Repo-relative path for the chunk IDs.
        blob_sha:         Git blob SHA for dedup.
        content:          Raw file bytes.
        grammar:          Tree-sitter `Language` for the parser.
        query_str:        S-expression query (capture conventions:
                          `@function`/`@_fn_name`,
                          `@class`/`@_cls_name`,
                          `@method`/`@_method_name`,
                          `@import`, `@_docstring`).
        import_extractor: Optional callable to extract structured
                          `ImportMeta` from import AST nodes.
        scope_types:      Node types that define naming scopes
                          (e.g. `class_definition`).
        strip_docstrings: When true, blank out bytes covered by
                          any `@_docstring` sub-capture from the
                          chunk content (replaced with whitespace
                          that preserves newlines, so `line_start`
                          and `line_end` stay valid).  Plugins
                          without a `@_docstring` capture are
                          unaffected.
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

                text = node.text.decode() if node.text else ""
                if strip_docstrings and node.text:
                    doc_nodes = capture_dict.get(_DOCSTRING_CAPTURE, [])
                    if doc_nodes:
                        ranges = [(d.start_byte, d.end_byte) for d in doc_nodes]
                        text = _redact_ranges(node.text, node.start_byte, node.end_byte, ranges)
                chunks.append(
                    Chunk(
                        id=_chunk_id(file_path, name, node.start_point[0]),
                        blob_sha=blob_sha,
                        file_path=file_path,
                        kind=actual_kind,
                        name=name,
                        scope=scope,
                        content=text,
                        line_start=node.start_point[0] + 1,
                        line_end=node.end_point[0] + 1,
                        metadata=metadata,
                    )
                )

    return chunks
