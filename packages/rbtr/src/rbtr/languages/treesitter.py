"""Tree-sitter parsing and symbol extraction.

Extracts structural chunks (functions, classes, methods, imports) from
source files using tree-sitter queries. Language-specific behaviour
(query, scope types, and name/scope/import resolution) is supplied by
the `LanguageRegistration` passed to `extract_symbols`, which delegates
resolution back to it.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from functools import lru_cache
from typing import TYPE_CHECKING

from tree_sitter import Language, Parser, Query, QueryCursor, Range

from rbtr.index.identity import compose_scope
from rbtr.index.models import Chunk, ChunkKind, ImportMeta
from rbtr.languages._resolvers import NAME_CAPTURE_KEY
from rbtr.languages.registration import QueryExtraction

if TYPE_CHECKING:
    from tree_sitter import Node

    from rbtr.languages.registration import LanguageRegistration

# ── Constants ────────────────────────────────────────────────────────

# The chunk kinds produced by a structural query capture. The capture name
# equals the `ChunkKind` value (a `StrEnum`), so `ChunkKind(capture_name)` is
# the reverse map; this set is the whitelist that keeps non-capture kinds
# (`RAW_CHUNK`, `MIGRATION`, …) out.
_CAPTURE_KINDS: frozenset[ChunkKind] = frozenset(
    {
        ChunkKind.FUNCTION,
        ChunkKind.METHOD,
        ChunkKind.CLASS,
        ChunkKind.VARIABLE,
        ChunkKind.IMPORT,
        ChunkKind.DOC_SECTION,
        ChunkKind.CONFIG_KEY,
        ChunkKind.COMMENT,
    }
)

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


def _contiguous(source: bytes, a_end: int, b_start: int) -> bool:
    """Whether `[a_end, b_start)` is only whitespace with no blank line.

    The sole primitive behind comment grouping (comment → comment) and
    leading-doc folding (block → symbol): two nodes belong together when no
    code and no empty line separate them. A trailing newline some grammars
    fold into the earlier node is discounted (rust's `line_comment` spans its
    `\n`; go's does not), so a blank line is ≥ 2 newlines in the gap.
    """
    if source[a_end:b_start].strip() != b"":
        return False
    end = a_end - 1 if a_end > 0 and source[a_end - 1 : a_end] == b"\n" else a_end
    return source.count(b"\n", end, b_start) < 2


def _starts_line(source: bytes, start: int) -> bool:
    """Whether only whitespace precedes *start* on its own line.

    A comment that trails code (`x = 1  # note`) returns False: it documents
    the preceding statement, not whatever follows, so it never folds into the
    next symbol.
    """
    line_begin = source.rfind(b"\n", 0, start) + 1
    return source[line_begin:start].strip(b" \t") == b""


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


def _scope_names(scopes: list[Node]) -> list[str]:
    """Names of enclosing scopes, outermost-first, for the `::` address.

    E.g. `["Outer", "Inner"]`, `["Svc", "start"]` — discovery only, not
    composed; the `Chunk.scope` validator forms the `::` address (via
    `compose_scope`) and drops empty/anonymous segments.
    """
    return [_scope_name(s) for s in reversed(scopes)]


def _nearest_scope_is_class(scopes: list[Node], class_scope_types: frozenset[str]) -> bool:
    """Whether the *nearest* enclosing scope is class-like.

    Drives function→method promotion in `_resolve_kind`. *scopes* is
    nearest-first (as returned by `_enclosing_scopes`).
    """
    return bool(scopes) and scopes[0].type in class_scope_types


def _doc_ranges_for_symbol(
    node: Node,
    content: bytes,
    capture_dict: dict[str, list[Node]],
    comment_nodes: list[Node],
) -> list[tuple[int, int]]:
    """Collect absolute byte ranges covering a symbol's documentation.

    Merges two sources:

    1. Interior `@_docstring` captures from the same match,
       clipped to *node*'s own byte span.
    2. The `@comment` block sitting flush before *node* (its leading
       docs), from *comment_nodes*.

    Returns sorted, possibly-empty byte ranges.  Used by
    `extract_doc_spans` to recover raw docstring text.
    """
    ranges: list[tuple[int, int]] = []
    for d in capture_dict.get(_DOCSTRING_CAPTURE, []):
        if node.start_byte <= d.start_byte and d.end_byte <= node.end_byte:
            ranges.append((d.start_byte, d.end_byte))
    for c in _leading_comment_block(node, comment_nodes, content):
        ranges.append((c.start_byte, c.end_byte))
    ranges.sort()
    return ranges


def _leading_comment_block(node: Node, comments: list[Node], source: bytes) -> list[Node]:
    """The contiguous `@comment` run ending flush before *node* (its docs).

    Scans source-ordered *comments* for the maximal run that ends flush
    before *node* — only whitespace and no blank line between the run and the
    node. Empty when a blank line or code separates them.
    """
    block: list[Node] = []
    for c in comments:
        if c.start_byte >= node.start_byte:
            break
        if block and not _contiguous(source, block[-1].end_byte, c.start_byte):
            block = []
        block.append(c)
    if block and _contiguous(source, block[-1].end_byte, node.start_byte):
        return block
    return []


# ── Public API ───────────────────────────────────────────────────────


def extract_symbols(
    registered_language: LanguageRegistration,
    file_path: str,
    blob_sha: str,
    content: bytes,
    grammar: Language,
    *,
    included_ranges: list[Range] | None = None,
) -> Iterator[Chunk]:
    """Parse *content* and yield structural chunks, in source order.

    Runs a tree-sitter query over the parsed AST; each capture's kind, name,
    scope, and metadata are derived from the query's capture conventions:

    - `@function` / `@_fn_name` → `ChunkKind.FUNCTION`
    - `@class` / `@_cls_name` → `ChunkKind.CLASS`
    - `@method` / `@_method_name` → `ChunkKind.METHOD`
    - `@import` / `@_import_module` → `ChunkKind.IMPORT`
    - `@doc_section` / `@_section_name` → `ChunkKind.DOC_SECTION`
    - `@config_key` / `@_section_name` → `ChunkKind.CONFIG_KEY`
    - `@comment` → `ChunkKind.COMMENT`

    Import metadata is built by `registered_language.resolve_import`.
    Functions inside a class scope are promoted to methods (`_resolve_kind`).

    **Comments.** Captured `@comment` nodes are grouped into blocks (a
    maximal `_contiguous` run) and routed: a block flush before a symbol
    folds into it (its docstring, possibly nested); a block inside a symbol's
    body is dropped (already carried by that chunk); any other block is a
    standalone `COMMENT` chunk. Imports are not a documented surface, so a
    block before one stays standalone.

    Parameters:
        registered_language:  The language's registration — the
                              `QueryExtraction`, name/scope/import resolution,
                              and the id stamped on each chunk.
        file_path:            Repo-relative path for chunk IDs.
        blob_sha:             Git blob SHA for dedup.
        content:              Raw file bytes.
        grammar:              Tree-sitter `Language`.
        included_ranges:      Byte/point ranges to restrict parsing to (e.g. a
                              `<script>` block inside an SFC); `None` parses
                              the whole content.
    """
    extraction = registered_language.extraction
    if not content or not isinstance(extraction, QueryExtraction):
        return

    parser = Parser(grammar)
    if included_ranges is not None:
        parser.included_ranges = included_ranges
    tree = parser.parse(content)
    query = _get_query(grammar, extraction.query)

    # Flatten chunk-producing captures into one source-ordered stream.
    items: list[tuple[Node, str, dict[str, list[Node]]]] = []
    for _pattern_idx, capture_dict in QueryCursor(query).matches(tree.root_node):
        for capture_name, nodes in capture_dict.items():
            if capture_name.startswith("_"):
                continue
            try:
                if ChunkKind(capture_name) not in _CAPTURE_KINDS:
                    continue
            except ValueError:
                continue
            for node in nodes:
                items.append((node, capture_name, capture_dict))
    items.sort(key=lambda it: it[0].start_byte)

    seen: list[tuple[int, int]] = []  # emitted symbol node spans (for interior comments)
    pending: list[Node] = []  # the comment block being accumulated

    def resolve(block: list[Node]) -> Chunk | None:
        """A comment block that folded nowhere: drop if interior, else standalone."""
        if any(s <= block[0].start_byte and block[-1].end_byte <= e for s, e in seen):
            return None
        end_byte = block[-1].end_byte
        line_end = block[-1].end_point[0] + 1
        if end_byte > block[-1].start_byte and content[end_byte - 1 : end_byte] == b"\n":
            end_byte -= 1
            line_end -= 1
        return Chunk.model_validate(
            {
                "blob_sha": blob_sha,
                "file_path": file_path,
                "kind": ChunkKind.COMMENT,
                "name": "<anonymous>",
                "scope": [],
                "language": registered_language.id,
                "content": content[block[0].start_byte : end_byte].decode(
                    "utf-8", errors="replace"
                ),
                "line_start": block[0].start_point[0] + 1,
                "line_end": line_end,
                "metadata": ImportMeta(),
            }
        )

    for node, capture_name, capture_dict in items:
        if capture_name == "comment":
            if not _starts_line(content, node.start_byte):
                # A comment trailing code documents the preceding statement,
                # not what follows: flush any block and stand it alone (or
                # drop if interior). It never folds into the next symbol.
                if pending and (chunk := resolve(pending)) is not None:
                    yield chunk
                pending = []
                if (chunk := resolve([node])) is not None:
                    yield chunk
                continue
            if pending and not _contiguous(content, pending[-1].end_byte, node.start_byte):
                if (chunk := resolve(pending)) is not None:
                    yield chunk
                pending = []
            pending.append(node)
            continue

        # A symbol capture. A pending block flush before it (and it is not an
        # import) folds in, extending the chunk's start; otherwise resolve it.
        start_byte = node.start_byte
        line_start = node.start_point[0] + 1
        if pending:
            if capture_name != "import" and _contiguous(
                content, pending[-1].end_byte, node.start_byte
            ):
                start_byte = pending[0].start_byte
                line_start = pending[0].start_point[0] + 1
            elif (chunk := resolve(pending)) is not None:
                yield chunk
            pending = []

        meta = ImportMeta()
        if capture_name == "import":
            meta = registered_language.resolve_import(node, capture_dict)
        scopes = _enclosing_scopes(node, extraction.scope_types)
        # The scope override's segments extend the tree-ancestry scope; the
        # default contributes the `@_scope` capture.
        scope_names = [
            *_scope_names(scopes),
            *registered_language.resolve_scope(capture_name, node, capture_dict),
        ]
        actual_kind = _resolve_kind(
            ChunkKind(capture_name),
            nearest_scope_is_class=_nearest_scope_is_class(scopes, extraction.class_scope_types),
        )
        seen.append((node.start_byte, node.end_byte))
        yield Chunk.model_validate(
            {
                "blob_sha": blob_sha,
                "file_path": file_path,
                "kind": actual_kind,
                "name": registered_language.resolve_name(capture_name, node, capture_dict),
                "scope": scope_names,
                "language": registered_language.id,
                "content": content[start_byte : node.end_byte].decode("utf-8", errors="replace"),
                "line_start": line_start,
                "line_end": node.end_point[0] + 1,
                "metadata": meta,
            }
        )

    if pending and (chunk := resolve(pending)) is not None:
        yield chunk


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
) -> Iterator[DocSpan]:
    """Yield `DocSpan` records for every documented symbol.

    Parses *content* once and collects doc byte ranges via interior
    `@_docstring` captures (clipped to the enclosing symbol) plus the
    `@comment` block sitting flush before each symbol. Symbols without any
    doc bytes are omitted.

    Line numbers reflect the leading comment when present, not the symbol
    node itself.

    Meant for tooling — the benchmark's query sampler decodes the
    returned ranges to recover each symbol's raw docstring.  Not
    used by the indexer itself.
    """
    if not content:
        return

    parser = Parser(grammar)
    tree = parser.parse(content)
    query = _get_query(grammar, query_str)
    matches = list(QueryCursor(query).matches(tree.root_node))

    comment_nodes = sorted(
        (n for _p, caps in matches for n in caps.get("comment", [])),
        key=lambda n: n.start_byte,
    )

    for _pattern_idx, capture_dict in matches:
        for capture_name, nodes in capture_dict.items():
            if capture_name not in NAME_CAPTURE_KEY:
                continue
            name_key = NAME_CAPTURE_KEY[capture_name]

            for node in nodes:
                doc_ranges = _doc_ranges_for_symbol(
                    node,
                    content,
                    capture_dict,
                    comment_nodes,
                )
                if not doc_ranges:
                    continue

                name_nodes = capture_dict.get(name_key, [])
                if not name_nodes or not name_nodes[0].text:
                    continue

                name = name_nodes[0].text.decode()
                scopes = _enclosing_scopes(node, scope_types)
                scope_names = _scope_names(scopes)
                nearest_is_class = _nearest_scope_is_class(scopes, class_scope_types)
                scope = compose_scope(scope_names)
                kind = _resolve_kind(
                    ChunkKind(capture_name), nearest_scope_is_class=nearest_is_class
                )

                lead = _leading_comment_block(node, comment_nodes, content)
                line_start = (lead[0] if lead else node).start_point[0] + 1

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
