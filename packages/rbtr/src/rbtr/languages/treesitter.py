"""Tree-sitter parsing and symbol extraction.

Extracts structural chunks (functions, classes, methods, imports) from
source files using tree-sitter queries. Language-specific behaviour
(query, scope types, and name/scope/import resolution) is supplied by
the `LanguageRegistration` passed to `extract_symbols`, which delegates
resolution back to it.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Iterator, Mapping
from functools import lru_cache
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Literal

from tree_sitter import Language, Parser, Query, QueryCursor, Range

from rbtr.config import config
from rbtr.domain.identity import compose_scope
from rbtr.domain.models import (
    Chunk,
    ChunkKind,
    Comment,
    Definition,
    Fragment,
    FragmentMeta,
    Import,
    ImportMeta,
    RawSlice,
    Role,
    Section,
    Span,
)
from rbtr.languages._resolvers import NAME_CAPTURE_KEY
from rbtr.languages.chunks import last_line
from rbtr.languages.manager import get_manager
from rbtr.languages.registration import QueryExtraction

if TYPE_CHECKING:
    from tree_sitter import Node, Tree

    from rbtr.languages.registration import LanguageRegistration

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


def _scope_names(scopes: list[Node], named: Mapping[int, str]) -> list[str]:
    """Names of enclosing scopes, outermost-first, for the `::` address.

    A scope the query captured is named as its capture names it (a
    Markdown `section` by its heading); any other by its `name` field. E.g.
    `["Outer", "Inner"]`, `["Svc", "start"]` — discovery only, not
    composed; the `Chunk.scope` validator forms the `::` address (via
    `compose_scope`) and drops empty/anonymous segments. *named* maps a
    captured node's id to its name.
    """
    return [named.get(s.id) or _scope_name(s) for s in reversed(scopes)]


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


def extract_symbols(
    registered_language: LanguageRegistration,
    file_path: str,
    blob_sha: str,
    content: bytes,
    grammar: Language,
    *,
    included_ranges: list[Range] | None = None,
) -> Iterator[Chunk]:
    """Parse *content* and yield the query's chunks in source order, then fragments.

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
    standalone `COMMENT` chunk. A comment trailing code on its line documents
    that statement, so it never folds into what follows. Imports are not a
    documented surface, so a block before one stays standalone.

    **Fragments.** After the captured chunks come `FRAGMENT` chunks for
    the code they leave (`_fragments`), so every line of the parsed
    range reaches a chunk.

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
    directives = _Directives(registered_language, _get_query(tree.language, extraction.query))
    items = _capture_items(tree, directives)
    # A node captured more than once (`a, b = …`) names a scope by its first capture.
    named: dict[int, str] = {}
    for item in items:
        if item.capture != "comment":
            named.setdefault(item.node.id, directives.name(item))
    sections = {item.node.id for item in items if item.capture == "doc_section"}
    flat = _flat_sections(items, directives, content, tree.root_node.end_byte)
    captured = list(
        _captured(
            registered_language,
            file_path,
            blob_sha,
            content,
            items,
            directives,
            named,
            sections,
            flat,
            extraction,
        )
    )
    yield from captured
    yield from _fragments(
        registered_language, file_path, blob_sha, content, tree, extraction, captured, named
    )


def _flat_sections(
    items: list[_Capture], directives: _Directives, content: bytes, end: int
) -> dict[int, tuple[int, int, list[str]]]:
    """Each flat section's end byte, 1-based last line and scope, by node id.

    A flat section is one whose match captures `@_level`: its grammar gives
    it no body to nest in (tree-sitter-rst's `section` holds its title
    alone). Its level is the order in which its style (its `@_level` texts,
    `#strip!` applied, joined) first appears; it is scoped under the nearest
    earlier flat section of a higher level, and holds its text up to the
    next flat section, or to *end*, less trailing whitespace.
    """
    flat = [item for item in items if item.capture == "doc_section" and "_level" in item.captures]
    ranks: dict[str, int] = {}
    above: list[tuple[int, str]] = []
    spans: dict[int, tuple[int, int, list[str]]] = {}
    for at, item in enumerate(flat):
        style = "".join(
            directives.strip(item.pattern, "_level", node.text.decode())
            for node in item.captures["_level"]
            if node.text
        )
        rank = ranks.setdefault(style, len(ranks))
        while above and above[-1][0] >= rank:
            above.pop()
        scope = [name for _, name in above]
        above.append((rank, directives.name(item)))
        stop = flat[at + 1].node.start_byte if at + 1 < len(flat) else end
        while stop > item.node.start_byte and content[stop - 1 : stop] in (
            b" ",
            b"\t",
            b"\r",
            b"\n",
        ):
            stop -= 1
        spans[item.node.id] = (stop, content.count(b"\n", 0, stop) + 1, scope)
    return spans


def _section_end(node: Node, sections: set[int], content: bytes) -> tuple[int, int]:
    """The end byte and 1-based last line of section *node*'s own text.

    When the section ends in captured sections (one of *sections*), nothing
    else after the first of them, its text is what comes before that one: a
    Markdown `# Title`'s chunk holds its text up to `## Install`, which has
    a chunk of its own. Any other section keeps its node's span: an HTML
    element's own content continues past a nested one, to its end tag
    at least. Trailing whitespace is not the section's.
    """
    first = next((i for i, c in enumerate(node.children) if c.id in sections), None)
    end = node.end_byte
    if first is not None and all(c.id in sections for c in node.children[first:]):
        end = node.children[first].start_byte
    while end > node.start_byte and content[end - 1 : end] in (b" ", b"\t", b"\r", b"\n"):
        end -= 1
    return end, content.count(b"\n", 0, end) + 1


def _fragment_named(meta: ImportMeta, file_path: str) -> ImportMeta:
    """*meta* with a URL fragment in its module read as the name it names.

    `page.html#intro` names `intro` in `page.html`; a fragment alone,
    `#intro`, names it in *file_path* itself.
    """
    if "#" not in meta.module:
        return meta
    module, _, fragment = meta.module.partition("#")
    return meta.model_copy(
        update={
            "module": module or PurePosixPath(file_path).name,
            "names": meta.names or fragment,
        }
    )


@dataclasses.dataclass(frozen=True)
class _Capture:
    """A chunk-producing capture: its *node*, its *capture* name, and the match it is in.

    *pattern* is the index of the query pattern that matched; *captures* is
    the match's every capture, helpers included.
    """

    pattern: int
    node: Node
    capture: str
    captures: dict[str, list[Node]]


@dataclasses.dataclass
class _Directives:
    """A query's directives, and the values a capture gives under them.

    tree-sitter leaves directives to its bindings, and py-tree-sitter applies
    `#set!` only (`Query.pattern_settings`). `#strip! @capture "regex"`
    removes what the regex matches from the capture's text (tree-sitter's
    query documentation, "Predicates and Directives"); the engine applies it
    to the name and import values read from a capture.
    """

    language: LanguageRegistration
    query: Query
    strips: dict[int, dict[str, re.Pattern[str]]] = dataclasses.field(default_factory=dict)

    def record(
        self,
        predicate: str,
        args: list[tuple[str, Literal["capture", "string"]]],
        pattern_index: int,
        captures: dict[str, list[Node]],  # unused; `QueryPredicate` passes it
    ) -> bool:
        """Note a `#strip!` while matching; every match is kept (`QueryCursor.matches`)."""
        if predicate == "strip!":
            (capture, _), (regex, _) = args
            self.strips.setdefault(pattern_index, {})[capture] = re.compile(regex)
        return True

    def strip(self, pattern: int, capture: str, text: str) -> str:
        """*text* with *pattern*'s `#strip!` for *capture* applied."""
        regex = self.strips.get(pattern, {}).get(capture)
        return regex.sub("", text) if regex else text

    def name(self, item: _Capture) -> str:
        """The name *item* gives, its name capture's `#strip!` applied."""
        name = self.language.resolve_name(item.capture, item.node, item.captures)
        key = NAME_CAPTURE_KEY.get(item.capture)
        return self.strip(item.pattern, key, name) if key else name

    def import_meta(self, item: _Capture) -> ImportMeta:
        """The import *item* records, its captures' `#strip!` applied."""
        meta = self.language.resolve_import(item.node, item.captures)
        return meta.model_copy(
            update={
                "module": self.strip(item.pattern, "_import_module", meta.module),
                "names": self.strip(item.pattern, "_import_names", meta.names),
            }
        )


def _line_imports(
    item: _Capture,
    modules: list[Node],
    directives: _Directives,
    registered_language: LanguageRegistration,
    file_path: str,
    blob_sha: str,
    content: bytes,
) -> Iterator[Chunk]:
    """One import per line of a quantified `@_import_module`'s nodes, named and spanned by it.

    For a grammar that gives a list's entries no node of their own (an rST
    `toctree`'s content is word tokens): a line's module is its text from
    its first captured node to its last, `#strip!` applied.
    """
    lines: dict[int, list[Node]] = {}
    for module in modules:
        lines.setdefault(module.start_point[0], []).append(module)
    base = directives.import_meta(item)
    for row, nodes in lines.items():
        text = content[nodes[0].start_byte : nodes[-1].end_byte].decode("utf-8", errors="replace")
        meta = base.model_copy(
            update={"module": directives.strip(item.pattern, "_import_module", text)}
        )
        yield Chunk(
            blob_sha=blob_sha,
            file_path=file_path,
            role=Import(name=text, meta=_fragment_named(meta, file_path)),
            language=registered_language.id,
            content=text,
            span=Span(start=row + 1, end=row + 1),
        )


def _capture_items(tree: Tree, directives: _Directives) -> list[_Capture]:
    """Each chunk-producing capture of the query, with its match, in source order."""
    items: list[_Capture] = []
    for pattern, capture_dict in QueryCursor(directives.query).matches(
        tree.root_node, directives.record
    ):
        for capture_name, nodes in capture_dict.items():
            if capture_name.startswith("_"):
                continue
            try:
                if ChunkKind(capture_name) not in _CAPTURE_KINDS:
                    continue
            except ValueError:
                continue
            for node in nodes:
                items.append(_Capture(pattern, node, capture_name, capture_dict))
    items.sort(key=lambda item: item.node.start_byte)
    return items


def _captured(
    registered_language: LanguageRegistration,
    file_path: str,
    blob_sha: str,
    content: bytes,
    items: list[_Capture],
    directives: _Directives,
    named: Mapping[int, str],
    sections: set[int],
    flat: Mapping[int, tuple[int, int, list[str]]],
    extraction: QueryExtraction,
) -> Iterator[Chunk]:
    """The chunks *items*, the query's captures in source order, make.

    *directives* gives each capture's name and import values. *named* maps
    each captured node's id to a name, for the scopes it opens.
    A captured section, one of *sections*, that ends in captured sections
    holds only the text before them (`_section_end`); a flat section, one of
    *flat*, has the end and scope `_flat_sections` gives it.
    """

    seen: list[tuple[int, int]] = []  # emitted symbol node spans (for interior comments)
    pending: list[Node] = []  # the comment block being accumulated

    def resolve(block: list[Node]) -> Chunk | None:
        """A comment block that folded nowhere: drop if interior, else standalone."""
        if any(s <= block[0].start_byte and block[-1].end_byte <= e for s, e in seen):
            return None
        end_byte = block[-1].end_byte
        # `last_line` already excludes a trailing newline's row; the byte range
        # still has to drop it so the content does not end with a blank line.
        if end_byte > block[-1].start_byte and content[end_byte - 1 : end_byte] == b"\n":
            end_byte -= 1
        return Chunk(
            blob_sha=blob_sha,
            file_path=file_path,
            role=Comment(),
            language=registered_language.id,
            content=content[block[0].start_byte : end_byte].decode("utf-8", errors="replace"),
            span=Span(start=block[0].start_point[0] + 1, end=last_line(block[-1])),
        )

    for item in items:
        node, capture_name, capture_dict = item.node, item.capture, item.captures
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

        scopes = _enclosing_scopes(node, extraction.scope_types)
        # The scope override's segments extend the tree-ancestry scope; the
        # default contributes the `@_scope` capture.
        scope_names = [
            *_scope_names(scopes, named),
            *registered_language.resolve_scope(capture_name, node, capture_dict),
        ]
        modules = capture_dict.get("_import_module", [])
        if capture_name == "import" and len(modules) > 1:
            yield from _line_imports(
                item, modules, directives, registered_language, file_path, blob_sha, content
            )
            seen.append((node.start_byte, node.end_byte))
            continue
        name = directives.name(item)
        role: Role
        match _resolve_kind(
            ChunkKind(capture_name),
            nearest_scope_is_class=_nearest_scope_is_class(scopes, extraction.class_scope_types),
        ):
            case ChunkKind.IMPORT:
                role = Import(
                    name=name, meta=_fragment_named(directives.import_meta(item), file_path)
                )
            case ChunkKind.DOC_SECTION | ChunkKind.CONFIG_KEY as kind:
                role = Section(kind=kind, name=name)
            case ChunkKind.COMMENT:
                role = Comment()
            case ChunkKind.RAW_CHUNK:
                role = RawSlice()
            case ChunkKind.FRAGMENT:
                role = Fragment(meta=FragmentMeta(node_types=(node.type,)))
            case kind:
                role = Definition(kind=kind, name=name)
        end_byte, line_end = node.end_byte, last_line(node)
        if node.id in flat:
            end_byte, line_end, scope_names = flat[node.id]
        elif capture_name == "doc_section":
            end_byte, line_end = _section_end(node, sections, content)
        seen.append((node.start_byte, end_byte))
        yield Chunk.model_validate(
            {
                "blob_sha": blob_sha,
                "file_path": file_path,
                "role": role,
                "scope": scope_names,
                "language": registered_language.id,
                "content": content[start_byte:end_byte].decode("utf-8", errors="replace"),
                "span": Span(start=line_start, end=line_end),
            }
        )

    if pending and (chunk := resolve(pending)) is not None:
        yield chunk


def injection_holders(registered_language: LanguageRegistration, tree: Tree) -> list[Node]:
    """The nodes holding a block *registered_language* delegates to another language.

    An HTML `<script>` element holds the javascript its injection query
    hands on: the element's tags belong to that block, not to the host. A
    block handed to a language that makes no fragments is not held: that
    language does not reach its every line (Markdown's text, handed to its
    inline half for links).
    """
    if registered_language.injection_query is None:
        return []
    query = _get_query(tree.language, registered_language.injection_query)
    holders: list[Node] = []
    for pattern, captures in QueryCursor(query).matches(tree.root_node):
        target = get_manager().get_registration(
            query.pattern_settings(pattern).get("injection.language") or ""
        )
        extraction = target.extraction if target is not None else None
        if isinstance(extraction, QueryExtraction) and not extraction.fragments:
            continue
        holders.extend(block.parent or block for block in captures.get("injection.content", []))
    return holders


def _fragments(
    registered_language: LanguageRegistration,
    file_path: str,
    blob_sha: str,
    content: bytes,
    tree: Tree,
    extraction: QueryExtraction,
    captured: list[Chunk],
    named: Mapping[int, str],
) -> Iterator[Chunk]:
    """Fragment chunks for the code *captured* leaves, so every line reaches a chunk.

    Walks the tree from the root by lines. A node no captured chunk
    touches is left over; a node one touches only in part, or a left-over
    one longer than `config.chunk_lines`, is looked into. Left-over siblings join one fragment until a blank line, a
    covered node or `config.chunk_lines` lines separate them, and a run
    of anonymous syntax alone (a closing brace) makes none. A node
    holding an injected block counts as covered: its language's
    extraction reaches it.
    """
    if not extraction.fragments:
        return
    covered = bytearray(content.count(b"\n") + 2)
    spans = [(c.line_start, c.line_end) for c in captured] + [
        (holder.start_point[0] + 1, last_line(holder))
        for holder in injection_holders(registered_language, tree)
    ]
    for start, end in spans:
        covered[start : end + 1] = b"\x01" * (end - start + 1)

    def runs(node: Node) -> Iterator[list[Node]]:
        run: list[Node] = []
        for child in node.children:
            lines = covered[child.start_point[0] + 1 : last_line(child) + 1]
            too_long = last_line(child) - child.start_point[0] > config.chunk_lines
            if 1 in lines or (too_long and child.named_child_count):
                if run:
                    yield run
                run = []
                if 0 in lines:
                    yield from runs(child)
                continue
            if run and (
                not _contiguous(content, run[-1].end_byte, child.start_byte)
                or last_line(child) - run[0].start_point[0] > config.chunk_lines
            ):
                yield run
                run = []
            run.append(child)
        if run:
            yield run

    for run in runs(tree.root_node):
        node_types = tuple(dict.fromkeys(n.type for n in run if n.is_named))
        text = content[run[0].start_byte : run[-1].end_byte].decode("utf-8", errors="replace")
        if not node_types or not text.strip():
            continue
        yield Chunk.model_validate(
            {
                "blob_sha": blob_sha,
                "file_path": file_path,
                "role": Fragment(meta=FragmentMeta(node_types=node_types)),
                "scope": _scope_names(_enclosing_scopes(run[0], extraction.scope_types), named),
                "language": registered_language.id,
                "content": text.rstrip("\n"),
                "span": Span(start=run[0].start_point[0] + 1, end=last_line(run[-1])),
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
                scope_names = _scope_names(scopes, {})
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
                    line_end=last_line(node),
                    ranges=doc_ranges,
                    start_byte=node.start_byte,
                    end_byte=node.end_byte,
                )
