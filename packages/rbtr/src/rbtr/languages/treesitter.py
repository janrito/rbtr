"""Tree-sitter parsing: a batch's files read into the rows the domain's rules take.

`read` parses each file of a batch, and each block it hands another
language (`sources`), and gives each parse's spans as `CaptureRow`s: its
query's captures (`query_captures`), then fragments for the code they
leave (`fragments`). Each capture's span is settled here, and the facts
only the tree or a plugin knows are recorded; `rbtr.domain.rules`
decides the rest from those values. Language-specific behaviour (query,
scope types, and name/scope/import resolution) is supplied by each
`LanguageRegistration`.
"""

from __future__ import annotations

import dataclasses
import itertools
import re
from collections.abc import Callable, Iterator, Mapping
from functools import lru_cache
from typing import TYPE_CHECKING, Literal, TypedDict

import dataframely as dy
import polars as pl
from tree_sitter import Language, Parser, Query, QueryCursor, Range

from rbtr.config import config
from rbtr.domain.identity import compose_scope
from rbtr.domain.models import Chunk, ChunkKind, ImportMeta
from rbtr.domain.rules import (
    DEFINITION_KINDS,
    CaptureRow,
    FileRow,
    MetadataCaptureName,
    MetadataCaptureRow,
    PlacedChunkRow,
)
from rbtr.languages._resolvers import NAME_CAPTURE_KEY, read_import
from rbtr.languages.chunks import chunk_rows, last_line
from rbtr.languages.manager import get_manager
from rbtr.languages.registration import ChunkExtraction, QueryExtraction

if TYPE_CHECKING:
    from tree_sitter import Node, Tree

    from rbtr.languages.registration import LanguageRegistration

CAPTURE_KINDS: frozenset[ChunkKind] = frozenset(
    {
        ChunkKind.FUNCTION,
        ChunkKind.METHOD,
        ChunkKind.CLASS,
        ChunkKind.VARIABLE,
        ChunkKind.TEST_FUNCTION,
        ChunkKind.IMPORT,
        ChunkKind.DOC_SECTION,
        ChunkKind.CONFIG_KEY,
        ChunkKind.COMMENT,
    }
)
"""The chunk kinds a query capture makes, each named by its kind's value.

The set keeps kinds no capture makes (`RAW_CHUNK`, `MIGRATION`, …) out.
"""

DOCSTRING_CAPTURE = "_docstring"
"""The helper capture marking a symbol's docstring: its spans within the symbol are its docs."""

MAX_INJECTION_DEPTH = 5
"""How deep blocks hand on blocks (a host embedded in a host). Insurance
only: delegation ends as ranges shrink, and at a language with no
`injection_query`."""


class ByteSpan(TypedDict):
    """A span of a source's bytes, as a capture's docs are kept."""

    start: int
    end: int


class Metadata(TypedDict):
    module: str | None
    names: str | None
    language_hint: str | None
    node_types: list[str] | None


# A TypedDict, not a dataclass: polars maps a dataclass's fields to a schema by
# position, and reads a list of nested dataclasses as null.
class Capture(TypedDict):
    """A `CaptureRow` as `read` builds it."""

    file: int
    source: int
    order: int
    match: int | None
    language: str
    kind: str
    name: str
    scope: str
    in_class: bool
    column: int
    docs: list[ByteSpan]
    start_byte: int
    end_byte: int
    line_start: int
    line_end: int
    metadata: Metadata | None


type Languages = Callable[[str], tuple[LanguageRegistration, Language | None] | None]
"""How `read` finds a language: its registration and grammar, or None when nothing registers it.

`extract_batch` injects `registered`; a test injects a registration no
plugin installs.
"""


class MetadataCapture(TypedDict):
    """A `MetadataCaptureRow` as `read` builds it."""

    file: int
    source: int
    match: int
    capture: str
    start_byte: int
    end_byte: int
    text: str


def registered(language: str) -> tuple[LanguageRegistration, Language | None] | None:
    """*language*'s registration and grammar, among the installed plugins'."""
    manager = get_manager()
    registration = manager.get_registration(language)
    return None if registration is None else (registration, manager.grammar(language))


@dataclasses.dataclass(frozen=True)
class Source:
    """One parse of a file of a batch: of the whole file, or of a block another language reads.

    *number* is its place among the batch's sources: a file's follow the
    file before, and a block follows the source that hands it on. *orders*
    numbers its file's rows, source by source. *ranges* restrict the parse
    to a block, and are None for a whole file. *holders* are the nodes
    holding the blocks it hands on to a language that reaches their every
    line.
    """

    file: int
    number: int
    orders: Iterator[int]
    path: str
    content: bytes
    registration: LanguageRegistration
    grammar: Language
    ranges: list[Range] | None
    tree: Tree
    holders: list[Node]


def read(
    files: dy.DataFrame[FileRow], languages: Languages
) -> tuple[
    dy.DataFrame[CaptureRow],
    dy.DataFrame[MetadataCaptureRow],
    dy.DataFrame[PlacedChunkRow],
    list[int],
]:
    """What tree-sitter and the plugins make of *files*, as rows.

    A file is read as its language's query or chunker, and each block it
    hands on the same way (`sources`): the query's spans are the
    `CaptureRow`s, its metadata captures the `MetadataCaptureRow`s, the
    chunkers' chunks the `PlacedChunkRow`s, numbered
    in one order per file, each checked against its schema in a lazy plan
    (cheaper than an eager check) and collected. A chunker's chunk is in its source's language
    when the chunker leaves it blank. The files whose language reads
    nothing and hands on no block are the list. *languages* finds each
    language.
    """
    captures: list[Capture] = []
    metadata_captures: list[MetadataCapture] = []
    chunked: list[tuple[Chunk, int, int, str, str]] = []
    plain: list[int] = []
    numbers = itertools.count()
    rows = files.select("file", "path", "blob_sha", "language", "content").iter_rows()
    for file, path, blob_sha, file_language, content in rows:
        registration, grammar = languages(file_language) or (None, None)
        reads = registration is not None and registration.extraction is not None
        hands_on = registration is not None and registration.injection_query is not None
        if not (reads and grammar is not None) and not hands_on:
            plain.append(file)
            continue
        if registration is None or grammar is None:
            continue
        orders = itertools.count()
        for source in sources(
            file, path, content, registration, grammar, languages, numbers, orders
        ):
            match source.registration.extraction:
                case ChunkExtraction(chunker=chunker):
                    for chunk in chunker(
                        path, blob_sha, content.decode(), source.grammar, source.ranges
                    ):
                        language = chunk.language or source.registration.id
                        chunked.append((chunk, file, next(orders), language, file_language))
                case QueryExtraction() as extraction if content:
                    captured, named, found = query_captures(source, extraction)
                    captures += captured
                    captures += fragments(source, extraction, captured, named)
                    metadata_captures += found
    return (
        pl.LazyFrame(captures, schema=CaptureRow.create_empty().schema)
        .pipe(CaptureRow.validate, cast=True)
        .collect(),
        pl.LazyFrame(metadata_captures, schema=MetadataCaptureRow.create_empty().schema)
        .pipe(MetadataCaptureRow.validate, cast=True)
        .collect(),
        pl.concat(
            [
                chunk_rows([chunk for chunk, *_ in chunked]).drop("language", "file_language"),
                pl.DataFrame(
                    [place for _, *place in chunked],
                    schema={
                        "file": pl.Int64,
                        "order": pl.Int64,
                        "language": pl.String,
                        "file_language": pl.String,
                    },
                    orient="row",
                ),
            ],
            how="horizontal",
        )
        .lazy()
        .pipe(PlacedChunkRow.validate, cast=True)
        .collect(),
        plain,
    )


def sources(
    file: int,
    path: str,
    content: bytes,
    registration: LanguageRegistration,
    grammar: Language,
    languages: Languages,
    numbers: Iterator[int],
    orders: Iterator[int],
) -> Iterator[Source]:
    """*content* parsed as *registration*'s language, then each block it hands on, depth first.

    Each source is parsed once, within its ranges, and numbered from
    *numbers*; its rows are numbered from *orders*, the file's. A source hands on its blocks down to `MAX_INJECTION_DEPTH`,
    each to a language *languages* finds with a grammar.
    """
    pending: list[tuple[LanguageRegistration, Language, list[Range] | None, int]] = [
        (registration, grammar, None, 0)
    ]
    while pending:
        registration, grammar, ranges, depth = pending.pop()
        parser = Parser(grammar)
        if ranges is not None:
            parser.included_ranges = ranges
        tree = parser.parse(content)
        blocks, holders = injections(registration, tree, languages)
        yield Source(
            file, next(numbers), orders, path, content, registration, grammar, ranges, tree, holders
        )
        if depth <= MAX_INJECTION_DEPTH:
            # Last block first: each is read, with its own blocks, before the next.
            pending.extend(
                (target, target_grammar, [block], depth + 1)
                for target, target_grammar, block in reversed(blocks)
            )


def injections(
    registration: LanguageRegistration, tree: Tree, languages: Languages
) -> tuple[list[tuple[LanguageRegistration, Language, Range]], list[Node]]:
    """The blocks *registration*'s injection query hands on from *tree*, and the nodes holding them.

    A block's language is a static `#set! injection.language`, or a
    captured `@injection.language` hint (`injection_language`). A block
    marked by several patterns takes the language of the highest
    `injection.priority`; an empty block, or one whose language
    *languages* finds no grammar for, is not read. The blocks are in
    source order.

    A block counts as covered when fragments are found: its lines are the
    language's that reads it. The host's syntax around it (an HTML
    `<script>` tag, a Markdown code fence, a section's other blocks) is
    the host's own. A block handed to a language that makes no fragments
    is not held: that language does not reach its every line (Markdown's
    text, handed to its inline half for links).
    """
    if registration.injection_query is None:
        return [], []
    query = compiled_query(tree.language, registration.injection_query)
    winner: dict[tuple[int, int], tuple[int, str, Range]] = {}
    holders: list[Node] = []
    for pattern, captures in QueryCursor(query).matches(tree.root_node):
        settings = query.pattern_settings(pattern)
        static, _ = languages(settings.get("injection.language") or "") or (None, None)
        extraction = static.extraction if static is not None else None
        if not (isinstance(extraction, QueryExtraction) and not extraction.fragments):
            holders.extend(captures.get("injection.content", []))
        target = settings.get("injection.language") or injection_language(captures)
        if target is None:
            continue
        priority = int(settings.get("injection.priority") or "0")
        for block in captures.get("injection.content", []):
            if block.text is None or not block.text.strip():
                continue
            span = (block.start_byte, block.end_byte)
            if span not in winner or priority > winner[span][0]:
                winner[span] = (priority, target, block.range)
    blocks: list[tuple[LanguageRegistration, Language, Range]] = []
    for _, target, block_range in winner.values():
        target_registration, target_grammar = languages(target) or (None, None)
        if target_registration is not None and target_grammar is not None:
            blocks.append((target_registration, target_grammar, block_range))
    return blocks, holders


def injection_language(captures: dict[str, list[Node]]) -> str | None:
    """The language id a captured `@injection.language` hint names, if any.

    A markdown fence tags its block with a free-form name (` ```python `,
    ` ```py `). The name is read as a language id, then as a file
    extension, so both spellings reach the python plugin. An unknown hint
    gives None, and the block is left unread.
    """
    nodes = captures.get("injection.language")
    if not nodes or nodes[0].text is None:
        return None
    hint = nodes[0].text.decode().strip().lower()
    manager = get_manager()
    return hint if manager.get_registration(hint) else manager.detect_language(f"x.{hint}")


def query_captures(
    source: Source, extraction: QueryExtraction
) -> tuple[list[Capture], dict[int, str], list[MetadataCapture]]:
    """*source*'s query captures as rows, in source order, the name each captured node gives,
    and its metadata captures (`MetadataCaptureName`) as rows.

    Each capture's kind, name, scope and metadata come from the query's
    capture conventions:

    - `@function` / `@_fn_name` → `ChunkKind.FUNCTION`
    - `@class` / `@_cls_name` → `ChunkKind.CLASS`
    - `@method` / `@_method_name` → `ChunkKind.METHOD`
    - `@import` / `@_import_module` → `ChunkKind.IMPORT`
    - `@doc_section` / `@_section_name` → `ChunkKind.DOC_SECTION`
    - `@config_key` / `@_section_name` → `ChunkKind.CONFIG_KEY`
    - `@comment` → `ChunkKind.COMMENT`

    Names, scopes and import metadata come from the registration's
    resolvers, `#strip!` applied. Each span is settled here: a captured
    section that ends in captured sections ends where the first of them
    begins (`section_end`), a flat section at the next one
    (`flat_sections`), and a quantified `@_import_module` makes one
    import per line (`line_imports`). Whether a capture's nearest scope is
    class-like, and the column it starts at, are recorded for the rules.
    A node captured more than once (`a, b = …`) names the scope it opens
    by its first capture.
    """
    directives = Directives(
        source.registration, compiled_query(source.tree.language, extraction.query)
    )
    nodes, metadata_captures = captured_nodes(source, directives)
    named: dict[int, str] = {}
    for captured in nodes:
        if captured.capture != "comment":
            named.setdefault(captured.node.id, directives.name(captured))
    sections = {captured.node.id for captured in nodes if captured.capture == "doc_section"}
    flat = flat_sections(nodes, directives, source.content, source.tree.root_node.end_byte)
    rows: list[Capture] = []
    for captured in nodes:
        node = captured.node
        if captured.capture == "comment":
            rows.append(capture(source, ChunkKind.COMMENT, node, node, match=captured.match))
            continue
        modules = captured.captures.get("_import_module", [])
        if captured.capture == "import" and len(modules) > 1:
            rows.extend(line_imports(source, captured, modules, directives))
            continue
        scopes = enclosing_scopes(node, extraction.scope_types)
        # The scope override's segments extend the tree-ancestry scope; the
        # default contributes the `@_scope` capture.
        scope = [
            *scope_names(scopes, named),
            *source.registration.resolve_scope(captured.capture, node, captured.captures),
        ]
        end: tuple[int, int] | None = None
        if node.id in flat:
            end_byte, line_end, scope = flat[node.id]
            end = (end_byte, line_end)
        elif captured.capture == "doc_section":
            end = section_end(node, sections, source.content)
        rows.append(
            capture(
                source,
                ChunkKind(captured.capture),
                node,
                node,
                end=end,
                name=directives.name(captured),
                scope=scope,
                in_class=bool(scopes) and scopes[0].type in extraction.class_scope_types,
                docs=[
                    ByteSpan(start=d.start_byte, end=d.end_byte)
                    for d in captured.captures.get(DOCSTRING_CAPTURE, [])
                    if node.start_byte <= d.start_byte and d.end_byte <= node.end_byte
                ],
                meta=directives.import_meta(captured) if captured.capture == "import" else None,
                match=captured.match,
            )
        )
    return rows, named, metadata_captures


def capture(
    source: Source,
    kind: ChunkKind,
    first: Node,
    last: Node,
    *,
    end: tuple[int, int] | None = None,
    name: str = "",
    scope: list[str] | None = None,
    in_class: bool = False,
    docs: list[ByteSpan] | None = None,
    meta: ImportMeta | None = None,
    node_types: list[str] | None = None,
    match: int | None = None,
) -> Capture:
    """A row for *source*'s span from *first*'s start to *last*'s end, numbered next in its file.

    *end*, when given, is where a section's own text ends: its end byte
    and last line. *meta* is an import's metadata, *node_types* a
    fragment's, *match* the query match that captured it.
    """
    end_byte, line_end = end if end is not None else (last.end_byte, last_line(last))
    metadata: Metadata | None = None
    if meta is not None:
        metadata = Metadata(
            module=meta.module,
            names=meta.names,
            language_hint=meta.language_hint,
            node_types=None,
        )
    elif node_types is not None or kind.value in DEFINITION_KINDS:
        metadata = Metadata(module=None, names=None, language_hint=None, node_types=node_types)
    return Capture(
        file=source.file,
        source=source.number,
        order=next(source.orders),
        match=match,
        language=source.registration.id,
        kind=kind.value,
        name=name,
        scope=compose_scope(scope or []),
        in_class=in_class,
        column=first.start_point[1],
        docs=docs or [],
        start_byte=first.start_byte,
        end_byte=end_byte,
        line_start=first.start_point[0] + 1,
        line_end=line_end,
        metadata=metadata,
    )


@dataclasses.dataclass(frozen=True)
class CapturedNode:
    """A node the query captured: its capture name, the pattern that matched, and the match.

    *match* numbers the match in its source; *captures* is its every
    capture, helpers included.
    """

    pattern: int
    match: int
    node: Node
    capture: str
    captures: dict[str, list[Node]]


def captured_nodes(
    source: Source, directives: Directives
) -> tuple[list[CapturedNode], list[MetadataCapture]]:
    """Each node the query captures as a chunk or a comment, with its match, in source order.

    Also each `MetadataCaptureName` capture of each match. The nodes one
    match captures under one such name are one span, from the first's start
    to the last's end: a grammar with no single callee node (Java's
    `method_invocation`, `object` then `name`) captures both parts.
    """
    nodes: list[CapturedNode] = []
    metadata_captures: list[MetadataCapture] = []
    metadata_names = {capture.value for capture in MetadataCaptureName}
    matches = QueryCursor(directives.query).matches(source.tree.root_node, directives.record)
    for match, (pattern, captures) in enumerate(matches):
        for capture_name, captured in captures.items():
            if capture_name in metadata_names:
                start = min(n.start_byte for n in captured)
                end = max(n.end_byte for n in captured)
                text = source.content[start:end].decode("utf-8", "replace")
                metadata_captures.append(
                    MetadataCapture(
                        file=source.file,
                        source=source.number,
                        match=match,
                        capture=capture_name,
                        start_byte=start,
                        end_byte=end,
                        text=directives.strip(pattern, capture_name, text),
                    )
                )
                continue
            if capture_name.startswith("_"):
                continue
            try:
                if ChunkKind(capture_name) not in CAPTURE_KINDS:
                    continue
            except ValueError:
                continue
            nodes.extend(
                CapturedNode(pattern, match, node, capture_name, captures) for node in captured
            )
    nodes.sort(key=lambda captured: captured.node.start_byte)
    return nodes, metadata_captures


@dataclasses.dataclass
class Directives:
    """A query's directives, and the values a capture gives under them.

    tree-sitter leaves directives to its bindings, and py-tree-sitter applies
    `#set!` only (`Query.pattern_settings`). `#strip! @capture "regex"`
    removes what the regex matches from the capture's text (tree-sitter's
    query documentation, "Predicates and Directives"); the engine applies it
    to the name and import values read from a capture. py-tree-sitter
    hands a `#strip!` to Python only while the query matches, so it is
    recorded then.
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

    def name(self, captured: CapturedNode) -> str:
        """The name *captured* gives, its name capture's `#strip!` applied."""
        name = self.language.resolve_name(captured.capture, captured.node, captured.captures)
        key = NAME_CAPTURE_KEY.get(captured.capture)
        return self.strip(captured.pattern, key, name) if key else name

    def import_meta(self, captured: CapturedNode) -> ImportMeta:
        """The import *captured* records: its module (`read_import`) as written, `#strip!`ped, and its pattern's `#set!` names and language hint.

        A pattern that sets `import.prefix` has it written before its
        module, unless the module already starts with a path step (`./`,
        `../`): Ruby's `require_relative "helpers"` is `./helpers`.

        The names it lists and the bindings it makes are metadata captures
        of their own, gathered over the statement's matches by the rules
        (`with_metadata`).
        """
        meta = read_import(captured.captures)
        settings = self.query.pattern_settings(captured.pattern)
        module = self.strip(captured.pattern, "_import_module", meta.module)
        prefix = settings.get("import.prefix")
        if prefix and module and not module.startswith(("./", "../")):
            module = f"{prefix}{module}"
        return meta.model_copy(
            update={
                "module": module,
                "names": settings.get("import.names") or meta.names,
                "language_hint": settings.get("import.language_hint") or meta.language_hint,
            }
        )


def flat_sections(
    nodes: list[CapturedNode], directives: Directives, content: bytes, end: int
) -> dict[int, tuple[int, int, list[str]]]:
    """Each flat section's end byte, 1-based last line and scope, by node id.

    A flat section is one whose match captures `@_level`: its grammar gives
    it no body to nest in (tree-sitter-rst's `section` holds its title
    alone). Its level is the order in which its style (its `@_level` texts,
    `#strip!` applied, joined) first appears; it is scoped under the nearest
    earlier flat section of a higher level, and holds its text up to the
    next flat section, or to *end*, less trailing whitespace.
    """
    flat = [
        captured
        for captured in nodes
        if captured.capture == "doc_section" and "_level" in captured.captures
    ]
    ranks: dict[str, int] = {}
    above: list[tuple[int, str]] = []
    spans: dict[int, tuple[int, int, list[str]]] = {}
    for at, captured in enumerate(flat):
        style = "".join(
            directives.strip(captured.pattern, "_level", node.text.decode())
            for node in captured.captures["_level"]
            if node.text
        )
        rank = ranks.setdefault(style, len(ranks))
        while above and above[-1][0] >= rank:
            above.pop()
        scope = [name for _, name in above]
        above.append((rank, directives.name(captured)))
        stop = flat[at + 1].node.start_byte if at + 1 < len(flat) else end
        while stop > captured.node.start_byte and content[stop - 1 : stop] in (
            b" ",
            b"\t",
            b"\r",
            b"\n",
        ):
            stop -= 1
        spans[captured.node.id] = (stop, content.count(b"\n", 0, stop) + 1, scope)
    return spans


def section_end(node: Node, sections: set[int], content: bytes) -> tuple[int, int]:
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


def line_imports(
    source: Source, captured: CapturedNode, modules: list[Node], directives: Directives
) -> Iterator[Capture]:
    """One import per line of a quantified `@_import_module`'s nodes, named and spanned by it.

    For a grammar that gives a list's entries no node of their own (an rST
    `toctree`'s content is word tokens): a line's module is its text from
    its first captured node to its last, `#strip!` applied.
    """
    lines: dict[int, list[Node]] = {}
    for module in modules:
        lines.setdefault(module.start_point[0], []).append(module)
    base = directives.import_meta(captured)
    for nodes in lines.values():
        text = source.content[nodes[0].start_byte : nodes[-1].end_byte].decode(
            "utf-8", errors="replace"
        )
        meta = base.model_copy(
            update={"module": directives.strip(captured.pattern, "_import_module", text)}
        )
        yield capture(
            source,
            ChunkKind.IMPORT,
            nodes[0],
            nodes[-1],
            name=text,
            meta=meta,
            match=captured.match,
        )


def enclosing_scopes(node: Node, scope_types: frozenset[str]) -> list[Node]:
    """Ancestor nodes that open a naming scope, nearest first."""
    scopes: list[Node] = []
    current = node.parent
    while current is not None:
        if current.type in scope_types:
            scopes.append(current)
        current = current.parent
    return scopes


def scope_names(scopes: list[Node], named: Mapping[int, str]) -> list[str]:
    """Names of enclosing *scopes*, outermost first, for the `::` address.

    A scope the query captured is named as its capture names it (a
    Markdown `section` by its heading, from *named*); any other by its
    `name` field, or Rust's `type`. `compose_scope` forms the address and
    drops empty segments.
    """
    names: list[str] = []
    for scope in reversed(scopes):
        field = scope.child_by_field_name("name")
        if field is None:
            field = scope.child_by_field_name("type")
        names.append(named.get(scope.id) or (field.text.decode() if field and field.text else ""))
    return names


def fragments(
    source: Source, extraction: QueryExtraction, captured: list[Capture], named: Mapping[int, str]
) -> list[Capture]:
    """Fragment rows for the code *captured* leaves, so every line reaches a chunk.

    A line is covered by a captured span, or by a node holding a block
    another language reads (`Source.holders`). The runs of uncovered
    nodes (`runs`) are fragments, scoped like a capture at their start. A
    run of anonymous syntax alone (`try:` around captured imports, the
    `#endif` closing a guard) is a fragment too, with no node types.
    """
    if not extraction.fragments:
        return []
    covered = bytearray(source.content.count(b"\n") + 2)
    lines = [(c["line_start"], c["line_end"]) for c in captured] + [
        (holder.start_point[0] + 1, last_line(holder)) for holder in source.holders
    ]
    for start, end in lines:
        covered[start : end + 1] = b"\x01" * (end - start + 1)
    rows: list[Capture] = []
    for run in runs(source.tree.root_node, covered, source.content):
        node_types = list(dict.fromkeys(node.type for node in run if node.is_named))
        scope = scope_names(enclosing_scopes(run[0], extraction.scope_types), named)
        rows.append(
            capture(source, ChunkKind.FRAGMENT, run[0], run[-1], scope=scope, node_types=node_types)
        )
    return rows


def runs(node: Node, covered: bytearray, content: bytes) -> Iterator[list[Node]]:
    """Runs of *node*'s children that no *covered* line reaches.

    A child no covered line touches is left over; a child one touches only
    in part, or a left-over one longer than `config.chunk_lines`, is
    looked into, and a leaf touched only in part is a run of its own. Left-over siblings join one run until a blank line, a
    covered node or `config.chunk_lines` lines separate them.
    """
    run: list[Node] = []
    for child in node.children:
        lines = covered[child.start_point[0] + 1 : last_line(child) + 1]
        too_long = last_line(child) - child.start_point[0] > config.chunk_lines
        if 1 in lines or (too_long and child.named_child_count):
            if run:
                yield run
            run = []
            if 0 in lines and child.child_count:
                yield from runs(child, covered, content)
            elif 0 in lines:
                # A leaf has nothing to look into: it is its own run, and its lines
                # are covered from here on.
                yield [child]
                covered[child.start_point[0] + 1 : last_line(child) + 1] = b"\x01" * len(lines)
            continue
        if run and (
            not contiguous(content, run[-1].end_byte, child.start_byte)
            or last_line(child) - run[0].start_point[0] > config.chunk_lines
        ):
            yield run
            run = []
        run.append(child)
    if run:
        yield run


def contiguous(content: bytes, a_end: int, b_start: int) -> bool:
    """Whether `[a_end, b_start)` is only whitespace with no blank line.

    A trailing newline some grammars fold into the earlier node is
    discounted (rust's `line_comment` spans its `\\n`; go's does not), so
    a blank line is two or more newlines in the gap.
    """
    if content[a_end:b_start].strip() != b"":
        return False
    end = a_end - 1 if a_end > 0 and content[a_end - 1 : a_end] == b"\n" else a_end
    return content.count(b"\n", end, b_start) < 2


@lru_cache(maxsize=32)
def compiled_query(grammar: Language, text: str) -> Query:
    """The query *text* compiled for *grammar*, cached.

    Compiling costs about 0.6 ms; caching saves about 0.9 s on a
    1,500-file Python repository, whose files all share one query.
    """
    return Query(grammar, text)
