"""Query extraction's rows, and the rules that make a batch's chunks of them.

The tree-sitter adapter reads a batch of files (`FileRow`) into captures
(`CaptureRow`) and its chunkers' chunks (`PlacedChunkRow`). The rules
make chunks of the captures (`capture_chunks`, which first folds the
comments: `fold_comments`), cut the files nothing reads into windows of
lines (`line_windows`), and put each file's chunks in order with a
presence chunk where it needs one (`assemble`). A chunk the store keeps
is a `ChunkRow`; a `PlacedChunkRow` is one at its place in a batch.
Every rule is polars over rows, run once for all the files of a batch.
"""

from __future__ import annotations

from enum import StrEnum

import dataframely as dy
import polars as pl

from rbtr.domain.models import ChunkKind, EdgeKind, Lookup, narrows

ASCII_WHITESPACE = " \t\n\r\x0b\x0c"
"""What `bytes.strip()` strips: a gap of these, and no blank line, is contiguous."""


class FileRow(dy.Schema):
    """A file of a batch, and the language it is read as.

    `content` is the file's bytes with each byte that is not UTF-8
    replaced by one `?`: offsets stay, and every slice is text.
    """

    file = dy.Int64(nullable=False, primary_key=True)
    path = dy.String(nullable=False)
    blob_sha = dy.String(nullable=False)
    language = dy.String(nullable=False)
    content = dy.Binary(nullable=False)


def reference_list() -> dy.List:
    """The calls and bases a chunk writes: each the edge it makes, a lookup, the names it follows and a local's type."""
    return dy.List(
        dy.Struct(
            {
                "kind": dy.String(nullable=False),
                "lookup": dy.String(nullable=False),
                "path": dy.List(dy.String(), nullable=False),
                "type": dy.List(dy.String(), nullable=True),
            }
        ),
        nullable=True,
    )


def binding_list(*, nullable: bool) -> dy.List:
    """The names an import binds: each a name and the local it is bound to."""
    return dy.List(
        dy.Struct({"name": dy.String(nullable=False), "local": dy.String(nullable=False)}),
        nullable=nullable,
    )


def capture_metadata() -> dy.Struct:
    """A capture's payload: one import's module, names, hint and bindings; a fragment's node types; the calls and bases it writes.

    Null for a role with no payload (sections, comments, raw slices).
    """
    return dy.Struct(
        {
            "module": dy.String(nullable=True),
            "names": dy.String(nullable=True),
            "language_hint": dy.String(nullable=True),
            "node_types": dy.List(dy.String(), nullable=True),
            "refs": reference_list(),
            "bindings": binding_list(nullable=True),
        },
        nullable=True,
    )


def import_list() -> dy.List:
    """The imports a chunk makes, each the import's text, its module, the names it lists, its hint and its bindings."""
    return dy.List(
        dy.Struct(
            {
                "text": dy.String(nullable=False),
                "module": dy.String(nullable=False),
                "names": dy.String(nullable=False),
                "language_hint": dy.String(nullable=False),
                "bindings": binding_list(nullable=False),
            }
        ),
        nullable=False,
    )


UNNAMED_KINDS = [
    ChunkKind.STATEMENT.value,
    ChunkKind.FRAGMENT.value,
    ChunkKind.COMMENT.value,
    ChunkKind.RAW_CHUNK.value,
]
"""The kinds of chunk that have no name: its rows have no names."""

DEFINITION_KINDS = [
    ChunkKind.FUNCTION.value,
    ChunkKind.CLASS.value,
    ChunkKind.METHOD.value,
    ChunkKind.VARIABLE.value,
    ChunkKind.MIGRATION.value,
    ChunkKind.TEST_FUNCTION.value,
    ChunkKind.API_ENDPOINT.value,
]
"""The kinds a definition or class role has: their rows carry `calls`."""


HOLDER_KINDS = [ChunkKind.FRAGMENT.value, ChunkKind.STATEMENT.value, *DEFINITION_KINDS]
"""The kinds whose rows hold the calls and bases they write: a definition, a statement or a fragment."""


def holds_references() -> pl.Expr:
    """Whether a row's chunk holds the calls and bases it writes (`HOLDER_KINDS`)."""
    return pl.col("kind").is_in(HOLDER_KINDS)


class ChunkRow(dy.Schema):
    """A chunk as a row: one span of a file.

    The role is spread over `kind`, `name` and `metadata`: the imports
    the chunk makes (empty when it makes none; a key or a section can
    import too); a fragment's node types (a diagnostic, read by rbtr-eval
    only), null on other kinds; the calls and bases a definition, class,
    statement or fragment writes (a base only on a class), null on other kinds.
    `docs` is the text of its documentation, null when it has none.
    `content` is the file's text from `start_byte` to `end_byte`, and
    `line_end` the line its last byte is on.
    """

    blob_sha = dy.String(nullable=False)
    file_language = dy.String(nullable=False)
    language = dy.String(nullable=False)
    kind = dy.Enum(k.value for k in ChunkKind)
    name = dy.String(nullable=False)
    scope = dy.String(nullable=False)
    content = dy.String(nullable=False)
    start_byte = dy.Int64(nullable=False)
    end_byte = dy.Int64(nullable=False)
    line_start = dy.Int32(nullable=False)
    line_end = dy.Int32(nullable=False)
    metadata = dy.Struct(
        {
            "node_types": dy.List(dy.String(), nullable=True),
            "refs": reference_list(),
            "imports": import_list(),
        },
        nullable=False,
    )
    docs = dy.String(nullable=True)

    @dy.rule()
    def no_name_on_unnamed_kinds(cls) -> pl.Expr:
        return ~cls.kind.col.is_in(UNNAMED_KINDS) | (cls.name.col == "")

    @dy.rule()
    def an_import_imports(cls) -> pl.Expr:
        imports = cls.metadata.col.struct.field("imports")
        return (cls.kind.col != ChunkKind.IMPORT.value) | (imports.list.len() > 0)

    @dy.rule()
    def node_types_only_on_fragments(cls) -> pl.Expr:
        node_types = cls.metadata.col.struct.field("node_types")
        return node_types.is_not_null() == (cls.kind.col == ChunkKind.FRAGMENT.value)

    @dy.rule()
    def refs_only_on_definitions_classes_and_fragments(cls) -> pl.Expr:
        refs = cls.metadata.col.struct.field("refs")
        return refs.is_not_null() == holds_references()

    @dy.rule()
    def bases_only_on_classes(cls) -> pl.Expr:
        refs = cls.metadata.col.struct.field("refs")
        has_base = refs.list.eval(
            pl.element().struct.field("kind") == EdgeKind.INHERITS.value
        ).list.any()
        return ~has_base.fill_null(value=False) | (cls.kind.col == ChunkKind.CLASS.value)

    @dy.rule()
    def refs_are_well_formed(cls) -> pl.Expr:
        """Each ref follows a name; a type is a value's; a base is a name path."""
        ref = pl.element().struct
        well_formed = (
            (ref.field("path").list.len() > 0)
            & (ref.field("type").is_null() | (ref.field("lookup") == Lookup.TYPE.value))
            & (
                (ref.field("kind") != EdgeKind.INHERITS.value)
                | (ref.field("lookup") == Lookup.UNQUALIFIED.value)
            )
        )
        refs = cls.metadata.col.struct.field("refs")
        return refs.list.eval(well_formed).list.all().fill_null(value=True)


def no_references() -> pl.Expr:
    """An empty `refs` list: a holder that writes no call or base."""
    return pl.lit([], dtype=reference_list().dtype)


def no_metadata() -> pl.Expr:
    """A chunk's `metadata` when it imports nothing and is no holder or fragment."""
    return pl.lit({"node_types": None, "refs": None, "imports": []}, dtype=ChunkRow.metadata.dtype)


def chunk_metadata() -> pl.Expr:
    """A capture's `metadata` as its chunk's: an import capture's import as a list of one, named by its text."""
    meta = pl.col("metadata").struct
    one_import = pl.concat_list(
        pl.struct(
            text=pl.col("name"),
            module=meta.field("module").fill_null(""),
            names=meta.field("names").fill_null(""),
            language_hint=meta.field("language_hint").fill_null(""),
            bindings=meta.field("bindings").fill_null(
                pl.lit([], dtype=binding_list(nullable=True).dtype)
            ),
        )
    )
    return pl.struct(
        node_types=meta.field("node_types"),
        refs=meta.field("refs"),
        imports=pl.when(pl.col("kind") == ChunkKind.IMPORT.value)
        .then(one_import)
        .otherwise(pl.lit([], dtype=import_list().dtype)),
    )


class PlacedChunkRow(ChunkRow):
    """A chunk at its place in a batch: the file it is read from, and its order among that file's chunks."""

    file = dy.Int64(nullable=False)
    order = dy.Int64(nullable=False)


class CaptureRow(dy.Schema):
    """A span of a source that makes a chunk: a query capture, a comment, or a fragment.

    A source is one parse: a file, or a block of it another language
    reads. `order` is the span's place among its file's spans, source by
    source. `kind` is as
    captured, a function not yet promoted; `in_class` is whether its
    nearest enclosing scope is class-like, and `column` the byte its span
    starts at within its line. `docs` are the spans of its documentation.
    `metadata` is an import's module and names, as resolved (its `#` not
    yet read), a fragment's node types, or what a definition calls.
    `match` is the query match that captured it, null for a fragment.
    A levelled capture (`levelled`) states its level as a number (`level`,
    from its pattern's `#set! level`) or as a style ranked by first use
    (`level_style`, its `@_level` texts). `source_end` is where its source
    ends: its file's, or the block's that a language read.
    """

    file = dy.Int64(nullable=False)
    source = dy.Int64(nullable=False)
    order = dy.Int64(nullable=False)
    match = dy.Int64(nullable=True)
    language = dy.String(nullable=False)
    kind = dy.Enum(k.value for k in ChunkKind)
    name = dy.String(nullable=False)
    scope = dy.String(nullable=False)
    in_class = dy.Bool(nullable=False)
    column = dy.Int64(nullable=False)
    docs = dy.List(
        dy.Struct({"start": dy.Int64(nullable=False), "end": dy.Int64(nullable=False)}),
        nullable=False,
    )
    start_byte = dy.Int64(nullable=False)
    end_byte = dy.Int64(nullable=False)
    line_start = dy.Int64(nullable=False)
    line_end = dy.Int64(nullable=False)
    metadata = capture_metadata()
    level = dy.Int32(nullable=True)
    level_style = dy.String(nullable=True)
    source_end = dy.Int64(nullable=False)

    @dy.rule()
    def one_level_statement(cls) -> pl.Expr:
        return cls.level.col.is_null() | cls.level_style.col.is_null()

    @dy.rule()
    def a_match_except_on_fragments(cls) -> pl.Expr:
        return cls.match.col.is_null() == (cls.kind.col == ChunkKind.FRAGMENT.value)

    @dy.rule()
    def metadata_of_roles_with_a_payload(cls) -> pl.Expr:
        return cls.metadata.col.is_not_null() == cls.kind.col.is_in(
            [ChunkKind.IMPORT.value, *HOLDER_KINDS]
        )


class MetadataCaptureName(StrEnum):
    """The captures a query marks a chunk's `metadata` with: its calls and bases, an import's names and bindings.

    A call's callee (`@_call`) and a class's base (`@_base`) are paths; a
    `self` or `super` marker (`@_self`, `@_super`) marks the head of the
    path starting where it starts, so the language's query, not core,
    says which names reach the enclosing class.
    """

    CALL = "_call"
    BASE = "_base"
    SELF = "_self"
    SUPER = "_super"
    IMPORT_NAME = "_import_names"
    """A name an import lists."""
    BIND_NAME = "_bind_name"
    """What a binding names: an import's name or module, a local's written type."""
    BIND_LOCAL = "_bind_local"
    """The name a binding gives; with `@_bind_name` in the same match, one binding."""
    SUPER_CALL = "_super_call"
    """A call to the bases' method of its function's own name: Ruby's bare `super`."""
    LOCAL = "_local"
    """A name a function holds a value under: a parameter, an assignment target."""
    LOCAL_TYPE = "_local_type"
    """The type written for the `@_local` of its match: an annotation, a constructor called."""


class MetadataCaptureRow(dy.Schema):
    """A node a query captured as a `MetadataCaptureName`: its source, query match, bytes and text."""

    file = dy.Int64(nullable=False)
    source = dy.Int64(nullable=False)
    match = dy.Int64(nullable=False)
    capture = dy.Enum(c.value for c in MetadataCaptureName)
    start_byte = dy.Int64(nullable=False)
    end_byte = dy.Int64(nullable=False)
    text = dy.String(nullable=False)


def text(start: pl.Expr, stop: pl.Expr) -> pl.Expr:
    """The file's text from *start* to *stop*; empty when *stop* is not after *start*."""
    return pl.col("content").bin.slice(start, pl.max_horizontal(stop - start, 0)).cast(pl.String)


def contiguous(a_end: pl.Expr, b_start: pl.Expr) -> pl.Expr:
    """Only whitespace, and no blank line, between *a_end* and *b_start*.

    A newline the earlier span ends with counts towards the blank line, as
    a grammar may fold it into a comment (rust's `line_comment`).
    """
    gap = text(a_end, b_start)
    folded = (a_end > 0) & (pl.col("content").bin.slice(a_end - 1, 1) == pl.lit(b"\n"))
    newlines = gap.str.count_matches("\n", literal=True) + folded.cast(pl.Int64)
    return (gap.str.strip_chars(ASCII_WHITESPACE) == "") & (newlines < 2)


def starts_line() -> pl.Expr:
    """Only spaces and tabs precede a span on its line.

    A comment that trails code (`x = 1  # note`) does not: it documents
    the statement before it, so it never folds into what follows.
    """
    return (
        text(pl.col("start_byte") - pl.col("column"), pl.col("start_byte"))
        .str.strip_chars(" \t")
        .eq("")
    )


def starts_block() -> pl.Expr:
    """Every span starts a block but a comment that starts its line, flush below another that does."""
    line_comment = (pl.col("kind") == ChunkKind.COMMENT.value) & pl.col("starts_line")
    joins = (
        line_comment
        & line_comment.shift(1).over("source")
        & contiguous(pl.col("end_byte").shift(1).over("source"), pl.col("start_byte"))
    )
    return ~joins.fill_null(False)


def takes_block_above() -> pl.Expr:
    """A captured symbol, not an import, takes the block of line comments flush above it."""
    above = (pl.col("kind") == ChunkKind.COMMENT.value) & pl.col("starts_line")
    return (
        ~pl.col("kind").is_in(
            [ChunkKind.COMMENT.value, ChunkKind.IMPORT.value, ChunkKind.FRAGMENT.value]
        )
        & above.shift(1).over("source")
        & contiguous(pl.col("end_byte").shift(1).over("source"), pl.col("start_byte"))
    ).fill_null(False)


def inside_earlier_symbol() -> pl.Expr:
    """A comment block that ends within a symbol before it: the symbol's chunk holds it."""
    reach = (
        pl.when(pl.col("kind") != ChunkKind.COMMENT.value)
        .then(pl.col("end_byte"))
        .cum_max()
        .forward_fill()
        .over("source")
    )
    return ((pl.col("kind") == ChunkKind.COMMENT.value) & (reach >= pl.col("end_byte"))).fill_null(
        False
    )


def promoted() -> pl.Expr:
    """A function whose nearest enclosing scope is class-like is a method.

    Grammars use one node type for both: the difference is where it is.
    A function nested in another function stays a function.
    """
    return (
        pl.when((pl.col("kind") == ChunkKind.FUNCTION.value) & pl.col("in_class"))
        .then(pl.lit(ChunkKind.METHOD.value))
        .otherwise(pl.col("kind"))
    )


def module_before_fragment() -> pl.Expr:
    """An import's module before its `#`, as a URL's: `page.html#intro` is `page.html`.

    A fragment alone, `#intro`, is in the importing file itself. Read
    within `metadata`.
    """
    module = pl.field("module")
    before = module.str.splitn("#", 2).struct.field("field_0")
    file_name = pl.col("path").str.split("/").list.last()
    return (
        pl.when(module.str.contains("#", literal=True))
        .then(pl.when(before == "").then(file_name).otherwise(before))
        .otherwise(module)
    )


def fragment_as_names() -> pl.Expr:
    """What follows an import's `#` names what it imports, unless its names already do.

    Read within `metadata`.
    """
    after = pl.field("module").str.splitn("#", 2).struct.field("field_1")
    return (
        pl.when(pl.field("module").str.contains("#", literal=True) & (pl.field("names") == ""))
        .then(after)
        .otherwise(pl.field("names"))
    )


def from_block_above(column: str) -> pl.Expr:
    """*column* as the block above has it, where a symbol takes that block; as it is elsewhere."""
    return pl.when("takes").then(pl.col(column).shift(1).over("source")).otherwise(column)


def docs_with_block_above() -> pl.Expr:
    """A symbol's docs, led by the spans of the block it takes."""
    return (
        pl.when("takes")
        .then(pl.concat_list(pl.col("spans").shift(1).over("source"), "docs"))
        .otherwise("docs")
    )


def doc_text() -> pl.Expr:
    """The text of one documentation span; null for none."""
    return text(pl.col("docs").struct.field("start"), pl.col("docs").struct.field("end"))


def joined_docs() -> pl.Expr:
    """A capture's documentation texts joined by newlines; null when it has none."""
    return pl.when(pl.col("doc").is_not_null().any()).then(
        pl.col("doc").drop_nulls().str.join("\n")
    )


def blank_fragment() -> pl.Expr:
    """A fragment with no text but whitespace: no chunk."""
    return (pl.col("kind") == ChunkKind.FRAGMENT.value) & (
        pl.col("content").str.strip_chars() == ""
    )


def without_final_empty_line() -> pl.Expr:
    """A file's lines less the empty one after its final newline."""
    return (
        pl.when(pl.col("lines").list.last() == "")
        .then(pl.col("lines").list.slice(0, pl.col("lines").list.len() - 1))
        .otherwise("lines")
    )


def window_lines(size: int) -> pl.Expr:
    """The *size* lines from a window's start, joined: its text before trimming."""
    return pl.col("lines").list.slice("start", size).list.join("\n")


def window_start_byte() -> pl.Expr:
    """Where a window's first line starts in its file: the lines before it, each with its newline."""
    return (
        pl.col("lines")
        .list.slice(0, "start")
        .list.eval(pl.element().str.len_bytes() + 1)
        .list.sum()
        .cast(pl.Int64)
    )


def fold_comments(
    files: dy.DataFrame[FileRow], captures: dy.LazyFrame[CaptureRow]
) -> dy.LazyFrame[CaptureRow]:
    """*captures* with each source's comments folded into blocks, and the blocks into symbols.

    Comments that start their line and touch, with only whitespace and no
    blank line between, are one block; a comment trailing code is a block
    of its own. A block ends where the furthest of its comments ends. A
    symbol takes the block of line comments flush above it, unless it is
    an import: its span starts where the block does, and the block's
    spans lead its docs. A block that ends within an earlier symbol is
    that symbol's; any other is a comment chunk.
    """
    return (
        # In one chunk: joined from several, polars copies the content into every row.
        captures.join(files.select("file", "content").rechunk().lazy(), on="file")
        .sort("source", "order")
        .with_columns(starts_line=starts_line())
        .with_columns(block=starts_block().cum_sum().over("source"))
        .group_by("source", "block", maintain_order=True)
        .agg(
            pl.exclude("end_byte", "line_end").first(),
            pl.col("end_byte", "line_end").max(),
            spans=pl.struct(start="start_byte", end="end_byte"),
        )
        .with_columns(takes=takes_block_above())
        .with_columns(
            taken=pl.col("takes").shift(-1).over("source").fill_null(False),
            start_byte=from_block_above("start_byte"),
            line_start=from_block_above("line_start"),
            docs=docs_with_block_above(),
        )
        .filter(~pl.col("taken"), ~inside_earlier_symbol())
        .pipe(CaptureRow.validate, cast=True)
    )


def written_names(text: pl.Expr) -> pl.Expr:
    """A callee, base or type written as *text*, as its names: split on `.`, `?.`, `::` and `->`."""
    return text.str.replace_all(r"\s", "").str.replace_all(r"\?\.|::|->", ".").str.split(".")


def all_names(path: pl.Expr) -> pl.Expr:
    """Whether *path* holds names only: letters, digits, `_` and `$`, not starting with a digit."""
    return (path.list.len() > 0) & path.list.eval(
        pl.element().str.contains(r"^(?:[^\W\d]|\$)[\w$]*$")
    ).list.all()


def level_rank() -> pl.Expr:
    """A levelled capture's rank: its stated `level`, else the order in which its `level_style` is first used in its source.

    Over a source's levelled captures.
    """
    first_use = pl.col("order").min().over("file", "source", "level_style")
    return pl.coalesce(
        "level",
        pl.when(pl.col("level_style").is_not_null()).then(
            first_use.rank("dense").over("file", "source")
        ),
    )


def next_levelled_start() -> pl.Expr:
    """Where the next levelled capture of the source starts, else where the source ends.

    Over a source's levelled captures, in order.
    """
    return pl.coalesce(pl.col("start_byte").shift(-1).over("file", "source"), "source_end")


def end_before_trailing_whitespace(stop: pl.Expr) -> pl.Expr:
    """The end of the text from `start_byte` to *stop*, less the whitespace it ends with."""
    return (
        pl.col("start_byte")
        + text(pl.col("start_byte"), stop).str.strip_chars_end(ASCII_WHITESPACE).str.len_bytes()
    )


def last_line_of_span() -> pl.Expr:
    """The line a span's last byte is on, from `line_start` and the newlines before that byte.

    A span ending with a newline ends on the line the newline ends, as
    tree-sitter's `last_line` reads a node ending at a line break.
    """
    span = text(pl.col("start_byte"), pl.col("end_byte"))
    return (
        pl.col("line_start")
        + span.str.count_matches("\n", literal=True)
        - span.str.ends_with("\n").cast(pl.Int64)
    )


def fragment_in_levelled_text() -> pl.Expr:
    """A fragment starting before the nearest levelled capture at or before it ends (`levelled_end`).

    False before a source's first levelled capture. Over a source's rows in
    byte order.
    """
    held_to = pl.col("levelled_end").forward_fill().over("file", "source")
    return (pl.col("kind") == ChunkKind.FRAGMENT) & (pl.col("start_byte") < held_to).fill_null(
        value=False
    )


def levelled(
    captures: dy.LazyFrame[CaptureRow], files: dy.DataFrame[FileRow]
) -> dy.LazyFrame[CaptureRow]:
    """*captures* with each levelled capture's text ended at the next and scoped under the ones above.

    A levelled capture is one whose row states a level: a number
    (`level`), or a style (`level_style`) ranked by the order in which
    it is first used in its source. Within a source, a levelled capture
    holds its text up to the next one, or to its source's end, less
    trailing whitespace. A later capture of the same or a lower rank
    closes it; it is scoped under every earlier one of a lower rank not
    closed before it, outermost first. A fragment starting inside a
    levelled capture's text is that text, and is dropped: a fragment never
    spans a captured node, so it ends before the next levelled capture.
    Other captures are unchanged.
    """
    key = ["file", "source"]
    rows = (
        captures.filter(pl.col("level").is_not_null() | pl.col("level_style").is_not_null())
        .sort(*key, "order")
        .with_columns(rank=level_rank(), stop=next_levelled_start())
    )
    # Each capture's closer: the first later one of the same or a lower rank.
    closers = (
        rows.select(*key, "order", "rank")
        .join(rows.select(*key, later="order", later_rank="rank"), on=key)
        .filter((pl.col("later") > pl.col("order")) & (pl.col("later_rank") <= pl.col("rank")))
        .group_by(*key, "order")
        .agg(closer=pl.col("later").min())
    )
    # Each capture's scope: the earlier ones of a lower rank still open at it.
    scopes = (
        rows.select(*key, "order", "rank")
        .join(
            rows.join(closers, on=[*key, "order"], how="left").select(
                *key, above="order", above_name="name", above_rank="rank", closer="closer"
            ),
            on=key,
        )
        .filter(
            (pl.col("above") < pl.col("order"))
            & (pl.col("above_rank") < pl.col("rank"))
            & (pl.col("closer").is_null() | (pl.col("closer") > pl.col("order")))
        )
        .group_by(*key, "order")
        .agg(chain=pl.col("above_name").sort_by("above").str.join("::"))
    )
    spans = (
        rows.join(files.select("file", "content").rechunk().lazy(), on="file")
        .with_columns(end_byte=end_before_trailing_whitespace(pl.col("stop")))
        .with_columns(line_end=last_line_of_span())
        .join(scopes, on=[*key, "order"], how="left")
        .select(
            *key,
            "order",
            levelled_end="end_byte",
            levelled_line_end="line_end",
            levelled_scope=pl.col("chain").fill_null(""),
        )
    )
    return (
        captures.join(spans, on=[*key, "order"], how="left")
        .with_columns(
            end_byte=pl.coalesce("levelled_end", "end_byte"),
            line_end=pl.coalesce("levelled_line_end", "line_end"),
            scope=pl.coalesce("levelled_scope", "scope"),
        )
        .sort(*key, "start_byte")
        .filter(~fragment_in_levelled_text())
        .sort("file", "order")
        .select(CaptureRow.column_names())
        .pipe(CaptureRow.validate, cast=True)
    )


def with_metadata(
    captures: dy.LazyFrame[CaptureRow], metadata_captures: dy.LazyFrame[MetadataCaptureRow]
) -> dy.LazyFrame[CaptureRow]:
    """*captures* with the metadata *metadata_captures* give them: an import's names and bindings, a chunk's calls and bases.

    A `@_bind_name` and a `@_bind_local` of one match are one binding. An
    import's every match (`from m import a as b, c` matches once per name)
    is one import, keyed by its statement and module, with the names its
    matches list, then its pattern's or resolver's, and the bindings they
    make: a name alone binds under itself, a local alone binds the module.
    A call or base belongs to the innermost capture around it that holds
    references, a base only to a class. A call starting where a `self` or
    `super` marker starts is a `MEMBER` or `BASE` lookup, its first name dropped
    (`self.save` is `self` → `save`), and a base is always a name path; a
    `@_super_call` (Ruby's bare `super`) calls the bases' method of its
    function's own name; a path through a value (`x[0].m`,
    `f().g`) holds a part that is not a name, and is none. A call whose
    head is a local of a function around it (an `@_local`, with its
    match's `@_local_type` as its written type) is on a value, its type
    the written type's names, unless that function defines the name
    itself. Each capture's references are in source order, once each.
    """
    by_match = ["file", "source", "match"]
    statement = ["file", "source", "start_byte", "end_byte", "module"]
    imports = captures.filter(pl.col("kind") == ChunkKind.IMPORT).with_columns(
        module=pl.col("metadata").struct.field("module")
    )
    pairs = (
        metadata_captures.filter(
            pl.col("capture").is_in([MetadataCaptureName.BIND_NAME, MetadataCaptureName.BIND_LOCAL])
        )
        # A binding sits where its local does; one naming alone, where its name does.
        .sort(pl.col("capture") != MetadataCaptureName.BIND_LOCAL)
        .group_by(by_match)
        .agg(
            name=pl.col("text").filter(pl.col("capture") == MetadataCaptureName.BIND_NAME).first(),
            local=pl.col("text")
            .filter(pl.col("capture") == MetadataCaptureName.BIND_LOCAL)
            .first(),
            start_byte=pl.col("start_byte").first(),
            end_byte=pl.col("end_byte").first(),
        )
    )
    # What each import's matches name and bind, gathered per statement.
    parts = pl.concat(
        [
            metadata_captures.filter(pl.col("capture") == MetadataCaptureName.IMPORT_NAME).select(
                *by_match, listed="text", at="start_byte"
            ),
            pairs.select(*by_match, bind_name="name", bind_local="local", at="start_byte"),
        ],
        how="diagonal_relaxed",
    )
    names = pl.col("metadata").struct.field("names")
    hint = pl.col("metadata").struct.field("language_hint")
    gathered = (
        imports.join(parts, on=by_match, how="left")
        .sort("order", "at", nulls_last=True)
        .group_by(statement, maintain_order=True)
        .agg(
            pl.exclude("listed", "bind_name", "bind_local", "at").first(),
            listed=pl.col("listed").drop_nulls().unique(maintain_order=True).str.join(","),
            set_names=names.filter(names != "").first(),
            set_hint=hint.filter(hint != "").first(),
            bound=pl.struct(
                name=pl.coalesce("bind_name", "module"),
                local=pl.coalesce("bind_local", "bind_name", "module"),
            ).filter(pl.col("bind_name").is_not_null() | pl.col("bind_local").is_not_null()),
        )
        .with_columns(
            metadata=pl.col("metadata").struct.with_fields(
                names=pl.when(pl.col("listed") != "")
                .then("listed")
                .otherwise(pl.coalesce("set_names", pl.field("names"))),
                language_hint=pl.coalesce("set_hint", pl.field("language_hint")),
                bindings=pl.col("bound").list.unique(maintain_order=True),
            )
        )
        .drop("module", "listed", "set_names", "set_hint", "bound")
        .select(CaptureRow.column_names())
    )
    markers = (
        metadata_captures.filter(
            pl.col("capture").is_in([MetadataCaptureName.SELF, MetadataCaptureName.SUPER])
        )
        .select(
            "file",
            "source",
            "start_byte",
            lookup=pl.when(pl.col("capture") == MetadataCaptureName.SELF)
            .then(pl.lit(Lookup.MEMBER.value))
            .otherwise(pl.lit(Lookup.BASE.value)),
        )
        .unique()
    )
    holders = captures.filter(holds_references()).select(
        "file",
        "source",
        "order",
        "kind",
        holder_name="name",
        holder_start="start_byte",
        holder_end="end_byte",
    )
    inside = (pl.col("holder_start") <= pl.col("start_byte")) & (
        pl.col("end_byte") <= pl.col("holder_end")
    )
    span = pl.col("holder_end") - pl.col("holder_start")
    # Each function's locals: an `@_local` and its match's `@_local_type`, kept
    # only where the innermost holder around it is a function.
    locals_ = (
        metadata_captures.filter(
            pl.col("capture").is_in([MetadataCaptureName.LOCAL, MetadataCaptureName.LOCAL_TYPE])
        )
        .sort(pl.col("capture") != MetadataCaptureName.LOCAL)
        .group_by(by_match)
        .agg(
            local=pl.col("text").filter(pl.col("capture") == MetadataCaptureName.LOCAL).first(),
            written=pl.col("text")
            .filter(pl.col("capture") == MetadataCaptureName.LOCAL_TYPE)
            .first(),
            start_byte=pl.col("start_byte").first(),
            end_byte=pl.col("end_byte").first(),
        )
        .drop_nulls("local")
        .with_columns(type=written_names(pl.col("written")))
        .with_columns(type=pl.when(all_names(pl.col("type"))).then("type"))
        .with_row_index("binding")
        .join(holders, on=["file", "source"])
        .filter(inside)
        .filter(span == span.min().over("binding"))
        .filter(
            pl.col("kind").is_in([ChunkKind.FUNCTION, ChunkKind.METHOD, ChunkKind.TEST_FUNCTION])
        )
        .select("file", "source", "local", "type", "holder_start", "holder_end")
    )
    defined = captures.filter(pl.col("kind").is_in(DEFINITION_KINDS)).select(
        "file", "source", "name", defined_at="start_byte"
    )
    paths = (
        metadata_captures.filter(
            pl.col("capture").is_in([MetadataCaptureName.CALL, MetadataCaptureName.BASE])
        )
        .join(
            markers.with_columns(
                capture=pl.lit(
                    MetadataCaptureName.CALL.value,
                    dtype=metadata_captures.collect_schema()["capture"],
                )
            ),
            on=["file", "source", "start_byte", "capture"],
            how="left",
        )
        .with_columns(
            pl.col("lookup").fill_null(Lookup.UNQUALIFIED.value),
            names=written_names(pl.col("text")),
        )
        .with_columns(
            path=pl.when(pl.col("lookup") == Lookup.UNQUALIFIED)
            .then("names")
            .otherwise(pl.col("names").list.slice(1))
        )
        .filter(all_names(pl.col("path")))
        .select("file", "source", "capture", "start_byte", "end_byte", "lookup", "path")
    )
    super_calls = metadata_captures.filter(
        pl.col("capture") == MetadataCaptureName.SUPER_CALL
    ).select(
        "file",
        "source",
        capture=pl.lit(
            MetadataCaptureName.CALL.value, dtype=metadata_captures.collect_schema()["capture"]
        ),
        start_byte="start_byte",
        end_byte="end_byte",
        lookup=pl.lit(Lookup.BASE.value),
        path=pl.lit([], dtype=pl.List(pl.String)),
    )
    # Numbered in a fixed order: the plan reads `paths` twice, and each read
    # must give a reference the same number.
    paths = (
        pl.concat([paths, super_calls], how="vertical_relaxed")
        .sort("file", "source", "start_byte", "end_byte", "capture", "lookup")
        .with_row_index("reference")
    )
    # A call whose head is a local of a function around it, unless that
    # function defines the name itself, is on a value: the innermost local wins,
    # one with a written type over one without.
    on_values = (
        paths.filter(
            (pl.col("capture") == MetadataCaptureName.CALL)
            & (pl.col("lookup") == Lookup.UNQUALIFIED)
        )
        .with_columns(head=pl.col("path").list.first())
        .join(locals_, left_on=["file", "source", "head"], right_on=["file", "source", "local"])
        .filter(inside)
        .join(
            defined,
            left_on=["file", "source", "head"],
            right_on=["file", "source", "name"],
            how="left",
        )
        .with_columns(
            defined_inside=(pl.col("holder_start") <= pl.col("defined_at"))
            & (pl.col("defined_at") <= pl.col("holder_end"))
        )
        .group_by("reference", "holder_start", "holder_end", "type")
        .agg(pl.col("defined_inside").any())
        .filter(~pl.col("defined_inside"))
        .sort(span, pl.col("type").is_null())
        .unique("reference", keep="first", maintain_order=True)
        .select("reference", "type", lookup=pl.lit(Lookup.TYPE.value))
    )
    paths = paths.join(on_values, on="reference", how="left", suffix="_value").with_columns(
        lookup=pl.coalesce("lookup_value", "lookup")
    )
    owned = (
        paths.drop("reference")
        .with_row_index("reference")
        .join(holders, on=["file", "source"])
        .filter(
            inside
            & (
                (pl.col("capture") != MetadataCaptureName.BASE)
                | (pl.col("kind") == ChunkKind.CLASS)
            )
        )
        .filter(span == span.min().over("reference"))
        .with_columns(
            path=pl.when(pl.col("path").list.len() > 0)
            .then("path")
            .when(pl.col("kind").is_in([ChunkKind.FUNCTION, ChunkKind.METHOD]))
            .then(pl.concat_list("holder_name"))
        )
        .drop_nulls("path")
        .sort("start_byte", "end_byte")
        .unique(
            ["file", "order", "capture", "lookup", "path", "type"],
            keep="first",
            maintain_order=True,
        )
        .group_by("file", "order", maintain_order=True)
        .agg(
            refs=pl.struct(
                kind=pl.when(pl.col("capture") == MetadataCaptureName.BASE)
                .then(pl.lit(EdgeKind.INHERITS.value))
                .otherwise(pl.lit(EdgeKind.CALLS.value)),
                lookup="lookup",
                path="path",
                type="type",
            )
        )
    )
    return (
        pl.concat([captures.filter(pl.col("kind") != ChunkKind.IMPORT), gathered])
        .join(owned, on=["file", "order"], how="left")
        .with_columns(
            metadata=pl.col("metadata").struct.with_fields(
                refs=pl.when(holds_references()).then(pl.coalesce("refs", no_references()))
            )
        )
        .drop("refs")
        .sort("file", "order")
        .pipe(CaptureRow.validate, cast=True)
    )


def capture_chunks(
    files: dy.DataFrame[FileRow], captures: dy.LazyFrame[CaptureRow]
) -> dy.LazyFrame[PlacedChunkRow]:
    """The chunks *captures* make: comments folded (`fold_comments`), then each made a chunk.

    A capture's text is its span of the file, and its docs the texts of
    its documentation's spans joined by newlines. A function in a class is a method
    (`promoted`), and an import's `#` is read (`module_before_fragment`,
    `fragment_as_names`). A fragment whose text is blank is no chunk.
    """
    return (
        fold_comments(files, captures)
        # In one chunk: joined from several, polars copies the content into every row.
        .join(
            files.select("file", "path", "blob_sha", "content", file_language="language")
            .rechunk()
            .lazy(),
            on="file",
        )
        .with_row_index("row")
        .explode("docs", empty_as_null=True)
        .with_columns(doc=doc_text())
        .group_by("row", maintain_order=True)
        .agg(pl.exclude("docs", "doc").first(), docs=joined_docs())
        .with_columns(
            kind=promoted(),
            metadata=pl.col("metadata").struct.with_fields(
                module=module_before_fragment(), names=fragment_as_names()
            ),
        )
        .with_columns(
            content=text(pl.col("start_byte"), pl.col("end_byte")),
            line_end=last_line_of_span(),
            metadata=chunk_metadata(),
        )
        .filter(~blank_fragment())
        .pipe(PlacedChunkRow.validate, cast=True)
    )


def line_windows(
    files: dy.DataFrame[FileRow], plain: list[int], size: int, overlap: int, language: str
) -> dy.LazyFrame[PlacedChunkRow]:
    """The files numbered in *plain*, as windows of *size* lines, each from *overlap* lines before the last ends.

    A file that ends in a newline has no empty last line. A window's span
    is its lines less whitespace at both ends, and a blank window is no
    chunk. The windows are in *language*, numbered in their file.
    """
    return (
        files.lazy()
        .filter(pl.col("file").is_in(plain))
        .select(
            "file",
            "blob_sha",
            file_language="language",
            lines=pl.col("content").cast(pl.String).str.split("\n"),
        )
        .with_columns(lines=without_final_empty_line())
        .with_columns(start=pl.int_ranges(0, pl.col("lines").list.len(), size - overlap))
        .explode("start", empty_as_null=False)
        .with_columns(raw=window_lines(size), first=window_start_byte())
        .with_columns(trimmed=pl.col("raw").str.strip_chars_start(ASCII_WHITESPACE))
        .with_columns(
            content=pl.col("trimmed").str.strip_chars_end(ASCII_WHITESPACE),
            start_byte=pl.col("first")
            + pl.col("raw").str.len_bytes()
            - pl.col("trimmed").str.len_bytes(),
            line_start=pl.col("start")
            + 1
            + pl.col("raw")
            .str.head(pl.col("raw").str.len_chars() - pl.col("trimmed").str.len_chars())
            .str.count_matches("\n", literal=True),
        )
        .with_columns(
            end_byte=pl.col("start_byte") + pl.col("content").str.len_bytes(),
            line_end=pl.col("line_start") + pl.col("content").str.count_matches("\n", literal=True),
        )
        .filter(pl.col("content") != "")
        .with_columns(
            order=pl.int_range(pl.len()).over("file"),
            language=pl.lit(language),
            kind=pl.lit(ChunkKind.RAW_CHUNK.value),
            name=pl.lit(""),
            scope=pl.lit(""),
            metadata=no_metadata(),
            docs=pl.lit(None, dtype=pl.String),
        )
        .pipe(PlacedChunkRow.validate, cast=True)
    )


class KindOverlapRow(dy.Schema):
    """Captures of more than one kind on one span: the file, the span's first line, the kinds in `ChunkKind` order, the kind its chunk took, and whether `narrows` decided it."""

    path = dy.String(nullable=False)
    line = dy.Int64(nullable=False)
    kinds = dy.List(dy.String(nullable=False), nullable=False)
    kept = dy.String(nullable=False)
    narrowed = dy.Bool(nullable=False)


def kind_rank(kind: pl.Expr) -> pl.Expr:
    """*kind*'s place in `ChunkKind` order."""
    return kind.cast(pl.String).replace_strict(
        [k.value for k in ChunkKind], list(range(len(ChunkKind))), return_dtype=pl.Int64
    )


def one_chunk_per_span(
    files: dy.DataFrame[FileRow], chunks: pl.LazyFrame
) -> tuple[pl.LazyFrame, dy.LazyFrame[KindOverlapRow]]:
    """*chunks* with the chunks of each span of a file made one, and the spans captured as more than one kind.

    A span's chunk takes the kind that narrows every other there
    (`narrows`), else the first in `ChunkKind` order, and its first
    chunk of that kind gives its scope, docs and content; its order is
    the span's first. Its name is every chunk's, in order, once each,
    joined by `, `, but an import's text when it is no import (a key that
    imports is named as the key). Its language is an embedded one if any
    chunk there has one. Its imports are every chunk's, in order, once
    each; its calls and bases, and a fragment's node types, likewise.
    """
    span = ["file", "start_byte", "end_byte"]
    counted = chunks.with_columns(chunks_at_span=pl.len().over(span))
    single = counted.filter(pl.col("chunks_at_span") == 1).drop("chunks_at_span")
    shared = counted.filter(pl.col("chunks_at_span") > 1).drop("chunks_at_span")
    allowed = pl.LazyFrame(
        {
            "kept": [kind.value for kind in ChunkKind],
            "allowed": [
                sorted({kind.value, *(other.value for other in ChunkKind if narrows(kind, other))})
                for kind in ChunkKind
            ],
        }
    )
    kinds = (
        shared.group_by(span)
        .agg(kinds=pl.col("kind").cast(pl.String).unique(), line=pl.col("line_start").min())
        .with_columns(
            kinds=pl.col("kinds").list.eval(pl.element().sort_by(kind_rank(pl.element())))
        )
    )
    choice = (
        kinds.with_columns(kept=pl.col("kinds"))
        .explode("kept")
        .join(allowed, on="kept")
        .with_columns(
            narrowed=pl.col("kinds").list.set_difference("allowed").list.len() == 0,
            rank=kind_rank(pl.col("kept")),
        )
        .sort("narrowed", "rank", descending=[True, False])
        .group_by(span)
        .agg(pl.col("kept", "narrowed", "kinds", "line").first())
    )
    of_kept = pl.col("kind").cast(pl.String) == pl.col("kept")
    every = {"keep_nulls": False, "empty_as_null": False}
    imports = pl.col("metadata").struct.field("imports").list.explode(**every)
    refs = pl.col("metadata").struct.field("refs").list.explode(**every)
    node_types = pl.col("metadata").struct.field("node_types").list.explode(**every)
    merged = (
        shared.join(choice.select(*span, "kept"), on=span)
        .sort("file", "order")
        .group_by(span, maintain_order=True)
        .agg(
            pl.exclude(*span, "kept", "order", "language", "metadata", "name")
            .filter(of_kept)
            .first(),
            name=pl.col("name")
            .filter(
                (pl.col("name") != "")
                & (
                    (pl.col("kind").cast(pl.String) != ChunkKind.IMPORT.value)
                    | (pl.col("kept") == ChunkKind.IMPORT.value)
                )
            )
            .unique(maintain_order=True)
            .str.join(", "),
            order=pl.col("order").min(),
            language=pl.coalesce(
                pl.col("language").filter(pl.col("language") != pl.col("file_language")).first(),
                pl.col("language").first(),
            ),
            imports=imports.unique(maintain_order=True),
            refs=refs.unique(maintain_order=True),
            node_types=node_types.unique(maintain_order=True),
        )
        .with_columns(
            metadata=pl.struct(
                node_types=pl.when(pl.col("kind") == ChunkKind.FRAGMENT.value).then("node_types"),
                refs=pl.when(holds_references()).then("refs"),
                imports="imports",
            )
        )
        .drop("imports", "refs", "node_types")
    )
    overlaps = (
        choice.filter(pl.col("kinds").list.len() > 1)
        .join(files.select("file", "path").lazy(), on="file")
        .sort("path", "line")
        .select(KindOverlapRow.column_names())
        .pipe(KindOverlapRow.validate, cast=True)
    )
    return pl.concat([single, merged], how="diagonal_relaxed"), overlaps


def assemble(
    files: dy.DataFrame[FileRow], *placed: dy.LazyFrame[PlacedChunkRow]
) -> tuple[dy.LazyFrame[PlacedChunkRow], dy.LazyFrame[KindOverlapRow]]:
    """The batch's chunks, each file's in order, *placed* and presence chunks, one per span; and the spans captured as more than one kind.

    The chunks of one span are one (`one_chunk_per_span`). A file with no
    chunk in its own language gets a content-less presence chunk in it,
    after its others, so the blob check can skip the file on later
    builds. Each file's chunks are numbered again from 0.
    """
    chunks, overlaps = one_chunk_per_span(files, pl.concat(placed))
    presence = (
        files.lazy()
        .join(chunks.filter(pl.col("language") == pl.col("file_language")), on="file", how="anti")
        .select(
            "file",
            "blob_sha",
            "language",
            file_language="language",
            kind=pl.lit(ChunkKind.RAW_CHUNK.value),
            name=pl.lit(""),
            scope=pl.lit(""),
            content=pl.lit(""),
            start_byte=pl.lit(0),
            end_byte=pl.lit(0),
            line_start=pl.lit(1),
            line_end=pl.lit(1),
            metadata=no_metadata(),
        )
    )
    return (
        pl.concat([chunks, presence], how="diagonal_relaxed")
        .sort("file", "order", nulls_last=True)
        .with_columns(order=pl.int_range(pl.len()).over("file"))
        .pipe(PlacedChunkRow.validate, cast=True)
    ), overlaps
