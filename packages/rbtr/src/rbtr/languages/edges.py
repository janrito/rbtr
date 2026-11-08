"""Edge inference: the chunks each import reaches.

`infer_import_edges` takes a snapshot's chunks as a frame and returns
an edge from each import to every chunk it reaches.  It reads only the
chunks and the repository's file list; the build writes the edges.

Each import takes one `ImportWay`:

- `MODULE`: its metadata names a module or a relative depth, which
  `languages.resolution` turns into files.  An import naming symbols
  (`from x import Y`, a link to `doc.md#heading`) reaches each named
  symbol in those files, or else a doc section of that name; one
  naming none (`import x`, `#include "x.h"`) reaches every chunk the
  files bring into scope.
- `NAME`: prose naming a symbol and no file (an rST `:func:` role)
  reaches the first symbol of that name anywhere.
- `TEXT`: an import whose extractor recorded nothing usable; each word
  of its text that is a file's stem reaches that file's first symbol.
  Imprecise, and the only way left.

Prose documents what it references, so its `MODULE` and `NAME` edges
are `DOCUMENTS`; every other edge is `IMPORTS`.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import PurePosixPath

import dataframely as dy
import polars as pl

from rbtr.domain.models import ChunkKind, Edge, EdgeKind, Edges, ImportMeta
from rbtr.languages.resolution import ImportResolution, import_files


class EdgeChunkRow(dy.Schema):
    """A chunk as edge inference reads it: where it is, what it is, and what it imports.

    The import fields hold an import's metadata and are null on every
    other kind.
    """

    id = dy.String(nullable=False)
    file_path = dy.String(nullable=False)
    line_start = dy.Int32(nullable=False)
    kind = dy.Enum(k.value for k in ChunkKind)
    name = dy.String(nullable=False)
    language = dy.String(nullable=False)
    module = dy.String(nullable=True)
    names = dy.String(nullable=True)
    dots = dy.String(nullable=True)
    language_hint = dy.String(nullable=True)


class ImportWay(StrEnum):
    """How an import reaches what it imports."""

    MODULE = "module"
    """Its metadata names a module or a relative depth: it reaches the module's files."""
    NAME = "name"
    """Prose naming a symbol and no file (rST `:func:`): it reaches that symbol."""
    TEXT = "text"
    """Neither: its words that are file stems reach those files."""


class ImportRow(dy.Schema):
    """An import, the way it reaches its targets, and the edge kind it makes.

    `target_paths` are the files its module resolves to, null unless
    its way is `MODULE`.
    """

    source_id = dy.String(nullable=False)
    source_path = dy.String(nullable=False)
    name = dy.String(nullable=False)
    names = dy.String(nullable=False)
    way = dy.Enum(w.value for w in ImportWay)
    kind = dy.Enum(k.value for k in EdgeKind)
    target_paths = dy.List(dy.String(), nullable=True)


class InferredEdgeRow(dy.Schema):
    """An edge one way of resolving imports infers: an import to a chunk it reaches."""

    source_id = dy.String(nullable=False)
    target_id = dy.String(nullable=False)
    kind = dy.Enum(k.value for k in EdgeKind)
    source_path = dy.String(nullable=False)
    target_path = dy.String(nullable=False)


def _named_by_imports() -> pl.Expr:
    """Chunks an import can name: a definition, or a data language's key.

    A `$ref` into a schema names a key, as an import names a function.
    """
    return pl.col("kind").is_in(
        [
            ChunkKind.FUNCTION,
            ChunkKind.CLASS,
            ChunkKind.METHOD,
            ChunkKind.VARIABLE,
            ChunkKind.CONFIG_KEY,
        ]
    )


def _brought_into_scope() -> pl.Expr:
    """Chunks a bare import brings into scope: a file's definitions, sections and raw slices.

    A raw slice includes the content-less chunk of an empty file, so
    `import pkg` reaches an empty `__init__.py`.
    """
    return pl.col("kind").is_in(
        [
            ChunkKind.FUNCTION,
            ChunkKind.CLASS,
            ChunkKind.METHOD,
            ChunkKind.VARIABLE,
            ChunkKind.MIGRATION,
            ChunkKind.TEST_FUNCTION,
            ChunkKind.API_ENDPOINT,
            ChunkKind.DOC_SECTION,
            ChunkKind.CONFIG_KEY,
            ChunkKind.RAW_CHUNK,
        ]
    )


def _way(prose: pl.Expr) -> pl.Expr:
    """The `ImportWay` of each import row; *prose* holds for an import written in prose."""
    return (
        pl.when((pl.col("module") != "") | (pl.col("dots") != ""))
        .then(pl.lit(ImportWay.MODULE))
        .when((pl.col("names") != "") & prose)
        .then(pl.lit(ImportWay.NAME))
        .otherwise(pl.lit(ImportWay.TEXT))
        .alias("way")
    )


def _edge_kind(prose: pl.Expr) -> pl.Expr:
    """Prose documents what it references; code imports it."""
    return (
        pl.when(prose)
        .then(pl.lit(EdgeKind.DOCUMENTS))
        .otherwise(pl.lit(EdgeKind.IMPORTS))
        .alias("kind")
    )


def _imports(
    chunks: dy.DataFrame[EdgeChunkRow],
    prose: pl.Expr,
    repo_files: set[str],
    rules: dict[str, ImportResolution],
) -> dy.DataFrame[ImportRow]:
    """*chunks*' imports, each with its way, its edge kind and, for a `MODULE` import, its module's files.

    A module resolves to files of *repo_files* with `import_files`, by
    the rules *rules* holds for each language.
    """
    imports = chunks.filter(pl.col("kind") == ChunkKind.IMPORT).with_columns(_way(prose))
    by_module = imports.filter(pl.col("way") == ImportWay.MODULE).with_columns(
        pl.struct(pl.all())
        .map_elements(
            lambda row: import_files(
                ImportMeta.model_validate(row), row["file_path"], row["language"], repo_files, rules
            ),
            return_dtype=pl.List(pl.String),
        )
        .alias("target_paths")
    )
    return (
        pl.concat([by_module, imports.filter(pl.col("way") != ImportWay.MODULE)], how="diagonal")
        .select(
            pl.col("id").alias("source_id"),
            pl.col("file_path").alias("source_path"),
            "name",
            "names",
            "way",
            _edge_kind(prose),
            "target_paths",
        )
        .pipe(ImportRow.validate, cast=True)
    )


def _module_files(imports: dy.DataFrame[ImportRow]) -> pl.DataFrame:
    """The `MODULE` imports, one row per file their module resolves to, as `target_path`."""
    return (
        imports.filter(pl.col("way") == ImportWay.MODULE)
        .explode("target_paths")
        .rename({"target_paths": "target_path"})
    )


def _named(imports: pl.DataFrame) -> pl.DataFrame:
    """*imports*, one row per name each one names, as `target_name`."""
    return imports.with_columns(pl.col("names").str.split(",").alias("target_name")).explode(
        "target_name"
    )


def _bare_edges(
    imports: dy.DataFrame[ImportRow], importable: dy.DataFrame[EdgeChunkRow]
) -> dy.DataFrame[InferredEdgeRow]:
    """A bare import (`import x`, `#include "x.h"`) reaches every chunk its files bring into scope."""
    return (
        _module_files(imports)
        .filter(pl.col("names") == "")
        .join(
            importable.select(
                pl.col("file_path").alias("target_path"), pl.col("id").alias("target_id")
            ),
            on="target_path",
        )
        .pipe(InferredEdgeRow.validate, cast=True)
    )


def _named_edges(
    imports: dy.DataFrame[ImportRow],
    symbols: dy.DataFrame[EdgeChunkRow],
    sections: dy.DataFrame[EdgeChunkRow],
) -> dy.DataFrame[InferredEdgeRow]:
    """A named import (`from X import Y`) reaches each named symbol in its files.

    A name no symbol there has reaches a doc section of that name
    instead (`doc.md#a-heading`).
    """

    def by_name(targets: dy.DataFrame[EdgeChunkRow], alias: str) -> pl.DataFrame:
        return targets.select(
            pl.col("file_path").alias("target_path"),
            pl.col("name").alias("target_name"),
            pl.col("id").alias(alias),
        )

    keys = ["target_path", "target_name"]
    return (
        _named(_module_files(imports).filter(pl.col("names") != ""))
        .join(by_name(symbols, "symbol_id"), on=keys, how="left")
        .join(by_name(sections, "section_id"), on=keys, how="left")
        .with_columns(target_id=pl.coalesce("symbol_id", "section_id"))
        .drop_nulls("target_id")
        .pipe(InferredEdgeRow.validate, cast=True)
    )


def _name_edges(
    imports: dy.DataFrame[ImportRow], symbols: dy.DataFrame[EdgeChunkRow]
) -> dy.DataFrame[InferredEdgeRow]:
    """A `NAME` import documents the first symbol of each name it names, wherever it is."""
    first_by_name = symbols.unique("name", keep="first", maintain_order=True)
    return (
        _named(imports.filter(pl.col("way") == ImportWay.NAME))
        .join(
            first_by_name.select(
                pl.col("name").alias("target_name"),
                pl.col("id").alias("target_id"),
                pl.col("file_path").alias("target_path"),
            ),
            on="target_name",
        )
        .with_columns(kind=pl.lit(EdgeKind.DOCUMENTS))
        .pipe(InferredEdgeRow.validate, cast=True)
    )


def _text_edges(
    imports: dy.DataFrame[ImportRow],
    symbols: dy.DataFrame[EdgeChunkRow],
    repo_files: set[str],
    index_stems: set[str],
) -> dy.DataFrame[InferredEdgeRow]:
    """A `TEXT` import reaches the first symbol of each file of *repo_files* whose stem is one of its words.

    No word reaches a stem of *index_stems*, the files languages declare
    as a directory's stand-in (`__init__`, `index`, `mod`), which name
    their directory rather than a module; nor a stem shorter than three
    characters (`a`, `io`), which too many words would match.
    """
    paths = sorted(repo_files)
    stems = pl.DataFrame(
        {"target_path": paths, "word": [PurePosixPath(path).stem for path in paths]},
        schema={"target_path": pl.String, "word": pl.String},
    ).filter(~pl.col("word").is_in(index_stems) & (pl.col("word").str.len_chars() >= 3))
    first_by_file = symbols.unique("file_path", keep="first", maintain_order=True)
    return (
        imports.filter(pl.col("way") == ImportWay.TEXT)
        .with_columns(
            pl.col("name").str.extract_all(r"[A-Za-z_]\w*").list.unique().alias("word"),
            kind=pl.lit(EdgeKind.IMPORTS),
        )
        .explode("word")
        .join(stems, on="word")
        .join(
            first_by_file.select(
                pl.col("file_path").alias("target_path"), pl.col("id").alias("target_id")
            ),
            on="target_path",
        )
        .pipe(InferredEdgeRow.validate, cast=True)
    )


def infer_import_edges(
    chunks: dy.DataFrame[EdgeChunkRow],
    repo_files: set[str],
    resolution_map: dict[str, ImportResolution] | None = None,
) -> list[Edge]:
    """Infer the edges *chunks*' imports make to the chunks they reach.

    Each import takes one `ImportWay`, resolved by `_bare_edges` and
    `_named_edges` (`MODULE`), `_name_edges` and `_text_edges`. A
    module resolves to files with `import_files`, by the rules
    *resolution_map* holds for each language.

    Where an import could reach several chunks and reaches one, it
    reaches the first by file path, then line.
    """
    chunks = chunks.sort("file_path", "line_start", maintain_order=True).pipe(EdgeChunkRow.validate)
    symbols = (
        chunks.filter(_named_by_imports())
        .unique(["file_path", "name"], keep="first", maintain_order=True)
        .pipe(EdgeChunkRow.validate)
    )
    sections = (
        chunks.filter(pl.col("kind") == ChunkKind.DOC_SECTION)
        .unique(["file_path", "name"], keep="first", maintain_order=True)
        .pipe(EdgeChunkRow.validate)
    )
    importable = chunks.filter(_brought_into_scope()).pipe(EdgeChunkRow.validate)
    rules = resolution_map or {}
    prose = pl.col("language").is_in([language for language, r in rules.items() if r.prose])
    index_stems = {PurePosixPath(f).stem for r in rules.values() for f in r.index_files}
    imports = _imports(chunks, prose, repo_files, rules)
    edges = pl.concat(
        [
            _bare_edges(imports, importable),
            _named_edges(imports, symbols, sections),
            _name_edges(imports, symbols),
            _text_edges(imports, symbols, repo_files, index_stems),
        ]
    )
    return Edges.validate_python(edges.to_dicts())
