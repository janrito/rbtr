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
- `NAME`: prose naming something and no file (an rST `:func:` or
  `:ref:` role) reaches the first chunk of that name in the language its
  `language_hint` says holds it, as a module says where a Python import
  finds its name; with no hint, the first symbol of that name anywhere.
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

from rbtr.domain.models import ChunkKind, EdgeKind, ImportMeta
from rbtr.languages.registration import SectionNames
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
    """Prose naming something and no file (rST `:func:`, `:ref:`): it reaches that chunk."""
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
    language_hint = dy.String(nullable=False)
    way = dy.Enum(w.value for w in ImportWay)
    kind = dy.Enum(k.value for k in EdgeKind)
    target_paths = dy.List(dy.String(), nullable=True)


class SectionKeyRow(dy.Schema):
    """A doc section with the key a reference reaches it by, and whether that key is folded.

    One row per file and key: of two sections a reference could reach,
    the earlier line.
    """

    id = dy.String(nullable=False)
    file_path = dy.String(nullable=False)
    line_start = dy.Int32(nullable=False)
    language = dy.String(nullable=False)
    key = dy.String(nullable=False)
    folded = dy.Bool(nullable=False)


class InferredEdgeRow(dy.Schema):
    """An edge: an import chunk at one location reaching a chunk at another.

    A chunk is content and sits at as many paths as hold those bytes, so
    the paths are part of the relationship: `docx/helpers.py` importing
    its sibling and `xlsx/helpers.py` importing a different one are two
    edges from a single import chunk, distinguished by where each end
    sits. `find-refs` reports the location that did the referring.
    """

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
            pl.col("language_hint").fill_null(""),
            "way",
            _edge_kind(prose),
            "target_paths",
        )
        .pipe(ImportRow.validate, cast=True)
    )


def _listed_names() -> pl.Expr:
    """The names an import lists (`from x import a, b`), as a list to explode into one per row."""
    return pl.col("names").str.split(",")


def _folded(name: pl.Expr) -> pl.Expr:
    """*name* as `SectionNames.FOLDED` compares it: letters and digits, lowercase, no accents.

    Leading digits do not count either: renderers drop them from a
    heading's anchor (`1. Intro` is linked as `#intro`).
    """
    return (
        name.str.normalize("NFKD")
        .str.to_lowercase()
        .str.replace_all(r"[^\p{L}\p{N}]", "")
        .str.replace(r"^\p{N}+", "")
    )


def _folds(folding: list[str]) -> pl.Expr:
    """Whether a section's language, one of *folding*, declares `SectionNames.FOLDED`."""
    return pl.col("language").is_in(folding)


def _section_key(folding: list[str]) -> pl.Expr:
    """The key a reference reaches a section by: its name, folded when its language folds."""
    return pl.when(_folds(folding)).then(_folded(pl.col("name"))).otherwise(pl.col("name"))


def _reference_keys() -> pl.Expr:
    """The keys a reference's `target_name` can match a section by: as written, and folded.

    Each is a `{key, folded}` struct, to explode into one row each and
    join against a section's `key` and `folded`.
    """
    return pl.concat_list(
        pl.struct(key=pl.col("target_name"), folded=pl.lit(value=False)),
        pl.struct(key=_folded(pl.col("target_name")), folded=pl.lit(value=True)),
    )


def _bare_edges(
    imports: dy.DataFrame[ImportRow], importable: dy.DataFrame[EdgeChunkRow]
) -> dy.DataFrame[InferredEdgeRow]:
    """A bare import (`import x`, `#include "x.h"`) reaches every chunk its files bring into scope."""
    return (
        imports.filter((pl.col("way") == ImportWay.MODULE) & (pl.col("names") == ""))
        .explode("target_paths")
        .rename({"target_paths": "target_path"})
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
    sections: dy.DataFrame[SectionKeyRow],
) -> dy.DataFrame[InferredEdgeRow]:
    """A named import (`from X import Y`) reaches each named symbol in its files.

    A name no symbol there has reaches the doc section whose key equals
    it, folded when the section's is (`doc.md#a-heading`).
    """
    named = (
        imports.filter((pl.col("way") == ImportWay.MODULE) & (pl.col("names") != ""))
        .explode("target_paths")
        .rename({"target_paths": "target_path"})
        .with_columns(target_name=_listed_names())
        .explode("target_name")
    )
    reached = ["source_id", "target_path", "target_name"]
    by_section = (
        named.with_columns(reference=_reference_keys())
        .explode("reference")
        .unnest("reference")
        .join(
            sections.select(
                pl.col("file_path").alias("target_path"),
                "key",
                "folded",
                pl.col("line_start").alias("section_line"),
                pl.col("id").alias("section_id"),
            ),
            on=["target_path", "key", "folded"],
        )
        .sort("section_line")
        .unique(reached, keep="first", maintain_order=True)
        .select(*reached, "section_id")
    )
    return (
        named.join(
            symbols.select(
                pl.col("file_path").alias("target_path"),
                pl.col("name").alias("target_name"),
                pl.col("id").alias("symbol_id"),
            ),
            on=["target_path", "target_name"],
            how="left",
        )
        .join(by_section, on=reached, how="left")
        .with_columns(target_id=pl.coalesce("symbol_id", "section_id"))
        .drop_nulls("target_id")
        .pipe(InferredEdgeRow.validate, cast=True)
    )


def _name_edges(
    imports: dy.DataFrame[ImportRow],
    symbols: dy.DataFrame[EdgeChunkRow],
    sections: dy.DataFrame[SectionKeyRow],
) -> dy.DataFrame[InferredEdgeRow]:
    """A `NAME` import documents the first chunk of each name it names, in its hinted language.

    The first symbol of that name in the language its `language_hint`
    names, or else the first section whose key equals it (an rST `:ref:`
    hinted `rst`); with no hint, the first symbol of that name anywhere.
    First by file, then line.
    """
    named = (
        imports.filter(pl.col("way") == ImportWay.NAME)
        .with_columns(target_name=_listed_names())
        .explode("target_name")
    )
    reached = ["source_id", "target_name"]
    by_symbol = (
        pl.concat(
            [
                symbols.select("language", "name", "id", "file_path"),
                symbols.select(pl.lit("").alias("language"), "name", "id", "file_path"),
            ]
        )
        .unique(["language", "name"], keep="first", maintain_order=True)
        .select(
            pl.col("language").alias("language_hint"),
            pl.col("name").alias("target_name"),
            pl.col("id").alias("symbol_id"),
            pl.col("file_path").alias("symbol_path"),
        )
    )
    by_section = (
        named.filter(pl.col("language_hint") != "")
        .with_columns(reference=_reference_keys())
        .explode("reference")
        .unnest("reference")
        .join(
            sections.select(
                pl.col("language").alias("language_hint"),
                "key",
                "folded",
                pl.col("id").alias("section_id"),
                pl.col("file_path").alias("section_path"),
                pl.col("line_start").alias("section_line"),
            ),
            on=["language_hint", "key", "folded"],
        )
        .sort("section_path", "section_line")
        .unique(reached, keep="first", maintain_order=True)
        .select(*reached, "section_id", "section_path")
    )
    return (
        named.join(by_symbol, on=["language_hint", "target_name"], how="left")
        .join(by_section, on=reached, how="left")
        .with_columns(
            target_id=pl.coalesce("symbol_id", "section_id"),
            target_path=pl.coalesce("symbol_path", "section_path"),
            kind=pl.lit(EdgeKind.DOCUMENTS),
        )
        .drop_nulls("target_id")
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
) -> dy.DataFrame[InferredEdgeRow]:
    """Infer the edges *chunks*' imports make to the chunks they reach.

    Each import takes one `ImportWay`, resolved by `_bare_edges` and
    `_named_edges` (`MODULE`), `_name_edges` and `_text_edges`. A doc
    section is reached by its key (`SectionKeyRow`), folded when its
    language declares `SectionNames.FOLDED`. A
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
    rules = resolution_map or {}
    folding = [lang for lang, r in rules.items() if r.section_names is SectionNames.FOLDED]
    sections = (
        chunks.filter(pl.col("kind") == ChunkKind.DOC_SECTION)
        .select(
            "id",
            "file_path",
            "line_start",
            "language",
            key=_section_key(folding),
            folded=_folds(folding),
        )
        .filter(pl.col("key") != "")
        .unique(["file_path", "key", "folded"], keep="first", maintain_order=True)
        .pipe(SectionKeyRow.validate, cast=True)
    )
    importable = chunks.filter(_brought_into_scope()).pipe(EdgeChunkRow.validate)
    prose = pl.col("language").is_in([language for language, r in rules.items() if r.prose])
    index_stems = {PurePosixPath(f).stem for r in rules.values() for f in r.index_files}
    imports = _imports(chunks, prose, repo_files, rules)
    edges = pl.concat(
        [
            _bare_edges(imports, importable),
            _named_edges(imports, symbols, sections),
            _name_edges(imports, symbols, sections),
            _text_edges(imports, symbols, repo_files, index_stems),
        ]
    )
    return edges.pipe(InferredEdgeRow.validate)
