"""Edge inference: the chunks a batch of imports reaches.

The entry point is `infer_edges`. It runs three steps over every import
of a batch at once, each a polars process with a typed result:

1. `requested_imports`: import chunks → `ImportRow`. Classifies each
   import's `ImportWay` and edge kind and normalises its fields.
2. `resolution.resolved_files`: imports → `FileMatchRow`. The files
   each `MODULE` or `TEXT` import reaches.
3. `reached_targets`: imports, their files and the snapshot's chunks →
   `InferredEdgeRow`. The chunks each import reaches in those files, or
   by name in its hinted language.

The functions returning `pl.Expr` are the column calculations these steps use.
"""

from __future__ import annotations

import dataframely as dy
import polars as pl

from rbtr.domain.edge_frames import (
    EdgeChunkRow,
    FileMatchRow,
    ImportRow,
    ImportWay,
    InferredEdgeRow,
    LanguageResolutionRow,
    SectionNames,
)
from rbtr.domain.models import ChunkKind, EdgeKind
from rbtr.domain.resolution import resolved_files, written_depth, written_module


def named_by_imports() -> pl.Expr:
    """Whether a chunk is a definition or data key an import can name."""
    return pl.col("kind").is_in(
        [
            ChunkKind.FUNCTION,
            ChunkKind.CLASS,
            ChunkKind.METHOD,
            ChunkKind.VARIABLE,
            ChunkKind.CONFIG_KEY,
        ]
    )


def brought_into_scope() -> pl.Expr:
    """Whether a bare import brings a chunk into scope, including an empty file's raw chunk."""
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


def import_way(prose: pl.Expr) -> pl.Expr:
    """Classify a request by its module, prose names, or remaining text."""
    return (
        pl.when(pl.col("module") != "")
        .then(pl.lit(ImportWay.MODULE))
        .when((pl.col("names") != "") & prose)
        .then(pl.lit(ImportWay.NAME))
        .otherwise(pl.lit(ImportWay.TEXT))
    )


def import_edge_kind(prose: pl.Expr) -> pl.Expr:
    """Prose module/name requests document targets; text fallback and code import them."""
    return (
        pl.when(prose & (pl.col("way") != ImportWay.TEXT))
        .then(pl.lit(EdgeKind.DOCUMENTS))
        .otherwise(pl.lit(EdgeKind.IMPORTS))
    )


def listed_names() -> pl.Expr:
    """The extracted comma-separated names as a list, empty for a bare import.

    `*` names every name (`from lib import *`), so it reaches what a bare
    import reaches.
    """
    return (
        pl.when(pl.col("names").is_in(["", "*"]))
        .then(pl.lit([], dtype=pl.List(pl.String)))
        .otherwise(pl.col("names").str.split(","))
    )


def folded_section_name(name: pl.Expr) -> pl.Expr:
    """Compare section names as lowercase letters/digits without accents or leading digits."""
    return (
        name.str.normalize("NFKD")
        .str.to_lowercase()
        .str.replace_all(r"[^\p{L}\p{N}]", "")
        .str.replace(r"^\p{N}+", "")
    )


def folds() -> pl.Expr:
    """Whether a resolution row's language compares section names folded."""
    return pl.col("section_names") == SectionNames.FOLDED


def section_key() -> pl.Expr:
    """A section's name, folded when its language `folds` names."""
    return (
        pl.when(pl.col("folds")).then(folded_section_name(pl.col("name"))).otherwise(pl.col("name"))
    )


def target_name_keys() -> pl.Expr:
    """The exact and folded comparison keys of an exploded target name."""
    return pl.concat_list(
        pl.struct(key=pl.col("target_name"), folded=pl.lit(value=False)),
        pl.struct(key=folded_section_name(pl.col("target_name")), folded=pl.lit(value=True)),
    )


def requested_imports(
    chunks: dy.DataFrame[EdgeChunkRow], resolutions: dy.DataFrame[LanguageResolutionRow]
) -> dy.DataFrame[ImportRow]:
    """The imports among *chunks*, one row each.

    Each keeps its source id, path and language and its text. Its way is
    `MODULE` when it names a module or a relative depth, `NAME` when prose
    names something and no file, else `TEXT`. Prose documents what it names
    (`DOCUMENTS`) by `MODULE` or `NAME`; every other edge `IMPORTS`. Names
    are split into a list; a relative module's prefix is read as its depth
    (`written_depth`, `written_module`).
    """
    if chunks.is_empty():
        return ImportRow.create_empty()
    prose = pl.col("prose").fill_null(value=False)
    return (
        chunks.filter(pl.col("kind") == ChunkKind.IMPORT)
        .join(
            resolutions.select("language", "prose"),
            on="language",
            how="left",
        )
        .with_columns(pl.col("module", "names", "language_hint").fill_null(""))
        .with_columns(way=import_way(prose))
        .select(
            source_id="id",
            source_path="file_path",
            source_language="language",
            text="name",
            way="way",
            kind=import_edge_kind(prose),
            depth=written_depth(pl.col("module")),
            module=written_module(pl.col("module")),
            names=listed_names(),
            language_hint="language_hint",
        )
        .pipe(ImportRow.validate, cast=True)
    )


def reached_targets(
    imports: dy.DataFrame[ImportRow],
    files: dy.DataFrame[FileMatchRow],
    chunks: dy.DataFrame[EdgeChunkRow],
    resolutions: dy.DataFrame[LanguageResolutionRow],
) -> dy.DataFrame[InferredEdgeRow]:
    """The chunks of *chunks* each import reaches, at each location it reaches them.

    An import's scope is chosen before any ranking, so an earlier chunk
    outside that scope cannot hide one inside it:

    - `MODULE` naming nothing (`import x`): every chunk its files bring into
      scope (`brought_into_scope`).
    - `MODULE` naming names (`from x import y`, `doc.md#heading`): in each
      of its files, the first symbol of each name by line, else the first
      section whose key matches it.
    - `NAME`: the first symbol of each name, by file then line, in its hinted
      language, else the first section there; unhinted, the first symbol of
      that name in any language.
    - `TEXT`: the first symbol of each file its words matched.
    """
    if imports.is_empty():
        return InferredEdgeRow.create_empty()
    location = ["source_id", "source_path"]
    columns = list(InferredEdgeRow.create_empty().columns)
    ordered = chunks.sort("file_path", "line_start", maintain_order=True)
    symbols = ordered.filter(named_by_imports()).select(
        target_path="file_path",
        target_name="name",
        target_id="id",
        target_language="language",
        target_line="line_start",
    )
    sections = (
        ordered.filter(pl.col("kind") == ChunkKind.DOC_SECTION)
        .join(resolutions.select("language", folds=folds()), on="language", how="left")
        .with_columns(pl.col("folds").fill_null(value=False))
        .select(
            target_path="file_path",
            target_id="id",
            target_language="language",
            target_line="line_start",
            key=section_key(),
            folded="folds",
        )
        .filter(pl.col("key") != "")
    )
    in_files = imports.join(files, on=location)
    bare = (
        in_files.filter((pl.col("way") == ImportWay.MODULE) & (pl.col("names").list.len() == 0))
        .join(
            ordered.filter(brought_into_scope()).select(target_path="file_path", target_id="id"),
            on="target_path",
        )
        .select(columns)
    )
    named_files = (
        in_files.filter((pl.col("way") == ImportWay.MODULE) & (pl.col("names").list.len() > 0))
        .with_columns(target_name=pl.col("names"))
        .explode("target_name")
    )
    file_symbols = named_files.join(symbols, on=["target_path", "target_name"]).with_columns(
        priority=pl.lit(0)
    )
    file_sections = (
        named_files.with_columns(key=target_name_keys())
        .explode("key")
        .unnest("key")
        .join(sections, on=["target_path", "key", "folded"])
        .with_columns(priority=pl.lit(1))
    )
    file_named = (
        pl.concat([file_symbols, file_sections], how="diagonal_relaxed")
        .sort("priority", "target_line", maintain_order=True)
        .unique([*location, "target_path", "target_name"], keep="first", maintain_order=True)
        .select(columns)
    )
    named = (
        imports.filter(pl.col("way") == ImportWay.NAME)
        .with_columns(target_name=pl.col("names"))
        .explode("target_name")
    )
    name_symbols = (
        named.join(symbols, on="target_name")
        .filter(
            (pl.col("language_hint") == "") | (pl.col("language_hint") == pl.col("target_language"))
        )
        .with_columns(priority=pl.lit(0))
    )
    name_sections = (
        named.filter(pl.col("language_hint") != "")
        .with_columns(key=target_name_keys())
        .explode("key")
        .unnest("key")
        .join(sections, on=["key", "folded"])
        .filter(pl.col("language_hint") == pl.col("target_language"))
        .with_columns(priority=pl.lit(1))
    )
    by_name = (
        pl.concat([name_symbols, name_sections], how="diagonal_relaxed")
        .sort("priority", "target_path", "target_line", maintain_order=True)
        .unique([*location, "target_name"], keep="first", maintain_order=True)
        .select(columns)
    )
    text = (
        in_files.filter(pl.col("way") == ImportWay.TEXT)
        .join(symbols.unique("target_path", keep="first", maintain_order=True), on="target_path")
        .select(columns)
    )
    return pl.concat([bare, file_named, by_name, text]).pipe(InferredEdgeRow.validate, cast=True)


def infer_edges(
    chunks: dy.DataFrame[EdgeChunkRow],
    repo_files: set[str],
    resolutions: dy.DataFrame[LanguageResolutionRow],
) -> dy.DataFrame[InferredEdgeRow]:
    """The edges the import chunks of a snapshot's *chunks* make to its other chunks.

    *repo_files* are the snapshot's paths and *resolutions* each
    language's resolution. A snapshot with no imports gives an empty frame.
    The module docstring lists the steps.
    """
    imports = requested_imports(chunks, resolutions)
    files = resolved_files(imports, repo_files, resolutions)
    return reached_targets(imports, files, chunks, resolutions)
