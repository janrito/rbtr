"""Which repository files imports name, as polars over every import of a batch.

`module_files` turns module lookups into the files each names, by one
language's `LanguageResolutionRow`; qualified-name lookup uses it directly.
`resolved_files` turns a batch's `ImportRow`s into `FileMatchRow`s:
module imports through `module_files`, text imports by matching
their words against file stems. Paths arrive as the native set the build
collects and become a frame once, inside each process.
"""

from __future__ import annotations

import dataframely as dy
import polars as pl

from rbtr.domain.edge_frames import (
    FileMatchRow,
    ImportRow,
    ImportWay,
    LanguageResolutionRow,
    ModuleFileRow,
    ModulePathRow,
)
from rbtr.domain.models import ModuleStyle


def path_extension(path: pl.Expr) -> pl.Expr:
    """*path*'s extension, dot included, as `PurePosixPath.suffix` gives it; empty when it has none."""
    return path.str.extract(r"[^/](\.[^./]+)$", 1).fill_null("")


def path_name(path: pl.Expr) -> pl.Expr:
    """*path*'s last segment, trailing slashes ignored."""
    return path.str.replace(r"/+$", "").str.extract(r"([^/]*)$", 1)


def parent_directory(path: pl.Expr) -> pl.Expr:
    """The directory holding *path*; null for a file at the repository root."""
    return path.str.extract(r"^(.*)/[^/]*$", 1)


def rewritten_module() -> pl.Expr:
    """The lookup path after its first matching substitution (`rewritten`), its separators slashes.

    A dotted language's dots and a scoped language's `::` become `/`.
    """
    path = pl.coalesce("rewritten", "path")
    return (
        pl.when(pl.col("module_style") == ModuleStyle.DOTTED)
        .then(path.str.replace_all(".", "/", literal=True))
        .when(pl.col("module_style") == ModuleStyle.SCOPED)
        .then(path.str.replace_all("::", "/", literal=True))
        .otherwise(path)
    )


def under_root() -> pl.Expr:
    """The rewritten module under one source root; the root `""` is the repository's."""
    return (
        pl.when(pl.col("source_roots") == "")
        .then(pl.col("resolved"))
        .otherwise(pl.col("source_roots") + "/" + pl.col("resolved"))
    )


def importer_preference() -> pl.Expr:
    """Rank a candidate file: 0 for the referring file's extension, 1 for another of its language's, else 2."""
    extension = path_extension(pl.col("target_path"))
    return (
        pl.when((pl.col("importer_extension") != "") & (extension == pl.col("importer_extension")))
        .then(0)
        .when(pl.col("own_extensions").list.contains(extension))
        .then(1)
        .otherwise(2)
    )


def module_files(
    lookups: dy.DataFrame[ModulePathRow],
    repo_files: set[str],
    resolutions: dy.DataFrame[LanguageResolutionRow],
) -> dy.DataFrame[ModuleFileRow]:
    """The files each lookup attempt names, by its language's resolution; a language without one names nothing.

    An attempt's path is rewritten by the first substitution its prefix
    matches, dots made slashes for a dotted language, then looked for:

    1. Under each source root in turn: as written, with an extension, or as
       a directory's index file. The first root that holds one wins; within
       it, as written beats an extension beats an index file.
    2. Otherwise, as a full-path suffix of two or more segments anywhere.
       Candidates with different paths are ambiguous and name nothing.
    3. Otherwise, for a language whose package is its directory, every file
       of its extensions directly inside the directory it names.

    Where steps 1 and 2 find files that differ only by extension, the
    referring file's extension wins, then its language's, then the
    earliest in `extensions`.
    """
    key = ["lookup", "attempt"]
    paths = pl.DataFrame({"target_path": sorted(repo_files)}, schema={"target_path": pl.String})
    attempts = lookups.join(resolutions, on="language")
    rewritten = (
        attempts.select(*key, "path", "substitutions")
        .with_columns(position=pl.int_ranges(0, pl.col("substitutions").list.len()))
        .explode("substitutions", "position")
        .unnest("substitutions")
        .filter(pl.col("path").str.starts_with(pl.col("prefix")))
        .sort("position")
        .unique(key, keep="first")
        .select(
            *key,
            rewritten=pl.col("replacement")
            + pl.col("path").str.slice(pl.col("prefix").str.len_chars()),
        )
    )
    attempts = attempts.join(rewritten, on=key, how="left").with_columns(
        resolved=rewritten_module()
    )
    rooted = (
        attempts.with_columns(root_order=pl.int_ranges(0, pl.col("source_roots").list.len()))
        .explode("source_roots", "root_order")
        .with_columns(base=under_root())
    )
    candidate = [*key, "root_order"]
    in_roots = (
        pl.concat(
            [
                rooted.select(
                    *candidate,
                    tier=pl.lit(0),
                    position=pl.lit(0, dtype=pl.Int64),
                    target_path="base",
                    importer_extension="importer_extension",
                    own_extensions="own_extensions",
                ),
                rooted.with_columns(position=pl.int_ranges(0, pl.col("extensions").list.len()))
                .explode("extensions", "position")
                .select(
                    *candidate,
                    tier=pl.lit(1),
                    position="position",
                    target_path=pl.col("base") + pl.col("extensions"),
                    importer_extension="importer_extension",
                    own_extensions="own_extensions",
                ),
                rooted.with_columns(position=pl.int_ranges(0, pl.col("index_files").list.len()))
                .explode("index_files", "position")
                .select(
                    *candidate,
                    tier=pl.lit(2),
                    position="position",
                    target_path=pl.col("base") + "/" + pl.col("index_files"),
                    importer_extension="importer_extension",
                    own_extensions="own_extensions",
                ),
            ]
        )
        .join(paths, on="target_path")
        .with_columns(preference=importer_preference())
        .sort("root_order", "tier", "preference", "position")
        .unique(key, keep="first")
        .select(*key, "target_path")
    )
    # Every suffix of every path that starts at a segment: `index.py` and
    # `rbtr/index.py` both name `src/rbtr/index.py`, `dex.py` names nothing.
    parts = pl.col("target_path").str.split("/")
    tails = (
        paths.with_columns(start=pl.int_ranges(0, parts.list.len()))
        .explode("start")
        .select("target_path", name=parts.list.slice(pl.col("start")).list.join("/"))
    )
    by_suffix = (
        attempts.join(in_roots, on=key, how="anti")
        .filter(pl.col("resolved").str.contains("/", literal=True))
        .with_columns(position=pl.int_ranges(0, pl.col("extensions").list.len()))
        .explode("extensions", "position")
        .with_columns(name=pl.col("resolved") + pl.col("extensions"))
        .join(tails, on="name")
        .filter(
            pl.col("target_path")
            .str.strip_suffix(path_extension(pl.col("target_path")))
            .n_unique()
            .over(key)
            == 1
        )
        .with_columns(preference=importer_preference())
        .sort("preference", "position")
        .unique(key, keep="first")
        .select(*key, "target_path")
    )
    in_directory = (
        attempts.join(pl.concat([in_roots, by_suffix]), on=key, how="anti")
        .filter(pl.col("package_directory") & (pl.col("resolved") != ""))
        .explode("extensions")
        .join(
            paths.with_columns(directory=parent_directory(pl.col("target_path"))),
            left_on="resolved",
            right_on="directory",
        )
        .filter(pl.col("target_path").str.ends_with(pl.col("extensions")))
        .select(*key, "target_path")
        .unique()
    )
    return pl.concat([in_roots, by_suffix, in_directory]).pipe(ModuleFileRow.validate, cast=True)


def leading_words(module: pl.Expr, word: pl.Expr) -> pl.Expr:
    """*module*'s leading run of *word* (`super::super::`); null when it starts with none, or *word* is null."""
    return module.str.extract(pl.concat_str(pl.lit("^(?:"), word, pl.lit("(?:::|$))+")), 0)


def in_directory_module(importer: pl.Expr) -> pl.Expr:
    """Whether *importer* is its directory's module (`directory_module_files`), not a module of its own inside it."""
    return (
        pl.col("directory_module_files").list.contains(path_name(importer)).fill_null(value=False)
    )


def written_depth(module: pl.Expr, importer: pl.Expr) -> pl.Expr:
    """*module*'s relative depth, read from its prefix: 1 is the *importer*'s own directory, 2 its parent.

    A path writes `./models` (1) and `../../shared/x` (3), and a directory
    alone as `.` (1) or `..` (2). A dotted module writes one leading dot per
    level, `.models` (1) and `..shared.x` (2). A module word reads from the
    importer's module (`in_directory_module`): the current word stays in
    it, `self::util` (1); each parent word climbs one level, `super::util`
    is 2 from a directory's module and 1 from a file's own. `react` is 0,
    not relative.
    """
    parents = leading_words(module, pl.col("parent_module"))
    climbs = module.str.extract(r"^((?:\.\./)*)", 1).str.len_chars() // 3
    return (
        pl.when(parents.is_not_null())
        .then(
            parents.str.split("::").list.eval(pl.element() != "").list.sum()
            + in_directory_module(importer).cast(pl.Int64)
        )
        .when(leading_words(module, pl.col("current_module")).is_not_null())
        .then(1)
        .when(module.str.starts_with("./"))
        .then(1)
        .when(climbs > 0)
        .then(climbs + 1)
        .when(module.is_in([".", ".."]) | (pl.col("module_style") == ModuleStyle.DOTTED))
        .then(module.str.extract(r"^(\.*)", 1).str.len_chars())
        .otherwise(0)
        .cast(pl.Int64)
    )


def written_module(module: pl.Expr, importer: pl.Expr) -> pl.Expr:
    """*module* with the prefix `written_depth` reads removed.

    A current word read in a file that is a module of its own names a
    child of that module, in the directory named by the file's stem:
    `self::util` in `order.rs` is `order::util`.
    """
    parents = leading_words(module, pl.col("parent_module"))
    current = leading_words(module, pl.col("current_module"))
    rest = module.str.slice(current.str.len_chars())
    return (
        pl.when(parents.is_not_null())
        .then(module.str.slice(parents.str.len_chars()))
        .when(current.is_not_null() & in_directory_module(importer))
        .then(rest)
        .when(current.is_not_null())
        .then(
            pl.concat_list(file_stem(importer), rest)
            .list.eval(pl.element().filter(pl.element() != ""))
            .list.join("::")
        )
        .when(module.str.starts_with("./"))
        .then(module.str.slice(2))
        .when(module.str.starts_with("../"))
        .then(module.str.replace(r"^(?:\.\./)+", ""))
        .when(module.is_in([".", ".."]) | (pl.col("module_style") == ModuleStyle.DOTTED))
        .then(module.str.strip_chars_start("."))
        .otherwise(module)
    )


def climbed(levels: pl.Expr) -> pl.Expr:
    """`module` read *levels* steps up from `source_path`: 1 is the file's own directory.

    Null when the file sits too shallow for the climb, or nothing is left to name.
    """
    parts = pl.col("source_path").str.split("/")
    remaining = parts.list.len().cast(pl.Int64) - levels
    joined = pl.concat_list(
        parts.list.head(pl.max_horizontal(remaining, 0)),
        pl.col("module").str.split("/").list.eval(pl.element().filter(pl.element() != "")),
    ).list.join("/")
    return pl.when((remaining >= 0) & (joined != "")).then(joined)


def attempt_paths() -> pl.Expr:
    """The paths a module import is looked for at, in order.

    A relative import has one, anchored to its file. Any other is looked
    for from the root, then, for a path-style language, beside its file:
    `@import "b.css"` in `css/a.css` means `css/b.css` when the root has no `b.css`.
    """
    beside = pl.when(pl.col("module_style") == ModuleStyle.PATH).then(climbed(pl.lit(1)))
    return (
        pl.when(pl.col("depth") > 0)
        .then(pl.concat_list(climbed(pl.col("depth"))))
        .when(pl.col("module") == "")
        .then(pl.lit([], dtype=pl.List(pl.String)))
        .otherwise(
            pl.concat_list(pl.col("module"), pl.when(beside != pl.col("module")).then(beside))
        )
        .list.drop_nulls()
    )


def file_stem(path: pl.Expr) -> pl.Expr:
    """*path*'s last segment without its extension, as `PurePosixPath.stem` gives it."""
    return path_name(path).str.strip_suffix(path_extension(path))


def resolved_files(
    imports: dy.DataFrame[ImportRow],
    repo_files: set[str],
    resolutions: dy.DataFrame[LanguageResolutionRow],
) -> dy.DataFrame[FileMatchRow]:
    """The files each import reaches, by the resolution of its hinted language, else its own.

    A `MODULE` import is looked for at each of its `attempt_paths` in
    turn through `module_files`; the first attempt that names a file is its
    answer. A file is its own answer only when the import names that
    file and something in it (`guide.md#usage` in `guide.md`). A `TEXT`
    import reaches every file whose stem is one of its words, except
    stems shorter than three characters and the stems of index files
    (`__init__`, `index`, `mod`), which name a directory. `NAME` imports
    name no file.
    """
    location = ["source_id", "source_path"]
    modules = (
        imports.filter(pl.col("way") == ImportWay.MODULE)
        .with_row_index("lookup")
        .with_columns(
            language=pl.when(pl.col("language_hint") != "")
            .then("language_hint")
            .otherwise("source_language")
        )
        .join(resolutions.select("language", "module_style"), on="language")
        .with_columns(path=attempt_paths())
        .with_columns(attempt=pl.int_ranges(0, pl.col("path").list.len()))
        .explode("path", "attempt")
        .drop_nulls("path")
    )
    lookups = modules.select(
        "lookup",
        "attempt",
        "language",
        "path",
        importer_extension=path_extension(pl.col("source_path")),
    ).pipe(ModulePathRow.validate, cast=True)
    by_module = (
        module_files(lookups, repo_files, resolutions)
        .filter(pl.col("attempt") == pl.col("attempt").min().over("lookup"))
        .join(modules.unique("lookup").select("lookup", *location, "module", "names"), on="lookup")
        .filter(
            (pl.col("target_path") != pl.col("source_path"))
            | (
                (pl.col("names").list.len() > 0)
                & (path_name(pl.col("module")) == path_name(pl.col("source_path")))
            )
        )
        .select(*location, "target_path")
    )
    index_stems = resolutions.select(target_path=pl.col("index_files").explode()).drop_nulls()
    stems = (
        pl.DataFrame({"target_path": sorted(repo_files)}, schema={"target_path": pl.String})
        .with_columns(word=file_stem(pl.col("target_path")))
        .join(index_stems.select(word=file_stem(pl.col("target_path"))), on="word", how="anti")
        .filter(pl.col("word").str.len_chars() >= 3)
    )
    by_text = (
        imports.filter(pl.col("way") == ImportWay.TEXT)
        .with_columns(word=pl.col("text").str.extract_all(r"[A-Za-z_]\w*").list.unique())
        .explode("word")
        .join(stems, on="word")
        .select(*location, "target_path")
    )
    return pl.concat([by_module, by_text]).unique().pipe(FileMatchRow.validate, cast=True)
