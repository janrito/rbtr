"""Edge inference: the chunks each import reaches.

`infer_import_edges` takes a snapshot's chunks as a frame and returns
an edge from each import to every chunk it reaches.  It reads only the
chunks and the repository's file list; the build writes the edges.

Each import takes one `ImportWay`:

- `MODULE`: its metadata names a module or a relative depth, which
  `_resolve_import_targets` turns into files.  An import naming symbols
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

import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath

import dataframely as dy
import polars as pl

from rbtr.domain.models import ChunkKind, Edge, EdgeKind, Edges, ImportMeta
from rbtr.languages.manager import LanguageManager
from rbtr.languages.registration import ModuleStyle, parse_path_relative


@dataclass(frozen=True)
class ImportResolution:
    """File resolution hints for imports from a specific language.

    Built by the orchestrator from plugin registrations and
    passed to edge inference functions as plain data.
    """

    extensions: tuple[str, ...]
    """File extensions to try (collected from target languages)."""
    index_files: tuple[str, ...]
    """Directory entry points to try (collected from target languages)."""
    source_roots: tuple[str, ...]
    """Directories to prepend when resolving absolute imports."""
    path_substitutions: tuple[tuple[str, str], ...]
    """Module path prefix replacements (e.g. `("crate/", "src/")`)."""
    module_style: ModuleStyle = ModuleStyle.PATH
    """How module strings map to file paths. See `ModuleStyle`."""
    own_extensions: frozenset[str] = frozenset()
    """The importing language's *own* extensions (distinct from the flattened
    target `extensions`). Used to break same-path ties in the suffix tier."""
    package_directory: bool = False
    """Whether a reference may name a directory whose every file it reaches."""
    prose: bool = False
    """Whether the language's references document what they name (see
    `LanguageRegistration.prose`)."""


def build_resolution_map(
    mgr: LanguageManager,
    manifests: dict[str, str] | None = None,
) -> dict[str, ImportResolution]:
    """Build language → resolution hints from plugin registrations.

    *manifests* maps a repository file's path to its text, for the files
    languages declare as their `manifest`. A language whose manifest is
    present gains the substitutions it reads from it, ahead of the ones
    its registration declares: Go's `go.mod` names the module prefix its
    imports carry, which is a fact about the repository rather than about
    the language.
    """
    manifests = manifests or {}
    result: dict[str, ImportResolution] = {}
    for lang_id in mgr.all_language_ids():
        reg = mgr.get_registration(lang_id)
        if reg is None:
            continue
        from_manifest: tuple[tuple[str, str], ...] = ()
        if reg.manifest and (text := manifests.get(reg.manifest)) is not None:
            from_manifest = reg.substitutions_from_manifest(text)
        target_langs = reg.import_targets or frozenset({lang_id})
        exts: list[str] = []
        idxs: list[str] = []
        for target in target_langs:
            target_reg = mgr.get_registration(target)
            if target_reg is not None:
                exts.extend(target_reg.extensions)
                idxs.extend(target_reg.index_files)
        result[lang_id] = ImportResolution(
            extensions=tuple(exts),
            index_files=tuple(idxs),
            source_roots=reg.source_roots,
            path_substitutions=from_manifest + reg.path_substitutions,
            module_style=reg.module_style,
            own_extensions=reg.extensions,
            package_directory=reg.package_directory,
            prose=reg.prose,
        )
    return result


def _substituted(module_path: str, resolution: ImportResolution) -> str:
    """Apply the first matching prefix substitution, then the module style."""
    resolved = module_path
    for prefix, replacement in resolution.path_substitutions:
        if resolved.startswith(prefix):
            resolved = replacement + resolved[len(prefix) :]
            break
    if resolution.module_style is ModuleStyle.DOTTED and "." in resolved:
        resolved = resolved.replace(".", "/")
    return resolved


def _under_source_roots(
    resolved: str,
    repo_files: set[str],
    resolution: ImportResolution,
) -> str | None:
    """Search each source root for the bare path, then extensions, then index files."""
    for root in resolution.source_roots:
        base = f"{root}/{resolved}" if root else resolved
        if base in repo_files:
            return base
        for ext in resolution.extensions:
            if f"{base}{ext}" in repo_files:
                return f"{base}{ext}"
        for idx in resolution.index_files:
            if f"{base}/{idx}" in repo_files:
                return f"{base}/{idx}"
    return None


def _by_path_suffix(
    resolved: str,
    repo_files: set[str],
    resolution: ImportResolution,
    importer_ext: str,
) -> str | None:
    """Match *resolved* as a full-path suffix, dropping genuine ambiguity.

    Requires >= 2 path segments so single-segment bare modules (`os`, `io`)
    cannot match a nested `.../os.py`. Candidates are grouped by path without
    extension: more than one group means distinct files, which is ambiguity
    rather than a choice, and resolves to nothing.
    """
    if "/" not in resolved:
        return None
    candidates = _suffix_matches(resolved, repo_files, resolution.extensions)
    if not candidates:
        return None
    groups: dict[str, list[str]] = {}
    for f in candidates:
        groups.setdefault(str(PurePosixPath(f).with_suffix("")), []).append(f)
    if len(groups) != 1:
        return None
    group = next(iter(groups.values()))
    if len(group) == 1:
        return group[0]
    return _pick_by_importer(group, importer_ext, resolution.own_extensions)


def _resolve_module_to_file(
    module_path: str,
    repo_files: set[str],
    resolution: ImportResolution,
    importer_ext: str = "",
) -> str | None:
    """Find a repo file matching a `/`-separated *module_path*.

    DOTTED: converts `.` to `/` before searching.
    PATH: searches as-is (bare path first, then extensions).

    Resolution proceeds in tiers: config-driven prefix matching against
    `source_roots` first, then a layout-independent full-path *suffix* match
    (Tier 3). *importer_ext* is the importing file's extension, used only to
    break ties when several files share one path but differ by extension.
    """
    resolved = _substituted(module_path, resolution)
    under_root = _under_source_roots(resolved, repo_files, resolution)
    if under_root is not None:
        return under_root
    # The index-file (package) form is not tried as a path suffix.
    return _by_path_suffix(resolved, repo_files, resolution, importer_ext)


def _suffix_matches(
    name: str,
    repo_files: set[str],
    extensions: tuple[str, ...],
) -> list[str]:
    """Collect repo files matching *name* as a full-path suffix.

    A file matches when it equals `name{ext}` or ends with `/name{ext}` for
    some extension. Used by the import resolver's Tier 3 (drop rather than
    guess on ambiguity).
    """
    matches: list[str] = []
    for ext in extensions:
        target = f"{name}{ext}"
        suffix = f"/{target}"
        matches.extend(f for f in repo_files if f == target or f.endswith(suffix))
    return matches


def _pick_by_importer(
    candidates: list[str],
    importer_ext: str,
    own_extensions: frozenset[str],
) -> str | None:
    """Pick the candidate closest to the importing file.

    Ranks by: (0) extension equals the importer's own extension, (1) extension
    belongs to the importer's language, (2) anything else. Returns the single
    best-ranked candidate, or `None` if two candidates tie at the top rank.
    """

    def rank(f: str) -> int:
        ext = PurePosixPath(f).suffix
        if importer_ext and ext == importer_ext:
            return 0
        if ext in own_extensions:
            return 1
        return 2

    ranked = sorted(candidates, key=rank)
    best = rank(ranked[0])
    if sum(1 for f in ranked if rank(f) == best) != 1:
        return None
    return ranked[0]


def _files_in_directory(
    path: str,
    repo_files: set[str],
    extensions: tuple[str, ...],
) -> list[str]:
    """The language's files sitting directly in the directory *path*.

    Some references name a directory rather than a file, and the files
    inside it are not named by any convention: a Go package is its
    directory and every `.go` file in it belongs to the package, and a
    Terraform module is a directory of `.tf` files. Where a language does
    name one — `__init__.py`, `index.js`, `mod.rs` — `index_files`
    resolves it to that single file first.
    """
    if not path or not extensions:
        return []
    inside = f"{path}/"
    return sorted(
        f
        for f in repo_files
        if f.startswith(inside) and "/" not in f[len(inside) :] and f.endswith(extensions)
    )


def module_files(
    path: str,
    repo_files: set[str],
    resolution: ImportResolution,
    importer_ext: str = "",
) -> list[str]:
    """The files a reference to the module *path* names: one file, or a directory's.

    *path* is `/`-separated (a dotted module style's `.` also works).
    *importer_ext* breaks ties between files that differ only by
    extension; a caller with no importing file leaves it empty.
    """
    found = _resolve_module_to_file(path, repo_files, resolution, importer_ext)
    if found is not None:
        return [found]
    if not resolution.package_directory:
        return []
    # `_resolve_module_to_file` substitutes prefixes on its own copy, so the
    # directory search has to be given the substituted path too.
    return _files_in_directory(_substituted(path, resolution), repo_files, resolution.extensions)


def _resolve_unanchored(
    module: str,
    file_path: str,
    repo_files: set[str],
    resolution: ImportResolution,
) -> list[str]:
    """Resolve a reference that names no starting point of its own.

    The repository root is tried first, so a reference that already
    resolved keeps the file it resolved to. A path-style reference that
    finds nothing there names a sibling: `@import "b.css"` in
    `css/a.css` means `css/b.css`.
    """
    importer_ext = PurePosixPath(file_path).suffix
    from_root = module_files(module, repo_files, resolution, importer_ext)
    if from_root or resolution.module_style is not ModuleStyle.PATH:
        return from_root
    beside_importer = _relative_module(file_path, 1, module)
    if beside_importer is None or beside_importer == module:
        return []
    return module_files(beside_importer, repo_files, resolution, importer_ext)


def _resolve_import_targets(
    meta: ImportMeta,
    file_path: str,
    repo_files: set[str],
    resolution: ImportResolution | None,
) -> list[str]:
    """Resolve import metadata to a repo file path.

    Handles relative imports (extractor-set `dots` or PATH-style
    `./`/`../` prefixes), then delegates to `_resolve_module_to_file`.

    A reference carrying its own depth (`../b`, or an extractor-set
    `dots`) is anchored to the importing file and has one place to look;
    anything else goes through `_resolve_unanchored`.
    """
    module = meta.module or ""

    # Determine relative depth: from extractor or from path prefix.
    dots = int(meta.dots) if meta.dots else 0
    if not dots and resolution is not None and resolution.module_style is ModuleStyle.PATH:
        dots, module = parse_path_relative(module)
    if resolution is None:
        return []

    if dots:
        anchored = _relative_module(file_path, dots, module)
        if anchored is None:
            return []
        return module_files(anchored, repo_files, resolution, PurePosixPath(file_path).suffix)

    if not module:
        return []
    return _resolve_unanchored(module, file_path, repo_files, resolution)


def _relative_module(file_path: str, dots: int, module: str) -> str | None:
    """Resolve a relative import *dots* levels above *file_path*.

    Returns `None` when the importing file sits too shallow for the climb,
    or when nothing is left to name after it.
    """
    parts = PurePosixPath(file_path).parts
    if len(parts) < dots:
        return None
    segs = list(parts[:-dots])
    if "/" in module:
        segs.extend(s for s in module.split("/") if s)
    elif module:
        segs.append(module)
    if not segs:
        return None
    return "/".join(segs)


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


def _build_stem_index(repo_files: set[str], index_stems: set[str]) -> dict[str, list[str]]:
    """Map a file stem a word of an import may name to the files with that stem.

    Left out: the stems of *index_stems*, the files languages declare as
    a directory's stand-in (`__init__`, `index`, `mod`), which name their
    directory rather than a module; and stems shorter than three
    characters (`a`, `io`), which too many words would match.
    """
    stem_to_files: dict[str, list[str]] = {}
    for f in repo_files:
        stem = PurePosixPath(f).stem
        if stem not in index_stems and len(stem) >= 3:
            stem_to_files.setdefault(stem, []).append(f)
    return stem_to_files


# Pre-compiled word tokeniser — splits on non-alphanumeric/underscore.
_WORD_RE = re.compile(r"[A-Za-z_]\w*")


def _resolution_for(
    language: str,
    meta: ImportMeta,
    resolution_map: dict[str, ImportResolution] | None,
) -> ImportResolution | None:
    """Pick the resolution rules for an import written in *language*.

    A `language_hint` (HTML `<script src>` naming JavaScript) resolves
    against the hinted language's import targets rather than the language
    of the file the import sits in.
    """
    if resolution_map is None:
        return None
    if meta.language_hint:
        return resolution_map.get(meta.language_hint)
    return resolution_map.get(language)


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
    module_targets: Callable[[dict[str, str]], list[str]],
) -> dy.DataFrame[ImportRow]:
    """*chunks*' imports, each with its way, its edge kind and, for a `MODULE` import, its module's files."""
    imports = chunks.filter(pl.col("kind") == ChunkKind.IMPORT).with_columns(_way(prose))
    by_module = imports.filter(pl.col("way") == ImportWay.MODULE).with_columns(
        pl.struct(pl.all())
        .map_elements(module_targets, return_dtype=pl.List(pl.String))
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
    stem_targets: Callable[[str], list[str]],
) -> dy.DataFrame[InferredEdgeRow]:
    """A `TEXT` import reaches the first symbol of each file whose stem is one of its words."""
    first_by_file = symbols.unique("file_path", keep="first", maintain_order=True)
    return (
        imports.filter(pl.col("way") == ImportWay.TEXT)
        .with_columns(
            pl.col("name")
            .map_elements(stem_targets, return_dtype=pl.List(pl.String))
            .alias("target_path"),
            kind=pl.lit(EdgeKind.IMPORTS),
        )
        .explode("target_path")
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
    module resolves to files with `_resolve_import_targets`, by the rules
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
    stem_index = _build_stem_index(repo_files, index_stems)

    def module_targets(row: dict[str, str]) -> list[str]:
        meta = ImportMeta.model_validate(row)
        resolution = _resolution_for(row["language"], meta, rules)
        return [
            f
            for f in _resolve_import_targets(meta, row["file_path"], repo_files, resolution)
            # A file does not import itself, however its neighbours are named.
            if f != row["file_path"]
        ]

    def stem_targets(name: str) -> list[str]:
        return [f for word in set(_WORD_RE.findall(name)) for f in stem_index.get(word, ())]

    imports = _imports(chunks, prose, module_targets)
    edges = pl.concat(
        [
            _bare_edges(imports, importable),
            _named_edges(imports, symbols, sections),
            _name_edges(imports, symbols),
            _text_edges(imports, symbols, stem_targets),
        ]
    )
    return Edges.validate_python(edges.to_dicts())
